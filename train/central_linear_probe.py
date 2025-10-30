import argparse
import os
import sys
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import hashlib
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RWF2000Dataset, build_video_transform
from models import build_videomae_model, LinearHead, trainable_parameters_filter, ClassifyLoss
from train.utils import load_config, default_device, make_adamw, set_seed


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = default_device()

    data_cfg = cfg['data']
    transform = build_video_transform(size=data_cfg.get('size', 224))

    # Setup plain-text logging similar to small-head finetune
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    def _infer_dp_subdir(cfg_dict: dict) -> str:
        try:
            ckpt = (cfg_dict or {}).get('model', {}).get('pretrained_checkpoint', '')
            low = str(ckpt).lower()
            if any(t in low for t in ['nodp', 'no-dp', 'no_dp']):
                return os.path.join('no-dp')
            import re
            m = re.search(r'epsilon[_\-]?(\d+)', low)
            if m:
                return os.path.join('dp', f'epsilon_{m.group(1)}')
        except Exception:
            pass
        return 'unknown'
    base_logs = os.path.join(project_root, 'logs')
    sub = _infer_dp_subdir(cfg)
    log_dir = os.path.join(base_logs, sub)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'linear_probe_{ts}.txt')
    def _log(msg: str):
        try:
            with open(log_path, 'a', encoding='utf-8') as _f:
                _f.write(msg.rstrip('\n') + '\n')
        except Exception:
            pass
    try:
        _log('=== Task: linear_probe ===')
        _log(f'timestamp: {ts}')
    except Exception:
        pass

    # Resolve roots: allow a single base root with subdirs train/evaluate
    base_root = data_cfg.get('root')
    explicit_train_root = data_cfg.get('train_root')
    explicit_eval_root = data_cfg.get('eval_root') or data_cfg.get('val_root')

    def _prefer(base: Optional[str], sub: str) -> Optional[str]:
        if base is None:
            return None
        p = os.path.join(base, sub)
        return p if os.path.isdir(p) else None

    train_root = explicit_train_root or _prefer(base_root, 'train') or base_root
    eval_root = explicit_eval_root or _prefer(base_root, 'evaluate') or _prefer(base_root, 'val') or base_root

    # Datasets + loaders
    train_ds = RWF2000Dataset(
        root=train_root, transform=transform,
        num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4)
    )
    eval_ds = RWF2000Dataset(
        root=eval_root, transform=transform,
        num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4)
    )

    bs = data_cfg.get('batch_size', 16)
    num_workers = data_cfg.get('num_workers', 4)
    pin_mem = bool(data_cfg.get('pin_memory', True))
    val_bs = int(max(1, data_cfg.get('val_batch_size', max(1, bs // 2))))

    from data.collate import video_collate
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=video_collate, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=val_bs, shuffle=False, num_workers=max(0, num_workers // 2),
        collate_fn=video_collate, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )

    # Model
    model_cfg = cfg['model']
    peft_config = model_cfg.get('peft', None)
    backbone = build_videomae_model(
        model_name=model_cfg.get('model_name'),
        pretrained=model_cfg.get('pretrained', True),
        mode='feature', num_classes=2, peft_config=peft_config,
        pretrained_checkpoint=None,
    )

    # Load pretrained weights (strict=False, same as federated version)
    ckpt_path = model_cfg.get('pretrained_checkpoint')
    if ckpt_path:
        # log checkpoint sha256
        try:
            h = hashlib.sha256()
            with open(ckpt_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            _log(f"[Init] Using pretrained checkpoint: {ckpt_path} | sha256={h.hexdigest()[:16]}")
        except Exception:
            pass
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading pretrained weights: {len(missing_keys)} keys")
            _log(f"[Warn] Missing keys when loading pretrained weights: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading pretrained weights: {len(unexpected_keys)} keys")
            _log(f"[Warn] Unexpected keys when loading pretrained weights: {len(unexpected_keys)}")
        print(f"Loaded pretrained weights from {ckpt_path}")
        _log(f"Loaded pretrained weights from {ckpt_path}")

        # Additionally load PEFT trainable params (LoRA/Prompt) if present
        try:
            tr = checkpoint.get('trainable', None) if isinstance(checkpoint, dict) else None
            if isinstance(tr, dict) and len(tr) > 0:
                ms = backbone.state_dict()
                to_load = {k: v for k, v in tr.items() if k in ms and getattr(ms[k], 'shape', None) == getattr(v, 'shape', None)}
                n_all = len(tr)
                n_ok = len(to_load)
                if n_ok > 0:
                    try:
                        backbone.load_state_dict(to_load, strict=False)
                    except Exception:
                        try:
                            # If wrapper nesting differs
                            backbone.backbone.load_state_dict(to_load, strict=False)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                _log(f"[Init] Loaded PEFT trainable: matched={n_ok}/{n_all}")
                print(f"[Init] Loaded PEFT trainable: matched={n_ok}/{n_all}")
            else:
                _log("[Init] No PEFT trainable found in checkpoint")
        except Exception:
            _log("[Init] Exception while loading PEFT trainable; continuing")

    # Log LoRA fingerprint for comparability
    try:
        sd = backbone.state_dict()
        keys = sorted([k for k in sd.keys() if k.endswith('.A') or k.endswith('.B')])
        h = hashlib.sha256()
        total = 0
        for k in keys:
            t = sd[k].detach().cpu().contiguous().numpy()
            total += int(t.size)
            h.update(k.encode('utf-8'))
            h.update(t.tobytes(order='C'))
        fp = h.hexdigest()[:16]
        msg = f"[Init] LoRA fingerprint: keys={len(keys)} elems={total} sha256={fp}"
        print(msg); _log(msg)
    except Exception:
        pass

    # Freeze backbone; only train linear head
    for p in backbone.parameters():
        p.requires_grad = False
    in_dim = backbone._feat_dim()
    import torch.nn as nn
    model = nn.Sequential(backbone, LinearHead(in_dim, model_cfg.get('head', {}).get('num_classes', 2)))
    model.to(device)

    # Optimizer
    train_cfg = cfg.get('training', {})
    lr = train_cfg.get('lr', 5e-3)
    weight_decay = train_cfg.get('weight_decay', 0.0)
    optimizer = make_adamw(trainable_parameters_filter(model), lr=lr, weight_decay=weight_decay)

    # Loss (support optional class weights)
    loss_cfg = train_cfg.get('loss', {}) or {}
    raw_cw = loss_cfg.get('class_weights', None)

    def _parse_class_weights(cw):
        if cw is None:
            return None
        # Handle string forms: 'none', '', 'null', '1.2,1.3', '[1.2, 1.3]'
        if isinstance(cw, str):
            s = cw.strip()
            if s.lower() in ('none', 'null', ''):
                return None
            # try JSON-like list
            try:
                import json as _json
                v = _json.loads(s)
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
            except Exception:
                pass
            # try comma-separated
            if ',' in s:
                try:
                    return [float(x.strip()) for x in s.split(',') if x.strip()]
                except Exception:
                    return None
            # fallback: single float
            try:
                return [float(s)]
            except Exception:
                return None
        if isinstance(cw, (list, tuple)):
            try:
                return [float(x) for x in cw]
            except Exception:
                return None
        return None

    class_weights = _parse_class_weights(raw_cw)
    if class_weights is not None:
        print(f"Using weighted loss with class weights: {class_weights}")
        _log(f"[Init] class_weights={class_weights}")
        loss_fn = ClassifyLoss(class_weights=class_weights, device=device)
    else:
        print("Using unweighted loss")
        _log("[Init] class_weights=unweighted")
        loss_fn = ClassifyLoss()

    # Train loop (centralized)
    epochs = int(train_cfg.get('epochs', 20))
    log_every = int(cfg.get('logging', {}).get('log_every', 50))
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total = 0
        correct = 0
        pbar = tqdm(total=len(train_loader.dataset), desc=f"[train][epoch {epoch}/{epochs}]", unit='samples', leave=True, ncols=120)
        # reset epoch-wise accumulators inside loss_fn (so get_detailed_metrics reflects this epoch)
        try:
            if hasattr(loss_fn, 'all_preds'):
                loss_fn.all_preds = []
            if hasattr(loss_fn, 'all_labels'):
                loss_fn.all_labels = []
            if hasattr(loss_fn, 'all_probs'):
                loss_fn.all_probs = []
        except Exception:
            pass
        for i, batch in enumerate(train_loader, start=1):
            x = batch['video'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_fn(model, x, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * y.size(0)
            total += int(y.size(0))
            if 'acc' in metrics:
                correct += int(metrics['acc'] * y.size(0))
            if i % max(1, log_every) == 0:
                cur_loss = running_loss / max(1, total)
                cur_acc = correct / max(1, total) if total > 0 else 0.0
                pbar.set_postfix({"loss": f"{cur_loss:.6f}", "acc": f"{cur_acc:.6f}"})
            pbar.update(y.size(0))
        pbar.close()
        # epoch summary
        ep_loss = running_loss / max(1, total)
        ep_acc = correct / max(1, total) if total > 0 else 0.0
        # detailed epoch metrics from loss_fn aggregator (macro/weighted)
        ep_prec_m = ep_rec_m = ep_f1_m = ep_f1_w = None
        try:
            det = loss_fn.get_detailed_metrics()
            ep_prec_m = float(det.get('precision_macro')) if det and 'precision_macro' in det else None
            ep_rec_m = float(det.get('recall_macro')) if det and 'recall_macro' in det else None
            ep_f1_m = float(det.get('f1_macro')) if det and 'f1_macro' in det else None
            ep_f1_w = float(det.get('f1_weighted')) if det and 'f1_weighted' in det else None
        except Exception:
            det = None
        # log + print
        msg = f"[Train] epoch={epoch} loss={ep_loss:.6f} acc={ep_acc:.6f}"
        if ep_f1_m is not None:
            msg += f" f1_macro={ep_f1_m:.6f}"
        if ep_f1_w is not None:
            msg += f" f1_weighted={ep_f1_w:.6f}"
        if ep_prec_m is not None:
            msg += f" precision_macro={ep_prec_m:.6f}"
        if ep_rec_m is not None:
            msg += f" recall_macro={ep_rec_m:.6f}"
        _log(msg)
        print(f"{msg.replace('epoch=', f'epoch={epoch}/{epochs} ')}")

    # Evaluate final model with detailed metrics (same logic as federated version)
    print("\n=== Final Model Evaluation ===")
    model.eval()
    with torch.no_grad():
        # Reset loss function accumulator
        loss_fn = ClassifyLoss()
        total_eval = 0
        correct_eval = 0
        running_eval_loss = 0.0
        pbar = tqdm(total=len(eval_loader.dataset), desc='[eval]', unit='samples', leave=False, ncols=120)
        for batch in eval_loader:
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            loss, metrics = loss_fn(model, video, labels)
            bs = int(labels.size(0))
            total_eval += bs
            running_eval_loss += float(loss.item()) * bs
            if 'acc' in metrics:
                correct_eval += int(metrics['acc'] * bs)
            cur_acc = correct_eval / max(1, total_eval)
            cur_loss = running_eval_loss / max(1, total_eval)
            pbar.set_postfix({"acc": f"{cur_acc:.6f}", "loss": f"{cur_loss:.6f}"})
            pbar.update(bs)
        pbar.close()
        detailed = loss_fn.get_detailed_metrics()
        if detailed:
            print(f"Accuracy: {detailed['accuracy']:.6f}")
            print(f"Precision (Weighted): {detailed['precision_weighted']:.6f}")
            print(f"Recall (Weighted): {detailed['recall_weighted']:.6f}")
            print(f"F1-Score (Weighted): {detailed['f1_weighted']:.6f}")
            print(f"Precision (Macro): {detailed['precision_macro']:.6f}")
            print(f"Recall (Macro): {detailed['recall_macro']:.6f}")
            print(f"F1-Score (Macro): {detailed['f1_macro']:.6f}")
            if 'roc_auc' in detailed:
                print(f"ROC-AUC: {detailed['roc_auc']:.6f}")
            if 'average_precision' in detailed:
                print(f"PR-AUC: {detailed['average_precision']:.6f}")
            cm = np.array(detailed['confusion_matrix'])
            print(f"\nConfusion Matrix:")
            print(f"         Predicted")
            print(f"         NonFight  Fight")
            print(f"Actual NonFight   {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"       Fight       {cm[1,0]:4d}   {cm[1,1]:4d}")
            # Save detailed metrics
            import json
            out_dir = cfg.get('output_dir', 'runs/linear_probe')
            os.makedirs(out_dir, exist_ok=True)
            metrics_file = os.path.join(out_dir, 'detailed_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(detailed, f, indent=2)
            print(f"\nDetailed metrics saved to: {metrics_file}")
            # Log key metrics
            _log('metrics:')
            for k in (
                'accuracy','precision_weighted','recall_weighted','f1_weighted',
                'precision_macro','recall_macro','f1_macro'
            ):
                if k in detailed:
                    _log(f"{k}: {float(detailed[k]):.6f}")
            for k in ('roc_auc','average_precision'):
                if k in detailed:
                    _log(f"{k}: {float(detailed[k]):.6f}")
            if 'confusion_matrix' in detailed:
                cm = detailed['confusion_matrix']
                try:
                    _log('confusion_matrix:')
                    _log(f"{cm[0][0]} {cm[0][1]}")
                    _log(f"{cm[1][0]} {cm[1][1]}")
                except Exception:
                    pass

            # Threshold scan (report best F1-macro and best Accuracy)
            try:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                y_true = np.array(getattr(loss_fn, 'all_labels', []))
                y_prob = np.array(getattr(loss_fn, 'all_probs', []))
                if y_true.size > 0 and y_prob.size > 0 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                    y_true = y_true.astype(int)
                    p1 = y_prob[:, 1]
                    thresholds = np.linspace(0.0, 1.0, 1001)
                    best_f1 = (-1.0, 0.5)  # (f1, thr)
                    best_acc = (-1.0, 0.5) # (acc, thr)
                    cache = {}
                    for thr in thresholds:
                        pred = (p1 >= thr).astype(int)
                        acc = accuracy_score(y_true, pred)
                        f1m = f1_score(y_true, pred, average='macro', zero_division=0)
                        cache[thr] = (acc, f1m, pred)
                        if f1m > best_f1[0]:
                            best_f1 = (f1m, thr)
                        if acc > best_acc[0]:
                            best_acc = (acc, thr)

                    # Recompute detailed metrics at best-F1 threshold
                    thr_f1 = float(best_f1[1])
                    pred_f1 = cache[thr_f1][2]
                    prec_m = precision_score(y_true, pred_f1, average='macro', zero_division=0)
                    rec_m = recall_score(y_true, pred_f1, average='macro', zero_division=0)
                    cm_f1 = np.zeros((2,2), dtype=int)
                    for t, p in zip(y_true, pred_f1):
                        cm_f1[t, p] += 1

                    thr_acc = float(best_acc[1])
                    pred_acc = cache[thr_acc][2]
                    f1_at_acc = f1_score(y_true, pred_acc, average='macro', zero_division=0)

                    print("\n=== Threshold Scan (eval) ===")
                    print(f"Best F1-macro threshold: {thr_f1:.2f}")
                    print(f"  F1-macro={best_f1[0]:.6f} Acc={cache[thr_f1][0]:.6f} Prec-macro={prec_m:.6f} Rec-macro={rec_m:.6f}")
                    print("  Confusion Matrix @best F1:")
                    print(f"         Predicted")
                    print(f"         NonFight  Fight")
                    print(f"Actual NonFight   {cm_f1[0,0]:4d}   {cm_f1[0,1]:4d}")
                    print(f"       Fight       {cm_f1[1,0]:4d}   {cm_f1[1,1]:4d}")
                    print(f"Best Accuracy threshold: {thr_acc:.2f}")
                    print(f"  Acc={best_acc[0]:.6f} F1-macro={f1_at_acc:.6f}")

                    _log('threshold_scan:')
                    _log(f"best_f1_macro_thr: {thr_f1:.6f}")
                    _log(f"  f1_macro: {best_f1[0]:.6f}")
                    _log(f"  acc_at_best_f1: {cache[thr_f1][0]:.6f}")
                    _log(f"  precision_macro_at_best_f1: {prec_m:.6f}")
                    _log(f"  recall_macro_at_best_f1: {rec_m:.6f}")
                    _log("  confusion_matrix_at_best_f1:")
                    _log(f"  {cm_f1[0,0]} {cm_f1[0,1]}")
                    _log(f"  {cm_f1[1,0]} {cm_f1[1,1]}")
                    _log(f"best_acc_thr: {thr_acc:.6f}")
                    _log(f"  acc: {best_acc[0]:.6f}")
                    _log(f"  f1_macro_at_best_acc: {f1_at_acc:.6f}")
            except Exception:
                pass

            # Optional: Bootstrap confidence intervals for metrics
            eval_cfg = cfg.get('evaluation', {}) or {}
            ci_cfg = (eval_cfg.get('ci') or {}) if isinstance(eval_cfg, dict) else {}
            method = str(ci_cfg.get('method', 'bootstrap')).lower()
            if method == 'bootstrap':
                n_boot = int(ci_cfg.get('n_samples', ci_cfg.get('n_bootstrap', 1000)))
                alpha = float(ci_cfg.get('alpha', 0.05))
                seed = int(ci_cfg.get('seed', cfg.get('seed', 42)))
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
                    # Collect eval arrays from loss_fn
                    y_true = np.array(getattr(loss_fn, 'all_labels', []))
                    y_prob = np.array(getattr(loss_fn, 'all_probs', []))
                    if y_true.size > 0 and y_prob.size > 0 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                        y_true = y_true.astype(int)
                        p1 = y_prob[:, 1]

                        def _ci(metric_fn: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float]:
                            rng = np.random.default_rng(seed)
                            vals = []
                            n = y_true.shape[0]
                            for _ in range(n_boot):
                                idx = rng.integers(0, n, size=n)
                                yt = y_true[idx]
                                yp = p1[idx]
                                try:
                                    vals.append(float(metric_fn(yt, yp)))
                                except Exception:
                                    continue
                            if len(vals) == 0:
                                return (float('nan'), float('nan'))
                            lo = float(np.percentile(vals, 100 * (alpha / 2)))
                            hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
                            return (lo, hi)

                        # Define metric wrappers
                        def _roc_auc(yt, yp):
                            return roc_auc_score(yt, yp)
                        def _pr_auc(yt, yp):
                            return average_precision_score(yt, yp)
                        def _acc_thr(yt, yp):
                            return accuracy_score(yt, (yp >= 0.5).astype(int))
                        def _f1_thr(yt, yp):
                            return f1_score(yt, (yp >= 0.5).astype(int), average='macro')

                        ci_roc = _ci(_roc_auc)
                        ci_pr = _ci(_pr_auc)
                        ci_acc = _ci(_acc_thr)
                        ci_f1 = _ci(_f1_thr)
                        _log(f"ci(method=bootstrap n={n_boot} alpha={alpha}):")
                        _log(f"  roc_auc_ci: [{ci_roc[0]:.6f}, {ci_roc[1]:.6f}]")
                        _log(f"  pr_auc_ci:  [{ci_pr[0]:.6f}, {ci_pr[1]:.6f}]")
                        _log(f"  acc@0.5_ci: [{ci_acc[0]:.6f}, {ci_acc[1]:.6f}]")
                        _log(f"  f1_macro@0.5_ci: [{ci_f1[0]:.6f}, {ci_f1[1]:.6f}]")
                        print("\n=== Confidence Intervals (bootstrap) ===")
                        print(f"ROC-AUC CI: [{ci_roc[0]:.6f}, {ci_roc[1]:.6f}]")
                        print(f"PR-AUC  CI: [{ci_pr[0]:.6f}, {ci_pr[1]:.6f}]")
                        print(f"Acc@0.5 CI: [{ci_acc[0]:.6f}, {ci_acc[1]:.6f}]")
                        print(f"F1@0.5   CI: [{ci_f1[0]:.6f}, {ci_f1[1]:.6f}]")
                except Exception:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
