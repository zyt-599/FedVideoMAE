import argparse
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RWF2000Dataset, RLVSDataset, build_video_transform, build_video_transform_train
from data.partition import detect_dataset_type
from models import build_videomae_model, MLPHead, trainable_parameters_filter
from train.utils import load_config, default_device, make_adamw, set_seed

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"timm(\.|$)"
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting vit_.* in registry",
    category=UserWarning,
)

def build_datasets(data_cfg: dict):
    # Stronger training-time augmentation; eval keeps center-crop pipeline
    aug_cfg = data_cfg.get('aug', {}) or {}
    transform_train = build_video_transform_train(
        size=data_cfg.get('size', 224),
        jitter_b=float(aug_cfg.get('jitter_b', 0.2)),
        jitter_c=float(aug_cfg.get('jitter_c', 0.15)),
        jitter_s=float(aug_cfg.get('jitter_s', 0.15)),
        grayscale_p=float(aug_cfg.get('grayscale_p', 0.0)),
        erase_p=float(aug_cfg.get('erase_p', 0.0)),
        erase_scale=tuple(aug_cfg.get('erase_scale', (0.02, 0.08))),
        erase_ratio=tuple(aug_cfg.get('erase_ratio', (0.3, 3.3))),
    )
    transform_val = build_video_transform(size=data_cfg.get('size', 224))

    root = data_cfg['root']
    val_root = data_cfg.get('val_root')
    if not val_root:
        # Try to derive val root by replacing trailing 'train' with 'val'
        norm = os.path.normpath(root)
        parts = norm.split(os.sep)
        if parts and parts[-1].lower() == 'train':
            parts[-1] = 'val'
            val_root = os.sep.join(parts)
        else:
            parent = os.path.dirname(norm)
            val_root = os.path.join(parent, 'val')

    ds_type = detect_dataset_type(root)
    if ds_type == 'rwf2000':
        train_ds = RWF2000Dataset(
            root=root, transform=transform_train,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
            random_offset=True,
        )
        val_ds = RWF2000Dataset(
            root=val_root, transform=transform_val,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
            random_offset=False,
        )
    else:
        train_ds = RLVSDataset(
            root=root, transform=transform_train,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
            random_offset=True,
        )
        val_ds = RLVSDataset(
            root=val_root, transform=transform_val,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
            random_offset=False,
        )

    return train_ds, val_ds


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module, thr_metric: str = 'f1_macro') -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    seen = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.inference_mode():
        total_target = len(loader.dataset) if hasattr(loader, 'dataset') else None
        pbar = tqdm(total=total_target, desc="[val]", leave=False, ncols=120, unit='samples')
        for batch in loader:
            x = batch['video'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = torch.argmax(logits, dim=1)
            prob = nn.functional.softmax(logits, dim=1)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((pred == y).sum().item())
            seen += bs
            all_preds.append(pred.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_probs.append(prob.detach().cpu())
            cur_acc = total_correct / max(1, seen)
            cur_loss = total_loss / max(1, seen)
            pbar.set_postfix({"acc": f"{cur_acc:.4f}", "loss": f"{cur_loss:.4f}"})
            pbar.update(bs)

    avg_loss = total_loss / max(1, seen)
    acc = total_correct / max(1, seen)

    # Extended metrics via sklearn
    try:
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support
        labels_np = torch.cat(all_labels).numpy()
        preds_np = torch.cat(all_preds).numpy()
        probs_np = torch.cat(all_probs).numpy()
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels_np, preds_np, average='weighted', zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels_np, preds_np, average='macro', zero_division=0)

        # Threshold scan (optimize selected metric)
        metric_name = (thr_metric or 'f1_macro').strip().lower()
        best_thr = 0.5
        best_score = -1.0
        p1 = probs_np[:, 1]
        for thr in [x / 100.0 for x in range(10, 91)]:  # 0.10..0.90
            preds_thr = (p1 >= thr).astype(int)
            p_thr, r_thr, f1_m_thr, _ = precision_recall_fscore_support(labels_np, preds_thr, average='macro', zero_division=0)
            if metric_name in ('accuracy', 'acc'):
                score = float((preds_thr == labels_np).mean())
            elif metric_name == 'precision_macro':
                score = float(p_thr)
            elif metric_name == 'recall_macro':
                score = float(r_thr)
            else:
                score = float(f1_m_thr)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        extra = {
            'val_precision_weighted': float(p_w),
            'val_recall_weighted': float(r_w),
            'val_f1_weighted': float(f1_w),
            'val_precision_macro': float(p_m),
            'val_recall_macro': float(r_m),
            'val_f1_macro': float(f1_m),
            'val_best_threshold': float(best_thr),
            'val_best_threshold_metric': metric_name,
            'val_best_threshold_score': float(best_score),
        }
    except Exception:
        extra = {}

    out = {
        'val_loss': avg_loss,
        'val_acc': acc,
        'total': seen,
        'correct': total_correct,
    }
    out.update(extra)
    return out


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = default_device()
    # Set up logging (group by DP/epsilon inferred from checkpoint path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _infer_dp_subdir(cfg: dict) -> str:
        try:
            ckpt = (cfg or {}).get('model', {}).get('pretrained_checkpoint', '')
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
    log_path = os.path.join(log_dir, f'small_head_finetune_{ts}.txt')
    def _log(msg: str):
        try:
            with open(log_path, 'a', encoding='utf-8') as _f:
                _f.write(msg.rstrip('\n') + '\n')
        except Exception:
            pass
    try:
        _log('=== Task: small_head_finetune ===')
        _log(f'timestamp: {ts}')
        _log('config:')
        _log(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        pass
    # Optional determinism for maximum stability
    det = bool(cfg.get('training', {}).get('deterministic', False))
    if det:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            print('[Init] Deterministic mode enabled')
        except Exception:
            pass

    out_dir = cfg.get('output_dir', 'runs/central_weak_finetune')
    os.makedirs(out_dir, exist_ok=True)

    # Build datasets and loaders
    data_cfg = cfg['data']
    train_ds, val_ds = build_datasets(data_cfg)

    from data.collate import video_collate
    # Optional weighted sampler to balance classes
    sampler = None
    if bool(cfg.get('training', {}).get('weighted_sampler', False)):
        try:
            items = getattr(train_ds, 'items', None)
            if items is not None:
                import numpy as np
                labels = np.array([y for _, y in items], dtype=np.int64)
                classes, counts = np.unique(labels, return_counts=True)
                weights_per_class = {int(c): (1.0 / n) if n > 0 else 0.0 for c, n in zip(classes, counts)}
                sample_weights = [weights_per_class[int(y)] for _, y in items]
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        except Exception:
            sampler = None

    pm = bool(data_cfg.get('pin_memory', True))
    val_pm = bool(data_cfg.get('val_pin_memory', pm))

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=video_collate,
        pin_memory=pm,
        persistent_workers=(data_cfg.get('num_workers', 4) > 0),
    )
    # Use a smaller validation batch size to avoid peak VRAM spikes at epoch boundary
    val_bs = int(max(1, data_cfg.get('val_batch_size', max(1, data_cfg.get('batch_size', 16) // 2))))
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=max(0, data_cfg.get('num_workers', 4) // 2),
        collate_fn=video_collate,
        pin_memory=val_pm,
        persistent_workers=(data_cfg.get('num_workers', 4) > 0),
    )

    # Build model (small head finetune style)
    model_cfg = cfg['model']
    is_full_finetune = False
    if model_cfg.get('pretrained_checkpoint'):
        print(f"[Init] Using pretrained checkpoint: {model_cfg['pretrained_checkpoint']}")
    backbone = build_videomae_model(
        model_name=model_cfg.get('model_name'),
        pretrained=False,  # wrapper ignores this and loads only if pretrained_checkpoint is provided
        mode='head' if is_full_finetune else 'feature',
        num_classes=model_cfg.get('head', {}).get('num_classes', 2),
        peft_config=model_cfg.get('peft', None),
        head_config=model_cfg.get('head', {}) if is_full_finetune else None,
        pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None),
        num_frames=data_cfg.get('num_frames', 16)
    )

    # After model is built (and LoRA injected), try to load PEFT trainable params from checkpoint['trainable']
    try:
        ckpt_path_peft = str(model_cfg.get('pretrained_checkpoint', ''))
        if ckpt_path_peft and os.path.isfile(ckpt_path_peft):
            ck = torch.load(ckpt_path_peft, map_location='cpu', weights_only=False)
            tr = ck.get('trainable', None)
            if isinstance(tr, dict) and len(tr) > 0:
                ms = backbone.state_dict()
                to_load = {k: v for k, v in tr.items() if k in ms and getattr(ms[k], 'shape', None) == getattr(v, 'shape', None)}
                n_all = len(tr)
                n_ok = len(to_load)
                if n_ok > 0:
                    # Load trainable PEFT weights (strict=False to ignore any leftover keys)
                    try:
                        backbone.load_state_dict(to_load, strict=False)
                    except Exception:
                        # Fallback: load into encoder/backbone if wrapper enforces nested modules
                        try:
                            backbone.backbone.load_state_dict(to_load, strict=False)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    msg = f"[Init] Loaded PEFT trainable from checkpoint: matched={n_ok}/{n_all}"
                    print(msg); _log(msg)
                else:
                    msg = f"[Init] No PEFT trainable matched current model (ckpt keys={n_all}); ensure LoRA cfg (r/targets) matches pretrain"
                    print(msg); _log(msg)
    except Exception:
        pass

    # Log LoRA fingerprint (A/B matrices) for deterministic comparability across runs
    try:
        import hashlib
        sd = backbone.state_dict()
        keys = sorted([k for k in sd.keys() if k.endswith('.A') or k.endswith('.B')])
        h = hashlib.sha256()
        total = 0
        for k in keys:
            t = sd[k].detach().cpu().contiguous().numpy()
            total += int(t.size)
            h.update(k.encode('utf-8'))
            h.update(t.tobytes(order='C'))
        msg = f"[Init] LoRA fingerprint: keys={len(keys)} elems={total} sha256={h.hexdigest()[:16]}"
        print(msg); _log(msg)
    except Exception:
        pass

    # Verify checkpoint loading & model fingerprint
    try:
        import hashlib
        ckpt_path = str(model_cfg.get('pretrained_checkpoint', ''))
        def _sha256(p: str, max_bytes: int = 0) -> str:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                if max_bytes and max_bytes > 0:
                    h.update(f.read(max_bytes))
                else:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(chunk)
            return h.hexdigest()
        if ckpt_path and os.path.isfile(ckpt_path):
            ck_hash = _sha256(ckpt_path)
            msg = f"[Init] Using pretrained checkpoint: {ckpt_path} | sha256={ck_hash[:16]}"
            print(msg); _log(msg)
        # Encoder fingerprint (selected tensors)
        enc = getattr(getattr(backbone, 'backbone', None), 'encoder', None)
        def _tensor_digest(t) -> str:
            try:
                import numpy as _np
                arr = t.detach().float().cpu().numpy().ravel()
                # sample first up to 1e6 elements
                arr = arr[: min(arr.size, 1000000)]
                return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            except Exception:
                return 'na'
        if enc is not None:
            fp_parts = []
            try:
                if hasattr(enc, 'pos_embed'):
                    fp_parts.append(('pos_embed', _tensor_digest(enc.pos_embed)))
            except Exception:
                pass
            # find a stable early block weight
            try:
                blk0 = enc.blocks[0]
                w = None
                if hasattr(blk0.attn, 'qkv') and hasattr(blk0.attn.qkv, 'weight'):
                    w = blk0.attn.qkv.weight
                elif hasattr(blk0.mlp, 'fc1') and hasattr(blk0.mlp.fc1, 'weight'):
                    w = blk0.mlp.fc1.weight
                if w is not None:
                    fp_parts.append(('blk0', _tensor_digest(w)))
            except Exception:
                pass
            if fp_parts:
                fp_str = ' '.join(f"{k}={v}" for k, v in fp_parts)
                msg = f"[Init] Encoder fingerprint: {fp_str}"
                print(msg); _log(msg)
            # Whole-encoder combined hash (robust difference check)
            try:
                import itertools
                h = hashlib.sha256()
                for k, v in enc.state_dict().items():
                    if hasattr(v, 'detach'):
                        t = v.detach().cpu().contiguous()
                        h.update(k.encode('utf-8'))
                        h.update(t.numpy().tobytes(order='C'))
                enc_hash = h.hexdigest()[:16]
                msg = f"[Init] Encoder full-hash: {enc_hash}"
                print(msg); _log(msg)
            except Exception:
                pass
    except Exception:
        pass

    # Freeze backbone; LoRA control
    peft_cfg = model_cfg.get('peft', {}) or {}
    use_lora = bool(peft_cfg.get('use_lora', False))
    # backprop_backbone: whether to backprop through backbone (default matches use_lora, can disable in config for speedup)
    backprop_backbone = bool(peft_cfg.get('backprop_backbone', use_lora))
    for p in backbone.parameters():
        p.requires_grad = False
    if use_lora:
        try:
            from models.peft_lora import LoRALinear
            # 1) Enable LoRA gradients on demand (when backprop_backbone is true)
            for m in backbone.modules():
                if isinstance(m, LoRALinear):
                    if getattr(m, 'A', None) is not None:
                        m.A.requires_grad = bool(backprop_backbone)
                    if getattr(m, 'B', None) is not None:
                        m.B.requires_grad = bool(backprop_backbone)
            # 2) Optional: train only LoRA in last K encoder blocks for further speedup
            last_k = int(peft_cfg.get('train_last_blocks', peft_cfg.get('last_trainable_blocks', 0)) or 0)
            if backprop_backbone and last_k and last_k > 0:
                try:
                    enc = backbone.backbone.encoder  # type: ignore[attr-defined]
                    total_blocks = len(list(enc.blocks))
                    cutoff = max(0, total_blocks - last_k)
                    for name, module in backbone.named_modules():
                        if isinstance(module, LoRALinear) and '.encoder.blocks.' in name:
                            try:
                                # parse block index from name
                                # e.g. 'backbone.encoder.blocks.12.attn.q_proj'
                                parts = name.split('.encoder.blocks.')[-1].split('.')
                                blk_idx = int(parts[0])
                            except Exception:
                                blk_idx = None
                            if blk_idx is not None and blk_idx < cutoff:
                                if getattr(module, 'A', None) is not None:
                                    module.A.requires_grad = False
                                if getattr(module, 'B', None) is not None:
                                    module.B.requires_grad = False
                    print(f"[Init] LoRA train_last_blocks={last_k} (total={total_blocks}); earlier blocks' LoRA frozen")
                except Exception:
                    pass
        except Exception:
            pass
    # Optional: unfreeze last N encoder blocks with low LR
    train_cfg = cfg.get('training', {})
    schedule_cfg = train_cfg.get('unfreeze_schedule', [])
    unfreeze_schedule = []
    if isinstance(schedule_cfg, (list, tuple)):
        for item in schedule_cfg:
            try:
                epoch = int(item.get('epoch'))
                last_blocks = int(item.get('last_blocks', 0))
                bb_lr_item = float(item.get('backbone_lr', train_cfg.get('backbone_lr', 1e-5)))
                if epoch > 0 and last_blocks > 0:
                    unfreeze_schedule.append({'epoch': epoch, 'last_blocks': last_blocks, 'backbone_lr': bb_lr_item})
            except Exception:
                continue
    unfreeze_schedule.sort(key=lambda x: x['epoch'])
    has_schedule = len(unfreeze_schedule) > 0

    # Gradient checkpointing explicit switch: strictly follow config boolean value
    try:
        gc_enable = bool(cfg.get('training', {}).get('grad_checkpoint', True)) and (use_lora or has_schedule)
        if hasattr(backbone.backbone, 'encoder') and hasattr(backbone.backbone.encoder, 'use_checkpoint'):
            setattr(backbone.backbone.encoder, 'use_checkpoint', bool(gc_enable))  # type: ignore[attr-defined]
            print('[Init] Gradient checkpointing enabled for encoder' if gc_enable else '[Init] Gradient checkpointing disabled for encoder')
    except Exception:
        pass

    in_dim = backbone._feat_dim()
    head_cfg = model_cfg.get('head', {})
    head = MLPHead(
        in_dim,
        head_cfg.get('hidden_dim', 512),
        head_cfg.get('num_classes', 2),
        num_layers=head_cfg.get('num_layers', 1),
        dropout=head_cfg.get('dropout', 0.0),
    )

    # Wrap model to optionally detach backbone outputs when LoRA is disabled
    class BackboneHead(nn.Module):
        def __init__(self, bb: nn.Module, head: nn.Module, detach_feats: bool):
            super().__init__()
            self.bb = bb
            self.head = head
            self.detach = detach_feats

        def forward(self, x):
            feats = self.bb(x)
            if self.detach:
                feats = feats.detach()
            return self.head(feats)

    # When backprop_backbone is False, detach features to avoid gradient computation on backbone (speedup without affecting LoRA inference)
    model = BackboneHead(backbone, head, detach_feats=not backprop_backbone)
    model.to(device)
    # EMA support
    ema_cfg = cfg.get('training', {}).get('ema', {}) if isinstance(cfg.get('training', {}).get('ema', {}), dict) else {}
    use_ema = bool(ema_cfg.get('enable', False))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    ema_update_every = int(ema_cfg.get('update_every', 1)) if use_ema else 1
    ema = None
    if use_ema:
        import copy
        class _ModelEMA:
            def __init__(self, src: nn.Module, decay: float = 0.999):
                self.ema = copy.deepcopy(src).to(device)
                for p in self.ema.parameters():
                    p.requires_grad = False
                self.decay = decay
            @torch.no_grad()
            def update(self, src: nn.Module):
                # Only update EMA for trainable parameters (LoRA + head, or unfrozen blocks)
                # This drastically reduces per-step overhead.
                trainable_names = [n for n, p in src.named_parameters() if p.requires_grad]
                if not trainable_names:
                    return
                msd = src.state_dict()
                esd = self.ema.state_dict()
                for k in trainable_names:
                    v_ema = esd.get(k, None)
                    v_src = msd.get(k, None)
                    if v_ema is None or v_src is None:
                        continue
                    src_val = v_src.detach()
                    # Align device/dtype for safe arithmetic
                    try:
                        if src_val.device != v_ema.device:
                            src_val = src_val.to(v_ema.device)
                        if src_val.dtype != v_ema.dtype:
                            src_val = src_val.to(v_ema.dtype)
                    except Exception:
                        pass
                    if v_ema.shape != src_val.shape:
                        continue
                    # EMA blend (float only)
                    if torch.is_floating_point(v_ema) and torch.is_floating_point(src_val):
                        v_ema.copy_(v_ema * self.decay + src_val * (1.0 - self.decay))
                    else:
                        v_ema.copy_(src_val)
        ema = _ModelEMA(model, decay=ema_decay)

    # Optimizer / scheduler (param groups: backbone last blocks vs LoRA/head)
    lr = float(train_cfg.get('lr', 5e-4))
    bb_lr = float(train_cfg.get('backbone_lr', 1e-5))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))

    main_params = []  # head + LoRA (+ prompt)
    bb_params = []    # unfrozen encoder blocks (base weights only)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Route LoRA low-rank matrices to main_params (use higher LR)
        if name.endswith('.A') or name.endswith('.B'):
            main_params.append(p)
            continue
        # Route unfrozen base encoder block params to bb_params (low LR)
        if 'bb.backbone.encoder.blocks.' in name:
            bb_params.append(p)
            continue
        # Everything else (head, recon_prompt, etc.) goes to main
        main_params.append(p)
    param_groups = []
    if main_params:
        param_groups.append({'params': main_params, 'lr': lr, 'weight_decay': weight_decay})
    if bb_params and not has_schedule:
        param_groups.append({'params': bb_params, 'lr': bb_lr, 'weight_decay': weight_decay})
    optimizer = make_adamw(param_groups if len(param_groups) > 1 else main_params, lr=lr, weight_decay=weight_decay)
    try:
        n_main = sum(p.numel() for p in main_params)
        n_bb = sum(p.numel() for p in bb_params)
        print(f"[Init] Optim groups: main={n_main} params @ lr={lr}, bb={n_bb} params @ lr={bb_lr}")
        _log(f"[Init] Optim groups: main={n_main} params @ lr={lr}, bb={n_bb} params @ lr={bb_lr}")
        # Trainable params breakdown
        total_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and (n.endswith('.A') or n.endswith('.B')))
        n_head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and (n.startswith('head.')))
        # backbone base weights only (exclude LoRA A/B)
        n_bb_train = sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad
            and ('bb.backbone.encoder.blocks.' in n)
            and (not n.endswith('.A'))
            and (not n.endswith('.B'))
        )
        msg = f"[Init] Trainable params: total={total_trainable} | lora={n_lora} | head={n_head} | backbone={n_bb_train}"
        print(msg); _log(msg)
    except Exception:
        pass
    seen_params = set()
    for _pg in optimizer.param_groups:
        for _p in _pg['params']:
            seen_params.add(id(_p))

    # Warmup + cosine scheduler (epoch-based)
    scheduler = None
    lr_mode = train_cfg.get('lr_scheduler', 'cosine')
    warmup_epochs = int(train_cfg.get('warmup_epochs', 0)) if lr_mode == 'cosine' else 0
    total_epochs = int(train_cfg.get('epochs', 10))
    min_lr_cfg = float(train_cfg.get('min_lr', 0.0)) if lr_mode == 'cosine' else 0.0
    if lr_mode == 'cosine' and min_lr_cfg < 0.0:
        min_lr_cfg = 0.0

    def _lr_factor(cur_epoch: int) -> float:
        if lr_mode != 'cosine':
            return 1.0
        if cur_epoch < warmup_epochs:
            return float(cur_epoch) / max(1, warmup_epochs)
        import math
        prog = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def _apply_lr(cur_epoch: int):
        if lr_mode != 'cosine':
            return
        factor = _lr_factor(cur_epoch)
        for pg in optimizer.param_groups:
            base_lr = pg.get('base_lr', pg['lr'])
            pg['base_lr'] = base_lr
            target = base_lr * factor
            if min_lr_cfg > 0.0 and base_lr >= min_lr_cfg:
                target = max(min_lr_cfg, target)
            pg['lr'] = target

    # Loss function (weak supervision could use class weights or focal loss)
    # Robust class_weights parsing: accept list/tuple of numbers or strings, a comma-separated string,
    # or a dict mapping (e.g., {0: 1.2, 1: 1.0})
    def _parse_class_weights(cfg_val):
        parsed_local = None
        if isinstance(cfg_val, (list, tuple)):
            parsed_local = [float(x) for x in cfg_val]
        elif isinstance(cfg_val, str):
            s_ = cfg_val.strip()
            if s_.startswith('[') and s_.endswith(']'):
                s_ = s_[1:-1]
            items_ = [p.strip() for p in s_.split(',') if p.strip()]
            if items_:
                parsed_local = [float(x) for x in items_]
        elif isinstance(cfg_val, dict):
            if all(k in cfg_val for k in (0, 1)):
                parsed_local = [float(cfg_val[0]), float(cfg_val[1])]
            else:
                keys_ = ['NonFight', 'Fight']
                if all(k in cfg_val for k in keys_):
                    parsed_local = [float(cfg_val['NonFight']), float(cfg_val['Fight'])]
        return parsed_local

    class_weights_cfg = train_cfg.get('class_weights', None)
    weights = None
    try:
        parsed = _parse_class_weights(class_weights_cfg)
        if parsed is not None:
            weights = torch.tensor(parsed, dtype=torch.float32, device=device)
    except Exception:
        weights = None

    # Holder to allow window-based adjustment of class weights during training
    weights_holder = { 'w': weights }

    use_focal = bool(train_cfg.get('use_focal', False))
    gamma = float(train_cfg.get('focal_gamma', 2.0))

    # label smoothing holder so we can adjust in-window
    ls_holder = { 'ls': float(train_cfg.get('label_smoothing', 0.0)) }
    def _ce_loss(logits, targets):
        ls_val = float(ls_holder.get('ls', 0.0))
        return nn.functional.cross_entropy(logits, targets, weight=weights_holder['w'], label_smoothing=ls_val if ls_val > 0.0 else 0.0)

    def focal_loss(logits, targets, gamma: float = 2.0):
        ce = nn.functional.cross_entropy(logits, targets, weight=weights_holder['w'], reduction='none')
        with torch.inference_mode():
            probs = nn.functional.softmax(logits, dim=1)
            pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-6)
        loss = ((1.0 - pt) ** gamma) * ce
        return loss.mean()

    def loss_fn_logits(logits, targets):
        if use_focal:
            return focal_loss(logits, targets, gamma=gamma)
        return _ce_loss(logits, targets)

    # AMP
    use_amp = bool(train_cfg.get('use_amp', False))
    try:
        from torch.amp import autocast as _autocast, GradScaler as _GradScaler  # type: ignore
        amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = _GradScaler(amp_device, enabled=use_amp)
        autocast_ctx = lambda: _autocast(amp_device, enabled=use_amp)
    except Exception:
        from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler  # type: ignore
        scaler = _GradScaler(enabled=use_amp)
        autocast_ctx = lambda: _autocast(enabled=use_amp)

    # Train loop with validation and best model saving
    epochs = train_cfg.get('epochs', 10)
    log_every = cfg.get('logging', {}).get('log_every', 50)
    clip_grad = train_cfg.get('clip_grad', 0.0)
    best_score = None
    best_path = os.path.join(out_dir, 'model_best.pth')

    accum = max(1, int(train_cfg.get('grad_accum_steps', 1)))
    micro_bs = int(train_cfg.get('micro_batch_size', 0))
    clear_steps = int(train_cfg.get('clear_cache_steps', 0))
    oom_autotune = not bool(train_cfg.get('disable_oom_autotune', False))
    # MixUp configuration (optionally delayed until first backbone unfreeze)
    mix_cfg_raw = train_cfg.get('mixup', {})
    mix_cfg = mix_cfg_raw if isinstance(mix_cfg_raw, dict) else {}
    mixup_enabled = bool(mix_cfg.get('enable', False))
    mixup_alpha = float(mix_cfg.get('alpha', 0.0))
    mixup_prob = float(mix_cfg.get('prob', 1.0))
    mixup_prob = min(1.0, max(0.0, mixup_prob))
    mixup_after_first_unfreeze = bool(mix_cfg.get('start_after_first_unfreeze', False))
    # allow config to control when to stop mixup (default=50 to preserve previous behavior)
    mixup_disable_after_epoch = int(mix_cfg.get('disable_after_epoch', 50)) if mixup_enabled else None
    beta_dist = None
    if mixup_enabled and mixup_alpha > 0.0:
        from torch.distributions import Beta
        beta_dist = Beta(mixup_alpha, mixup_alpha)
    else:
        mixup_enabled = False
        mixup_alpha = 0.0

    mixup_active = mixup_enabled and (not mixup_after_first_unfreeze or not has_schedule)

    # Optional window adjustments: allow mid-training tweaks for smoothing/mixup/ema
    win_cfg = train_cfg.get('window_adjustments', [])
    def _apply_window(epoch: int):
        try:
            for w in (win_cfg or []):
                se = int(w.get('start_epoch', 0)); ee = int(w.get('end_epoch', -1))
                if se <= epoch <= ee:
                    changed = []
                    if 'label_smoothing' in w:
                        ls_holder['ls'] = float(w['label_smoothing']); changed.append(f"ls={ls_holder['ls']}")
                    if 'mixup_prob' in w:
                        nonlocal mixup_prob
                        mixup_prob = float(w['mixup_prob']); changed.append(f"mixup_prob={mixup_prob}")
                    if 'ema_decay' in w and ('ema' in locals() and ema is not None):
                        ema.decay = float(w['ema_decay']); changed.append(f"ema_decay={ema.decay}")
                    if 'class_weights' in w:
                        cw = w.get('class_weights')
                        parsed_cw = _parse_class_weights(cw)
                        if parsed_cw is not None:
                            try:
                                weights_holder['w'] = torch.tensor(parsed_cw, dtype=torch.float32, device=device)
                                changed.append(f"class_weights={parsed_cw}")
                            except Exception:
                                pass
                    if changed:
                        print(f"[Schedule] Window adjustments at epoch {epoch}: " + ', '.join(changed))
        except Exception:
            pass


    # Control whether unfreeze_schedule affects base encoder weights (heavy) or only LoRA A/B (lightweight)
    unfreeze_base = bool(train_cfg.get('unfreeze_base', False))
    schedule_idx = 0
    for epoch in range(1, int(epochs) + 1):
        if mixup_disable_after_epoch is not None and epoch >= mixup_disable_after_epoch:
            mixup_active = False
            print(f"Mixup Disabled after Epoch:{epoch}")
        while schedule_idx < len(unfreeze_schedule) and epoch >= unfreeze_schedule[schedule_idx]['epoch']:
            event = unfreeze_schedule[schedule_idx]
            last_blocks_event = max(0, int(event.get('last_blocks', 0)))
            event_lr = float(event.get('backbone_lr', bb_lr))
            newly_trainable = []
            try:
                enc = backbone.backbone.encoder  # type: ignore[attr-defined]
                blocks = list(enc.blocks)
                sel = blocks[-last_blocks_event:] if last_blocks_event <= len(blocks) else blocks
                if unfreeze_base:
                    # Heavy mode: unfreeze base encoder weights of selected blocks
                    for blk in sel:
                        for p in blk.parameters():
                            if not p.requires_grad:
                                p.requires_grad = True
                                newly_trainable.append(p)
                else:
                    # Lightweight mode: only enable LoRA A/B in selected blocks
                    from models.peft_lora import LoRALinear  # type: ignore
                    # Determine selected block indices
                    total_blocks = len(blocks)
                    cutoff = max(0, total_blocks - last_blocks_event)
                    for name, module in backbone.named_modules():
                        if isinstance(module, LoRALinear) and '.encoder.blocks.' in name:
                            try:
                                parts = name.split('.encoder.blocks.')[-1].split('.')
                                blk_idx = int(parts[0])
                            except Exception:
                                blk_idx = None
                            if blk_idx is None or blk_idx < cutoff:
                                continue
                            # Enable A/B only
                            if getattr(module, 'A', None) is not None and not module.A.requires_grad:
                                module.A.requires_grad = True
                                newly_trainable.append(module.A)
                            if getattr(module, 'B', None) is not None and not module.B.requires_grad:
                                module.B.requires_grad = True
                                newly_trainable.append(module.B)
            except Exception:
                newly_trainable = []
            newly_trainable = [p for p in newly_trainable if id(p) not in seen_params]
            if newly_trainable:
                if hasattr(model, 'detach'):
                    setattr(model, 'detach', False)
                optimizer.add_param_group({'params': newly_trainable, 'lr': event_lr, 'weight_decay': weight_decay})
                new_group = optimizer.param_groups[-1]
                new_group['base_lr'] = event_lr
                for _p in newly_trainable:
                    seen_params.add(id(_p))
                try:
                    total_new = sum(p.numel() for p in newly_trainable)
                except Exception:
                    total_new = 0
                mode_tag = 'base+LoRA' if unfreeze_base else 'LoRA-only'
                print(f"[Schedule] Unfroze last {last_blocks_event} blocks ({mode_tag}) at epoch {epoch}; added {total_new} params @ lr={event_lr}")
                _apply_lr(epoch)
                if mixup_after_first_unfreeze and mixup_enabled:
                    mixup_active = True
            schedule_idx += 1

        _apply_lr(epoch)
        _apply_window(epoch)
        model.train()
        total = 0
        correct = 0.0
        running_loss = 0.0
        total_train_samples = len(train_ds)
        pbar = tqdm(total=total_train_samples, desc=f"Epoch {epoch}/{epochs} [train]", ncols=120, unit='samples')
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader, start=1):
            x = batch['video'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            B = y.size(0)
            mixup_applied = False
            lam = 1.0
            y_mix_partner = None
            if mixup_active and B > 1:
                if mixup_prob >= 1.0 or float(torch.rand(1).item()) < mixup_prob:
                    lam = float(beta_dist.sample().item())
                    if lam < 0.5:
                        lam = 1.0 - lam
                    perm = torch.randperm(B, device=y.device)
                    x_perm = x[perm].clone()
                    x = lam * x + (1.0 - lam) * x_perm
                    y_mix_partner = y[perm]
                    mixup_applied = True
            local_micro = micro_bs if (micro_bs and micro_bs > 0 and micro_bs < B) else B

            # OOM-resilient micro-batch loop
            c = 0
            while c * local_micro < B:
                s = c * local_micro
                e = min(B, s + local_micro)
                x_c = x[s:e]
                y_c = y[s:e]
                try:
                    y_mix_c = None
                    with autocast_ctx():
                        logits = model(x_c)
                        if mixup_applied:
                            y_mix_c = y_mix_partner[s:e]
                            loss_main = loss_fn_logits(logits, y_c)
                            loss_aux = loss_fn_logits(logits, y_mix_c)
                            loss_raw = lam * loss_main + (1.0 - lam) * loss_aux
                        else:
                            loss_raw = loss_fn_logits(logits, y_c)
                        loss = loss_raw / (accum * max(1, (B + local_micro - 1) // local_micro))
                    scaler.scale(loss).backward()

                    running_loss += float(loss_raw.item()) * y_c.size(0)
                    pred = torch.argmax(logits, dim=1)
                    if mixup_applied:
                        mix_correct_a = (pred == y_c).sum().item()
                        mix_correct_b = (pred == y_mix_c).sum().item()
                        correct += mix_correct_a * lam + mix_correct_b * (1.0 - lam)
                    else:
                        correct += int((pred == y_c).sum().item())
                    total += int(y_c.size(0))
                    c += 1
                except RuntimeError as err:
                    if 'CUDA out of memory' in str(err):
                        # Either auto-tune to continue, or raise to keep training strict & deterministic
                        if not oom_autotune:
                            raise
                        torch.cuda.empty_cache()
                        if local_micro > 1:
                            local_micro = max(1, local_micro // 2)
                            print(f"[OOM] Reduced micro_batch_size to {local_micro} and retrying...")
                            continue
                        else:
                            accum *= 2
                            print(f"[OOM] Increased grad_accum_steps to {accum} and retrying...")
                            continue
                    else:
                        raise
            # Only unscale + clip right before stepping to avoid double unscale across accumulation steps
            if (i % accum == 0) or (i == len(train_loader)):
                if clip_grad and clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_parameters_filter(model), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                if use_ema and ema is not None:
                    if ema_update_every <= 1:
                        ema.update(model)
                    else:
                        # Only update EMA every N optimizer steps
                        if (i % max(1, ema_update_every)) == 0:
                            ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            if i % max(1, log_every) == 0:
                cur_loss = running_loss / max(1, total)
                cur_acc = correct / max(1, total)
                pbar.set_postfix({"loss": f"{cur_loss:.4f}", "acc": f"{cur_acc:.4f}"})
            # advance by actual samples processed
            pbar.update(y.size(0))

            if clear_steps and (i % clear_steps == 0):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        # ensure bar shows final stats
        cur_loss = running_loss / max(1, total)
        cur_acc = correct / max(1, total)
        pbar.set_postfix({"loss": f"{cur_loss:.4f}", "acc": f"{cur_acc:.4f}"})
        pbar.close()


        # Validation (prefer EMA if available)
        thr_metric = (train_cfg.get('threshold_scan_metric', 'f1_macro') or 'f1_macro')
        eval_model = ema.ema if ('ema' in locals() and ema is not None) else model
        metrics = evaluate(eval_model, val_loader, device, _ce_loss, thr_metric)
        # Pretty print key metrics
        line = f"[Val] epoch={epoch}"
        if 'val_acc' in metrics:
            line += f" acc={metrics['val_acc']:.6f}"
        if 'val_loss' in metrics:
            line += f" loss={metrics['val_loss']:.6f}"
        if 'val_f1_macro' in metrics:
            line += f" f1_macro={metrics['val_f1_macro']:.6f}"
        if 'val_f1_weighted' in metrics:
            line += f" f1_weighted={metrics['val_f1_weighted']:.6f}"
        if 'val_best_threshold' in metrics and 'val_best_threshold_score' in metrics:
            tag = metrics.get('val_best_threshold_metric', 'f1_macro')
            line += f" | thr@{tag} {metrics['val_best_threshold']:.2f}/{metrics['val_best_threshold_score']:.6f}"
        print(line + "\n")
        _log(line)

        # Save best by configurable metric
        sel = train_cfg.get('select_best_metric', 'acc')
        # normalize metric key
        key = sel.strip().lower()
        if key in ('acc', 'accuracy'):
            metric_key = 'val_acc'; maximize = True
        elif key in ('loss', 'val_loss'):
            metric_key = 'val_loss'; maximize = False
        elif key in ('f1_macro',):
            metric_key = 'val_f1_macro'; maximize = True
        elif key in ('f1_weighted',):
            metric_key = 'val_f1_weighted'; maximize = True
        elif key in ('precision_macro',):
            metric_key = 'val_precision_macro'; maximize = True
        elif key in ('recall_macro',):
            metric_key = 'val_recall_macro'; maximize = True
        elif key in ('f1_macro_thr', 'thresholded_f1_macro', 'thr_metric'):
            # Use scanned-best metric score (depends on threshold_scan_metric)
            metric_key = 'val_best_threshold_score'; maximize = True
        else:
            metric_key = 'val_acc'; maximize = True

        cur_val = float(metrics.get(metric_key, 0.0))
        cur_score = cur_val if maximize else -cur_val
        if best_score is None or cur_score >= best_score:
            best_score = cur_score
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                **({'model_ema': ema.ema.state_dict()} if (ema is not None) else {}),
                'optimizer': optimizer.state_dict(),
                **{k: v for k, v in metrics.items() if k.startswith('val_')},
            }, best_path)
            print(f"[Save] New best ({metric_key}={cur_val:.4f}) saved to {best_path}\n")
            _log(f"[Best] epoch={epoch} {metric_key}={cur_val:.4f} saved={best_path}")

        # Optional cache clear to release allocator caches after val
        if bool(train_cfg.get('clear_cache_each_epoch', False)):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Only keep best model; skip saving final snapshot per request
    _log('[Done] finetune completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Centralized weakly supervised small-head fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args)
