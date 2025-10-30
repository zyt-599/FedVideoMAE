import argparse
import torch
import warnings
import torch.nn.functional as F
from datetime import datetime
import yaml
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

# Suppress specific noisy warnings (timm deprecations & model registry overwrites)
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RWF2000Dataset, RLVSDataset, build_video_transform
from data.partition import detect_dataset_type
from models import build_videomae_model, PretrainLoss
from data.collate import video_collate
from train.utils import load_config, default_device, set_seed
import torch.nn.functional as F


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def calculate_psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate PSNR metric."""
    mse = F.mse_loss(recon, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_ssim(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate SSIM metric (simplified version)."""
    # Simplified SSIM calculation
    mu1 = recon.mean()
    mu2 = target.mean()
    sigma1 = recon.var()
    sigma2 = target.var()
    sigma12 = ((recon - mu1) * (target - mu2)).mean()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return ssim.item()


def evaluate_reconstruction_quality(model, dataloader, device, num_samples=50) -> Dict[str, float]:
    """Evaluate reconstruction quality."""
    model.eval()
    psnr_values = []
    ssim_values = []
    mse_values = []
    mae_values = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating reconstruction quality", total=min(num_samples, len(dataloader)))
        for i, batch in enumerate(pbar):
            if i >= num_samples:
                break
                
            x = batch['video'].to(device)
            
            # Get reconstruction results
            recon, target, mask, _ = model(x, return_mask=True)
            
            # Calculate metrics
            psnr = calculate_psnr(recon, target)
            ssim = calculate_ssim(recon, target)
            mse = F.mse_loss(recon, target).item()
            mae = F.l1_loss(recon, target).item()
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)
            mae_values.append(mae)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'PSNR': f'{psnr:.2f}',
                'SSIM': f'{ssim:.4f}',
                'MSE': f'{mse:.4f}'
            })
    
    return {
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
    }


def _threshold_scan(labels_np, probs_np, metric='f1_macro', thr_min: float = 0.10, thr_max: float = 0.90, thr_step: float = 0.01):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    best_thr = 0.5
    best_metric = -1
    best_stats = None
    p1 = probs_np[:, 1]
    import numpy as _np
    thr_min = float(thr_min); thr_max = float(thr_max); thr_step = float(thr_step)
    if thr_step <= 0:
        thr_step = 0.01
    # inclusive range
    grid = _np.arange(thr_min, thr_max + 1e-8, thr_step)
    for thr in grid:
        preds = (p1 >= thr).astype(int)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_np, preds, average='macro', zero_division=0)
        if metric == 'accuracy':
            val = (preds == labels_np).mean()
        else:
            val = {'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro}[metric]
        if val > best_metric:
            best_metric = val
            best_thr = thr
            best_stats = (preds, precision_macro, recall_macro, f1_macro, confusion_matrix(labels_np, preds))
    return best_thr, best_metric, best_stats


def _probs_with_tta(model, x, tta_hflip: bool = False, temporal_clips: int = 1, deterministic_temporal: bool = False, *, logit_avg: bool = False, temperature: float = 1.0):
    """Compute probs with TTA. If logit_avg=True, average logits then softmax(logits/T). Else average probabilities of softmax(logits/T)."""
    with torch.no_grad():
        temporal_clips = max(1, int(temporal_clips))
        T = x.size(1)
        shifts = []
        if temporal_clips > 1 and deterministic_temporal:
            for k in range(temporal_clips):
                shifts.append(int(round(k * T / temporal_clips)) % max(1, T))

        if logit_avg:
            logits_accum = None
            for k in range(temporal_clips):
                xk = x
                if temporal_clips > 1 and xk.size(1) > 1:
                    shift = shifts[k] if deterministic_temporal else int(torch.randint(low=0, high=xk.size(1), size=(1,), device=xk.device).item())
                    xk = torch.roll(xk, shifts=shift, dims=1)
                logits = model(xk)
                if tta_hflip:
                    logits_h = model(xk.flip(-1))
                    logits = 0.5 * (logits + logits_h)
                logits_accum = logits if logits_accum is None else (logits_accum + logits)
            logits_mean = logits_accum / float(temporal_clips)
            if temperature and temperature != 1.0:
                logits_mean = logits_mean / float(temperature)
            return F.softmax(logits_mean, dim=1)
        else:
            all_probs = []
            for k in range(temporal_clips):
                xk = x
                if temporal_clips > 1 and xk.size(1) > 1:
                    shift = shifts[k] if deterministic_temporal else int(torch.randint(low=0, high=xk.size(1), size=(1,), device=xk.device).item())
                    xk = torch.roll(xk, shifts=shift, dims=1)
                logits = model(xk)
                if temperature and temperature != 1.0:
                    logits = logits / float(temperature)
                prob = F.softmax(logits, dim=1)
                if tta_hflip:
                    logits_h = model(xk.flip(-1))
                    if temperature and temperature != 1.0:
                        logits_h = logits_h / float(temperature)
                    prob_h = F.softmax(logits_h, dim=1)
                    prob = 0.5 * (prob + prob_h)
                all_probs.append(prob)
            if len(all_probs) == 1:
                return all_probs[0]
            return torch.stack(all_probs, dim=0).mean(0)


def _collect_logits(model, dataloader, device, *, tta_hflip: bool = False, temporal_clips: int = 1, deterministic_temporal: bool = False, logit_avg: bool = True, max_samples: int = None):
    """Collect per-sample logits (pre-softmax) for temperature fitting.

    Returns (logits_tensor[B, C], labels_tensor[B]). Uses TTA with logit averaging.
    """
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        seen = 0
        for batch in dataloader:
            x = batch['video'].to(device)
            y = batch['label'].to(device)

            # Optional sample cap
            if max_samples is not None:
                take = min(x.size(0), max_samples - seen)
                if take <= 0:
                    break
                if take < x.size(0):
                    x = x[:take]
                    y = y[:take]

            # Build logits via TTA with logit averaging
            temporal_clips = max(1, int(temporal_clips))
            T = x.size(1)
            shifts = []
            if temporal_clips > 1 and deterministic_temporal:
                for k in range(temporal_clips):
                    shifts.append(int(round(k * T / temporal_clips)) % max(1, T))

            # Always average logits here
            logits_accum = None
            for k in range(temporal_clips):
                xk = x
                if temporal_clips > 1 and xk.size(1) > 1:
                    shift = shifts[k] if deterministic_temporal else int(torch.randint(low=0, high=xk.size(1), size=(1,), device=xk.device).item())
                    xk = torch.roll(xk, shifts=shift, dims=1)
                logits = model(xk)
                if tta_hflip:
                    logits_h = model(xk.flip(-1))
                    logits = 0.5 * (logits + logits_h)
                logits_accum = logits if logits_accum is None else (logits_accum + logits)
            logits_mean = logits_accum / float(temporal_clips)

            logits_all.append(logits_mean.detach().cpu())
            labels_all.append(y.detach().cpu())
            seen += x.size(0)
    if not logits_all:
        return torch.empty(0, device='cpu'), torch.empty(0, dtype=torch.long, device='cpu')
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def _fit_temperature_grid(logits: torch.Tensor, labels: torch.Tensor):
    """Grid search temperature T>0 to minimize NLL on provided logits/labels.

    Returns best T as float.
    """
    if isinstance(logits, np.ndarray):
        logits_np = logits
    else:
        logits_np = logits.detach().cpu().numpy()
    if isinstance(labels, np.ndarray):
        labels_np = labels
    else:
        labels_np = labels.detach().cpu().numpy()

    def nll_for_T(T: float) -> float:
        z = logits_np / float(T)
        # log-softmax stable
        z = z - z.max(axis=1, keepdims=True)
        expz = np.exp(z)
        log_probs = z - np.log(expz.sum(axis=1, keepdims=True))
        idx = np.arange(z.shape[0])
        nll = -log_probs[idx, labels_np].mean()
        return float(nll)

    # coarse grid then refine
    Ts = np.linspace(0.5, 2.0, 31)  # 0.5..2.0 step ≈0.05
    best_T = 1.0
    best_nll = float('inf')
    for T in Ts:
        cur = nll_for_T(T)
        if cur < best_nll:
            best_nll = cur
            best_T = float(T)
    # refine around best
    lo = max(0.1, best_T - 0.3); hi = best_T + 0.3
    Ts2 = np.linspace(lo, hi, 41)
    for T in Ts2:
        cur = nll_for_T(T)
        if cur < best_nll:
            best_nll = cur
            best_T = float(T)
    return best_T


def evaluate_classification_balanced(model, dataloader, device, num_samples=400, threshold_scan=False, decision_threshold: float = None, tta_hflip: bool = False, temporal_clips: int = 1, deterministic_temporal: bool = False, scan_metric: str = 'f1_macro', *, logit_avg: bool = False, temperature: float = 1.0, thr_min: float = 0.10, thr_max: float = 0.90, thr_step: float = 0.01) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    import random
    model.eval()

    # Target totals
    target_total = min(num_samples, len(dataloader.dataset)) if hasattr(dataloader, 'dataset') else num_samples
    per_class_target = max(1, target_total // 2)
    class_0, class_1 = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting samples for balanced evaluation")
        for batch in pbar:
            if len(class_0) >= per_class_target and len(class_1) >= per_class_target:
                break
            x = batch['video'].to(device)
            y = batch['label'].to(device)

            prob = _probs_with_tta(model, x, tta_hflip=tta_hflip, temporal_clips=temporal_clips, deterministic_temporal=deterministic_temporal, logit_avg=logit_avg, temperature=temperature)
            pred = prob.argmax(dim=1)

            for i in range(x.size(0)):
                item = {
                    'pred': int(pred[i].cpu().numpy()),
                    'label': int(y[i].cpu().numpy()),
                    'prob': prob[i].cpu().numpy(),
                }
                if item['label'] == 0 and len(class_0) < per_class_target:
                    class_0.append(item)
                elif item['label'] == 1 and len(class_1) < per_class_target:
                    class_1.append(item)
            pbar.set_postfix({'Class_0': f"{len(class_0)}/{per_class_target}", 'Class_1': f"{len(class_1)}/{per_class_target}"})

    print(f"Class 0 (NonFight) collected: {len(class_0)}")
    print(f"Class 1 (Fight) collected: {len(class_1)}")
    per_class = min(per_class_target, len(class_0), len(class_1))
    if per_class < per_class_target:
        print(f"Warning: Not enough per-class samples. Using {per_class} per class")

    random.shuffle(class_0)
    random.shuffle(class_1)
    selected = class_0[:per_class] + class_1[:per_class]
    random.shuffle(selected)
    print(f"Selected {len(selected)} samples for evaluation ({per_class} per class)")

    correct = 0
    total = 0
    predictions, labels, probabilities = [], [], []
    with torch.no_grad():
        pbar = tqdm(selected, desc="Evaluating classification")
        for s in pbar:
            label = s['label']
            prob = s['prob']
            pred = int(float(prob[1]) >= float(decision_threshold)) if decision_threshold is not None else int(s['pred'])
            correct += int(pred == label)
            total += 1
            predictions.append(pred)
            labels.append(label)
            probabilities.append(prob)
            pbar.set_postfix({'Accuracy': f"{(correct/total if total>0 else 0):.4f}"})

    accuracy = correct / total if total > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    cm = confusion_matrix(labels, predictions)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(labels, [p[1] for p in probabilities])
        avg_precision = average_precision_score(labels, [p[1] for p in probabilities])
    except:
        roc_auc = 0.0
        avg_precision = 0.0

    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
    }
    if threshold_scan and probabilities:
        labels_np = np.array(labels)
        probs_np = np.array(probabilities)
        thr, _, stats = _threshold_scan(labels_np, probs_np, metric=scan_metric, thr_min=thr_min, thr_max=thr_max, thr_step=thr_step)
        preds_thr, p_macro, r_macro, f_macro, cm_thr = stats
        result.update({
            'best_threshold': thr,
            'f1_macro_thr': f_macro,
            'precision_macro_thr': p_macro,
            'recall_macro_thr': r_macro,
            'confusion_matrix_thr': cm_thr.tolist(),
        })
    return result


def evaluate_classification(model, dataloader, device, num_samples=50, threshold_scan=False, decision_threshold: float = None, tta_hflip: bool = False, temporal_clips: int = 1, deterministic_temporal: bool = False, scan_metric: str = 'f1_macro', *, logit_avg: bool = False, temperature: float = 1.0, thr_min: float = 0.10, thr_max: float = 0.90, thr_step: float = 0.01) -> Dict[str, float]:
    """Evaluate classification performance.
    num_samples means total samples to evaluate (not batches).
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
    
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    probabilities = []
    
    with torch.no_grad():
        target_total = min(num_samples, len(dataloader.dataset)) if hasattr(dataloader, 'dataset') else num_samples
        seen = 0
        pbar = tqdm(total=target_total, desc="Evaluating classification", unit='samples')
        for batch in dataloader:
            if seen >= target_total:
                break
            x = batch['video'].to(device)
            y = batch['label'].to(device)

            take = min(x.size(0), target_total - seen)
            if take < x.size(0):
                x = x[:take]
                y = y[:take]

            prob = _probs_with_tta(model, x, tta_hflip=tta_hflip, temporal_clips=temporal_clips, deterministic_temporal=deterministic_temporal, logit_avg=logit_avg, temperature=temperature)
            if decision_threshold is not None:
                pred = (prob[:, 1] >= float(decision_threshold)).long()
            else:
                pred = prob.argmax(dim=1)

            batch_correct = (pred == y).sum().item()
            batch_total = y.size(0)
            correct += batch_correct
            total += batch_total
            predictions.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())

            seen += batch_total
            pbar.update(batch_total)
            current_acc = correct / total if total > 0 else 0
            pbar.set_postfix({'Accuracy': f'{current_acc:.4f}', 'Correct': f'{correct}/{total}'})
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist(),
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities,
    }
    if threshold_scan and probabilities:
        labels_np = np.array(labels)
        probs_np = np.array(probabilities)
        thr, _, stats = _threshold_scan(labels_np, probs_np, metric=scan_metric, thr_min=thr_min, thr_max=thr_max, thr_step=thr_step)
        preds_thr, p_macro, r_macro, f_macro, cm_thr = stats
        result.update({
            'best_threshold': thr,
            'f1_macro_thr': f_macro,
            'precision_macro_thr': p_macro,
            'recall_macro_thr': r_macro,
            'confusion_matrix_thr': cm_thr.tolist(),
        })
    return result


def visualize_reconstruction(model, dataloader, device, save_dir, num_samples=5):
    """Visualize reconstruction results."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Generating visualizations", total=min(num_samples, len(dataloader)))
        for i, batch in enumerate(pbar):
            if i >= num_samples:
                break
                
            x = batch['video'].to(device)
            recon, target, mask, _ = model(x, return_mask=True)
            
            # Select first sample for visualization
            x_sample = x[0].cpu()  # [T, C, H, W]
            recon_sample = recon[0].cpu()  # [N_mask, C]
            target_sample = target[0].cpu()  # [N_mask, C]
            
            # Create visualization plot
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Display original video frames
            for t in range(min(4, x_sample.shape[0])):
                frame = x_sample[t].permute(1, 2, 0)  # [H, W, C]
                # Denormalize
                frame = frame * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                frame = torch.clamp(frame, 0, 1)
                
                axes[0, t].imshow(frame)
                axes[0, t].set_title(f'Original Frame {t}')
                axes[0, t].axis('off')
            
            # Display reconstruction statistics
            axes[1, 0].hist(recon_sample.flatten().numpy(), bins=50, alpha=0.7, label='Reconstruction')
            axes[1, 0].hist(target_sample.flatten().numpy(), bins=50, alpha=0.7, label='Target')
            axes[1, 0].set_title('Value Distribution')
            axes[1, 0].legend()
            
            # Display reconstruction error
            error = (recon_sample - target_sample).abs()
            axes[1, 1].hist(error.flatten().numpy(), bins=50)
            axes[1, 1].set_title('Reconstruction Error')
            
            # Display token norms
            recon_norms = recon_sample.norm(dim=-1).numpy()
            target_norms = target_sample.norm(dim=-1).numpy()
            axes[1, 2].scatter(target_norms, recon_norms, alpha=0.6)
            axes[1, 2].plot([0, max(target_norms.max(), recon_norms.max())], 
                           [0, max(target_norms.max(), recon_norms.max())], 'r--')
            axes[1, 2].set_xlabel('Target Norm')
            axes[1, 2].set_ylabel('Reconstruction Norm')
            axes[1, 2].set_title('Token Norm Comparison')
            
            # Display metrics
            mse = F.mse_loss(recon_sample, target_sample).item()
            mae = F.l1_loss(recon_sample, target_sample).item()
            psnr = calculate_psnr(recon_sample, target_sample)
            ssim = calculate_ssim(recon_sample, target_sample)
            
            axes[1, 3].text(0.1, 0.8, f'MSE: {mse:.4f}', fontsize=12)
            axes[1, 3].text(0.1, 0.6, f'MAE: {mae:.4f}', fontsize=12)
            axes[1, 3].text(0.1, 0.4, f'PSNR: {psnr:.2f} dB', fontsize=12)
            axes[1, 3].text(0.1, 0.2, f'SSIM: {ssim:.4f}', fontsize=12)
            axes[1, 3].set_title('Metrics')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'reconstruction_sample_{i}.png'), dpi=150, bbox_inches='tight')
            plt.close()


def extract_features(model, dataloader, device, save_path, num_samples=100):
    """Extract features for subsequent analysis."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting features", total=min(num_samples, len(dataloader)))
        for i, batch in enumerate(pbar):
            if i >= num_samples:
                break
                
            x = batch['video'].to(device)
            label = batch['label'].to(device)
            
            # Extract features using encoder's patch embedding
            # Convert input to encoder expected format
            x_in = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
            
            # Ensure 3-channel input
            c = x_in.shape[1]
            if c == 1:
                x_in = x_in.repeat(1, 3, 1, 1, 1)
            elif c == 4:
                x_in = x_in[:, :3]
            elif c != 3:
                raise ValueError(f'Unsupported channel count {c}; expected 1/3/4')
            
            # Use encoder's patch embedding to extract features
            encoded_features = model.backbone.encoder.patch_embed(x_in)  # [B, N, C]
            
            # Use global average pooling to get feature vector for each sample
            pooled_features = encoded_features.mean(dim=1)  # [B, C]
            
            features.append(pooled_features.cpu())
            labels.append(label.cpu())
    
    # Concatenate all features
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Debug info: check label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"Total samples: {len(labels)}")
    
    # Save features
    torch.save({
        'features': features,
        'labels': labels
    }, save_path)
    
    print(f"Extracted features saved to {save_path}")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = default_device()
    data_cfg = cfg['data']
    # Set up plain-text logging for evaluation (group by DP/epsilon)
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
    log_path = os.path.join(log_dir, f'evaluate_{ts}.txt')
    def _log(msg: str):
        try:
            with open(log_path, 'a', encoding='utf-8') as _f:
                _f.write(msg.rstrip('\n') + '\n')
        except Exception:
            pass
    try:
        _log('=== Task: evaluate ===')
        _log(f'timestamp: {ts}')
        _log('config:')
        _log(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        pass
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data (built after model-type resolution below)
    
    # Determine model type and mode based on arguments
    if args.model_type == 'pretrained':
        # Evaluate pretrained model (reconstruction)
        model_cfg = cfg['model']
        model = build_videomae_model(
            model_name=model_cfg.get('model_name'),
            pretrained=model_cfg.get('pretrained', True),
            mode='pretrain',  # Force pretrain mode for reconstruction
            num_classes=2,
            mask_ratio=model_cfg.get('mask_ratio', 0.9),
            peft_config=model_cfg.get('peft', None),
            pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None),
            num_frames=data_cfg.get('num_frames', 16)
        )
        eval_mode = 'pretrain'
    elif args.model_type == 'small_head':
        # Evaluate small head fine-tuned model (classification)
        model_cfg = cfg['model']
        model = build_videomae_model(
            model_name=model_cfg.get('model_name'),
            pretrained=model_cfg.get('pretrained', True),
            mode='head',  # Force head mode for classification
            num_classes=2,
            mask_ratio=model_cfg.get('mask_ratio', 0.9),
            peft_config=model_cfg.get('peft', None),
            head_config=model_cfg.get('head', {}),  # Pass head configuration
            pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None),
            num_frames=data_cfg.get('num_frames', 16)
        )
        eval_mode = 'head'
    else:  # auto mode - use config file settings
        # Use config file settings (backward compatibility)
        model_cfg = cfg['model']
        model = build_videomae_model(
            model_name=model_cfg.get('model_name'),
            pretrained=model_cfg.get('pretrained', True),
            mode=model_cfg.get('mode', 'pretrain'),
            num_classes=2,
            mask_ratio=model_cfg.get('mask_ratio', 0.9),
            peft_config=model_cfg.get('peft', None),
            pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None),
            num_frames=data_cfg.get('num_frames', 16)
        )
        eval_mode = model_cfg.get('mode', 'pretrain')
    
    # Build dataset (supports optional JSON val list)
    transform = build_video_transform(size=data_cfg.get('size', 224), normalize=True)
    root = data_cfg['root']
    val_list = None
    parts = data_cfg.get('partitions')
    # Normalize relative partitions path to repository root (parent of 'federated_videomae')
    if parts and not os.path.isabs(parts):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../federated_videomae
        repo_root = os.path.dirname(project_root)  # parent dir of the package
        cand1 = os.path.join(repo_root, parts)
        cand2 = os.path.join(project_root, parts)
        if os.path.exists(cand1):
            parts = cand1
        elif os.path.exists(cand2):
            parts = cand2
        else:
            # Try relative to current working directory
            cand3 = os.path.join(os.getcwd(), parts)
            if os.path.exists(cand3):
                parts = cand3
        # else keep original (will print not found note below)
    if parts and os.path.exists(parts):
        try:
            with open(parts, 'r', encoding='utf-8') as f:
                js = json.load(f)
            entries = js.get('val', js)  # accept either {"val": [...]} or plain list
            # entries: {path,label}
            val_list = [(e['path'], int(e['label'])) for e in entries]
            print(f"Loaded validation list from {parts}: {len(val_list)} samples")
        except Exception as e:
            print(f"Warning: failed to read partitions file {parts}: {e}")
            val_list = None
    else:
        if parts:
            print(f"Note: partitions file not found: {parts}. Evaluating full dataset under {root}")

    ds_type = detect_dataset_type(root)
    print(f"Detected dataset type: {ds_type}")
    if val_list:
        print(f"Using validation list with {len(val_list)} samples")
        # Check label distribution in val_list
        label_counts = {}
        for _, label in val_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution in val_list: {label_counts}")
    
    if ds_type == 'rwf2000':
        dataset = RWF2000Dataset(
            root=root,
            transform=transform,
            num_frames=data_cfg.get('num_frames', 16),
            frame_stride=data_cfg.get('frame_stride', 4),
            index_list=val_list,
        )
    else:
        dataset = RLVSDataset(
            root=root,
            transform=transform,
            num_frames=data_cfg.get('num_frames', 16),
            frame_stride=data_cfg.get('frame_stride', 4),
            index_list=val_list,
        )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=video_collate,
        pin_memory=True,
    )

    # Determine checkpoint path solely from YAML configuration
    ckpt_path = cfg.get('model', {}).get('pretrained_checkpoint', None)
    if not ckpt_path:
        raise FileNotFoundError('Checkpoint path not provided in config (model.pretrained_checkpoint).')
    # Check if model file exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found!")
    
    # Load pretrained weights
    print(f"Loading checkpoint from {ckpt_path}")
    # Log checkpoint SHA256 for provenance
    try:
        import hashlib
        def _sha256(p: str, max_bytes: int = 0) -> str:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                if max_bytes and max_bytes > 0:
                    h.update(f.read(max_bytes))
                else:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(chunk)
            return h.hexdigest()
        ck_hash = _sha256(ckpt_path)
        _log(f"[Init] Using pretrained checkpoint: {ckpt_path} | sha256={ck_hash[:16]}")
    except Exception:
        pass
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Optionally apply best decision threshold from training (opt-in)
    try:
        if getattr(args, 'use_checkpoint_threshold', False) and isinstance(checkpoint, dict) and args.threshold is None:
            bt = None
            tag = None
            if 'best_threshold' in checkpoint:
                bt = checkpoint.get('best_threshold')
                tag = checkpoint.get('best_threshold_metric', None)
            elif 'val_best_threshold' in checkpoint:
                bt = checkpoint.get('val_best_threshold')
                tag = checkpoint.get('val_best_threshold_metric', None)
            if isinstance(bt, (int, float)):
                args.threshold = float(bt)
                print(f"[Eval] Using checkpoint best_threshold={args.threshold:.2f} (metric={tag or args.scan_metric})")
    except Exception:
        pass
    # Prefer EMA weights if present unless --no_ema
    if (not getattr(args, 'no_ema', False)) and ('model_ema' in checkpoint):
        state_dict = checkpoint['model_ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # If checkpoint saved as BackboneHead (bb.+head.), rebuild same graph to load directly
    has_bb = any(k.startswith('bb.') for k in state_dict.keys())
    has_head = any(k.startswith('head.') for k in state_dict.keys())
    if args.model_type == 'small_head' and (has_bb or has_head):
        # Try to infer LoRA rank from checkpoint to build a matching model
        def _infer_lora_r(sd: dict) -> int:
            try:
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor) and k.endswith('.A') and v.dim() == 2:
                        return int(v.shape[0])
            except Exception:
                pass
            return None
        from models import MLPHead
        peft_cfg = model_cfg.get('peft', None)
        # If LoRA is used but rank unspecified or mismatched, override or disable to match checkpoint
        inferred_r = _infer_lora_r(state_dict)
        if peft_cfg and peft_cfg.get('use_lora', False):
            peft_cfg = dict(peft_cfg)  # shallow copy
            if inferred_r:
                peft_cfg['lora_r'] = inferred_r
                if 'lora_alpha' not in peft_cfg or not peft_cfg['lora_alpha']:
                    peft_cfg['lora_alpha'] = inferred_r * 2
            else:
                # Checkpoint has no LoRA A/B; disable LoRA injection for a clean load
                peft_cfg['use_lora'] = False
                print('[Eval] LoRA disabled for evaluation (no A/B matrices found in checkpoint)')
        bb = build_videomae_model(
            model_name=model_cfg.get('model_name'),
            pretrained=model_cfg.get('pretrained', True),
            mode='feature',
            num_classes=2,
            mask_ratio=model_cfg.get('mask_ratio', 0.9),
            peft_config=peft_cfg,
            pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None),
            num_frames=data_cfg.get('num_frames', 16)
        )
        in_dim = bb._feat_dim()
        head_cfg = model_cfg.get('head', {})
        head = MLPHead(
            in_dim,
            head_cfg.get('hidden_dim', 512),
            head_cfg.get('num_classes', 2),
            num_layers=head_cfg.get('num_layers', 1),
            dropout=head_cfg.get('dropout', 0.0),
        )
        import torch.nn as nn
        class BackboneHead(nn.Module):
            def __init__(self, bb, head):
                super().__init__()
                self.bb = bb
                self.head = head
            def forward(self, x):
                feats = self.bb(x)
                return self.head(feats)
        model = BackboneHead(bb, head)
        # strip only module. prefix
        sd2 = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            sd2[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(sd2, strict=False)
        if missing_keys:
            print(f"Note: Missing keys (first 10): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Note: Unexpected keys (first 10): {unexpected_keys[:10]}")
        print("Loaded BackboneHead checkpoint (strict=False)")
        model.to(device)
        print(f"Loaded model (BackboneHead) from {ckpt_path}")
        # Optional: also load 'trainable' PEFT weights if provided (and shape-matching)
        try:
            tr = checkpoint.get('trainable', None) if isinstance(checkpoint, dict) else None
            if isinstance(tr, dict) and len(tr) > 0:
                ms = model.state_dict()
                tl = {k: v for k, v in tr.items() if k in ms and getattr(ms[k], 'shape', None) == getattr(v, 'shape', None)}
                if tl:
                    model.load_state_dict(tl, strict=False)
                    print(f"Loaded additional PEFT trainable: matched={len(tl)}/{len(tr)}")
        except Exception:
            pass
    
    # Fix/Filter keys: remove prefixes; skip recon_prompt/decoder/LoRA A,B; map *.base.* to plain; map head
    def fix_state_dict_keys(state_dict):
        fixed = {}
        # Detect whether current model contains LoRA modules; if so, we must keep A/B and base.* keys
        try:
            from models.peft_lora import LoRALinear  # type: ignore
            lora_present = any(isinstance(m, LoRALinear) for m in model.modules())
        except Exception:
            lora_present = False
        for key, value in state_dict.items():
            # strip numeric prefixes like '0.'
            if '.' in key and key.split('.')[0].isdigit():
                key = '.'.join(key.split('.')[1:])
            # strip ddp/module prefix if present
            if key.startswith('module.'):
                key = key[len('module.'):]
            # map 'bb.' prefix (from Sequential(backbone=head) save) to nothing so it matches wrapper keys
            if key.startswith('bb.'):
                key = key[len('bb.'):]
            # skip recon_prompt / decoder; keep LoRA A/B if LoRA is present in the eval model
            if 'recon_prompt' in key or '.decoder.' in key:
                continue
            if (key.endswith('.A') or key.endswith('.B')) and not lora_present:
                # Only drop A/B when evaluation model has no LoRA
                continue
            # Only map base.* to plain when there is no LoRA in the eval model
            if (not lora_present) and key.endswith('.base.weight'):
                key = key[:-11] + '.weight'
            elif (not lora_present) and key.endswith('.base.bias'):
                key = key[:-9] + '.bias'
            fixed[key] = value

        # head mode: map MLPHead layers to classifier
        if eval_mode == 'head':
            mapped = {}
            for k, v in fixed.items():
                # support both bare 'net.*' and prefixed 'head.net.*'
                if k in ('net.2.weight', 'head.net.2.weight'):
                    mapped['classifier.weight'] = v
                elif k in ('net.2.bias', 'head.net.2.bias'):
                    mapped['classifier.bias'] = v
                elif k.startswith('net.0.') or k.startswith('head.net.0.'):
                    # Map first layer of MLP
                    new_k = k.replace('head.', '').replace('net.0.', 'classifier.0.')
                    mapped[new_k] = v
                elif k.startswith('net.1.') or k.startswith('head.net.1.'):
                    # Map second layer of MLP
                    new_k = k.replace('head.', '').replace('net.1.', 'classifier.2.')
                    mapped[new_k] = v
                elif k.startswith('head.'):
                    # drop 'head.' prefix for any remaining keys
                    mapped[k[len('head.'):]] = v
                else:
                    mapped[k] = v
            return mapped
        return fixed
    # If we already rebuilt BackboneHead above, skip wrapper-head mapping path
    if args.model_type == 'small_head' and (has_bb or has_head):
        pass
    else:
    # Fix the state dict keys
        state_dict = fix_state_dict_keys(state_dict)
        
        # Try to load weights, ignore mismatched keys
        # Load permissively after filtering
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Note: Missing keys after filter: {len(missing_keys)} (showing first 10): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Note: Unexpected keys after filter: {len(unexpected_keys)} (showing first 10): {unexpected_keys[:10]}")
        print("Loaded pretrained weights with strict=False (filtered)")
        
        model.to(device)
        print(f"Loaded model from {ckpt_path}")
        # Optional: also load 'trainable' PEFT weights if provided (and shape-matching)
        try:
            tr = checkpoint.get('trainable', None) if isinstance(checkpoint, dict) else None
            if isinstance(tr, dict) and len(tr) > 0:
                ms = model.state_dict()
                tl = {k: v for k, v in tr.items() if k in ms and getattr(ms[k], 'shape', None) == getattr(v, 'shape', None)}
                if tl:
                    model.load_state_dict(tl, strict=False)
                    print(f"Loaded additional PEFT trainable: matched={len(tl)}/{len(tr)}")
        except Exception:
            pass
    
    # Optional: fit temperature for calibration (classification modes only)
    if args.fit_temperature and eval_mode != 'pretrain':
        print('Fitting temperature via grid search...')
        # We will reuse the same dataloader; collect log-probs as proxy and grid search T
        log_probs, labels = _collect_logits(model, dataloader, device, tta_hflip=args.tta_hflip, temporal_clips=args.temporal_clips, deterministic_temporal=args.deterministic_temporal, logit_avg=True)
        T = _fit_temperature_grid(log_probs, labels)
        args.temperature = float(T)
        print(f'[Calibration] Chosen temperature T={args.temperature:.3f}')

    # Evaluate based on mode
    if eval_mode == 'pretrain':
        print("Evaluating reconstruction quality...")
        metrics = evaluate_reconstruction_quality(model, dataloader, device, num_samples=args.num_samples)
        print("\n=== Reconstruction Quality Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        _log('metrics:')
        for key, value in metrics.items():
            _log(f"{key}: {float(value):.4f}")
    else:
        print("Evaluating classification performance...")
        if args.balanced_sampling:
            metrics = evaluate_classification_balanced(
                model, dataloader, device,
                num_samples=args.num_samples,
                threshold_scan=args.threshold_scan,
                decision_threshold=args.threshold,
                tta_hflip=args.tta_hflip,
                temporal_clips=args.temporal_clips,
                deterministic_temporal=args.deterministic_temporal,
                scan_metric=args.scan_metric,
                logit_avg=args.logit_avg,
                temperature=args.temperature,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                thr_step=args.thr_step,
            )
            if args.apply_best_threshold and 'best_threshold' in metrics:
                thr = metrics['best_threshold']
                print(f"\n[ApplyBestThreshold] Re-evaluating with threshold={thr:.2f}")
                metrics = evaluate_classification_balanced(
                    model, dataloader, device,
                    num_samples=args.num_samples,
                    threshold_scan=False,
                    decision_threshold=thr,
                    tta_hflip=args.tta_hflip,
                    temporal_clips=args.temporal_clips,
                    deterministic_temporal=args.deterministic_temporal,
                    scan_metric=args.scan_metric,
                    logit_avg=args.logit_avg,
                    temperature=args.temperature,
                    thr_min=args.thr_min,
                    thr_max=args.thr_max,
                    thr_step=args.thr_step,
                )
        else:
            metrics = evaluate_classification(
                model, dataloader, device,
                num_samples=args.num_samples,
                threshold_scan=args.threshold_scan,
                decision_threshold=args.threshold,
                tta_hflip=args.tta_hflip,
                temporal_clips=args.temporal_clips,
                deterministic_temporal=args.deterministic_temporal,
                scan_metric=args.scan_metric,
                logit_avg=args.logit_avg,
                temperature=args.temperature,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                thr_step=args.thr_step,
            )
            if args.apply_best_threshold and 'best_threshold' in metrics:
                thr = metrics['best_threshold']
                print(f"\n[ApplyBestThreshold] Re-evaluating with threshold={thr:.2f}")
                metrics = evaluate_classification(
                    model, dataloader, device,
                    num_samples=args.num_samples,
                    threshold_scan=False,
                    decision_threshold=thr,
                    tta_hflip=args.tta_hflip,
                    temporal_clips=args.temporal_clips,
                    deterministic_temporal=args.deterministic_temporal,
                    scan_metric=args.scan_metric,
                    logit_avg=args.logit_avg,
                    temperature=args.temperature,
                    thr_min=args.thr_min,
                    thr_max=args.thr_max,
                    thr_step=args.thr_step,
                )
        print("\n=== Classification Performance Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Correct: {metrics['correct']}/{metrics['total']}")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        # Log summary metrics
        try:
            _log('metrics:')
            for k in ['accuracy','precision_weighted','recall_weighted','f1_weighted','precision_macro','recall_macro','f1_macro']:
                if k in metrics:
                    _log(f"{k}: {float(metrics[k]):.4f}")
            _log(f"correct: {int(metrics.get('correct',0))}/{int(metrics.get('total',0))}")
            if 'best_threshold' in metrics and 'best_threshold_metric' in metrics and 'best_threshold_score' in metrics:
                _log(f"best_threshold: {metrics['best_threshold']:.2f} ({metrics['best_threshold_metric']}) score={metrics['best_threshold_score']:.4f}")
        except Exception:
            pass
        
        print("\n=== Per-Class Metrics ===")
        class_names = ['NonFight', 'Fight']
        precision_per_class = metrics.get('precision_per_class', [])
        recall_per_class = metrics.get('recall_per_class', [])
        f1_per_class = metrics.get('f1_per_class', [])
        support_per_class = metrics.get('support_per_class', [])
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            if i < len(precision_per_class):
                print(f"  Precision: {precision_per_class[i]:.4f}")
            else:
                print(f"  Precision: N/A")
            if i < len(recall_per_class):
                print(f"  Recall: {recall_per_class[i]:.4f}")
            else:
                print(f"  Recall: N/A")
            if i < len(f1_per_class):
                print(f"  F1-Score: {f1_per_class[i]:.4f}")
            else:
                print(f"  F1-Score: N/A")
            if i < len(support_per_class):
                print(f"  Support: {support_per_class[i]}")
            else:
                print(f"  Support: N/A")
        
        print("\n=== Confusion Matrix ===")
        cm = np.array(metrics['confusion_matrix'])
        if cm.ndim == 1:
            print("No samples available for confusion matrix")
        else:
            print("         Predicted")
            print("         NonFight  Fight")
            if cm.shape[0] >= 2 and cm.shape[1] >= 2:
                print(f"Actual NonFight   {cm[0,0]:4d}   {cm[0,1]:4d}")
                print(f"       Fight       {cm[1,0]:4d}   {cm[1,1]:4d}")
            else:
                print("Confusion matrix shape:", cm.shape)
        
        # If threshold scan results available, print them
        if 'best_threshold' in metrics:
            print("\n=== Threshold Scan (macro-F1) ===")
            print(f"Best threshold: {metrics['best_threshold']:.2f}")
            print(f"Precision (Macro, thr): {metrics['precision_macro_thr']:.4f}")
            print(f"Recall (Macro, thr): {metrics['recall_macro_thr']:.4f}")
            print(f"F1-Score (Macro, thr): {metrics['f1_macro_thr']:.4f}")
            cm_t = np.array(metrics['confusion_matrix_thr'])
            print("Confusion Matrix @ best thr:")
            print(f"         Predicted")
            print(f"         NonFight  Fight")
            print(f"Actual NonFight   {cm_t[0,0]:4d}   {cm_t[0,1]:4d}")
            print(f"       Fight       {cm_t[1,0]:4d}   {cm_t[1,1]:4d}")
    
    # Visualize reconstruction results
    if args.visualize:
        print("Generating reconstruction visualizations...")
        visualize_reconstruction(
            model, dataloader, device, 
            save_dir=args.output_dir, 
            num_samples=args.num_samples
        )
        print(f"Visualizations saved to {args.output_dir}")
    
    # Extract features
    if args.extract_features:
        print("Extracting features...")
        feature_path = os.path.join(args.output_dir, 'features.pth')
        extract_features(model, dataloader, device, feature_path, num_samples=args.num_samples)
    
    # Save metrics
    # Convert numpy types to Python native types for JSON serialization
    metrics_serializable = convert_numpy_types(metrics)
    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VideoMAE models')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model_type', type=str, choices=['pretrained', 'small_head', 'auto'], 
                       default='auto', help='Model type: pretrained (reconstruction), small_head (classification), or auto (from config)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=400, help='Number of samples to evaluate')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--extract_features', action='store_true', help='Extract features for analysis')
    parser.add_argument('--threshold_scan', action='store_true', help='Scan decision threshold for best metric (classification only)')
    parser.add_argument('--balanced_sampling', action='store_true', default=False, help='Use balanced sampling for classification evaluation')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold for class=1 (overrides argmax)')
    parser.add_argument('--apply_best_threshold', action='store_true', help='Apply best threshold from scan to recompute metrics')
    parser.add_argument('--use_checkpoint_threshold', action='store_true', help='Use best_threshold saved in checkpoint (if present) unless --threshold is given')
    parser.add_argument('--no_ema', action='store_true', help='Ignore EMA weights even if present in checkpoint')
    parser.add_argument('--scan_metric', type=str, default='f1_macro', choices=['f1_macro','precision_macro','recall_macro','accuracy'], help='Metric used when scanning threshold')
    parser.add_argument('--thr_min', type=float, default=0.10, help='Threshold scan min (inclusive)')
    parser.add_argument('--thr_max', type=float, default=0.90, help='Threshold scan max (inclusive)')
    parser.add_argument('--thr_step', type=float, default=0.01, help='Threshold scan step size')
    parser.add_argument('--deterministic_temporal', action='store_true', help='Use fixed, evenly spaced temporal shifts for multi-clip TTA')
    # TTA / multi-clip options
    parser.add_argument('--tta_hflip', action='store_true', help='Enable horizontal flip TTA (averages original+flipped)')
    parser.add_argument('--temporal_clips', type=int, default=1, help='Evaluate N temporal jitter clips and average probs (approximate)')
    # Aggregation & calibration options
    parser.add_argument('--logit_avg', action='store_true', help='Average logits instead of probabilities across TTA clips')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling for logits before softmax (T=1.0 means no scaling)')
    parser.add_argument('--fit_temperature', action='store_true', help='Fit a temperature on-the-fly (grid search) to minimize NLL, then apply')
    args = parser.parse_args()

    main(args)


