import os
import random
from collections import defaultdict
from typing import Dict, Any
import math
import numpy as np
import torch
import wandb
import sys
import gc
from datetime import datetime
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.aggregator import fedavg
from fl.serialization import load_trainable_state_dict, get_trainable_state_dict
from fl.privacy import SecureAggregator, fedavg_with_privacy
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from torch.utils.data import DataLoader, Subset
from data.collate import video_collate

import warnings
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

def select_clients(all_clients, m: int):
    return random.sample(all_clients, m) if m < len(all_clients) else list(all_clients)


def run_federated(
    model: torch.nn.Module,
    clients_data: Dict[str, Any],
    make_optimizer,
    loss_fn,
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    clip_grad: float = 0.0,
    log_every: int = 50,
    out_dir: str = 'runs',
    wandb_project: str = None,
    wandb_entity: str = None,
    experiment_name: str = None,
    use_amp: bool = False,
    aggregate_top_k: int = 0,
    explore_prob: float = 0.0,
    full_agg_period: int = 0,
    score_metric: str = 'acc',  # 'acc' or 'loss'
    score_ema_alpha: float = 0.6,
    # dataloader/memory controls
    reuse_loaders: bool = False,
    dataloader_persistent_workers: bool = False,
    dataloader_prefetch_factor: int = 2,
    dataloader_pin_memory: bool = True,
    # validation controls
    val_dataset=None,
    val_batch_size: int = None,
    val_num_workers: int = 0,
    select_best_metric: str = 'acc',
    # Differential Privacy parameters
    use_dp: bool = False,
    target_epsilon: float = 10.0,
    target_delta: float = 1e-5,
    noise_multiplier: float = None,
    max_grad_norm: float = 1.0,
    # Secure Aggregation parameters
    use_secure_agg: bool = False,
    secure_agg_threshold: int = None,
    # Privacy-preserving aggregation
    server_noise_scale: float = 0.0,
    # Memory controls
    clear_cache_steps: int = 0,
    dp_max_physical_batch_size: int = None,
):
    os.makedirs(out_dir, exist_ok=True)
    all_clients = list(clients_data.keys())
    # Set up plain-text logging (server-only)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Group logs by DP/epsilon for easier comparison
    def _infer_dp_subdir(enabled: bool, eps: float) -> str:
        if not enabled:
            return os.path.join('no-dp')
        try:
            e = int(round(float(eps)))
        except Exception:
            e = 0
        return os.path.join('dp', f'epsilon_{e}')
    base_logs = os.path.join(project_root, 'logs')
    sub = _infer_dp_subdir(use_dp, target_epsilon)
    log_dir = os.path.join(base_logs, sub)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'pretrain_{ts}.txt')
    def _log(msg: str):
        try:
            with open(log_path, 'a', encoding='utf-8') as _f:
                _f.write(msg.rstrip('\n') + '\n')
        except Exception:
            pass
    # Log task header + config snapshot
    _log('=== Task: pretrain (server) ===')
    _log(f'timestamp: {ts}')
    cfg_snapshot = {
        'rounds': rounds,
        'clients_per_round': clients_per_round,
        'local_epochs': local_epochs,
        'optimizer': {'lr': lr, 'weight_decay': weight_decay},
        'batch_size': batch_size,
        'num_workers': num_workers,
        'use_amp': use_amp,
        'dp': {
            'enabled': use_dp,
            'target_epsilon': target_epsilon,
            'target_delta': target_delta,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
        },
        'secure_aggregation': {
            'enabled': use_secure_agg,
            'threshold': secure_agg_threshold or clients_per_round,
        },
        'server_noise_scale': server_noise_scale,
    }
    try:
        _log('config:')
        _log(yaml.safe_dump(cfg_snapshot, sort_keys=False))
    except Exception:
        pass

    # Hash helper for encoder/backbone to track changes
    def _hash_tensor_map(sd: dict) -> str:
        import hashlib as _hl
        h = _hl.sha256()
        for k in sorted(sd.keys()):
            v = sd[k]
            if isinstance(v, torch.Tensor):
                tb = v.detach().cpu().contiguous()
                try:
                    h.update(k.encode('utf-8'))
                    h.update(tb.numpy().tobytes(order='C'))
                except Exception:
                    h.update(k.encode('utf-8'))
                    h.update(tb.flatten().to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()

    # Log initial encoder hash (or full state if no encoder attr)
    try:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
            init_hash = _hash_tensor_map(model.backbone.encoder.state_dict())
            _log(f'[Init] encoder_full_hash={init_hash[:16]}')
        else:
            init_hash = _hash_tensor_map(model.state_dict())
            _log(f'[Init] model_full_hash={init_hash[:16]}')
    except Exception:
        pass
    
    # Differential privacy: let clients compute per-round, per-client noise if not provided explicitly.
    # This avoids mismatches due to heterogeneous client sizes and per-round accountants.
    
    # Setup secure aggregation if enabled
    secure_aggregator = None
    if use_secure_agg:
        try:
            secure_aggregator = SecureAggregator(
                num_clients=len(all_clients),
                threshold=(secure_agg_threshold or clients_per_round),
            )
            print(f"[Server] Secure aggregation enabled. Expected participants per round: {clients_per_round}")
        except Exception as e:
            print(f"[Server] Failed to setup secure aggregation: {e}")
            print(f"[Server] Falling back to standard aggregation")
            use_secure_agg = False
    
    # Initialize wandb if project is specified
    if wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=experiment_name,
            config={
                'rounds': rounds,
                'clients_per_round': clients_per_round,
                'local_epochs': local_epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'num_clients': len(all_clients),
                'device': str(device),
            }
        )
        print(f"[Server] Wandb initialized: project={wandb_project}, entity={wandb_entity}, name={experiment_name}")
    
    # Build persistent DataLoaders per client once
    client_loaders: Dict[str, DataLoader] = {}
    for cid in all_clients:
        dataset, indices = clients_data[cid]['dataset'], clients_data[cid]['indices']
        subset = Subset(dataset, indices)
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=video_collate,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        if num_workers > 0:
            dl_kwargs['prefetch_factor'] = 2
        client_loaders[cid] = DataLoader(subset, **dl_kwargs)

    # Pre-compute DP configuration per client (noise multiplier and sampling rate)
    client_dp_nm: Dict[str, float] = {}
    client_dp_sr: Dict[str, float] = {}
    client_dp_steps_per_round: Dict[str, int] = {}
    client_dp_steps_seen: Dict[str, int] = {}
    if use_dp:
        for cid in all_clients:
            indices = clients_data[cid]['indices']
            n = max(1, len(indices))
            # Poisson-like sampling rate approximation for DP-SGD
            sr = min(1.0, float(batch_size) / float(n))
            # Steps per round for this client (optimizer updates)
            steps_per_epoch = int(math.ceil(n / max(1, batch_size)))
            steps_rnd = max(1, steps_per_epoch * max(1, local_epochs))
            total_steps_worst = steps_rnd * max(1, rounds)  # worst-case: selected every round
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Optimal order is the largest alpha.*",
                        category=UserWarning,
                        module=r"opacus\.accountants\.analysis\.rdp",
                    )
                    nm = get_noise_multiplier(
                        target_epsilon=target_epsilon,
                        target_delta=target_delta,
                        sample_rate=sr,
                        steps=total_steps_worst,
                        accountant="rdp",
                    )
                nm = float(max(nm, 0.1))
            except Exception:
                # Fallback to Gaussian mech bound
                nm = float(max(math.sqrt(2.0 * math.log(1.25 / max(target_delta, 1e-12))) / max(target_epsilon, 1e-6), 0.1))
            client_dp_nm[cid] = nm
            client_dp_sr[cid] = sr
            client_dp_steps_per_round[cid] = steps_rnd
            client_dp_steps_seen[cid] = 0
        # Summarize DP per-client settings
        try:
            nms = list(client_dp_nm.values()); srs = list(client_dp_sr.values()); steps = list(client_dp_steps_per_round.values())
            def _stat(x):
                import numpy as _np
                a = _np.array(x, dtype=float)
                return float(a.min()), float(a.mean()), float(a.max())
            nm_min, nm_mean, nm_max = _stat(nms)
            sr_min, sr_mean, sr_max = _stat(srs)
            st_min, st_mean, st_max = _stat(steps)
            _log(f'[DP] noise_multiplier min/mean/max: {nm_min:.4f}/{nm_mean:.4f}/{nm_max:.4f}')
            _log(f'[DP] sample_rate min/mean/max: {sr_min:.6f}/{sr_mean:.6f}/{sr_max:.6f}')
            _log(f'[DP] steps_per_round min/mean/max: {st_min:.0f}/{st_mean:.1f}/{st_max:.0f}')
        except Exception:
            pass

    def _epsilon_from_rdp(nm: float, sr: float, steps: int, delta: float) -> float:
        if steps <= 0:
            return 0.0
        # Standard alpha grid from Opacus examples
        orders = [1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 14, 16, 20, 32, 64, 128, 256, 384, 512, 768, 1024]
        rdp = compute_rdp(q=sr, noise_multiplier=nm, steps=int(steps), orders=orders)
        # Opacus versions may define get_privacy_spent with keyword-only args or different naming
        def _extract_eps(res):
            # Opacus may return (eps, best_order) or (eps, best_order, opt_alpha)
            if isinstance(res, (list, tuple)):
                if len(res) >= 1:
                    return float(res[0])
            # As a last resort, treat as scalar
            try:
                return float(res)
            except Exception:
                return 0.0
        try:
            res = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
            eps = _extract_eps(res)
        except TypeError:
            # Fallback for APIs expecting 'target_delta'
            res = get_privacy_spent(orders=orders, rdp=rdp, target_delta=delta)
            eps = _extract_eps(res)
        return float(eps)

    # Maintain EMA scores per client across rounds
    score_ema = {cid: None for cid in all_clients}

    # Optional: build persistent loaders once if requested
    client_loaders = {}
    if reuse_loaders:
        for cid in all_clients:
            dataset, indices = clients_data[cid]['dataset'], clients_data[cid]['indices']
            from fl.client import _make_loader as _mk
            client_loaders[cid] = _mk(
                dataset, indices, batch_size, num_workers,
                persistent_workers=dataloader_persistent_workers,
                prefetch_factor=dataloader_prefetch_factor,
                pin_memory=dataloader_pin_memory,
            )

    # Prepare validation dataloader (optional)
    val_loader = None
    if val_dataset is not None:
        from data.collate import video_collate as _val_collate
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            collate_fn=_val_collate,
            pin_memory=True,
            persistent_workers=(val_num_workers > 0),
        )

    best_score = None
    best_path = None

    secure_agg_template = get_trainable_state_dict(model) if use_secure_agg and secure_aggregator else None

    for rnd in range(1, rounds + 1):
        selected = select_clients(all_clients, clients_per_round)
        print(f'\n[Server] Round {rnd}/{rounds} | selected={selected}')

        if use_secure_agg and secure_aggregator:
            try:
                # Recompute parameter template just-in-time to ensure availability
                secure_agg_template = get_trainable_state_dict(model)
                secure_aggregator.start_round(selected, secure_agg_template)
            except Exception as e:
                print(f"[Server] Secure aggregation setup failed for round {rnd}: {e}")
                print("[Server] Falling back to standard aggregation for this round")
                secure_aggregator = None
                use_secure_agg = False
                secure_agg_template = None

        # Save global trainable snapshot at round start
        global_start = get_trainable_state_dict(model)

        client_states = []
        client_ids = []
        round_metrics = defaultdict(list)
        for cid in selected:
            dataset, indices = clients_data[cid]['dataset'], clients_data[cid]['indices']
            # fresh optimizer per client on the same model instance using PEFT params only
            opt = make_optimizer(model, lr=lr, weight_decay=weight_decay)
            # Ensure every client starts from identical global state
            try:
                load_trainable_state_dict(model, global_start)
            except Exception:
                pass
            # Resolve per-client noise multiplier (fixed across rounds)
            nm_this = None
            if use_dp:
                nm_this = client_dp_nm.get(cid, noise_multiplier)
            state = client_update_fn(
                model=model,
                optimizer=opt,
                loss_fn=loss_fn,
                dataset=dataset,
                indices=indices,
                batch_size=batch_size,
                num_workers=num_workers,
                local_epochs=local_epochs,
                device=device,
                client_id=cid,
                clip_grad=clip_grad,
                log_every=log_every,
                loader=client_loaders[cid] if reuse_loaders else None,
                use_amp=use_amp,
                persistent_workers=dataloader_persistent_workers and reuse_loaders,
                prefetch_factor=dataloader_prefetch_factor,
                pin_memory=dataloader_pin_memory,
                # Differential Privacy parameters
                use_dp=use_dp,
                noise_multiplier=nm_this if nm_this is not None else noise_multiplier,
                max_grad_norm=max_grad_norm,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=batch_size / len(indices),
                # Memory controls
                clear_cache_steps=clear_cache_steps,
                max_physical_batch_size=dp_max_physical_batch_size,
                # Apply secure aggregation masking on client side before returning
                mask_fn=(
                    (lambda st, _cid=cid: secure_aggregator.mask_client_update(_cid, st))
                    if (use_secure_agg and secure_aggregator is not None)
                    else None
                ),
            )
            client_states.append(state)
            client_ids.append(cid)

            # Accumulate final metrics from each client
            if 'metrics' in state:
                for k, v in state['metrics'].items():
                    round_metrics[k].append(v)

            # Server-side no longer applies additional masking to avoid accessing unmasked data

        # Calculate and print global average metrics (if available)
        if round_metrics:
            global_metrics = {k: np.mean(v) for k, v in round_metrics.items()}
            # Compact logging: keep only core metrics; drop redundant/noisy ones
            # Allowed (in order): loss, pixel_mse, token_norm, max_token_norm, privacy_epsilon
            ordered_keys = [
                'loss',
                'pixel_mse',
                'token_norm',
                'max_token_norm',
                'privacy_epsilon',
            ]
            metrics_parts = []
            for k in ordered_keys:
                if k in global_metrics:
                    metrics_parts.append(f"{k}={global_metrics[k]:.4f}")
            # Add server-side DP cumulative epsilon logging (average over selected clients this round)
            if use_dp and selected:
                eps_prev = []
                eps_after = []
                eps_inc = []
                for cid in selected:
                    sr = client_dp_sr.get(cid, batch_size / max(1, len(clients_data[cid]['indices'])))
                    nm = client_dp_nm.get(cid, noise_multiplier if noise_multiplier is not None else 1.0)
                    steps_before = client_dp_steps_seen.get(cid, 0)
                    steps_add = client_dp_steps_per_round.get(cid, 0)
                    e_prev = _epsilon_from_rdp(nm, sr, steps_before, target_delta)
                    e_after = _epsilon_from_rdp(nm, sr, steps_before + steps_add, target_delta)
                    eps_prev.append(e_prev)
                    eps_after.append(e_after)
                    eps_inc.append(max(0.0, e_after - e_prev))
                    client_dp_steps_seen[cid] = steps_before + steps_add
                # Aggregate (mean) for logging
                eps_cum_avg = float(np.mean(eps_after)) if eps_after else 0.0
                eps_inc_avg = float(np.mean(eps_inc)) if eps_inc else 0.0
                metrics_parts.append(f"privacy_epsilon_cum={eps_cum_avg:.4f}")
                metrics_parts.append(f"privacy_epsilon_inc={eps_inc_avg:.4f}")
                metrics_parts.append(f"privacy_delta={target_delta:.1e}")
            metrics_str = ' '.join(metrics_parts)
            print(f'[Server] Round {rnd} global metrics: {metrics_str}')
            _log(f'[Round {rnd}] global metrics: {metrics_str}')
            
            # Log to wandb
            if wandb_project:
                wandb_metrics = {f'global_{k}': v for k, v in global_metrics.items()}
                wandb_metrics['round'] = rnd
                
                # Add communication and parameter efficiency metrics
                if client_states:
                    # Calculate communication cost (number of trainable parameters)
                    sample_state = client_states[0]
                    trainable_params = sum(p.numel() for p in sample_state.values() if isinstance(p, torch.Tensor))
                    total_params = sum(p.numel() for p in model.parameters())
                    
                    wandb_metrics['communication_cost'] = trainable_params
                    wandb_metrics['total_params'] = total_params
                    
                    # Avoid division by zero
                    if total_params > 0:
                        wandb_metrics['parameter_efficiency'] = trainable_params / total_params
                    else:
                        wandb_metrics['parameter_efficiency'] = 0.0
                    
                    if trainable_params > 0:
                        wandb_metrics['compression_ratio'] = total_params / trainable_params
                    else:
                        wandb_metrics['compression_ratio'] = 0.0
                
                wandb.log(wandb_metrics)

        # Periodic full aggregation
        force_full_agg = (full_agg_period > 0 and (rnd % full_agg_period == 0))

        if (aggregate_top_k and aggregate_top_k > 0 and client_states and not force_full_agg):
            # Construct and update EMA scores
            scored = []
            for cid, st in zip(client_ids, client_states):
                m = st.get('metrics', {})
                raw = 0.0
                if score_metric == 'acc':
                    raw = float(m.get('val_acc', m.get('acc', 0.0)))
                    # Higher acc is better
                    signed = raw
                else:
                    raw = float(m.get('val_loss', m.get('loss', 0.0)))
                    # Lower loss is better, use negative sign
                    signed = -raw
                prev = score_ema.get(cid, None)
                cur = signed if prev is None else (score_ema_alpha * signed + (1.0 - score_ema_alpha) * prev)
                score_ema[cid] = cur
                scored.append((cur, cid, st))

            scored.sort(key=lambda x: x[0], reverse=True)
            k = min(aggregate_top_k, len(scored))
            top = scored[:k]

            # Exploration: randomly replace a proportion of non-top clients
            if explore_prob > 0 and len(scored) > k:
                import random as _random
                explore_n = max(1, int(round(explore_prob * k)))
                pool = scored[k:]
                _random.shuffle(pool)
                replace = pool[:explore_n]
                # Combine: keep top (k - explore_n) + random explore_n
                top = top[:max(0, k - explore_n)] + replace

            top_states = [t[2] for t in top]
            top_ids = [t[1] for t in top]
            print(f"[Server] Aggregating {len(top_states)} clients (top-k with explore): {top_ids}")
            agg = fedavg(top_states)
        else:
            # Aggregate all client parameters (full aggregation or top-k disabled)
            if use_secure_agg and secure_aggregator:
                # Use secure aggregation
                client_updates_dict = {cid: state for cid, state in zip(client_ids, client_states)}
                agg = secure_aggregator.aggregate(client_updates_dict)
            elif server_noise_scale > 0.0:
                # Use privacy-preserving aggregation with server-side noise
                agg = fedavg_with_privacy(client_states, noise_scale=server_noise_scale)
            else:
                # Standard FedAvg
                agg = fedavg(client_states)
        load_trainable_state_dict(model, agg)
        # Log encoder hash after aggregation to track global updates
        try:
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                enc_hash = _hash_tensor_map(model.backbone.encoder.state_dict())
                _log(f'[Round {rnd}] encoder_full_hash={enc_hash[:16]}')
        except Exception:
            pass

        if secure_agg_template is not None:
            secure_agg_template = get_trainable_state_dict(model)

        # Memory cleanup after each round
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        # Save checkpoint for current round
        ckpt = {
            'round': rnd,
            'trainable': get_trainable_state_dict(model),
            'global_metrics': global_metrics if round_metrics else None,
        }
        torch.save(ckpt, os.path.join(out_dir, f'round_{rnd:03d}.pt'))

        # Optional validation on aggregated global model
        if val_loader is not None:
            model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    video = batch['video'].to(device)
                    label = batch['label'].to(device)
                    pred_loss, metrics = loss_fn(model, video, label)
                    bs = label.size(0)
                    total_loss += float(pred_loss.item()) * bs
                    total_correct += int((torch.argmax(model(video), dim=1) == label).sum().item())
                    total_samples += bs
            val_loss = total_loss / max(1, total_samples)
            val_acc = total_correct / max(1, total_samples)
            print(f"[Server] Validation @ round {rnd}: acc={val_acc:.4f} loss={val_loss:.4f}")
            _log(f"[Round {rnd}] validation: acc={val_acc:.4f} loss={val_loss:.4f}")

            # Track best model
            cur_score = val_acc if select_best_metric == 'acc' else -val_loss
            if best_score is None or cur_score > best_score:
                best_score = cur_score
                best_ckpt = {
                    'round': rnd,
                    'model': model.state_dict(),
                    'trainable': get_trainable_state_dict(model),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }
                best_path = os.path.join(out_dir, 'model_best.pth')
                torch.save(best_ckpt, best_path)
                print(f"[Server] New best model saved to {best_path} (metric={select_best_metric}, score={best_score:.6f})")
                _log(f"[Round {rnd}] new best saved: metric={select_best_metric} score={best_score:.6f}")
        
        # Save final model file at last round
        if rnd == rounds:
            final_ckpt = {
                'model': model.state_dict(),  # Save complete model state
                'trainable': get_trainable_state_dict(model),  # Save only trainable parameters
                'final_metrics': global_metrics if round_metrics else None,
                'total_rounds': rounds,
            }
            torch.save(final_ckpt, os.path.join(out_dir, 'model_final.pth'))
            print(f"\n[Server] Training completed! Final model saved to: {os.path.join(out_dir, 'model_final.pth')}")
            _log('[Done] training completed. final checkpoint saved.')
    
    # Finish wandb run
    if wandb_project:
        wandb.finish()
        print("[Server] Wandb run finished")


# import here to avoid circular
from fl.client import client_update as client_update_fn
