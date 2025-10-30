from typing import Dict, Optional, Callable
import copy as _copy
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collate import video_collate
from fl.serialization import get_trainable_state_dict
from fl.privacy import DPOptimizerWrapper, create_dp_optimizer


def _make_loader(dataset, indices, batch_size, num_workers, *,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True):
    subset = Subset(dataset, indices)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=video_collate,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        kwargs['prefetch_factor'] = int(prefetch_factor)
    return DataLoader(subset, **kwargs)


def client_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    dataset,
    indices,
    batch_size: int,
    num_workers: int,
    local_epochs: int,
    device: torch.device,
    client_id: str = "Client",
    clip_grad: float = 0.0,
    log_every: int = 50,
    loader: Optional[DataLoader] = None,
    use_amp: Optional[bool] = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    # Optional secure aggregation masking callable (applied before returning state)
    mask_fn: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    # Differential Privacy parameters
    use_dp: bool = False,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_epsilon: float = 10.0,
    target_delta: float = 1e-5,
    sample_rate: float = 1.0,
    # Memory controls
    clear_cache_steps: int = 0,
    max_physical_batch_size: Optional[int] = None,
) -> Dict:
    model.train()
    if loader is None:
        loader = _make_loader(
            dataset, indices, batch_size, num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
    
    # Setup differential privacy if enabled
    dp_wrapper: Optional[DPOptimizerWrapper] = None
    dp_loader: Optional[DataLoader] = None
    if use_dp:
        try:
            # Use a local copy of the model to avoid persistent hooks on the shared instance
            local_model = _copy.deepcopy(model)
            # Mirror base optimizer hyperparameters for the local copy
            base_opt_cls = optimizer.__class__
            base_defaults = getattr(optimizer, 'defaults', {}).copy() if hasattr(optimizer, 'defaults') else {}
            # Filter out keys not accepted by the optimizer constructor (e.g. decoupled_weight_decay)
            try:
                import inspect as _inspect
                _sig = _inspect.signature(base_opt_cls.__init__)
                _allowed = {p.name for p in _sig.parameters.values()}
                # Common kwargs across torch optimizers
                _allowed.update({
                    'lr', 'momentum', 'weight_decay', 'betas', 'eps', 'amsgrad',
                    'maximize', 'foreach', 'capturable', 'differentiable', 'fused'
                })
                _filtered = {k: v for k, v in base_defaults.items() if k in _allowed}
                if len(_filtered) != len(base_defaults):
                    # Only print when something was dropped to help debugging env differences
                    dropped = sorted(set(base_defaults.keys()) - set(_filtered.keys()))
                    print(f"[Client{client_id}] Filtering optimizer defaults (dropped: {dropped})")
                base_defaults = _filtered
            except Exception:
                # Fallback: drop known problematic keys
                for _k in ('decoupled_weight_decay', 'weight_decay_norm', 'weight_decay_bias'):
                    if _k in base_defaults:
                        base_defaults.pop(_k, None)
            local_params = [p for p in local_model.parameters() if p.requires_grad]
            local_optimizer = base_opt_cls(local_params, **base_defaults) if local_params else base_opt_cls(local_model.parameters(), **base_defaults)

            dp_wrapper, dp_loader = create_dp_optimizer(
                model=local_model,
                optimizer=local_optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                data_loader=loader,
                num_steps=(local_epochs * len(loader)) if loader is not None else None,
            )
            # Log the resolved (possibly auto-computed) noise multiplier
            try:
                resolved_nm = getattr(dp_wrapper, 'noise_multiplier', None)
            except Exception:
                resolved_nm = None
            nm_str = f"{resolved_nm}" if resolved_nm is not None else (f"{noise_multiplier}" if noise_multiplier is not None else "auto")
            print(f"[Client{client_id}] Differential privacy enabled: noise_multiplier={nm_str}, max_grad_norm={max_grad_norm}")
        except Exception as e:
            print(f"[Client{client_id}] Failed to setup differential privacy: {e}")
            print(f"[Client{client_id}] Falling back to standard (non-DP) training")
            dp_wrapper, dp_loader = None, None

    # Select active components depending on DP availability
    active_model = dp_wrapper.grad_sample_module if dp_wrapper else model
    active_loader = dp_loader if dp_wrapper and dp_loader is not None else loader
    active_optimizer = dp_wrapper.optimizer if dp_wrapper else optimizer
    
    # Calculate total steps and samples for progress bar
    total_batches = len(active_loader) * local_epochs
    total_samples = len(indices) * local_epochs
    
    # Create progress bar with custom format (track samples as total)
    pbar = tqdm(
        total=total_samples,
        desc=f'[Client{client_id}]',
        unit='samples',
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {postfix} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # Set initial postfix
    pbar.set_postfix_str(f'0/{total_samples} samples')
    
    step = 0
    processed_samples = 0
    # AMP setup (disable under DP to avoid dtype issues with functorch/Opacus)
    amp_enabled = bool(use_amp) and (dp_wrapper is None)
    try:
        from torch.amp import autocast as _autocast, GradScaler as _GradScaler  # type: ignore
        amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = _GradScaler(amp_device, enabled=amp_enabled)
        autocast_ctx = lambda: _autocast(amp_device, enabled=amp_enabled)
    except Exception:
        from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler  # type: ignore
        scaler = _GradScaler(enabled=amp_enabled)
        autocast_ctx = lambda: _autocast(enabled=amp_enabled)
    
    for epoch in range(local_epochs):
        # Always try Opacus BatchMemoryManager to cap physical batch size (works for DP and non-DP)
        iter_loader = active_loader
        _ctx = None
        if active_loader is not None:
            try:
                from opacus.utils.batch_memory_manager import BatchMemoryManager
                phys_bs = int(max_physical_batch_size) if (isinstance(max_physical_batch_size, int) and max_physical_batch_size > 0) else (max(1, batch_size // 4) if batch_size and batch_size > 1 else 1)
                # When DP is disabled, provide optimizer shim for signal_skip_step
                if dp_wrapper is None and not hasattr(active_optimizer, "_bmm_skip_wrap"):
                    import types as _types
                    _orig_step = active_optimizer.step
                    def _signal_skip_step(self, do_skip: bool):
                        self._bmm_skip_next = bool(do_skip)
                    def _wrapped_step(self, *args, **kwargs):
                        if getattr(self, "_bmm_skip_next", False):
                            self._bmm_skip_next = False
                            return None
                        return _orig_step(*args, **kwargs)
                    active_optimizer._bmm_skip_next = False
                    active_optimizer.step = _types.MethodType(_wrapped_step, active_optimizer)
                    active_optimizer.signal_skip_step = _types.MethodType(_signal_skip_step, active_optimizer)
                    active_optimizer._bmm_skip_wrap = True
                _ctx = BatchMemoryManager(data_loader=active_loader, max_physical_batch_size=phys_bs, optimizer=active_optimizer)
                iter_loader = _ctx
            except Exception:
                _ctx = None

        if _ctx is not None:
            _cm = _ctx
        else:
            # Dummy context manager
            from contextlib import nullcontext as _nullctx
            _cm = _nullctx()

        with _cm as _safe_loader:
            real_loader = _safe_loader if _ctx is not None else iter_loader
            for batch in real_loader:
                # Support dict batches (our datasets) and tuple batches (Opacus)
                if isinstance(batch, dict):
                    x = batch['video']
                    y = batch.get('label')
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch  # type: ignore
                else:
                    # Fallback: treat as single tensor input (unsupervised)
                    x, y = batch, None
                # Skip empty batches (can happen with Opacus Poisson sampling)
                try:
                    if x is None or (hasattr(x, 'numel') and x.numel() == 0) or (hasattr(x, 'size') and x.size(0) == 0):
                        continue
                except Exception:
                    pass
                x = x.to(device, non_blocking=True)  # (B, T, C, H, W)
                if y is not None:
                    y = y.to(device, non_blocking=True)
                # Standard single-batch backward; BMM will split internally if needed
                with autocast_ctx():
                    loss_out = loss_fn(active_model, x, y)
                if isinstance(loss_out, tuple):
                    loss, metrics = loss_out
                else:
                    loss = loss_out
                    metrics = {'loss': loss.item()}
                active_optimizer.zero_grad()
                scaler.scale(loss).backward()
                if clip_grad > 0 and not dp_wrapper:
                    scaler.unscale_(active_optimizer)
                    torch.nn.utils.clip_grad_norm_(active_model.parameters(), clip_grad)
                scaler.step(active_optimizer)
                scaler.update()
                step += 1
                
                # Calculate current sample count using actual batch size
                bs_cur = int(x.size(0)) if hasattr(x, 'size') else batch_size
                processed_samples = min(processed_samples + bs_cur, total_samples)
                current_samples = processed_samples
                
                # Update progress bar with sample count only
                pbar.set_postfix_str(f'{current_samples}/{total_samples} samples')
                
                # Advance by actual processed samples; clamp to remaining
                inc = max(0, min(bs_cur, total_samples - pbar.n))
                if inc:
                    pbar.update(inc)

                # Optional in-loop cache clearing to reduce fragmentation
                if clear_cache_steps and (step % max(1, int(clear_cache_steps)) == 0):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
    
    pbar.close()

    # Detach privacy engine to avoid double-hook issues on the shared model instance
    if dp_wrapper is not None:
        try:
            pe = getattr(dp_wrapper, 'privacy_engine', None)
            if pe is not None and hasattr(pe, 'detach'):
                pe.detach()
        except Exception:
            pass
    
    # Print final metrics and privacy information
    metrics_str = ' '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
    print(f'[Client{client_id}] Final: {metrics_str}')
    
    # Add privacy information if DP was used
    if dp_wrapper:
        try:
            epsilon, delta = dp_wrapper.get_privacy_spent()
            print(f'[Client{client_id}] Privacy spent: epsilon={epsilon:.4f}, delta={delta:.6f}')
            metrics['privacy_epsilon'] = epsilon
            metrics['privacy_delta'] = delta
        except Exception as e:
            print(f'[Client{client_id}] Could not get privacy information: {e}')

    # Return final state including model parameters and metrics from last batch
    final_state = get_trainable_state_dict(active_model)
    final_state['metrics'] = metrics  # Add metrics from last batch
    # If secure aggregation mask function is provided, apply mask on client side before returning
    if mask_fn is not None:
        try:
            final_state = mask_fn(final_state)
        except Exception as _e:
            # On failure, return unmasked state, let upper layer handle it
            pass
    return final_state
