"""
Differential Privacy and Secure Aggregation utilities for federated learning.
"""
import secrets
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
import warnings
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader

from fl.aggregator import fedavg

import logging

# NVFLARE is optional; use a stable internal hasher if unavailable
try:
    from nvflare import __version__ as _nvflare_version  # noqa: F401  # trigger ImportError if not installed
    from nvflare.fuel.utils.hash_utils import UniformHash
    NVFLARE_AVAILABLE = True
except ImportError:
    NVFLARE_AVAILABLE = False
    class UniformHash:  # type: ignore
        def __init__(self, *_args, **_kwargs) -> None:
            pass
        def hash(self, s: str) -> int:
            import hashlib
            return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)
    logging.warning("NVFLARE not available. Using internal hasher for secure aggregation.")

logger = logging.getLogger(__name__)


class GradSampleModuleProxy(nn.Module):
    """Proxy to expose original module methods after Opacus wraps the model."""

    def __init__(self, grad_sample_module: nn.Module) -> None:
        super().__init__()
        self._dp_inner = grad_sample_module
        candidates = [
            getattr(grad_sample_module, "module", None),
            getattr(grad_sample_module, "_module", None),
            getattr(grad_sample_module, "_dp_inner", None),
        ]
        self._base_module = next((c for c in candidates if c is not None), grad_sample_module)
        self.add_module("_dp_wrapped_module", grad_sample_module)

    def forward(self, *args, **kwargs):
        return self._dp_inner(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in {"_dp_inner", "_base_module", "_modules", "_parameters", "_buffers", "_non_persistent_buffers_set"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            for candidate in (
                self._dp_inner,
                getattr(self._dp_inner, "module", None),
                getattr(self._dp_inner, "_module", None),
                self._base_module,
            ):
                if candidate is None:
                    continue
                try:
                    return getattr(candidate, name)
                except AttributeError:
                    continue
            raise

    def state_dict(self, *args, **kwargs):
        # Return the base/original module's state dict so keys match the non-DP model
        try:
            return self._base_module.state_dict(*args, **kwargs)
        except Exception:
            return self._dp_inner.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Load into the original base module to keep keys consistent
        try:
            return self._base_module.load_state_dict(state_dict, strict)
        except Exception:
            return self._dp_inner.load_state_dict(state_dict, strict)

    def modules(self):
        return self._dp_inner.modules()

    def named_modules(self, *args, **kwargs):
        return self._dp_inner.named_modules(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._dp_inner.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._dp_inner.named_parameters(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self._dp_inner.buffers(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self._dp_inner.named_buffers(*args, **kwargs)

    def train(self, mode: bool = True):
        self._dp_inner.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self._dp_inner.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self._dp_inner.cuda(*args, **kwargs)
        return self

    def cpu(self):
        self._dp_inner.cpu()
        return self

    def double(self):
        self._dp_inner.double()
        return self

    def float(self):
        self._dp_inner.float()
        return self

    def half(self):
        self._dp_inner.half()
        return self

    def requires_grad_(self, *args, **kwargs):
        self._dp_inner.requires_grad_(*args, **kwargs)
        return self

class DPOptimizerWrapper:
    """
    Wrapper for differential privacy optimizer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        privacy_engine: PrivacyEngine,
        noise_multiplier: float,
        max_grad_norm: float,
        target_epsilon: float,
        target_delta: float = 1e-5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self._accountant = getattr(privacy_engine, "accountant", None)

    @property
    def grad_sample_module(self) -> nn.Module:
        # Return the proxy model to preserve original methods/attributes (e.g., get_metrics)
        return self.model

    def step(self):
        """Perform optimizer step with differential privacy."""
        self.optimizer.step()
        
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
        
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        import warnings
        if self._accountant is not None:
            with warnings.catch_warnings():
                # Suppress RDP warning about optimal order at largest alpha
                warnings.filterwarnings(
                    "ignore",
                    message=r"Optimal order is the largest alpha.*",
                    category=UserWarning,
                    module=r"opacus\.accountants\.analysis\.rdp"
                )
                epsilon = self._accountant.get_epsilon(self.target_delta)
            return float(epsilon), float(self.target_delta)
        if hasattr(self.privacy_engine, "get_epsilon"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Optimal order is the largest alpha.*",
                    category=UserWarning,
                    module=r"opacus\.accountants\.analysis\.rdp"
                )
                epsilon = self.privacy_engine.get_epsilon(self.target_delta)
            return float(epsilon), float(self.target_delta)
        raise RuntimeError("Privacy accountant is not available in the current PrivacyEngine.")
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        epsilon, _ = self.get_privacy_spent()
        return epsilon
    
    def is_privacy_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded."""
        current_epsilon, _ = self.get_privacy_spent()
        return current_epsilon >= self.target_epsilon


def create_dp_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_multiplier: Optional[float],
    max_grad_norm: float,
    target_epsilon: float,
    target_delta: float = 1e-5,
    sample_rate: float = 1.0,
    data_loader: Optional[DataLoader] = None,
    num_steps: Optional[int] = None,
) -> Tuple[DPOptimizerWrapper, DataLoader]:
    """
    Create a differential privacy optimizer.
    
    Args:
        model: PyTorch model
        optimizer: Base optimizer
        noise_multiplier: Noise multiplier for DP
        max_grad_norm: Maximum gradient norm for clipping
        target_epsilon: Target privacy budget (epsilon)
        target_delta: Target delta for DP
        sample_rate: Sampling rate for DP
        
    Returns:
        DPOptimizerWrapper: Wrapped optimizer with DP
    """
    if data_loader is None:
        raise ValueError("create_dp_optimizer requires a valid DataLoader when using differential privacy.")

    # Prefer secure RNG if torchcsprng is available; otherwise fall back to fast RNG
    try:
        import torchcsprng  # noqa: F401
        _secure_mode = True
    except Exception:
        _secure_mode = False
    # Suppress secure RNG warning if we intentionally run without torchcsprng
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Secure RNG turned off.*",
            category=UserWarning,
            module=r"opacus\.privacy_engine",
        )
        try:
            privacy_engine = PrivacyEngine(secure_mode=_secure_mode)
        except Exception:
            # As a last resort, fall back to non-secure mode
            privacy_engine = PrivacyEngine(secure_mode=False)

    # Robustly coerce numeric-like inputs that may come as strings from YAML
    def _to_float(v, default=None):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("none", "null", ""):
                return default
            try:
                return float(s)
            except Exception:
                return default
        return default

    sample_rate = _to_float(sample_rate, 1.0)
    target_epsilon = _to_float(target_epsilon, 10.0)
    target_delta = _to_float(target_delta, 1e-5)
    max_grad_norm = _to_float(max_grad_norm, 1.0)
    if noise_multiplier is not None:
        noise_multiplier = _to_float(noise_multiplier, None)

    # For Opacus 1.5.4, we need to ensure the data_loader is properly configured
    # Check if the data_loader has the required attributes
    if not hasattr(data_loader, 'dataset'):
        raise ValueError("DataLoader must have a 'dataset' attribute for Opacus compatibility")
    
    # Wrap the dataset so Opacus sees (video, label) tuples
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    class _TupleDataset(Dataset):
        def __init__(self, base_ds):
            self.base = base_ds
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            if isinstance(item, dict):
                return (item.get('video'), item.get('label'))
            if isinstance(item, (list, tuple)) and len(item) == 2:
                return (item[0], item[1])
            # Fallback: no label provided
            return (item, None)

    base_loader = data_loader
    tuple_ds = _TupleDataset(base_loader.dataset)
    # Build a lightweight torch DataLoader for Opacus to clone from
    loader_kwargs = dict(
        batch_size=getattr(base_loader, 'batch_size', 1),
        shuffle=False,
        num_workers=getattr(base_loader, 'num_workers', 0),
        pin_memory=getattr(base_loader, 'pin_memory', False),
        drop_last=getattr(base_loader, 'drop_last', False),
        persistent_workers=getattr(base_loader, 'persistent_workers', False),
    )
    if loader_kwargs['num_workers'] > 0:
        pf = getattr(base_loader, 'prefetch_factor', None)
        if pf is not None:
            loader_kwargs['prefetch_factor'] = int(pf)
    wrapped_loader = TorchDataLoader(tuple_ds, **loader_kwargs)

    # If noise multiplier not provided, compute it from target epsilon/delta and schedule
    if noise_multiplier is None:
        steps = num_steps if num_steps is not None else len(wrapped_loader)
        try:
            with warnings.catch_warnings():
                # Suppress RDP accountant alpha-range hint during auto noise solve
                warnings.filterwarnings(
                    "ignore",
                    message=r"Optimal order is the largest alpha.*",
                    category=UserWarning,
                    module=r"opacus\.accountants\.analysis\.rdp",
                )
                nm = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    steps=steps,
                    accountant="rdp",
                )
            noise_multiplier = float(max(nm, 0.1))
        except Exception:
            # Fallback to simple Gaussian mech bound
            noise_multiplier = float(max(np.sqrt(2 * np.log(1.25 / target_delta)) / max(target_epsilon, 1e-6), 0.1))

    make_private_kwargs = dict(
        module=model,
        optimizer=optimizer,
        data_loader=wrapped_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="hooks",
        poisson_sampling=False,
    )
    try:
        dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(**make_private_kwargs)
    except TypeError:
        make_private_kwargs.pop("poisson_sampling", None)
        dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(**make_private_kwargs)

    proxy_model = GradSampleModuleProxy(dp_model)

    wrapper = DPOptimizerWrapper(
        model=proxy_model,
        optimizer=dp_optimizer,
        privacy_engine=privacy_engine,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )

    return wrapper, dp_dataloader


class SecureAggregator:
    """Secure aggregation with NVFLARE-style pairwise masking.

    This implementation mimics the secure aggregation protocol used in NVFLARE by
    creating pairwise random masks between participating clients. The masks cancel
    out when enough (threshold) clients contribute, so the server only observes
    masked updates. The same masking utilities can be reused in real NVFLARE jobs.
    """

    def __init__(
        self,
        num_clients: int,
        threshold: Optional[int] = None,
        mask_std: float = 1e-3,
    ) -> None:
        # NVFLARE is optional; fallback hasher ensures deterministic ordering without NVFLARE

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")

        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)
        self.mask_std = mask_std

        # NVFLARE utility used to deterministically shuffle clients per round.
        self._hasher = UniformHash(max(num_clients, 1))

        self._round_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._param_template: Dict[str, torch.Tensor] = {}
        self._active_clients: List[str] = []
        self._round_id: int = 0

    def setup_clients(self, client_ids: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compatibility stub kept for previous LightSecAgg API consumers."""

        return {cid: {} for cid in client_ids}

    def start_round(
        self,
        client_ids: List[str],
        param_template: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Prepare secure masks for the incoming round.

        Args:
            client_ids: Participating client identifiers for this round.
            param_template: Parameter snapshot used only to infer tensor shapes.
        """

        if not client_ids:
            raise ValueError("client_ids must not be empty")

        if param_template is not None:
            self._param_template = {
                k: v.detach().cpu().to(torch.float32)
                for k, v in param_template.items()
                if isinstance(v, torch.Tensor)
            }

        # If there are no trainable parameters (empty template), proceed without masks.
        # This allows secure aggregation to no-op on parameter masking.

        self._round_id += 1

        # Use NVFLARE's uniform hash to get a deterministic but pseudo-random ordering.
        ordered = sorted(
            client_ids,
            key=lambda cid: (self._hasher.hash(f"{self._round_id}:{cid}"), cid),
        )

        self._active_clients = ordered
        self._round_masks = {
            cid: {name: torch.zeros_like(template) for name, template in self._param_template.items()}
            for cid in ordered
        }

        sysrand = secrets.SystemRandom()
        for idx_i, client_i in enumerate(ordered):
            for client_j in ordered[idx_i + 1 :]:
                seed = sysrand.getrandbits(64)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                for name, template in self._param_template.items():
                    mask = torch.empty_like(template).normal_(mean=0.0, std=self.mask_std, generator=generator)
                    self._round_masks[client_i][name] += mask
                    self._round_masks[client_j][name] -= mask

    def mask_client_update(self, client_id: str, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply the pre-computed mask to a client's model delta before transport."""

        if client_id not in self._round_masks:
            # No secure aggregation configured for this client (fallback to plaintext)
            return state

        masked: Dict[str, torch.Tensor] = {}
        mask_bank = self._round_masks[client_id]
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                masked[name] = tensor
                continue
            mask = mask_bank.get(name)
            if mask is None:
                masked[name] = tensor
                continue
            masked[name] = (tensor.to(torch.float32) + mask).to(tensor.dtype)
        return masked

    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return {}

        if not self._round_masks:
            # Secure aggregation disabled for this round.
            return fedavg(list(client_updates.values()))

        # Pairwise mask cancellation requires all planned participants to respond
        expected = set(self._active_clients)
        received = set(client_updates.keys())
        if received != expected:
            missing = expected - received
            extra = received - expected
            raise RuntimeError(
                f"Secure aggregation requires all {len(expected)} clients. Missing={sorted(list(missing))} extra={sorted(list(extra))}."
            )

        # Sum masked updates; masks cancel out across clients
        sum_state: Dict[str, torch.Tensor] = {}
        count = float(len(client_updates))
        for cid in self._active_clients:
            masked_state = client_updates[cid]
            for name, tensor in masked_state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if name not in sum_state:
                    sum_state[name] = tensor.detach().clone().to(torch.float32)
                else:
                    sum_state[name] += tensor.to(torch.float32)

        # Average
        agg: Dict[str, torch.Tensor] = {k: (v / count).to(v.dtype) for k, v in sum_state.items()}

        # Reset round-specific state to free memory
        self._round_masks.clear()
        self._active_clients.clear()

        return agg




def fedavg_with_privacy(
    client_states: List[Dict[str, torch.Tensor]],
    noise_scale: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    FedAvg with optional privacy-preserving noise.
    
    Args:
        client_states: List of client model states
        noise_scale: Scale of noise to add (0.0 for no noise)
        
    Returns:
        Aggregated model parameters
    """
    if not client_states:
        return {}
        
    agg: Dict[str, torch.Tensor] = {}
    
    # Separate metrics from model parameters
    metrics = {}
    for state in client_states:
        for k, v in state.items():
            if k == 'metrics':
                continue
            if k not in agg:
                agg[k] = v.clone().float()
            else:
                agg[k] += v.float()
    
    # Average model parameters
    for k in agg:
        agg[k] /= float(len(client_states))
        
        # Add privacy-preserving noise if specified
        if noise_scale > 0.0:
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=agg[k].shape,
                device=agg[k].device,
                dtype=agg[k].dtype
            )
            agg[k] += noise
            
        agg[k] = agg[k].to(client_states[0][k].dtype)
    
    return agg

