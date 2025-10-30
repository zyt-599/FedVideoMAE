from typing import Dict
import torch


def _unwrap_base_module(m: torch.nn.Module) -> torch.nn.Module:
    """Best‑effort unwrap to the original (non‑DP/non‑DDP) module.

    Handles Opacus/GradSample wrappers and common proxy attributes so that
    state_dict() keys align with named_parameters() names.
    """
    # Opacus GradSampleModuleProxy used in this repo exposes `_base_module`
    for attr in ("_base_module", "module", "_module", "_dp_inner"):
        base = getattr(m, attr, None)
        if isinstance(base, torch.nn.Module):
            # Keep unwrapping recursively in case of multiple wrappers
            return _unwrap_base_module(base)
    return m


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state dict containing only trainable parameters.

    Robust to DP/Opacus wrappers by unwrapping to the base module for both
    parameter name discovery and tensor extraction.
    """
    base = _unwrap_base_module(model)

    # Discover trainable parameter names from the base module
    trainable_names = {name for name, p in base.named_parameters() if p.requires_grad}

    # Use the base module's state_dict to ensure keys match the discovered names
    full_sd = base.state_dict()

    # Filter and move tensors to CPU for transport/aggregation
    return {k: v.detach().cpu() for k, v in full_sd.items() if k in trainable_names}


def load_trainable_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    missing = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k].copy_(v)
        else:
            missing.append(k)
    if missing:
        print(f'[Warn] Missing keys during load: {len(missing)}')
