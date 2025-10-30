from typing import Iterable, List, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_name(name: str, targets: List[str]) -> bool:
    return any(t in name for t in targets)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        if self.r > 0:
            self.A = nn.Parameter(torch.zeros((self.r, base.in_features)))
            self.B = nn.Parameter(torch.zeros((base.out_features, self.r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # align dtype to avoid matmul dtype mismatch
        try:
            w_dtype = self.base.weight.dtype
            if x.dtype != w_dtype:
                x = x.to(w_dtype)
        except Exception:
            pass
        out = self.base(x)
        if self.r > 0:
            lora = (self.B @ (self.A @ x.transpose(-1, -2))).transpose(-1, -2)
            out = out + self.dropout(lora) * self.scaling
        return out


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[List[str]] = None,
    dp_training: bool = False,
) -> List[str]:
    if r <= 0:
        return []
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2', 'proj']

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _match_name(name, target_modules):
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last = parts[-1]
            lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, last, lora_module)
            replaced.append(name)

    if len(replaced) == 0:
        # fall back: if recon_prompt exists, train it only; otherwise keep original flags
        has_prompt = hasattr(model, 'recon_prompt') and isinstance(getattr(model, 'recon_prompt'), nn.Module)
        if has_prompt:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.recon_prompt.parameters():
                p.requires_grad = True
            print('[LoRA] No target Linear matched; training recon_prompt only.')
        else:
            print('[LoRA] No target Linear matched; keeping model as-is.')
        return replaced

    # freeze non-LoRA params, unfreeze LoRA (+ prompt if any)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A is not None:
                m.A.requires_grad = True
            if m.B is not None:
                m.B.requires_grad = True
            for p in m.base.parameters():
                p.requires_grad = False
    if hasattr(model, 'recon_prompt') and isinstance(getattr(model, 'recon_prompt'), nn.Module):
        for p in model.recon_prompt.parameters():
            p.requires_grad = True

    # Debug summary: how many layers got LoRA, and a few samples
    try:
        show = 8
        sample = replaced[:show]
        more = ' ...' if len(replaced) > show else ''
        # quick trainable count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] Injected {len(replaced)} layers -> LoRALinear; samples: {sample}{more}; trainable params={trainable_params}")
    except Exception:
        pass

    return replaced


def trainable_parameters_filter(model: nn.Module) -> Iterable[nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def _strip_prefixes(name: str) -> str:
    """Normalize a parameter/module name by stripping common leading wrappers.

    Drops leading numeric indices and wrappers such as 'module', 'bb', 'backbone'.
    Keeps 'encoder' since it disambiguates scope within the backbone.
    """
    parts = name.split('.')
    while parts and (parts[0].isdigit() or parts[0] in ("module", "bb", "backbone")):
        parts = parts[1:]
    return '.'.join(parts)


def load_lora_from_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> int:
    """Load LoRA A/B matrices from a checkpoint state_dict into the model.

    Args:
        model: Model that already has LoRA modules injected.
        state: A flat state_dict loaded from checkpoint (e.g., checkpoint['model']).

    Returns:
        Number of LoRA adapter pairs (A,B) successfully loaded.
    """
    # Build normalized lookup for checkpoint keys
    ck_norm_map: Dict[str, str] = {}
    for k in state.keys():
        ck_norm_map[_strip_prefixes(k)] = k

    loaded = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a_key_norm = _strip_prefixes(f"{name}.A")
        b_key_norm = _strip_prefixes(f"{name}.B")
        src_a_key = ck_norm_map.get(a_key_norm, None)
        src_b_key = ck_norm_map.get(b_key_norm, None)
        if src_a_key is None or src_b_key is None:
            continue
        try:
            A = state[src_a_key]
            B = state[src_b_key]
            if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and A.shape == module.A.shape and B.shape == module.B.shape:  # type: ignore[union-attr]
                with torch.no_grad():
                    module.A.copy_(A)  # type: ignore[union-attr]
                    module.B.copy_(B)  # type: ignore[union-attr]
                loaded += 1
        except Exception:
            # Skip incompatible shapes or unexpected entries
            continue
    if loaded > 0:
        print(f"[LoRA] Loaded {loaded} adapter(s) from checkpoint state_dict")
    else:
        print("[LoRA] No matching A/B adapters found in checkpoint state_dict")
    return loaded


def try_load_lora_from_checkpoint(model: nn.Module, ckpt_path: Optional[str]) -> int:
    """Best-effort load of LoRA A/B from a checkpoint file.

    Returns number of loaded adapter pairs. Safe to call when the checkpoint
    has no LoRA weights or the model has no LoRA modules.
    """
    if not ckpt_path:
        return 0
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get('model', raw) if isinstance(raw, dict) else raw
        if not isinstance(state, dict):
            return 0
        return load_lora_from_state_dict(model, state)
    except Exception:
        return 0
