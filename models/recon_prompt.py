import torch
import torch.nn as nn


class ReconPrompt(nn.Module):
    """
    A simple reconstructive prompt: a learnable affine on reconstruction tokens.
    Can be extended to token-injection if backbone APIs are exposed.
    Input: reconstruction tokens from VideoMAE decoder, shape (B, N, decoder_dim)
    where decoder_dim = 3 * patch_size * patch_size (usually 1536 = 3 * 16 * 16)
    """
    def __init__(self, dim: int = 768, length: int = 8):
        super().__init__()
        self.prompt_dim = dim  # Use passed dimension
        self.decoder_dim = 1536  # VideoMAE decoder fixed output dimension
        
        # Initialize parameters using prompt dimension
        self.prompt = nn.Parameter(torch.zeros(length, self.prompt_dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)
        
        # Projection layer: from prompt dimension to decoder dimension
        self.proj = nn.Linear(self.prompt_dim, self.decoder_dim)

    def forward(self, recon_tokens: torch.Tensor) -> torch.Tensor:
        # recon_tokens: (B, N, decoder_dim) from decoder output
        if recon_tokens.dim() < 3:
            return self.proj(recon_tokens)

        # Project prompt to decoder dimension
        p = self.prompt.mean(dim=0, keepdim=True)  # (1, prompt_dim)
        p_proj = self.proj(p)  # (1, decoder_dim)

        # Add to reconstruction tokens
        out = recon_tokens + p_proj
        return out


# Opacus grad sampler support so DP training can compute per-sample gradients
try:
    from opacus.grad_sample.utils import create_or_extend_grad_sample, register_grad_sampler

    @register_grad_sampler(ReconPrompt)
    def _recon_prompt_grad_sampler(module: ReconPrompt, activations, backprops):
        if not backprops:
            return
        grad_out = backprops[0]
        if grad_out is None:
            return
        if grad_out.dim() < 2:
            grad_out = grad_out.unsqueeze(1)
        grad_out = grad_out.reshape(grad_out.shape[0], -1, grad_out.shape[-1])

        # Each prompt copy is broadcast across tokens; sum token gradients per sample
        grad_prompt_proj = grad_out.sum(dim=1)  # (batch, decoder_dim)

        proj_module = module.proj
        weight = proj_module.weight.detach()
        try:
            from models.peft_lora import LoRALinear  # local import to avoid cycle on init
            if isinstance(proj_module, LoRALinear):
                base_weight = proj_module.base.weight.detach()
                if getattr(proj_module, "r", 0) > 0 and proj_module.A is not None and proj_module.B is not None:
                    lora_update = proj_module.B.detach() @ proj_module.A.detach()
                    weight = base_weight + proj_module.scaling * lora_update
                else:
                    weight = base_weight
        except Exception:
            pass
        weight = weight.to(grad_prompt_proj.dtype)
        grad_prompt = grad_prompt_proj @ weight  # (batch, prompt_dim)

        if not getattr(module.prompt, "requires_grad", False):
            return

        prompt_rows = module.prompt.shape[0]
        if prompt_rows > 0:
            grad_prompt = grad_prompt / float(prompt_rows)
            grad_prompt = grad_prompt.unsqueeze(1).repeat(1, prompt_rows, 1)

        grad_prompt = grad_prompt.to(module.prompt.dtype).contiguous()
        create_or_extend_grad_sample(module.prompt, grad_prompt, batch_first=True)
except ImportError:
    pass
