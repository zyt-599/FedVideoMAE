import torch

def get_masked_patch_tokens(x: torch.Tensor, bool_mask: torch.Tensor, patch_embed) -> torch.Tensor:
    """Get patch tokens from masked positions in the original video input.
    
    Args:
        x: Input video tensor with shape [B, C, T, H, W]
        bool_mask: Boolean mask with shape [B, num_patches], True indicates masked position
        patch_embed: PatchEmbed module for converting video to tokens
        
    Returns:
        Patch tokens with shape [B, num_masked, patch_dim]
    """
    # First get all patch tokens through patch_embed
    all_tokens = patch_embed(x)  # [B, N, C]
    
    # Select masked tokens
    B, N, C = all_tokens.shape
    masked_tokens = all_tokens[bool_mask].reshape(B, -1, C)
    
    return masked_tokens
