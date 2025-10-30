from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import random
import math


def build_video_transform(size: int = 224, normalize: bool = True):
    """Vectorized, semantics-equivalent transform (Resize -> CenterCrop -> Normalize).

    Input: sample['video'] as (T, H, W, C) uint8
    Output: (T, C, size, size) float32 (ImageNet-normalized when normalize=True)
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _apply(sample: Dict):
        v = sample['video']  # (T, H, W, C)
        if v.dtype != torch.float32:
            v = v.float().mul_(1.0 / 255.0)

        # (T, C, H, W)
        v = v.permute(0, 3, 1, 2).contiguous()

        _, _, H, W = v.shape
        if H == 0 or W == 0:
            v = torch.zeros((v.shape[0], 3, size, size), dtype=v.dtype)
        else:
            # scale shorter side to `size` then center crop
            scale = float(size) / float(min(H, W))
            new_h = max(size, int(round(H * scale)))
            new_w = max(size, int(round(W * scale)))
            v = F.interpolate(v, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
            top = (new_h - size) // 2
            left = (new_w - size) // 2
            v = v[:, :, top:top + size, left:left + size]

        if normalize:
            m = mean.to(v.device, dtype=v.dtype)
            s = std.to(v.device, dtype=v.dtype)
            v = (v - m) / s

        sample['video'] = v
        return sample

    return _apply


def build_video_transform_train(
    size: int = 224,
    normalize: bool = True,
    *,
    jitter_b: float = 0.2,
    jitter_c: float = 0.15,
    jitter_s: float = 0.15,
    grayscale_p: float = 0.0,
    erase_p: float = 0.0,
    erase_scale: Tuple[float, float] = (0.02, 0.08),
    erase_ratio: Tuple[float, float] = (0.3, 3.3),
):
    """Augmentations for training: RandomResizedCrop -> RandHFlip -> ColorJitter -> Normalize.

    Input: sample['video'] as (T, H, W, C) uint8
    Output: (T, C, size, size) float32 (ImageNet-normalized when normalize=True)
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _random_resized_crop(v: torch.Tensor) -> torch.Tensor:
        # v: (T, C, H, W) in [0,1]
        _, _, H, W = v.shape
        if H == 0 or W == 0:
            return torch.zeros((v.shape[0], 3, size, size), dtype=v.dtype, device=v.device)
        area = H * W
        for _ in range(10):
            target_area = random.uniform(0.6, 1.0) * area
            aspect = math.exp(random.uniform(math.log(3/4), math.log(4/3)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            if 0 < crop_h <= H and 0 < crop_w <= W:
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                v = v[:, :, top:top + crop_h, left:left + crop_w]
                v = F.interpolate(v, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
                return v
        # fallback to center crop after resize (shorter to = size)
        scale = float(size) / float(min(H, W))
        new_h = max(size, int(round(H * scale)))
        new_w = max(size, int(round(W * scale)))
        v = F.interpolate(v, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
        top = (new_h - size) // 2
        left = (new_w - size) // 2
        return v[:, :, top:top + size, left:left + size]

    def _hflip(v: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(v, dims=[-1])
        return v

    def _color_jitter(v: torch.Tensor, b: float, c: float, s: float) -> torch.Tensor:
        # brightness
        bf = 1.0 + random.uniform(-b, b)
        v = v * bf
        # contrast (around per-frame mean)
        if c > 0:
            cf = 1.0 + random.uniform(-c, c)
            mean = v.mean(dim=[2, 3], keepdim=True)
            v = (v - mean) * cf + mean
        # saturation: blend with grayscale
        if s > 0:
            sf = 1.0 + random.uniform(-s, s)
            gray = (0.2989 * v[:, 0:1] + 0.5870 * v[:, 1:2] + 0.1140 * v[:, 2:3])
            v = v * sf + gray * (1.0 - sf)
        return v.clamp_(0.0, 1.0)

    def _random_grayscale(v: torch.Tensor, p: float) -> torch.Tensor:
        if p > 0.0 and random.random() < p:
            gray = (0.2989 * v[:, 0:1] + 0.5870 * v[:, 1:2] + 0.1140 * v[:, 2:3])
            v = gray.repeat(1, 3, 1, 1)
        return v

    def _random_erasing(v: torch.Tensor, p: float, scale: Tuple[float, float], ratio: Tuple[float, float]) -> torch.Tensor:
        if p <= 0.0 or random.random() >= p:
            return v
        _, _, H, W = v.shape
        area = H * W
        for _ in range(10):
            target_area = random.uniform(scale[0], scale[1]) * area
            log_ratio_min, log_ratio_max = math.log(ratio[0]), math.log(ratio[1])
            aspect = math.exp(random.uniform(log_ratio_min, log_ratio_max))
            h = int(round(math.sqrt(target_area / aspect)))
            w = int(round(math.sqrt(target_area * aspect)))
            if 0 < h <= H and 0 < w <= W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                # erase with zeros (before normalization)
                v[:, :, top:top + h, left:left + w] = 0.0
                break
        return v

    def _apply(sample: Dict):
        v = sample['video']  # (T, H, W, C)
        if v.dtype != torch.float32:
            v = v.float().mul_(1.0 / 255.0)
        # (T, C, H, W)
        v = v.permute(0, 3, 1, 2).contiguous()
        # random resized crop
        v = _random_resized_crop(v)
        # flip
        v = _hflip(v, p=0.5)
        # jitter
        v = _color_jitter(v, b=jitter_b, c=jitter_c, s=jitter_s)
        # optional grayscale
        v = _random_grayscale(v, p=grayscale_p)
        # optional random erasing
        v = _random_erasing(v, p=erase_p, scale=erase_scale, ratio=erase_ratio)
        if normalize:
            m = mean.to(v.device, dtype=v.dtype)
            s_ = std.to(v.device, dtype=v.dtype)
            v = (v - m) / s_
        sample['video'] = v
        return sample

    return _apply
