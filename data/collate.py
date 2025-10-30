from typing import List, Dict, Tuple
import torch


def video_collate(batch: List[Dict]):
    videos = [b['video'] for b in batch]
    labels = [b.get('label', None) for b in batch]
    paths = [b.get('path', '') for b in batch]
    # pad time dimension to max T
    max_t = max(v.shape[0] for v in videos)
    C = videos[0].shape[1]
    H = videos[0].shape[2]
    W = videos[0].shape[3]
    padded = []
    for v in videos:
        if v.shape[0] == max_t:
            padded.append(v)
        else:
            pad = v[-1:].repeat(max_t - v.shape[0], 1, 1, 1)
            padded.append(torch.cat([v, pad], dim=0))
    x = torch.stack(padded, dim=0)  # (B, T, C, H, W)
    y = torch.stack([l for l in labels], dim=0) if labels[0] is not None else None
    return {'video': x, 'label': y, 'path': paths}


def video_collate_dp(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function compatible with Opacus DP training.
    Returns (video_tensor, label_tensor) tuple instead of dict.
    """
    videos = [b['video'] for b in batch]
    labels = [b.get('label', None) for b in batch]
    
    # pad time dimension to max T
    max_t = max(v.shape[0] for v in videos)
    padded = []
    for v in videos:
        if v.shape[0] == max_t:
            padded.append(v)
        else:
            pad = v[-1:].repeat(max_t - v.shape[0], 1, 1, 1)
            padded.append(torch.cat([v, pad], dim=0))
    
    x = torch.stack(padded, dim=0)  # (B, T, C, H, W)

    if all(lbl is None for lbl in labels):
        y = torch.zeros(len(labels), dtype=torch.long)
    else:
        y = torch.stack(
            [
                lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long)
                for lbl in labels
            ],
            dim=0,
        )

    return x, y


def video_collate_opacus(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function specifically designed for Opacus compatibility.
    Returns a tuple of tensors (video_tensor, label_tensor) as Opacus expects.
    """
    try:
        videos = [b['video'] for b in batch]
        labels = [b.get('label', None) for b in batch]
        
        # pad time dimension to max T
        max_t = max(v.shape[0] for v in videos)
        padded = []
        for v in videos:
            if v.shape[0] == max_t:
                padded.append(v)
            else:
                pad = v[-1:].repeat(max_t - v.shape[0], 1, 1, 1)
                padded.append(torch.cat([v, pad], dim=0))
        
        x = torch.stack(padded, dim=0)  # (B, T, C, H, W)

        if all(lbl is None for lbl in labels):
            y = torch.zeros(len(labels), dtype=torch.long)
        else:
            y = torch.stack(
                [
                    lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long)
                    for lbl in labels
                ],
                dim=0,
            )

        return x, y
    except Exception as e:
        print(f"Error in video_collate_opacus: {e}")
        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch)}")
        if batch:
            print(f"First item type: {type(batch[0])}")
            print(f"First item keys: {list(batch[0].keys()) if isinstance(batch[0], dict) else 'Not a dict'}")
        raise
