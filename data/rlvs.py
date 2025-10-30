import os
import glob
import random
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge('torch')
except Exception:
    decord = None

try:
    from torchvision.io import read_video
except Exception:
    read_video = None


CLASS_TO_LABEL = {
    'NonViolence': 0,
    'Violence': 1,
}


def _scan_videos(root: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for cls, y in CLASS_TO_LABEL.items():
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            for p in glob.glob(os.path.join(cls_dir, ext)):
                items.append((p, y))
    return items


def _decode_video(path: str) -> torch.Tensor:
    """Return tensor T x H x W x C in uint8."""
    # Try decord first (faster)
    if decord is not None:
        try:
            vr = decord.VideoReader(path)
            frames = vr.get_batch(range(len(vr)))  # (T, H, W, C) torch.uint8
            return frames
        except Exception as e:
            print(f"Warning: decord failed for {path}: {e}, falling back to OpenCV")
    
    # Try torchvision.io as second option
    if read_video is not None:
        try:
            frames, _, _ = read_video(path, pts_unit='sec')  # (T, H, W, C) uint8
            return frames
        except Exception as e:
            print(f"Warning: torchvision.io failed for {path}: {e}, falling back to OpenCV")
    
    # Fallback to OpenCV
    import cv2
    import os
    
    # Suppress FFmpeg warnings
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suppress all FFmpeg logs
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {path} with any backend")
        return torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        print(f"Warning: No frames read from {path}")
        return torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    
    # Convert to tensor: (T, H, W, C)
    frames_tensor = torch.from_numpy(np.stack(frames))
    return frames_tensor


def _sample_indices(num_frames_total: int, num_frames: int, stride: int, random_offset: bool = False) -> List[int]:
    if num_frames_total <= 0:
        return []
    step = max(1, stride)
    start = 0
    if random_offset and num_frames_total > 0:
        # Offset within one step to avoid bias; ensure start < num_frames_total
        start = random.randint(0, min(step - 1, max(0, num_frames_total - 1)))
    idx = list(range(start, num_frames_total, step))
    if len(idx) >= num_frames:
        return idx[:num_frames]
    # pad by repeating last
    if not idx:
        idx = [0]
    while len(idx) < num_frames:
        idx.append(idx[-1])
    return idx


class RLVSDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        num_frames: int = 16,
        frame_stride: int = 4,
        index_list: Optional[List[Tuple[str, int]]] = None,
        random_offset: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.random_offset = random_offset
        self.items: List[Tuple[str, int]] = index_list if index_list is not None else _scan_videos(root)
        if len(self.items) == 0:
            raise RuntimeError(f'No videos found under {root} (expected Violence/ and NonViolence/)')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.items[idx]
        # Prefer selective decoding with decord when available
        if decord is not None:
            try:
                vr = decord.VideoReader(path)
                t = len(vr)
                inds = _sample_indices(t, self.num_frames, self.frame_stride, self.random_offset)
                frames = vr.get_batch(inds)
            except Exception:
                # fallback to full decode then sample
                frames = _decode_video(path)
                t = frames.shape[0]
                inds = _sample_indices(t, self.num_frames, self.frame_stride, self.random_offset)
                frames = frames[inds]
        else:
            frames = _decode_video(path)
            t = frames.shape[0]
            inds = _sample_indices(t, self.num_frames, self.frame_stride, self.random_offset)
            frames = frames[inds]
        sample = {
            'video': frames,  # uint8
            'label': torch.tensor(label, dtype=torch.long),
            'path': path,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
