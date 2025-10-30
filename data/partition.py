import json
import os
import random
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.rwf2000 import _scan_videos
    from data.rlvs import _scan_videos as _scan_videos_rlvs
except ImportError:
    from rwf2000 import _scan_videos
    from rlvs import _scan_videos as _scan_videos_rlvs


def _iid_split(items: List[Tuple[str, int]], num_clients: int) -> Dict[str, List[int]]:
    """IID split with 1-based client ids ('1'..'num_clients')."""
    idxs = list(range(len(items)))
    random.shuffle(idxs)
    shards = {str(i + 1): [] for i in range(num_clients)}
    for i, idx in enumerate(idxs):
        shards[str((i % num_clients) + 1)].append(idx)
    return shards


def _by_label_split(items: List[Tuple[str, int]], num_clients: int) -> Dict[str, List[int]]:
    by_label: Dict[int, List[int]] = {}
    for i, (_, y) in enumerate(items):
        by_label.setdefault(y, []).append(i)
    for v in by_label.values():
        random.shuffle(v)
    # 1-based client ids
    shards = {str(i + 1): [] for i in range(num_clients)}
    # alternate labels to create mild non-iid
    i = 0
    while any(by_label.values()):
        for y in sorted(by_label.keys()):
            if by_label[y]:
                shards[str((i % num_clients) + 1)].append(by_label[y].pop())
                i += 1
    return shards


def save_partitions(out_json: str, shards: Dict[str, List[int]]):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(shards, f)


def load_partitions(path: str) -> Dict[str, List[int]]:
    with open(path, 'r') as f:
        return json.load(f)


def detect_dataset_type(data_root: str) -> str:
    """Detect dataset type based on directory structure."""
    if os.path.exists(os.path.join(data_root, 'Fight')) and os.path.exists(os.path.join(data_root, 'NonFight')):
        return 'rwf2000'
    elif os.path.exists(os.path.join(data_root, 'Violence')) and os.path.exists(os.path.join(data_root, 'NonViolence')):
        return 'rlvs'
    else:
        raise ValueError(f"Unknown dataset structure in {data_root}. Expected Fight/NonFight or Violence/NonViolence directories.")


def load_partitions_or_default(data_root: str, partitions: Optional[str], num_clients: int) -> Dict[str, List[int]]:
    dataset_type = detect_dataset_type(data_root)
    if dataset_type == 'rwf2000':
        items = _scan_videos(data_root)
    elif dataset_type == 'rlvs':
        items = _scan_videos_rlvs(data_root)
    
    if partitions is None:
        return _iid_split(items, num_clients)
    return load_partitions(partitions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='iid', choices=['iid', 'by_label'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    # Auto-detect dataset type
    dataset_type = detect_dataset_type(args.data_root)
    print(f"Detected dataset type: {dataset_type}")
    
    if dataset_type == 'rwf2000':
        items = _scan_videos(args.data_root)
    elif dataset_type == 'rlvs':
        items = _scan_videos_rlvs(args.data_root)
    
    print(f"Found {len(items)} videos")
    
    if args.strategy == 'iid':
        shards = _iid_split(items, args.num_clients)
    else:
        shards = _by_label_split(items, args.num_clients)
    save_partitions(args.out_json, shards)
    print(f'Saved partitions to {args.out_json}')
