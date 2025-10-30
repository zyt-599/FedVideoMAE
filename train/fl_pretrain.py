import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RWF2000Dataset, RLVSDataset, build_video_transform, load_partitions_or_default
from models import build_videomae_model, ReconPrompt, inject_lora, trainable_parameters_filter, PretrainLoss
from fl import run_federated
from train.utils import load_config, default_device, make_adamw, set_seed

# Suppress the noisy full backward hook warning from torch (benign for training)
warnings.filterwarnings(
    "ignore",
    message=r"Full backward hook is firing.*",
    category=UserWarning,
)


def detect_dataset_type(data_root: str) -> str:
    """Detect dataset type based on directory structure."""
    import os
    if os.path.exists(os.path.join(data_root, 'Fight')) and os.path.exists(os.path.join(data_root, 'NonFight')):
        return 'rwf2000'
    elif os.path.exists(os.path.join(data_root, 'Violence')) and os.path.exists(os.path.join(data_root, 'NonViolence')):
        return 'rlvs'
    else:
        raise ValueError(f"Unknown dataset structure in {data_root}. Expected Fight/NonFight or Violence/NonViolence directories.")


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = default_device()

    # data + partitions
    data_cfg = cfg['data']
    fed_cfg = cfg['federated']  # Move this before it's used
    
    # For VideoMAE pretraining, use proper normalization to match the model's expectations
    transform = build_video_transform(size=data_cfg.get('size', 224), normalize=True)
    
    # Auto-detect dataset type and load appropriate dataset
    dataset_type = detect_dataset_type(data_cfg['root'])
    print(f"[Init] Detected dataset type: {dataset_type}")
    
    if dataset_type == 'rwf2000':
        full_dataset = RWF2000Dataset(
            root=data_cfg['root'], transform=transform,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4)
        )
    elif dataset_type == 'rlvs':
        full_dataset = RLVSDataset(
            root=data_cfg['root'], transform=transform,
            num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4)
        )
    
    shards = load_partitions_or_default(data_cfg['root'], data_cfg.get('partitions'), fed_cfg['num_clients'])
    clients = {
        cid: {'dataset': full_dataset, 'indices': idxs}
        for cid, idxs in shards.items()
    }

    # model + peft + prompt
    model_cfg = cfg['model']
    prompt = ReconPrompt(dim=model_cfg.get('prompt', {}).get('prompt_dim', 768),
                         length=model_cfg.get('prompt', {}).get('prompt_len', 8)) if model_cfg.get('prompt', {}).get('use_recon_prompt', False) else None
    model = build_videomae_model(
        model_name=model_cfg.get('model_name'),
        pretrained=model_cfg.get('pretrained', True),
        mode='pretrain',
        num_classes=2,
        recon_prompt=prompt,
        mask_ratio=model_cfg.get('mask_ratio', 0.9),
        peft_config=model_cfg.get('peft', None),
        dp_training=fed_cfg.get('differential_privacy', {}).get('enabled', False)
    )
    # LoRA injection is now handled in build_videomae_model
    model.to(device)

    # optimizer factory uses only trainable (PEFT) params
    def make_optimizer(m, lr, weight_decay):
        return make_adamw(trainable_parameters_filter(m), lr=lr, weight_decay=weight_decay)

    loss_fn = PretrainLoss(mask_ratio=model_cfg.get('mask_ratio', 0.9))

    run_federated(
        model=model,
        clients_data=clients,
        make_optimizer=make_optimizer,
        loss_fn=loss_fn,
        rounds=fed_cfg.get('rounds', 10),
        clients_per_round=fed_cfg.get('clients_per_round', fed_cfg.get('num_clients', 1)),
        local_epochs=fed_cfg.get('local_epochs', 1),
        lr=fed_cfg.get('lr', 1e-3),
        weight_decay=fed_cfg.get('weight_decay', 0.0),
        batch_size=data_cfg.get('batch_size', 4),
        num_workers=data_cfg.get('num_workers', 4),
        device=device,
        clip_grad=fed_cfg.get('clip_grad', 0.0),
        log_every=cfg.get('logging', {}).get('log_every', 50),
        out_dir=cfg.get('output_dir', 'runs/pretrain'),
        experiment_name=cfg.get('experiment', 'FedVideomae_DP'),
        use_amp=fed_cfg.get('use_amp', False),
        # Differential Privacy parameters
        use_dp=fed_cfg.get('differential_privacy', {}).get('enabled', False),
        target_epsilon=fed_cfg.get('differential_privacy', {}).get('target_epsilon', 10.0),
        target_delta=fed_cfg.get('differential_privacy', {}).get('target_delta', 1e-5),
        noise_multiplier=fed_cfg.get('differential_privacy', {}).get('noise_multiplier'),
        max_grad_norm=fed_cfg.get('differential_privacy', {}).get('max_grad_norm', 1.0),
        # Secure Aggregation parameters
        use_secure_agg=fed_cfg.get('secure_aggregation', {}).get('enabled', False),
        secure_agg_threshold=fed_cfg.get('secure_aggregation', {}).get('threshold'),
        # Privacy-preserving aggregation
        server_noise_scale=fed_cfg.get('server_noise_scale', 0.0),
        # Memory controls
        dp_max_physical_batch_size=fed_cfg.get('differential_privacy', {}).get('max_physical_batch_size', None),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
