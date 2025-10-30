import argparse
import torch
from torch import nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RWF2000Dataset, build_video_transform, load_partitions_or_default
from models import build_videomae_model, MLPHead, trainable_parameters_filter, ClassifyLoss
from fl import run_federated
from train.utils import load_config, default_device, make_adamw, set_seed


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = default_device()

    data_cfg = cfg['data']
    transform = build_video_transform(size=data_cfg.get('size', 224))
    full_dataset = RWF2000Dataset(
        root=data_cfg['root'], transform=transform,
        num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
        random_offset=True
    )
    # Build a validation dataset from RWF-2000 val directory
    def _derive_val_root(train_root: str) -> str:
        norm = os.path.normpath(train_root)
        parts = norm.split(os.sep)
        if parts and parts[-1].lower() == 'train':
            parts[-1] = 'val'
            return os.sep.join(parts)
        parent = os.path.dirname(norm)
        return os.path.join(parent, 'val')

    val_root = cfg['data'].get('val_root', _derive_val_root(cfg['data']['root']))
    val_dataset = RWF2000Dataset(
        root=val_root, transform=build_video_transform(size=data_cfg.get('size', 224)),
        num_frames=data_cfg.get('num_frames', 16), frame_stride=data_cfg.get('frame_stride', 4),
        random_offset=False
    )
    fed_cfg = cfg['federated']
    shards = load_partitions_or_default(data_cfg['root'], data_cfg.get('partitions'), fed_cfg['num_clients'])
    clients = {cid: {'dataset': full_dataset, 'indices': idxs} for cid, idxs in shards.items()}

    model_cfg = cfg['model']
    peft_config = model_cfg.get('peft', None)
    # Check if this is full fine-tuning or small head fine-tuning
    is_full_finetune = cfg.get('experiment', '').startswith('full_finetune')
    
    backbone = build_videomae_model(
        model_name=model_cfg.get('model_name'),
        pretrained=model_cfg.get('pretrained', True),
        mode='head' if is_full_finetune else 'feature', 
        num_classes=2,
        peft_config=peft_config,
        head_config=model_cfg.get('head', {}) if is_full_finetune else None,
        pretrained_checkpoint=model_cfg.get('pretrained_checkpoint', None)
    )
    
    # Load pretrained weights (robust to LoRA/recon_prompt/decoder keys)
    if model_cfg.get('pretrained_checkpoint'):
        ckpt = torch.load(model_cfg['pretrained_checkpoint'], map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)

        model_sd = backbone.state_dict()
        filtered = {}
        remapped = 0
        skipped = 0

        def maybe_remap_key(k: str) -> str:
            # Map LoRA base weights to plain weights: *.base.weight -> *.weight
            if k.endswith('.base.weight'):
                return k[:-11] + '.weight'
            if k.endswith('.base.bias'):
                return k[:-9] + '.bias'
            return k

        for k, v in state.items():
            # Skip recon prompt, LoRA A/B, and decoder weights in head/feature finetune
            if ('.A' in k) or ('.B' in k) or ('recon_prompt' in k) or ('.decoder.' in k):
                skipped += 1
                continue
            k2 = maybe_remap_key(k)
            if k2 in model_sd and model_sd[k2].shape == v.shape:
                filtered[k2] = v
                if k2 != k:
                    remapped += 1

        missing_before = len([k for k in model_sd.keys() if k not in filtered])
        print(f"[Init] Loading pretrained (filtered) weights: kept={len(filtered)}, remapped={remapped}, skipped={skipped}, missing(before load)={missing_before}")
        backbone.load_state_dict(filtered, strict=False)
        print(f"Loaded pretrained weights (strict=False) from {model_cfg['pretrained_checkpoint']}")
    
    if is_full_finetune:
        # Full fine-tuning: train all parameters
        print('[Init] Full fine-tuning: training all parameters')
        for p in backbone.parameters():
            p.requires_grad = True
        model = backbone  # backbone already has classifier in head mode
    else:
        # Small head fine-tuning: freeze backbone, train only head
        print('[Init] Small head fine-tuning: training only head')
        if peft_config and peft_config.get('use_lora', False):
            print('[Init] LoRA enabled: training LoRA params + head; base backbone remains frozen')
        else:
            for p in backbone.parameters():
                p.requires_grad = False
        in_dim = backbone._feat_dim()
        head_cfg = model_cfg['head']
        head = MLPHead(
            in_dim,
            head_cfg.get('hidden_dim', 512),
            head_cfg.get('num_classes', 2),
            num_layers=head_cfg.get('num_layers', 1),
            dropout=head_cfg.get('dropout', 0.0),
        )
        model = nn.Sequential(backbone, head)
    model.to(device)

    def make_optimizer(m, lr, weight_decay):
        return make_adamw(trainable_parameters_filter(m), lr=lr, weight_decay=weight_decay)

    # Create loss function with class weights if specified
    loss_config = fed_cfg.get('loss', {})
    class_weights = loss_config.get('class_weights', None)
    if class_weights is not None:
        print(f"Using weighted loss with class weights: {class_weights}")
        loss_fn = ClassifyLoss(class_weights=class_weights, device=device)
    else:
        loss_fn = ClassifyLoss()

    run_federated(
        model=model,
        clients_data=clients,
        make_optimizer=make_optimizer,
        loss_fn=loss_fn,
        rounds=fed_cfg.get('rounds', 10),
        clients_per_round=fed_cfg.get('clients_per_round', fed_cfg.get('num_clients', 1)),
        local_epochs=fed_cfg.get('local_epochs', 1),
        lr=fed_cfg.get('lr', 5e-3),
        weight_decay=fed_cfg.get('weight_decay', 0.01),
        batch_size=data_cfg.get('batch_size', 8),
        num_workers=data_cfg.get('num_workers', 4),
        device=device,
        clip_grad=fed_cfg.get('clip_grad', 0.0),
        log_every=cfg.get('logging', {}).get('log_every', 50),
        out_dir=cfg.get('output_dir', 'runs/finetune_head'),
        use_amp=fed_cfg.get('use_amp', False),
        aggregate_top_k=fed_cfg.get('aggregate_top_k', 0),
        explore_prob=fed_cfg.get('explore_prob', 0.0),
        full_agg_period=fed_cfg.get('full_agg_period', 0),
        score_metric=fed_cfg.get('score_metric', 'acc'),
        score_ema_alpha=fed_cfg.get('score_ema_alpha', 0.6),
        reuse_loaders=fed_cfg.get('dataloader', {}).get('reuse_loaders', False),
        dataloader_persistent_workers=fed_cfg.get('dataloader', {}).get('persistent_workers', False),
        dataloader_prefetch_factor=fed_cfg.get('dataloader', {}).get('prefetch_factor', 2),
        dataloader_pin_memory=fed_cfg.get('dataloader', {}).get('pin_memory', True),
        # validation
        val_dataset=val_dataset,
        val_batch_size=data_cfg.get('batch_size', 8),
        val_num_workers=max(0, data_cfg.get('num_workers', 4) // 2),
        select_best_metric=fed_cfg.get('score_metric', 'acc'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
