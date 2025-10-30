# Federated VideoMAE (pMAE-style) for Self-Supervised Video Classification

This project adapts the pMAE paradigm of "parameter-efficient + reconstructive prompt + server-side reconstruction/fine-tuning" to the video domain, based on VideoMAE for self-supervised pretraining, and performs federated evaluation (linear probing/small head fine-tuning) on RWF-2000. It can be smoothly extended to similar datasets like RLVS, HockeyFight, UCF-Crime, etc.

## Features Overview
- Federated self-supervised pretraining (FedAvg, parameter-efficient: LoRA + reconstructive prompt)
- Client-side: Only train PEFT parameters (LoRA, Prompt, small head), freeze backbone
- Server-side: Aggregate parameters, optional "server-side reconstruction/fine-tuning" for prompt or head alignment
- Evaluation:
  - Linear probing (freeze backbone + federated training of linear head)
  - Small head fine-tuning (freeze backbone + federated fine-tuning of small head with few parameters)
- Dataset: RWF-2000 (extensible to other datasets)

## Dependencies
Recommended Python 3.9

- torch==2.8.0+cu128
- torchvision==0.23.0+cu128
- torchaudio==2.8.0+cu128
- transformers==4.30.2
- timm>=0.9
- decord>=0.6.0
- opencv-python-headless>=4.8
- numpy>=1.21
- scikit-learn>=1.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- pyyaml
- rich
- tqdm
- opacus>=1.4.0
- nvflare==2.6.2

Quick installation:

```bash
pip install -r requirements.txt
```

The `requirements.txt` provides minimum dependencies. If you already have local repositories (like `VideoMAE/`, `pMAE/`), you don't need to install them.

## Data Preparation (RWF-2000)

### 1. Dataset Structure

Organize your RWF-2000 dataset as follows:

```
RWF-2000/
├── train/
│   ├── Fight/
│   │   └── *.avi
│   └── NonFight/
│       └── *.avi
└── val/
    ├── Fight/
    │   └── *.avi
    └── NonFight/
        └── *.avi
```

The dataset contains training and validation splits, with each split containing two categories: Fight (violent behavior) and NonFight (non-violent behavior). Video files are in AVI format.

### 2. Create Client Data Partitions (Optional)

Pre-partition client data (IID/non-IID):

```bash
python -m FedVideomae_DP.data.partition \
  --data_root /your/data/RWF-2000 \
  --out_json partitions/rwf2000_iid_10c.json \
  --num_clients 10 --strategy iid
```

### 3. Update Configuration Files

Update `data.root` and `data.partitions` paths in `configs/*rwf2000.yaml` to point to your actual paths.

## Training and Evaluation

### 1. Federated Self-Supervised Pretraining

```bash
python -m FedVideomae_DP.train.fl_pretrain \
  --config configs/pretrain_rwf2000.yaml
```

### 2. Evaluate Pretrained Model

```bash
python -m FedVideomae_DP.train.evaluate_pretrained \
  --config configs/evaluate_pretrained.yaml \
  --checkpoint /path/to/pretrained/model.pth \
  --output_dir evaluation_results \
  --num_samples 20 \
  --visualize \
  --extract_features
```

**Note**: Replace `/path/to/pretrained/model.pth` with your actual model checkpoint path.

### 3. Linear Probing

```bash
python -m FedVideomae_DP.train.fl_eval_linear \
  --config configs/linear_probe_rwf2000.yaml
```

### 4. Small Head Fine-tuning

```bash
python -m FedVideomae_DP.train.fl_eval_tunehead \
  --config configs/finetune_head_rwf2000.yaml
```

### 5. Feature Analysis

```bash
python -m FedVideomae_DP.train.analyze_features \
  --feature_path evaluation_results/features.pth \
  --output_dir feature_analysis
```

## Key Concepts
- **Parameter Efficiency (PEFT)**: Insert LoRA into attention/MLP linear layers, train only small parameters, reducing communication/privacy leakage risks.
- **Reconstructive Prompt**: Inject learnable prompt tokens during self-supervised reconstruction to assist masked video block reconstruction.
- **Server-side Reconstruction/Fine-tuning**: Server maintains (aggregated) prompts/small heads, optionally performs reconstruction/consistency fine-tuning on small public buffer to align cross-client feature spaces.

## Code Structure
```
FedVideomae_DP/
  configs/          # Configuration files
  data/            # Data loading and preprocessing
  fl/              # Federated learning components
  models/          # Model definitions and wrappers
  train/           # Training and evaluation scripts
  scripts/         # Utility scripts
```

## Environment Check

To verify your environment setup:

```bash
python scripts/check_env.py
```

This will print versions of key libraries and basic introspection of the Transformers VideoMAE API.

## Notes
- By default uses HuggingFace `VideoMAEForPreTraining` as backbone; will automatically download weights on first run (can set `model_name`).
- If running environment cannot connect to internet or you don't want automatic downloads, set `model_name: null` in config and provide local weight path, or only use feature mode (avoid pretraining).
- Initial version is a runnable baseline skeleton, easy to extend as needed (e.g., RLVS, HockeyFight, UCF-Crime).

## Configuration Files

- `configs/pretrain_rwf2000.yaml` - Federated pretraining configuration
- `configs/linear_probe_rwf2000.yaml` - Linear probing configuration  
- `configs/finetune_head_rwf2000.yaml` - Small head fine-tuning configuration
- `configs/evaluate_pretrained.yaml` - Pretrained model evaluation configuration

## Output Files

After running evaluation scripts, you'll find:
- `evaluation_results/` - Reconstruction quality metrics and visualizations
- `feature_analysis/` - Feature distribution analysis and visualizations
- `runs/` - Training logs and model checkpoints