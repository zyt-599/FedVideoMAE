from typing import Dict, Optional, Tuple, List
import os
import torch
import torch.nn as nn
import numpy as np


class PretrainLoss(nn.Module):
    def __init__(self, mask_ratio: float = 0.9):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.criterion = nn.MSELoss()

    def forward(self, model: nn.Module, video: torch.Tensor, _: Optional[torch.Tensor] = None):
        # model expected to return (recon, target, mask, optional_loss)
        out = model(video, return_mask=True)
        if isinstance(out, (list, tuple)) and len(out) == 4:
            recon, target, mask, opt_loss = out
            if opt_loss is not None:
                return opt_loss, {'custom_loss': opt_loss.item()}
        else:
            recon, target, mask = out
            opt_loss = None

        # If target not provided, fall back to L2 on reconstruction against itself (stop-grad)
        if target is None:
            target = recon.detach()

        # For official VideoMAE, recon only contains masked tokens, so no need to filter with mask again
        # recon: [B, N_mask, C], target: [B, N_mask, C] or None
        if mask is not None:
            # Calculate MSE loss directly without sigmoid normalization
            # VideoMAE reconstruction target should be original pixel values, no additional normalization needed
            recon_loss = ((recon - target) ** 2).mean()
            
            # Reduce L2 regularization strength to avoid over-constraining the model
            l2_reg = 0.001 * recon.pow(2).mean()
            total_loss = recon_loss + l2_reg
            
            # Get monitoring metrics
            metrics = model.get_metrics(recon, target)
            metrics.update({
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'l2_reg': l2_reg.item(),
            })
            
            return total_loss, metrics
            
        loss = self.criterion(recon, target)
        metrics = {
            'loss': loss.item(),
            'pixel_mse': ((recon - target) ** 2).mean().item(),
            'pixel_mae': (recon - target).abs().mean().item()
        }
        return loss, metrics


class ClassifyLoss(nn.Module):
    def __init__(self, class_weights=None, device=None):
        super().__init__()
        if class_weights is not None:
            # Convert to tensor if it's a list
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            # Move weights to the specified device
            if device is not None:
                class_weights = class_weights.to(device)
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, model: nn.Module, video: torch.Tensor, y: Optional[torch.Tensor]):
        logits = model(video)
        loss = self.ce(logits, y)
        
        # Calculate classification metrics
        metrics = {
            'loss': loss.item(),
        }
        if y is not None:
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y).float().mean().item()
            metrics['acc'] = acc
            
            # Store predictions and labels for detailed metrics calculation
            if not hasattr(self, 'all_preds'):
                self.all_preds = []
                self.all_labels = []
                self.all_probs = []
            
            self.all_preds.extend(pred.detach().cpu().numpy())
            self.all_labels.extend(y.detach().cpu().numpy())
            self.all_probs.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
            
        return loss, metrics
    
    def get_detailed_metrics(self):
        """Calculate detailed classification metrics"""
        if not hasattr(self, 'all_preds') or len(self.all_preds) == 0:
            return {}
        
        from sklearn.metrics import (
            precision_recall_fscore_support, confusion_matrix, 
            roc_auc_score, average_precision_score, classification_report
        )
        
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        all_probs = np.array(self.all_probs)
        
        # Basic metrics
        accuracy = (all_preds == all_labels).mean()
        
        # Precision, Recall, F1
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC-AUC and Average Precision
        try:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            avg_precision = average_precision_score(all_labels, all_probs[:, 1])
        except:
            roc_auc = 0.0
            avg_precision = 0.0
        
        # Classification report
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        
        detailed_metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Reset stored predictions
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
        return detailed_metrics


class VideoMAEWrapper(nn.Module):
    """
    A thin wrapper over official VideoMAE for pretraining (reconstruction) and feature/classification mode.
    Input: (B, T, C, H, W) normalized with ImageNet statistics.
    """
    def get_metrics(self, recon_tokens: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate reconstruction metrics.
        
        Args:
            recon_tokens: Reconstructed tokens with shape [B, N_mask, C]
            target_tokens: Target tokens (used in pretrain mode)

        Returns:
            Dictionary containing metrics: pixel_mse, pixel_mae, pixel_rmse, etc.
        """
        metrics = {}
        if self.mode == 'pretrain':
            with torch.no_grad():  # Ensure metric calculation doesn't affect gradients
                if target_tokens is not None:
                    # Calculate reconstruction error metrics
                    pixel_mse = ((recon_tokens - target_tokens) ** 2).mean().item()
                    pixel_mae = (recon_tokens - target_tokens).abs().mean().item()
                    pixel_rmse = pixel_mse ** 0.5
                    pixel_max = recon_tokens.max().item()
                    pixel_min = recon_tokens.min().item()
                else:
                    # If no target, calculate reconstruction value statistics
                    pixel_mse = recon_tokens.pow(2).mean().item()
                    pixel_mae = recon_tokens.abs().mean().item()
                    pixel_rmse = pixel_mse ** 0.5
                    pixel_max = recon_tokens.max().item()
                    pixel_min = recon_tokens.min().item()

                metrics.update({
                    'pixel_mse': pixel_mse,
                    'pixel_mae': pixel_mae,
                    'pixel_rmse': pixel_rmse,
                    'pixel_max': pixel_max,
                    'pixel_min': pixel_min,
                })

                # Add token-level metrics
                avg_token_norm = recon_tokens.norm(dim=-1).mean().item()
                max_token_norm = recon_tokens.norm(dim=-1).max().item()
                metrics.update({
                    'token_norm': avg_token_norm,
                    'max_token_norm': max_token_norm,
                })

        return metrics

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch tokens from raw video input.

        Args:
            x: Video input with shape [B, C, T, H, W]
            
        Returns:
            Patch tokens with shape [B, N, patch_dim]
        """
        # Use official patch_embed layer
        patch_tokens = self.backbone.encoder.patch_embed(x)  # [B, N, C]
        return patch_tokens

    def __init__(
        self,
        model_name: Optional[str] = 'MCG-NJU/videomae-base',
        pretrained: bool = True,
        mode: str = 'pretrain',  # 'pretrain' | 'feature' | 'head'
        num_classes: int = 2,
        recon_prompt: Optional[nn.Module] = None,
        mask_ratio: float = 0.9,
        head_config: Optional[dict] = None,
        pretrained_checkpoint: Optional[str] = None,
        num_frames: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.recon_prompt = recon_prompt
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes
        self.num_frames = int(num_frames) if num_frames is not None else 16

        self.backbone = None
        self.decoder = None
        self.classifier = None

        self._init_hf(model_name, pretrained, pretrained_checkpoint)
        if self.mode in ('head', 'feature'):
            if self.mode == 'feature':
                self.classifier = nn.Identity()
            else:  # head mode
                # Use head_config if provided, otherwise use default feat_dim
                if head_config and 'hidden_dim' in head_config:
                    # Build MLP head with specified hidden_dim
                    hidden_dim = head_config['hidden_dim']
                    num_layers = head_config.get('num_layers', 2)
                    dropout = head_config.get('dropout', 0.1)
                    
                    layers = []
                    in_dim = self._feat_dim()
                    for i in range(num_layers):
                        if i == 0:
                            layers.append(nn.Linear(in_dim, hidden_dim))
                        else:
                            layers.append(nn.Linear(hidden_dim, hidden_dim))
                        layers.append(nn.GELU())
                        if dropout > 0:
                            layers.append(nn.Dropout(dropout))
                    
                    # Final classification layer
                    layers.append(nn.Linear(hidden_dim, num_classes))
                    self.classifier = nn.Sequential(*layers)
                else:
                    # Fallback to simple linear classifier
                    in_dim = self._feat_dim()
                    self.classifier = nn.Linear(in_dim, num_classes)

    def _init_hf(self, model_name: Optional[str], pretrained: bool, pretrained_checkpoint: Optional[str] = None):
        # Prefer local DP project's VideoMAE implementation; fallback to federated_videomae if not present
        try:
            from videomae_official.modeling_pretrain import (
                pretrain_videomae_small_patch16_224,
                pretrain_videomae_base_patch16_224,
                pretrain_videomae_large_patch16_224,
            )
            from videomae_official.modeling_finetune import (
                vit_small_patch16_224,
                vit_base_patch16_224,
                vit_large_patch16_224,
            )
        except Exception:
            from federated_videomae.videomae_official.modeling_pretrain import (
                pretrain_videomae_small_patch16_224,
                pretrain_videomae_base_patch16_224,
                pretrain_videomae_large_patch16_224,
            )
            from federated_videomae.videomae_official.modeling_finetune import (
                vit_small_patch16_224,
                vit_base_patch16_224,
                vit_large_patch16_224,
            )

        # Decide architecture (small/base/large) & tubelet_size by peeking checkpoint shapes if available
        ckpt_path = pretrained_checkpoint
        arch = 'base'
        tubelet = 2
        embed_dim = 768
        all_frames = self.num_frames if self.num_frames is not None else 16
        ckpt_mode = 'pretrain'  # 'pretrain' or 'finetune'
        if ckpt_path is not None:
            try:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_all = raw.get('model', raw)
                # find patch_embed.proj.weight, normalize key prefix
                def _strip(k: str) -> str:
                    parts = k.split('.')
                    while parts and (parts[0].isdigit() or parts[0] in ('module','bb','backbone','encoder')):
                        parts = parts[1:]
                    return '.'.join(parts)
                pe_key = None
                for k in list(state_all.keys()):
                    sk = _strip(k)
                    if sk == 'patch_embed.proj.weight':
                        pe_key = k
                        break
                if pe_key and isinstance(state_all[pe_key], torch.Tensor):
                    w = state_all[pe_key]
                    # weight shape: [embed_dim, in_chans, tubelet, p, p]
                    if w.ndim == 5:
                        embed_dim = int(w.shape[0])
                        tubelet = int(w.shape[2])
                        if embed_dim <= 384:
                            arch = 'small'
                            embed_dim = 384
                        elif embed_dim >= 1024:
                            arch = 'large'
                            embed_dim = 1024
                        else:
                            arch = 'base'
                            embed_dim = 768
                # infer frames from pos_embed length if present (assume 224/16 -> 14x14 spatial)
                pos_len = None
                for k in list(state_all.keys()):
                    sk = _strip(k)
                    if sk == 'pos_embed':
                        v = state_all[k]
                        if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
                            pos_len = int(v.shape[1])
                            break
                if pos_len is not None:
                    spatial = 14 * 14
                    if pos_len % spatial == 0:
                        T_src = pos_len // spatial
                        ckpt_frames = max(1, T_src * tubelet)
                        if self.num_frames is None:
                            all_frames = ckpt_frames
                # detect finetune vs pretrain style by presence of decoder vs head
                names_stripped = set(_strip(k) for k in state_all.keys())
                has_encoder_tokens = any('encoder.' in _strip(k) for k in state_all.keys())
                has_decoder = any(s.startswith('decoder.') or s in ('encoder_to_decoder.weight','mask_token') for s in names_stripped)
                has_head = any(s.startswith('head.') for s in names_stripped)
                if has_encoder_tokens:
                    ckpt_mode = 'pretrain'
                else:
                    ckpt_mode = 'pretrain' if has_decoder else ('finetune' if has_head else 'pretrain')
            except Exception:
                pass

        # Heuristic override: if using long clips (>=32 frames), increase temporal tubelet to keep token count similar
        if all_frames >= 32:
            tubelet = 4

        # Helper: construct model passing only supported kwargs
        def _construct_with_supported_args(ctor, required_kwargs: dict, optional_kwargs: dict):
            """Call ctor with required kwargs and any optional kwargs that are supported.
            If the callable accepts **kwargs, pass all optional kwargs through.
            """
            try:
                import inspect
                sig = inspect.signature(ctor)
                call_kwargs = dict(required_kwargs)
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if has_var_kw:
                    call_kwargs.update(optional_kwargs)
                else:
                    for k, v in optional_kwargs.items():
                        if k in sig.parameters:
                            call_kwargs[k] = v
                return ctor(**call_kwargs)
            except Exception:
                base = dict(required_kwargs)
                return ctor(**base)

        # Construct backbone with inferred arch & tubelet_size & frames (no auto-loading)
        if ckpt_mode == 'finetune':
            ctor = vit_small_patch16_224 if arch == 'small' else (vit_large_patch16_224 if arch == 'large' else vit_base_patch16_224)
            self.backbone = _construct_with_supported_args(
                ctor,
                required_kwargs={'pretrained': False},
                optional_kwargs={'all_frames': all_frames, 'tubelet_size': tubelet, 'num_frames': all_frames}
            )
        else:
            ctor = pretrain_videomae_small_patch16_224 if arch == 'small' else (pretrain_videomae_large_patch16_224 if arch == 'large' else pretrain_videomae_base_patch16_224)
            self.backbone = _construct_with_supported_args(
                ctor,
                required_kwargs={'pretrained': False, 'init_ckpt': None},
                optional_kwargs={'tubelet_size': tubelet, 'all_frames': all_frames, 'num_frames': all_frames}
            )

        # Only load our provided pretrained checkpoint now
        if ckpt_path is None:
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get('model', ckpt)

            # Build a shape-aware, encoder-only filtered dict to avoid missing/unexpected
            # target state dict (encoder for pretrain backbone, whole for finetune backbone)
            if hasattr(self.backbone, 'encoder'):
                target_sd = self.backbone.encoder.state_dict()  # type: ignore[attr-defined]
                target_prefix = ''
            else:
                target_sd = self.backbone.state_dict()
                target_prefix = ''

            def normalize_key_to_target(k: str) -> Optional[str]:
                key = k
                # strip leading numeric or 'module' prefixes like '0.' or 'module.'
                parts = key.split('.')
                while parts and (parts[0].isdigit() or parts[0] in ('module','bb')):
                    parts = parts[1:]
                key = '.'.join(parts)
                # remove wrapper/backbone prefix if present
                if key.startswith('backbone.'):
                    key = key[len('backbone.'):]
                # also strip 'bb.' (BackboneHead saves)
                if key.startswith('bb.'):
                    key = key[len('bb.'):]
                # In some checkpoints, encoder params are prefixed with 'encoder.' (pretrain ckpt)
                if key.startswith('encoder.'):
                    key = key[len('encoder.'):]
                # Skip prompt / LoRA low-rank matrices
                if any(s in key for s in ('recon_prompt', '.A', '.B')):
                    return None
                # For pretrain-target, drop decoder; for finetune-target, drop head
                if hasattr(self.backbone, 'encoder'):
                    if key.startswith('decoder.') or '.decoder.' in key or key in ('encoder_to_decoder.weight','mask_token'):
                        return None
                else:
                    if key.startswith('head.'):
                        return None
                # Map LoRA base weights to plain
                if '.base.' in key:
                    key = key.replace('.base.', '.')
                if key.endswith('.base.weight'):
                    key = key[:-11] + '.weight'
                elif key.endswith('.base.bias'):
                    key = key[:-9] + '.bias'
                # Map potential *orig naming to plain
                if key.endswith('.weight_orig'):
                    key = key[:-12] + '.weight'
                if key.endswith('.bias_orig'):
                    key = key[:-10] + '.bias'
                # Normalize alternative MLP naming
                key = key.replace('.mlp.linear1.', '.mlp.fc1.').replace('.mlp.linear2.', '.mlp.fc2.')
                key = key.replace('.mlp.fc.0.', '.mlp.fc1.').replace('.mlp.fc.1.', '.mlp.fc2.')
                return key

            enc_filtered = {}
            for k, v in state.items():
                nk = normalize_key_to_target(k)
                if nk is None:
                    continue
                # only keep keys that exactly exist in encoder and match shape
                if nk in target_sd and isinstance(v, torch.Tensor) and v.shape == target_sd[nk].shape:
                    enc_filtered[nk] = v

            # Special-case adaptation: patch_embed.proj.weight temporal kernel size mismatch -> temporal interpolate
            try:
                want_key = 'patch_embed.proj.weight'
                # find raw key for patch_embed in ckpt
                src_key = None
                for k in state.keys():
                    nk = normalize_key_to_target(k) or ''
                    if nk == want_key:
                        src_key = k
                        break
                if src_key and want_key in target_sd and isinstance(state[src_key], torch.Tensor):
                    src = state[src_key]
                    dst = target_sd[want_key]
                    if src.ndim == 5 and dst.ndim == 5 and src.shape[:2] == dst.shape[:2] and src.shape[3:] == dst.shape[3:] and src.shape[2] != dst.shape[2]:
                        # interpolate along temporal kernel dim
                        t_src, t_dst = src.shape[2], dst.shape[2]
                        if t_src == 1:
                            adapted = src.repeat(1,1,t_dst,1,1) / t_dst
                        else:
                            adapted = torch.nn.functional.interpolate(src.permute(2,0,1,3,4).float(), size=t_dst, mode='linear', align_corners=True).permute(1,2,0,3,4).to(dst.dtype)
                        enc_filtered[want_key] = adapted
            except Exception:
                pass

            # Special-case adaptation: positional embedding length mismatch -> temporal interpolate by frames
            try:
                pe_key = 'pos_embed'
                # locate src key for pos_embed
                src_key = None
                for k in state.keys():
                    nk = normalize_key_to_target(k) or ''
                    if nk == pe_key:
                        src_key = k
                        break
                if src_key and pe_key in target_sd and isinstance(state[src_key], torch.Tensor):
                    src = state[src_key]
                    dst = target_sd[pe_key]
                    if src.ndim == 3 and dst.ndim == 3 and src.shape[0] == 1 and dst.shape[0] == 1 and src.shape[2] == dst.shape[2] and src.shape[1] != dst.shape[1]:
                        N_src = src.shape[1]; N_dst = dst.shape[1]
                        # assume 224/16 => 14x14 spatial grid
                        spatial = 14 * 14
                        if N_src % spatial == 0 and N_dst % spatial == 0:
                            T_src = N_src // spatial
                            T_dst = N_dst // spatial
                            src_t = src.view(1, T_src, spatial, -1).permute(1,0,2,3)  # [T_src, 1, spatial, dim]
                            src_t = src_t.reshape(T_src, 1, spatial * src.shape[2])  # [T_src, 1, spatial*dim]
                            # interpolate along T
                            interp = torch.nn.functional.interpolate(src_t.float(), size=T_dst, mode='linear', align_corners=True)
                            interp = interp.reshape(T_dst, 1, spatial, src.shape[2]).permute(1,0,2,3).contiguous().view(1, N_dst, src.shape[2]).to(dst.dtype)
                            enc_filtered[pe_key] = interp
            except Exception:
                pass

            # Helper: hash a mapping of tensors deterministically
            def _hash_tensor_map(tensors_map: dict) -> str:
                import hashlib as _hl
                h = _hl.sha256()
                for _k in sorted(tensors_map.keys()):
                    _t = tensors_map[_k]
                    if not isinstance(_t, torch.Tensor):
                        continue
                    _tb = _t.detach().cpu().contiguous()
                    try:
                        h.update(_k.encode('utf-8'))
                        h.update(_tb.numpy().tobytes(order='C'))
                    except Exception:
                        h.update(_k.encode('utf-8'))
                        h.update(_tb.flatten().to(torch.float32).cpu().numpy().tobytes())
                return h.hexdigest()

            # Merge into a full encoder dict to make strict loading clean (no missing/unexpected)
            new_sd = target_sd.copy()
            new_sd.update(enc_filtered)
            # Compute source-subset hash (normalized/adapted tensors we will load)
            try:
                src_subset_hash = _hash_tensor_map(enc_filtered)
            except Exception:
                src_subset_hash = 'na'
            # load back to encoder or whole backbone
            if hasattr(self.backbone, 'encoder'):
                self.backbone.encoder.load_state_dict(new_sd, strict=True)  # type: ignore[attr-defined]
                tgt_tag = 'encoder'
            else:
                self.backbone.load_state_dict(new_sd, strict=True)
                tgt_tag = 'backbone'
            total_enc_keys = len(target_sd)
            kept = len(enc_filtered)
            pct = (kept / total_enc_keys * 100.0) if total_enc_keys else 0.0
            print(f"[VideoMAEWrapper] Loaded {tgt_tag} weights from {ckpt_path} | kept={kept}/{total_enc_keys} ({pct:.1f}%) missing=0 unexpected=0")
            # Compute full target encoder hash after loading
            try:
                if hasattr(self.backbone, 'encoder'):
                    tgt_sd_all = self.backbone.encoder.state_dict()  # type: ignore[attr-defined]
                else:
                    tgt_sd_all = self.backbone.state_dict()
                target_full_hash = _hash_tensor_map(tgt_sd_all)
                print(f"[VideoMAEWrapper][hash] src_subset={src_subset_hash[:16]} -> target_full={target_full_hash[:16]}")
            except Exception:
                pass

            # Optional diagnostics
            if kept < total_enc_keys:
                src_map = {}
                for k, v in state.items():
                    nk = normalize_key_to_target(k)
                    if nk is not None and isinstance(v, torch.Tensor):
                        src_map[nk] = v.shape
                missing_in_ckpt = []
                shape_mismatch = []
                for k, v in target_sd.items():
                    if k in enc_filtered:
                        continue
                    if k not in src_map:
                        missing_in_ckpt.append(k)
                    else:
                        if tuple(src_map[k]) != tuple(v.shape):
                            shape_mismatch.append((k, tuple(src_map[k]), tuple(v.shape)))
                if missing_in_ckpt:
                    print(f"[VideoMAEWrapper][diag] not found in ckpt: {len(missing_in_ckpt)} e.g. {missing_in_ckpt[:5]}")
                if shape_mismatch:
                    print(f"[VideoMAEWrapper][diag] shape mismatch: {len(shape_mismatch)} e.g. {shape_mismatch[:5]}")
        except Exception as e:
            print(f"[VideoMAEWrapper] Warning: failed to load pretrained checkpoint {ckpt_path}: {e}")

    def _feat_dim(self) -> int:
        # Get model hidden dimension
        # Support both pretrain backbone (has encoder) and finetune backbone (no encoder)
        if hasattr(self.backbone, 'encoder'):
            return getattr(self.backbone.encoder, 'embed_dim', 768)
        return getattr(self.backbone, 'embed_dim', 768)

    def _decoder_dim(self) -> int:
        # Decoder output dimension is fixed: 3 * patch_size * patch_size
        patch_size = getattr(self.backbone.encoder.patch_embed, 'patch_size', [16])[0]
        return 3 * patch_size * patch_size  # Usually 1536 (3 * 16 * 16)

    def _make_bool_mask(self, batch_size: int) -> torch.Tensor:
        """Generate boolean mask for VideoMAE masking.

        Args:
            batch_size: Current batch size

        Returns:
            Boolean mask with shape [B, num_patches], True indicates masked position
        """
        encoder = self.backbone.encoder
        num_patches = encoder.patch_embed.num_patches
        device = next(encoder.parameters()).device
        
        # Calculate number of tokens to mask
        num_mask = int(self.mask_ratio * num_patches)
        
        # Use argsort to generate mask (more efficient than topk)
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Generate mask: first num_mask positions are True (indicating masked)
        mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        # x: (B, T, C, H, W)
        if self.mode == 'pretrain':
            encoder = self.backbone.encoder
            img_size = getattr(encoder.patch_embed, 'img_size', [224])[0]
            patch_size = getattr(encoder.patch_embed, 'patch_size', [16])[0]
            tubelet = getattr(encoder.patch_embed, 'tubelet_size', 2)

            if x.dim() != 5:
                raise ValueError('Expected 5D video tensor')

            # Input is (B, T, C, H, W), need to convert to (B, C, T, H, W)
            x_in = x.permute(0, 2, 1, 3, 4)
            batch_size = x_in.shape[0]

            # Ensure 3-channel input (dim=1 after permute)
            c = x_in.shape[1]
            if c == 1:
                x_in = x_in.repeat(1, 3, 1, 1, 1)
            elif c == 4:
                x_in = x_in[:, :3]
            elif c != 3:
                raise ValueError(f'Unsupported channel count {c}; expected 1/3/4')

            # Generate mask using current batch actual size
            bool_mask = self._make_bool_mask(batch_size)
            # Note: backbone only reconstructs masked parts
            recon = self.backbone(x_in, bool_mask)  # Returns [B, N_mask, 3 * patch_size^2]

            # Convert mask to float and add dimension for broadcasting
            mask = bool_mask.float().unsqueeze(-1)

            if return_mask:
                # Get pixel-level targets for masked positions in original video
                with torch.no_grad():
                    patch_size = getattr(self.backbone.encoder.patch_embed, 'patch_size', [16])[0]
                    tubelet_size = getattr(self.backbone.encoder.patch_embed, 'tubelet_size', 2)
                    num_patches = self.backbone.encoder.patch_embed.num_patches

                    # Generate targets matching decoder output dimension
                    # Decoder output dimension is 3 * patch_size * patch_size = 1536
                    decoder_dim = 3 * patch_size * patch_size
                    
                    # Convert input video to patches and flatten to pixel-level targets
                    B, C, T, H, W = x_in.shape
                    # Reshape to patches: [B, T//tubelet, H//P, W//P, C, tubelet, P, P]
                    x_patch = x_in.view(B, C, T//tubelet_size, tubelet_size, 
                                      H//patch_size, patch_size, 
                                      W//patch_size, patch_size)
                    x_patch = x_patch.permute(0, 2, 4, 6, 1, 3, 5, 7)
                    
                    # Flatten each patch to target vector: [B, N, C * tubelet * P * P]
                    target = x_patch.reshape(B, -1, C * tubelet_size * patch_size * patch_size)
                    
                    # Select masked patches as targets
                    target = target[bool_mask].reshape(batch_size, -1, target.shape[-1])

                # Apply reconstruction prompt if needed
                if self.recon_prompt is not None:
                    recon = self.recon_prompt(recon)

                # Return reconstruction results, targets and mask
                return (recon, target, mask, None)
            
            # If mask not needed, only return reconstruction results
            if self.recon_prompt is not None:
                recon = self.recon_prompt(recon)
            return recon

        # feature/head mode
        if x.dim() != 5:
            raise ValueError('Expected 5D video tensor')
        # Input is (B, T, C, H, W), need to convert to (B, C, T, H, W)
        x_in = x.permute(0, 2, 1, 3, 4)
        # Ensure 3-channel input (dim=1 after permute)
        c = x_in.shape[1]
        if c == 1:
            x_in = x_in.repeat(1, 3, 1, 1, 1)
        elif c == 4:
            x_in = x_in[:, :3]
        elif c != 3:
            raise ValueError(f'Unsupported channel count {c}; expected 1/3/4')

        # If backbone is a finetune VisionTransformer (no encoder attr), use forward_features directly
        if not hasattr(self.backbone, 'encoder'):
            pooled = self.backbone.forward_features(x_in)
            return pooled if self.classifier is None or isinstance(self.classifier, nn.Identity) else self.classifier(pooled)

        # In feature mode, call encoder with all-False mask (no tokens masked)
        B = x_in.shape[0]
        N = self.backbone.encoder.patch_embed.num_patches
        mask = torch.zeros(B, N, dtype=torch.bool, device=x_in.device)  # All False, no tokens masked
        features = self.backbone.encoder(x_in, mask)  # [B, N, C]
        pooled = features.mean(1)  # Average over sequence dimension [B, C]

        # Return features or classification results
        return pooled if self.classifier is None or isinstance(self.classifier, nn.Identity) else self.classifier(pooled)


def build_videomae_model(
    model_name: Optional[str], pretrained: bool, mode: str, num_classes: int,
    recon_prompt: Optional[nn.Module] = None, mask_ratio: float = 0.9,
    peft_config: Optional[dict] = None, head_config: Optional[dict] = None,
    pretrained_checkpoint: Optional[str] = None,
    num_frames: Optional[int] = None,
    dp_training: bool = False,
) -> nn.Module:
    # Create Reconstructive Prompt only for pretrain mode (decoder space)
    # In feature/head modes, recon_prompt is unnecessary and may cause dim warnings
    if recon_prompt is None and peft_config and peft_config.get('use_lora', False) and mode == 'pretrain':
        from models.recon_prompt import ReconPrompt
        # Default to encoder embed dim prompt; internal module can adapt as needed
        recon_prompt = ReconPrompt(dim=768, length=8)
    
    model = VideoMAEWrapper(
        model_name=model_name, pretrained=pretrained, mode=mode, num_classes=num_classes,
        recon_prompt=recon_prompt, mask_ratio=mask_ratio, head_config=head_config,
        pretrained_checkpoint=pretrained_checkpoint,
        num_frames=num_frames,
    )
    
    # Apply PEFT if configured
    if peft_config and peft_config.get('use_lora', False):
        from models.peft_lora import inject_lora, try_load_lora_from_checkpoint
        inject_lora(
            model=model,
            r=peft_config.get('lora_r', 4),
            alpha=peft_config.get('lora_alpha', 8),
            dropout=peft_config.get('lora_dropout', 0.0),
            target_modules=peft_config.get('target_modules', None),
            dp_training=bool(dp_training),
        )
        # If a pretrained checkpoint is provided, try to load LoRA A/B weights
        # so that downstream tasks benefit from PEFT pretraining.
        try_load_lora_from_checkpoint(model, pretrained_checkpoint)
    
    return model
