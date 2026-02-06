#!/scratch/rpp/vlm_image_compositionality/bin/python
"""
DiST Video Embedding Extraction
Paper: "Disentangling Spatial and Temporal Learning for Efficient Image-to-Video Transfer Learning"

Architecture (per config vit-b16-32+64f.yaml):
- Spatial Encoder: Frozen CLIP ViT-B/16 (processes 32 sparse frames)
- Temporal Encoder: Lightweight 3D Conv (processes 64 dense frames, 96 channels)
- Integration Branch: Fuses spatial-temporal features (384 channels, 12 layers)
"""

import os
import sys
import json
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import decord
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add DiST paths
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
import clip
from models.module_zoo.branches.dist import DiSTNetwork
from easydict import EasyDict as edict

# ============================================================
# Configuration
# ============================================================
# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
VIDEO_FOLDER = PROJECT_ROOT / "data/sth-sth-v2/s2s clips"
JSON_FOLDER = PROJECT_ROOT / "data/sth-sth-v2"
CHECKPOINT_PATH = _script_dir / "checkpoints/dist_vit-b16_32+64f_ssv2_70.9.pth"
OUTPUT_FOLDER = PROJECT_ROOT / "embeddings"

# Architecture parameters from config
NUM_SPATIAL_FRAMES = 32  # Sparse sampling for spatial encoder
NUM_TEMPORAL_FRAMES = 64  # Dense sampling for temporal encoder
SPARSE_SAMPLE_ALPHA = 2  # 64/32 = 2
S_PATCH_SIZE = 16
T_PATCH_SIZE = 5
TEMPORAL_DIM = 96  # β = 1/8 of 768
INTEGRATION_DIM = 384  # α = 1/2 of 768
SELECTED_LAYERS = list(range(12))  # All 12 transformer layers
ADA_POOLING_LAYERS = 2

CHECKPOINT_EVERY = 100
LOG_EVERY = 10

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "checkpoints"), exist_ok=True)

# ============================================================
# Logging
# ============================================================
def setup_logger(split='val'):
    """Setup logging for extraction."""
    log_file = os.path.join(OUTPUT_FOLDER, f"dist_{split}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

print("=" * 80)
print(" " * 25 + "DiST Video Embedding Extraction")
print("=" * 80)
print(f"Architecture: Dual-encoder (Spatial + Temporal + Integration)")
print(f"Spatial Encoder: Frozen CLIP ViT-B/16 ({NUM_SPATIAL_FRAMES} sparse frames)")
print(f"Temporal Encoder: 3D Conv ({NUM_TEMPORAL_FRAMES} dense frames, {TEMPORAL_DIM}D)")
print(f"Integration Branch: {INTEGRATION_DIM}D across {len(SELECTED_LAYERS)} layers")
print(f"Output: 512-dim embeddings")
print("=" * 80)

# ============================================================
# DiST Model
# ============================================================
class DiSTVideoEncoder(nn.Module):
    """Complete DiST architecture for video encoding."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        # Load frozen CLIP (spatial encoder)
        logger.info("Loading CLIP ViT-B/16 (Spatial Encoder)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        
        # CRITICAL: Convert CLIP to FP32 (OpenAI CLIP uses FP16 by default)
        self.clip_model = self.clip_model.float()
        for param in self.clip_model.parameters():
            param.data = param.data.float()
            param.requires_grad = False
        
        self.clip_model.eval()
        logger.info("✓ CLIP loaded and frozen (FP32)")
        
        # Build config for DiST
        cfg = edict({
            'DATA': {
                'NUM_INPUT_FRAMES': NUM_TEMPORAL_FRAMES,
                'SPARSE_SAMPLE_ALPHA': SPARSE_SAMPLE_ALPHA
            },
            'VIDEO': {
                'BACKBONE': {
                    'DIST': {
                        'S_PATCH_SIZE': S_PATCH_SIZE,
                        'T_PATCH_SIZE': T_PATCH_SIZE,
                        'TEMPORAL_DIM': TEMPORAL_DIM,
                        'INTEGRATION_DIM': INTEGRATION_DIM,
                        'SELECTED_LAYERS': SELECTED_LAYERS,
                        'ADA_POOLING_LAYERS': ADA_POOLING_LAYERS,
                        'TEMPORAL_KERNEL_SIZE': 3,
                        'TEMPORAL_CONV_MLP_RATIO': 1,
                        'TEMPORAL_USE_LAYER_NORM': True,
                        'INTEGRATION_MLP_RATIO': 1,
                        'INTEGRATION_TEMPORAL_MLP_RATIO': 0.25,
                        'SPATIAL_CLS_TOKEN_INIT_AS_ZERO': False
                    }
                }
            }
        })
        
        # Initialize DiST temporal + integration networks
        logger.info("Initializing DiST networks (Temporal + Integration)...")
        d_model = 768  # CLIP ViT-B/16 hidden dim
        output_dim = 512  # CLIP projection dim
        self.dist_network = DiSTNetwork(cfg, d_model, d_model, output_dim)
        
        # Load pretrained checkpoint
        if CHECKPOINT_PATH.exists():
            logger.info(f"Loading pretrained DiST checkpoint: {CHECKPOINT_PATH}")
            checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
            state_dict = checkpoint['model_state']
            
            # Extract DiST network weights (remove 'backbone.base_encoder.dist_net.' prefix)
            dist_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.base_encoder.dist_net.'):
                    new_key = k.replace('backbone.base_encoder.dist_net.', '')
                    dist_state[new_key] = v
            
            # Load weights
            missing, unexpected = self.dist_network.load_state_dict(dist_state, strict=False)
            logger.info(f"✓ Loaded pretrained DiST weights (70.9% SSV2 accuracy)")
            if missing:
                logger.warning(f"   Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"   Unexpected keys: {len(unexpected)}")
        else:
            logger.warning(f"No checkpoint found at {CHECKPOINT_PATH}, using random initialization")
        
        self.dist_network = self.dist_network.to(device)
        self.dist_network = self.dist_network.float()
        for param in self.dist_network.parameters():
            param.data = param.data.float()
            param.requires_grad = False  # Freeze DiST for embedding extraction
        
        self.dist_network.eval()
        
        # Verify dtypes
        logger.info(f"   CLIP dtype: {next(self.clip_model.parameters()).dtype}")
        logger.info(f"   DiST dtype: {next(self.dist_network.parameters()).dtype}")
        logger.info(f"✓ Model ready on {device}")
    
    def extract_clip_features(self, frames):
        """
        Extract intermediate features from CLIP visual encoder.
        Args:
            frames: Tensor (B, 3, H, W) - batch of frames
        Returns:
            List of 12 tensors (one per layer) with shape (L, B, D)
        """
        x = self.clip_model.visual.conv1(frames)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid**2)
        x = x.permute(0, 2, 1)  # (B, grid**2, width)
        
        # Add class embedding
        x = torch.cat([
            self.clip_model.visual.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)
        
        # Add positional embedding
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Collect intermediate features from all 12 layers
        intermediate_features = []
        for layer in self.clip_model.visual.transformer.resblocks:
            x = layer(x)
            intermediate_features.append(x)
        
        return intermediate_features
    
    def forward(self, spatial_frames, temporal_frames):
        """
        Forward pass through complete DiST architecture.
        Args:
            spatial_frames: Tensor (32, 3, 224, 224) - sparse frames for spatial encoder
            temporal_frames: Tensor (64, 3, 224, 224) - dense frames for temporal encoder
        Returns:
            embedding: Tensor (512,) - video-level embedding
        """
        with torch.no_grad():
            # Step 1: Extract spatial features from CLIP (frozen)
            spatial_features = self.extract_clip_features(spatial_frames)
            
            # Step 2: Prepare input for DiST
            input_dict = {
                'images': temporal_frames,  # Dense frames for temporal encoder
                'mid_feat': {'img': spatial_features}  # Spatial features for integration
            }
            
            # Step 3: Process through DiST (temporal + integration)
            embedding, _ = self.dist_network(input_dict)
            
            # Step 4: Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        return embedding.squeeze(0)

# ============================================================
# Video Loading
# ============================================================
def load_video_frames(video_path, num_frames, resize=(224, 224)):
    """Load uniformly sampled frames from video."""
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        try:
            frames = vr.get_batch(indices).asnumpy()
        except:
            frames = []
            for idx in indices:
                try:
                    frame = vr[idx].asnumpy()
                    frames.append(frame)
                except:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))
            frames = np.stack(frames, axis=0)
        
        # Resize if needed
        if frames.shape[1:3] != resize:
            resized_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame)
                pil_frame = pil_frame.resize((resize[1], resize[0]), Image.BILINEAR)
                resized_frames.append(np.array(pil_frame))
            frames = np.stack(resized_frames, axis=0)
        
        return frames
        
    except Exception as e:
        logger.error(f"Failed to load {video_path}: {str(e)}")
        return None

def preprocess_frames(frames, preprocess_fn):
    """Preprocess frames using CLIP preprocessing."""
    processed = []
    for frame in frames:
        frame_pil = Image.fromarray(frame)
        frame_tensor = preprocess_fn(frame_pil)
        processed.append(frame_tensor)
    return torch.stack(processed)

# ============================================================
# Main Extraction
# ============================================================
def extract_embeddings(split='train', device='cuda'):
    """Extract embeddings for specified split."""
    global logger
    logger = setup_logger(split)
    
    json_path = os.path.join(JSON_FOLDER, f"{split}_pairs.json")
    output_file = os.path.join(OUTPUT_FOLDER, f"dist_{split}.pt")
    checkpoint_file = os.path.join(OUTPUT_FOLDER, "checkpoints", f"dist_{split}_checkpoint.pt")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"\nProcessing: {split} split ({len(data)} videos)")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        embeddings_dict = checkpoint['embeddings']
        processed_ids = checkpoint['processed_ids']
        start_idx = checkpoint['last_index'] + 1
        errors = checkpoint.get('errors', 0)
        missing = checkpoint.get('missing', 0)
        logger.info(f"✓ Resuming from video {start_idx}/{len(data)}")
    else:
        embeddings_dict = {}
        processed_ids = set()
        start_idx = 0
        errors = 0
        missing = 0
        logger.info("Starting from beginning")
    
    # Initialize model
    model = DiSTVideoEncoder(device=device)
    successful = len(processed_ids)
    
    # Process videos
    for idx in tqdm(range(start_idx, len(data)), desc=f"Extracting {split}"):
        record = data[idx]
        video_id = str(record['id'])
        
        if video_id in processed_ids:
            continue
        
        video_path = os.path.join(VIDEO_FOLDER, f"{video_id}.webm")
        if not os.path.exists(video_path):
            missing += 1
            continue
        
        # Load frames (spatial: 32 sparse, temporal: 64 dense)
        frames_spatial = load_video_frames(video_path, NUM_SPATIAL_FRAMES, (224, 224))
        frames_temporal = load_video_frames(video_path, NUM_TEMPORAL_FRAMES, (224, 224))
        
        if frames_spatial is None or frames_temporal is None:
            errors += 1
            continue
        
        try:
            # Preprocess
            spatial_input = preprocess_frames(frames_spatial, model.clip_preprocess).to(device)
            temporal_input = preprocess_frames(frames_temporal, model.clip_preprocess).to(device)
            
            # Extract embedding
            embedding = model(spatial_input, temporal_input)
            
            # Store
            embeddings_dict[video_id] = embedding.cpu()
            processed_ids.add(video_id)
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed {video_id}: {str(e)}")
            errors += 1
            continue
        
        # Log progress
        if (idx + 1) % LOG_EVERY == 0:
            logger.info(f"[{idx+1}/{len(data)}] ✓ {successful} videos | errors: {errors} | missing: {missing}")
        
        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            checkpoint = {
                'embeddings': embeddings_dict,
                'processed_ids': processed_ids,
                'last_index': idx,
                'errors': errors,
                'missing': missing,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, checkpoint_file)
            logger.info(f"💾 Checkpoint saved")
    
    # Final save
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Completed {split}: {successful}/{len(data)} videos")
    logger.info(f"Errors: {errors} | Missing: {missing}")
    logger.info("=" * 80)
    
    torch.save(embeddings_dict, output_file)
    logger.info(f"💾 Saved to {output_file}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return successful, errors, missing

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract DiST video embeddings')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test', 'all'],
                        help='Which split to extract (default: val)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
        torch.set_num_threads(4)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print(" " * 25 + "DiST Video Embedding Extraction")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Output: {OUTPUT_FOLDER}")
    print("=" * 80)
    
    # Determine splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    # Process each split
    results = {}
    for split in splits:
        try:
            successful, errors, missing = extract_embeddings(split, device)
            results[split] = {'successful': successful, 'errors': errors, 'missing': missing}
        except Exception as e:
            logger.error(f"Failed to process {split}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    for split, res in results.items():
        print(f"{split:10s}: {res['successful']:5d} extracted | "
              f"{res['errors']:3d} errors | {res['missing']:3d} missing")
    print("=" * 80)
