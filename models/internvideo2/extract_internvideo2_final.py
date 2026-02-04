#!/usr/bin/env python3
"""
InternVideo2 Video Embedding Extraction for Something-Something v2

This script extracts video embeddings using InternVideo2-Stage2-6B model.
This version is based on a working example and uses a more robust video reader.

Usage:
    python extract_internvideo2_final.py --split train --device cpu
    python extract_internvideo2_final.py --split val --device cuda
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import importlib.util
import torchvision.transforms as T
import cv2
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VIDEO_FOLDER = PROJECT_ROOT / "data/sth-sth-v2/s2s clips"
JSON_FOLDER = PROJECT_ROOT / "data/sth-sth-v2"
OUTPUT_FOLDER = PROJECT_ROOT / "embeddings"
CHECKPOINT_DIR = OUTPUT_FOLDER / "checkpoints"

# Checkpoint path
INTERNVIDEO2_CHECKPOINT = PROJECT_ROOT / "models/internvideo2/checkpoints/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt"

# InternVideo2 paths
INTERN_PATH = Path('/scratch/rpp/vlm_image_compositionality/InternVideo/InternVideo2/multi_modality')
INTERN_BACKBONE_PATH = INTERN_PATH / 'models' / 'backbones' / 'internvideo2'

# Create output directories
OUTPUT_FOLDER.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

def setup_logger(split='train'):
    """Setup logging for the extraction process"""
    log_file = OUTPUT_FOLDER / f"internvideo2_{split}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log on new run
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = None

def load_internvideo2_model(device='cpu'):
    """
    Load InternVideo2 model using dynamic imports to avoid relative import issues.
    """
    global logger
    if logger is None:
        logger = setup_logger('default')
    logger.info("Loading InternVideo2 model...")
    logger.info(f"Device: {device}")
    
    sys.path.insert(0, str(INTERN_PATH))
    
    # Dynamically load required modules to bypass relative import errors
    for module_name in ['pos_embed', 'flash_attention_class']:
        module_path = str(INTERN_BACKBONE_PATH / f'{module_name}.py')
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    # Patch and load the main internvideo2 model file
    internvideo2_path = INTERN_BACKBONE_PATH / 'internvideo2.py'
    with open(internvideo2_path, 'r') as f:
        code = f.read().replace('from .', 'from ')
    
    spec = importlib.util.spec_from_loader("internvideo2_module", loader=None)
    internvideo2_module = importlib.util.module_from_spec(spec)
    sys.modules['internvideo2_module'] = internvideo2_module
    exec(code, internvideo2_module.__dict__)
    
    # Simple config based on working example
    # IMPORTANT: use_flash_attn=False to use normal attention (flash_attn not installed)
    class SimpleConfig:
        def __init__(self):
            self.vision_encoder = type('obj', (object,), {
                'clip_embed_dim': 768, 'use_flash_attn': False, 'use_fused_rmsnorm': False,
                'use_fused_mlp': False, 'num_frames': 4, 'tubelet_size': 1,
                'sep_image_video_pos_embed': True, 'use_checkpoint': False, 'checkpoint_num': 0,
                'clip_teacher_embed_dim': 3200, 'clip_teacher_final_dim': 768,
                'clip_norm_type': 'l2', 'clip_return_layer': 1, 'clip_student_return_interval': 1,
                'pretrained': None, 'get': lambda self, key, default: getattr(self, key, default)
            })()
    
    config = SimpleConfig()
    
    # Create model
    model = internvideo2_module.pretrain_internvideo2_6b_patch14_224(config)
    
    # Load checkpoint
    if not INTERNVIDEO2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {INTERNVIDEO2_CHECKPOINT}")
    
    logger.info(f"Loading checkpoint: {INTERNVIDEO2_CHECKPOINT} (28GB)...")
    checkpoint = torch.load(INTERNVIDEO2_CHECKPOINT, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # Extract vision encoder weights
    vision_state = {k.replace('vision_encoder.', ''): v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
    
    missing, unexpected = model.load_state_dict(vision_state, strict=False)
    logger.info(f"✓ Model loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    
    model = model.to(device).eval()
    logger.info(f"✓ InternVideo2 model ready on {device}")
    
    return model

class InternVideo2VideoEncoder:
    """
    InternVideo2 video encoder for embedding extraction.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.model = load_internvideo2_model(device)
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.num_frames = 4
    
    def preprocess_video(self, video_path):
        """Preprocess video using OpenCV (cv2) for robustness."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Cannot open video: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None

            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                elif frames:
                    frames.append(frames[-1]) # Use last valid frame on error
            
            cap.release()

            if not frames:
                return None

            # Ensure we have the correct number of frames
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

            pixel_values = torch.stack([self.transform(frame) for frame in frames])
            return pixel_values
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess video {video_path}: {e}")
            return None
    
    @torch.no_grad()
    def extract_embedding(self, video_path):
        """Extract embedding for a single video."""
        try:
            pixel_values = self.preprocess_video(video_path)
            if pixel_values is None: return None
            
            # Add batch dimension and rearrange: [T, C, H, W] -> [B, C, T, H, W]
            pixel_values = pixel_values.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
            pixel_values = pixel_values.to(self.device)
            
            if self.device == 'cuda':
                pixel_values = pixel_values.to(torch.bfloat16)
            else:
                pixel_values = pixel_values.to(torch.float32)

            # Forward pass through InternVideo2 model (use_image=False for video)
            # Returns: (x_vis, x_pool_vis, x_clip_align, x_align)
            # x_pool_vis is the pooled visual representation [B, clip_embed_dim]
            x_vis, x_pool_vis, x_clip_align, x_align = self.model(pixel_values, mask=None, use_image=False)
            
            # Use x_pool_vis as the video embedding (already pooled by clip_projector)
            pooled_embedding = x_pool_vis.squeeze(0)  # [clip_embed_dim]
            
            embedding = pooled_embedding.cpu().float().numpy()
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract embedding from {video_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

def load_video_list(split='train'):
    """Load list of videos for given split"""
    json_file = JSON_FOLDER / f"{split}_pairs.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return [{'id': str(item['id']), 'path': str(VIDEO_FOLDER / f"{item['id']}.webm")} for item in data]

def extract_embeddings(split='train', device='cuda'):
    """Extract embeddings for all videos in a split."""
    global logger
    logger = setup_logger(split)
    
    logger.info(f"{'='*80}\nInternVideo2 Video Embedding Extraction (Final)\nSplit: {split}, Device: {device}\n{'='*80}")
    
    output_path = OUTPUT_FOLDER / f"internvideo2_{split}.pt"
    checkpoint_path = CHECKPOINT_DIR / f"internvideo2_{split}_checkpoint.pt"
    
    videos = load_video_list(split)
    logger.info(f"Processing: {split} split ({len(videos)} videos)")
    
    embeddings, processed_ids, start_idx = {}, set(), 0
    if checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        embeddings = checkpoint.get('embeddings', {})
        processed_ids = set(checkpoint.get('processed_ids', []))
        start_idx = checkpoint.get('last_index', 0)
        logger.info(f"Resuming from index {start_idx} ({len(embeddings)} embeddings loaded)")
    
    encoder = InternVideo2VideoEncoder(device=device)
    
    errors, missing = 0, 0
    with tqdm(total=len(videos), desc=f"Extracting {split}", initial=start_idx) as pbar:
        for idx in range(start_idx, len(videos)):
            video, video_id = videos[idx], videos[idx]['id']
            if video_id in processed_ids:
                pbar.update(1)
                continue
            
            if not Path(video['path']).exists():
                missing += 1
                pbar.update(1)
                continue
            
            embedding = encoder.extract_embedding(video['path'])
            if embedding is not None:
                embeddings[video_id] = torch.from_numpy(embedding)
                processed_ids.add(video_id)
            else:
                errors += 1
            
            pbar.update(1)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"[{idx + 1}/{len(videos)}] ✓ {len(embeddings)} videos | errors: {errors} | missing: {missing}")
                torch.save({
                    'embeddings': embeddings, 'processed_ids': list(processed_ids), 'last_index': idx + 1,
                }, checkpoint_path)
                logger.info("💾 Checkpoint saved")
    
    logger.info(f"\n{'='*80}\nExtraction complete! Total: {len(embeddings)}, Errors: {errors}, Missing: {missing}\n{'='*80}")
    torch.save(embeddings, output_path)
    logger.info(f"✓ Saved to: {output_path}")
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()

def main():
    parser = argparse.ArgumentParser(description='Extract InternVideo2 video embeddings')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    if args.split == 'all':
        for split in ['train', 'val', 'test']:
            extract_embeddings(split, device)
    else:
        extract_embeddings(args.split, device)

if __name__ == '__main__':
    main()
