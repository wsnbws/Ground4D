"""
Unified Visualization Tool for DGGT Models

Supports multiple visualization modes:
- dggt: Base DGGT model with temporal decay
- voxel: Voxel-based model with temporal fusion

Usage:
    python tools/visualize.py --mode dggt --image_dir /path/to/images --ckpt_path /path/to/ckpt
    python tools/visualize.py --mode voxel --image_dir /path/to/images --ckpt_path /path/to/ckpt
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ground4d.models.vggt import VGGT
from ground4d.utils.pose_enc import pose_encoding_to_extri_intri
from ground4d.utils.geometry import unproject_depth_map_to_point_map
from ground4d.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from datasets.orad_dataset import OradDataset

from tools.configs.base_config import BaseVisualizationConfig
from tools.configs.dggt_config import DGGTVisualizationConfig
from tools.configs.voxel_config import VoxelVisualizationConfig


# ============================================================================
# Common Utilities
# ============================================================================

class PoseInterpolator:
    """Handles camera pose interpolation."""
    
    @staticmethod
    def interpolate(
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        interval: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linearly interpolate camera poses."""
        S = extrinsics.shape[0]
        new_S = (S - 1) * interval + 1
        device = extrinsics.device
        
        new_extrinsics = torch.zeros((new_S, 4, 4), device=device)
        new_intrinsics = torch.zeros((new_S, 3, 3), device=device)
        
        new_extrinsics[::interval] = extrinsics
        new_intrinsics[::interval] = intrinsics
        
        for i in range(S - 1):
            start_idx = i * interval
            for step in range(1, interval):
                alpha = step / interval
                curr_idx = start_idx + step
                new_extrinsics[curr_idx] = (1 - alpha) * extrinsics[i] + alpha * extrinsics[i + 1]
                new_intrinsics[curr_idx] = (1 - alpha) * intrinsics[i] + alpha * intrinsics[i + 1]
        
        return new_extrinsics, new_intrinsics
    
    @staticmethod
    def get_interpolated_timestamp(idx: int, interval: int, timestamps: torch.Tensor) -> torch.Tensor:
        """Get timestamp for interpolated frame."""
        progress = idx / interval
        base_idx = min(int(progress), len(timestamps) - 2)
        local_progress = progress - base_idx
        return timestamps[base_idx] * (1 - local_progress) + timestamps[base_idx + 1] * local_progress
    
    @staticmethod
    def get_nearest_frame_idx(idx: int, interval: int, max_idx: int) -> int:
        """Get nearest original frame index for interpolated frame."""
        progress = idx / interval
        nearest_idx = int(round(progress))
        return min(nearest_idx, max_idx)


class ImageUtils:
    """Utilities for image processing and visualization."""
    
    @staticmethod
    def add_labels_to_image(
        image: Image.Image,
        labels: List[str],
        panel_width: int,
        font_size: int = 16,
        text_color: str = "white",
        bg_color: str = "black",
        padding: int = 5
    ) -> Image.Image:
        """Add text labels on top of each panel in the image."""
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        
        for i, label in enumerate(labels):
            x_start = i * panel_width
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text in the panel
            x = x_start + (panel_width - text_width) // 2
            y = padding
            
            # Draw background rectangle
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=bg_color
            )
            
            # Draw text
            draw.text((x, y), label, fill=text_color, font=font)
        
        return image
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor [C, H, W] or [H, W, C] to PIL Image."""
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3, 4]:  # [C, H, W]
                return T.ToPILImage()(tensor.clamp(0, 1))
            else:  # [H, W, C]
                return T.ToPILImage()(tensor.permute(2, 0, 1).clamp(0, 1))
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


class ModelLoader:
    """Handles model checkpoint loading."""
    
    @staticmethod
    def load_dggt_checkpoint(ckpt_path: str, device: torch.device) -> VGGT:
        """Load DGGT model from checkpoint."""
        model = VGGT().to(device)
        
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        
        # Remove 'module.' prefix if present (DDP models)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model
    
    @staticmethod
    def load_voxel_checkpoint(ckpt_path: str, config: 'VoxelVisualizationConfig', device: torch.device):
        """Load Voxel model from checkpoint."""
        from ground4d.voxelize_v2 import GaussianVoxelizerV2, TemporalVoxelFusion
        
        model = VGGT().to(device)
        fusion_module = TemporalVoxelFusion(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        fusion_module.load_state_dict(checkpoint['fusion_module'], strict=False)
        
        model.eval()
        fusion_module.eval()
        
        voxelizer = GaussianVoxelizerV2(voxel_size=config.voxel_size)
        
        return model, fusion_module, voxelizer


# ============================================================================
# Base Visualizer
# ============================================================================

class BaseVisualizer(ABC):
    """Abstract base class for visualizers."""
    
    def __init__(self, config: BaseVisualizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.interpolator = PoseInterpolator()
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Load dataset
        self.dataset = OradDataset(config.image_dir, sequence_length=config.sequence_length)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _prepare_batch(self, sample_idx: int) -> dict:
        """Load and prepare a batch from dataset."""
        batch = self.dataset[sample_idx]
        
        for key in batch:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            if isinstance(batch[key], torch.Tensor):
                if batch[key].dtype == torch.float64:
                    batch[key] = batch[key].float()
                batch[key] = batch[key].unsqueeze(0)
        
        return batch
    
    def _process_geometry(self, predictions: dict, images: torch.Tensor) -> dict:
        """Process model predictions into geometric representations."""
        H, W = images.shape[-2:]
        
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
        extrinsic = extrinsics[0]
        
        # Add homogeneous row
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device)
        bottom = bottom.view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
        extrinsic = torch.cat([extrinsic, bottom], dim=1)
        
        intrinsic = intrinsics[0]
        
        # Unproject depth to 3D points
        depth_map = predictions["depth"][0]
        point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])
        point_map = torch.from_numpy(point_map[None, ...]).to(self.device).float()
        
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'point_map': point_map,
            'H': H,
            'W': W
        }
    
    def _save_frame(
        self,
        combined: torch.Tensor,
        sample_dir: Path,
        frame_idx: int,
        labels: List[str],
        panel_width: int
    ):
        """Save a single frame with optional labels."""
        # Convert to PIL
        if combined.shape[0] in [1, 3, 4]:  # [C, H, W]
            pil_image = T.ToPILImage()(combined.clamp(0, 1))
        else:  # [H, W, C]
            pil_image = T.ToPILImage()(combined.permute(2, 0, 1).clamp(0, 1))
        
        # Add labels if enabled
        if self.config.add_labels:
            pil_image = ImageUtils.add_labels_to_image(
                pil_image,
                labels,
                panel_width,
                font_size=self.config.label_font_size,
                text_color=self.config.label_color,
                bg_color=self.config.label_bg_color
            )
        
        # Save
        pil_image.save(sample_dir / f"frame_{frame_idx:02d}.png")
    
    @abstractmethod
    def _visualize_sample(self, batch: dict, sample_idx: int):
        """Visualize a single sample. Must be implemented by subclasses."""
        pass
    
    @torch.no_grad()
    def run(self):
        """Execute visualization pipeline."""
        print(f"Mode: {self.config.mode}")
        print(f"Loading dataset from {self.config.image_dir}...")
        print(f"Output directory: {self.config.output_dir}")
        print(f"Start index: {self.config.start_index}")
        print(f"Rendering {self.config.num_samples} sample(s)...\n")
        
        end_index = self.config.start_index + min(self.config.num_samples, len(self.dataset))
        
        for sample_idx in tqdm(range(self.config.start_index, end_index), desc="Visualizing"):
            batch = self._prepare_batch(sample_idx)
            self._visualize_sample(batch, sample_idx)
        
        print(f"\nDone! Results saved to {self.config.output_dir}")


# ============================================================================
# DGGT Visualizer
# ============================================================================

class DGGTVisualizer(BaseVisualizer):
    """Visualizer for DGGT base model."""
    
    def __init__(self, config: DGGTVisualizationConfig):
        super().__init__(config)
        self.model = ModelLoader.load_dggt_checkpoint(config.ckpt_path, self.device)
    
    @staticmethod
    def compute_temporal_decay(
        timestamps: torch.Tensor,
        t0: torch.Tensor,
        opacity: torch.Tensor,
        confidence: torch.Tensor,
        gamma0: float = 1.0,
        gamma1: float = 0.1
    ) -> torch.Tensor:
        """Compute temporal decay for static Gaussians."""
        sigma = torch.log(torch.tensor(gamma1, device=confidence.device)) / (gamma0**2 + 1e-6)
        decay = torch.exp(sigma * (t0 - timestamps)**2)
        return opacity * decay
    
    def _split_gaussians(
        self,
        point_map: torch.Tensor,
        gs_map: torch.Tensor,
        dy_map: torch.Tensor,
        gs_conf: torch.Tensor,
        bg_mask: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[List, torch.Tensor]:
        """Split Gaussians into static and dynamic components."""
        static_mask = torch.ones_like(bg_mask)
        static_points = point_map[static_mask].reshape(-1, 3)
        
        dynamic_probs = dy_map[static_mask].sigmoid()
        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
        static_opacity = static_opacity * (1 - dynamic_probs)
        static_gs_conf = gs_conf[static_mask]
        
        frame_indices = torch.nonzero(static_mask, as_tuple=False)[:, 1]
        gs_timestamps = timestamps[frame_indices]
        
        static_gs_list = [static_points, static_rgbs, static_opacity, static_scales, static_rotations, static_gs_conf]
        return static_gs_list, gs_timestamps
    
    def _extract_dynamic_gaussians(
        self,
        point_map: torch.Tensor,
        gs_map: torch.Tensor,
        dy_map: torch.Tensor,
        bg_mask: torch.Tensor
    ) -> List[Tuple]:
        """Extract dynamic Gaussians for each frame."""
        dynamic_gaussians = []
        
        for i in range(dy_map.shape[1]):
            points = point_map[:, i][bg_mask[:, i]].reshape(-1, 3)
            rgbs, opacity, scales, rotations = get_split_gs(gs_map[:, i], bg_mask[:, i])
            
            dynamic_probs = dy_map[:, i][bg_mask[:, i]].sigmoid()
            opacity = opacity * dynamic_probs
            
            dynamic_gaussians.append((points, rgbs, opacity, scales, rotations))
        
        return dynamic_gaussians
    
    def _render_frame(
        self,
        static_gs_list: List,
        dynamic_gaussians: Tuple,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render a single frame."""
        static_points, static_rgbs, static_opacity, static_scales, static_rotations = static_gs_list[:5]
        
        if dynamic_gaussians:
            points, rgbs, opacity, scales, rotations = concat_list(
                [static_points, static_rgbs, static_opacity, static_scales, static_rotations],
                list(dynamic_gaussians)
            )
        else:
            points, rgbs, opacity, scales, rotations = static_gs_list[:5]
        
        render, alpha, _ = rasterization(
            means=points,
            quats=rotations,
            scales=scales,
            opacities=opacity,
            colors=rgbs,
            viewmats=extrinsic[None],
            Ks=intrinsic[None],
            width=W,
            height=H
        )
        
        return render[0], alpha[0]
    
    def _visualize_sample(self, batch: dict, sample_idx: int):
        """Visualize a single sample."""
        images = batch['images'].to(self.device)
        sky_mask = batch['masks'].to(self.device).permute(0, 1, 3, 4, 2)
        bg_mask = (sky_mask == 0).any(dim=-1)
        timestamps = batch['timestamps'][0].to(self.device)
        
        # Forward pass
        predictions = self.model(images)
        geometry = self._process_geometry(predictions, images)
        H, W = geometry['H'], geometry['W']
        
        # Prepare poses
        extrinsic = geometry['extrinsic']
        intrinsic = geometry['intrinsic']
        original_extrinsic, original_intrinsic = extrinsic.clone(), intrinsic.clone()
        
        if self.config.interp_interval > 0:
            extrinsic, intrinsic = self.interpolator.interpolate(
                extrinsic, intrinsic, self.config.interp_interval
            )
            print(f"  Interpolated: {original_extrinsic.shape[0]} -> {extrinsic.shape[0]} frames")
        
        # Split Gaussians
        static_gs_list, gs_timestamps = self._split_gaussians(
            geometry['point_map'],
            predictions['gs_map'],
            predictions['dynamic_conf'].squeeze(-1),
            predictions['gs_conf'],
            bg_mask,
            timestamps
        )
        
        dynamic_gaussians = self._extract_dynamic_gaussians(
            geometry['point_map'],
            predictions['gs_map'],
            predictions['dynamic_conf'].squeeze(-1),
            bg_mask
        )
        
        # Create sample directory
        sample_dir = self.config.output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Render all frames
        n_frames = extrinsic.shape[0]
        
        for idx in range(n_frames):
            # Determine current time and dynamic frame
            if self.config.interp_interval > 0:
                t0 = self.interpolator.get_interpolated_timestamp(idx, self.config.interp_interval, timestamps)
                dyn_idx = self.interpolator.get_nearest_frame_idx(idx, self.config.interp_interval, len(timestamps) - 1)
            else:
                t0 = timestamps[idx]
                dyn_idx = idx
            
            # Apply temporal decay
            if self.config.disable_lifespan:
                static_opacity = static_gs_list[2]
            else:
                static_opacity = self.compute_temporal_decay(
                    gs_timestamps, t0, static_gs_list[2], static_gs_list[5]
                )
            
            current_static_gs = static_gs_list[:2] + [static_opacity] + static_gs_list[3:5]
            
            # Render frame
            render, alpha = self._render_frame(
                current_static_gs,
                dynamic_gaussians[dyn_idx] if dynamic_gaussians else None,
                extrinsic[idx],
                intrinsic[idx],
                H, W
            )
            
            # Sky background
            if self.config.interp_interval > 0:
                bg_render = self.model.sky_model.forward_with_new_pose(
                    images, original_extrinsic, original_intrinsic, extrinsic[idx:idx+1], intrinsic[idx:idx+1]
                )
            else:
                bg_render = self.model.sky_model(images[:, idx:idx+1], extrinsic[idx:idx+1], intrinsic[idx:idx+1])
            
            final_render = alpha * render + (1 - alpha) * bg_render[0]
            
            # Prepare visualizations
            render_vis = final_render.cpu().clamp(0, 1)
            alpha_vis = alpha.repeat(1, 1, 3).cpu().clamp(0, 1)
            # dynamic_conf shape: [1, S, H, W, 1] or [1, S, H, W] -> squeeze and handle
            dyn_conf = predictions['dynamic_conf'][0, dyn_idx].squeeze()  # [H, W]
            dynamic_vis = dyn_conf.sigmoid().unsqueeze(-1).repeat(1, 1, 3).cpu().clamp(0, 1)
            
            # Combine visualizations
            is_keyframe = (self.config.interp_interval == 0) or (idx % self.config.interp_interval == 0)
            keyframe_idx = idx // self.config.interp_interval if self.config.interp_interval > 0 else idx
            
            if is_keyframe and keyframe_idx < images.shape[1]:
                gt = images[0, keyframe_idx].permute(1, 2, 0).cpu().clamp(0, 1)
                combined = torch.cat([gt, render_vis, alpha_vis, dynamic_vis], dim=1)
                labels = self.config.panel_labels
            else:
                combined = torch.cat([render_vis, alpha_vis, dynamic_vis], dim=1)
                labels = self.config.interp_panel_labels
            
            self._save_frame(combined, sample_dir, idx, labels, W)
        
        print(f"  Sample {sample_idx}: Saved {n_frames} frames to {sample_dir}")


# ============================================================================
# Voxel Visualizer
# ============================================================================

class VoxelVisualizer(BaseVisualizer):
    """Visualizer for Voxel-based model with temporal fusion."""
    
    def __init__(self, config: VoxelVisualizationConfig):
        super().__init__(config)
        self.model, self.fusion_module, self.voxelizer = ModelLoader.load_voxel_checkpoint(
            config.ckpt_path, config, self.device
        )
    
    def _extract_gaussian_features(
        self,
        point_map: torch.Tensor,
        gs_map: torch.Tensor,
        dy_map: torch.Tensor,
        timestamps: torch.Tensor,
        bg_mask: torch.Tensor
    ):
        """Extract Gaussian features for voxelization."""
        static_mask = torch.ones_like(bg_mask)
        static_points = point_map[static_mask].reshape(-1, 3)
        
        gs_dynamic_list = dy_map[static_mask].sigmoid()
        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
        static_opacity = static_opacity * (1 - gs_dynamic_list)
        
        frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
        gs_timestamps = timestamps[frame_idx].float()
        
        gaussian_features = torch.cat([
            static_points,
            static_rgbs,
            static_opacity if static_opacity.dim() == 2 else static_opacity.unsqueeze(-1),
            static_scales,
            static_rotations,
            gs_timestamps.unsqueeze(-1) if gs_timestamps.dim() == 1 else gs_timestamps
        ], dim=-1)
        
        return gaussian_features, static_points, frame_idx
    
    @staticmethod
    def _unpack_gaussian_features(fused_gaussians: torch.Tensor):
        """Unpack fused Gaussians."""
        points = fused_gaussians[:, 0:3]
        rgbs = fused_gaussians[:, 3:6]
        opacity = fused_gaussians[:, 6:7].squeeze(-1)
        scales = fused_gaussians[:, 7:10]
        rotations = fused_gaussians[:, 10:14]
        return points, rgbs, opacity, scales, rotations
    
    def _render_frame(
        self,
        fused_static_gs: Tuple,
        dynamic_gs: Optional[Tuple],
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render a frame."""
        if dynamic_gs is not None:
            world_points, rgbs, opacity, scales, rotation = concat_list(
                list(fused_static_gs),
                list(dynamic_gs)
            )
        else:
            world_points, rgbs, opacity, scales, rotation = fused_static_gs
        
        renders, alphas, _ = rasterization(
            means=world_points,
            quats=rotation,
            scales=scales,
            opacities=opacity,
            colors=rgbs,
            viewmats=extrinsic[None],
            Ks=intrinsic[None],
            width=W,
            height=H,
        )
        
        return renders, alphas
    
    def _visualize_sample(self, batch: dict, sample_idx: int):
        """Visualize a single sample with temporal fusion."""
        images = batch['images'].to(self.device)
        sky_mask = batch['masks'].to(self.device).permute(0, 1, 3, 4, 2)
        bg_mask = (sky_mask == 0).any(dim=-1)
        timestamps = batch['timestamps'][0].to(self.device)
        
        # Forward pass
        predictions = self.model(images)
        geometry = self._process_geometry(predictions, images)
        H, W = geometry['H'], geometry['W']
        
        gs_map = predictions["gs_map"]
        dy_map = predictions["dynamic_conf"].squeeze(-1)
        
        # Extract features and voxelize
        gaussian_features, static_points, frame_indices = self._extract_gaussian_features(
            geometry['point_map'], gs_map, dy_map, timestamps, bg_mask
        )
        
        voxel_indices, num_voxels, voxel_counts = self.voxelizer.voxelize(
            static_points, gaussian_features
        )
        
        print(f"  Sample {sample_idx}: {static_points.shape[0]} Gaussians -> {num_voxels} Voxels "
              f"(compression: {static_points.shape[0]/num_voxels:.1f}x)")
        
        # Prepare dynamic Gaussians
        dynamic_gs_per_frame = []
        for i in range(dy_map.shape[1]):
            point_map_i = geometry['point_map'][:, i]
            bg_mask_i = bg_mask[:, i]
            if bg_mask_i.sum() > 0:
                dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(
                    gs_map[:, i], bg_mask_i
                )
                gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                dynamic_gs_per_frame.append(
                    (dynamic_point, dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation)
                )
            else:
                dynamic_gs_per_frame.append(None)
        
        # Create sample directory
        sample_dir = self.config.output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original parameters
        extrinsic = geometry['extrinsic']
        intrinsic = geometry['intrinsic']
        original_extrinsic = extrinsic.clone()
        original_intrinsic = intrinsic.clone()
        original_timestamps = timestamps
        n_original_frames = len(timestamps)
        
        # Interpolate if needed
        interp_interval = max(1, self.config.interp_interval)
        if self.config.interp_interval > 1:
            extrinsic, intrinsic = self.interpolator.interpolate(extrinsic, intrinsic, interp_interval)
            print(f"  Interpolated poses: {n_original_frames} -> {extrinsic.shape[0]} frames")
        
        n_interp_frames = extrinsic.shape[0]
        
        # Render interpolated frames
        for frame_idx in range(n_interp_frames):
            # Get interpolated timestamp
            if self.config.interp_interval > 1:
                t_current = self.interpolator.get_interpolated_timestamp(frame_idx, interp_interval, original_timestamps)
            else:
                t_current = timestamps[frame_idx]
            
            if not isinstance(t_current, torch.Tensor):
                t_current = torch.tensor(t_current, device=self.device, dtype=torch.float32)
            else:
                t_current = t_current.to(dtype=torch.float32)
            
            cam_idx = min(int(round(frame_idx / interp_interval)), n_original_frames - 1)
            
            # Fuse Gaussians
            fused_gaussians, attn_weights = self.fusion_module(
                gaussian_features, voxel_indices, num_voxels, t_current
            )
            
            fused_static_gs = self._unpack_gaussian_features(fused_gaussians)
            
            # Render
            renders, alphas = self._render_frame(
                fused_static_gs,
                dynamic_gs_per_frame[cam_idx],
                extrinsic[frame_idx],
                intrinsic[frame_idx],
                H, W
            )
            
            # Sky background
            bg_render = self.model.sky_model(
                images[:, cam_idx:cam_idx+1],
                original_extrinsic[cam_idx:cam_idx+1],
                original_intrinsic[cam_idx:cam_idx+1]
            )
            final_render = alphas * renders + (1 - alphas) * bg_render
            
            # Prepare visualizations
            rendered_img = final_render[0].permute(2, 0, 1).detach().cpu().clamp(0, 1)
            
            # Attention map visualization - Separate per frame
            attn_vis_list = []
            for frame_id in range(n_original_frames):
                frame_mask = (frame_indices == frame_id)
                if frame_mask.sum() > 0:
                    frame_attn_weights = attn_weights[frame_mask].cpu()
                    # Reshape to H, W
                    # Note: frame_attn_weights length matches H*W if we extracted all pixels
                    attn_map_frame = frame_attn_weights.reshape(H, W)
                    attn_vis_frame = attn_map_frame.unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)
                else:
                    attn_vis_frame = torch.zeros((3, H, W))
                attn_vis_list.append(attn_vis_frame)
            
            alpha_vis = alphas[0, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu().clamp(0, 1)
            
            # Combine visualizations
            is_keyframe = (frame_idx % interp_interval == 0)
            keyframe_idx = frame_idx // interp_interval
            
            # Row 1: GT (if available), Render, Alpha, Dynamic (if available)
            row1_imgs = []
            row1_labels = []
            
            if is_keyframe and keyframe_idx < images.shape[1]:
                gt_img = images[0, keyframe_idx].detach().cpu().clamp(0, 1)
                row1_imgs.append(gt_img)
                row1_labels.append("GT")
            else:
                # Add placeholder for GT or skip?
                # To maintain alignment, let's add black placeholder if user expects GT
                row1_imgs.append(torch.zeros_like(rendered_img))
                row1_labels.append("GT (N/A)")

            row1_imgs.append(rendered_img)
            row1_labels.append("Render")
            
            row1_imgs.append(alpha_vis)
            row1_labels.append("Alpha")
            
            # Dynamic map logic
            if is_keyframe and keyframe_idx < images.shape[1]:
                # Use actual frame dynamic map
                dyn_vis = dy_map[0, keyframe_idx].sigmoid().unsqueeze(0).repeat(3, 1, 1).cpu().clamp(0, 1)
                row1_imgs.append(dyn_vis)
                row1_labels.append("Dynamic")
            else:
                # Use nearest keyframe dynamic map or placeholder
                # Here we reuse the logic from DGGT visualizer if available, or just use interpolated/nearest
                # Given 'dy_map' is only for original frames, we pick nearest
                nearest_idx = min(int(round(frame_idx / interp_interval)), n_original_frames - 1)
                dyn_vis = dy_map[0, nearest_idx].sigmoid().unsqueeze(0).repeat(3, 1, 1).cpu().clamp(0, 1)
                row1_imgs.append(dyn_vis)
                row1_labels.append("Dynamic (Nearest)")
            
            # Row 2: Attention maps per frame
            row2_imgs = attn_vis_list
            row2_labels = [f"Attn Frame {i}" for i in range(n_original_frames)]
            
            # Stitch Row 1
            row1 = torch.cat(row1_imgs, dim=2) # [C, H, W_row1]
            
            # Stitch Row 2
            row2 = torch.cat(row2_imgs, dim=2) # [C, H, W_row2]
            
            # Pad to match width
            W_row1 = row1.shape[2]
            W_row2 = row2.shape[2]
            max_W = max(W_row1, W_row2)
            
            if W_row1 < max_W:
                pad = torch.zeros((3, H, max_W - W_row1))
                row1 = torch.cat([row1, pad], dim=2)
            
            if W_row2 < max_W:
                pad = torch.zeros((3, H, max_W - W_row2))
                row2 = torch.cat([row2, pad], dim=2)
                
            # Combine rows vertically
            combined = torch.cat([row1, row2], dim=1) # [C, H_total, max_W]
            
            # Convert to PIL and add labels manually here, instead of using _save_frame's built-in label logic
            # because _save_frame assumes single row equal panels
            if combined.shape[0] == 3:
                pil_combined = T.ToPILImage()(combined.clamp(0, 1))
            else:
                pil_combined = T.ToPILImage()(combined.permute(2, 0, 1).clamp(0, 1))

            if self.config.add_labels:
                # Add labels to Row 1
                pil_combined = ImageUtils.add_labels_to_image(
                    pil_combined,
                    row1_labels,
                    panel_width=W,
                    font_size=self.config.label_font_size,
                    text_color=self.config.label_color,
                    bg_color=self.config.label_bg_color
                )
                
                # Add labels to Row 2 (need to offset y)
                # ImageUtils.add_labels_to_image doesn't support y-offset, so we need to enhance it or mock it
                # We can crop the bottom half, label it, and paste it back? 
                # Or just manually draw.
                draw = ImageDraw.Draw(pil_combined)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.config.label_font_size)
                except:
                    font = ImageFont.load_default()
                
                for i, label in enumerate(row2_labels):
                    x_start = i * W
                    y_start = H + 5 # Start of second row + padding
                    
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    x = x_start + (W - text_width) // 2
                    
                    # Background
                    draw.rectangle(
                        [x - 5, y_start - 5, x + text_width + 5, y_start + text_height + 5],
                        fill=self.config.label_bg_color
                    )
                    draw.text((x, y_start), label, fill=self.config.label_color, font=font)

            # Save directly
            pil_combined.save(sample_dir / f"frame_{frame_idx:02d}.png")
        
        print(f"  Saved {n_interp_frames} frames to {sample_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def create_visualizer(mode: str, config: BaseVisualizationConfig) -> BaseVisualizer:
    """Factory function to create visualizer based on mode."""
    if mode == "dggt":
        return DGGTVisualizer(config)
    elif mode == "voxel":
        return VoxelVisualizer(config)
    else:
        raise ValueError(f"Unknown visualization mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Unified Visualization Tool for DGGT Models")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default="dggt", choices=["dggt", "voxel"],
                        help="Visualization mode: dggt (base model) or voxel (temporal fusion)")
    
    # Common arguments
    parser.add_argument('--image_dir', type=str, required=True, help="Dataset images directory")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Model checkpoint path")
    parser.add_argument('--output_dir', type=str, default="vis/output", help="Output directory")
    parser.add_argument('--sequence_length', type=int, default=4, help="Sequence length")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to visualize")
    parser.add_argument('--start_index', type=int, default=0, help="Starting sample index")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--interp_interval', type=int, default=0, help="Pose interpolation interval (0=disabled)")
    
    # Label options
    parser.add_argument('--no_labels', action='store_true', help="Disable text labels on images")
    parser.add_argument('--label_font_size', type=int, default=16, help="Font size for labels")
    
    # DGGT specific
    parser.add_argument('--disable_lifespan', action='store_true', help="Disable temporal decay (DGGT mode)")
    
    # Voxel specific
    parser.add_argument('--voxel_size', type=float, default=0.002, help="Voxel size (voxel mode)")
    parser.add_argument('--feature_dim', type=int, default=64, help="Feature dimension (voxel mode)")
    parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden dimension (voxel mode)")
    
    args = parser.parse_args()
    
    # Create config based on mode
    if args.mode == "dggt":
        config = DGGTVisualizationConfig(
            image_dir=args.image_dir,
            ckpt_path=args.ckpt_path,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            start_index=args.start_index,
            seed=args.seed,
            interp_interval=args.interp_interval,
            add_labels=not args.no_labels,
            label_font_size=args.label_font_size,
            disable_lifespan=args.disable_lifespan
        )
    else:  # voxel
        config = VoxelVisualizationConfig(
            image_dir=args.image_dir,
            ckpt_path=args.ckpt_path,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            start_index=args.start_index,
            seed=args.seed,
            interp_interval=args.interp_interval,
            add_labels=not args.no_labels,
            label_font_size=args.label_font_size,
            voxel_size=args.voxel_size,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim
        )
    
    # Create and run visualizer
    visualizer = create_visualizer(args.mode, config)
    visualizer.run()


if __name__ == "__main__":
    main()
