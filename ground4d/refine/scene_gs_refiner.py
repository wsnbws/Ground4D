# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene-level Gaussian Consistency Refiner

This module refines multi-view fused 3D Gaussians using PointTransformer.
It addresses inconsistencies from multi-view fusion by:
1. Adjusting Gaussian positions based on spatial neighborhood
2. Refining Gaussian shapes (scales and rotations)
3. Predicting confidence scores for adaptive pruning/weighting
"""

import torch
import torch.nn as nn
from typing import Dict
from ground4d.splatformer.pointtransformer_v3 import PointTransformerV3Model


class MinMaxScaler:
    """Simple min-max scaler for normalizing 3D coordinates to [0,1]^3"""
    def __init__(self):
        self.scale_ = None
        self.trans_ = None
        self.center = None
        
    def fit_transform(self, X):
        """Fit and transform coordinates to [0,1] range"""
        data_min = X.min(dim=0)[0]
        data_max = X.max(dim=0)[0]
        data_range = data_max - data_min
        
        # Avoid division by zero
        data_range = torch.where(data_range < 1e-6, torch.ones_like(data_range), data_range)
        
        # Use uniform scale to preserve aspect ratio
        self.scale_ = 1.0 / data_range.max()
        self.center = torch.tensor([0.5, 0.5, 0.5], device=X.device)
        
        scaled_X = X * self.scale_
        scaled_X_mid = (scaled_X.min(dim=0)[0] + scaled_X.max(dim=0)[0]) / 2
        self.trans_ = self.center - scaled_X_mid
        
        return scaled_X + self.trans_
    
    def inverse_transform(self, X_scaled):
        """Transform normalized coordinates back to original space"""
        X = X_scaled - self.trans_
        return X / self.scale_


class SceneGaussianConsistencyRefiner(nn.Module):
    """
    Refines 3D Gaussian splats using PointTransformer for multi-view consistency.
    
    Args:
        enable_flash: Whether to use flash attention (faster for large point clouds)
        grid_resolution: Voxel grid resolution for spatial hashing
        use_input_in_head: Whether to concatenate input features to head input
        enc_dim: Encoder dimension (controls model size)
        output_dim: Output feature dimension
        enc_depths: Number of blocks per encoder stage (controls depth)
    """
    
    def __init__(
        self,
        enable_flash: bool = True,
        grid_resolution: int = 384,
        use_input_in_head: bool = False,
        enc_dim: int = 64,
        output_dim: int = 96,
        enc_depths: tuple = (1, 1, 1, 2, 1),  # Lightweight: fewer blocks
        enc_num_head: tuple = (2, 4, 8, 16, 32),
        dec_depths: tuple = (1, 1, 1, 1),     # Lightweight decoder
        dec_num_head: tuple = (4, 4, 8, 16),
        stride: tuple = (2, 2, 2, 2),         # Downsampling strides
    ):
        super().__init__()
        
        self.grid_resolution = grid_resolution
        self.use_input_in_head = use_input_in_head
        
        # Input features: colors (3) + opacity (1) + scales (3) + quats (4) = 11
        self.in_channels = 11
        
        # PointTransformer V3 backbone (lightweight version)
        self.backbone = PointTransformerV3Model(
            in_channels=self.in_channels,
            enable_flash=enable_flash,
            enc_dim=enc_dim,
            output_dim=output_dim,
            turn_off_bn=False,
            stride=stride,
            embedding_type='MLP',
            enc_depths=enc_depths,
            enc_num_head=enc_num_head,
            dec_depths=dec_depths,
            dec_num_head=dec_num_head,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )
        
        head_input_dim = self.backbone.output_dim
        if self.use_input_in_head:
            head_input_dim += self.in_channels
        
        # Output heads for refinement
        # We predict residuals for most attributes
        hidden_dim = 128
        
        # Position residual (small adjustments)
        self.position_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),  # Bounded adjustments
        )
        
        # Scale residual (log-space adjustments)
        self.scale_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        
        # Rotation residual (quaternion adjustments)
        self.rotation_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        
        # Color residual (small color adjustments)
        self.color_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        
        # Confidence score (per-Gaussian quality/reliability)
        self.confidence_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize last layers to zero for stable training
        self._zero_init_heads()
        
    def _zero_init_heads(self):
        """Initialize output layers to zero for residual learning"""
        for head in [self.position_head, self.scale_head, self.rotation_head, 
                     self.color_head, self.confidence_head]:
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
    
    def normalize_gs(self, gs_dict: Dict[str, torch.Tensor]):
        """Normalize Gaussian positions and scales to [0,1] range"""
        scaler = MinMaxScaler()
        
        normalized_gs = {}
        normalized_gs['means'] = scaler.fit_transform(gs_dict['means'])
        
        # Adjust scales in log-space
        normalized_gs['scales'] = gs_dict['scales'] + torch.log(scaler.scale_).to(gs_dict['scales'].device)
        
        # Copy other attributes as-is
        normalized_gs['features_dc'] = gs_dict['features_dc']
        normalized_gs['opacities'] = gs_dict['opacities']
        normalized_gs['quats'] = gs_dict['quats']
        
        return normalized_gs, scaler
    
    def unnormalize_gs(self, gs_dict: Dict[str, torch.Tensor], scaler: MinMaxScaler):
        """Transform normalized Gaussians back to original space"""
        unnormalized_gs = {}
        
        unnormalized_gs['means'] = scaler.inverse_transform(gs_dict['means'])
        unnormalized_gs['scales'] = gs_dict['scales'] - torch.log(scaler.scale_).to(gs_dict['scales'].device)
        
        # Copy other attributes
        unnormalized_gs['features_dc'] = gs_dict['features_dc']
        unnormalized_gs['opacities'] = gs_dict['opacities']
        unnormalized_gs['quats'] = gs_dict['quats']
        
        if 'confidence' in gs_dict:
            unnormalized_gs['confidence'] = gs_dict['confidence']
        
        return unnormalized_gs
    
    def forward(self, gs_dict: Dict[str, torch.Tensor], return_confidence: bool = False):
        """
        Refine 3D Gaussians using spatial context from PointTransformer.
        
        Args:
            gs_dict: Dictionary containing:
                - means: (N, 3) Gaussian positions
                - features_dc: (N, 3) RGB colors
                - opacities: (N, 1) opacity values
                - scales: (N, 3) scale parameters
                - quats: (N, 4) rotation quaternions
            return_confidence: If True, return confidence separately for pruning.
                               If False (training), modulate opacity by confidence.
        
        Returns:
            Refined gs_dict with modulated opacity (and optionally confidence key)
        """
        device = gs_dict['means'].device
        N = gs_dict['means'].shape[0]
        
        # Handle edge case: empty or very small point cloud
        if N < 10:
            if return_confidence:
                gs_dict['confidence'] = torch.ones((N, 1), device=device)
            return gs_dict
        
        # 1. Normalize coordinates to [0,1]^3
        normalized_gs, scaler = self.normalize_gs(gs_dict)
        
        # 2. Prepare input features for PointTransformer
        # Concatenate all Gaussian attributes as features
        opacities = normalized_gs['opacities']
        if opacities.dim() == 1:
            opacities = opacities[:, None]
        
        input_feat = torch.cat([
            normalized_gs['features_dc'],  # 3
            opacities,                      # 1
            normalized_gs['scales'],        # 3
            normalized_gs['quats'],         # 4
        ], dim=-1)  # (N, 11)
        
        # 3. Build Point structure for PointTransformer
        model_input = {
            'coord': normalized_gs['means'],
            'feat': input_feat,
            'offset': torch.tensor([N], device=device),
            'grid_size': torch.ones([3], device=device) / self.grid_resolution,
        }
        model_input['grid_coord'] = torch.floor(
            model_input['coord'] * self.grid_resolution
        ).int()
        
        # 4. PointTransformer forward pass
        output = self.backbone(model_input)
        hidden_feat = output['feat']  # (N, output_dim)
        
        # Optionally concatenate input features
        if self.use_input_in_head:
            hidden_feat = torch.cat([hidden_feat, input_feat], dim=-1)
        
        # 5. Predict refinements and confidence
        pos_residual = self.position_head(hidden_feat)  # (N, 3), bounded by tanh
        scale_residual = self.scale_head(hidden_feat)    # (N, 3)
        rot_residual = self.rotation_head(hidden_feat)   # (N, 4)
        color_residual = self.color_head(hidden_feat)    # (N, 3)
        confidence_logit = self.confidence_head(hidden_feat)  # (N, 1)
        
        # 6. Apply residuals
        refined_gs = {}
        
        # Position: small bounded adjustment (tanh gives [-1,1], scale to small range)
        pos_scale = 0.05  # Max 5% adjustment in normalized space
        refined_gs['means'] = normalized_gs['means'] + pos_scale * pos_residual
        
        # Scale: log-space adjustment with small magnitude
        scale_adjustment_scale = 0.1
        refined_gs['scales'] = normalized_gs['scales'] + scale_adjustment_scale * scale_residual
        
        # Rotation: add small quaternion residual and renormalize
        rot_adjustment_scale = 0.1
        refined_quats = normalized_gs['quats'] + rot_adjustment_scale * rot_residual
        refined_gs['quats'] = refined_quats / (refined_quats.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Color: small adjustment
        color_adjustment_scale = 0.05
        refined_gs['features_dc'] = normalized_gs['features_dc'] + color_adjustment_scale * color_residual
        refined_gs['features_dc'] = refined_gs['features_dc'].clamp(0, 1)
        
        # Confidence: sigmoid to [0,1]
        confidence = torch.sigmoid(confidence_logit)
        
        # Opacity: modulate by confidence (for training) or keep separate (for pruning)
        if return_confidence:
            refined_gs['opacities'] = normalized_gs['opacities']
            refined_gs['confidence'] = confidence
        else:
            # During training: bake confidence into opacity for gradient flow
            opacities = normalized_gs['opacities']
            if opacities.dim() == 1:
                opacities = opacities[:, None]
            refined_gs['opacities'] = opacities * confidence
        
        # 7. Unnormalize back to original space
        refined_gs = self.unnormalize_gs(refined_gs, scaler)
        
        return refined_gs
    
    def prune(self, gs_dict: Dict[str, torch.Tensor], w_thresh: float = 0.1):
        """
        Prune low-confidence Gaussians (for inference/evaluation).
        Also modulates opacity by confidence for the remaining Gaussians.
        
        Args:
            gs_dict: Gaussian dictionary with 'confidence' key
            w_thresh: Confidence threshold (prune if confidence < w_thresh)
        
        Returns:
            Pruned gs_dict without confidence key (confidence baked into opacity)
        """
        if 'confidence' not in gs_dict:
            return gs_dict
        
        confidence = gs_dict['confidence']  # (N, 1) or (N,)
        if confidence.dim() == 2:
            confidence = confidence.squeeze(-1)  # (N,)
        
        mask = confidence >= w_thresh
        
        pruned_gs = {}
        for key, val in gs_dict.items():
            if key == 'confidence':
                continue  # Will be baked into opacity
            pruned_gs[key] = val[mask]
        
        # Modulate opacity by confidence for remaining Gaussians
        confidence_kept = confidence[mask]
        opacities = pruned_gs['opacities']
        if opacities.dim() == 1:
            opacities = opacities[:, None]
        pruned_gs['opacities'] = opacities * confidence_kept[:, None]
        
        return pruned_gs

