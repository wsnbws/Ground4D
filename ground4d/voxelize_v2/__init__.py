"""
Voxelize V2 Module: Temporal-aware Gaussian Voxelization with Dynamic Fusion

This module implements a more sophisticated voxelization strategy where:
1. Multiple Gaussians can exist in a single voxel (from different views/times)
2. At render time, Gaussians within each voxel are dynamically fused based on:
   - Temporal proximity to target time
   - Feature quality/confidence
3. Each voxel outputs a single representative Gaussian for rendering

This reduces the number of Gaussians for rendering while maintaining temporal consistency.
"""

from .voxelizer_v2 import GaussianVoxelizerV2
from .temporal_fusion_v1 import TemporalVoxelFusion
from .temporal_fusion_v3 import TemporalVoxelFusionV3

__all__ = ['GaussianVoxelizerV2', 'TemporalVoxelFusion', 'TemporalVoxelFusionV3']
