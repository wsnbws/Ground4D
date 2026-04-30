"""
Improved Gaussian Voxelizer that preserves all Gaussian information
for later dynamic fusion based on temporal proximity.
"""

import torch
import torch.nn as nn


class GaussianVoxelizerV2(nn.Module):
    """
    Voxelizer that groups Gaussians spatially but preserves individual Gaussian data.
    
    Unlike the original voxelizer that immediately fuses Gaussians, this version:
    - Assigns each Gaussian to a voxel ID
    - Preserves all original Gaussian parameters
    - Allows downstream modules to decide how to fuse based on temporal/feature info
    """
    
    def __init__(self, voxel_size=0.02):
        """
        Args:
            voxel_size: Size of each voxel cube in world coordinates
        """
        super().__init__()
        self.voxel_size = voxel_size
    
    def voxelize(self, positions, gaussian_features):
        """
        Voxelize Gaussians by spatial location, preserving all information.

        Args:
            positions: [N, 3] 3D positions of Gaussians
            gaussian_features: [N, D] Full Gaussian features including:
                - positions (3)
                - colors (3)
                - opacity (1)
                - scales (3)
                - rotations (4)
                - timestamp (1)
                Total: D = 15

        Returns:
            voxel_indices: [N] Voxel ID for each Gaussian (0 to M-1)
            num_voxels: M, number of unique voxels
            voxel_counts: [M] Number of Gaussians in each voxel
        """
        # Quantize positions to integer voxel grid coordinates.
        voxel_coords = (positions / self.voxel_size).round().long()  # [N, 3]

        # torch.unique on the full [N, 3] coordinate tensor gives an exact grouping
        # with zero risk of hash collisions.  It is O(N log N) — same asymptotic
        # complexity as hash-based approaches but without correctness concerns.
        _, voxel_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        num_voxels = int(voxel_indices.max().item()) + 1

        # Count Gaussians per voxel
        voxel_counts = torch.bincount(voxel_indices, minlength=num_voxels)

        return voxel_indices, num_voxels, voxel_counts
    
    def get_voxel_statistics(self, voxel_indices, num_voxels, gaussian_features, timestamps):
        """
        Compute per-voxel statistics for debugging/analysis.
        
        Args:
            voxel_indices: [N] Voxel assignment
            num_voxels: M
            gaussian_features: [N, D] Gaussian features
            timestamps: [N] Timestamp for each Gaussian
        
        Returns:
            Dictionary with per-voxel statistics
        """
        device = gaussian_features.device
        M = num_voxels
        
        # Count Gaussians per voxel
        counts = torch.bincount(voxel_indices, minlength=M)
        
        # Average timestamp per voxel
        voxel_time_sum = torch.zeros(M, device=device)
        voxel_time_sum.scatter_add_(0, voxel_indices, timestamps)
        avg_time = voxel_time_sum / (counts.float() + 1e-8)
        
        # Time span per voxel (max - min)
        voxel_time_min = torch.full((M,), float('inf'), device=device)
        voxel_time_max = torch.full((M,), float('-inf'), device=device)
        voxel_time_min.scatter_reduce_(0, voxel_indices, timestamps, reduce='amin')
        voxel_time_max.scatter_reduce_(0, voxel_indices, timestamps, reduce='amax')
        time_span = voxel_time_max - voxel_time_min
        time_span[counts == 1] = 0  # Single Gaussian voxels have 0 span
        
        return {
            'counts': counts,
            'avg_time': avg_time,
            'time_span': time_span,
            'single_gaussian_voxels': (counts == 1).sum().item(),
            'multi_gaussian_voxels': (counts > 1).sum().item(),
            'max_gaussians_per_voxel': counts.max().item(),
        }
