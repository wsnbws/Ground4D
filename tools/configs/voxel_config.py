"""
Configuration for Voxel-based model visualization.
"""

from dataclasses import dataclass
from typing import List
from .base_config import BaseVisualizationConfig


@dataclass
class VoxelVisualizationConfig(BaseVisualizationConfig):
    """Configuration for Voxel-based model visualization with temporal fusion."""
    # Voxel specific parameters
    voxel_size: float = 0.002
    feature_dim: int = 64
    hidden_dim: int = 32
    
    @property
    def mode(self) -> str:
        return "voxel"
    
    @property
    def panel_labels(self) -> List[str]:
        return ["GT", "Render", "Alpha", "Dynamic", "Attention"]
    
    @property
    def interp_panel_labels(self) -> List[str]:
        return ["Render", "Alpha", "Attention"]
