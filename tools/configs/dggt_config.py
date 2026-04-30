"""
Configuration for DGGT base model visualization.
"""

from dataclasses import dataclass
from typing import List
from .base_config import BaseVisualizationConfig


@dataclass
class DGGTVisualizationConfig(BaseVisualizationConfig):
    """Configuration for DGGT base model visualization."""
    # DGGT specific parameters
    disable_lifespan: bool = False
    
    @property
    def mode(self) -> str:
        return "dggt"
    
    @property
    def panel_labels(self) -> List[str]:
        return ["GT", "Render", "Alpha", "Dynamic"]
    
    @property
    def interp_panel_labels(self) -> List[str]:
        return ["Render", "Alpha", "Dynamic"]
