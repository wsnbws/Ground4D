"""
Base configuration for visualization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class BaseVisualizationConfig:
    """Base configuration for all visualization modes."""
    # Required paths
    image_dir: str = ""
    ckpt_path: str = ""
    output_dir: str = "vis/output"
    
    # Dataset parameters
    sequence_length: int = 4
    
    # Sampling parameters
    num_samples: int = 1
    start_index: int = 0
    seed: int = 42
    
    # Interpolation
    interp_interval: int = 0  # 0 = disabled
    
    # Visualization options
    add_labels: bool = True  # Add text labels on top of images
    label_font_size: int = 16
    label_color: str = "white"
    label_bg_color: str = "black"
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def mode(self) -> str:
        """Visualization mode name."""
        return "base"
    
    @property
    def panel_labels(self) -> List[str]:
        """Labels for each panel in the visualization."""
        return ["GT", "Render", "Alpha", "Dynamic"]
    
    @property
    def interp_panel_labels(self) -> List[str]:
        """Labels for interpolated frames (no GT)."""
        return ["Render", "Alpha", "Dynamic"]
