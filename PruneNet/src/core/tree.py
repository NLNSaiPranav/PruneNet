"""
tree.py

Defines the Tree object used throughout the PruneNet pipeline.

Each detected tree is represented as a Tree instance and is
updated as it progresses through segmentation, canopy analysis,
localization, and path planning.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class Tree:
    """
    Represents one detected tree.
    """

    # ==========================================================
    # Basic Information
    # ==========================================================
    tree_id: int
    image_name: str
    image_path: str = ""
    # ==========================================================
    # Segmentation
    # ==========================================================

    mask: Optional[np.ndarray] = None

    segmented_tree_path: str = ""

    overlay_path: str = ""

    bbox: Optional[Tuple[int, int, int, int]] = None
    # (x1, y1, x2, y2)

    confidence: float = 0.0

    mask_area: int = 0

    # ==========================================================
    # IWCP
    # ==========================================================

    opening_percentage: float = 0.0

    opening_pixels: int = 0

    roi_area: int = 0

    pruning_threshold: float = 0.5

    needs_pruning: bool = False

    iwcp_overlay_path: str = ""

    # ==========================================================
    # Localization
    # ==========================================================

    latitude: Optional[float] = None

    longitude: Optional[float] = None

    gps_image: str = ""

    orthomosaic_bbox: Optional[Tuple[int,int,int,int]] = None

    orthomosaic_center: Optional[Tuple[int,int]] = None

    orthomosaic_radius: float = 0.0

    is_duplicate: bool = False

    match_score: int = 0

    # ==========================================================
    # Path Planning
    # ==========================================================

    visit_order: Optional[int] = None

    cumulative_distance: Optional[float] = None

    planner_index: Optional[int] = None

    # ==========================================================
    # Extra Metadata
    # ==========================================================

    metadata: dict = field(default_factory=dict)

    # ==========================================================
    # Helper Methods
    # ==========================================================

    def center(self):
        """
        Returns the center of the segmentation bounding box.
        """

        if self.bbox is None:
            return None

        x1, y1, x2, y2 = self.bbox

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def width(self):

        if self.bbox is None:
            return None

        return self.bbox[2] - self.bbox[0]

    def height(self):

        if self.bbox is None:
            return None

        return self.bbox[3] - self.bbox[1]

    def __repr__(self):

        return (
            f"Tree("
            f"id={self.tree_id}, "
            f"image='{self.image_name}', "
            f"opening={self.opening_percentage:.2f}%, "
            f"prune={self.needs_pruning})"
        )
