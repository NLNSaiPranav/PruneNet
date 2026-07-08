"""
iwcp.py

Intensity-Weighted Canopy Projection (IWCP)

This module estimates canopy opening from each segmented tree
using pseudo-depth generated from RGB intensity.
"""

import os
import cv2
import numpy as np

from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter
)

from src.core.tree import Tree
from src.core.database import TreeDatabase


class IWCPAnalyzer:

    def __init__(
        self,
        pruning_threshold=8.3,
        analysis_margin=0.10,
        percentile=10
    ):

        self.pruning_threshold = pruning_threshold
        self.analysis_margin = analysis_margin
        self.percentile = percentile

    # =====================================================
    # Largest Inscribed Analysis Region
    # =====================================================

    def create_analysis_mask(
        self,
        mask
    ):

        dist_transform = distance_transform_edt(mask)

        max_dist = np.max(dist_transform)

        center_y, center_x = np.unravel_index(
            np.argmax(dist_transform),
            dist_transform.shape
        )

        radius = int(max_dist)

        margin = int(
            radius *
            self.analysis_margin
        )

        analysis_mask = (
            dist_transform >= margin
        )

        return (
            analysis_mask,
            radius,
            (center_x, center_y)
        )

    # =====================================================
    # Pseudo Depth
    # =====================================================

    def create_pseudo_depth(
        self,
        gray,
        analysis_mask
    ):

        gray = gaussian_filter(
            gray.astype(np.float32),
            sigma=5
        )

        pseudo_depth = (
            gray /
            255.0
        ) * 100.0

        pseudo_values = pseudo_depth[
            analysis_mask
        ]

        threshold = np.percentile(
            pseudo_values,
            self.percentile
        )

        opening_mask = (
            pseudo_depth <= threshold
        ) & analysis_mask

        return (
            pseudo_depth,
            opening_mask,
            threshold
        )

    # =====================================================
    # Point Cloud
    # =====================================================

    def create_point_cloud(
        self,
        gray,
        analysis_mask
    ):

        circle_y, circle_x = np.where(
            analysis_mask
        )

        depth = (
            gray[
                circle_y,
                circle_x
            ].astype(np.float32)
            /
            255.0
        ) * 100.0

        h, w = gray.shape

        x = circle_x.astype(np.float32)
        y = circle_y.astype(np.float32)

        y = h - y

        x = w - x
        y = h - y

        points = np.column_stack(
            (
                x,
                y,
                depth
            )
        )

        sort_idx = np.argsort(
            -depth
        )

        return (
            points[sort_idx],
            circle_x,
            circle_y,
            sort_idx
        )

    # =====================================================
    # Opening Overlay
    # =====================================================

    def create_overlay(
        self,
        masked_tree,
        opening_mask,
        mask
    ):

        overlay = masked_tree.copy()

        overlay[
            opening_mask
        ] = [255, 0, 0]

        ys, xs = np.where(mask)

        ymin = np.min(ys)
        ymax = np.max(ys)

        xmin = np.min(xs)
        xmax = np.max(xs)

        overlay = overlay[
            ymin:ymax + 1,
            xmin:xmax + 1
        ]

        overlay = cv2.copyMakeBorder(
            overlay,
            50,
            50,
            50,
            50,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        overlay = cv2.resize(
            overlay,
            (600, 600)
        )

        return overlay
    # =====================================================
    # Analyze Single Tree
    # =====================================================

    def analyze_tree(
        self,
        tree: Tree,
        image_rgb
    ):

        mask = tree.mask

        # ==============================================
        # Extract Tree
        # ==============================================

        masked_tree = cv2.imread(
            tree.segmented_tree_path
        )

        masked_tree = cv2.cvtColor(
            masked_tree,
            cv2.COLOR_BGR2RGB
        )

        gray = cv2.cvtColor(
            masked_tree,
            cv2.COLOR_RGB2GRAY
        )

        # ==============================================
        # Analysis Region
        # ==============================================

        (
            analysis_mask,
            radius,
            center
        ) = self.create_analysis_mask(mask)

        # ==============================================
        # Pseudo Depth
        # ==============================================

        (
            pseudo_depth,
            opening_mask,
            threshold
        ) = self.create_pseudo_depth(
            gray,
            analysis_mask
        )

        # ==============================================
        # Point Cloud
        # ==============================================

        (
            points_3d,
            circle_x,
            circle_y,
            sort_idx
        ) = self.create_point_cloud(
            gray,
            analysis_mask
        )

        # ==============================================
        # Point Colors
        # ==============================================

        reduced_tree = cv2.bitwise_and(
            masked_tree,
            masked_tree,
            mask=analysis_mask.astype(np.uint8) * 255
        )

        colors = (
            reduced_tree[
                circle_y,
                circle_x
            ] / 255.0
        )

        colors = colors[
            sort_idx
        ]

        opening_flags = opening_mask[
            circle_y,
            circle_x
        ]

        opening_flags = opening_flags[
            sort_idx
        ]

        colors[
            opening_flags
        ] = [1.0, 0.0, 0.0]

        # ==============================================
        # Opening Statistics
        # ==============================================

        total_tree_pixels = int(
            np.sum(mask)
        )

        opening_pixels = int(
            np.sum(opening_mask)
        )

        opening_percentage = (
            opening_pixels
            /
            total_tree_pixels
        ) * 100.0

        needs_pruning = (
            opening_percentage
            <
            self.pruning_threshold
        )

        # ==============================================
        # Overlay
        # ==============================================

        overlay = self.create_overlay(
            masked_tree,
            opening_mask,
            mask
        )

        # ==============================================
        # Update Tree Object
        # ==============================================

        tree.opening_percentage = float(
            opening_percentage
        )

        tree.opening_pixels = opening_pixels

        tree.roi_area = int(
            np.sum(analysis_mask)
        )

        tree.pruning_threshold = (
            self.pruning_threshold
        )

        tree.needs_pruning = (
            needs_pruning
        )

        tree.metadata["analysis_radius"] = radius

        tree.metadata["analysis_center"] = center

        tree.metadata["pseudo_depth_threshold"] = float(
            threshold
        )

        tree.metadata["point_cloud"] = points_3d

        tree.metadata["point_colors"] = colors

        return (
            tree,
            overlay,
            opening_mask,
            points_3d,
            colors
        )
    # =====================================================
    # Analyze Database
    # =====================================================

    def analyze_database(
        self,
        database: TreeDatabase,
        image_rgb,
        output_folder=None
    ):

        if output_folder is not None:

            overlay_folder = os.path.join(
                output_folder,
                "iwcp_overlays"
            )

            os.makedirs(
                overlay_folder,
                exist_ok=True
            )

        for tree in database:

            (
                tree,
                overlay,
                opening_mask,
                points_3d,
                colors
            ) = self.analyze_tree(
                tree,
                image_rgb
            )

            if output_folder is not None:

                overlay_name = (
                    os.path.splitext(tree.image_name)[0]
                    +
                    f"_tree_{tree.tree_id}_iwcp.png"
                )

                overlay_path = os.path.join(
                    overlay_folder,
                    overlay_name
                )

                cv2.imwrite(
                    overlay_path,
                    cv2.cvtColor(
                        overlay,
                        cv2.COLOR_RGB2BGR
                    )
                )

                tree.iwcp_overlay_path = overlay_path

        return database

    # =====================================================
    # Process Image
    # =====================================================

    def process_image(
        self,
        database: TreeDatabase,
        image_rgb,
        output_folder=None
    ):

        database = self.analyze_database(
            database,
            image_rgb,
            output_folder
        )

        return database

    # =====================================================
    # Export Results
    # =====================================================

    def export_results(
        self,
        database: TreeDatabase,
        output_folder
    ):

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        df = database.to_dataframe()

        df.to_excel(
            os.path.join(
                output_folder,
                "iwcp_results.xlsx"
            ),
            index=False
        )

        pruned_df = df[
            df["needs_pruning"] == True
        ]

        pruned_df.to_excel(
            os.path.join(
                output_folder,
                "trees_to_prune.xlsx"
            ),
            index=False
        )

        return df

    # =====================================================
    # Complete Pipeline
    # =====================================================

    def run(
        self,
        database: TreeDatabase,
        image_rgb,
        output_folder=None
    ):

        database = self.process_image(
            database,
            image_rgb,
            output_folder
        )

        if output_folder is not None:

            self.export_results(
                database,
                output_folder
            )

        return database