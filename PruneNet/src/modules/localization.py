"""
localization.py

Maps segmented trees onto the orthomosaic.

Pipeline

Segmented Tree
      │
      ▼
GPS Extraction
      │
      ▼
Orthomosaic Crop
      │
      ▼
SIFT Matching
      │
      ▼
Orthomosaic Projection
"""

import os
import cv2
import numpy as np
import rasterio

from PIL import Image
from PIL import ExifTags

from pyproj import Transformer

from rasterio.windows import Window
from src.core.database import TreeDatabase
import torchvision
import torch


class LocalizationModule:

    def __init__(

        self,

        orthomosaic_path,

        image_dataset,

        crop_width=2000,

        crop_height=2000

    ):

        self.orthomosaic_path = orthomosaic_path

        self.image_dataset = image_dataset

        self.crop_width = crop_width

        self.crop_height = crop_height

        # Load orthomosaic only once

        self.src = rasterio.open(
            orthomosaic_path
        )

        self.full_image = (
            self.src.read()
            .transpose(1,2,0)
            .astype(np.uint8)
        )[..., :3]

        self.full_image_bgr = cv2.cvtColor(
            self.full_image,
            cv2.COLOR_RGB2BGR
        )

        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            self.src.crs,
            always_xy=True
        )
    # =====================================================
    # GPS Utilities
    # =====================================================

    def dms_to_decimal(
        self,
        dms,
        ref
    ):

        d = float(dms[0])

        m = float(dms[1])

        s = float(dms[2])

        value = d + m/60 + s/3600

        if ref in ["S","W"]:

            value *= -1

        return value
    
    def extract_gps(

        self,

        image_path

    ):

        img = Image.open(
            image_path
        )

        exif = img._getexif()

        if exif is None:
            raise ValueError(
                f"No EXIF found in {image_path}"
            )

        gps = {}

        for tag, value in exif.items():

            decoded = ExifTags.TAGS.get(
                tag,
                tag
            )

            if decoded == "GPSInfo":

                for t in value:

                    gps[
                        ExifTags.GPSTAGS.get(
                            t,
                            t
                        )
                    ] = value[t]

        latitude = self.dms_to_decimal(

            gps["GPSLatitude"],

            gps["GPSLatitudeRef"]

        )

        longitude = self.dms_to_decimal(

            gps["GPSLongitude"],

            gps["GPSLongitudeRef"]

        )

        return latitude, longitude

    # =====================================================
    # Orthomosaic Crop
    # =====================================================

    def crop_from_gps(

        self,

        latitude,

        longitude

    ):

        x, y = self.transformer.transform(

            longitude,

            latitude

        )

        px, py = (

            ~self.src.transform

        ) * (x, y)

        px = int(px)

        py = int(py)

        x1 = max(

            0,

            px - self.crop_width // 2

        )

        y1 = max(

            0,

            py - self.crop_height // 2

        )

        x2 = min(

            self.src.width,

            x1 + self.crop_width

        )

        y2 = min(

            self.src.height,

            y1 + self.crop_height

        )

        window = Window(

            x1,

            y1,

            x2 - x1,

            y2 - y1

        )

        crop = self.src.read(
            window=window
        )

        crop = (
            crop.transpose(1,2,0)
            .astype(np.uint8)
        )[..., :3]

        return crop, x1, y1

    # =====================================================
    # SIFT Matching
    # =====================================================

    def match_tree(
        self,
        tree
    ):

        latitude, longitude = self.extract_gps(
            tree.image_path
        )

        tree.latitude = latitude
        tree.longitude = longitude

        crop, crop_x, crop_y = self.crop_from_gps(
            latitude,
            longitude
        )

        segmented_tree = cv2.imread(
            tree.segmented_tree_path
        )

        segmented_tree = cv2.cvtColor(
            segmented_tree,
            cv2.COLOR_BGR2RGB
        )

        gray = cv2.cvtColor(
            segmented_tree,
            cv2.COLOR_RGB2GRAY
        )

        _, binary = cv2.threshold(
            gray,
            10,
            255,
            cv2.THRESH_BINARY
        )

        coords = cv2.findNonZero(binary)

        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)

        template = segmented_tree[
            y:y+h,
            x:x+w
        ]

        result = self.compute_homography(
            template,
            crop,
            crop_x,
            crop_y
        )

        if result is None:
            return None

        bbox, score = result

        tree.match_score = score

        return bbox

    # =====================================================
    # Compute Homography
    # =====================================================

    def compute_homography(
        self,
        template,
        crop,
        crop_x,
        crop_y
    ):

        sift = cv2.SIFT_create()

        template_gray = cv2.cvtColor(
            template,
            cv2.COLOR_RGB2GRAY
        )

        crop_gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY
        )

        kp1, des1 = sift.detectAndCompute(
            template_gray,
            None
        )

        kp2, des2 = sift.detectAndCompute(
            crop_gray,
            None
        )

        if des1 is None or des2 is None:
            return None

        FLANN_INDEX_KDTREE = 1

        flann = cv2.FlannBasedMatcher(

            dict(
                algorithm=FLANN_INDEX_KDTREE,
                trees=5
            ),

            dict(
                checks=50
            )

        )

        matches = flann.knnMatch(
            des1,
            des2,
            k=2
        )

        good = []

        for m, n in matches:

            if m.distance < 0.75 * n.distance:

                good.append(m)

        if len(good) < 25:

            return None

        src_pts = np.float32(

            [

                kp1[m.queryIdx].pt

                for m in good

            ]

        ).reshape(-1,1,2)

        dst_pts = np.float32(

            [

                kp2[m.trainIdx].pt

                for m in good

            ]

        ).reshape(-1,1,2)

        H, _ = cv2.findHomography(

            src_pts,

            dst_pts,

            cv2.RANSAC,

            5.0

        )

        if H is None:

            return None

        h, w = template_gray.shape

        corners = np.float32([

            [0,0],

            [0,h-1],

            [w-1,h-1],

            [w-1,0]

        ]).reshape(-1,1,2)

        projected = cv2.perspectiveTransform(

            corners,

            H

        )

        projected = np.int32(projected)

        x_min = np.min(projected[:,0,0])

        y_min = np.min(projected[:,0,1])

        x_max = np.max(projected[:,0,0])

        y_max = np.max(projected[:,0,1])

        return (
            (
                crop_x + x_min,
                crop_y + y_min,
                crop_x + x_max,
                crop_y + y_max
            ),
            len(good)
        )

    # =====================================================
    # Localize Every Tree
    # =====================================================

    def localize_database(
        self,
        database
    ):

        all_boxes = []
        tree_lookup = []

        for tree in database:

            bbox = self.match_tree(tree)

            if bbox is None:
                continue

            tree.orthomosaic_bbox = bbox

            x1, y1, x2, y2 = bbox

            tree.orthomosaic_center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            tree.orthomosaic_radius = float( (
                min(
                    x2 - x1,
                    y2 - y1
                ) * 0.4
            ))

            all_boxes.append(bbox)

            tree_lookup.append(tree)

        return self.remove_duplicates(
            tree_lookup,
            all_boxes
        )
    # =====================================================
    # Duplicate Removal
    # =====================================================

    def remove_duplicates(
        self,
        trees,
        boxes
    ):

        if len(boxes) == 0:
            return trees

        boxes_tensor = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        scores = torch.ones(
            (
                len(boxes),
            )
        )

        keep = torchvision.ops.nms(

            boxes_tensor,

            scores,

            iou_threshold=0.30

        )
        keep = keep.cpu().numpy()
        keep = set(keep.tolist())
        unique_trees = []

        for i, tree in enumerate(trees):

            if i in keep:

                tree.is_duplicate = False

                unique_trees.append(tree)

            else:

                tree.is_duplicate = True

        return unique_trees

    # =====================================================
    # Update Tree Database
    # =====================================================

    def update_database(
        self,
        database,
        localized_trees
    ):

        new_database = TreeDatabase()

        for tree in localized_trees:

            if tree.is_duplicate:
                continue

            new_database.add_tree(tree)

        return new_database

    # =====================================================
    # Main Function
    # =====================================================

    def process_database(
        self,
        database
    ):

        localized = self.localize_database(
            database
        )

        database = self.update_database(

            database,

            localized

        )

        return database

    # =====================================================
    # Draw Localization Result
    # =====================================================

    def draw_localization(
        self,
        database,
        output_folder
    ):

        output_dir = os.path.join(
            output_folder,
            "localization"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        image = self.full_image_bgr.copy()

        for tree in database:

            if tree.orthomosaic_center is None:
                continue

            cx, cy = tree.orthomosaic_center

            radius = int(
                tree.orthomosaic_radius
            )

            color = (
                (0, 0, 255)
                if tree.needs_pruning
                else
                (0, 255, 0)
            )

            cv2.circle(
                image,
                (cx, cy),
                radius,
                color,
                20
            )

            cv2.putText(
                image,
                str(tree.tree_id),
                (cx-20, cy-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

        save_path = os.path.join(
            output_dir,
            "localized_trees.png"
        )

        cv2.imwrite(
            save_path,
            image
        )

        return save_path
    # =====================================================
    # Export Localization
    # =====================================================

    def export_results(
        self,
        database,
        output_folder
    ):

        output_dir = os.path.join(
            output_folder,
            "localization"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        df = database.to_dataframe()

        df.to_excel(

            os.path.join(
                output_dir,
                "localized_trees.xlsx"
            ),

            index=False

        )

        pruned = df[
            df["needs_pruning"] == True
        ]

        pruned.to_excel(

            os.path.join(
                output_dir,
                "trees_to_prune.xlsx"
            ),

            index=False

        )
    # =====================================================
    # Run
    # =====================================================

    def run(
        self,
        database,
        output_folder
    ):

        database = self.process_database(
            database
        )

        self.draw_localization(
            database,
            output_folder
        )

        self.export_results(
            database,
            output_folder
        )

        print(
            "Localization complete."
        )

        return database