"""
Mask R-CNN based tree segmentation module for PruneNet.

This module performs:

1. Model loading
2. Instance segmentation
3. Overlap removal
4. Overlay generation
5. Tree object creation
"""

import cv2
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.core.tree import Tree
from src.core.database import TreeDatabase


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


class MaskRCNNSegmenter:

    def __init__(
        self,
        model_path,
        confidence_threshold=0.6,
        overlap_threshold=0.30,
        device=None
    ):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = self.load_model()

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(self):

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False
        )

        in_features = (
            model.roi_heads.box_predictor.cls_score.in_features
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            2
        )

        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            2
        )

        model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device
            )
        )

        model.to(self.device)
        model.eval()

        return model

    # =====================================================
    # Remove Duplicate Masks
    # =====================================================

    def filter_overlapping_masks(
        self,
        masks,
        boxes,
        scores
    ):

        if len(masks) == 0:
            return [], [], []

        sorted_indices = np.argsort(
            [np.sum(m) for m in masks]
        )[::-1]

        masks = [masks[i] for i in sorted_indices]
        boxes = [boxes[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        keep = np.ones(
            len(masks),
            dtype=bool
        )

        for i in range(len(masks)):

            if not keep[i]:
                continue

            for j in range(i + 1, len(masks)):

                if not keep[j]:
                    continue

                overlap = np.logical_and(
                    masks[i],
                    masks[j]
                ).sum()

                min_area = min(
                    masks[i].sum(),
                    masks[j].sum()
                )

                if min_area == 0:
                    continue

                overlap_ratio = overlap / min_area

                if overlap_ratio > self.overlap_threshold:
                    keep[j] = False

        masks = [
            masks[i]
            for i in range(len(masks))
            if keep[i]
        ]

        boxes = [
            boxes[i]
            for i in range(len(boxes))
            if keep[i]
        ]

        scores = [
            scores[i]
            for i in range(len(scores))
            if keep[i]
        ]

        return masks, boxes, scores

    # =====================================================
    # Create Overlay
    # =====================================================

    def create_overlay(
        self,
        image_rgb,
        masks
    ):

        overlay = image_rgb.copy()

        for idx, mask in enumerate(masks):

            color = COLORS[idx % len(COLORS)]

            color_layer = np.zeros_like(
                overlay,
                dtype=np.uint8
            )

            for c in range(3):
                color_layer[:, :, c] = (
                    mask.astype(np.uint8)
                    * color[c]
                )

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                color_layer,
                0.45,
                0
            )

            ys, xs = np.where(mask)

            if len(xs):

                cx = int(xs.mean())
                cy = int(ys.mean())

                cv2.putText(
                    overlay,
                    str(idx),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

        return overlay
    # =====================================================
    # Predict
    # =====================================================

    def predict(self, image_rgb):
        """
        Run Mask R-CNN inference on an RGB image.

        Returns
        -------
        masks : list[np.ndarray]
        boxes : list[np.ndarray]
        scores : list[float]
        """

        tensor_img = (
            F.to_tensor(Image.fromarray(image_rgb))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            prediction = self.model(tensor_img)[0]

        pred_masks_raw = prediction["masks"].cpu().numpy()
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()

        masks = []
        boxes = []
        scores = []

        for i in range(len(pred_scores)):

            if pred_scores[i] < self.confidence_threshold:
                continue

            mask = pred_masks_raw[i, 0] > 0.5

            masks.append(mask)
            boxes.append(pred_boxes[i])
            scores.append(float(pred_scores[i]))

        masks, boxes, scores = self.filter_overlapping_masks(
            masks,
            boxes,
            scores
        )

        return masks, boxes, scores

    # =====================================================
    # Create Tree Database
    # =====================================================

    def create_tree_database(
        self,
        image_name,
        masks,
        boxes,
        scores
    ):
        """
        Convert predictions into Tree objects.
        """

        database = TreeDatabase()

        for idx, (mask, box, score) in enumerate(
            zip(masks, boxes, scores)
        ):

            x1, y1, x2, y2 = box.astype(int)

            tree = Tree(

                tree_id=idx,

                image_name=os.path.basename(image_name),

                image_path=image_name,

                mask=mask,

                bbox=(x1, y1, x2, y2),

                confidence=float(score),

                mask_area=int(np.sum(mask))
            )

            database.add_tree(tree)

        return database

    # =====================================================
    # Segment Image
    # =====================================================

    def segment_image(
        self,
        image_path
    ):
        """
        Segment one RGB image.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        database : TreeDatabase

        overlay : np.ndarray

        image_rgb : np.ndarray
        """

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        masks, boxes, scores = self.predict(
            image_rgb
        )

        database = self.create_tree_database(
            image_name=image_path,
            masks=masks,
            boxes=boxes,
            scores=scores
        )

        overlay = self.create_overlay(
            image_rgb,
            masks
        )

        return database, overlay, image_rgb
    # =====================================================
    # Run Single Image
    # =====================================================

    def run(
        self,
        image_path
    ):
        """
        Run segmentation on a single image.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        TreeDatabase
        overlay
        image_rgb
        """

        database, overlay, image_rgb = self.segment_image(
            image_path
        )

        return database, overlay, image_rgb

    # =====================================================
    # Run Folder
    # =====================================================

    def run_folder(
        self,
        folder_path
    ):
        """
        Segment every image inside a folder.

        Parameters
        ----------
        folder_path : str

        Returns
        -------
        TreeDatabase
        """

        import os

        master_database = TreeDatabase()

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        )

        image_files = sorted(

            [

                os.path.join(folder_path, file)

                for file in os.listdir(folder_path)

                if file.lower().endswith(valid_extensions)

            ]

        )

        tree_counter = 0

        for image_path in image_files:

            database, _, _ = self.run(
                image_path
            )

            for tree in database:

                tree.tree_id = tree_counter

                master_database.add_tree(tree)

                tree_counter += 1

        return master_database
    # =====================================================
    # Save Masks
    # =====================================================

    def save_segmented_trees(
        self,
        database,
        image_rgb,
        output_folder
    ):

        segmented_folder = os.path.join(
            output_folder,
            "segmented_trees"
        )

        os.makedirs(
            segmented_folder,
            exist_ok=True
        )

        for tree in database:

            segmented_tree = image_rgb.copy()

            segmented_tree[~tree.mask] = 0

            filename = (
                f"{os.path.splitext(tree.image_name)[0]}"
                f"_tree_{tree.tree_id}.png"
            )

            save_path = os.path.join(
                segmented_folder,
                filename
            )

            cv2.imwrite(
                save_path,
                cv2.cvtColor(
                    segmented_tree,
                    cv2.COLOR_RGB2BGR
                )
            )

            tree.segmented_tree_path = save_path

    # =====================================================
    # Save Overlay
    # =====================================================

    def save_overlay(
        self,
        overlay,
        image_name,
        output_folder
    ):

        overlay_folder = os.path.join(
            output_folder,
            "overlays"
        )

        os.makedirs(
            overlay_folder,
            exist_ok=True
        )

        filename = (
            os.path.splitext(
                os.path.basename(image_name)
            )[0]
            + "_overlay.png"
        )

        save_path = os.path.join(
            overlay_folder,
            filename
        )

        cv2.imwrite(
            save_path,
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )

        return save_path

    # =====================================================
    # Run Complete Segmentation
    # =====================================================

    def process_image(
        self,
        image_path,
        output_folder=None
    ):

        database, overlay, image_rgb = self.run(
            image_path
        )

        if output_folder is not None:

            self.save_segmented_trees(
                database,
                image_rgb,
                output_folder
            )

            if len(database) > 0:

                overlay_path = self.save_overlay(
                    overlay,
                    database.get_all()[0].image_name,
                    output_folder
                )

                for tree in database:
                    tree.overlay_path = overlay_path

        return database, overlay, image_rgb