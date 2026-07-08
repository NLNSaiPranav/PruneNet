"""
pipeline.py

Runs the complete PruneNet pipeline.

RGB Images
    │
    ▼
Mask R-CNN
    │
    ▼
IWCP
    │
    ▼
Master Tree Database
"""

import os

from src.modules.segmentation import MaskRCNNSegmenter
from src.modules.iwcp import IWCPAnalyzer
from src.core.database import TreeDatabase
from src.modules.localization import LocalizationModule
from src.modules.planner import OrchardPlanner
class PruneNetPipeline:

    def __init__(
        self,
        model_path,
        orthomosaic_path,
        image_dataset,
        pruning_threshold=8.3
    ):

        self.segmenter = MaskRCNNSegmenter(
            model_path=model_path
        )

        self.iwcp = IWCPAnalyzer(
            pruning_threshold=pruning_threshold
        )

        self.localization = LocalizationModule(
            orthomosaic_path=orthomosaic_path,
            image_dataset=image_dataset
        )
        self.localization = LocalizationModule(
            orthomosaic_path=orthomosaic_path,
            image_dataset=image_dataset
        )
        self.planner = OrchardPlanner()
        self.orthomosaic_path = orthomosaic_path
    # =====================================================
    # Process One Image
    # =====================================================

    def process_image(
        self,
        image_path,
        output_folder
    ):

        print(f"\nProcessing {os.path.basename(image_path)}")

        database, overlay, image_rgb = self.segmenter.process_image(
            image_path,
            output_folder
        )

        print(f"Detected Trees : {len(database)}")

        database = self.iwcp.process_image(
            database,
            image_rgb,
            output_folder
        )

        print("IWCP complete")

        return database

    # =====================================================
    # Process Entire Folder
    # =====================================================

    def process_folder(
        self,
        image_folder,
        output_folder
    ):

        valid_ext = (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        )

        image_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(valid_ext)
        ])

        master_database = TreeDatabase()

        for image_path in image_files:

            database = self.process_image(
                image_path,
                output_folder
            )

            for tree in database:
                master_database.add_tree(tree)

        print("\nExporting IWCP results...")

        self.iwcp.export_results(
            master_database,
            output_folder
        )

        print("\nRunning localization...")

        master_database = self.localization.run(
            master_database,
            output_folder
        )

        print("\nRunning planner...")

        master_database = self.planner.run(

            database=master_database,

            orthomosaic=self.localization.full_image_bgr,

            output_folder=output_folder

        )

        print(f"Total Trees : {len(master_database)}")

        print(
            f"Trees to Prune : {master_database.total_pruned()}"
        )

        print("\nPipeline Complete.")

        return master_database