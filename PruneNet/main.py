from pipeline import PruneNetPipeline

MODEL = "models/maskrcnn/mango_maskrcnn.pth"

IMAGE_FOLDER = "data/input/rgb_images"

ORTHOMOSAIC = "data/input/orthomosaic/Orthomosaic.rgb_modified.tif"

OUTPUT_FOLDER = "outputs"

pipeline = PruneNetPipeline(

    model_path=MODEL,

    orthomosaic_path=ORTHOMOSAIC,

    image_dataset=IMAGE_FOLDER,

    pruning_threshold=8.3

)

pipeline.process_folder(

    IMAGE_FOLDER,

    OUTPUT_FOLDER

)