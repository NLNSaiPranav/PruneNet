# PruneNet [[Playground]](https://prunenetdemo-bbwzgu5f93k2ayszlswfbh.streamlit.app/)**

An end-to-end UAV-based mango tree pruning decision support system using deep learning, canopy analysis, geospatial localization, and autonomous path planning.

Authors : Nidugondi L N Sai Pranav, Dr. Sandeep Manjanna

# Abstract
Pruning trees plays a crucial role in maintaining canopy health, improving sunlight penetration, and maximizing orchard productivity. As precision agriculture continues to advance, there is a growing opportunity to complement traditional human expertise with intelligent, data-driven approaches that enable more consistent, efficient, and scalable pruning decisions. In this paper, we present PruneNet, an end-to-end computer vision framework that analyzes aerial imagery to identify trees that require pruning. The proposed pipeline starts with an instance-level tree canopy segmentation fine-tuned on custom built orchard dataset. Each segmented canopy is then analyzed using Intensity-Weighted Canopy Projection (IWCP), which estimates canopy transparency from RGB images by using the intensity variations as a structural proxy for the canopy density. Expert-in-the-loop calibration grounds transparency scores in agricultural expertise, producing pruning recommendations that are both explainable and practically actionable. Identified trees are subsequently georeferenced onto an orthomosaic of the orchard, and an optimized traversal path is generated to support efficient robotic inspection and intervention. Experimental evaluations demonstrate tree segmentation IoU of 86.53%, with 95.07% of the predicted masks exceeding IoU > 0.5, and reliable canopy transparency estimations with an accuracy of 86.67%. We also present empirical evaluations to illustrate the generalization across multiple orchard datasets,
highlighting the potential of PruneNet as a practical decision support pipeline for precision orchard management.


## Pipeline

![Pipeline](Images/3D.png)

## Video Demonstration

[![Watch the Demo](Images/youtube.png)](https://youtu.be/c6rzDiAI3Ic)

# Setup

PruneNet has been developed and tested on **Ubuntu 22.04 LTS** using **Python 3.12**. We recommend using a dedicated Conda environment for dependency isolation.

---

## Clone Repository

```bash
git clone https://github.com/NLNSaiPranav/PruneNet.git
```

---

## Create Conda Environment

```bash
conda create -n prunenet python=3.12 -y

conda activate prunenet
```

---

## Install Dependencies

```bash
cd PruneNet

pip install -r requirements.txt
```

---

## Download Model Weights

Download the trained **Mask R-CNN** weights and place them in the following directory.

```text
PruneNet/
└── models/
    └── maskrcnn/
        └── mango_maskrcnn.pth
```

> **Note:** The trained model weights will be released after publication of the paper.

---

## Prepare Dataset

Organize the dataset using the following directory structure.

```text
PruneNet/
└── data/
    └── input/
        ├── rgb_images/
        │   ├── IMG_0001.JPG
        │   ├── IMG_0002.JPG
        │   └── ...
        │
        └── orthomosaic/
            └── Orthomosaic.tif
```

A small sample dataset is included to demonstrate the complete pipeline. The full UAV dataset used in our experiments is not included due to its size. Researchers interested in obtaining the complete dataset and the orthomosaic stitched image for academic purposes are to contact the authors.

---

## Configure the Pipeline

Before running the pipeline, update the following paths in **main.py**.

```python
MODEL = "models/maskrcnn/mango_maskrcnn.pth"

IMAGE_FOLDER = "data/input/rgb_images"

ORTHOMOSAIC = "data/input/orthomosaic/Orthomosaic.tif"

OUTPUT_FOLDER = "outputs"
```

---

## Run the Pipeline

```bash
python main.py
```

---

## Generated Outputs

After successful execution, the following outputs are generated automatically.

```text
outputs/
│
├── segmentation/
│
├── iwcp/
│
├── localization/
│
├── planner/
│
├── results.xlsx
│
└── planner/
    └── elkai_route.xlsx
```

---

If you have any questions regarding the paper, codebase, dataset, or would like to collaborate, please feel free to reach out at.

**Nidugondi L. N. Sai Pranav**  
📧 Email: `nidugondi.pranav@plaksha.edu.in`

