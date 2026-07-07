# 🌳 PruneNet

An end-to-end UAV-based mango tree pruning decision support system using deep learning, pseudo-depth canopy analysis, geospatial localization, and autonomous path planning.
## Overview

PruneNet is an end-to-end precision agriculture framework designed for automated mango orchard analysis from UAV imagery.

The framework performs:

- Tree instance segmentation using Mask R-CNN
- Canopy transparency estimation using Intensity-Weighted Canopy Projection (IWCP)
- Automatic pruning recommendation
- Tree localization on an orthomosaic
- Voronoi-based navigation graph construction
- Elkai Traveling Salesman path optimization

Unlike existing approaches that stop at tree detection, PruneNet generates a complete pruning workflow from aerial imagery to an optimized traversal path for field workers or autonomous robots.

## Pipeline

![Pipeline](Images/3D.png)

## 🎥 Video Demonstration

[![PruneNet Demo](README_images/demo_thumbnail.png)](https://youtu.be/c6rzDiAI3Ic)
