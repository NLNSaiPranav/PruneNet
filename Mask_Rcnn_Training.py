import os
import torch
import json
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection.mask_rcnn as maskrcnn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🔹 Detect Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Set fixed seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 🔹 Custom Mango Mask Dataset (Skips Images Without JSON Files)
class MangoMaskDataset(Dataset):
    def __init__(self, root_dir, annotation_file=None, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith((".JPG", ".jpg", ".png"))]

        self.image_ids = []
        self.annotations = {}

        valid_images = []  # Store images that have JSON files

        if annotation_file:
            with open(annotation_file, "r") as f:
                coco_data = json.load(f)

            self.image_ids = [img["id"] for img in coco_data["images"]]
            self.image_filenames = {img["id"]: img["file_name"] for img in coco_data["images"]}

            self.annotations = {img_id: [] for img_id in self.image_ids}
            for ann in coco_data["annotations"]:
                self.annotations[ann["image_id"]].append(ann)

            self.image_ids = [img_id for img_id in self.image_ids if any("segmentation" in ann for ann in self.annotations[img_id])]

        else:
            for img_file in self.image_files:
                json_file = os.path.join(root_dir, img_file.replace(".JPG", ".json").replace(".png", ".json"))
                
                if os.path.exists(json_file):  # ✅ Only add images with JSON files
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        self.annotations[img_file] = data["shapes"]
                    valid_images.append(img_file)  # Keep only valid images
                
            self.image_ids = valid_images  # ✅ Update to only use valid images

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, image_id)

        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)

        boxes, labels, masks = [], [], []
        for ann in self.annotations.get(image_id, []):
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
            elif "points" in ann:
                points = ann["points"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])

                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                poly = np.array(ann["points"], np.int32)
                cv2.fillPoly(mask, [poly], 1)
                masks.append(mask.astype(np.uint8))

                labels.append(1)

        if not boxes or not masks:
            print(f"⚠️ Skipping image {image_path}: No valid bounding boxes or masks.")
            return None  # Skip this image

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.uint8),
        }

        return image, target

# 🔹 DataLoader Collate Function
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch)) if batch else None

# 🔹 Define Paths
TRAIN_ROOT = "Mango_Tree_Data/100FPLAN"
TEST_ROOTS = ["Mango_Tree_Data/101FPLAN", "Mango_Tree_Data/102FPLAN"]

# 🔹 Create Datasets & DataLoaders
datasets = {
    "train": MangoMaskDataset(TRAIN_ROOT),
    "test": MangoMaskDataset(TEST_ROOTS[0]),
    "test2": MangoMaskDataset(TEST_ROOTS[1]),
}

dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=4, shuffle=True, collate_fn=collate_fn),
    "test": DataLoader(datasets["test"], batch_size=4, shuffle=False, collate_fn=collate_fn),
    "test2": DataLoader(datasets["test2"], batch_size=4, shuffle=False, collate_fn=collate_fn),
}

# 🔹 Load Mask R-CNN Model
def get_mask_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = maskrcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

num_classes = 2  # Background + Mango Tree
model = get_mask_model(num_classes).to(device)

# 🔹 Optimizer & Learning Rate Scheduler
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 🔹 Train Mask R-CNN Model
def train_mask_model(model, dataloaders, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Train the model
train_mask_model(model, dataloaders, num_epochs=50)

# Save Trained Model
torch.save(model.state_dict(), "mango_maskrcnn.pth")

# 🔹 Run Inference & Visualize Masks
def run_mask_inference(image_path, model, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred_boxes = predictions[0]["boxes"].cpu().numpy()
    pred_scores = predictions[0]["scores"].cpu().numpy()
    pred_masks = predictions[0]["masks"].cpu().numpy()

    image_np = np.array(image)
    for i in range(len(pred_boxes)):
        if pred_scores[i] > threshold:
            x1, y1, x2, y2 = map(int, pred_boxes[i])
            mask = (pred_masks[i, 0] > 0.5).astype(np.uint8) * 255
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            image_np[:, :, 1] = np.maximum(image_np[:, :, 1], mask)

    plt.imshow(image_np)
    plt.axis("off")
    plt.show()

test_image = "Mango_Tree_Data/100FPLAN/DJI_0390.JPG"
run_mask_inference(test_image, model)
