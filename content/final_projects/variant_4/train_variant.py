import os
import cv2
import copy
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "/content/dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK_DIR = "/content/dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation"
VAL_IMG_DIR = "/content/dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK_DIR = "/content/dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation"

IMAGE_SIZE = (320, 320)
CLASS_VALUES = [0, 100, 200, 300, 500, 550, 800, 7100, 10000]
VALUE_TO_CLASS = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 800: 6, 7100: 7, 10000: 8}
NUM_CLASSES = 9

CONFIG = {'name': 'variant_4', 'seed': 99, 'encoder': 'mobilenet_v2', 'lr': 0.0007, 'batch_size': 8, 'epochs': 9, 'optimizer': 'adamw'}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        mapped_mask = np.zeros_like(mask, dtype=np.int64)
        for orig_val, cls_idx in VALUE_TO_CLASS.items():
            mapped_mask[mask == orig_val] = cls_idx

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mapped_mask, dtype=torch.long)

def build_model(encoder_name="resnet34"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )

def calculate_iou(preds, masks, num_classes):
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / (union + 1e-8))

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))

ce_loss = nn.CrossEntropyLoss()
dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

def combined_loss(outputs, masks):
    return 0.7 * ce_loss(outputs, masks) + 0.3 * dice_loss(outputs, masks)

def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    total_iou = 0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        iou = calculate_iou(preds, masks, num_classes)

        total_loss += loss.item()
        total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device, num_classes):
    model.eval()
    total_loss = 0
    total_iou = 0

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        iou = calculate_iou(preds, masks, num_classes)

        total_loss += loss.item()
        total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)

if __name__ == "__main__":
    print("Variant config:", CONFIG)
    print("This script matches the training setup used for this variant.")