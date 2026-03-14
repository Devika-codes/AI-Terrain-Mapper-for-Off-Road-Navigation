##AI Terrain Mapper for Off-Road Navigation

An intelligent image-processing tool that breaks down off-road scenes pixel-by-pixel. By classifying everything from vegetation to obstacles, this project provides the "eyes" for self-driving vehicles, ensuring they can distinguish between a clear path and a dangerous barrier.
---

## Overview

This project trains a segmentation model to classify every pixel in an off-road image into one of 9 terrain classes such as sky, ground, vegetation, and obstacles. It uses a combined Cross-Entropy and Dice loss function and achieves a best validation IoU of **0.5309** over 9 training epochs.

---

## Project Structure

```
variant_4_project/
├── content/final_projects/variant_4/
│   ├── train_variant.py          # Training pipeline
│   ├── inference.py              # Run predictions on new images
│   ├── variant_4_best.pth        # Best saved model weights
│   ├── config.json               # Hyperparameter configuration
│   ├── results.txt               # Best IoU summary
│   ├── variant_4_loss.png        # Training/validation loss curve
│   ├── variant_4_iou.png         # Training/validation IoU curve
│   ├── sample_image.png          # Sample input image
│   ├── sample_original_mask.png  # Ground truth mask
│   ├── sample_mapped_mask.png    # Predicted segmentation mask
│   └── README.md
```

---

## Model Configuration

| Parameter   | Value         |
|-------------|---------------|
| Encoder     | MobileNetV2   |
| Weights     | ImageNet      |
| Optimizer   | AdamW         |
| Learning Rate | 0.0007      |
| Batch Size  | 8             |
| Epochs      | 9             |
| Seed        | 99            |
| Image Size  | 320 x 320     |
| Classes     | 9             |

---

## Classes

| Class Index | Pixel Value |
|-------------|-------------|
| 0           | 0           |
| 1           | 100         |
| 2           | 200         |
| 3           | 300         |
| 4           | 500         |
| 5           | 550         |
| 6           | 800         |
| 7           | 7100        |
| 8           | 10000       |

---

## Results

| Metric              | Value  |
|---------------------|--------|
| Best Validation IoU | 0.5309 |
| Final Train Loss    | ~0.352 |
| Final Val Loss      | ~0.352 |

---

## Installation

```bash
pip install torch torchvision segmentation-models-pytorch opencv-python numpy matplotlib tqdm
```

---

## Loss Function

A combined loss is used for training:

```
Loss = 0.7 × CrossEntropyLoss + 0.3 × DiceLoss
```

---

## Requirements

- Python 3.8+
- PyTorch
- segmentation-models-pytorch
- OpenCV
- NumPy
- Matplotlib
- tqdm
