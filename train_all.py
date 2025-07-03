import os
import torch
from train import train
from utils import calculate_iou
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
from model_baseline import UNet as UNetBase
from model_deep import DeepUNet
from config import CLASS_WEIGHTS
import torch.nn as nn

def evaluate(model, dataloader, device):
    model.eval()
    return calculate_iou(model, dataloader, device)

def load_model(model_name, num_classes, device):
    if model_name == "baseline":
        model = UNetBase(num_classes)
    elif model_name == "deep":
        model = DeepUNet(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(f"checkpoints/unet_{model_name}.pth", map_location=device))
    return model.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_WEIGHTS)
    val_dataset = SegmentationDataset("data/val/images", "data/val/masks")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model_names = ["baseline", "deep"]
    results = {}

    for name in model_names:
        print(f"\n🔧 Training model: {name}")
        train(name)

        print(f"🔍 Evaluating model: {name}")
        model = load_model(name, num_classes, device)
        iou = evaluate(model, val_loader, device)
        print(f"✅ Model '{name}' IoU: {iou:.4f}")
        results[name] = iou

    # Определение лучшей модели
    best_model = max(results, key=results.get)
    print("\n🏆 Best model:", best_model)
    print("📈 IoU scores:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
