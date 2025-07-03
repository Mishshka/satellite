import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from utils import calculate_iou
from model_baseline import UNet as UNetBase
from model_deep import DeepUNet
from config import CLASS_WEIGHTS
import os

def train(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Параметры
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-4
    num_classes = len(CLASS_WEIGHTS)

    # Датасеты и DataLoader
    train_dataset = SegmentationDataset("data/train/images", "data/train/masks", augment=True)
    val_dataset = SegmentationDataset("data/val/images", "data/val/masks")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Выбор модели
    if model_name == "baseline":
        model = UNetBase(num_classes)
    elif model_name == "deep":
        model = DeepUNet(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)

    # Лосс с весами классов
    class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Валидация
        model.eval()
        with torch.no_grad():
            iou = calculate_iou(model, val_loader, device)
            print(f"Validation IoU: {iou:.4f}")

    # Сохранение модели
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/unet_{model_name}.pth")

if __name__ == "__main__":
    train("baseline")  # по умолчанию baseline, если запускать напрямую
