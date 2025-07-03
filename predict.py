import torch
import numpy as np
import cv2
from model_baseline import UNet
from config import COLOR_ENCODING
from utils import rgb_to_class, class_to_rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(COLOR_ENCODING)


def load_model(model_path='checkpoints/unet_baseline.pth'):
    model = UNet(n_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(image, model, img_size=256):
    original_size = (image.shape[1], image.shape[0])

    # Предобработка
    image_resized = cv2.resize(image, (img_size, img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device)

    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Преобразование в RGB и resize обратно
    pred_mask_rgb = class_to_rgb(pred_mask, COLOR_ENCODING)  # Передаем COLOR_ENCODING
    pred_mask_rgb = cv2.resize(pred_mask_rgb, original_size, interpolation=cv2.INTER_NEAREST)

    return pred_mask_rgb