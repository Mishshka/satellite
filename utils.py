import numpy as np
import torch
from config import COLOR_ENCODING

def calculate_iou(model, dataloader, device, num_classes=11):
    model.eval()
    ious = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for pred, true in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                iou = calculate_single_iou(pred, true, num_classes)
                ious.append(iou)

    return np.nanmean(ious)


def calculate_single_iou(pred_mask, target_mask, num_classes=11):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (target_mask == cls)

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            ious.append(np.nan)  # Пропускаем класс, если он отсутствует
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def rgb_to_class(mask_rgb, COLOR_ENCODING):
    """
    Преобразует RGB-маску (H, W, 3) в маску классов (H, W)
    """
    class_mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

    for class_idx, (_, color) in enumerate(COLOR_ENCODING.items()):
        matches = np.all(mask_rgb == color, axis=-1)
        class_mask[matches] = class_idx

    return class_mask


def class_to_rgb(class_mask, COLOR_ENCODING):
    """
    Преобразует маску классов (H, W) в RGB-маску (H, W, 3)
    """
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, (_, color) in enumerate(COLOR_ENCODING.items()):
        rgb_mask[class_mask == class_idx] = color

    return rgb_mask
