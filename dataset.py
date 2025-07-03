import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from config import COLOR_ENCODING

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.image_size = image_size

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        self.image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20)
        ])

        self.mask_transform = T.Compose([
            T.Resize(image_size, interpolation=Image.NEAREST)
        ])

        self.class_map = {v: i for i, (k, v) in enumerate(COLOR_ENCODING.items())}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("RGB")

        if self.augment:
            seed = np.random.randint(0, 10000)
            T.RandomApply([self.augment_transform], p=0.8)
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            mask = self.augment_transform(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = np.array(mask)

        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)

        for rgb, cls_idx in self.class_map.items():
            matches = np.all(mask == rgb, axis=-1)
            label_mask[matches] = cls_idx

        return image, torch.tensor(label_mask, dtype=torch.long)
