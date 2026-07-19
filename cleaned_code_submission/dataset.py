import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class BreastUltrasoundDataset(Dataset):

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        if split == 'train':
            self.img_dir = os.path.join(root_dir, split, 'img', 'ori')
            self.gt_dir = os.path.join(root_dir, split, 'gt', 'ori')
        else:
            self.img_dir = os.path.join(root_dir, split, 'img')
            self.gt_dir = os.path.join(root_dir, split, 'gt')
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.gt_dir))
        if len(self.images) != len(self.masks):
            print(f'Warning: Mismatch in number of images ({len(self.images)}) and masks ({len(self.masks)}) for split {split}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        mask_name = os.path.join(self.gt_dir, self.masks[idx])
        image = cv2.imread(img_name)
        if image is None:
            raise ValueError(f'Failed to load image: {img_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'Failed to load mask: {mask_name}')
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if not torch.is_tensor(image):
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return (image, mask)
