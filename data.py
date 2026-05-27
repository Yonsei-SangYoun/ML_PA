import random
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation pipelines ────────────────────────────────────────────────────
#
# Why CoarseDropout was added (Bundle B):
#   Cuts random rectangular holes in the input image. The model has to predict
#   the correct mask for those pixels using *surrounding context*, since the
#   image content is gone. This forces the model to rely on shape/context
#   instead of memorizing local fur textures specific to cats/dogs, which
#   helps a lot for out-of-distribution animals (hamsters, owls, turtles).
#   Important: mask_fill_value=None means the ground-truth mask is NOT modified
#   — only the image gets the holes. This is the standard "Cutout" recipe.

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
        p=0.7, border_mode=0
    ),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussNoise(var_limit=(10, 50), p=1.0),
    ], p=0.3),
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        min_holes=1, min_height=16, min_width=16,
        fill_value=0,
        mask_fill_value=None,   # leave ground-truth mask untouched
        p=0.3
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class FullDataset(Dataset):
    def __init__(self, dataset1, dataset2, augment=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1     = len(dataset1)
        self.len2     = len(dataset2)
        self.transform = train_transform if augment else val_transform

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, idx):
        if idx < self.len1:
            image, mask = self.dataset1[idx]
        else:
            image, mask = self.dataset2[idx - self.len1]

        image = np.array(image.convert("RGB"))
        mask  = np.array(mask, dtype=np.int64) - 1   # (1,2,3) -> (0,1,2)

        out   = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


# ── Load Oxford Pet splits ────────────────────────────────────────────────────

data_dir = './oxford_pet_data'

train_dataset = datasets.OxfordIIITPet(
    root=data_dir, split='trainval', download=True, target_types='segmentation'
)
test_dataset = datasets.OxfordIIITPet(
    root=data_dir, split='test', download=True, target_types='segmentation'
)

full_dataset_train = FullDataset(train_dataset, test_dataset, augment=True)
full_dataset_val   = FullDataset(train_dataset, test_dataset, augment=False)

total_size = len(full_dataset_train)
train_size = int(0.9 * total_size)
val_size   = total_size - train_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")

# KEEP this seed=42 fixed across all training runs.
# We want the same train/val split every time so val mIoU is comparable
# between seed=42, seed=123, seed=7 runs for ensembling.
torch.manual_seed(42)
train_indices, val_indices = random_split(range(total_size), [train_size, val_size])

train_set = torch.utils.data.Subset(full_dataset_train, train_indices.indices)
val_set   = torch.utils.data.Subset(full_dataset_val,   val_indices.indices)

print(f"\nFinal split:")
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

# ── Data loaders ──────────────────────────────────────────────────────────────

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"\nData loaders created successfully!")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")