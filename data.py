import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2


# training augmentations
# coarsedropout cut random holes in the image so model have to use
# surrounding context to predict what was there, this help alot for
# out of distribution animal like hamster, owl, turtle
# mask_fill_value=None means we only put holes in image not in mask
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
        mask_fill_value=None,
        p=0.3,
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# for validation we dont augment, just resize and normalize
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# dataset class that combine trainval and test splits into one
# we do this so we have more data to train on
# augment flag decide which transform to use
class PetDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, augment=False):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.size_a = len(dataset_a)
        self.size_b = len(dataset_b)
        self.transform = train_transform if augment else val_transform

    def __len__(self):
        return self.size_a + self.size_b

    def __getitem__(self, idx):
        # if idx is in first dataset use that, otherwise go to second
        if idx < self.size_a:
            image, mask = self.dataset_a[idx]
        else:
            image, mask = self.dataset_b[idx - self.size_a]

        image = np.array(image.convert("RGB"))
        # mask come as values 1,2,3 but we want 0,1,2 for the model
        mask = np.array(mask, dtype=np.int64) - 1

        result = self.transform(image=image, mask=mask)
        return result["image"], result["mask"].long()


# download and load oxford pet, both splits
data_dir = "./oxford_pet_data"

trainval_data = datasets.OxfordIIITPet(
    root=data_dir, split="trainval", download=True, target_types="segmentation"
)
test_data = datasets.OxfordIIITPet(
    root=data_dir, split="test", download=True, target_types="segmentation"
)

# one version with aug for training, one without for validation
full_train = PetDataset(trainval_data, test_data, augment=True)
full_val = PetDataset(trainval_data, test_data, augment=False)

total_size = len(full_train)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")

# keep seed=42 here so the split is same every run
# this is important because for ensemble we train with different seed in main.py
# but we want val mIoU comparable across run so val set must stay same
torch.manual_seed(42)
train_idx, val_idx = random_split(range(total_size), [train_size, val_size])

train_set = Subset(full_train, train_idx.indices)
val_set = Subset(full_val, val_idx.indices)

print(f"\nFinal split:")
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")


# dataloaders
batch_size = 16

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"\nData loaders created successfully!")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")