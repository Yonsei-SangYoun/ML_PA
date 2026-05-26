import torch
import random
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, random_split
import numpy as np

# Define data transform (You can freely modify this part to suit your needs)
# CHANGED: split into two transforms — images need normalization, masks must NOT
# be normalized because their values (0,1,2) are class labels, not pixel colors
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def apply_transforms(image, mask, augment=False):
    # Resize — NEAREST for mask to avoid blending class labels
    image = TF.resize(image, (256, 256))
    mask = TF.resize(mask, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)

    # Augmentation: random horizontal flip applied to both image AND mask together
    if augment and random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    image = normalize(TF.to_tensor(image))
    mask = torch.tensor(np.array(mask), dtype=torch.long) - 1  # (1,2,3) -> (0,1,2)
    return image, mask

# Download and load the entire dataset
data_dir = './oxford_pet_data'

# Load train split
# CHANGED: target_types='segmentation' — gives pixel-level masks instead of breed labels
train_dataset = datasets.OxfordIIITPet(
    root=data_dir,
    split='trainval',  # This includes both train and validation from original split
    download=True,
    target_types='segmentation'  # CHANGED FROM 'category'
)

# Load test split
test_dataset = datasets.OxfordIIITPet(
    root=data_dir,
    split='test',
    download=True,
    target_types='segmentation'  # CHANGED FROM 'category'
)

# Combine train and test datasets
# CHANGED: added transform logic — applies image_transform to photo, mask_transform
# to mask separately. Also converts mask labels from (1,2,3) to (0,1,2) because
# PyTorch CrossEntropyLoss expects class indices starting from 0
class FullDataset(Dataset):
    def __init__(self, dataset1, dataset2, augment=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.augment = augment

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, idx):
        if idx < self.len1:
            image, mask = self.dataset1[idx]
        else:
            image, mask = self.dataset2[idx - self.len1]

        return apply_transforms(image, mask, augment=self.augment)

# Temporary combined dataset — used only to calculate total_size for the split
full_dataset = FullDataset(train_dataset, test_dataset)

# Calculate split sizes (90% train, 10% val)
total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")

# Two FullDataset instances with different augment flags — same indices, different transforms
# Train split gets random hflip augmentation, val split does not
full_dataset_train = FullDataset(train_dataset, test_dataset, augment=True)
full_dataset_val   = FullDataset(train_dataset, test_dataset, augment=False)

torch.manual_seed(42)
train_indices, val_indices = random_split(range(total_size), [train_size, val_size])

train_set = torch.utils.data.Subset(full_dataset_train, train_indices.indices)
val_set   = torch.utils.data.Subset(full_dataset_val,   val_indices.indices)

print(f"\nFinal split:")
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

# Create data loaders
batch_size = 16  # CHANGED FROM 32 — safer for 6GB VRAM on RTX 3060

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # CHANGED FROM 4 — num_workers > 0 causes crashes on Windows
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # CHANGED FROM 4 — same reason
    pin_memory=True
)

print(f"\nData loaders created successfully!")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")