import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import numpy as np

# Define data transform (You can freely modify this part to suit your needs)
# CHANGED: split into two transforms — images need normalization, masks must NOT
# be normalized because their values (0,1,2) are class labels, not pixel colors
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    # NEAREST interpolation is critical — mask values are class labels (0,1,2)
    # other interpolation modes blend neighboring values which corrupts the labels
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
])

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
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, idx):
        if idx < self.len1:
            image, mask = self.dataset1[idx]
        else:
            image, mask = self.dataset2[idx - self.len1]

        image = image_transform(image)
        mask = mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long) - 1  # (1,2,3) -> (0,1,2)

        return image, mask

# Combine datasets
full_dataset = FullDataset(train_dataset, test_dataset)

# Calculate split sizes (90% train, 10% val)
total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")

# Random split with fixed seed for reproducibility
torch.manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size])

print(f"\nFinal split:")
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

# Create data loaders
batch_size = 16  # CHANGED FROM 32 — safer for 6GB VRAM on RTX 3060

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # CHANGED FROM 4 — num_workers > 0 causes crashes on Windows
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0  # CHANGED FROM 4 — same reason
)

print(f"\nData loaders created successfully!")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")