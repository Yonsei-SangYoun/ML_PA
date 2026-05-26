import torch
import torch.nn as nn
import torch.optim as optim

from data import train_loader, val_loader


def double_conv(in_channels, out_channels):
    # Two conv layers back to back, each followed by BatchNorm and ReLU.
    # This is the fundamental building block of U-Net.
    # padding=1 keeps spatial dimensions unchanged after each conv.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        # TODO: define encoder (contracting path)
        # Each level: double_conv to learn features, then MaxPool2d to halve spatial size.
        # Filter count doubles at each level (64->128->256->512) to learn richer features.
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # TODO: define bottleneck (e.g., 512 -> 1024)
        self.bottleneck = double_conv(512, 1024)

        # TODO: define decoder (expanding path)
        # ConvTranspose2d doubles spatial size (reverse of pooling).
        # After upsampling we concat the skip connection from the same encoder level,
        # so in_channels doubles (e.g. 512 up + 512 skip = 1024 into dec4).
        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4  = double_conv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3  = double_conv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2  = double_conv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1  = double_conv(128, 64)

        # TODO: define final 1x1 conv (out_channels = num_classes)
        # Maps 64 channels -> num_classes channels (one score per class per pixel)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encoder forward, store feature maps for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # TODO: bottleneck forward
        b = self.bottleneck(self.pool(e4))

        # TODO: decoder forward, concat with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # TODO: return segmentation logits via final 1x1 conv
        return self.final_conv(d1)


class DiceLoss(nn.Module):
    """Dice loss — measures mask overlap. Perfect overlap = 0, no overlap = 1."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        dice = sum(
            (2 * (probs[:, c] * targets_one_hot[:, c]).sum() + self.smooth) /
            (probs[:, c].sum() + targets_one_hot[:, c].sum() + self.smooth)
            for c in range(num_classes)
        )
        return 1 - dice / num_classes


class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    def __init__(self):
        super().__init__()
        # TODO: define the loss to use
        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss
        # Using CrossEntropy + Dice: CE handles per-pixel classification,
        # Dice directly optimises mask overlap which is closer to mIoU
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: compute and return loss from logits and targets
        return self.cross_entropy(logits, targets) + self.dice(logits, targets)


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        # TODO: iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss
        total_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader) -> tuple:
        self.model.eval()
        # TODO: run forward only and compute loss / metrics (IoU, Dice, etc.)
        total_loss, total_miou = 0.0, 0.0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            total_loss += self.criterion(logits, masks).item()
            total_miou += self._compute_miou(logits.argmax(dim=1), masks)
        return total_loss / len(loader), total_miou / len(loader)

    def _compute_miou(self, preds, targets, num_classes=3) -> float:
        ious = []
        for c in range(num_classes):
            intersection = ((preds == c) & (targets == c)).sum().item()
            union = ((preds == c) | (targets == c)).sum().item()
            if union > 0:
                ious.append(intersection / union)
        return sum(ious) / len(ious) if ious else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 50           # CHANGED FROM 10 — 10 epochs is too few to learn
    learning_rate = 1e-4
    num_classes = 3           # CHANGED FROM 1 — we have 3 classes (fg, bg, boundary)

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    trainer = Trainer(model, criterion, optimizer, device)

    best_miou = 0.0
    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_miou = trainer.validate(val_loader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}")

        # TODO: save best model checkpoint / visualize predictions / report metrics
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Best model saved (mIoU={best_miou:.4f})")


if __name__ == "__main__":
    main()