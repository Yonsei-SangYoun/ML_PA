import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from data import train_loader, val_loader


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ResNetUNet(nn.Module):
    """U-Net with pretrained ResNet34 encoder (ImageNet weights).
    
    ResNet34 feature map sizes for 256x256 input:
        encoder0 (conv1): 128x128, 64ch
        pool:              64x64
        encoder1 (layer1): 64x64,  64ch
        encoder2 (layer2): 32x32, 128ch
        encoder3 (layer3): 16x16, 256ch
        encoder4 (layer4):  8x8,  512ch
    
    Decoder upsample path: 8→16→32→64→128→256
    Skip connections from encoder0, encoder1, encoder2, encoder3.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # --- Encoder: pretrained ResNet34 ---
        try:
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            # older torchvision
            backbone = models.resnet34(pretrained=True)

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, H/2
        self.pool     = backbone.maxpool                                              # H/4
        self.encoder1 = backbone.layer1   # 64ch,  H/4
        self.encoder2 = backbone.layer2   # 128ch, H/8
        self.encoder3 = backbone.layer3   # 256ch, H/16
        self.encoder4 = backbone.layer4   # 512ch, H/32

        # --- Decoder ---
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256, 256)   # up4 + encoder3

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)   # up3 + encoder2

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)      # up2 + encoder1

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64 + 64, 64)      # up1 + encoder0

        # one more upsample to recover the factor-of-2 from conv1 stride
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = double_conv(32, 32)            # no skip here

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.encoder0(x)           # 64ch, H/2
        e1 = self.encoder1(self.pool(e0))  # 64ch, H/4
        e2 = self.encoder2(e1)           # 128ch, H/8
        e3 = self.encoder3(e2)           # 256ch, H/16
        e4 = self.encoder4(e3)           # 512ch, H/32

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.dec0(self.up0(d1))

        return self.final_conv(d0)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
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
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets) + self.dice(logits, targets)


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device    = device

    def train_one_epoch(self, loader) -> float:
        self.model.train()
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
    def validate(self, loader, num_classes=3) -> tuple:
        self.model.eval()
        total_loss = 0.0
        # Accumulate intersection and union across the WHOLE val set, not per-batch.
        # Per-batch averaging inflates mIoU when a class is absent from some batches.
        inter = torch.zeros(num_classes, device=self.device)
        union = torch.zeros(num_classes, device=self.device)
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            total_loss += self.criterion(logits, masks).item()
            preds = logits.argmax(dim=1)
            for c in range(num_classes):
                inter[c] += ((preds == c) & (masks == c)).sum()
                union[c] += ((preds == c) | (masks == c)).sum()
        iou  = inter / (union + 1e-9)
        miou = iou.mean().item()
        return total_loss / len(loader), miou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs  = 50
    num_classes = 3

    model     = ResNetUNet(num_classes=num_classes).to(device)
    criterion = SegmentationLoss()

    # Lower LR for pretrained encoder, higher for randomly-init decoder
    encoder_params = list(model.encoder0.parameters()) + \
                     list(model.encoder1.parameters()) + \
                     list(model.encoder2.parameters()) + \
                     list(model.encoder3.parameters()) + \
                     list(model.encoder4.parameters())
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-3},
    ], weight_decay=1e-5)

    # Reduce LR when val mIoU plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )

    trainer    = Trainer(model, criterion, optimizer, device)
    best_miou  = 0.0

    for epoch in range(num_epochs):
        train_loss          = trainer.train_one_epoch(train_loader)
        val_loss, val_miou  = trainer.validate(val_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}")

        scheduler.step(val_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Best model saved (mIoU={best_miou:.4f})")


if __name__ == "__main__":
    main()