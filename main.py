import argparse
import random

import numpy as np
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
    """U-Net with pretrained ResNet34 encoder (ImageNet weights)."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        try:
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            backbone = models.resnet34(pretrained=True)

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool     = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)
        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64 + 64, 64)
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = double_conv(32, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
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
    """CrossEntropy + Dice. Optional class weights for CE.

    Why class weights:
        Class 2 (boundary) is much smaller than classes 0/1, and it's also
        the hardest because it lives in thin 1-2 pixel strips. Without weighting,
        the model can score well on val mIoU just by getting foreground+background
        right and ignoring boundaries. Boosting class 2's CE weight pushes the
        model to learn boundaries properly, which helps mIoU and helps OOD
        animals where boundary detection is even harder.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice          = DiceLoss()

    def forward(self, logits, targets):
        return self.cross_entropy(logits, targets) + self.dice(logits, targets)


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device    = device

    def train_one_epoch(self, loader):
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
    def validate(self, loader, num_classes=3):
        self.model.eval()
        total_loss = 0.0
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed. Use different values per run for ensemble diversity.")
    p.add_argument("--epochs",    type=int, default=80,
                   help="Total epochs. Cosine LR decays over this whole range, so longer = smoother decay.")
    p.add_argument("--save_path", default="best_model.pth",
                   help="Where to save the best model. Use different paths per run, e.g. best_seed42.pth")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed everything AFTER data.py has been imported.
    # data.py uses its own fixed seed=42 for the train/val split (we want that
    # constant across runs so val mIoU is comparable). Setting the seed here
    # affects model init, dataloader shuffle order, and aug RNG — which is
    # what we want to vary across ensemble members.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    print(f"Config: seed={args.seed} epochs={args.epochs} save_path={args.save_path} device={device}")

    model = ResNetUNet(num_classes=num_classes).to(device)

    # Bundle B: class weights — boundary class gets 2x.
    # [foreground, background, boundary]
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
    criterion     = SegmentationLoss(class_weights=class_weights)

    # Differential LR: pretrained encoder gets 10x lower LR than random-init decoder
    encoder_params = (
        list(model.encoder0.parameters()) +
        list(model.encoder1.parameters()) +
        list(model.encoder2.parameters()) +
        list(model.encoder3.parameters()) +
        list(model.encoder4.parameters())
    )
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-3},
    ], weight_decay=1e-5)

    # Bundle A: Cosine annealing instead of ReduceLROnPlateau.
    # LR starts at the initial values above and smoothly decays to eta_min over
    # `epochs` epochs. Smoother than ReduceLROnPlateau (no plateau detection
    # needed), and the long tail of small LR at the end refines the model
    # without disrupting it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    trainer   = Trainer(model, criterion, optimizer, device)
    best_miou = 0.0

    for epoch in range(args.epochs):
        train_loss         = trainer.train_one_epoch(train_loader)
        val_loss, val_miou = trainer.validate(val_loader)
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_mIoU={val_miou:.4f} lr_enc={lrs[0]:.2e} lr_dec={lrs[1]:.2e}")

        # Cosine: step every epoch, no metric argument needed
        scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Best model saved (mIoU={best_miou:.4f}) to {args.save_path}")

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f} (saved to {args.save_path})")


if __name__ == "__main__":
    main()