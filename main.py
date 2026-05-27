import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from data import train_loader, val_loader


# basic double conv block, two 3x3 convs with bn and relu
# this is the standard unet decoder block
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# unet but with resnet34 pretrained on imagenet as the encoder
# decoder is normal unet style with skip connections
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # load resnet34 with imagenet weights
        # try new api first, fallback to old one if torchvision is older
        try:
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            backbone = models.resnet34(pretrained=True)

        # split resnet into stages for the encoder
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # decoder going back up
        # at each step we upsample then concat with the matching encoder feature
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = double_conv(32, 32)

        # final 1x1 conv to get the class scores
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder path, save each stage for skip connection later
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # decoder path, concat with skip features at each stage
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.dec0(self.up0(d1))
        return self.final_conv(d0)


# dice loss is useful for segmentation because it handle class imbalance
# better than plain crossentropy
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        # convert target into one hot encoding
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        dice = sum(
            (2 * (probs[:, c] * targets_one_hot[:, c]).sum() + self.smooth) /
            (probs[:, c].sum() + targets_one_hot[:, c].sum() + self.smooth)
            for c in range(num_classes)
        )
        return 1 - dice / num_classes


# combined loss, crossentropy + dice
# class_weights boost the boundary class because its much smaller
# without weighting the model can get high miou just by predicting
# fg and bg correctly, but boundary is the hardest part and we want
# the model to actually learn it
class SegmentationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


# run one epoch of training and return average loss
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# evaluate on val set and compute miou
# we accumulate intersection and union over the whole set then divide at the end
# this is more correct than averaging per batch miou
@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = 0.0
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        total_loss += criterion(logits, masks).item()

        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            inter[c] += ((preds == c) & (masks == c)).sum()
            union[c] += ((preds == c) | (masks == c)).sum()

    iou = inter / (union + 1e-9)
    miou = iou.mean().item()
    return total_loss / len(loader), miou


def parse_args():
    parser = argparse.ArgumentParser()
    # seed change the model init and dataloader order, use different value for ensemble
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--save_path", default="best_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    # seed everything for reproducibility
    # note data.py already use seed=42 for the train/val split
    # so changing seed here only affect model init and aug rng order
    # which is what we want for ensemble diversity
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    print(f"Config: seed={args.seed} epochs={args.epochs} save_path={args.save_path} device={device}")

    model = ResNetUNet(num_classes=num_classes).to(device)

    # weights are [foreground, background, boundary], boundary class get 2x
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
    criterion = SegmentationLoss(class_weights=class_weights)

    # differential learning rate
    # encoder is pretrained so it dont need to change much, give it lower lr
    # decoder is random init so it need higher lr to learn faster
    encoder_params = (
        list(model.encoder0.parameters())
        + list(model.encoder1.parameters())
        + list(model.encoder2.parameters())
        + list(model.encoder3.parameters())
        + list(model.encoder4.parameters())
    )
    decoder_params = [
        p for p in model.parameters()
        if not any(p is ep for ep in encoder_params)
    ]

    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": 1e-4},
            {"params": decoder_params, "lr": 1e-3},
        ],
        weight_decay=1e-5,
    )

    # cosine annealing schedule
    # lr smoothly decay from initial value down to eta_min over the epochs
    # the slow lr at the end help the model settle into a good minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_miou = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device)

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} lr_enc={lrs[0]:.2e} lr_dec={lrs[1]:.2e}"
        )

        # step scheduler every epoch
        scheduler.step()

        # save best model so far
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.save_path)
            print(f"  >> Best model saved (mIoU={best_miou:.4f})")

    print(f"\nDone. Best val mIoU: {best_miou:.4f}, saved to {args.save_path}")


if __name__ == "__main__":
    main()