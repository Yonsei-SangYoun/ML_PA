"""Run inference on Kaggle test images with hflip-only TTA.

Why hflip-only TTA:
    The model was trained with HorizontalFlip augmentation, so it's invariant
    to hflips at test time. But vflip and 180° rotation were NEVER in training,
    so the model gives unreliable predictions on upside-down images.
    Averaging good predictions with bad ones drags the final score down —
    which is why the previous 4-way TTA hurt instead of helped.

Usage:
    python inference.py \
        --model_path  best_model.pth \
        --test_dir    kaggle/test_images \
        --pred_dir    kaggle/predictions \
        --sample      kaggle/sample_submission.csv

Then generate submission CSV:
    python make_submission.py \
        --pred_dir kaggle/predictions \
        --sample   kaggle/sample_submission.csv \
        --out      kaggle/submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms


# ── Model definition (must match the architecture used during training) ──────

import torchvision.models as models

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
    def __init__(self, num_classes=3):
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


# ── Inference ────────────────────────────────────────────────────────────────

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


@torch.no_grad()
def predict(model, image_path, device):
    """Return a (H, W) uint8 numpy array with values in {0, 1, 2}
    at the ORIGINAL image resolution, using hflip-only TTA.
    
    Process:
      1. Run model on original image -> probability map p_orig
      2. Run model on hflipped image -> p_h, then unflip p_h
      3. Average p_orig and p_h, then argmax
    Averaging probabilities (not predictions) keeps confidence info intact.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size   # PIL gives (width, height)

    def to_tensor(pil_img):
        return normalize(TF.to_tensor(TF.resize(pil_img, (256, 256)))).unsqueeze(0).to(device)

    inp   = to_tensor(img)
    inp_h = to_tensor(TF.hflip(img))

    # Forward + softmax for both versions
    p_orig = torch.softmax(model(inp),   dim=1)
    p_h    = torch.softmax(model(inp_h), dim=1)

    # Undo hflip on the flipped-image probabilities (flip back along width axis)
    p_h = torch.flip(p_h, dims=[3])

    # Average probability maps, then argmax
    avg_probs = (p_orig + p_h) / 2.0
    pred = avg_probs.argmax(dim=1)    # (1, 256, 256)

    # Resize back to original resolution using nearest neighbour
    pred_img = TF.to_pil_image(pred.squeeze(0).byte())
    pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)

    return np.array(pred_img, dtype=np.uint8)   # (H, W), values 0/1/2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="best_model.pth")
    p.add_argument("--test_dir",   default="kaggle/test_images")
    p.add_argument("--pred_dir",   default="kaggle/predictions")
    p.add_argument("--sample",     default="kaggle/sample_submission.csv")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ResNetUNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    test_dir = Path(args.test_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(test_dir.glob("*"))
    image_paths = [p for p in image_paths
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Found {len(image_paths)} test images.")

    for i, img_path in enumerate(image_paths):
        pred = predict(model, img_path, device)
        npy_path = pred_dir / f"{img_path.stem}.npy"
        np.save(npy_path, pred)
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name} -> {npy_path.name}")

    print(f"\nDone. Predictions saved to: {pred_dir}")
    print("\nNow run:")
    print(f"  python make_submission.py --pred_dir {args.pred_dir} "
          f"--sample {args.sample} --out kaggle/submission.csv")


if __name__ == "__main__":
    main()