import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


# same double conv block as main.py
# model architecture have to match exactly so we can load the weights
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# same resnet34 unet from main.py, need to be identical to load checkpoint
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        try:
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            backbone = models.resnet34(pretrained=True)

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

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


# imagenet normalization, same as training
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# predict mask for one image with hflip tta
# we only do hflip because model was trained with hflip aug so its invariant to it
# vflip and 180 rotation were not in training so the model give bad prediction on them
# averaging good prediction with bad one just drag the score down which is what happen before
@torch.no_grad()
def predict(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # helper to resize, normalize and move to gpu
    def to_tensor(pil_image):
        resized = TF.resize(pil_image, (256, 256))
        tensor = TF.to_tensor(resized)
        return normalize(tensor).unsqueeze(0).to(device)

    # original and horizontally flipped version
    inp = to_tensor(img)
    inp_flipped = to_tensor(TF.hflip(img))

    # get softmax probabilities for both
    probs_orig = torch.softmax(model(inp), dim=1)
    probs_flipped = torch.softmax(model(inp_flipped), dim=1)

    # the flipped probs are still flipped so we unflip them along width axis
    probs_flipped = torch.flip(probs_flipped, dims=[3])

    # average the two prob maps then argmax to get final class
    # averaging probs is better than averaging predictions because it keep confidence info
    avg_probs = (probs_orig + probs_flipped) / 2.0
    pred = avg_probs.argmax(dim=1)

    # convert back to pil and resize to original size with nearest neighbor
    # nearest is important because we have integer class labels not continuous value
    pred_pil = TF.to_pil_image(pred.squeeze(0).byte())
    pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)

    return np.array(pred_pil, dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pth")
    parser.add_argument("--test_dir", default="kaggle/test_images")
    parser.add_argument("--pred_dir", default="kaggle/predictions")
    parser.add_argument("--sample", default="kaggle/sample_submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # build model and load trained weights
    model = ResNetUNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    test_dir = Path(args.test_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # gather all image files in the test folder
    image_paths = sorted(test_dir.glob("*"))
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    print(f"Found {len(image_paths)} test images.")

    # run prediction for each image and save as npy
    for i, img_path in enumerate(image_paths):
        pred = predict(model, img_path, device)
        npy_path = pred_dir / f"{img_path.stem}.npy"
        np.save(npy_path, pred)

        # print progress every 50 images so we know its still running
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name} -> {npy_path.name}")

    print(f"\nDone. Predictions saved to: {pred_dir}")
    print("\nNow run:")
    print(
        f"  python make_submission.py --pred_dir {args.pred_dir} "
        f"--sample {args.sample} --out kaggle/submission.csv"
    )


if __name__ == "__main__":
    main()