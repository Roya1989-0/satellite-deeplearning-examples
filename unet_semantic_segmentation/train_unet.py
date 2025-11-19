# train_unet.py
"""
Small U-Net training pipeline on free Sentinel-2 data.

Expected folder structure (chips created from free Sentinel-2 L2A tiles):

unet_semantic_segmentation/
  data/
    images/
      train/*.npy  # (4, H, W) float32, normalized
      val/*.npy
    masks/
      train/*.npy  # (H, W) int64 with class ids
      val/*.npy
  models/
"""

import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------
# 1. Dataset
# -----------------------
class ChipsDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.npy")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
        assert len(self.img_paths) == len(self.mask_paths), "Image/mask count mismatch!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype("float32")  # (C, H, W)
        mask = np.load(self.mask_paths[idx]).astype("int64")  # (H, W)

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask


# -----------------------
# 2. U-Net building blocks
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, n_classes=4):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)
        p3 = self.pool(c3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        x3 = torch.cat([u3, c3], dim=1)
        c3 = self.conv3(x3)

        u2 = self.up2(c3)
        x2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(x2)

        u1 = self.up1(c2)
        x1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(x1)

        logits = self.out_conv(c1)
        return logits


# -----------------------
# 3. Simple IoU metric
# -----------------------
def compute_iou(pred, target, num_classes):
    """
    pred, target: (H, W) with class ids.
    returns mean IoU over classes present in target.
    """
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)

        if target_c.sum() == 0:
            continue  # skip empty class

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)


# -----------------------
# 4. Pipeline class
# -----------------------
class UNetSegmentationPipeline:
    """
    Small training/evaluation pipeline for Sentinel-2 segmentation.

    Steps:
      - load chips from free Sentinel-2 data
      - train U-Net
      - evaluate mean IoU on validation set
      - save model weights
    """

    def __init__(
        self,
        img_train_dir="data/images/train",
        mask_train_dir="data/masks/train",
        img_val_dir="data/images/val",
        mask_val_dir="data/masks/val",
        in_channels=4,
        num_classes=4,
        batch_size=4,
        lr=1e-4,
        n_epochs=10,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_epochs = n_epochs
        self.num_classes = num_classes

        # datasets & loaders
        train_ds = ChipsDataset(img_train_dir, mask_train_dir)
        val_ds = ChipsDataset(img_val_dir, mask_val_dir)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # model, loss, optimizer
        self.model = UNet(in_ch=in_channels, n_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}/{self.n_epochs}"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)              # (B, C, H, W)
            loss = self.criterion(logits, masks)   # (B, H, W)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        iou_scores = []

        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)  # (B, H, W)
                for b in range(preds.shape[0]):
                    iou = compute_iou(
                        preds[b].cpu(), masks[b].cpu(), num_classes=self.num_classes
                    )
                    iou_scores.append(iou)

        val_loss /= len(self.val_loader)
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        return val_loss, mean_iou

    def run(self):
        best_iou = 0.0
        os.makedirs("models", exist_ok=True)

        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, mean_iou = self.evaluate()

            print(
                f"Epoch {epoch+1}/{self.n_epochs} "
                f"- Train loss: {train_loss:.4f} "
                f"- Val loss: {val_loss:.4f} "
                f"- Mean IoU: {mean_iou:.4f}"
            )

            # save best model
            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save(self.model.state_dict(), "models/unet_sentinel2.pt")
                print(f"  ✅ New best model saved (mean IoU = {best_iou:.4f})")


def main():
    pipeline = UNetSegmentationPipeline(
        img_train_dir="data/images/train",
        mask_train_dir="data/masks/train",
        img_val_dir="data/images/val",
        mask_val_dir="data/masks/val",
        in_channels=4,
        num_classes=4,
        batch_size=4,
        lr=1e-4,
        n_epochs=10,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
