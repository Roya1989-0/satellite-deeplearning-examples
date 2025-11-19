# train_unet.py
import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as TF
from tqdm import tqdm

# -----------------------
# 1. Simple Dataset
# -----------------------
class ChipsDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.npy")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
        assert len(self.img_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype("float32")  # (C,H,W)
        mask = np.load(self.mask_paths[idx]).astype("int64")  # (H,W)

        # to tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask

# -----------------------
# 2. Minimal U-Net
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
# 3. Simple train loop
# -----------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ChipsDataset("data/images/train", "data/masks/train")
    val_ds   = ChipsDataset("data/images/val", "data/masks/val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = UNet(in_ch=4, n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)             # (B,4,H,W)
            loss = criterion(logits, masks)  # masks: (B,H,W)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # very quick val loop (just loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Val loss: {val_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_sentinel2.pt")
    print("Model saved to models/unet_sentinel2.pt")


if __name__ == "__main__":
    train()
