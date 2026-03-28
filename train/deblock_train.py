"""Train the JPEG deblocking model for Artefex.

A lightweight U-Net that removes 8x8 block artifacts from JPEG-compressed images.
Exports to ONNX for use with the artefex restore pipeline.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("Training requires PyTorch. Install with: pip install torch torchvision")

from PIL import Image


class DeblockNet(nn.Module):
    """Lightweight U-Net for JPEG artifact removal."""

    def __init__(self, channels=1):
        super().__init__()
        # Encoder
        self.enc1 = self._block(channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)

        self.out = nn.Conv2d(64, channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Residual learning: predict the difference
        return x + self.out(d1)


class ImagePairDataset(Dataset):
    """Loads paired (degraded, clean) images from directories."""

    def __init__(self, data_dir: Path, grayscale: bool = True):
        self.degraded_dir = data_dir / "degraded"
        self.clean_dir = data_dir / "clean"
        self.grayscale = grayscale

        self.files = sorted(self.degraded_dir.glob("*.png"))
        if not self.files:
            raise ValueError(f"No training data found in {self.degraded_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        clean_path = self.clean_dir / deg_path.name

        if self.grayscale:
            deg = np.array(Image.open(deg_path).convert("L"), dtype=np.float32) / 255.0
            clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
            deg = deg[np.newaxis, :, :]
            clean = clean[np.newaxis, :, :]
        else:
            deg = np.array(Image.open(deg_path).convert("RGB"), dtype=np.float32) / 255.0
            clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0
            deg = deg.transpose(2, 0, 1)
            clean = clean.transpose(2, 0, 1)

        return torch.from_numpy(deg), torch.from_numpy(clean)


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on: {device}")
    print(f"Data: {data_dir}")
    print(f"Epochs: {epochs}")

    dataset = ImagePairDataset(data_dir, grayscale=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DeblockNet(channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            output = model(degraded)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "deblock_v1_best.pth")

    # Export to ONNX
    model.eval()
    model.load_state_dict(torch.load(output_dir / "deblock_v1_best.pth", weights_only=True))

    dummy = torch.randn(1, 1, 256, 256).to(device)
    onnx_path = output_dir / "deblock_v1.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17,
    )

    print(f"\nTraining complete.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"ONNX model: {onnx_path}")
    print(f"\nTo use: artefex models import deblock-v1 {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Artefex JPEG deblocking model")
    parser.add_argument("--data", type=Path, required=True, help="Training data directory")
    parser.add_argument("--output", type=Path, default=Path("./models"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
