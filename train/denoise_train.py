"""Train the adaptive denoising model for Artefex.

A lightweight U-Net that removes noise while preserving edges.
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


class DenoiseNet(nn.Module):
    """Lightweight U-Net for image denoising (3-channel RGB)."""

    def __init__(self, channels=3):
        super().__init__()
        self.enc1 = self._block(channels, 48)
        self.enc2 = self._block(48, 96)
        self.enc3 = self._block(96, 192)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._block(192, 384)

        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = self._block(384, 192)
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = self._block(192, 96)
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec1 = self._block(96, 48)

        self.out = nn.Conv2d(48, channels, kernel_size=1)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return x + self.out(d1)


class ImagePairDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.degraded_dir = data_dir / "degraded"
        self.clean_dir = data_dir / "clean"
        self.files = sorted(self.degraded_dir.glob("*.png"))
        if not self.files:
            raise ValueError(f"No training data found in {self.degraded_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        deg_path = self.files[idx]
        clean_path = self.clean_dir / deg_path.name

        deg = np.array(Image.open(deg_path).convert("RGB"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0

        return torch.from_numpy(deg.transpose(2, 0, 1)), torch.from_numpy(clean.transpose(2, 0, 1))


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

    dataset = ImagePairDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DenoiseNet(channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Combined L1 + perceptual-style loss
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            output = model(degraded)
            loss = l1(output, clean) * 0.7 + mse(output, clean) * 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "denoise_v1_best.pth")

    # Export to ONNX
    model.eval()
    model.load_state_dict(torch.load(output_dir / "denoise_v1_best.pth", weights_only=True))

    dummy = torch.randn(1, 3, 256, 256).to(device)
    onnx_path = output_dir / "denoise_v1.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17,
    )

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"ONNX model: {onnx_path}")
    print(f"\nTo use: artefex models import denoise-v1 {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Artefex denoising model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
