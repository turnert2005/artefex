"""Generate synthetic training data by applying controlled degradations to clean images."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def degrade_jpeg(img: Image.Image, quality_range=(5, 40)) -> Image.Image:
    """Apply JPEG compression at a random quality level."""
    import io
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade_noise(img: Image.Image, sigma_range=(10, 50)) -> Image.Image:
    """Add Gaussian noise with random sigma."""
    arr = np.array(img, dtype=np.float64)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def degrade_blur(img: Image.Image, radius_range=(1, 4)) -> Image.Image:
    """Apply Gaussian blur to simulate resolution loss."""
    radius = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade_color_shift(img: Image.Image, shift_range=(0.8, 1.2)) -> Image.Image:
    """Randomly shift color channels."""
    arr = np.array(img, dtype=np.float64)
    for ch in range(3):
        factor = random.uniform(*shift_range)
        arr[:, :, ch] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade_multi_compress(img: Image.Image, rounds=(2, 5), quality_range=(15, 50)) -> Image.Image:
    """Apply multiple rounds of JPEG compression."""
    import io
    n_rounds = random.randint(*rounds)
    for _ in range(n_rounds):
        quality = random.randint(*quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")
    return img


DEGRADATIONS = {
    "deblock": degrade_jpeg,
    "denoise": degrade_noise,
    "sharpen": degrade_blur,
    "color": degrade_color_shift,
    "multi": degrade_multi_compress,
}


def generate_pairs(
    source_dir: Path,
    output_dir: Path,
    degradation: str,
    crop_size: int = 256,
    pairs_per_image: int = 4,
):
    """Generate (degraded, clean) pairs from source images."""
    degrade_fn = DEGRADATIONS[degradation]
    clean_dir = output_dir / "clean"
    degraded_dir = output_dir / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    source_images = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not source_images:
        print(f"No images found in {source_dir}")
        return

    pair_idx = 0
    for img_path in source_images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        w, h = img.size
        if w < crop_size or h < crop_size:
            continue

        for _ in range(pairs_per_image):
            # Random crop
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            clean_crop = img.crop((x, y, x + crop_size, y + crop_size))

            # Apply degradation
            degraded_crop = degrade_fn(clean_crop)

            # Ensure same size after degradation
            if degraded_crop.size != clean_crop.size:
                degraded_crop = degraded_crop.resize(clean_crop.size, Image.LANCZOS)

            # Save
            clean_dir_path = clean_dir / f"{pair_idx:06d}.png"
            degraded_dir_path = degraded_dir / f"{pair_idx:06d}.png"
            clean_crop.save(clean_dir_path)
            degraded_crop.save(degraded_dir_path)
            pair_idx += 1

    print(f"Generated {pair_idx} pairs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for Artefex models")
    parser.add_argument("--source", type=Path, required=True, help="Directory of clean source images")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for training pairs")
    parser.add_argument(
        "--type",
        choices=list(DEGRADATIONS.keys()),
        default="deblock",
        help="Type of degradation to generate",
    )
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size for training patches")
    parser.add_argument("--pairs-per-image", type=int, default=4, help="Number of pairs per source image")
    args = parser.parse_args()

    generate_pairs(args.source, args.output, args.type, args.crop_size, args.pairs_per_image)


if __name__ == "__main__":
    main()
