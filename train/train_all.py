"""Orchestrate the full Artefex training pipeline.

Downloads or accepts source images, generates degraded/clean pairs for each
degradation type, trains all four restoration models, validates them, and
imports the resulting ONNX files into the artefex model registry.

Usage examples:
    python train_all.py --source ./my_clean_images
    python train_all.py --source ./images --epochs 50 --models deblock,denoise
    python train_all.py --source ./images --skip-generate --skip-train
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODELS = ["deblock", "denoise", "sharpen", "color"]

# Maps model name -> (training script filename, ONNX output name, registry key)
MODEL_CONFIG = {
    "deblock": {
        "script": "deblock_train.py",
        "onnx": "deblock_v1.onnx",
        "registry_key": "deblock-v1",
        "degradation": "deblock",
        "channels": 1,
    },
    "denoise": {
        "script": "denoise_train.py",
        "onnx": "denoise_v1.onnx",
        "registry_key": "denoise-v1",
        "degradation": "denoise",
        "channels": 3,
    },
    "sharpen": {
        "script": "sharpen_train.py",
        "onnx": "sharpen_v1.onnx",
        "registry_key": "sharpen-v1",
        "degradation": "sharpen",
        "channels": 3,
    },
    "color": {
        "script": "color_train.py",
        "onnx": "color_correct_v1.onnx",
        "registry_key": "color-correct-v1",
        "degradation": "color",
        "channels": 3,
    },
}

TRAIN_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    """Format seconds as a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_subprocess(cmd: list[str], label: str) -> bool:
    """Run a subprocess and stream its output. Returns True on success."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}\n")
    result = subprocess.run(
        cmd,
        cwd=str(TRAIN_DIR),
    )
    if result.returncode != 0:
        print(f"\n[FAIL] {label} exited with code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 1 - Data generation
# ---------------------------------------------------------------------------

def generate_data(
    source: Path,
    output: Path,
    models: list[str],
    crop_size: int,
    pairs_per_image: int,
) -> None:
    """Generate training pairs for each requested degradation type."""
    print("\n" + "#" * 72)
    print("# STAGE 1: Generating training data")
    print("#" * 72)

    for model_name in models:
        cfg = MODEL_CONFIG[model_name]
        data_dir = output / "data" / model_name
        if (data_dir / "degraded").exists() and (data_dir / "clean").exists():
            n_existing = len(list((data_dir / "degraded").glob("*.png")))
            if n_existing > 0:
                print(
                    f"\n[SKIP] {model_name}: "
                    f"{n_existing} pairs already exist in {data_dir}"
                )
                continue

        cmd = [
            sys.executable,
            str(TRAIN_DIR / "generate_data.py"),
            "--source", str(source),
            "--output", str(data_dir),
            "--type", cfg["degradation"],
            "--crop-size", str(crop_size),
            "--pairs-per-image", str(pairs_per_image),
        ]
        ok = run_subprocess(cmd, f"Generating {model_name} data")
        if not ok:
            print(f"[ERROR] Data generation failed for {model_name}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 2 - Training
# ---------------------------------------------------------------------------

def train_models(
    output: Path,
    models: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> dict[str, float]:
    """Train each model sequentially. Returns dict of training times."""
    print("\n" + "#" * 72)
    print("# STAGE 2: Training models")
    print("#" * 72)

    device = detect_device(device)
    print(f"\nDevice: {device}")

    train_times: dict[str, float] = {}

    for model_name in models:
        cfg = MODEL_CONFIG[model_name]
        data_dir = output / "data" / model_name
        model_dir = output / "models"

        if not (data_dir / "degraded").exists():
            print(
                f"\n[ERROR] No training data for {model_name} at "
                f"{data_dir}. Run data generation first."
            )
            sys.exit(1)

        script = TRAIN_DIR / cfg["script"]
        if not script.exists():
            print(f"\n[ERROR] Training script not found: {script}")
            sys.exit(1)

        cmd = [
            sys.executable,
            str(script),
            "--data", str(data_dir),
            "--output", str(model_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--device", device,
        ]

        t0 = time.time()
        ok = run_subprocess(cmd, f"Training {model_name} ({epochs} epochs)")
        elapsed = time.time() - t0
        train_times[model_name] = elapsed

        if not ok:
            print(f"[ERROR] Training failed for {model_name}")
            sys.exit(1)

        onnx_path = model_dir / cfg["onnx"]
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(
                f"\n[OK] {model_name} - ONNX saved "
                f"({size_mb:.2f} MB) in {fmt_time(elapsed)}"
            )
        else:
            print(f"\n[WARN] Expected ONNX not found: {onnx_path}")

    return train_times


# ---------------------------------------------------------------------------
# Stage 3 - Validation
# ---------------------------------------------------------------------------

def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def validate_model(
    onnx_path: Path,
    data_dir: Path,
    channels: int,
    num_test: int = 5,
) -> float:
    """Run an ONNX model on test images and return mean PSNR improvement.

    Returns the average PSNR improvement (restored vs degraded, relative
    to clean ground truth). Positive means the model improved quality.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "[WARN] onnxruntime not installed - skipping validation. "
            "Install with: pip install onnxruntime"
        )
        return 0.0

    degraded_dir = data_dir / "degraded"
    clean_dir = data_dir / "clean"

    test_files = sorted(degraded_dir.glob("*.png"))[:num_test]
    if not test_files:
        print(f"[WARN] No test images found in {degraded_dir}")
        return 0.0

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    improvements = []

    for deg_path in test_files:
        clean_path = clean_dir / deg_path.name
        if not clean_path.exists():
            continue

        if channels == 1:
            deg_img = np.array(
                Image.open(deg_path).convert("L"), dtype=np.float32
            )
            clean_img = np.array(
                Image.open(clean_path).convert("L"), dtype=np.float32
            )
            # Shape: (1, 1, H, W)
            input_tensor = deg_img[np.newaxis, np.newaxis, :, :] / 255.0
        else:
            deg_img = np.array(
                Image.open(deg_path).convert("RGB"), dtype=np.float32
            )
            clean_img = np.array(
                Image.open(clean_path).convert("RGB"), dtype=np.float32
            )
            # Shape: (1, 3, H, W)
            input_tensor = deg_img.transpose(2, 0, 1)[np.newaxis] / 255.0

        output = sess.run(None, {input_name: input_tensor})[0]

        # Convert back to uint8 image space
        if channels == 1:
            restored = np.clip(output[0, 0] * 255.0, 0, 255).astype(np.uint8)
            deg_uint8 = deg_img.astype(np.uint8)
            clean_uint8 = clean_img.astype(np.uint8)
        else:
            restored = np.clip(
                output[0].transpose(1, 2, 0) * 255.0, 0, 255
            ).astype(np.uint8)
            deg_uint8 = deg_img.astype(np.uint8)
            clean_uint8 = clean_img.astype(np.uint8)

        psnr_before = compute_psnr(deg_uint8, clean_uint8)
        psnr_after = compute_psnr(restored, clean_uint8)
        improvement = psnr_after - psnr_before
        improvements.append(improvement)

        print(
            f"  {deg_path.name}: "
            f"before={psnr_before:.2f} dB, "
            f"after={psnr_after:.2f} dB, "
            f"delta={improvement:+.2f} dB"
        )

    if not improvements:
        return 0.0
    return float(np.mean(improvements))


def validate_all(
    output: Path,
    models: list[str],
) -> dict[str, float]:
    """Validate all trained models. Returns dict of PSNR improvements."""
    print("\n" + "#" * 72)
    print("# STAGE 3: Validating models")
    print("#" * 72)

    results: dict[str, float] = {}

    for model_name in models:
        cfg = MODEL_CONFIG[model_name]
        onnx_path = output / "models" / cfg["onnx"]
        data_dir = output / "data" / model_name

        if not onnx_path.exists():
            print(f"\n[SKIP] {model_name}: ONNX file not found at {onnx_path}")
            results[model_name] = 0.0
            continue

        print(f"\n--- Validating {model_name} ---")
        improvement = validate_model(
            onnx_path, data_dir, cfg["channels"]
        )
        results[model_name] = improvement

        threshold = 1.0
        status = "PASS" if improvement > threshold else "FAIL"
        print(
            f"  Mean PSNR improvement: {improvement:+.2f} dB "
            f"[{status}] (target: >{threshold} dB)"
        )

    return results


# ---------------------------------------------------------------------------
# Stage 4 - Import into artefex registry
# ---------------------------------------------------------------------------

def import_models(output: Path, models: list[str]) -> None:
    """Import trained ONNX models into the artefex model registry."""
    print("\n" + "#" * 72)
    print("# STAGE 4: Importing models into artefex registry")
    print("#" * 72)

    for model_name in models:
        cfg = MODEL_CONFIG[model_name]
        onnx_path = output / "models" / cfg["onnx"]

        if not onnx_path.exists():
            print(f"\n[SKIP] {model_name}: ONNX not found at {onnx_path}")
            continue

        checksum = sha256_file(onnx_path)
        size_mb = onnx_path.stat().st_size / (1024 * 1024)

        print(f"\n  {model_name}:")
        print(f"    File:   {onnx_path}")
        print(f"    Size:   {size_mb:.2f} MB")
        print(f"    SHA256: {checksum}")

        cmd = [
            sys.executable, "-m", "artefex",
            "models", "import", cfg["registry_key"], str(onnx_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("    Status: Imported successfully")
            if result.stdout.strip():
                print(f"    {result.stdout.strip()}")
        else:
            # Fallback - try direct registry import via Python
            print("    CLI import returned non-zero; trying direct import...")
            try:
                # Add src to path so we can import artefex
                src_dir = TRAIN_DIR.parent / "src"
                if str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))
                from artefex.models_registry import ModelRegistry
                registry = ModelRegistry()
                info = registry.import_model(onnx_path, cfg["registry_key"])
                print(f"    Status: Imported to {info.local_path}")
            except Exception as e:
                print(f"    [ERROR] Import failed: {e}")


# ---------------------------------------------------------------------------
# Stage 5 - Summary
# ---------------------------------------------------------------------------

def print_summary(
    models: list[str],
    train_times: dict[str, float],
    psnr_results: dict[str, float],
    total_start: float,
) -> None:
    """Print a final summary of all results."""
    total_elapsed = time.time() - total_start

    print("\n" + "#" * 72)
    print("# SUMMARY")
    print("#" * 72)

    print(f"\n{'Model':<12} {'Train Time':<14} {'PSNR Gain':<14} {'Status'}")
    print("-" * 52)

    for model_name in models:
        t = train_times.get(model_name, 0.0)
        psnr = psnr_results.get(model_name, 0.0)
        status = "PASS" if psnr > 1.0 else "FAIL"
        time_str = fmt_time(t) if t > 0 else "skipped"
        print(
            f"{model_name:<12} {time_str:<14} {psnr:+.2f} dB"
            f"       {status}"
        )

    print(f"\nTotal time: {fmt_time(total_elapsed)}")

    print("\n--- Next steps ---")
    print("  1. Run 'artefex models list' to verify installed models")
    print("  2. Test on real images: artefex analyze image.jpg")
    print(
        "  3. Restore with neural models: "
        "artefex restore image.jpg -o restored.jpg"
    )
    print(
        "  4. For better results, train with more epochs "
        "(--epochs 200) or more data"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artefex full training pipeline - generate data, train models, "
            "validate, and import into the artefex registry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_all.py --source ./clean_images\n"
            "  python train_all.py --source ./images --epochs 50 "
            "--models deblock,denoise\n"
            "  python train_all.py --source ./images "
            "--skip-train  # validate and import only\n"
        ),
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory of clean source images (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./training_output"),
        help="Output directory for data, models, and logs "
        "(default: ./training_output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per model (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device (default: auto)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Crop size for training patches (default: 256)",
    )
    parser.add_argument(
        "--pairs-per-image",
        type=int,
        default=8,
        help="Number of training pairs per source image (default: 8)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation (use existing data in output dir)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (just validate and import existing models)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated list of models to train "
            "(default: all). Options: deblock, denoise, sharpen, color"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    source = args.source.resolve()
    output = args.output.resolve()

    # Parse model list
    if args.models == "all":
        models = list(ALL_MODELS)
    else:
        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if m not in ALL_MODELS:
                print(
                    f"[ERROR] Unknown model: {m}. "
                    f"Choose from: {', '.join(ALL_MODELS)}"
                )
                sys.exit(1)

    # Validate source directory
    if not source.exists() or not source.is_dir():
        print(f"[ERROR] Source directory not found: {source}")
        sys.exit(1)

    total_start = time.time()

    print("#" * 72)
    print("#  Artefex Training Pipeline")
    print("#" * 72)
    print(f"\n  Source images: {source}")
    print(f"  Output dir:   {output}")
    print(f"  Models:       {', '.join(models)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Device:       {detect_device(args.device)}")
    print(f"  Crop size:    {args.crop_size}")
    print(f"  Pairs/image:  {args.pairs_per_image}")

    output.mkdir(parents=True, exist_ok=True)

    # Stage 1 - Generate data
    if not args.skip_generate:
        generate_data(
            source, output, models,
            args.crop_size, args.pairs_per_image,
        )
    else:
        print("\n[SKIP] Data generation (--skip-generate)")

    # Stage 2 - Train
    train_times: dict[str, float] = {}
    if not args.skip_train:
        train_times = train_models(
            output, models,
            args.epochs, args.batch_size, args.lr, args.device,
        )
    else:
        print("\n[SKIP] Training (--skip-train)")

    # Stage 3 - Validate
    psnr_results = validate_all(output, models)

    # Stage 4 - Import
    import_models(output, models)

    # Stage 5 - Summary
    print_summary(models, train_times, psnr_results, total_start)


if __name__ == "__main__":
    main()
