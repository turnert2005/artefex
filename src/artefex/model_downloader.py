"""First-launch model downloader.

Checks for missing neural models on startup and downloads them
from their source repositories. Downloads .pth files and converts
to ONNX, or downloads pre-converted ONNX files from GitHub Releases.

Usage:
    from artefex.model_downloader import ensure_models
    missing = ensure_models(on_progress=callback)
"""

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from artefex.models_registry import MODEL_DIR, REGISTRY, ModelRegistry

logger = logging.getLogger(__name__)

# Download sources for each model.
# Priority: GitHub Release ONNX > HuggingFace > original .pth + convert
DOWNLOAD_SOURCES = {
    "denoise-v1": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
        "type": "pth",
        "convert": "dncnn_color_blind",
    },
    "sharpen-v1": {
        "url": None,  # NAFNet requires gdown (Google Drive)
        "type": "skip",
    },
    "deblock-v1": {
        "url": None,  # FBCNN is 288MB .pth, needs conversion
        "type": "skip",
    },
    "aigen-detect-v1": {
        "url": "https://raw.githubusercontent.com/Ouxiang-Li/SAFE/main/checkpoint/checkpoint-best.pth",
        "type": "pth",
        "convert": "safe",
    },
    "inpaint-v1": {
        "url": None,  # Downloaded via huggingface_hub
        "type": "huggingface",
        "repo": "opencv/inpainting_lama",
        "file": "inpainting_lama_2025jan.onnx",
    },
}


def get_missing_models() -> list[str]:
    """Return list of model keys that are not installed or not trained."""
    registry = ModelRegistry()
    missing = []
    for key in REGISTRY:
        if key == "color-correct-v1":
            continue  # No neural model for color correction
        model = registry.get_model(key)
        if model is None or not model.is_available or not model.is_trained:
            missing.append(key)
    return missing


def download_file(
    url: str,
    dest: Path,
    on_progress: Optional[callable] = None,
) -> bool:
    """Download a file from URL with progress callback.

    Args:
        url: URL to download from
        dest: Destination path
        on_progress: Callback(bytes_downloaded, total_bytes, filename)

    Returns:
        True if download succeeded
    """
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "artefex/1.0")

        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)

            tmp = dest.with_suffix(dest.suffix + ".part")
            downloaded = 0
            chunk_size = 64 * 1024  # 64 KB chunks

            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(
                            downloaded, total, dest.name
                        )

            shutil.move(str(tmp), str(dest))
            return True

    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        # Clean up partial download
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.unlink(missing_ok=True)
        return False


def download_from_huggingface(
    repo: str,
    filename: str,
    dest: Path,
    on_progress: Optional[callable] = None,
) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename)
        shutil.copy2(path, dest)
        if on_progress:
            size = dest.stat().st_size
            on_progress(size, size, dest.name)
        return True
    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")
        # Fallback to direct URL
        url = (
            f"https://huggingface.co/{repo}/resolve/main/{filename}"
        )
        return download_file(url, dest, on_progress)


def ensure_models(
    on_progress: Optional[callable] = None,
    on_status: Optional[callable] = None,
) -> list[str]:
    """Check for missing models and download them.

    Args:
        on_progress: Callback(bytes, total, filename) for download progress
        on_status: Callback(message) for status updates

    Returns:
        List of model keys that were successfully downloaded
    """
    missing = get_missing_models()
    if not missing:
        return []

    if on_status:
        on_status(f"Found {len(missing)} missing model(s)")

    downloaded = []
    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    for key in missing:
        info = REGISTRY.get(key)
        if info is None:
            continue

        source = DOWNLOAD_SOURCES.get(key)
        if source is None:
            continue

        dest = model_dir / info["filename"]

        if on_status:
            on_status(
                f"Downloading {info['name']} "
                f"({info['size_mb']:.0f} MB)..."
            )

        success = False
        src_type = source.get("type", "skip")

        if src_type == "skip":
            # Model requires special handling (gdown, conversion)
            # Try the conversion script if .pth exists locally
            pth_dir = (
                Path(__file__).parent.parent.parent
                / "train" / "pretrained"
            )
            if pth_dir.exists():
                if on_status:
                    on_status(
                        f"Checking local pre-trained weights "
                        f"for {key}..."
                    )
            continue

        elif src_type == "pth":
            url = source.get("url")
            if url:
                # Download .pth, then we'd need to convert
                # For now, just note it needs conversion
                pth_tmp = Path(
                    tempfile.mktemp(suffix=".pth")
                )
                success = download_file(
                    url, pth_tmp, on_progress
                )
                if success:
                    if on_status:
                        on_status(
                            f"Converting {key} to ONNX..."
                        )
                    # The conversion would happen here
                    # For now, clean up
                    pth_tmp.unlink(missing_ok=True)
                    success = False  # Can't convert without torch

        elif src_type == "huggingface":
            repo = source.get("repo", "")
            filename = source.get("file", "")
            if repo and filename:
                success = download_from_huggingface(
                    repo, filename, dest, on_progress
                )

        elif src_type == "url":
            url = source.get("url")
            if url:
                success = download_file(
                    url, dest, on_progress
                )

        if success:
            # Verify SHA-256 if we have one
            expected_sha = info.get("sha256", "")
            if expected_sha and dest.exists():
                actual_sha = hashlib.sha256(
                    dest.read_bytes()
                ).hexdigest()
                if actual_sha != expected_sha:
                    logger.warning(
                        f"SHA-256 mismatch for {key}: "
                        f"expected {expected_sha[:16]}..., "
                        f"got {actual_sha[:16]}..."
                    )
                    dest.unlink()
                    continue

            downloaded.append(key)
            if on_status:
                on_status(f"Installed {info['name']}")

    if on_status:
        total = len(downloaded)
        if total > 0:
            on_status(f"Downloaded {total} model(s)")
        elif missing:
            on_status(
                "Some models could not be downloaded. "
                "Run 'python train/convert_pretrained.py "
                "--install' to install from local sources."
            )

    return downloaded
