"""First-launch model downloader.

Downloads pre-converted ONNX models from GitHub Releases on first
launch. All models are served directly as ONNX files - no PyTorch
or conversion needed on the user's machine.

Usage:
    from artefex.model_downloader import ensure_models
    missing = ensure_models(on_progress=callback)
"""

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from artefex.models_registry import MODEL_DIR, REGISTRY, ModelRegistry

logger = logging.getLogger(__name__)

# GitHub Release base URL for pre-converted ONNX models
_RELEASE_BASE = (
    "https://github.com/turnert2005/artefex"
    "/releases/download/v1.0.0"
)

# Download sources - all models served as ONNX from GitHub Releases
DOWNLOAD_SOURCES = {
    "deblock-v1": {
        "url": f"{_RELEASE_BASE}/deblock_v1.onnx",
        "type": "url",
    },
    "denoise-v1": {
        "url": f"{_RELEASE_BASE}/denoise_v1.onnx",
        "type": "url",
    },
    "sharpen-v1": {
        "url": f"{_RELEASE_BASE}/sharpen_v1.onnx",
        "type": "url",
    },
    "aigen-detect-v1": {
        "url": f"{_RELEASE_BASE}/aigen_detect_v1.onnx",
        "type": "url",
    },
    "inpaint-v1": {
        "url": f"{_RELEASE_BASE}/inpaint_v1.onnx",
        "type": "url",
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

        url = source.get("url")
        if not url:
            continue

        success = download_file(url, dest, on_progress)

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
