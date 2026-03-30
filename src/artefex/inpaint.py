"""Inpainting module - detects and repairs physical damage in images.

Uses classical CV to detect damage (scratches, tears, white flaking,
stains) and generates a binary mask, then uses the LaMa neural
inpainting model to fill in damaged regions.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from artefex.models import Degradation


def detect_damage_mask(img: Image.Image, sensitivity: float = 0.5):
    """Detect physical damage and return a binary mask.

    Detects: white flaking/peeling, dark scratches, tears, stains,
    and other physical damage patterns common in old photographs.

    Args:
        img: PIL Image (RGB)
        sensitivity: 0.0 (conservative) to 1.0 (aggressive).
            Higher values detect more damage but risk marking
            valid image content.

    Returns:
        Tuple of (mask as PIL Image mode 'L', damage_pct as float)
        Mask: 255 = damaged pixel, 0 = intact pixel
    """
    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Detect very bright spots (white flaking/peeling)
    # Old photos have damage that appears as bright white patches
    brightness = np.max(arr, axis=2)
    bright_thresh = 240 - int(sensitivity * 20)
    bright_mask = brightness > bright_thresh

    # Check if bright spots are surrounded by darker content
    # (to avoid marking legitimately bright image areas)
    local_mean = _box_filter(gray, 15)
    contrast = brightness - local_mean
    bright_damage = bright_mask & (contrast > 60 - int(sensitivity * 30))
    mask[bright_damage] = 255

    # 2. Detect scratches (thin bright or dark lines)
    # Use gradient magnitude to find high-contrast linear features
    grad_h = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad_v = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    grad_mag = np.sqrt(grad_h ** 2 + grad_v ** 2)

    # Scratches have very high local gradient
    grad_thresh = np.percentile(grad_mag, 98 - int(sensitivity * 5))
    scratch_mask = grad_mag > grad_thresh

    # Only count as damage if it's a thin feature (not an edge)
    # Check by looking at gradient consistency in neighborhood
    local_grad = _box_filter(grad_mag, 5)
    thin_features = scratch_mask & (local_grad < grad_thresh * 0.6)
    mask[thin_features] = 255

    # 3. Detect unusual color patches (stains, discoloration spots)
    # Look for regions where color deviates strongly from neighbors
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    local_r = _box_filter(r, 21)
    local_g = _box_filter(g, 21)
    local_b = _box_filter(b, 21)
    color_diff = np.sqrt(
        (r - local_r) ** 2 + (g - local_g) ** 2 + (b - local_b) ** 2
    )
    stain_thresh = np.percentile(color_diff, 97 - int(sensitivity * 5))
    stain_mask = color_diff > stain_thresh

    # Only mark as damage if the deviation is extreme
    if sensitivity > 0.3:
        mask[stain_mask & (color_diff > stain_thresh * 1.5)] = 255

    # 4. Dilate the mask to ensure coverage of damage boundaries
    mask = _dilate(mask, radius=max(1, int(2 + sensitivity * 3)))

    # Calculate damage percentage
    damage_pct = np.sum(mask > 0) / (h * w)

    return Image.fromarray(mask, mode="L"), damage_pct


def _box_filter(arr, size):
    """Box/mean filter using PIL on uint8."""
    from PIL import ImageFilter
    # Convert to uint8 for PIL compatibility
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return arr.copy()
    scaled = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    pil = Image.fromarray(scaled, mode="L")
    filtered = pil.filter(ImageFilter.BoxBlur(size // 2))
    result = np.array(filtered, dtype=np.float64)
    return result / 255.0 * (mx - mn) + mn


def _dilate(mask, radius=2):
    """Simple binary dilation using box filter."""
    kernel_size = radius * 2 + 1
    flt = _box_filter(mask.astype(np.float64), kernel_size)
    return (flt > 0).astype(np.uint8) * 255


def inpaint_image(
    img: Image.Image,
    mask: Image.Image,
    model_path: Optional[Path] = None,
) -> Image.Image:
    """Inpaint damaged regions using the LaMa neural model.

    Args:
        img: Original image (RGB)
        mask: Binary damage mask (L mode, 255=damaged)
        model_path: Path to inpaint_v1.onnx. If None, looks in
            the default model directory.

    Returns:
        Inpainted image (RGB, same size as input)
    """
    if model_path is None:
        from artefex.models_registry import MODEL_DIR
        model_path = MODEL_DIR / "inpaint_v1.onnx"

    if not model_path.exists():
        return img  # No model available, return original

    try:
        import onnxruntime as ort
    except ImportError:
        return img

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    orig_size = img.size  # (w, h)

    # Resize to 512x512 for LaMa
    img_resized = img.convert("RGB").resize(
        (512, 512), Image.LANCZOS
    )
    mask_resized = mask.convert("L").resize(
        (512, 512), Image.NEAREST
    )

    # Prepare image input: NCHW, float32, [0, 1]
    img_arr = (
        np.array(img_resized, dtype=np.float32) / 255.0
    )
    img_blob = img_arr.transpose(2, 0, 1)[np.newaxis]

    # Prepare mask input: NCHW, float32, binary
    mask_arr = np.array(mask_resized, dtype=np.float32) / 255.0
    mask_blob = mask_arr[np.newaxis, np.newaxis]
    mask_blob = (mask_blob > 0.5).astype(np.float32)

    # Run inference
    inputs = session.get_inputs()
    input_dict = {
        inputs[0].name: img_blob,
        inputs[1].name: mask_blob,
    }
    output = session.run(None, input_dict)[0]

    # Post-process: output is NCHW, [0, 255] uint8 range
    result = output[0].transpose(1, 2, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Resize back to original dimensions
    result_img = result_img.resize(orig_size, Image.LANCZOS)

    return result_img


def detect_physical_damage(
    img: Image.Image, arr: np.ndarray
) -> Optional[Degradation]:
    """Detect physical damage for the analysis pipeline.

    Returns a Degradation object if significant physical damage
    is detected, or None if the image appears undamaged.
    """
    if len(arr.shape) < 3 or arr.shape[2] < 3:
        return None

    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return None

    mask, damage_pct = detect_damage_mask(img, sensitivity=0.5)

    if damage_pct < 0.005:  # Less than 0.5% damage
        return None

    severity = min(1.0, damage_pct * 5)  # 20% damage = severity 1.0
    confidence = min(1.0, damage_pct * 10 + 0.3)

    detail = (
        f"Physical damage detected covering {damage_pct:.1%} of "
        f"the image. Includes scratches, tears, flaking, or stains."
    )

    return Degradation(
        name="Physical Damage",
        confidence=confidence,
        severity=severity,
        detail=detail,
        category="physical",
    )
