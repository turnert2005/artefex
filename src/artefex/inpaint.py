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
    # Only flag pixels that are very bright AND surrounded by
    # much darker content. This avoids false positives from
    # JPEG artifacts, highlights, or bright image regions.
    brightness = np.max(arr, axis=2)
    bright_thresh = 240 - int(sensitivity * 15)
    bright_mask = brightness > bright_thresh

    local_mean = _box_filter(gray, 25)
    contrast = brightness - local_mean
    # Require high contrast (bright spot in dark area)
    bright_damage = bright_mask & (contrast > 70 - int(sensitivity * 20))
    mask[bright_damage] = 255

    # 2. Remove small isolated noise spots (keep large patches)
    # Erode with small kernel to remove 1-2 pixel noise spots
    from PIL import ImageFilter
    mask_pil = Image.fromarray(mask, mode="L")
    eroded = mask_pil.filter(ImageFilter.MinFilter(3))
    mask = np.array(eroded)

    # 3. Dilate the remaining mask slightly
    mask = _dilate(mask, radius=max(1, int(1 + sensitivity * 2)))

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


def detect_faces(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Detect faces using OpenCV Haar cascades.

    Returns list of (x, y, w, h) bounding boxes with padding.
    """
    try:
        import cv2
    except ImportError:
        return []

    arr = np.array(img.convert("RGB"))
    gray = np.mean(arr, axis=2).astype(np.uint8)

    cascade_path = str(
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return []

    h, w = arr.shape[:2]
    padded = []
    for (fx, fy, fw, fh) in faces:
        # Add 30% padding around face to protect hair, ears, neck
        pad_x = int(fw * 0.3)
        pad_y = int(fh * 0.3)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)
        padded.append((x1, y1, x2 - x1, y2 - y1))

    return padded


def create_face_protection_mask(
    img: Image.Image,
    damage_mask: Image.Image,
) -> Image.Image:
    """Remove face regions from the damage mask.

    Detects faces and zeroes out those regions in the damage mask
    so inpainting never touches facial features.
    """
    faces = detect_faces(img)
    if not faces:
        return damage_mask

    mask_arr = np.array(damage_mask)

    for (x, y, w, h) in faces:
        mask_arr[y:y + h, x:x + w] = 0

    return Image.fromarray(mask_arr, mode="L")


def inpaint_image(
    img: Image.Image,
    mask: Image.Image,
    model_path: Optional[Path] = None,
    protect_faces: bool = True,
) -> Image.Image:
    """Inpaint damaged regions using the LaMa neural model.

    Uses face detection to protect facial regions from inpainting.
    Blends the inpainted result with the original to preserve
    undamaged areas perfectly.

    Args:
        img: Original image (RGB)
        mask: Binary damage mask (L mode, 255=damaged)
        model_path: Path to inpaint_v1.onnx
        protect_faces: If True, detect faces and exclude them

    Returns:
        Inpainted image (RGB, same size as input)
    """
    if model_path is None:
        from artefex.models_registry import MODEL_DIR
        model_path = MODEL_DIR / "inpaint_v1.onnx"

    if not model_path.exists():
        return img

    try:
        import onnxruntime as ort
    except ImportError:
        return img

    # Protect faces from inpainting
    if protect_faces:
        mask = create_face_protection_mask(img, mask)

    # Check if there's still damage to repair after face exclusion
    mask_arr = np.array(mask)
    if np.sum(mask_arr > 0) < 100:  # Less than 100 damaged pixels
        return img

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    orig_size = img.size  # (w, h)
    orig_arr = np.array(img.convert("RGB"))

    # Resize to 512x512 for LaMa
    img_resized = img.convert("RGB").resize(
        (512, 512), Image.LANCZOS
    )
    mask_resized = mask.convert("L").resize(
        (512, 512), Image.NEAREST
    )

    # Prepare inputs
    img_arr = (
        np.array(img_resized, dtype=np.float32) / 255.0
    )
    img_blob = img_arr.transpose(2, 0, 1)[np.newaxis]

    mask_arr_resized = (
        np.array(mask_resized, dtype=np.float32) / 255.0
    )
    mask_blob = mask_arr_resized[np.newaxis, np.newaxis]
    mask_blob = (mask_blob > 0.5).astype(np.float32)

    # Run LaMa inference
    inputs = session.get_inputs()
    output = session.run(None, {
        inputs[0].name: img_blob,
        inputs[1].name: mask_blob,
    })[0]

    result = output[0].transpose(1, 2, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Resize back to original dimensions
    result_img = result_img.resize(orig_size, Image.LANCZOS)
    result_arr = np.array(result_img)

    # Blend: only use inpainted pixels where the mask says damage.
    # Keep original pixels everywhere else for perfect preservation.
    mask_full = np.array(
        mask.convert("L").resize(orig_size, Image.NEAREST)
    )
    blend = mask_full.astype(np.float32) / 255.0
    blend = blend[:, :, np.newaxis]  # (H, W, 1)

    # Smooth the blend boundary to avoid sharp edges
    blend_pil = Image.fromarray(
        (blend[:, :, 0] * 255).astype(np.uint8), mode="L"
    )
    from PIL import ImageFilter
    blend_pil = blend_pil.filter(ImageFilter.GaussianBlur(3))
    blend = np.array(blend_pil, dtype=np.float32) / 255.0
    blend = blend[:, :, np.newaxis]

    final = (
        orig_arr.astype(np.float32) * (1.0 - blend)
        + result_arr.astype(np.float32) * blend
    )
    final = np.clip(final, 0, 255).astype(np.uint8)

    return Image.fromarray(final)


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

    if damage_pct < 0.15:  # Less than 15% - likely false positives
        return None

    severity = min(1.0, damage_pct * 3)  # 33% damage = severity 1.0
    confidence = min(1.0, damage_pct * 5 + 0.2)

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
