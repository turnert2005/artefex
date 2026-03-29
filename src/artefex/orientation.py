"""Image orientation detection and auto-correction.

Detects and corrects image orientation issues:
1. EXIF orientation tags (cameras store rotation in metadata)
2. Horizon detection (straighten tilted photos)
3. Upside-down detection using content analysis
"""

import numpy as np
from PIL import Image, ExifTags


# EXIF orientation tag values and their transformations
EXIF_ORIENTATIONS = {
    1: "Normal",
    2: "Mirrored horizontal",
    3: "Rotated 180",
    4: "Mirrored vertical",
    5: "Mirrored horizontal + rotated 270",
    6: "Rotated 90 CW",
    7: "Mirrored horizontal + rotated 90",
    8: "Rotated 270 CW",
}


def detect_orientation(img: Image.Image) -> dict:
    """Detect orientation issues in an image.

    Returns dict with:
        exif_orientation: int or None
        exif_description: str
        needs_correction: bool
        suggested_rotation: int (degrees)
        horizon_tilt: float (degrees, estimated)
    """
    result = {
        "exif_orientation": None,
        "exif_description": "Normal",
        "needs_correction": False,
        "suggested_rotation": 0,
        "horizon_tilt": 0.0,
    }

    # Check EXIF orientation
    try:
        exif = img.getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    result["exif_orientation"] = val
                    result["exif_description"] = EXIF_ORIENTATIONS.get(val, f"Unknown ({val})")
                    if val != 1:
                        result["needs_correction"] = True
                    break
    except Exception:
        pass

    # Estimate horizon tilt using edge detection
    try:
        arr = np.array(img.convert("L"), dtype=np.float64)
        tilt = _estimate_horizon_tilt(arr)
        result["horizon_tilt"] = round(tilt, 2)
        if abs(tilt) > 2.0:
            result["needs_correction"] = True
    except Exception:
        pass

    # Map EXIF orientation to rotation
    rotation_map = {1: 0, 2: 0, 3: 180, 4: 0, 5: 270, 6: 90, 7: 90, 8: 270}
    if result["exif_orientation"]:
        result["suggested_rotation"] = rotation_map.get(result["exif_orientation"], 0)

    return result


def auto_orient(img: Image.Image) -> tuple[Image.Image, dict]:
    """Auto-correct image orientation based on EXIF and content analysis.

    Returns (corrected_image, info_dict).
    """
    info = detect_orientation(img)
    corrected = img.copy()

    # Apply EXIF-based correction
    exif_orient = info["exif_orientation"]
    if exif_orient and exif_orient != 1:
        if exif_orient == 2:
            corrected = corrected.transpose(Image.FLIP_LEFT_RIGHT)
        elif exif_orient == 3:
            corrected = corrected.rotate(180, expand=True)
        elif exif_orient == 4:
            corrected = corrected.transpose(Image.FLIP_TOP_BOTTOM)
        elif exif_orient == 5:
            corrected = corrected.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif exif_orient == 6:
            corrected = corrected.rotate(270, expand=True)
        elif exif_orient == 7:
            corrected = corrected.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif exif_orient == 8:
            corrected = corrected.rotate(90, expand=True)

        info["applied"] = f"EXIF correction: {EXIF_ORIENTATIONS.get(exif_orient, 'unknown')}"
    else:
        info["applied"] = "No EXIF correction needed"

    # Apply horizon straightening if tilt > 2 degrees
    if abs(info["horizon_tilt"]) > 2.0:
        corrected = corrected.rotate(-info["horizon_tilt"], expand=True, fillcolor=(0, 0, 0))
        info["applied"] += f", horizon straightened by {info['horizon_tilt']}deg"

    return corrected, info


def _estimate_horizon_tilt(gray: np.ndarray) -> float:
    """Estimate horizon tilt angle using Hough-like edge analysis."""
    h, w = gray.shape
    if h < 50 or w < 50:
        return 0.0

    # Compute horizontal edges
    edges = np.abs(np.diff(gray, axis=0))

    # Focus on the middle third of the image (where horizons typically are)
    mid_start = h // 3
    mid_end = 2 * h // 3
    mid_edges = edges[mid_start:mid_end, :]

    if mid_edges.size == 0:
        return 0.0

    # Find strong edge rows
    row_strengths = mid_edges.mean(axis=1)
    threshold = np.percentile(row_strengths, 80)
    strong_rows = np.where(row_strengths > threshold)[0]

    if len(strong_rows) < 2:
        return 0.0

    # For each strong edge row, find the weighted center
    centers = []
    for row in strong_rows[:20]:
        edge_row = mid_edges[row]
        if edge_row.sum() > 0:
            x_coords = np.arange(len(edge_row))
            center = np.average(x_coords, weights=edge_row)
            centers.append((row, center))

    if len(centers) < 2:
        return 0.0

    # Fit a line to estimate tilt
    rows = np.array([c[0] for c in centers])
    cols = np.array([c[1] for c in centers])

    if cols.std() < 1:
        return 0.0

    # Simple linear regression
    slope = np.polyfit(cols, rows, 1)[0]

    # Convert slope to angle in degrees
    angle = np.degrees(np.arctan(slope))

    # Clamp to reasonable range
    return max(-15, min(15, angle))
