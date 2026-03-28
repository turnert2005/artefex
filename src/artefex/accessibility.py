"""Color accessibility checker - simulate color blindness and check contrast.

Simulates how an image appears to people with different types of color vision
deficiency (CVD) and checks if the image meets accessibility standards.

Types:
- Protanopia (no red cones, ~1% of males)
- Deuteranopia (no green cones, ~1% of males)
- Tritanopia (no blue cones, very rare)
- Achromatopsia (total color blindness, very rare)
"""

from pathlib import Path

import numpy as np
from PIL import Image


# Color blindness simulation matrices (Brettel et al.)
# These transform RGB to simulate how people with CVD see colors

PROTANOPIA_MATRIX = np.array([
    [0.567, 0.433, 0.000],
    [0.558, 0.442, 0.000],
    [0.000, 0.242, 0.758],
])

DEUTERANOPIA_MATRIX = np.array([
    [0.625, 0.375, 0.000],
    [0.700, 0.300, 0.000],
    [0.000, 0.300, 0.700],
])

TRITANOPIA_MATRIX = np.array([
    [0.950, 0.050, 0.000],
    [0.000, 0.433, 0.567],
    [0.000, 0.475, 0.525],
])

ACHROMATOPSIA_MATRIX = np.array([
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114],
])

SIMULATION_TYPES = {
    "protanopia": ("Protanopia (no red)", PROTANOPIA_MATRIX),
    "deuteranopia": ("Deuteranopia (no green)", DEUTERANOPIA_MATRIX),
    "tritanopia": ("Tritanopia (no blue)", TRITANOPIA_MATRIX),
    "achromatopsia": ("Achromatopsia (no color)", ACHROMATOPSIA_MATRIX),
}


def simulate_cvd(img: Image.Image, cvd_type: str) -> Image.Image:
    """Simulate how an image appears with a specific color vision deficiency.

    Args:
        img: Input image.
        cvd_type: One of "protanopia", "deuteranopia", "tritanopia", "achromatopsia".

    Returns:
        Simulated image.
    """
    if cvd_type not in SIMULATION_TYPES:
        raise ValueError(f"Unknown CVD type: {cvd_type}. Use: {list(SIMULATION_TYPES.keys())}")

    _, matrix = SIMULATION_TYPES[cvd_type]

    arr = np.array(img.convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = arr.shape

    # Apply transformation matrix
    flat = arr.reshape(-1, 3)
    transformed = flat @ matrix.T
    transformed = np.clip(transformed, 0, 1)

    result = (transformed.reshape(h, w, 3) * 255).astype(np.uint8)
    return Image.fromarray(result)


def check_accessibility(img: Image.Image) -> dict:
    """Check image accessibility for different types of color blindness.

    Returns dict with:
        - information_loss: dict of CVD type -> percentage of color information lost
        - contrast_ratio: estimated foreground/background contrast
        - recommendations: list of suggestions
    """
    arr = np.array(img.convert("RGB"), dtype=np.float64)

    info_loss = {}
    for cvd_type, (name, matrix) in SIMULATION_TYPES.items():
        simulated = simulate_cvd(img, cvd_type)
        sim_arr = np.array(simulated, dtype=np.float64)

        # Measure color difference (Delta E approximation)
        diff = np.sqrt(np.sum((arr - sim_arr) ** 2, axis=2))
        mean_diff = diff.mean()

        # Normalize: max possible diff is sqrt(255^2 * 3) = 441.67
        loss_pct = mean_diff / 441.67
        info_loss[cvd_type] = {
            "name": name,
            "loss_pct": round(float(loss_pct * 100), 1),
            "mean_color_diff": round(float(mean_diff), 1),
        }

    # Contrast analysis
    gray = np.mean(arr, axis=2)
    p10 = np.percentile(gray, 10)
    p90 = np.percentile(gray, 90)

    # WCAG contrast ratio approximation
    l1 = (p90 / 255 + 0.05)
    l2 = (p10 / 255 + 0.05)
    contrast_ratio = l1 / l2 if l2 > 0 else 0

    # Recommendations
    recommendations = []
    for cvd_type, data in info_loss.items():
        if data["loss_pct"] > 15:
            recommendations.append(
                f"High information loss for {data['name']} ({data['loss_pct']}%). "
                f"Consider adding patterns/textures alongside color coding."
            )

    if contrast_ratio < 4.5:
        recommendations.append(
            f"Low contrast ratio ({contrast_ratio:.1f}:1). "
            f"WCAG AA requires 4.5:1 for text. Consider increasing contrast."
        )

    return {
        "information_loss": info_loss,
        "contrast_ratio": round(float(contrast_ratio), 2),
        "wcag_aa_pass": contrast_ratio >= 4.5,
        "recommendations": recommendations,
    }


def generate_cvd_comparison(
    file_path: Path,
    output_dir: Path,
) -> dict:
    """Generate comparison images for all CVD types.

    Returns dict with output paths for each simulation.
    """
    img = Image.open(file_path).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for cvd_type in SIMULATION_TYPES:
        simulated = simulate_cvd(img, cvd_type)
        out_path = output_dir / f"{file_path.stem}_{cvd_type}.png"
        simulated.save(out_path)
        outputs[cvd_type] = str(out_path)

    return outputs
