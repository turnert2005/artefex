"""Color palette extraction - extract dominant colors from an image."""

import numpy as np
from PIL import Image


def extract_palette(img: Image.Image, n_colors: int = 8) -> list[dict]:
    """Extract dominant colors using k-means-like clustering.

    Returns list of: [{"rgb": (r, g, b), "hex": "#RRGGBB", "percentage": float}]
    """
    # Resize for speed
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((200, 200))
    arr = np.array(thumb).reshape(-1, 3).astype(np.float64)

    # Simple k-means
    n_pixels = len(arr)
    if n_pixels < n_colors:
        n_colors = n_pixels

    # Initialize centroids with k-means++
    centroids = _kmeans_pp_init(arr, n_colors)

    for _ in range(20):
        # Assign
        dists = np.array([np.sum((arr - c) ** 2, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)

        # Update
        new_centroids = np.zeros_like(centroids)
        for k in range(n_colors):
            mask = labels == k
            if mask.any():
                new_centroids[k] = arr[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1):
            break
        centroids = new_centroids

    # Compute percentages
    dists = np.array([np.sum((arr - c) ** 2, axis=1) for c in centroids])
    labels = np.argmin(dists, axis=0)

    palette = []
    for k in range(n_colors):
        count = np.sum(labels == k)
        pct = count / n_pixels
        r, g, b = int(centroids[k, 0]), int(centroids[k, 1]), int(centroids[k, 2])
        palette.append({
            "rgb": (r, g, b),
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "percentage": round(float(pct * 100), 1),
        })

    palette.sort(key=lambda x: x["percentage"], reverse=True)
    return palette


def _kmeans_pp_init(data: np.ndarray, k: int) -> np.ndarray:
    """K-means++ initialization."""
    n = len(data)
    centroids = [data[np.random.randint(n)]]

    for _ in range(1, k):
        dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        idx = np.random.choice(n, p=probs)
        centroids.append(data[idx])

    return np.array(centroids)


def render_palette_ascii(palette: list[dict]) -> str:
    """Render palette as ASCII art blocks."""
    lines = []
    for color in palette:
        bar_len = max(1, int(color["percentage"] / 2))
        bar = "#" * bar_len
        lines.append(f"  {color['hex']}  {bar} {color['percentage']}%")
    return "\n".join(lines)
