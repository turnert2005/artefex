"""Image similarity and duplicate detection using perceptual hashing."""

from pathlib import Path

import numpy as np
from PIL import Image


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute perceptual hash (pHash) of an image.

    Resizes to small grayscale, applies DCT, and thresholds to produce
    a compact binary hash. Similar images produce similar hashes.
    """
    # Resize to hash_size*4 x hash_size*4 then DCT
    size = hash_size * 4
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)

    # Simple DCT approximation using numpy
    from numpy.fft import fft2
    dct = np.abs(fft2(arr))[:hash_size, :hash_size]

    # Exclude the DC component
    dct[0, 0] = 0
    median = np.median(dct)

    # Threshold
    bits = (dct > median).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def ahash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute average hash (aHash) of an image."""
    img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    mean = arr.mean()

    bits = (arr > mean).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) of an image."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)

    # Compare adjacent pixels
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Count differing bits between two hashes."""
    x = hash1 ^ hash2
    count = 0
    while x:
        count += 1
        x &= x - 1
    return count


def similarity_score(hash1: int, hash2: int, hash_size: int = 8) -> float:
    """Compute similarity between two hashes (0.0 = different, 1.0 = identical)."""
    max_dist = hash_size * hash_size
    dist = hamming_distance(hash1, hash2)
    return 1.0 - (dist / max_dist)


def find_duplicates(
    files: list[Path],
    threshold: float = 0.9,
    hash_fn: str = "phash",
    on_progress=None,
) -> list[dict]:
    """Find duplicate or near-duplicate images in a list of files.

    Returns list of groups: [{"files": [path1, path2, ...], "similarity": float}]
    """
    hash_func = {"phash": phash, "ahash": ahash, "dhash": dhash}.get(hash_fn, phash)

    # Compute hashes
    hashes = []
    for i, file in enumerate(files):
        try:
            img = Image.open(file)
            h = hash_func(img)
            hashes.append((file, h))
        except Exception:
            pass

        if on_progress:
            on_progress(i + 1, len(files))

    # Find groups
    used = set()
    groups = []

    for i in range(len(hashes)):
        if i in used:
            continue

        group_files = [hashes[i][0]]
        min_sim = 1.0

        for j in range(i + 1, len(hashes)):
            if j in used:
                continue

            sim = similarity_score(hashes[i][1], hashes[j][1])
            if sim >= threshold:
                group_files.append(hashes[j][0])
                min_sim = min(min_sim, sim)
                used.add(j)

        if len(group_files) > 1:
            used.add(i)
            groups.append({
                "files": [str(f) for f in group_files],
                "similarity": round(min_sim, 3),
            })

    return groups
