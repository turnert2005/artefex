"""Copy-move forgery detection - finds cloned/duplicated regions within an image.

A common image manipulation technique is to copy a region of an image and
paste it elsewhere to hide or duplicate objects. This module detects such
manipulations by finding similar patches that appear in different locations.

Method:
1. Extract overlapping patches across the image
2. Compute compact feature vectors for each patch
3. Find suspiciously similar patches that are spatially distant
4. Cluster matches into forgery regions
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from artefex.models import Degradation


@dataclass
class ForgeryRegion:
    """A detected copy-move forgery region."""
    source: tuple[int, int]  # (x, y) center of source region
    target: tuple[int, int]  # (x, y) center of cloned region
    similarity: float
    patch_size: int


class CopyMoveDetector:
    """Detects copy-move forgery by finding duplicated image regions."""

    def __init__(self, patch_size: int = 32, stride: int = 16, threshold: float = 0.95):
        self.patch_size = patch_size
        self.stride = stride
        self.threshold = threshold

    def detect(self, img: Image.Image, arr: np.ndarray) -> Degradation | None:
        if len(arr.shape) < 3:
            return None

        h, w = arr.shape[:2]
        if h < self.patch_size * 3 or w < self.patch_size * 3:
            return None

        regions = self._find_cloned_regions(arr)

        if not regions:
            return None

        n = len(regions)
        avg_sim = np.mean([r.similarity for r in regions])

        detail_parts = [f"{n} cloned region(s) detected, avg similarity {avg_sim:.2f}"]
        for r in regions[:3]:
            detail_parts.append(
                f"({r.source[0]},{r.source[1]}) -> ({r.target[0]},{r.target[1]}) "
                f"sim={r.similarity:.3f}"
            )

        return Degradation(
            name="Copy-Move Forgery",
            confidence=min(1.0, avg_sim * 0.9),
            severity=min(0.8, n * 0.15),
            detail="; ".join(detail_parts),
            category="forgery",
        )

    def _find_cloned_regions(self, arr: np.ndarray) -> list[ForgeryRegion]:
        """Extract patches, compute features, find matches."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)
        h, w = gray.shape

        # Extract patches and compute feature vectors
        patches = []
        positions = []

        for y in range(0, h - self.patch_size, self.stride):
            for x in range(0, w - self.patch_size, self.stride):
                patch = gray[y:y + self.patch_size, x:x + self.patch_size]
                feat = self._patch_features(patch)
                patches.append(feat)
                positions.append((x + self.patch_size // 2, y + self.patch_size // 2))

        if len(patches) < 10:
            return []

        patches = np.array(patches)
        positions = np.array(positions)

        # Normalize features
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        patches_norm = patches / norms

        # Find similar pairs using sorted feature matching
        # Sort by first feature for efficient search
        sort_idx = np.argsort(patches_norm[:, 0])
        sorted_feats = patches_norm[sort_idx]
        sorted_pos = positions[sort_idx]

        regions = []
        min_spatial_dist = self.patch_size * 2  # Must be spatially distant

        # Sliding window comparison on sorted features
        window = min(50, len(sorted_feats))
        for i in range(len(sorted_feats)):
            for j in range(i + 1, min(i + window, len(sorted_feats))):
                # Quick reject on first feature
                if abs(sorted_feats[i, 0] - sorted_feats[j, 0]) > 0.1:
                    break

                # Full similarity
                sim = np.dot(sorted_feats[i], sorted_feats[j])

                if sim >= self.threshold:
                    # Check spatial distance
                    dx = sorted_pos[i, 0] - sorted_pos[j, 0]
                    dy = sorted_pos[i, 1] - sorted_pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)

                    if dist > min_spatial_dist:
                        regions.append(ForgeryRegion(
                            source=tuple(sorted_pos[i]),
                            target=tuple(sorted_pos[j]),
                            similarity=float(sim),
                            patch_size=self.patch_size,
                        ))

            if len(regions) >= 20:  # Cap for performance
                break

        # Deduplicate overlapping detections
        return self._deduplicate(regions)

    def _patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Compute compact feature vector for a patch."""
        # Use a combination of:
        # 1. Block means (4x4 grid = 16 features)
        # 2. Edge statistics (4 features)
        # 3. Texture (4 features)

        ps = self.patch_size
        bs = ps // 4
        features = []

        # Block means
        for by in range(4):
            for bx in range(4):
                block = patch[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs]
                features.append(block.mean())

        # Edge features
        dx = np.abs(np.diff(patch, axis=1))
        dy = np.abs(np.diff(patch, axis=0))
        features.append(dx.mean())
        features.append(dy.mean())
        features.append(dx.std())
        features.append(dy.std())

        # Texture (variance in blocks)
        for by in range(2):
            for bx in range(2):
                block = patch[by * (ps // 2):(by + 1) * (ps // 2),
                              bx * (ps // 2):(bx + 1) * (ps // 2)]
                features.append(block.var())

        return np.array(features)

    def _deduplicate(self, regions: list[ForgeryRegion]) -> list[ForgeryRegion]:
        """Remove overlapping detections."""
        if not regions:
            return []

        # Sort by similarity
        regions.sort(key=lambda r: r.similarity, reverse=True)

        kept = []
        for r in regions:
            overlap = False
            for k in kept:
                # Check if source or target overlaps
                for p1, p2 in [(r.source, k.source), (r.target, k.target),
                               (r.source, k.target), (r.target, k.source)]:
                    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if dist < self.patch_size:
                        overlap = True
                        break
                if overlap:
                    break

            if not overlap:
                kept.append(r)

        return kept
