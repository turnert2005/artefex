"""Plugin system for community detectors and restorers.

Plugins are Python packages that register themselves via entry points
in their pyproject.toml:

    [project.entry-points."artefex.detectors"]
    my_detector = "my_package:MyDetector"

    [project.entry-points."artefex.restorers"]
    my_restorer = "my_package:MyRestorer"

Detector plugins must implement:
    class MyDetector:
        name: str  - unique name for this detector
        def detect(self, img: PIL.Image, arr: np.ndarray) -> Degradation | None

Restorer plugins must implement:
    class MyRestorer:
        name: str  - must match a degradation name
        def restore(self, img: PIL.Image, degradation: Degradation) -> PIL.Image
"""

import importlib.metadata
from typing import Optional

import numpy as np
from PIL import Image

from artefex.models import Degradation


DETECTOR_GROUP = "artefex.detectors"
RESTORER_GROUP = "artefex.restorers"


class PluginRegistry:
    """Discovers and manages plugins via entry points."""

    def __init__(self):
        self._detectors: dict[str, object] = {}
        self._restorers: dict[str, object] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return

        # Load detector plugins
        for ep in importlib.metadata.entry_points(group=DETECTOR_GROUP):
            try:
                cls = ep.load()
                instance = cls()
                self._detectors[instance.name] = instance
            except Exception as e:
                print(f"Warning: failed to load detector plugin '{ep.name}': {e}")

        # Load restorer plugins
        for ep in importlib.metadata.entry_points(group=RESTORER_GROUP):
            try:
                cls = ep.load()
                instance = cls()
                self._restorers[instance.name] = instance
            except Exception as e:
                print(f"Warning: failed to load restorer plugin '{ep.name}': {e}")

        self._loaded = True

    @property
    def detectors(self) -> dict[str, object]:
        self._load()
        return self._detectors

    @property
    def restorers(self) -> dict[str, object]:
        self._load()
        return self._restorers

    def run_detectors(
        self, img: Image.Image, arr: np.ndarray
    ) -> list[Degradation]:
        """Run all plugin detectors and return any findings."""
        results = []
        for name, detector in self.detectors.items():
            try:
                degradation = detector.detect(img, arr)
                if degradation is not None:
                    results.append(degradation)
            except Exception as e:
                print(f"Warning: detector plugin '{name}' failed: {e}")
        return results

    def run_restorer(
        self, img: Image.Image, degradation: Degradation
    ) -> Optional[Image.Image]:
        """Try to restore using a plugin restorer. Returns None if no plugin handles it."""
        restorer = self.restorers.get(degradation.name)
        if restorer is None:
            return None
        try:
            return restorer.restore(img, degradation)
        except Exception as e:
            print(f"Warning: restorer plugin '{degradation.name}' failed: {e}")
            return None

    def list_plugins(self) -> dict:
        """Return summary of loaded plugins."""
        self._load()
        return {
            "detectors": list(self._detectors.keys()),
            "restorers": list(self._restorers.keys()),
        }


# Global singleton
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
