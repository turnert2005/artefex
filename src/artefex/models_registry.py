"""Model registry - manages neural model discovery, download, and loading."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODEL_DIR = Path.home() / ".artefex" / "models"

REGISTRY = {
    "deblock-v1": {
        "name": "JPEG Deblocking v1",
        "description": "Removes JPEG 8x8 block artifacts using a lightweight CNN",
        "filename": "deblock_v1.onnx",
        "input_size": (256, 256),
        "channels": 1,
        "category": "compression",
        "version": "1.0.0",
    },
    "denoise-v1": {
        "name": "Adaptive Denoiser v1",
        "description": "Edge-preserving neural denoiser",
        "filename": "denoise_v1.onnx",
        "input_size": (256, 256),
        "channels": 3,
        "category": "noise",
        "version": "1.0.0",
    },
    "sharpen-v1": {
        "name": "Detail Recovery v1",
        "description": "Recovers high-frequency detail lost to downscaling",
        "filename": "sharpen_v1.onnx",
        "input_size": (256, 256),
        "channels": 3,
        "category": "resolution",
        "version": "1.0.0",
    },
    "color-correct-v1": {
        "name": "Color Correction v1",
        "description": "Corrects color shifts from format conversions",
        "filename": "color_correct_v1.onnx",
        "input_size": (256, 256),
        "channels": 3,
        "category": "color",
        "version": "1.0.0",
    },
}


@dataclass
class ModelInfo:
    key: str
    name: str
    description: str
    filename: str
    input_size: tuple[int, int]
    channels: int
    category: str
    version: str
    local_path: Optional[Path] = None

    @property
    def is_available(self) -> bool:
        return self.local_path is not None and self.local_path.exists()


class ModelRegistry:
    """Discovers, manages, and loads neural models for restoration."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[ModelInfo]:
        """List all known models and their availability."""
        models = []
        for key, info in REGISTRY.items():
            local_path = self.model_dir / info["filename"]
            models.append(
                ModelInfo(
                    key=key,
                    name=info["name"],
                    description=info["description"],
                    filename=info["filename"],
                    input_size=tuple(info["input_size"]),
                    channels=info["channels"],
                    category=info["category"],
                    version=info["version"],
                    local_path=local_path if local_path.exists() else None,
                )
            )
        return models

    def get_model(self, key: str) -> Optional[ModelInfo]:
        """Get info about a specific model."""
        if key not in REGISTRY:
            return None
        info = REGISTRY[key]
        local_path = self.model_dir / info["filename"]
        return ModelInfo(
            key=key,
            name=info["name"],
            description=info["description"],
            filename=info["filename"],
            input_size=tuple(info["input_size"]),
            channels=info["channels"],
            category=info["category"],
            version=info["version"],
            local_path=local_path if local_path.exists() else None,
        )

    def get_model_for_category(self, category: str) -> Optional[ModelInfo]:
        """Find the best model for a degradation category."""
        for key, info in REGISTRY.items():
            if info["category"] == category:
                return self.get_model(key)
        return None

    def import_model(self, source_path: Path, key: str) -> ModelInfo:
        """Import a model file into the registry."""
        if key not in REGISTRY:
            raise ValueError(f"Unknown model key: {key}. Known keys: {list(REGISTRY.keys())}")

        info = REGISTRY[key]
        dest = self.model_dir / info["filename"]

        import shutil
        shutil.copy2(source_path, dest)

        return ModelInfo(
            key=key,
            name=info["name"],
            description=info["description"],
            filename=info["filename"],
            input_size=tuple(info["input_size"]),
            channels=info["channels"],
            category=info["category"],
            version=info["version"],
            local_path=dest,
        )

    def model_path(self, key: str) -> Optional[Path]:
        """Get the local path for a model, or None if not available."""
        model = self.get_model(key)
        if model and model.is_available:
            return model.local_path
        return None
