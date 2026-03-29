"""Neural inference engine - runs ONNX models for restoration."""

from typing import Optional

import numpy as np
from PIL import Image

from artefex.models_registry import ModelInfo, ModelRegistry


class NeuralEngine:
    """Loads and runs ONNX neural models for image restoration."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self._sessions: dict[str, object] = {}
        self._onnx_available = self._check_onnx()

    def _check_onnx(self) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("onnxruntime") is not None
        except (ImportError, ValueError):
            return False

    @property
    def available(self) -> bool:
        return self._onnx_available

    def _get_session(self, model_info: ModelInfo):
        """Get or create an ONNX inference session for a model."""
        if not self._onnx_available:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install it with: pip install artefex[neural]"
            )

        if model_info.key not in self._sessions:
            import onnxruntime as ort

            if not model_info.is_available:
                raise FileNotFoundError(
                    f"Model '{model_info.name}' not found at {model_info.local_path}. "
                    f"Import it with: artefex models import {model_info.key} <path>"
                )

            session = ort.InferenceSession(
                str(model_info.local_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._sessions[model_info.key] = session

        return self._sessions[model_info.key]

    def run(self, model_key: str, img: Image.Image) -> Image.Image:
        """Run a neural model on an image, handling tiling for large images."""
        model_info = self.registry.get_model(model_key)
        if model_info is None:
            raise ValueError(f"Unknown model: {model_key}")

        if not model_info.is_available:
            raise FileNotFoundError(f"Model '{model_key}' is not downloaded")

        session = self._get_session(model_info)
        input_h, input_w = model_info.input_size

        # Convert to numpy
        if model_info.channels == 1:
            work_img = img.convert("L")
            arr = np.array(work_img, dtype=np.float32) / 255.0
            arr = arr[np.newaxis, np.newaxis, :, :]  # NCHW
        else:
            work_img = img.convert("RGB")
            arr = np.array(work_img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]  # NCHW

        h, w = arr.shape[2], arr.shape[3]

        # Tile if image is larger than model input
        if h > input_h or w > input_w:
            result = self._run_tiled(session, arr, model_info)
        else:
            result = self._run_padded(session, arr, model_info)

        # Convert back to PIL
        if model_info.channels == 1:
            result_2d = np.clip(result[0, 0] * 255, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(result_2d, mode="L")
            # Merge back with original color
            if img.mode == "RGB":
                orig_rgb = img.convert("YCbCr")
                orig_y, orig_cb, orig_cr = orig_rgb.split()
                result_img = result_img.resize(orig_y.size, Image.LANCZOS)
                merged = Image.merge("YCbCr", (result_img, orig_cb, orig_cr))
                return merged.convert("RGB")
            return result_img
        else:
            result_rgb = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(result_rgb)

    def _run_padded(self, session, arr: np.ndarray, model_info: ModelInfo) -> np.ndarray:
        """Run model on a single image, padding to fit input size."""
        input_h, input_w = model_info.input_size
        h, w = arr.shape[2], arr.shape[3]

        # Pad to input size
        padded = np.zeros(
            (arr.shape[0], arr.shape[1], input_h, input_w), dtype=np.float32
        )
        padded[:, :, :h, :w] = arr

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: padded})[0]

        return output[:, :, :h, :w]

    def _run_tiled(self, session, arr: np.ndarray, model_info: ModelInfo) -> np.ndarray:
        """Run model with overlapping tiles for large images."""
        input_h, input_w = model_info.input_size
        h, w = arr.shape[2], arr.shape[3]
        overlap = 16

        result = np.zeros_like(arr)
        weight = np.zeros((1, 1, h, w), dtype=np.float32)

        for y in range(0, h, input_h - overlap):
            for x in range(0, w, input_w - overlap):
                y_end = min(y + input_h, h)
                x_end = min(x + input_w, w)
                y_start = max(0, y_end - input_h)
                x_start = max(0, x_end - input_w)

                tile = arr[:, :, y_start:y_end, x_start:x_end]

                # Pad if tile is smaller than input size
                if tile.shape[2] < input_h or tile.shape[3] < input_w:
                    padded = np.zeros(
                        (arr.shape[0], arr.shape[1], input_h, input_w),
                        dtype=np.float32,
                    )
                    padded[:, :, : tile.shape[2], : tile.shape[3]] = tile
                    tile = padded

                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: tile})[0]

                out_h = min(input_h, y_end - y_start)
                out_w = min(input_w, x_end - x_start)

                result[:, :, y_start:y_end, x_start:x_end] += output[:, :, :out_h, :out_w]
                weight[:, :, y_start:y_end, x_start:x_end] += 1.0

        # Average overlapping regions
        weight = np.maximum(weight, 1e-8)
        result /= weight

        return result

    def has_model_for(self, category: str) -> bool:
        """Check if a trained neural model is available for a category.

        Returns False for untrained test models to prevent quality
        degradation. Only trained models (> 10 KB) are used.
        """
        model = self.registry.get_model_for_category(category)
        return model is not None and model.is_available and model.is_trained
