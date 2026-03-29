"""Generate lightweight ONNX test models for the Artefex neural pipeline.

Creates small Conv->ReLU->Conv models with a residual skip connection
for each model defined in the Artefex registry. These are functional
but untrained - useful for integration testing only.

Requires: pip install onnx numpy
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# Model definitions matching src/artefex/models_registry.py
MODELS = [
    {
        "filename": "deblock_v1.onnx",
        "in_channels": 1,
        "category": "compression",
        "description": "JPEG deblocking test model (1-ch grayscale)",
    },
    {
        "filename": "denoise_v1.onnx",
        "in_channels": 3,
        "category": "noise",
        "description": "Denoising test model (3-ch RGB)",
    },
    {
        "filename": "sharpen_v1.onnx",
        "in_channels": 3,
        "category": "resolution",
        "description": "Sharpening test model (3-ch RGB)",
    },
    {
        "filename": "color_correct_v1.onnx",
        "in_channels": 3,
        "category": "color",
        "description": "Color correction test model (3-ch RGB)",
    },
]

HIDDEN_CHANNELS = 8
KERNEL_SIZE = 3
OPSET_VERSION = 17
DEFAULT_MODEL_DIR = Path.home() / ".artefex" / "models"


def _make_conv_weights(
    out_ch: int, in_ch: int, kernel: int, rng: np.random.Generator
) -> np.ndarray:
    """Create small random conv weights with shape (out_ch, in_ch, kernel, kernel)."""
    scale = 0.1 / np.sqrt(in_ch * kernel * kernel)
    return (rng.standard_normal((out_ch, in_ch, kernel, kernel)) * scale).astype(
        np.float32
    )


def _make_bias(channels: int, rng: np.random.Generator) -> np.ndarray:
    """Create small random bias values."""
    return (rng.standard_normal(channels) * 0.01).astype(np.float32)


def build_model(in_channels: int, seed: int = 42) -> onnx.ModelProto:
    """Build a Conv->ReLU->Conv + residual skip ONNX model.

    Architecture:
        input -> Conv1(in_ch, 8, 3x3, pad=1) -> ReLU -> Conv2(8, in_ch, 3x3, pad=1)
                   |                                                          |
                   +----------------------------------------------------------+
                                           Add -> output

    Args:
        in_channels: Number of input (and output) channels.
        seed: Random seed for weight initialization.

    Returns:
        A checked onnx.ModelProto ready to save.
    """
    rng = np.random.default_rng(seed)
    hidden = HIDDEN_CHANNELS
    k = KERNEL_SIZE
    pad = k // 2

    # --- Weight tensors ---
    w1 = numpy_helper.from_array(
        _make_conv_weights(hidden, in_channels, k, rng), name="conv1_w"
    )
    b1 = numpy_helper.from_array(_make_bias(hidden, rng), name="conv1_b")
    w2 = numpy_helper.from_array(
        _make_conv_weights(in_channels, hidden, k, rng), name="conv2_w"
    )
    b2 = numpy_helper.from_array(_make_bias(in_channels, rng), name="conv2_b")

    # --- Nodes ---
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "conv1_w", "conv1_b"],
        outputs=["conv1_out"],
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        name="conv1",
    )
    relu = helper.make_node(
        "Relu",
        inputs=["conv1_out"],
        outputs=["relu_out"],
        name="relu1",
    )
    conv2 = helper.make_node(
        "Conv",
        inputs=["relu_out", "conv2_w", "conv2_b"],
        outputs=["conv2_out"],
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        name="conv2",
    )
    add = helper.make_node(
        "Add",
        inputs=["input", "conv2_out"],
        outputs=["output"],
        name="residual_add",
    )

    # --- Graph I/O with dynamic spatial axes ---
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, in_channels, "height", "width"]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, in_channels, "height", "width"]
    )

    graph = helper.make_graph(
        nodes=[conv1, relu, conv2, add],
        name="residual_conv_net",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[w1, b1, w2, b2],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def create_all_models(output_dir: Path) -> list[Path]:
    """Create all test models and save them to output_dir.

    Args:
        output_dir: Directory to write .onnx files into.

    Returns:
        List of paths to the created model files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for i, spec in enumerate(MODELS):
        model = build_model(in_channels=spec["in_channels"], seed=42 + i)
        path = output_dir / spec["filename"]
        onnx.save(model, str(path))
        size_kb = path.stat().st_size / 1024
        print(
            f"  Created {spec['filename']} "
            f"({spec['in_channels']}ch, {spec['category']}) "
            f"- {size_kb:.1f} KB"
        )
        created.append(path)

    return created


def install_models(source_dir: Path, install_dir: Path) -> None:
    """Copy generated models to the install directory.

    Args:
        source_dir: Directory containing the generated .onnx files.
        install_dir: Target directory (typically ~/.artefex/models/).
    """
    install_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        src = source_dir / spec["filename"]
        if not src.exists():
            print(f"  WARNING: {spec['filename']} not found in {source_dir}")
            continue
        dst = install_dir / spec["filename"]
        shutil.copy2(src, dst)
        print(f"  Installed {spec['filename']} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lightweight ONNX test models for Artefex."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models",
        help=(
            "Directory to write generated models. "
            "Defaults to train/models/ next to this script."
        ),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help=(
            "Copy generated models to ~/.artefex/models/ "
            "so they are discoverable by the Artefex model registry."
        ),
    )
    args = parser.parse_args()

    print(f"Generating test models in: {args.output}")
    create_all_models(args.output)
    print("Done.\n")

    if args.install:
        print(f"Installing models to: {DEFAULT_MODEL_DIR}")
        install_models(args.output, DEFAULT_MODEL_DIR)
        print("Install complete.")


if __name__ == "__main__":
    main()
