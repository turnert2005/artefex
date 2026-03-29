"""Build a distributable Windows package of Artefex using PyInstaller."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DIST_DIR = Path("dist")
BUILD_DIR = Path("build")
APP_NAME = "Artefex"
MODELS_DIR = Path.home() / ".artefex" / "models"


def ensure_pyinstaller():
    """Install PyInstaller if it is not already available."""
    try:
        import PyInstaller  # noqa: F401
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller"]
        )


def find_onnx_models():
    """Find ONNX model files in the user model directory."""
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.onnx"):
            models.append(f)
        if models:
            print(f"Found {len(models)} ONNX model(s) in {MODELS_DIR}")
        else:
            print(f"No ONNX models found in {MODELS_DIR}")
    else:
        print(f"Models directory not found: {MODELS_DIR}")
        print("The build will continue without bundled models.")
    return models


def build_pyinstaller_args(models):
    """Construct the PyInstaller command line arguments."""
    args = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--console",
        # Hidden imports for the web stack and runtime
        "--hidden-import", "artefex",
        "--hidden-import", "artefex.web",
        "--hidden-import", "artefex.analyze",
        "--hidden-import", "artefex.restore",
        "--hidden-import", "artefex.neural",
        "--hidden-import", "artefex.models",
        "--hidden-import", "artefex.models_registry",
        "--hidden-import", "artefex.report",
        "--hidden-import", "artefex.report_html",
        "--hidden-import", "artefex.heatmap",
        "--hidden-import", "artefex.grade",
        "--hidden-import", "artefex.config",
        "--hidden-import", "artefex.plugins",
        "--hidden-import", "artefex.api",
        "--hidden-import", "artefex.story",
        "--hidden-import", "artefex.gallery",
        "--hidden-import", "artefex.dashboard",
        "--hidden-import", "artefex.fingerprint",
        "--hidden-import", "artefex.detect_aigen",
        "--hidden-import", "artefex.detect_stego",
        "--hidden-import", "artefex.detect_camera",
        "--hidden-import", "artefex.detect_forgery",
        "--hidden-import", "artefex.parallel",
        "--hidden-import", "artefex.gif_analyze",
        "--hidden-import", "artefex.video",
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "uvicorn.lifespan.off",
        "--hidden-import", "fastapi",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "multipart",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "typer",
        "--hidden-import", "rich",
        # Collect all submodules for packages that need it
        "--collect-submodules", "artefex",
        "--collect-submodules", "uvicorn",
        "--collect-submodules", "fastapi",
    ]

    # Add ONNX models as data files
    for model_path in models:
        # Bundle into models/ subfolder in the dist
        args.extend([
            "--add-data",
            f"{model_path}{os.pathsep}models",
        ])

    # Entry point
    args.append("launcher.py")
    return args


def create_readme(output_dir):
    """Create a README file in the distribution folder."""
    readme_path = output_dir / "README_INSTALL.txt"
    readme_path.write_text(
        "Artefex - Neural Forensic Restoration\n"
        "======================================\n"
        "\n"
        "Getting Started\n"
        "---------------\n"
        "1. Extract this zip to any folder.\n"
        "2. Double-click Artefex.exe to start.\n"
        "3. Your browser will open automatically.\n"
        "4. Drag and drop images to analyze them.\n"
        "5. Close the terminal window to stop the server.\n"
        "\n"
        "Troubleshooting\n"
        "---------------\n"
        "- If Windows SmartScreen blocks the app, click\n"
        '  "More info" then "Run anyway".\n'
        "- If port 8787 is busy, the app will pick the\n"
        "  next available port automatically.\n"
        "- Make sure no antivirus is blocking the app.\n"
        "\n"
        "For more information visit:\n"
        "  https://github.com/turnert2005/artefex\n",
        encoding="utf-8",
    )
    print(f"Created {readme_path}")


def create_zip(source_dir, zip_path):
    """Create a zip archive from the distribution folder."""
    print(f"Creating {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = f"{APP_NAME}/{file_path.relative_to(source_dir)}"
                zf.write(file_path, arcname)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Done - {zip_path} ({size_mb:.1f} MB)")


def main():
    """Run the full build process."""
    print("=" * 52)
    print("  Artefex Windows Build")
    print("=" * 52)
    print()

    ensure_pyinstaller()
    models = find_onnx_models()

    # Clean previous builds
    app_dist = DIST_DIR / APP_NAME
    if app_dist.exists():
        print(f"Removing old build at {app_dist} ...")
        shutil.rmtree(app_dist)

    # Run PyInstaller
    args = build_pyinstaller_args(models)
    print()
    print("Running PyInstaller (this may take a few minutes) ...")
    print()
    subprocess.check_call(args)

    # Copy models into dist if they were not bundled via --add-data
    # (belt-and-suspenders approach)
    models_dest = app_dist / "models"
    if models and not models_dest.exists():
        models_dest.mkdir(parents=True)
        for m in models:
            shutil.copy2(m, models_dest / m.name)
            print(f"Copied model: {m.name}")

    # Create README inside the distribution
    create_readme(app_dist)

    # Zip it up
    zip_path = DIST_DIR / "Artefex-Windows.zip"
    if zip_path.exists():
        zip_path.unlink()
    create_zip(app_dist, zip_path)

    # Try to build installer if Inno Setup is available
    build_installer(app_dist)

    print()
    print("Build complete!")
    print(f"  Folder: {app_dist.resolve()}")
    print(f"  Zip:    {zip_path.resolve()}")


def build_installer(app_dist):
    """Build a Windows installer using Inno Setup if available."""
    iss_path = Path("installer/artefex_setup.iss")
    if not iss_path.exists():
        return

    # Search for Inno Setup compiler
    iscc_paths = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    iscc = None
    for p in iscc_paths:
        if p.exists():
            iscc = p
            break

    if iscc is None:
        print()
        print(
            "Inno Setup not found - skipping installer build."
        )
        print(
            "Install from https://jrsoftware.org/isinfo.php "
            "to build the .exe installer."
        )
        return

    print()
    print("Building Windows installer with Inno Setup ...")
    try:
        subprocess.check_call([str(iscc), str(iss_path)])
        setup_exe = DIST_DIR / "Artefex-1.0.0-Setup.exe"
        if setup_exe.exists():
            size = setup_exe.stat().st_size / (1024 * 1024)
            print(f"  Installer: {setup_exe} ({size:.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"  Installer build failed: {e}")


if __name__ == "__main__":
    main()
