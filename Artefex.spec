# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['artefex', 'artefex.web', 'artefex.analyze', 'artefex.restore', 'artefex.neural', 'artefex.models', 'artefex.models_registry', 'artefex.report', 'artefex.report_html', 'artefex.heatmap', 'artefex.grade', 'artefex.config', 'artefex.plugins', 'artefex.api', 'artefex.story', 'artefex.gallery', 'artefex.dashboard', 'artefex.fingerprint', 'artefex.detect_aigen', 'artefex.detect_stego', 'artefex.detect_camera', 'artefex.detect_forgery', 'artefex.parallel', 'artefex.gif_analyze', 'artefex.video', 'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off', 'fastapi', 'onnxruntime', 'multipart', 'PIL', 'numpy', 'typer', 'rich']
hiddenimports += collect_submodules('artefex')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('fastapi')


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\scott\\.artefex\\models\\aigen_detect_v1.onnx', 'models'), ('C:\\Users\\scott\\.artefex\\models\\color_correct_v1.onnx', 'models'), ('C:\\Users\\scott\\.artefex\\models\\deblock_v1.onnx', 'models'), ('C:\\Users\\scott\\.artefex\\models\\denoise_v1.onnx', 'models'), ('C:\\Users\\scott\\.artefex\\models\\sharpen_v1.onnx', 'models')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'onnxscript', 'sympy', 'pytest', 'setuptools'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Artefex',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Artefex',
)
