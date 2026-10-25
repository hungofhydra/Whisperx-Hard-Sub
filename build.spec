# PyInstaller spec for WhisperX Hardsub Studio.
#
# Build (on the target Windows machine, inside your venv):
#     pyinstaller build.spec
#
# Output goes to dist/WhisperXHardsubStudio/ (onedir build - deliberately
# NOT onefile: torch/ctranslate2 are large and onefile mode re-extracts
# everything into a temp dir on every launch, which is slow and fragile
# for ML apps. Zip the whole dist/WhisperXHardsubStudio folder and that
# IS your distributable "exe" package - the .exe inside it is what users
# double-click).
#
# See README.md for the full first-time setup / troubleshooting notes.

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# These packages ship data files / dynamic libs / do runtime plugin
# discovery that static analysis misses, so pull them in wholesale.
for pkg in [
    "whisperx",
    "faster_whisper",
    "ctranslate2",
    "torch",
    "torchaudio",
    "torchvision",
    "pyannote",
    "lightning_fabric",
    "lightning_utilities",
    "pytorch_lightning",
    "speechbrain",
    "transformers",
    "hydra",
    "hydra_core",
    "omegaconf",
    "torch_audiomentations",
    "torchmetrics",
    "asteroid_filterbanks",
    "julius",
    "antlr4",
    "torchcodec",
    "imageio_ffmpeg",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[build.spec] warning: could not collect {pkg}: {e}")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperXHardsubStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX + CUDA dlls do not mix well; leave off
    console=True,  # TEMP: keep True while debugging import errors; see README
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WhisperXHardsubStudio",
)
