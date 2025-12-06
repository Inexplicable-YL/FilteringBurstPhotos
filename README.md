# FilteringBurstPhotos

A Windows-friendly prototype for grouping and filtering burst photos with a PySide6 GUI. The project focuses on a clear pipeline for scanning folders, computing perceptual hashes, grouping likely bursts, and reviewing them visually.

## Features

- Supports JPEG/PNG/BMP/TIFF alongside common RAW formats (CR2/CR3/NEF/ARW/RAF/ORF/RW2/DNG).
- Perceptual hash (pHash) scoring to measure similarity between shots.
- Burst grouping based on configurable time and hash thresholds.
- Settings persistence to keep the latest thresholds across GUI sessions.
- PySide6 interface for scanning folders, previewing thumbnails, and selecting photos to keep.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Use the GUI to choose a folder and filter burst photos.

## Roadmap

- Enhance the PySide6 GUI with overlay/flipbook modes and batch actions.
- Cached hashing to accelerate rescans.
- Enhanced EXIF and RAW metadata handling for more reliable capture times.
- Packaging for Windows (PyInstaller) with installer options.
