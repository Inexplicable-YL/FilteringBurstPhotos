# FilteringBurstPhotos

A Windows-friendly prototype for grouping and filtering burst photos, including RAW formats. The project focuses on a clear pipeline for scanning folders, computing perceptual hashes, grouping likely bursts, and moving discarded files to a safe backup directory.

## Features

- Supports JPEG/PNG/BMP/TIFF alongside common RAW formats (CR2/CR3/NEF/ARW/RAF/ORF/RW2/DNG).
- Perceptual hash (pHash) scoring to measure similarity between shots.
- Burst grouping based on configurable time and hash thresholds.
- Settings persistence to keep the latest thresholds and duplicate folder name.
- Command-line prototype that scans a folder and reports grouping results while preparing for a future PySide6 GUI.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py <photo_directory> --recursive --dry-run
```

Launch the GUI prototype (folder can be chosen later if omitted):

```bash
python main.py --gui [photo_directory]
```

Command-line flags:

- `--recursive`: scan subfolders.
- `--time-threshold`: time gap in seconds to consider photos part of the same burst.
- `--hash-threshold`: maximum Hamming distance between hashes inside a burst.
- `--min-group-size`: minimum number of photos to keep a burst grouped; otherwise photos are treated as singletons.
- `--duplicate-dir`: folder name for moved duplicates (default `_duplicates`).
- `--dry-run`: preview moves without touching files.

## Roadmap

- PySide6-based GUI with thumbnail grouping, large previews, and overlay/flipbook modes.
- Cached hashing to accelerate rescans.
- Enhanced EXIF and RAW metadata handling for more reliable capture times.
- Packaging for Windows (PyInstaller) with installer options.
