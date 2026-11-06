"""Projektpfade für die lokale, editierbare Installation, unabhängig vom cwd."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Während der Umordnung kann Windows eine geöffnete MAT-Datei sperren.
# Den neuen Speicherort bevorzugen, die bisherige Datei weiterhin unterstützen.
MAT_PATH = PROJECT_ROOT / "data" / "raw" / "MegadataStruct.mat"
if not MAT_PATH.is_file():
    MAT_PATH = PROJECT_ROOT / "MegadataStruct.mat"
