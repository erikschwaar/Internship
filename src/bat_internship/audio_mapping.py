"""Portable manifest and WAV checks shared by datasets and visualization.

Manifest paths are relative to an explicitly supplied audio root. This module
has no knowledge of the internship workspace or ROI polygons.
"""
import csv
import hashlib
from pathlib import Path
import wave

import numpy as np


def read_audio_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"matrix", "site_index_python", "index_python", "wav_path",
                    "match_status", "matlab_category", "pcm_sha256", "sample_rate_hz"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Audio manifest is missing required columns")
        rows = {}
        for row in reader:
            if row["matrix"] != "VocalizationMat":
                continue
            key = (int(row["site_index_python"]), int(row["index_python"]))
            if min(key) < 0 or key in rows:
                raise ValueError(f"Duplicate or invalid manifest position: {key}")
            rows[key] = row
    return rows


def load_mapped_audio(mapping, audio_root, category):
    """Verify category, rate and PCM hash before decoding mono 16-bit audio."""
    if mapping["match_status"] not in ("exact", "extension_added"):
        raise ValueError(f"WAV-Zuordnung nicht bestätigt: {mapping['match_status']}")
    if mapping["matlab_category"] != category:
        raise ValueError("Kategorie der WAV-Zuordnung stimmt nicht mit der Matrix überein")
    relative = Path(mapping["wav_path"])
    root = Path(audio_root).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Manifest WAV paths must stay relative to audio_root")
    with wave.open(str(path), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        pcm = wav.readframes(wav.getnframes())
    if hashlib.sha256(pcm).hexdigest() != mapping["pcm_sha256"]:
        raise ValueError("WAV wurde seit der Zuordnungsprüfung verändert; Audit erneut ausführen")
    if rate != int(mapping["sample_rate_hz"]):
        raise ValueError("WAV-Abtastrate stimmt nicht mit der Zuordnungsprüfung überein")
    if channels != 1 or width != 2:
        raise ValueError("Dieses Tool erwartet die gelieferten Mono-WAVs mit 16-bit PCM")
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    if not audio.size:
        raise ValueError("Leere WAV-Datei")
    return audio, rate
