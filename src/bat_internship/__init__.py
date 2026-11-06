"""Datenzugriff und Vorbereitung der Fledermausdaten für deepSTRF."""

__all__ = ["BatAudioDataset"]


def __getattr__(name):
    # Keep metadata-only tools lightweight: import PyTorch only on demand.
    if name == "BatAudioDataset":
        from .bat_audio_dataset import BatAudioDataset
        return BatAudioDataset
    raise AttributeError(name)
