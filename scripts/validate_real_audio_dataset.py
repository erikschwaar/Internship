"""Reproduce the single-site real-audio check without training a model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from deepSTRF.utils.data import neural_collate
from bat_internship.bat_audio_dataset import BatAudioDataset, plot_audio_example
from bat_internship.bat_data import BatData
from bat_internship.paths import MAT_PATH, OUTPUT_ROOT, PROJECT_ROOT


def main():
    ds = BatAudioDataset(MAT_PATH, site_index=24,
                         audio_root=PROJECT_ROOT / "Autorendaten/OneDrive_1_10-5-2026",
                         mapping_path=PROJECT_ROOT / "data/metadata/regular_audio_manifest.csv")
    frontend = ds.make_wav2spec()
    checked = 0
    with BatData(MAT_PATH) as source:
        for s in range(len(ds.stims)):
            original = source.dataset(24, "VocalizationMat")[:, :, s, :]
            actual = torch.stack(ds.responses[s]).numpy().transpose(2, 1, 0)
            np.testing.assert_array_equal(actual, original)
            assert actual.dtype == np.float64
            checked += original.size
            with torch.no_grad():
                features = frontend(ds.stims[s][None]).numpy()
            assert np.isfinite(features).all() and np.any(features > 0)
            np.testing.assert_array_equal(features[..., ds.time_ms <= 0], 0)
            # All bins entirely after WAV end must also be exactly silent.
            end = ds.stim_meta[s]["audio_duration_ms"]
            np.testing.assert_array_equal(features[..., ds.time_ms - 32 >= end], 0)
    batch = neural_collate([ds[0], ds[1]])
    assert batch["responses"].dtype == torch.float32
    assert batch["valid_mask"].all()
    assert tuple(batch["stims"].shape) == (2, 1, len(ds.time_ms) * ds.hop)
    assert tuple(frontend(batch["stims"]).shape) == (2, 1, ds.F, len(ds.time_ms))
    assert tuple(batch["responses"].shape) == (2, ds.N_neurons, 10, len(ds.time_ms))
    assert not ds.is_placeholder and not ds.timing_confirmed
    output = OUTPUT_ROOT / "real_audio_dataset"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(site_index=24, stimuli=len(ds), neurons=ds.N_neurons,
                  audio_storage="native_grid_locked_waveform",
                  wav2spec=repr(frontend),
                  response_values_exactly_verified=checked,
                  stimulus_shape=list(ds.stims[0].shape),
                  response_per_roi_shape=list(ds.responses[0][0].shape),
                  batch_stimulus_shape=list(batch["stims"].shape),
                  batch_response_shape=list(batch["responses"].shape),
                  all_batch_values_valid=True, dt_ms_assumed=32,
                  audio_offset_ms_assumed=0, timing_confirmed=False,
                  no_training_performed=True)
    (output / "site_24_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "site_24_metadata.json").write_text(json.dumps(ds.stim_meta, indent=2), encoding="utf-8")
    figure = plot_audio_example(ds, 10, 3)
    figure.savefig(output / "site_24_stimulus_10_roi_003.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
