"""Verify all recorded and missing pairs for the two-site real-audio pilot."""
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
from bat_internship.paths import MAT_PATH, PROJECT_ROOT, OUTPUT_ROOT


def main():
    ds = BatAudioDataset(MAT_PATH, site_indices=[0, 13], dt_ms=32,
                        audio_root=PROJECT_ROOT / "Autorendaten/OneDrive_1_10-5-2026",
                        mapping_path=PROJECT_ROOT / "data/metadata/regular_audio_manifest.csv")
    checked = missing = 0
    with BatData(MAT_PATH) as source:
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            for local, global_s in enumerate(ds.local_to_global[site]):
                values = torch.stack(ds.responses[global_s][bounds]).numpy().transpose(2, 1, 0)
                original = source.dataset(site, "VocalizationMat")[:, :, local, :]
                np.testing.assert_array_equal(values, original)
                assert values.dtype == np.float64
                checked += original.size
            for s in set(range(len(ds))) - set(ds.local_to_global[site]):
                for response in ds.responses[s][bounds]:
                    assert tuple(response.shape) == (1, 1) and torch.isnan(response).all()
                    missing += 1
    assert len(ds) == 17 and ds.N_neurons == 778
    assert len(set(ds.local_to_global[0]) & set(ds.local_to_global[13])) == 12
    # Verify the upstream batch mask for every stimulus, not just one example.
    for s in range(len(ds)):
        batch = neural_collate([ds[s]])
        assert torch.equal(batch["valid_mask"], torch.isfinite(batch["responses"]))
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            mask = batch["valid_mask"][0, bounds]
            assert bool(mask.all()) if s in ds.local_to_global[site] else not bool(mask.any())
    report = dict(site_indices=ds.site_indices, local_stimulus_counts=[12, 17],
                  unique_stimuli=len(ds), shared_stimuli=12, neurons=ds.N_neurons,
                  roi_counts=[b.stop - b.start for b in ds.roi_ranges],
                  original_values_exactly_verified=checked,
                  missing_stimulus_roi_pairs=missing,
                  all_stimulus_batch_masks_verified=True,
                  local_to_global=ds.local_to_global,
                  dt_ms_assumed=32, audio_offset_ms_assumed=0,
                  timing_confirmed=False, training_performed=False)
    output = OUTPUT_ROOT / "merged_audio_dataset"
    output.mkdir(parents=True, exist_ok=True)
    (output / "two_sites_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "two_sites_metadata.json").write_text(
        json.dumps(dict(stimuli=ds.stim_meta, neurons=ds.nrn_meta), indent=2), encoding="utf-8")
    fig = plot_audio_example(ds, ds.local_to_global[13][4], ds.roi_ranges[1].start + 3)
    fig.savefig(output / "two_sites_example.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
