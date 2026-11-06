"""Select a RAM-limited pilot from live MAT metadata and optionally verify it."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bat_internship.site_selection import select_diverse_sites, require_available_memory
from bat_internship.paths import MAT_PATH, PROJECT_ROOT, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sites", type=int, default=5)
    parser.add_argument("--response-budget-mib", type=float, default=64)
    parser.add_argument("--validate", action="store_true",
                        help="Load selected sites and compare every recorded value")
    args = parser.parse_args()
    from bat_internship.bat_data import BatData
    with BatData(MAT_PATH) as source:
        metadata = [source.vocal_metadata(site) for site in range(source.n_sites)]
    selection = select_diverse_sites(
        metadata, max_sites=args.max_sites, response_budget_mib=args.response_budget_mib)
    output = OUTPUT_ROOT / "merged_audio_dataset"
    output.mkdir(parents=True, exist_ok=True)
    (output / "diverse_sites_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)
    if not args.validate:
        return
    # Do this before importing PyTorch and before loading any calcium responses.
    require_available_memory(selection["response_bytes"])
    import numpy as np
    import torch
    from deepSTRF.utils.data import neural_collate
    from bat_internship.bat_audio_dataset import BatAudioDataset
    ds = BatAudioDataset(
        MAT_PATH, site_indices=selection["site_indices"],
        audio_root=PROJECT_ROOT / "Autorendaten/OneDrive_1_10-5-2026",
        mapping_path=PROJECT_ROOT / "data/metadata/regular_audio_manifest.csv")
    checked = missing = 0
    with BatData(MAT_PATH) as source:
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            for local, global_s in enumerate(ds.local_to_global[site]):
                original = source.dataset(site, "VocalizationMat")[:, :, local, :]
                # Compare ROI by ROI without stacking a second full-site tensor.
                for roi, response in enumerate(ds.responses[global_s][bounds]):
                    np.testing.assert_array_equal(response.numpy().T, original[:, :, roi])
                    assert response.dtype == torch.float64
                checked += original.size
                del original
            for stim in set(range(len(ds))) - set(ds.local_to_global[site]):
                for response in ds.responses[stim][bounds]:
                    assert tuple(response.shape) == (1, 1) and torch.isnan(response).all()
                    missing += 1
    for stim in range(len(ds)):
        batch = neural_collate([ds[stim]])
        assert torch.equal(batch["valid_mask"], torch.isfinite(batch["responses"]))
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            mask = batch["valid_mask"][0, bounds]
            assert bool(mask.all()) if stim in ds.local_to_global[site] else not bool(mask.any())
        del batch
    report = dict(
        site_indices=ds.site_indices, bat_ids=selection["bat_ids"],
        unique_stimuli=len(ds), neurons=ds.N_neurons,
        original_values_exactly_verified=checked,
        missing_stimulus_roi_pairs=missing, all_stimulus_batch_masks_verified=True,
        response_bytes=selection["response_bytes"], timing_confirmed=False,
        training_performed=False)
    (output / "diverse_sites_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output / "diverse_sites_metadata.json").write_text(
        json.dumps(dict(sites=ds.site_metadata, stimuli=ds.stim_meta, neurons=ds.nrn_meta),
                   indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
