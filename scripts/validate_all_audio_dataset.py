"""Merge every regular site and verify original values, masks and process RAM.

Run in a separate process so the dataset is released on exit. Requires psutil
for Windows working-set/private-commit measurements; does not train a model.
"""
import json
import math
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bat_internship.bat_data import BatData
from bat_internship.paths import MAT_PATH, PROJECT_ROOT, OUTPUT_ROOT
from bat_internship.site_selection import require_available_memory


def main():
    started = time.perf_counter()
    process = psutil.Process()
    snapshots = []

    def snapshot(stage):
        info = process.memory_info()
        entry = dict(stage=stage, rss_bytes=info.rss,
                     peak_rss_bytes=getattr(info, "peak_wset", info.rss),
                     private_bytes=getattr(info, "private", None),
                     available_system_bytes=psutil.virtual_memory().available)
        snapshots.append(entry)
        print(json.dumps(entry), flush=True)

    snapshot("before_imports")
    with BatData(MAT_PATH) as source:
        metadata = [source.vocal_metadata(i) for i in range(source.n_sites)]
    indices = [m["site_index"] for m in metadata]
    response_bytes = sum(math.prod(m["shape_time_repeat_stimulus_roi"]) * 8
                         for m in metadata)
    require_available_memory(response_bytes)
    import numpy as np
    import torch
    from deepSTRF.utils.data import neural_collate
    from bat_internship.bat_audio_dataset import BatAudioDataset

    snapshot("before_dataset")
    ds = BatAudioDataset(
        MAT_PATH, site_indices=indices,
        audio_root=PROJECT_ROOT / "Autorendaten/OneDrive_1_10-5-2026",
        mapping_path=PROJECT_ROOT / "data/metadata/regular_audio_manifest.csv",
        dt_ms=32.0, audio_offset_ms=0.0)
    snapshot("dataset_loaded")
    checked = missing = 0
    with BatData(MAT_PATH) as source:
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            for local, global_s in enumerate(ds.local_to_global[site]):
                original = source.dataset(site, "VocalizationMat")[:, :, local, :]
                for roi, response in enumerate(ds.responses[global_s][bounds]):
                    np.testing.assert_array_equal(response.numpy().T, original[:, :, roi])
                    assert response.dtype == torch.float64
                checked += original.size
                del original
            for stim in set(range(len(ds))) - set(ds.local_to_global[site]):
                for response in ds.responses[stim][bounds]:
                    assert tuple(response.shape) == (1, 1) and torch.isnan(response).all()
                    missing += 1
            print(f"Original values verified: site {site}", flush=True)
    assert checked * 8 == response_bytes
    assert ds.N_neurons == sum(m["shape_time_repeat_stimulus_roi"][3] for m in metadata)
    snapshot("original_values_verified")
    for stim in range(len(ds)):
        batch = neural_collate([ds[stim]])
        assert torch.equal(batch["valid_mask"], torch.isfinite(batch["responses"]))
        for site, bounds in zip(ds.site_indices, ds.roi_ranges):
            mask = batch["valid_mask"][0, bounds]
            assert bool(mask.all()) if stim in ds.local_to_global[site] else not bool(mask.any())
        snapshot(f"batch_{stim}_verified")
        del batch
    report = dict(
        site_indices=indices, bat_ids=sorted({m["bat_id"] for m in metadata}),
        unique_stimuli=len(ds), neurons=ds.N_neurons,
        original_values_exactly_verified=checked,
        missing_stimulus_roi_pairs=missing, all_stimulus_batch_masks_verified=True,
        response_bytes=response_bytes, memory_snapshots=snapshots,
        peak_process_rss_bytes=max(s["peak_rss_bytes"] for s in snapshots),
        memory_scope="Separate validation process, batch size 1, no training; RSS peak from OS",
        elapsed_seconds=time.perf_counter() - started,
        dt_ms_assumed=32.0, audio_offset_ms_assumed=0.0,
        timing_confirmed=False, training_performed=False)
    output = OUTPUT_ROOT / "merged_audio_dataset"
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_sites_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output / "all_sites_metadata.json").write_text(
        json.dumps(dict(sites=ds.site_metadata, stimuli=ds.stim_meta,
                        neurons=ds.nrn_meta, local_to_global=ds.local_to_global), indent=2),
        encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "memory_snapshots"}, indent=2))


if __name__ == "__main__":
    main()
