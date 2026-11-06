# Audio manifest for the regular vocalization recordings

`regular_audio_manifest.csv` contains 428 site-specific sound assignments from
the checked author-data audit. It stores metadata and checksums. WAV paths are
relative to the audio directory supplied to the dataset loader.

The original audio directory is `Autorendaten/OneDrive_1_10-5-2026`. The directory
can be moved as long as its internal structure is preserved and the caller
provides the new root. The manifest has no absolute user-specific paths.

## Loader checks

For each selected site, the manifest must cover exactly the regular stimulus
positions. The loader checks the sound category, sample rate and PCM checksum
against the recordings and WAV files. Missing, duplicate or unconfirmed
assignments are rejected.

The dataset loader uses the completed manifest directly. The audit scripts and
previous output files are only needed to regenerate it.

## Regenerate the manifest

From the repository root, after producing the author-data audit:

```powershell
.venv/Scripts/python.exe scripts/export_regular_audio_manifest.py --audit outputs/author_stimulus_audit/site_stimulus_mapping.csv --audit-root . --audio-root Autorendaten/OneDrive_1_10-5-2026 --output data/metadata/regular_audio_manifest.csv
```

The audit itself also requires the authors' stimulus key and original audio files.
See [audit_author_stimuli.py](../../scripts/audit_author_stimuli.py) for its inputs.

## Validation

The combined dataset of all 29 sites has been checked separately. Running
[scripts/validate_all_audio_dataset.py](../../scripts/validate_all_audio_dataset.py)
produces `outputs/merged_audio_dataset/all_sites_validation.json`. The checks
cover preservation of original response values and the masks for missing
sound/ROI combinations.

Joint linear training provides an additional check of the prepared dataset's use
in deepSTRF. See the [training guide](../../docs/training_pilot.md).
