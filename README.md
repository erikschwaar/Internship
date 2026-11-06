# Bat calcium recordings for deepSTRF

This internship project prepares bat calcium-imaging recordings and sound files
for use with [deepSTRF](https://github.com/urancon/deepSTRF). It matches sounds to
recorded responses, merges recording sites into a common dataset and checks that
the result works with the library's audio processing, batching and linear model.

The main contribution is the dataset integration. Linear-model training provides
a practical way to validate it.

The recordings come from Lawlor et al., *Spatially clustered neurons in the bat
midbrain encode vocalization categories*. [Original study](https://www.nature.com/articles/s41593-025-01932-3)

## Start with the notebook

[From bat recordings to a deepSTRF dataset](notebooks/dataset_integration.ipynb)
walks through the implementation with explanations, code and saved figures.
It covers the recording inventory, sound matching, site merging, preservation of
original values, missing-data masks, native mel features and the linear interface.

The saved demonstration uses five sites from five bats, with 809 ROIs and
17 distinct sounds. Training is opt-in. Set `SITE_MODE = "all"` to run the data
walkthrough on all sites, or use `"manual"` with your own site indices.

## Installation

Use Python 3.10 or later. Development and local checks used Python 3.13 on Windows.
From the repository root, create a virtual environment and install the dependencies.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements/deepstrf.txt
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m ipykernel install --user --name bat-internship --display-name "Python (bat internship)"
```

Select **Python (bat internship)** when opening the notebook. On Linux or macOS,
use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

The requirements pin deepSTRF to commit
`5dba99f60a5cb51548f9f9769941cca377771d01`, the version used for this integration.

## Input data

The workflow needs the original MATLAB recordings and the authors' WAV files.
Obtain these through the study's data-access arrangements. The committed manifest
contains relative file paths, categories and checksums used to verify the sound
assignments.

The command-line scripts expect the following locations relative to the repository
root. Keep the WAV subdirectories and filenames as supplied by the authors.

```text
data/raw/MegadataStruct.mat
Autorendaten/OneDrive_1_10-5-2026/
data/metadata/regular_audio_manifest.csv
```

The MATLAB file can also be placed directly in the repository root. The dataset
class accepts explicit input paths, and the notebook exposes them in its setup
cell. The command-line trainer currently uses the repository locations above.

The [manifest documentation](data/metadata/README.md) explains the mapping and how
it can be regenerated from the author-data audit.

## Load and merge recording sites

Run this example from the repository root after installation.

```python
from pathlib import Path
from bat_internship import BatAudioDataset
from bat_internship.paths import MAT_PATH

root = Path.cwd()
dataset = BatAudioDataset(
    MAT_PATH,
    site_indices=[0, 13],
    audio_root=root / "Autorendaten/OneDrive_1_10-5-2026",
    mapping_path=root / "data/metadata/regular_audio_manifest.csv",
    dt_ms=32.0,
    audio_offset_ms=0.0,
    return_waveform=True,
    n_mels=32,
)

print(len(dataset), dataset.N_neurons)  # 17 sounds, 778 ROIs
print(dataset.local_to_global)         # Local sound indices mapped to shared identities
```

Site indices are zero-based. Use `site_indices=list(range(29))` for the complete
current dataset, or `site_index=24` for a single site.

The merged dataset has one shared sound axis. Sounds are matched using verified
PCM checksums and sample rates. Each ROI retains its site, animal and local index
in `nrn_meta`; `roi_ranges` identifies each site's section of the output axis.

Original responses remain float64. Missing sound/ROI pairs use NaN sentinels.
Repeats belong to their original recording sites, which were recorded separately.
The native `neural_collate` function pads unequal repeat counts and creates a
fine-grained validity mask. Its waveform batches use float32.

Use `site_indices` to merge sites by shared sound identity. The inherited
`dataset_a + dataset_b` operation appends stimulus axes and does not deduplicate
shared sounds.

## Native audio processing

The dataset stores time-aligned mono waveforms. `dataset.make_wav2spec()` creates
deepSTRF's native `CausalMelSpectrogram`, which is attached to the model's
`wav2spec` slot. The starting settings are 32 mel bands, a 32 ms window and hop,
power spectra and log1p compression.

```python
from bat_internship.training import build_model
from deepSTRF.utils.data import neural_collate

model = build_model("linear", dataset)
batch = neural_collate([dataset[0]])
prediction = model(batch["stims"])
print(prediction.shape)  # batch × ROIs × 1 × response frames
```

The implementation is a local adapter extending `AudioNeuralDataset`.
The reusable code is in [bat_audio_dataset.py](src/bat_internship/bat_audio_dataset.py).

## Check the integration

Run the automated tests after installing the project.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests
```

With the original recordings and audio available, run the real-data checks.

```powershell
.venv/Scripts/python.exe scripts/validate_merged_audio_dataset.py
.venv/Scripts/python.exe scripts/validate_diverse_audio_dataset.py --validate
.venv/Scripts/python.exe scripts/validate_all_audio_dataset.py
```

These check a two-site dataset, a smaller selection across animals, and the full
dataset respectively. They compare merged responses with the original values and
verify missing-pair and batch masks. Reports are written under
`outputs/merged_audio_dataset/`.

The full dataset was locally checked with 29 sites, five bats, 9,443 ROIs and
17 distinct sounds. That check compared 69,034,110 original values exactly and
verified 25,170 missing sound/ROI pairs. The repository notebook reproduces the
same kind of checks on its selected sites.

## Train one joint linear model

Training validates the dataset's use in deepSTRF. One linear model predicts all
ROIs in the selected sites. A short smoke run performs three epochs.

```powershell
# Two sites merged into one training dataset
.venv/Scripts/python.exe scripts/train_audio_models.py --sites 0 13 --smoke

# All sites, three epochs
.venv/Scripts/python.exe scripts/train_audio_models.py --all-sites --smoke

# All sites, up to 100 epochs with early stopping
.venv/Scripts/python.exe scripts/train_audio_models.py --all-sites --epochs 100 --patience 15
```

All repeats and sites for a given sound stay in the same split. Three sounds are
used for validation, three remain reserved for a future test, and the rest are
used for training. Test sounds are not evaluated by this workflow.

Output initialization and the constant per-ROI baseline use only observed
training responses. Missing targets are excluded from loss and evaluation.
Results include aggregate and per-site scores, prediction plots, checkpoints,
sound assignments and ROI metadata. Each run gets its own directory under
`outputs/linear_dataset_validation/`.

The [training guide](docs/training_pilot.md) documents the configuration and
saved files. Two-site joint smoke runs have completed successfully, including a
split with missing validation responses. The full 29-site training run remains
unverified locally because the memory check stopped it before loading.

## Memory and scientific assumptions

- The full original response payload is about 527 MiB. Loading, batching and
  training require additional memory. Scripts check available RAM before loading;
  a previous full-dataset training attempt required about 2.7 GiB of free headroom.
  Restart a notebook kernel holding a loaded dataset before running a separate
  training process if memory is limited.
- A 32 ms frame interval and zero audio-onset offset are explicit working
  assumptions. The supplied time metadata still need reconciliation.
- Fluorescence is used in its supplied scale. The exported F/F₀ versus ΔF/F₀
  convention remains to be confirmed.
- Prediction scores complement exact data checks and interface checks. A small
  held-out sound set limits conclusions about performance on other recordings.

## Code to explore

- [Dataset and site merging](src/bat_internship/bat_audio_dataset.py)
- [MATLAB data access](src/bat_internship/bat_data.py)
- [Sound mapping and file verification](src/bat_internship/audio_mapping.py)
- [Metadata-based site selection](src/bat_internship/site_selection.py)
- [Joint linear training](scripts/train_audio_models.py)
- [Training statistics and metrics](src/bat_internship/training.py)
- [Multi-site tests](tests/test_bat_audio_multisite.py)
- [Missing-value and training tests](tests/test_training.py)
