# Linear training to validate the dataset integration

The purpose of training is to check that the prepared sounds and calcium
responses can be used together in deepSTRF. A single native Linear model predicts
all ROIs in the selected sites. Dataset checks and prediction results together
support this validation. Prediction accuracy alone cannot establish correct
sound mapping or timing.

## Run from the project root

Short check on two merged sites, including different sound inventories:

```powershell
.venv/Scripts/python.exe scripts/train_audio_models.py --sites 0 13 --smoke
```

Short check on the complete dataset:

```powershell
.venv/Scripts/python.exe scripts/train_audio_models.py --all-sites --smoke
```

Longer joint training on all sites:

```powershell
.venv/Scripts/python.exe scripts/train_audio_models.py --all-sites --epochs 100 --patience 15
```

Use `--sites 5 14 19 25 28` for a smaller merged selection or `--site 24`
for a single-site run. Indices are zero-based. Without a selection, site 24
is used for compatibility. Site selection flags are mutually exclusive.
All runs use only Linear. The older `--models linear` and `--validation-only`
flags remain accepted. Validation-only evaluation is now the default and test
sounds are always reserved. The old model-comparison launcher is retired.

## Protocol

- Split the common audio identities before training. All repeats of an identity
  remain together across every site. Three sounds are held out for validation,
  three for a future test; the remaining sounds are used for training.
- The default category-coverage strategy tries to cover categories in validation.
  `--split-seed` controls the split; `--seed` controls model initialization and
  minibatch order. Both default to 42. Each ROI must have training observations.
- The dataset preserves the original fluorescence. Batches convert to float32.
  Repeat means are calculated with NaN-aware averaging. Missing sound/ROI pairs
  and padded repetitions do not contribute to the loss or metrics.
- Output initialization uses only observed training repeat means, with equal
  weight per observed sound/time bin. It sets the Linear readout BatchNorm bias
  and scale. `--linear-init default` optionally uses the library initialization.
- The constant baseline uses the same training-only ROI means. Metrics compare
  model and baseline on identical observed targets. Overall MSE is the mean of
  the per-ROI MSE values over ROIs with observations; unobserved ROIs have null
  scores and explicit counts. Per-site metrics retain the original ROI mapping.
- Native CausalMelSpectrogram, 32 bands, power/log1p, 32 ms window and hop.
  Linear uses 16 context frames and Identity output. Timing and sound onset
  remain the documented working assumptions.
- CPU, two threads, batch size two, AdamW, learning rate 0.0003 and weight decay
  0.0001. `--learning-rate` can be set explicitly. Earlier linear diagnostics
  used 0.001 and should be treated as a different protocol.
- `--smoke` runs three epochs. Regular runs use at most 100 epochs and patience
  15 by default. The native Fitter restores the validation-selected checkpoint.
- A RAM check includes the response payload and headroom for model parameters
  and optimizer state. The all-site model has 9,443 outputs. Close a previously
  loaded notebook dataset or restart its kernel if memory is insufficient.

## Saved results

Every run has its own directory under `outputs/linear_dataset_validation`.
Single-site names start with `site_`; merged runs start with `sites_`.

- `config.json` records sites, ROI metadata and ranges, sound identities and
  splits, model settings and the actual learning rate.
- `linear_initialization.json` records the training indices, per-ROI moments
  and observed-bin counts when training-statistics initialization is used.
- `linear/` contains checkpoints, epoch histories, validation predictions,
  metrics, learning curves and example responses with the constant baseline.
- Metrics include aggregate scores, post-onset scores and scores for each site.
  Example ROIs are fixed; each uses its first observed validation sound.
- `summary.json` is written only after the run completes. Incomplete folders
  remain available for diagnosis and are excluded from the notebook display.

Notebook sections 4.8–4.10 prepare a joint run, optionally launch it and display
matching complete results. `LINEAR_ALL_SITES = True` selects all sites.
For a manual selection, set it to False and set `LINEAR_MANUAL_SITES`.
`RUN_LINEAR_TRAINING = False` keeps execution in preparation mode.

The small number of validation sounds limits biological conclusions. Training
here validates the integration and provides examples for the internship report.
