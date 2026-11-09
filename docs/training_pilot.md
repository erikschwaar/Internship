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
- The readout uses fixed normalization moments measured on the initial filter
  activations for observed training bins. These moments are held constant in
  both training and evaluation. No validation or test targets enter calibration.
- Baseline initialization sets the STRF filters to zero and compensates the
  readout offset so every ROI initially predicts its training response mean.
  Output scales come from training repeat means. All filter and affine parameters
  remain learnable.
- The constant baseline uses the same training-only ROI means. Metrics compare
  model and baseline on identical observed targets. Overall MSE is the mean of
  the per-ROI MSE values over ROIs with observations; unobserved ROIs have null
  scores and explicit counts. Per-site metrics retain the original ROI mapping.
- Native CausalMelSpectrogram, 32 bands, power/threshold-clipped log, 32 ms window and hop.
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
  and observed-bin counts when training-statistics or baseline initialization is used.
- `linear_normalization.json` records the fixed moments and their training indices.
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

## Investigating normalization in the linear model

The native readout uses BatchNorm. Its running statistics can differ substantially
from the current activations, particularly early in training with small batches.
This can make early validation predictions nearly constant. Recalibrating a saved
checkpoint alone does not guarantee better predictions.

The `--linear-normalization fixed-training` option measures the initial
readout activations on observed training sound/time bins, then holds those
moments fixed throughout training and evaluation. The affine parameters remain
learnable. Calibration excludes missing targets and never uses validation or test
responses. Its values and training indices are saved in `linear_normalization.json`.
The model remains affine in its mel features.

The default combination is `--linear-normalization fixed-training --linear-init baseline`.
For the previous behavior, explicitly use
`--linear-normalization running --linear-init train-statistics`.
Fixed normalization with the old random filter initialization was also checked
and performed worse on this validation split; its result remains in the diagnostic outputs.
When restoring a checkpoint, pass its recorded mode to
`build_model("linear", dataset, normalization=config.get("linear_normalization", "running"))`.
Older configurations without that key use `running`.

Epoch logs now also include `train_eval_mse`, measured on the training sounds in
inference mode after each epoch. This makes differences between training-batch
loss and inference-mode error visible. The training evaluation loader is not
shuffled and is separate from the optimization loader.

The original five-site run's validation MSE was 0.00634070, slightly worse than
its 0.00633559 constant baseline. With fixed normalization and baseline
initialization, training seed 42 reached 0.00624210 (1.48% below the baseline).
The median ROI correlation rose from 0.0268 to 0.1071. These are development
results on the same validation sounds, not an independent generalization test.
The test split was not evaluated.

The same fixed split was also checked with training seeds 43 and 44. Their MSE
improvements were 1.59% and 1.55%, with median ROI correlations of 0.1113 and
0.1083. Together with seed 42, the three runs show a small repeatable improvement
on this validation split. They do not establish accuracy on new sounds or animals.
The diagnostic comparison, including the unsuccessful fixed-normalization-only
run, is saved locally in `outputs/linear_normalization_diagnostics/comparison.json`.

Frontend configuration update: new runs use `compression="log"` (floor `1e-4`). Existing runs produced with `log1p` retain their original configuration and results; retrain to evaluate the new frontend.
