"""Linear validation of single-site and merged deepSTRF datasets."""
import numpy as np
import torch
from pathlib import Path
import time
from deepSTRF.models.audio import Linear


class CheckpointPath(type(Path())):
    """Retry transient Windows locks during atomic checkpoint replacement."""
    def replace(self, target):
        for attempt in range(6):
            try:
                return super().replace(target)
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * 2**attempt)


def split_stimuli(identities, seed=42, categories=None):
    """Keep legacy test sounds; optionally diversify validation categories.

    Prefer one validation sound per category with at least two remaining sounds,
    preserving training representation. Fill remaining slots in seeded order.
    This is category coverage, not proportional stratification.
    """
    if len(identities) != len(set(identities)) or len(identities) < 7:
        raise ValueError("Need at least seven distinct sound identities")
    order = np.random.default_rng(seed).permutation(len(identities)).tolist()
    if categories is None:
        return dict(train=order[6:], validation=order[3:6], test=order[:3])
    if len(categories) != len(identities) or any(not c for c in categories):
        raise ValueError("Need one nonempty category per identity")
    remaining = order[3:]
    groups = {}
    for i in remaining:
        groups.setdefault(categories[i], []).append(i)
    validation = [ids[0] for ids in groups.values() if len(ids) >= 2][:3]
    for i in remaining:
        if len(validation) == 3:
            break
        if i not in validation:
            validation.append(i)
    return dict(train=[i for i in remaining if i not in validation],
                validation=validation, test=order[:3])


def build_model(name, dataset):
    common = dict(n_frequency_bands=dataset.F, out_neurons=dataset.N_neurons,
                  output_activation=torch.nn.Identity())
    if name != "linear":
        raise ValueError(f"Only the linear model is supported, got: {name}")
    model = Linear(**common, temporal_window_size=16)
    model.wav2spec = dataset.make_wav2spec()
    model.validate()
    return model


def training_response_statistics(dataset, indices):
    """Per-ROI moments of observed training repeat means; equal weight per bin.

    Each ROI may have different repeat counts and missing sound recordings.
    Read one ROI at a time to avoid padding the entire dataset in memory.
    """
    total = np.zeros(dataset.N_neurons, dtype=np.float64)
    total_squared = np.zeros_like(total)
    count = np.zeros(dataset.N_neurons, dtype=np.int64)
    for index in indices:
        for roi, response in enumerate(dataset.responses[index]):
            values = response.numpy()
            if np.isnan(values).all():
                continue
            if not np.isfinite(values).all():
                raise ValueError("Recorded responses must be finite")
            mean = values.mean(axis=0)
            total[roi] += mean.sum()
            total_squared[roi] += np.square(mean).sum()
            count[roi] += mean.size
    if np.any(count == 0):
        missing = np.flatnonzero(count == 0).tolist()
        raise ValueError(f"No training observations for ROI indices {missing[:20]}")
    location = total / count
    variance = np.maximum(total_squared / count - location ** 2, 0)
    return location, np.maximum(np.sqrt(variance), 1e-6), count


def per_roi_scores(prediction, target):
    """Score observed bins only; unobserved ROIs and undefined correlations are null.

    Overall MSE is the equal-weight mean over ROIs with observations, matching
    the library's neuron reduction. Nonfinite predictions always fail.
    """
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError("Expected matching (sounds, neurons, time) arrays")
    x = prediction.transpose(1, 0, 2).reshape(target.shape[1], -1).astype(np.float64)
    y = target.transpose(1, 0, 2).reshape(target.shape[1], -1).astype(np.float64)
    if not np.isfinite(x).all() or np.isinf(y).any():
        raise ValueError("Predictions must be finite; targets may only use NaN for missing data")
    valid = np.isfinite(y)
    count = valid.sum(axis=1)
    present = count > 0
    if not present.any():
        raise ValueError("No observed targets in evaluation subset")
    safe_count = np.maximum(count, 1)
    residual = np.where(valid, x - y, 0)
    mse = np.square(residual).sum(axis=1) / safe_count
    xmean = np.where(valid, x, 0).sum(axis=1) / safe_count
    ymean = np.where(valid, y, 0).sum(axis=1) / safe_count
    xc = np.where(valid, x - xmean[:, None], 0)
    yc = np.where(valid, y - ymean[:, None], 0)
    denom = np.sqrt(np.square(xc).sum(axis=1) * np.square(yc).sum(axis=1))
    xrange = np.max(np.where(valid, x, -np.inf), axis=1) - np.min(np.where(valid, x, np.inf), axis=1)
    yrange = np.max(np.where(valid, y, -np.inf), axis=1) - np.min(np.where(valid, y, np.inf), axis=1)
    corr = np.divide((xc * yc).sum(axis=1), denom,
                     out=np.full(len(mse), np.nan),
                     where=(count > 1) & (denom > 0) & (xrange > 0) & (yrange > 0))
    return dict(mean_mse=float(mse[present].mean()),
                mse_per_roi=[float(v) if ok else None for v, ok in zip(mse, present)],
                correlation_per_roi=[float(v) if np.isfinite(v) else None for v in corr],
                observed_bins_per_roi=count.tolist(), rois_evaluated=int(present.sum()))


def comparison_scores(prediction, target, baseline):
    """Compare model and training-only baseline on exactly the same observed bins."""
    model = per_roi_scores(prediction, target)
    constant = per_roi_scores(baseline, target)
    correlations = [v for v in model['correlation_per_roi'] if v is not None]
    return dict(model=model, constant_baseline=constant,
                relative_mse_improvement=(1-model['mean_mse']/constant['mean_mse']
                                          if constant['mean_mse'] > 0 else None),
                median_roi_correlation=float(np.median(correlations)) if correlations else None,
                rois_beating_baseline=sum(a < b for a, b in
                    zip(model['mse_per_roi'], constant['mse_per_roi']) if a is not None))
