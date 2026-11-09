"""Validate a deepSTRF dataset with one linear model across selected sites. --smoke runs 3 epochs."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from deepSTRF.training import Fitter
from deepSTRF.metrics import mse_loss
from deepSTRF.utils.data import neural_collate
from bat_internship.bat_data import BatData
from bat_internship.bat_audio_dataset import BatAudioDataset
from bat_internship.paths import MAT_PATH, PROJECT_ROOT, OUTPUT_ROOT
from bat_internship.site_selection import require_available_memory
from bat_internship.training import (
    split_stimuli, build_model, comparison_scores, CheckpointPath,
    training_response_statistics, calibrate_linear_normalization,
    initialize_linear_from_baseline,
)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                    encoding="utf-8")


def predict(model, loader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(batch["stims"]).squeeze(2).numpy())
            targets.append(batch["responses"].nanmean(dim=2).numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--site", type=int, help="One zero-based site index (default: 24)")
    selection.add_argument("--sites", type=int, nargs="+", help="Train one model on these merged sites")
    selection.add_argument("--all-sites", action="store_true", help="Train one model on all sites")
    parser.add_argument("--models", nargs="+", choices=["linear"],
                        default=["linear"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["category-coverage", "legacy"],
                        default="category-coverage")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "linear_dataset_validation")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--linear-init", choices=["default", "train-statistics", "baseline"],
                        default="baseline")
    parser.add_argument("--linear-normalization", choices=["running", "fixed-training"], default="fixed-training")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.linear_init == "baseline" and args.linear_normalization != "fixed-training":
        parser.error("baseline initialization requires --linear-normalization fixed-training")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with BatData(MAT_PATH) as source:
        sites = list(range(source.n_sites)) if args.all_sites else args.sites
        if sites is None:
            sites = [24 if args.site is None else args.site]
        if len(set(sites)) != len(sites) or any(i < 0 or i >= source.n_sites for i in sites):
            parser.error("Site indices must be distinct and within the dataset")
        metadata = [source.vocal_metadata(i) for i in sites]
    response_bytes = sum(math.prod(m["shape_time_repeat_stimulus_roi"]) * 8 for m in metadata)
    # Reserve additional headroom for parameters, gradients and Adam state.
    roi_count = sum(m["shape_time_repeat_stimulus_roi"][-1] for m in metadata)
    model_mib = math.ceil(roi_count * 32 * 16 * 4 * 6 / 1024**2)
    require_available_memory(response_bytes, reserve_mib=1024 + model_mib)
    started = time.perf_counter()
    ds = BatAudioDataset(MAT_PATH, site_indices=sites,
                        audio_root=PROJECT_ROOT / "Autorendaten/OneDrive_1_10-5-2026",
                        mapping_path=PROJECT_ROOT / "data/metadata/regular_audio_manifest.csv",
                        dt_ms=32.0, audio_offset_ms=0.0, return_waveform=True, n_mels=32)
    splits = split_stimuli(ds.common_ids, args.split_seed,
                          categories=[m['category'] for m in ds.stim_meta]
                          if args.split_strategy == 'category-coverage' else None)
    evaluate_test = False  # Training validates the integration using held-out validation sounds.
    location, scale, training_counts = training_response_statistics(ds, splits["train"])
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S_%f")
    label = f"site_{sites[0]:02d}" if len(sites) == 1 else "sites_" + "_".join(map(str, sites))
    out = args.output_root / f"{label}_{'smoke' if args.smoke else 'train'}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    config = dict(site=sites[0] if len(sites) == 1 else None, site_indices=sites,
                  purpose="dataset integration validation", seed=args.seed, smoke=args.smoke,
                  split_seed=args.split_seed, split_strategy=args.split_strategy,
                  test_evaluation='exploratory' if evaluate_test else 'disabled',
                  epochs=3 if args.smoke else args.epochs, patience=args.patience,
                  batch_size=2, device="cpu", num_workers=0, torch_threads=2,
                  optimizer="AdamW", learning_rate=args.learning_rate, weight_decay=0.0001,
                  monitor="val_loss", target="repeat mean of original fluorescence",
                  normalization="none", baseline_subtraction=False,
                  linear_initialization=args.linear_init, linear_normalization=args.linear_normalization,
                  dt_ms=32, audio_offset_ms=0, n_mels=ds.F,
                  frontend=repr(ds.make_wav2spec()),
                  deepstrf_commit="5dba99f60a5cb51548f9f9769941cca377771d01",
                  torch_version=torch.__version__, roi_count=ds.N_neurons,
                  models={"linear":dict(temporal_window_size=16)},
                  roi_metadata=ds.nrn_meta,
                  roi_ranges=[dict(site_index=site, start=r.start, stop=r.stop)
                              for site, r in zip(sites, ds.roi_ranges)],
                  output_activation="Identity", time_ms=ds.time_ms.tolist(),
                  split={key:[dict(index=i, identity=ds.common_ids[i],
                                  wav_path=ds.stim_meta[i]["wav_path"],
                                  category=ds.stim_meta[i]["category"]) for i in ids]
                         for key,ids in splits.items()})
    write_json(out / "config.json", config)
    print(f"Output: {out}\nSplit: { {k:len(v) for k,v in splits.items()} }", flush=True)
    summaries = {}
    for name in dict.fromkeys(args.models):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        loaders = {key:DataLoader(Subset(ds, ids), batch_size=2,
                   shuffle=key=="train", num_workers=0, collate_fn=neural_collate,
                   generator=torch.Generator().manual_seed(args.seed))
                   for key,ids in splits.items() if key!="test" or evaluate_test}
        model = build_model(name, ds, normalization=args.linear_normalization)
        train_evaluation_loader = DataLoader(
            Subset(ds, splits["train"]), batch_size=2, shuffle=False,
            num_workers=0, collate_fn=neural_collate,
        )
        if name == "linear" and args.linear_init in ("train-statistics", "baseline"):
            # Use only observed repeat means from training sounds.
            location_tensor = torch.from_numpy(location).float()
            scale_tensor = torch.from_numpy(scale).float()
            norms = [m for m in model.readout.modules()
                     if isinstance(m, torch.nn.BatchNorm1d)]
            if len(norms) != 1 or norms[0].num_features != ds.N_neurons:
                raise ValueError("Expected one per-ROI output BatchNorm")
            with torch.no_grad():
                norms[0].bias.copy_(location_tensor)
                norms[0].weight.copy_(scale_tensor)
            write_json(out / "linear_initialization.json",
                       dict(source="training sounds, repeat means",
                            indices=splits["train"], offset=location.tolist(),
                            scale=scale.tolist(), observed_bins_per_roi=training_counts.tolist()))
        if args.linear_normalization == "fixed-training":
            calibration = calibrate_linear_normalization(model, train_evaluation_loader)
            calibration["indices"] = splits["train"]
            write_json(out / "linear_normalization.json", calibration)
        if args.linear_init == "baseline":
            initialize_linear_from_baseline(model, location, scale)
        model_out = out / name
        model_out.mkdir()
        # Check real shapes, finite loss and gradients before fitting.
        batch = neural_collate([ds[i] for i in splits["train"][:2]])
        model.eval()  # Contract check must not change BatchNorm running statistics.
        pred = model(batch["stims"])
        assert tuple(pred.shape)==(2,ds.N_neurons,1,len(ds.time_ms))
        loss = mse_loss(pred, batch["responses"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite initial loss")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        model.zero_grad(set_to_none=True)
        del pred, loss, batch
        def log(row):
            # Evaluate a fixed, non-shuffled training loader in inference mode.
            # This separates generalization from train/eval normalization effects.
            train_pred, train_target = predict(model, train_evaluation_loader)
            train_baseline = np.broadcast_to(location[None, :, None], train_target.shape)
            row["train_eval_mse"] = comparison_scores(
                train_pred, train_target, train_baseline)["model"]["mean_mse"]
            if not all(np.isfinite(v) for v in row.values()):
                raise ValueError(f"Nonfinite training history: {row}")
            with (model_out / "history.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(name, row, flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        fitter = Fitter(model, loaders["train"], loaders["validation"],
                        optimizer=optimizer,
                        max_epochs=config["epochs"], patience=args.patience,
                        monitor="val_loss", mode="min", val_metrics={},
                        ckpt_path=model_out/"best.pt", state_path=model_out/"state.pt",
                        track_train_metrics=False, log_fn=log)
        # Fitter coerces constructor paths to Path; specialize only its atomic
        # state replacement, without modifying the installed deepSTRF package.
        fitter.state_path = CheckpointPath(fitter.state_path)
        history = fitter.fit()
        write_json(model_out/"history.json", history)
        constant = location
        result = dict(parameters=sum(p.numel() for p in model.parameters()),
                      epochs_completed=len(history),
                      best_validation_loss=min(row["val_loss"] for row in history),
                      best_epoch=min(history, key=lambda row: row["val_loss"])["epoch"],
                      test_evaluated=evaluate_test)
        for subset in (["validation","test"] if evaluate_test else ["validation"]):
            pred, target = predict(model, loaders[subset])
            baseline = np.broadcast_to(constant[None,:,None],target.shape)
            result[subset]=comparison_scores(pred,target,baseline)
            post = ds.time_ms >= 0
            result[subset]['post_onset'] = comparison_scores(
                pred[:,:,post], target[:,:,post], baseline[:,:,post])
            result[subset]['sites'] = {}
            for site, bounds in zip(sites, ds.roi_ranges):
                site_target = target[:, bounds]
                if not np.isfinite(site_target).any():
                    result[subset]['sites'][str(site)] = dict(status="no_observed_targets")
                    continue
                result[subset]['sites'][str(site)] = comparison_scores(
                    pred[:, bounds], site_target, baseline[:, bounds])
            np.savez_compressed(model_out/f"{subset}_predictions.npz",
                                prediction=pred,target=target,time_ms=ds.time_ms,
                                stimulus_indices=np.array(splits[subset]))
        write_json(model_out/"metrics.json",result)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4),layout="constrained")
        for key, label in [("train_loss", "Training batches"),
                           ("train_eval_mse", "Training in evaluation mode"),
                           ("val_loss", "Validation")]:
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=label)
        ax.axhline(result["validation"]["constant_baseline"]["mean_mse"],
                   color="gray", ls="--", label="Validation constant baseline")
        ax.axvline(result["best_epoch"], color="gray", ls=":", label="Selected epoch")
        ax.set(xlabel="Epoch",ylabel="MSE",title=f"{name} / {len(sites)} site(s)")
        ax.legend(); fig.savefig(model_out/"learning_curve.png"); plt.close(fig)
        fig, axes = plt.subplots(3,1,figsize=(8,7),sharex=True,layout="constrained")
        for ax,roi in zip(axes,[0,ds.N_neurons//2,ds.N_neurons-1]):
            observed_sounds = np.flatnonzero(np.isfinite(target[:, roi]).any(axis=1))
            if not len(observed_sounds):
                ax.text(0.05, 0.5, f"ROI {roi}: no validation recording", transform=ax.transAxes)
                continue
            sound = observed_sounds[0]
            ax.plot(ds.time_ms,target[sound,roi],label="Measured repeat mean")
            ax.plot(ds.time_ms,pred[sound,roi],label="Prediction")
            ax.plot(ds.time_ms,baseline[sound,roi],label="Training mean", ls=":")
            ax.set_title(f"Sound {splits[subset][sound]}")
            ax.axvline(0,color="gray",ls="--"); ax.set_ylabel(f"ROI {roi}")
        axes[0].legend(); axes[-1].set_xlabel("Time from sound onset (ms; 32 ms/frame)")
        fig.suptitle(f"{name}: {subset} (fixed ROIs, first observed sound per ROI)")
        fig.savefig(model_out/"predictions.png"); plt.close(fig)
        summaries[name]={k:result[k] for k in ["parameters","epochs_completed","best_validation_loss","test_evaluated"]}
        del fitter,model,loaders
    memory=psutil.Process().memory_info()
    write_json(out/"summary.json",dict(status="complete",models=summaries,
               elapsed_seconds=time.perf_counter()-started,
               peak_rss_bytes=getattr(memory,"peak_wset",memory.rss)))
    print(f"Complete: {out}",flush=True)


if __name__ == "__main__":
    main()
