"""Read-only access to Lawlor et al. data; axes remain in h5py order.

No inferred audio identities, time correction, or fluorescence conversion.
"""
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


class BatData:
    def __init__(self, path):
        self.file = h5py.File(Path(path), "r")
        self.struct = self.file["MegadataStruct"]
        shapes = {d.shape for d in self.struct.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            self.file.close()
            raise ValueError("Unexpected struct dimensions")
        self.n_sites = self.struct["BatId"].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def dataset(self, site, field):
        if not 0 <= site < self.n_sites:
            raise IndexError(f"site must be in [0, {self.n_sites})")
        return self.file[self.struct[field][site, 0]]

    def resolve(self, obj):
        if isinstance(obj, h5py.Reference):
            if not obj:
                return None
            obj = self.file[obj]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Expected a dataset, got {type(obj)}")
        # MATLAB stores the empty array's dimensions as data, e.g. [0, 0].
        if obj.attrs.get("MATLAB_empty", 0):
            return np.empty(0)
        data = obj[()]
        cls = obj.attrs.get("MATLAB_class", b"")
        if isinstance(cls, bytes):
            cls = cls.decode()
        if cls == "char":
            return np.asarray(data, dtype="<u2").tobytes().decode("utf-16-le")
        if h5py.check_dtype(ref=obj.dtype) is not None:
            result = np.empty(data.shape, dtype=object)
            for index in np.ndindex(data.shape):
                result[index] = self.resolve(data[index])
            return result
        return data

    def field(self, site, field):
        return self.resolve(self.dataset(site, field))

    def vocal_metadata(self, site):
        d = self.dataset(site, "VocalizationMat")
        labels = list(self.field(site, "VocalizationMatDimension").ravel())
        expected = ["ROI number", "Vocalization stimulus", "Repetition number", "Frame number"]
        if labels != expected or d.ndim != 4:
            raise ValueError(f"Unexpected dimensions at site {site}: {labels}, {d.shape}")
        categories = list(self.field(site, "VocalizationCategory").ravel())
        time = self.field(site, "FrametoSecond").ravel()
        n_roi = int(self.field(site, "RoiNb").item())
        if (len(time), len(categories), n_roi) != (d.shape[0], d.shape[2], d.shape[3]):
            raise ValueError(f"Metadata/shape mismatch at site {site}")
        if not np.isfinite(time).all() or not (np.diff(time) > 0).all():
            raise ValueError("Invalid time axis")
        zeros = np.flatnonzero(np.isclose(time, 0))
        if len(zeros) != 1:
            raise ValueError("Expected exactly one stimulus-onset frame")
        step = float(np.median(np.diff(time)))
        rate = float(self.field(site, "FrameRate").item())
        return {
            "site_index": site,
            "bat_id": int(self.field(site, "BatId").item()),
            "site_number": int(self.field(site, "SiteNb").item()),
            "shape_time_repeat_stimulus_roi": list(d.shape),
            "axes_h5py": labels[::-1],
            "categories": categories,
            "category_counts": dict(Counter(categories)),
            "time_stored": time.tolist(),
            "time_unit": "not explicitly encoded; provisionally interpreted as ms",
            "time_step_stored": step,
            "uniform_time_steps": bool(np.allclose(np.diff(time), step)),
            "onset_frame_index": int(zeros[0]),
            "frame_rate_stored": rate,
            "dt_ms_if_frame_rate_is_hz": 1000 / rate,
            "time_rate_mismatch_if_ms_and_hz": not bool(np.isclose(step, 1000 / rate)),
        }

    def inventory(self):
        rows = []
        for site in range(self.n_sites):
            row = self.vocal_metadata(site)
            row["fields"] = {}
            for field in self.struct:
                d = self.dataset(site, field)
                empty = bool(d.attrs.get("MATLAB_empty", 0))
                row["fields"][field] = {
                    "shape": list(d.shape), "dtype": str(d.dtype), "matlab_empty": empty,
                    "scalar_nan": bool(not empty and d.size == 1 and d.dtype.kind == "f"
                                       and np.isnan(d[()]).all()),
                }
            rows.append(row)
        return rows

    def vocal_roi(self, site, roi):
        """Load only one ROI: (time, repetition, stimulus)."""
        self.vocal_metadata(site)
        d = self.dataset(site, "VocalizationMat")
        if not 0 <= roi < d.shape[-1]:
            raise IndexError("ROI index out of bounds")
        return d[:, :, :, roi]

    def vocal_quality(self, site, roi_block=32):
        """Scan all values at one site in small ROI blocks; never load full MAT."""
        if roi_block < 1:
            raise ValueError("roi_block must be positive")
        meta = self.vocal_metadata(site)
        d = self.dataset(site, "VocalizationMat")
        counts = {"nan": 0, "inf": 0, "finite": 0}
        minimum, maximum = np.inf, -np.inf
        baseline_medians = []
        before = np.asarray(meta["time_stored"]) < 0
        for start in range(0, d.shape[-1], roi_block):
            block = d[..., start:start + roi_block]
            counts["nan"] += int(np.isnan(block).sum())
            counts["inf"] += int(np.isinf(block).sum())
            finite = block[np.isfinite(block)]
            counts["finite"] += int(finite.size)
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
            for roi in range(block.shape[-1]):
                baseline = block[before, :, :, roi]
                baseline = baseline[np.isfinite(baseline)]
                if baseline.size:
                    baseline_medians.append(float(np.median(baseline)))
        return {
            "site_index": site, "values": int(d.size), **counts,
            "minimum": minimum if counts["finite"] else None,
            "maximum": maximum if counts["finite"] else None,
            "median_of_roi_baseline_medians": float(np.median(baseline_medians)) if baseline_medians else None,
            "baseline_definition": "all stored time values < 0; descriptive only",
        }


def finite_mean(values, axis):
    """Ignore nonfinite observations without turning missing samples into zeros."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
