"""Metadata-only, deterministic pilot selection; never rank neural responses."""
from itertools import combinations
from math import prod, isfinite

MIB = 1024 ** 2


def select_diverse_sites(metadata, *, max_sites=5, response_budget_mib=64):
    """Choose at most one site per animal on a common relative frame grid.

    Priorities: animal count, stimulus-count variants, category coverage, then
    smallest original float64 response payload. Ties use ascending site indices.
    This is metadata diversity, not evidence of biological representativeness.
    Budget excludes runtime, Python objects, audio, batches and trainin
    """
    if (isinstance(max_sites, bool) or not isinstance(max_sites, int) or max_sites < 1
            or not isfinite(response_budget_mib) or response_budget_mib <= 0):
        raise ValueError("Positive site count and finite response budget required")
    candidates = {}
    seen = set()
    for meta in metadata:
        site = meta["site_index"]
        if site in seen:
            raise ValueError("Duplicate site index")
        seen.add(site)
        frames, repeats, stimuli, rois = meta["shape_time_repeat_stimulus_roi"]
        if min(frames, repeats, stimuli, rois) <= 0:
            raise ValueError("Empty response shape")
        row = dict(site_index=site, bat_id=meta["bat_id"],
                   site_number=meta["site_number"], roi_count=rois,
                   stimulus_count=stimuli, repeat_count=repeats,
                   categories=sorted(set(meta["categories"])),
                   frame_grid=[frames, meta["onset_frame_index"]],
                   response_bytes=prod((frames, repeats, stimuli, rois)) * 8)
        # A larger site with identical selection features cannot improve the score.
        key = (row["bat_id"], stimuli, tuple(row["categories"]), tuple(row["frame_grid"]))
        old = candidates.get(key)
        if old is None or (row["response_bytes"], site) < (old["response_bytes"], old["site_index"]):
            candidates[key] = row
    rows = sorted(candidates.values(), key=lambda r: r["site_index"])
    best, best_score = None, None
    for size in range(1, min(max_sites, len({r["bat_id"] for r in rows})) + 1):
        for group in combinations(rows, size):
            if len({r["bat_id"] for r in group}) != size:
                continue
            if len({tuple(r["frame_grid"]) for r in group}) != 1:
                continue
            payload = sum(r["response_bytes"] for r in group)
            if payload > response_budget_mib * MIB:
                continue
            score = (size, len({r["stimulus_count"] for r in group}),
                     len(set().union(*(r["categories"] for r in group))), -payload)
            if best_score is None or score > best_score:
                best, best_score = group, score
    if best is None:
        raise ValueError("No site fits the response budget")
    return dict(
        site_indices=[r["site_index"] for r in best],
        bat_ids=sorted(r["bat_id"] for r in best), sites=list(best),
        response_bytes=sum(r["response_bytes"] for r in best),
        response_budget_mib=response_budget_mib, max_sites=max_sites,
        policy="one site per bat; maximize bats, stimulus-count variants, categories; minimize response bytes",
        budget_scope="Original float64 responses only; excludes runtime, objects, audio, batches and training",
        limitations="No selection by neural activity. Sex, age and anatomical diversity not established.",
    )


def require_available_memory(response_bytes, *, reserve_mib=768):
    """Windows preflight: heuristic headroom, not a bound on peak RAM usage."""
    import ctypes
    import sys
    if sys.platform != "win32":
        return

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_page", "avail_page",
                "total_virtual", "avail_virtual", "avail_extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot determine available physical memory")
    required = response_bytes * 3 + reserve_mib * MIB
    if status.avail_phys < required:
        raise MemoryError(
            f"Only {status.avail_phys / MIB:.0f} MiB available; "
            f"pilot requires at least {required / MIB:.0f} MiB of headroom. "
            "Save work and close unused apps/kernels before retrying.")
