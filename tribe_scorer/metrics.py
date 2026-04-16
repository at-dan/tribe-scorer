"""
Compute creative performance metrics from TRIBE V2 neural predictions.

Pipeline:
  1. Map prediction vertices to ROI masks (from regions.py)
  2. Compute per-ROI mean activation timeseries
  3. Derive summary statistics per metric
  4. Normalize across a batch for 0-100 scoring
"""

import numpy as np
from scipy import stats as sp_stats

from .regions import METRICS, build_roi_masks


# ── Timeseries extraction ──────────────────────────────────────────────────

def compute_roi_timeseries(
    predictions: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Mean activation timeseries per ROI.

    Args:
        predictions: (n_timesteps, n_vertices)
        masks: {metric_name: bool mask of shape (n_vertices,)}

    Returns:
        {metric_name: array of shape (n_timesteps,)}
    """
    ts = {}
    for name, mask in masks.items():
        n_verts = mask.sum()
        if n_verts == 0:
            ts[name] = np.zeros(predictions.shape[0])
        else:
            ts[name] = predictions[:, mask].mean(axis=1)
    return ts


# ── Per-metric summary statistics ──────────────────────────────────────────

def score_timeseries(ts: np.ndarray, tr_sec: float = 1.5) -> dict:
    """
    Derive summary scores from one ROI's activation timeseries.

    Args:
        ts:     (n_timesteps,) activation values
        tr_sec: seconds per timestep (fMRI TR)

    Returns:
        dict of summary metrics
    """
    if len(ts) == 0 or np.all(ts == 0):
        return {
            "mean_activation": 0.0,
            "peak_activation": 0.0,
            "peak_time_sec": 0.0,
            "sustained_ratio": 0.0,
            "trend": 0.0,
        }

    mean_val = float(np.mean(ts))
    peak_val = float(np.max(ts))
    peak_idx = int(np.argmax(ts))
    peak_time = round(peak_idx * tr_sec, 1)

    # Sustained engagement: fraction of timesteps above the global median
    median_val = float(np.median(ts))
    sustained = float(np.mean(ts > median_val))

    # Trend: slope of linear regression across time
    # Positive → engagement is building; negative → declining
    if len(ts) > 2:
        slope = float(sp_stats.linregress(np.arange(len(ts)), ts).slope)
    else:
        slope = 0.0

    return {
        "mean_activation": round(mean_val, 6),
        "peak_activation": round(peak_val, 6),
        "peak_time_sec": peak_time,
        "sustained_ratio": round(sustained, 4),
        "trend": round(slope, 8),
    }


# ── Full creative scoring ──────────────────────────────────────────────────

def compute_creative_scores(
    predictions: np.ndarray,
    tr_sec: float = 1.5,
    masks: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Full scoring pipeline for a single creative.

    Args:
        predictions: (n_timesteps, n_vertices) from TRIBE V2
        tr_sec: temporal resolution in seconds
        masks: precomputed ROI masks (computed on first call if None)

    Returns:
        {metrics, overall_raw, timeline, duration_sec, n_timesteps, n_vertices}
    """
    if masks is None:
        masks = build_roi_masks(predictions.shape[1])

    timeseries = compute_roi_timeseries(predictions, masks)

    # Per-metric summary
    metric_scores: dict[str, dict] = {}
    for name, ts in timeseries.items():
        summary = score_timeseries(ts, tr_sec)
        metric_scores[name] = {
            "label": METRICS[name]["label"],
            "description": METRICS[name]["description"],
            **summary,
        }

    # Weighted composite (using mean_activation as the base signal)
    w_sum = 0.0
    w_total = 0.0
    for name, scores in metric_scores.items():
        w = METRICS[name]["weight"]
        w_sum += scores["mean_activation"] * w
        w_total += w
    overall_raw = round(w_sum / w_total, 6) if w_total > 0 else 0.0

    # Second-by-second timeline
    n_t = predictions.shape[0]
    timeline = []
    for t in range(n_t):
        point = {"time_sec": round(t * tr_sec, 1)}
        for name, ts in timeseries.items():
            point[name] = round(float(ts[t]), 6)
        timeline.append(point)

    return {
        "metrics": metric_scores,
        "overall_raw": overall_raw,
        "timeline": timeline,
        "duration_sec": round(n_t * tr_sec, 1),
        "n_timesteps": n_t,
        "n_vertices": predictions.shape[1],
    }


# ── Batch normalization ────────────────────────────────────────────────────

def normalize_batch_scores(all_scores: list[dict]) -> list[dict]:
    """
    Normalize raw scores across a batch into 0-100 percentile scores.

    Mutates each dict in-place, adding a ``score`` key to each metric
    and an ``overall_score`` key at the top level.
    """
    if len(all_scores) <= 1:
        # Absolute scoring: map raw activation to 0-100 where
        # 50 = zero (baseline brain response), 100 = strong positive,
        # 0 = strong negative.  Scale factor of 0.15 maps the typical
        # fMRI activation range to a readable score.
        SCALE = 0.15
        for s in all_scores:
            for name, m in s["metrics"].items():
                raw = m["mean_activation"]
                absolute = 50 + (raw / SCALE) * 50
                m["score"] = round(max(0.0, min(100.0, absolute)), 1)

            w_sum = 0.0
            w_total = 0.0
            for name, m in s["metrics"].items():
                w = METRICS[name]["weight"]
                w_sum += m["score"] * w
                w_total += w
            s["overall_score"] = round(w_sum / w_total, 1)
        return all_scores

    # Collect raw mean_activations
    metric_vals: dict[str, list[float]] = {n: [] for n in METRICS}
    overall_vals: list[float] = []

    for s in all_scores:
        for name in METRICS:
            metric_vals[name].append(s["metrics"][name]["mean_activation"])
        overall_vals.append(s["overall_raw"])

    # Percentile-rank each creative per metric
    for i, s in enumerate(all_scores):
        for name in METRICS:
            arr = np.array(metric_vals[name])
            pct = sp_stats.percentileofscore(arr, arr[i], kind="mean")
            s["metrics"][name]["score"] = round(pct, 1)

        ov = np.array(overall_vals)
        s["overall_score"] = round(
            sp_stats.percentileofscore(ov, ov[i], kind="mean"), 1
        )

    return all_scores
