# calibration.py
# Probability calibration & confidence post-processing for the cloud classifier.
#
# Why this matters: a raw sigmoid output from a network trained with cross-entropy
# is often *over-confident* (or under-confident) — e.g. it says 95% when its true
# accuracy at that score is only ~80%. Calibration maps raw scores to trustworthy
# probabilities, and temperature scaling lets us sharpen them so confident-and-correct
# predictions report higher (and empirically more accurate) confidence values.
#
# This module provides:
#   * fit_temperature()      – learn a single scalar T on a held-out split
#   * apply_temperature()    – calibrated + sharpened probability
#   * expected_calibration_error() – standard calibration quality metric
#   * evaluate_confidence()  – avg confidence on correct vs incorrect predictions
#   * save/load helpers for models/calibration.json
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def apply_temperature(score: float, temperature: float) -> float:
    """Map a raw sigmoid score through temperature T.

    T < 1  → sharpened (more decisive, higher confidence on clear cases)
    T > 1  → softened  (more conservative probabilities)
    T = 1  → identity
    """
    eps = 1e-7
    p = float(np.clip(score, eps, 1 - eps))
    logit = np.log(p / (1.0 - p)) / max(float(temperature), 1e-6)
    return float(1.0 / (1.0 + np.exp(-logit)))


def nll_loss(scores: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Negative log-likelihood of the temperature-scaled probabilities."""
    probs = np.array([apply_temperature(s, temperature) for s in scores])
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    ll = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
    return float(-np.mean(ll))


def fit_temperature(scores, labels, grid_lo: float = 0.3, grid_hi: float = 3.0,
                    coarse_steps: int = 55, fine_steps: int = 40) -> float:
    """Learn the optimal temperature T by grid search minimizing NLL.

    Two-stage search (coarse → refined) — robust and dependency-free.
    Guards against degenerate fits: returns T = 1.0 when there is too little
    calibration data (< MIN_CALIBRATION_SAMPLES), only one class present, or
    when scaling does not improve NLL over the identity mapping.
    """
    MIN_CALIBRATION_SAMPLES = 16
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.size < MIN_CALIBRATION_SAMPLES or not np.array_equal(np.unique(labels), [0, 1]):
        return 1.0

    # Coarse grid
    best_t, best_loss = 1.0, nll_loss(scores, labels, 1.0)
    for t in np.linspace(grid_lo, grid_hi, coarse_steps):
        loss = nll_loss(scores, labels, t)
        if loss < best_loss:
            best_t, best_loss = float(t), loss

    # Fine grid around the coarse optimum
    span = (grid_hi - grid_lo) / coarse_steps
    for t in np.linspace(max(grid_lo, best_t - span), min(grid_hi, best_t + span), fine_steps):
        loss = nll_loss(scores, labels, t)
        if loss < best_loss:
            best_t, best_loss = float(t), loss

    # Only adopt a non-identity temperature if it meaningfully improves NLL
    if abs(best_loss - nll_loss(scores, labels, 1.0)) < 1e-4:
        return 1.0
    return round(best_t, 4)


# ---------------------------------------------------------------------------
# Calibration quality metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(scores, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): weighted gap between confidence and accuracy."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    confidences = np.where(scores >= 0.5, scores, 1.0 - scores)
    accuracies = ((scores > 0.5).astype(int) == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (mask.sum() / len(scores)) * abs(avg_acc - avg_conf)
    return float(ece)


def confidence_report(scores, labels, temperature: float = 1.0) -> dict:
    """Confidence diagnostics before/after temperature scaling."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    def stats(raw_scores):
        preds = (raw_scores > 0.5).astype(int)
        confs = np.where(preds == 1, raw_scores, 1.0 - raw_scores)
        correct = preds == labels
        return {
            "avg_confidence_overall": round(float(confs.mean()), 4),
            "avg_confidence_correct": round(float(confs[correct].mean()), 4) if correct.any() else None,
            "avg_confidence_incorrect": round(float(confs[~correct].mean()), 4) if (~correct).any() else None,
            "high_conf_ratio_0.9": round(float((confs >= 0.9).mean()), 4),
            "ece": round(expected_calibration_error(raw_scores, labels), 4),
        }

    base = stats(scores)
    scaled = stats(np.array([apply_temperature(s, temperature) for s in scores]))
    return {"temperature": temperature, "before_scaling": base, "after_scaling": scaled}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_calibration(calibration_dir: Path, data: dict) -> Path:
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    out = calibration_dir / "calibration.json"
    out.write_text(json.dumps(data, indent=2))
    return out


def load_calibration(calibration_path: Path) -> dict | None:
    """Load calibration artifact; returns None if missing/corrupt (caller falls back)."""
    try:
        data = json.loads(Path(calibration_path).read_text())
        if "temperature" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def get_temperature(model_path: Path, default: float = 1.0) -> float:
    """Convenience: read the temperature associated with a model artifact."""
    cal = load_calibration(Path(model_path).parent / "calibration.json")
    if cal is None:
        return default
    return float(cal.get("temperature", default))
