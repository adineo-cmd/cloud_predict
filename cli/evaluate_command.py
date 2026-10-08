# cli/evaluate_command.py
# Rigorous evaluation: full classification metrics, ROC/AUC, PR curves,
# optimal-threshold search, JSON report export.
import json
from pathlib import Path

import click
import numpy as np

from errors import ModelNotFoundError


def _load_split(split: str):
    from config import (DATASET_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT,
                        CLASS_MODE, SHUFFLE_VAL)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 validation_split=VALIDATION_SPLIT)
    data = datagen.flow_from_directory(
        str(DATASET_DIR), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation" if split == "validation" else "training",
        shuffle=False)
    return data


def find_optimal_threshold(y_true, scores):
    """Grid-search decision threshold maximizing F1; returns (best_t, best_f1, curve)."""
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.05, 0.95, 91)
    curve = []
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (scores > t).astype(int), zero_division=0)
        curve.append({"threshold": round(float(t), 3), "f1": round(float(f1), 4)})
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1, curve


def run_evaluate(model_path=None, threshold=None, find_optimal=True,
                 plot_roc=True, split="validation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 roc_curve, auc, precision_recall_curve, average_precision_score)
    from tensorflow.keras.models import load_model
    from config import MODEL_PATH, PREDICTION_THRESHOLD, RESULTS_DIR, PLOT_CONFUSION_MATRIX

    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(path)
    model = load_model(path)
    thr = threshold if threshold is not None else PREDICTION_THRESHOLD

    data = _load_split(split)
    class_names = list(data.class_indices.keys())
    click.secho(f"▶ Evaluating {path.name} on {len(data.filenames)} '{split}' samples...", fg="cyan")

    scores = model.predict(data, verbose=1).flatten()
    y_true = np.array(data.classes)
    y_pred = (scores > thr).astype(int)

    # --- Confidence calibration diagnostics (temperature scaling) ---------------
    from config import CALIBRATION_PATH, DEFAULT_TEMPERATURE
    from calibration import load_calibration, confidence_report
    cal = load_calibration(CALIBRATION_PATH)
    temperature = float(cal["temperature"]) if cal else DEFAULT_TEMPERATURE
    conf_stats = confidence_report(scores, y_true, temperature)
    b, a = conf_stats["before_scaling"], conf_stats["after_scaling"]
    click.echo("\nConfidence Diagnostics:")
    click.echo(f"  Temperature T          : {temperature:.4f}"
               + (" (from calibration.json)" if cal else " (no artifact — raw scores)"))
    click.echo(f"  Avg conf overall       : {b['avg_confidence_overall']:.4f} → "
               f"{a['avg_confidence_overall']:.4f} (calibrated)")
    click.echo(f"  Avg conf on correct    : {b['avg_confidence_correct']} → "
               f"{a['avg_confidence_correct']}")
    click.echo(f"  Avg conf on incorrect  : {b['avg_confidence_incorrect']} → "
               f"{a['avg_confidence_incorrect']}")
    click.echo(f"  ECE                    : {b['ece']:.4f} → {a['ece']:.4f}")
    if not cal:
        click.secho("  💡 Run 'python -m cli calibrate' to fit a temperature and boost "
                    "trustworthy confidence.", fg="yellow")

    accuracy = float(np.mean(y_pred == y_true))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    click.echo("\n" + "=" * 70)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 70)
    click.echo("\nClassification Report:\n")
    click.echo(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    click.echo(f"Confusion Matrix (rows=true, cols=pred):\n{cm}")
    click.echo(f"\nAccuracy : {accuracy:.4f}\nF1 (pos='{class_names[1]}') : {f1:.4f}")

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, scores)
    click.echo(f"ROC-AUC  : {roc_auc:.4f}\nAvgPrec  : {ap:.4f}")

    ratio = float(np.mean(y_pred))
    if ratio > 0.8 or ratio < 0.2:
        click.secho(f"⚠ Class-collapse warning: positive predictions at {ratio*100:.1f}%", fg="yellow")

    report = {
        "model": str(path), "split": split, "threshold": thr,
        "n_samples": int(len(y_true)),
        "accuracy": round(accuracy, 4), "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4), "average_precision": round(ap, 4),
        "confusion_matrix": cm.tolist(), "class_names": class_names,
        "calibration": {"temperature": temperature, "artifact_present": bool(cal),
                        "confidence_diagnostics": conf_stats},
    }

    # Optimal threshold search
    if find_optimal:
        best_t, best_f1, curve = find_optimal_threshold(y_true, scores)
        click.secho(f"\nOptimal threshold: {best_t:.2f} (F1={best_f1:.4f})", fg="green", bold=True)
        report["optimal_threshold"] = {"threshold": best_t, "f1": best_f1, "curve": curve}

        if plot_roc:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot([t["threshold"] for t in curve], [t["f1"] for t in curve])
            axes[0].axvline(best_t, ls="--", c="r", label=f"best={best_t:.2f}")
            axes[0].set(title="F1 vs Decision Threshold", xlabel="threshold", ylabel="F1")
            axes[0].legend(); axes[0].grid(alpha=.3)
            prec, rec, _ = precision_recall_curve(y_true, scores)
            axes[1].plot(rec, prec, label=f"AP={ap:.3f}")
            axes[1].set(title="Precision-Recall", xlabel="recall", ylabel="precision")
            axes[1].legend(); axes[1].grid(alpha=.3)
            fig.tight_layout()
            p = RESULTS_DIR / "threshold_pr_curves.png"
            fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
            click.echo(f"✓ Saved {p}")

    # Plots
    if plot_roc:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=.5)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(); ax.grid(alpha=.3)
        p = RESULTS_DIR / "roc_curve.png"
        fig.tight_layout(); fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        click.echo(f"✓ Saved {p}")

    if PLOT_CONFUSION_MATRIX:
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True"); plt.xlabel("Predicted")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.tight_layout()
        p = RESULTS_DIR / "confusion_matrix.png"
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        click.echo(f"✓ Saved {p}")

    out = RESULTS_DIR / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2))
    click.secho(f"\n✓ Report written to {out}", fg="green")
    return report
