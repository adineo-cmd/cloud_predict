# cli/evaluate_command.py
# Rigorous evaluation: full classification metrics, ROC/AUC, PR curves,
# optimal-threshold search, JSON report export.
#
# By default this scores the held-out TEST split — files never used for
# training, checkpointing, threshold search or calibration — so the numbers
# are honest generalization estimates, not rescored training data.
import json
from pathlib import Path

import click
import numpy as np

from errors import ModelNotFoundError


def _load_split(split: str):
    """Generator over a named split ('test' | 'validation' | 'training'), no augmentation."""
    from cli.data import load_split

    return load_split(split, augment=False, seed=None)


def find_optimal_threshold(y_true, scores):
    """Grid-search decision threshold maximizing macro F1 (balanced across classes
    under class imbalance); returns (best_t, best_f1, curve)."""
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.05, 0.95, 91)
    curve = []
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (scores > t).astype(int), average="macro", zero_division=0)
        curve.append({"threshold": round(float(t), 3), "f1": round(float(f1), 4)})
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1, curve


def _ensemble_probabilities(models, data):
    """Mean sigmoid probability across ensemble members for a generator."""
    per_model = [m.predict(data, verbose=0).flatten() for m in models]
    return np.clip(np.mean(np.stack(per_model), axis=0), 1e-6, 1 - 1e-6)


def run_evaluate(model_path=None, threshold=None, find_optimal=True,
                 plot_roc=True, split="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 roc_curve, auc, precision_recall_curve, average_precision_score)
    from config import PREDICTION_THRESHOLD, RESULTS_DIR, PLOT_CONFUSION_MATRIX
    from cli.predict_command import load_model_ensemble

    thr = threshold if threshold is not None else PREDICTION_THRESHOLD

    models, mpaths = load_model_ensemble(model_path)
    data = _load_split(split)
    class_names = list(data.class_indices.keys())
    click.secho(f"▶ Evaluating {len(mpaths)} model(s) on {len(data.filenames)} "
                f"'{split}' samples...", fg="cyan")

    # --- Calibration artifact (validation-fitted) --------------------------------
    from config import CALIBRATION_PATH, CALIBRATION_TRUSTED_FIT
    from calibration import load_calibration, confidence_report, apply_calibration
    from cli.predict_command import resolve_calibration
    cal = load_calibration(CALIBRATION_PATH)
    # Report on the scored split only what inference would actually use.
    trusted = bool(cal) and cal.get("fitted_on") == CALIBRATION_TRUSTED_FIT
    temperature, platt = resolve_calibration(None)
    method = "platt" if platt else ("temperature" if temperature != 1.0 else "none")

    scores = _ensemble_probabilities(models, data)
    y_true = np.array(data.classes)
    # Decisions use the calibrated probability (same mapping inference uses);
    # raw scores are kept for threshold-free ranking metrics (ROC/PR).
    y_scores = np.array([apply_calibration(s, temperature, platt) for s in scores])
    y_pred = (y_scores > thr).astype(int)

    conf_stats = confidence_report(scores, y_true, temperature)
    b, a = conf_stats["before_scaling"], conf_stats["after_scaling"]
    click.echo("\nConfidence Diagnostics (calibration artifact: "
               f"{cal.get('fitted_on') if cal else 'none'}, method={method}):")
    click.echo(f"  Temperature T          : {temperature:.4f}"
               + (" (from calibration.json)" if trusted else " (not trusted — raw scores)"))
    if platt:
        click.echo(f"  Platt scaling          : a={platt['a']:.4f} b={platt['b']:.4f}")
    click.echo(f"  Avg conf overall       : {b['avg_confidence_overall']:.4f} → "
               f"{a['avg_confidence_overall']:.4f} (calibrated)")
    click.echo(f"  Avg conf on correct    : {b['avg_confidence_correct']} → "
               f"{a['avg_confidence_correct']}")
    click.echo(f"  Avg conf on incorrect  : {b['avg_confidence_incorrect']} → "
               f"{a['avg_confidence_incorrect']}")
    click.echo(f"  ECE                    : {b['ece']:.4f} → {a['ece']:.4f}")
    if not trusted:
        click.secho("  💡 Run 'python -m cli calibrate' (validation split) to fit a trusted "
                    "temperature.", fg="yellow")

    accuracy = float(np.mean(y_pred == y_true))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    click.echo("\n" + "=" * 70)
    click.echo(f"EVALUATION RESULTS  (split='{split}', threshold={thr})")
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
        "model": [str(p) for p in mpaths], "split": split, "threshold": thr,
        "n_samples": int(len(y_true)),
        "accuracy": round(accuracy, 4), "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4), "average_precision": round(ap, 4),
        "confusion_matrix": cm.tolist(), "class_names": class_names,
        "calibration": {"method": method, "temperature": temperature,
                        "platt": platt, "artifact_present": bool(cal), "trusted": trusted,
                        "ece_raw": b["ece"], "ece_calibrated": a["ece"],
                        "confidence_diagnostics": conf_stats},
    }

    # Optimal threshold: search on validation (never test), apply to the scored split.
    if find_optimal and split != "validation":
        try:
            val_data = _load_split("validation")
            val_scores_raw = _ensemble_probabilities(models, val_data)
            val_scores = np.array([apply_calibration(s, temperature, platt)
                                   for s in val_scores_raw])
            val_true = np.array(val_data.classes)
            best_t, best_f1_val, _ = find_optimal_threshold(val_true, val_scores)
            y_pred_opt = (y_scores > best_t).astype(int)
            acc_opt = float(np.mean(y_pred_opt == y_true))
            f1_opt = float(f1_score(y_true, y_pred_opt, zero_division=0))
            click.secho(f"\nVal-selected threshold {best_t:.2f} applied to '{split}': "
                        f"acc={acc_opt:.4f} F1={f1_opt:.4f} (vs thr=0.5: "
                        f"acc={accuracy:.4f} F1={f1:.4f})", fg="green", bold=True)
            report["val_selected_threshold"] = {"threshold": best_t,
                                                "f1_val": best_f1_val,
                                                "accuracy_on_split": round(acc_opt, 4),
                                                "f1_on_split": round(f1_opt, 4)}
        except Exception as e:
            click.secho(f"⚠ Threshold transfer skipped: {e}", fg="yellow")

    # Optimal threshold search on the scored split itself (kept for the validation
    # split case, where selection and scoring coincide).
    if find_optimal and split == "validation":
        best_t, best_f1, curve = find_optimal_threshold(y_true, y_scores)
        click.secho(f"\nOptimal threshold: {best_t:.2f} (F1={best_f1:.4f})", fg="green", bold=True)
        report["optimal_threshold"] = {"threshold": best_t, "f1": best_f1, "curve": curve}

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

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        axes[0].set(title="ROC", xlabel="False Positive Rate", ylabel="True Positive Rate")
        axes[0].legend(); axes[0].grid(alpha=.3)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        axes[1].plot(rec, prec, label=f"AP={ap:.3f}")
        axes[1].set(title="Precision-Recall", xlabel="recall", ylabel="precision")
        axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        p = RESULTS_DIR / "threshold_pr_curves.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        click.echo(f"✓ Saved {p}")

    if PLOT_CONFUSION_MATRIX:
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix ({split} split)")
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
