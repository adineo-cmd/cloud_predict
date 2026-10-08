# cli/calibrate_command.py
# Standalone confidence calibration: fits a temperature on a chosen split and
# writes models/calibration.json used by predict / evaluate / the web app.
import click
import numpy as np


def run_calibrate(model_path=None, split="validation", apply_only=False):
    """Fit (or report) the temperature-scaling calibration for a model."""
    from pathlib import Path
    from config import (MODEL_PATH, CALIBRATION_PATH, TEMPERATURE_BOUNDS,
                        DEFAULT_TEMPERATURE)
    from calibration import (fit_temperature, save_calibration, load_calibration,
                             confidence_report)
    from tensorflow.keras.models import load_model
    from cli.evaluate_command import _load_split

    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        from errors import ModelNotFoundError
        raise ModelNotFoundError(path)

    if apply_only:
        cal = load_calibration(CALIBRATION_PATH)
        if cal is None:
            click.secho("⚠ No calibration artifact found — run without --apply-only to fit one.",
                        fg="yellow")
            return None
        click.secho(f"Active calibration: T={cal['temperature']}", fg="cyan")
        click.echo(f"  ECE before → after : {cal.get('ece_before')} → {cal.get('ece_after')}")
        click.echo(f"  Avg conf (correct) : {cal.get('avg_confidence_correct_before')} → "
                   f"{cal.get('avg_confidence_correct_after')}")
        click.echo(f"  Fitted at          : {cal.get('trained_at')} on '{cal.get('fitted_on')}'")
        return cal

    click.secho(f"▶ Loading model {path.name} and scoring '{split}' split...", fg="cyan")
    model = load_model(path)
    data = _load_split(split)
    scores = np.clip(model.predict(data, verbose=0).flatten(), 1e-6, 1 - 1e-6)
    labels = np.array(data.classes, dtype=float)

    temperature = fit_temperature(scores, labels,
                                  grid_lo=TEMPERATURE_BOUNDS[0],
                                  grid_hi=TEMPERATURE_BOUNDS[1])
    report = confidence_report(scores, labels, temperature)

    save_calibration(CALIBRATION_PATH.parent, {
        "temperature": temperature,
        "fitted_on": f"{split}_split",
        "n_samples": int(len(scores)),
        "ece_before": report["before_scaling"]["ece"],
        "ece_after": report["after_scaling"]["ece"],
        "avg_confidence_correct_before": report["before_scaling"]["avg_confidence_correct"],
        "avg_confidence_correct_after": report["after_scaling"]["avg_confidence_correct"],
        "model": str(path),
    })

    b, a = report["before_scaling"], report["after_scaling"]
    click.echo("\n" + "=" * 64)
    click.secho(f"CONFIDENCE CALIBRATION  |  T = {temperature:.4f}", fg="green", bold=True)
    click.echo("=" * 64)
    click.echo(f"{'metric':<34}{'before':>14}{'after':>14}")
    rows = [
        ("Expected Calibration Error", b["ece"], a["ece"]),
        ("Avg confidence (overall)", b["avg_confidence_overall"], a["avg_confidence_overall"]),
        ("Avg confidence (correct)", b["avg_confidence_correct"], a["avg_confidence_correct"]),
        ("Avg confidence (incorrect)", b["avg_confidence_incorrect"], a["avg_confidence_incorrect"]),
        ("% predictions ≥ 90% conf", b["high_conf_ratio_0.9"], a["high_conf_ratio_0.9"]),
    ]
    for name, x, y in rows:
        xs = "-" if x is None else f"{x:.4f}"
        ys = "-" if y is None else f"{y:.4f}"
        click.echo(f"{name:<34}{xs:>14}{ys:>14}")
    click.echo("=" * 64)
    click.secho(f"✓ Saved {CALIBRATION_PATH} (predictions now use calibrated confidence)",
                fg="green")
    return report
