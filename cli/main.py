# cli/main.py
# Unified professional CLI: python -m cli <command> [options]
# Commands: train | predict | evaluate | history | benchmark
import sys
import logging

import click

from cli import __version__
from errors import CloudClassifierError

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def _configure_logging(level: str, log_file: str = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name="cloud-classifier")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Console/CLI logging verbosity.")
def cli(log_level):
    """Cumulonimbus Cloud Classifier — transfer-learning toolkit.

    \b
    Quick start:
      python -m cli train --epochs 15
      python -m cli predict path/to/image.jpg --json
      python -m cli evaluate --threshold 0.5
      python -m cli history
      python -m cli benchmark
    """
    from config import LOG_FILE_PATH
    _configure_logging(log_level, str(LOG_FILE_PATH))


@cli.command()
@click.option("--epochs", type=int, default=None, help="Override training epochs.")
@click.option("--batch-size", type=int, default=None, help="Override batch size.")
@click.option("--lr", "learning_rate", type=float, default=None, help="Override learning rate.")
@click.option("--finetune/--no-finetune", default=None,
              help="Two-phase training: frozen head, then unfreeze top base layers.")
@click.option("--seed", type=int, default=42, show_default=True, help="Global RNG seed.")
@click.option("--device", type=click.Choice(["cpu", "gpu"]), default=None,
              help="Override compute device.")
def train(epochs, batch_size, lr, finetune, seed, device):
    """Train the MobileNetV2 transfer-learning model on the dataset."""
    from cli.train_command import run_train

    run_train(
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
        finetune=finetune, seed=seed, device=device,
    )


@cli.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=False))
@click.option("--model", "model_path", default=None, help="Path to a .keras model file.")
@click.option("--threshold", type=float, default=None, help="Decision threshold (0-1).")
@click.option("--tta", "tta_runs", type=int, default=None,
              help="Test-time augmentation views (default: CLOUD_TTA or 5; averaging "
                   "diverse views increases stable confidence).")
@click.option("--calibrate/--no-calibrate", default=None,
              help="Apply temperature-scaled calibrated confidence (default: on).")
@click.option("--gradcam/--no-gradcam", default=False, help="Overlay Grad-CAM explanation image.")
@click.option("--save-vis", is_flag=True, help="Save annotated visualization alongside input.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON result.")
@click.option("--debug", is_flag=True, help="Show raw scores and threshold analysis.")
def predict(images, model_path, threshold, tta_runs, gradcam, save_vis, as_json, debug,
            calibrate):
    """Predict one or more images (supports glob patterns and directories).

    \b
    Confidence is boosted the honest way: multi-view TTA + final/best model
    ensemble + temperature-scaled calibration fitted on validation data.
    """
    from cli.predict_command import run_predict

    exit_code = run_predict(
        images=list(images), model_path=model_path, threshold=threshold,
        tta_runs=tta_runs, gradcam=gradcam, save_vis=save_vis,
        as_json=as_json, debug=debug, calibrate=calibrate,
    )
    sys.exit(exit_code)


@cli.command()
@click.option("--model", "model_path", default=None, help="Path to a .keras model file.")
@click.option("--threshold", type=float, default=None, help="Decision threshold (0-1).")
@click.option("--optimal-threshold/--no-optimal-threshold", default=True,
              help="Search thresholds and report the best F1 operating point.")
@click.option("--roc/--no-roc", default=True, help="Plot ROC / PR curves.")
@click.option("--split", default="validation", show_default=True,
              type=click.Choice(["validation", "training"]), help="Dataset split to score.")
def evaluate(model_path, threshold, optimal_threshold, roc, split):
    """Evaluate the model: precision/recall/F1, confusion matrix, ROC/PR curves."""
    from cli.evaluate_command import run_evaluate

    run_evaluate(
        model_path=model_path, threshold=threshold,
        find_optimal=optimal_threshold, plot_roc=roc, split=split,
    )


@cli.command()
@click.option("--model", "model_path", default=None, help="Path to a .keras model file.")
@click.option("--split", default="validation", show_default=True,
              type=click.Choice(["validation", "training"]), help="Split to fit temperature on.")
@click.option("--apply-only", is_flag=True,
              help="Just print the currently active calibration artifact.")
def calibrate(model_path, split, apply_only):
    """Fit confidence calibration (temperature scaling) for higher trustworthy confidence."""
    from cli.calibrate_command import run_calibrate

    run_calibrate(model_path=model_path, split=split, apply_only=apply_only)


@cli.command()
@click.option("--csv-file", default=None, help="History CSV produced by training.")
@click.option("--plot/--no-plot", default=True, help="Render accuracy/loss curves.")
def history(csv_file, plot):
    """Summarize a previous training run from its metrics CSV."""
    from cli.history_command import run_history

    run_history(csv_file=csv_file, plot=plot)


@cli.command()
@click.option("--runs", type=int, default=10, show_default=True, help="Timed inference runs.")
@click.option("--warmup", type=int, default=3, show_default=True, help="Warm-up iterations.")
@click.option("--model", "model_path", default=None, help="Path to a .keras model file.")
def benchmark(runs, warmup, model_path):
    """Measure single-image inference latency (p50/p95) and throughput."""
    from cli.benchmark_command import run_benchmark

    run_benchmark(runs=runs, warmup=warmup, model_path=model_path)


def main():
    try:
        cli(standalone_mode=False)
    except CloudClassifierError as e:
        click.secho(f"✗ {e}", fg="red", bold=True)
        sys.exit(e.exit_code)
    except click.exceptions.UsageError as e:
        e.show()
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        click.secho("\nInterrupted.", fg="yellow")
        sys.exit(130)


if __name__ == "__main__":
    main()
