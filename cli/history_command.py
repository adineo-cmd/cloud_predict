# cli/history_command.py
# Post-training analysis: read the metrics CSV, print a summary table, plot curves.
from pathlib import Path

import click


def run_history(csv_file=None, plot=True):
    import pandas as pd
    from config import RESULTS_DIR

    path = Path(csv_file) if csv_file else RESULTS_DIR / "training_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"No history CSV at {path}. Train first or pass --csv-file.")

    df = pd.read_csv(path, index_col="epoch")
    epochs_run = len(df)

    click.secho(f"▶ Training history: {path} ({epochs_run} epochs)", fg="cyan")

    # Summary table with rich if available, else plain
    key_cols = [c for c in ["loss", "accuracy", "precision", "recall",
                            "val_loss", "val_accuracy", "val_precision", "val_recall",
                            "lr"] if c in df.columns]
    summary = df[key_cols].describe().loc[["min", "max", "mean"]].round(4)

    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title="Metrics Summary")
        table.add_column("metric", style="cyan")
        for stat in ["min", "max", "mean"]:
            table.add_column(stat, justify="right")
        for col in key_cols:
            row = [col] + [f"{summary.loc[stat, col]:.4f}" for stat in ["min", "max", "mean"]]
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        click.echo(summary.to_string())

    best_epoch = df["val_accuracy"].idxmax() if "val_accuracy" in df else None
    if best_epoch is not None:
        click.secho(f"✓ Best val_accuracy {df['val_accuracy'][best_epoch]:.4f} at epoch {best_epoch}",
                    fg="green", bold=True)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import MODEL_PATH  # noqa (kept for parity of artifacts dir)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if {"accuracy", "val_accuracy"} <= set(df.columns):
            axes[0].plot(df["accuracy"], label="train acc")
            axes[0].plot(df["val_accuracy"], label="val acc")
            axes[0].set(title="Accuracy", xlabel="epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
        if {"loss", "val_loss"} <= set(df.columns):
            axes[1].plot(df["loss"], label="train loss")
            axes[1].plot(df["val_loss"], label="val loss")
            axes[1].set(title="Loss", xlabel="epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        out = RESULTS_DIR / "training_curves.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        click.echo(f"✓ Saved {out}")
