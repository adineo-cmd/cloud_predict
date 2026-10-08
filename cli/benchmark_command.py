# cli/benchmark_command.py
# Latency/throughput benchmark for single-image inference — useful for deployment sizing.
import time
from pathlib import Path

import click
import numpy as np

from errors import ModelNotFoundError


def run_benchmark(runs=10, warmup=3, model_path=None):
    from config import MODEL_PATH, INPUT_SHAPE
    from tensorflow.keras.models import load_model

    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(path)

    click.secho(f"▶ Loading {path.name} ...", fg="cyan")
    t_load = time.perf_counter()
    model = load_model(path)
    load_time = time.perf_counter() - t_load

    dummy = np.random.uniform(-1, 1, size=(1,) + INPUT_SHAPE).astype("float32")

    for _ in range(warmup):
        model.predict(dummy, verbose=0)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000  # ms
    click.echo("\n" + "=" * 50)
    click.echo("INFERENCE BENCHMARK")
    click.echo("=" * 50)
    click.echo(f"  Model load time : {load_time:.2f} s")
    click.echo(f"  Runs (batch=1)  : {runs} (+{warmup} warmup)")
    click.echo(f"  Latency mean    : {lat.mean():.1f} ms")
    click.echo(f"  Latency p50     : {np.percentile(lat, 50):.1f} ms")
    click.echo(f"  Latency p95     : {np.percentile(lat, 95):.1f} ms")
    click.echo(f"  Throughput      : {1000 / lat.mean():.1f} img/s")
    click.echo("=" * 50)
