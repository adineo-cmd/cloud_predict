#!/usr/bin/env python3
"""
run.py — One-command pipeline for the Cumulonimbus Cloud Classifier.

You do NOT need to run each script separately. This single entry point
orchestrates the whole workflow end-to-end:

    1. setup       Install / verify dependencies (optional, --install)
    2. train       Train the MobileNetV2 transfer-learning model
    3. calibrate   Fit temperature scaling for trustworthy confidence
    4. evaluate    Precision / Recall / F1, confusion matrix, ROC/PR
    5. benchmark   Inference latency (p50/p95) and throughput
    6. demo        Predict every sample image with Grad-CAM overlays
    7. web         Launch the Flask web UI (default final step)

Usage examples:
    python run.py                 # full pipeline: train → calibrate → evaluate → web UI
    python run.py quick           # skip training if a model already exists, then web UI
    python run.py train           # just train
    python run.py predict img.jpg --gradcam
    python run.py web --port 8080
    python run.py all --epochs 20 # explicit full run with custom epochs

Every subcommand simply forwards to the unified CLI (python -m cli ...),
so nothing is duplicated — run.py is only a convenience orchestrator.
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cloud_model.keras"


def _cli(args, abort_on_error: bool = True):
    """Run a CLI command in-process (no extra Python startup cost)."""
    from cli.main import cli
    print(f"\n\033[1;36m▶ cloud-classifier {' '.join(args)}\033[0m")
    print("-" * 60)
    try:
        cli(args=args, standalone_mode=False)
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:  # noqa: BLE001
        print(f"\033[1;31m✗ Command failed: {e}\033[0m")
        if abort_on_error:
            sys.exit(1)
        return 1


def _deps_ok() -> bool:
    try:
        import tensorflow  # noqa: F401
        import flask  # noqa: F401
        import click  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError as e:
        print(f"\033[1;33mMissing dependency: {e.name}\033[0m")
        return False


def _install_deps():
    import subprocess
    req = BASE_DIR / "requirements.txt"
    print("\033[1;36m▶ Installing dependencies from requirements.txt ...\033[0m")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def cmd_setup(**_):
    if _deps_ok():
        print("\033[1;32m✓ All required packages are available.\033[0m")
    else:
        _install_deps()
        print("\033[1;32m✓ Environment ready.\033[0m")


def cmd_train(argv):
    return _cli(["train"] + argv)


def cmd_calibrate(argv):
    return _cli(["calibrate"] + argv)


def cmd_evaluate(argv):
    return _cli(["evaluate"] + argv)


def cmd_benchmark(argv):
    return _cli(["benchmark"] + argv)


def cmd_predict(argv):
    return _cli(["predict"] + argv)


def cmd_history(argv):
    return _cli(["history"] + argv)


def cmd_web(argv):
    import subprocess
    port, host = 5000, "127.0.0.1"
    i = 0
    while i < len(argv):
        if argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1]); i += 2
        elif argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]; i += 2
        else:
            i += 1
    if not (BASE_DIR / "app.py").exists():
        return _fail("app.py not found")
    print(f"\033[1;36m▶ Starting web UI at http://{host}:{port}  (Ctrl+C to stop)\033[0m")
    return subprocess.call([sys.executable, str(BASE_DIR / "app.py"),
                            "--host", host, "--port", str(port)])


def _fail(msg):
    print(f"\033[1;31m✗ {msg}\033[0m")
    return 1


def cmd_demo(argv):
    """Predict a handful of dataset samples so you can see results instantly."""
    samples = sorted((BASE_DIR / "dataset" / "cumulonimbus").glob("*.jpg"))[:2]
    samples += sorted((BASE_DIR / "dataset" / "other").glob("*.jpg"))[:2]
    if not samples:
        return _fail("No sample images found under dataset/.")
    return _cli(["predict", *[str(p) for p in samples], "--gradcam", "--save-vis"])


def cmd_all(argv):
    """Full one-shot pipeline: setup → train → calibrate → evaluate → benchmark → demo → web."""
    t0 = time.time()
    extras = list(argv)

    cmd_setup()
    if not MODEL_PATH.exists() or "--force-train" in extras:
        extras = [a for a in extras if a != "--force-train"]
        cmd_train(extras)
    else:
        print("\033[1;33m• Model already exists — skipping training "
              "(use 'python run.py train' or --force-train to retrain).\033[0m")
    cmd_calibrate([])
    cmd_evaluate([])
    cmd_benchmark([])
    cmd_demo([])
    print(f"\n\033[1;32m✓ Pipeline complete in {time.time() - t0:.1f}s\033[0m")
    cmd_web([])


def cmd_quick(argv):
    """Fastest path to the web UI: reuse existing model, otherwise run full pipeline."""
    if MODEL_PATH.exists():
        cmd_setup()
        cmd_demo([])
        cmd_web(argv)
    else:
        cmd_all(argv)


HELP = """\033[1mCumulonimbus Cloud Classifier — one-command runner\033[0m

Usage: python run.py [command] [options]

Commands:
  (no args)     Same as 'all': full pipeline, ends with the web UI
  all           setup → train → calibrate → evaluate → benchmark → demo → web
  quick         Reuse existing model and jump straight to demo + web UI
  setup         Verify/install dependencies
  train [...]   Train the model (forwards flags to 'python -m cli train')
  calibrate     Fit temperature-scaled confidence calibration
  evaluate      Full metrics report + ROC/PR plots
  benchmark     Latency & throughput measurement
  predict ...   Predict images (supports --gradcam, --json, --tta N, ...)
  history       Show previous training-run summary & curves
  demo          Predict 4 sample images with Grad-CAM visualizations
  web [--port]  Launch the Flask web interface

Examples:
  python run.py                       # everything in one go
  python run.py quick                 # model exists → straight to the UI
  python run.py train --epochs 30 --finetune
  python run.py predict photo.jpg --gradcam --json
"""


COMMANDS = {
    "all": cmd_all, "quick": cmd_quick, "setup": cmd_setup, "train": cmd_train,
    "calibrate": cmd_calibrate, "evaluate": cmd_evaluate, "benchmark": cmd_benchmark,
    "predict": cmd_predict, "history": cmd_history, "demo": cmd_demo, "web": cmd_web,
}


def main():
    if not _deps_ok():
        _install_deps()

    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        if not argv:
            return cmd_all([])
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(HELP)
        return _fail(f"Unknown command: {cmd!r}")
    return COMMANDS[cmd](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
