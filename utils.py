# utils.py
# Shared utilities: reproducible seeding, hardware detection, logging, metrics export
import os
import random
import logging

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", os.getenv("TF_ENABLE_ONEDNN_OPTS", "0"))

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Make training reproducible across Python, NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # TF not installed / not needed for this entry point


def configure_tf_runtime(use_gpu: bool = False, allow_growth: bool = True) -> str:
    """Configure GPU visibility / CPU thread settings. Returns active device mode."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if use_gpu and gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, allow_growth)
            except RuntimeError:
                pass  # already initialized
        tf.config.set_visible_devices(gpus, "GPU")
        return f"GPU ({gpus[0].name})"

    tf.config.set_visible_devices([], "GPU")
    cpu_count = os.cpu_count() or 4
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, cpu_count // 2))
    except RuntimeError:
        pass
    return "CPU"


def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Create a logger with console output and optional rotating file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_metrics_csv(history, csv_path) -> str:
    """Persist a Keras History object to CSV for downstream analysis/reporting."""
    import pandas as pd
    from pathlib import Path

    df = pd.DataFrame(history.history)
    df.index.name = "epoch"
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    return str(csv_path)


def get_git_version() -> str:
    """Best-effort semantic version tag for model artifacts (falls back to timestamp)."""
    try:
        import subprocess
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return tag or "dev"
    except Exception:
        return "dev"
