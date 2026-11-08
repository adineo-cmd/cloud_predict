# utils.py
# Shared utilities: reproducible seeding, hardware detection, logging, metrics export,
# seeded dataset splitting.
import os
import json
import hashlib
import random
import logging
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", os.getenv("TF_ENABLE_ONEDNN_OPTS", "0"))

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def _is_dataset_noise(path: Path) -> bool:
    """Model-generated files must never enter a training split (Grad-CAM overlays)."""
    name = path.name.lower()
    return name.endswith("_gradcam.jpg") or name.endswith("_gradcam.jpeg") \
        or name.endswith("_gradcam.png")


def _inventory_hash(dataset_dir: Path) -> str:
    """Stable hash of the dataset inventory — re-split when files change."""
    h = hashlib.sha1()
    for cls_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS and not _is_dataset_noise(f):
                h.update(f"{cls_dir.name}/{f.name}".encode())
    return h.hexdigest()


def stratified_split(dataset_dir, train_frac: float = 0.8, val_frac: float = 0.1,
                     test_frac: float = 0.1, seed: int = 42, cache_path=None,
                     user_photos_per_class: int = 2) -> dict:
    """Deterministic stratified train/val/test split of the dataset folders.

    Returns {"train": {class: [Path]}, "val": {...}, "test": {...}}.

    Two guarantees over the old validation_split=0.2 approach:
      * a real held-out test split exists (never used for training, checkpointing,
        threshold search or calibration);
      * a few of the user's own camera photos (files without the 'ccsn_' prefix)
        are reserved for val+test when available, so held-out metrics reflect the
        deployment domain rather than only the downloaded data distribution.

    The result is cached (keyed by seed, fractions and dataset inventory) so
    train / evaluate / calibrate always see identical splits.
    """
    dataset_dir = Path(dataset_dir)
    classes = ["cumulonimbus", "other"]
    key = {
        "seed": seed, "train": train_frac, "val": val_frac, "test": test_frac,
        "inventory": _inventory_hash(dataset_dir),
    }
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                return {
                    split: {cls: [Path(p) for p in files]
                            for cls, files in cached["splits"][split].items()}
                    for split in ("train", "val", "test")
                }
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # corrupt/legacy cache — recompute below

    rng = random.Random(seed)
    splits = {s: {c: [] for c in classes} for s in ("train", "val", "test")}

    for cls in classes:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        user_files = sorted(f for f in cls_dir.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS
                            and not _is_dataset_noise(f)
                            and not f.name.lower().startswith("ccsn_"))
        ccsn_files = sorted(f for f in cls_dir.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS
                            and not _is_dataset_noise(f)
                            and f.name.lower().startswith("ccsn_"))
        rng.shuffle(user_files)
        rng.shuffle(ccsn_files)

        n_total = len(user_files) + len(ccsn_files)
        n_val = max(1, round(val_frac * n_total))
        n_test = max(1, round(test_frac * n_total))

        # Reserve user photos for val/test first (they are the scarce, deployment-
        # domain images), then fill the remaining slots from the downloaded pool.
        take_val_user = min(user_photos_per_class, len(user_files), n_val)
        take_test_user = min(user_photos_per_class, len(user_files) - take_val_user, n_test)
        splits["val"][cls] += user_files[:take_val_user]
        splits["test"][cls] += user_files[take_val_user:take_val_user + take_test_user]
        train_user = user_files[take_val_user + take_test_user:]

        fill_val = n_val - take_val_user
        fill_test = n_test - take_test_user
        splits["val"][cls] += ccsn_files[:fill_val]
        splits["test"][cls] += ccsn_files[fill_val:fill_val + fill_test]
        # Everything not reserved for val/test goes to train (user photos + pool).
        splits["train"][cls] += train_user + ccsn_files[fill_val + fill_test:]

    for split in splits.values():
        for cls in classes:
            split[cls].sort(key=lambda p: str(p))

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        serial = {s: {c: [str(p) for p in files] for c, files in pairs.items()}
                  for s, pairs in splits.items()}
        cache_file.write_text(json.dumps({"key": key, "splits": serial}, indent=2))

    sizes = {s: sum(len(v) for v in splits[s].values()) for s in splits}
    logging.getLogger(__name__).info(f"Dataset split sizes: {sizes}")
    return splits
