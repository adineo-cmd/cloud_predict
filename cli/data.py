# cli/data.py
# Shared data plumbing: the seeded stratified split turned into Keras generators,
# plus backbone-correct preprocessing (shared by train / evaluate / calibrate so
# every split definition and pixel pipeline is identical everywhere).
import numpy as np

from utils import stratified_split


def get_preprocess_input(backbone: str = None):
    """Return the preprocessing function matching the configured backbone.

    EfficientNetV2 normalizes inside the graph → pass-through ([0, 255] floats);
    MobileNetV2 needs its preprocess_input (scales to [-1, 1]).
    """
    from config import BACKBONE, BACKBONES

    name = backbone or BACKBONE
    entry = BACKBONES[name]
    if entry["preprocess"] == "identity":
        def preprocess_input(x):
            x = np.asarray(x, dtype="float32")
            return x
        return preprocess_input
    from tensorflow.keras.applications import mobilenet_v2
    return mobilenet_v2.preprocess_input


def _split_dataframe(split_files: dict):
    """Split dict {class: [paths]} → pandas DataFrame(filepath, class)."""
    import pandas as pd

    rows = [(str(p), cls) for cls, files in split_files.items() for p in files]
    return pd.DataFrame(rows, columns=["filepath", "class"])


def load_split(split: str = "train", batch_size: int = None, augment: bool = False,
               seed: int = None, split_cache_path=None):
    """Build a generator over a named split ('train' | 'val' | 'test').

    Uses the cached seeded stratified split so train/evaluate/calibrate all see
    exactly the same files. Augmentation is applied to the train split only.
    """
    from config import (DATASET_DIR, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
                        BATCH_SIZE, CLASS_MODE, AUGMENTATION_CONFIG, CLASS_INDICES,
                        SPLIT_CACHE_PATH, IMG_SIZE, SEED)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    split = {"validation": "val"}.get(split, split)
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split!r}")

    files = stratified_split(
        DATASET_DIR, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
        seed=SEED, cache_path=split_cache_path or SPLIT_CACHE_PATH,
    )
    df = _split_dataframe(files[split])
    if df.empty:
        raise ValueError(f"Split {split!r} is empty — run training/data download first.")

    preprocess_input = get_preprocess_input()
    batch_size = batch_size or BATCH_SIZE
    target_size = (IMG_SIZE, IMG_SIZE)

    if augment:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     **AUGMENTATION_CONFIG)
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    gen = datagen.flow_from_dataframe(
        dataframe=df, directory=None, x_col="filepath", y_col="class",
        target_size=target_size, batch_size=batch_size, class_mode=CLASS_MODE,
        shuffle=(augment and split == "train"), seed=seed,
    )
    if gen.class_indices != CLASS_INDICES:
        raise RuntimeError(
            f"Class index mismatch: generator {gen.class_indices} vs config {CLASS_INDICES}")
    return gen


def split_sizes():
    """Per-class file counts of every split (for logging / manifests)."""
    from config import (DATASET_DIR, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
                        SEED, SPLIT_CACHE_PATH)

    files = stratified_split(
        DATASET_DIR, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
        seed=SEED, cache_path=SPLIT_CACHE_PATH,
    )
    return {s: {c: len(v) for c, v in pairs.items()} for s, pairs in files.items()}
