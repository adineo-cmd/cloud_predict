# config.py
# Centralized configuration for Cumulonimbus Cloud Classification Project
# Supports environment-variable overrides (12-factor style), e.g.:
#   CLOUD_EPOCHS=30 CLOUD_BATCH_SIZE=32 python -m cli train
import os
from pathlib import Path


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ============================================
# 📁 PATHS & DIRECTORIES
# ============================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("CLOUD_DATASET_DIR", str(BASE_DIR / "dataset")))
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cloud_model.keras"   # best ensemble member (backward-compat copy)
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Ensure directories exist
for directory in [MODEL_DIR, LOG_DIR, RESULTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# 🖼️ IMAGE & DATA SETTINGS
# ============================================
IMG_SIZE = _env_int("CLOUD_IMG_SIZE", 256)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
BATCH_SIZE = _env_int("CLOUD_BATCH_SIZE", 32)
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
CLASS_MODE = "binary"

# Seeded stratified train/val/test split (replaces ImageDataGenerator's
# positional validation_split, which had no test set at all).
TRAIN_FRACTION = _env_float("CLOUD_TRAIN_FRACTION", 0.8)
VAL_FRACTION = _env_float("CLOUD_VAL_FRACTION", 0.1)
TEST_FRACTION = _env_float("CLOUD_TEST_FRACTION", 0.1)
SPLIT_CACHE_PATH = RESULTS_DIR / "dataset_split.json"

# CRITICAL: Class indices (alphabetical order from flow_from_directory)
# 'cumulonimbus' < 'other' alphabetically, so:
CLASS_INDICES = {"cumulonimbus": 0, "other": 1}

# ============================================
# 🎨 DATA AUGMENTATION (Training Only)
# ============================================
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": False,
    "fill_mode": "nearest",
    "brightness_range": [0.8, 1.2]
}

# ============================================
# 🧠 MODEL ARCHITECTURE
# ============================================
# Backbone registry: name -> (tf.keras.applications class, preprocessing).
# EfficientNetV2 models normalize internally (preprocess_input is a pass-through),
# so images stay in [0, 255]; MobileNetV2 expects its preprocess_input ([-1, 1]).
BACKBONE = os.getenv("CLOUD_BACKBONE", "efficientnet_v2_b0")
BACKBONES = {
    "efficientnet_v2_b0": {"ctor": "EfficientNetV2B0", "preprocess": "identity"},
    "mobilenet_v2": {"ctor": "MobileNetV2", "preprocess": "mobilenet_v2"},
}
BASE_MODEL_WEIGHTS = "imagenet"
INCLUDE_TOP = False
BASE_MODEL_TRAINABLE = False

HEAD_LAYERS = [
    {"type": "GlobalAveragePooling2D", "params": {}},
    {"type": "Dropout", "params": {"rate": 0.3}},
    {"type": "Dense", "params": {"units": 256, "activation": "relu"}},
    {"type": "Dropout", "params": {"rate": 0.4}},
]

OUTPUT_ACTIVATION = "sigmoid"
OUTPUT_UNITS = 1

# ============================================
# ⚙️ TRAINING HYPERPARAMETERS
# ============================================
OPTIMIZER = "adam"
LEARNING_RATE = _env_float("CLOUD_LR", 1e-3)
EPOCHS = _env_int("CLOUD_EPOCHS", 15)
LOSS_FUNCTION = "binary_crossentropy"
LABEL_SMOOTHING = _env_float("CLOUD_LABEL_SMOOTHING", 0.05)
METRICS = ["accuracy", "precision", "recall"]

# Two-phase fine-tuning (used by the CLI `train` command)
FINETUNE_ENABLED = os.getenv("CLOUD_FINETUNE", "1") == "1"
FINETUNE_FRACTION_UNFROZEN = 0.35   # top 35% of base layers (BatchNorm stays frozen)
FINETUNE_LR_DIVISOR = 10            # fine-tune at lr / 10
SEED = _env_int("CLOUD_SEED", 42)

# ============================================
# 🧬 SEED ENSEMBLE
# ============================================
# Real ensemble: N members trained with different seeds (head init + batch order).
# Their probabilities are averaged at inference — a genuine variance reduction,
# unlike the previous setup that averaged one model with its own checkpoint.
ENSEMBLE_SIZE = _env_int("CLOUD_ENSEMBLE_SIZE", 3)
ENSEMBLE_SEED_STRIDE = _env_int("CLOUD_ENSEMBLE_SEED_STRIDE", 1000)
ENSEMBLE_DIR = Path(os.getenv("CLOUD_ENSEMBLE_DIR", str(MODEL_DIR / "ensemble")))
ENSEMBLE_MANIFEST_PATH = ENSEMBLE_DIR / "manifest.json"

# ============================================
# 🛡️ CALLBACKS
# ============================================
EARLY_STOPPING = {
    "enabled": True,
    "monitor": "val_loss",
    "patience": 5,
    "restore_best_weights": True,
    "mode": "min"
}

TENSORBOARD = {
    "enabled": True,
    "log_dir": str(LOG_DIR / "tensorboard"),
    "histogram_freq": 0,
    "write_graph": True,
    "write_images": False
}

# ============================================
# 🎯 CONFIDENCE CALIBRATION (Platt scaling / temperature)
# ============================================
# Post-training, a 2-parameter Platt map (or single temperature T) is fitted on
# the VALIDATION split only — never on training data (that sharpens confidence
# spuriously) and never on the test split (that would leak evaluation).
# Inference rejects artifacts fitted on other splits, falling back to raw
# scores instead of distorted confidence.
CALIBRATION_ENABLED = os.getenv("CLOUD_CALIBRATE", "1") == "1"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
CALIBRATION_TRUSTED_FIT = "validation_split"
CALIBRATION_MIN_SAMPLES = 16
TEMPERATURE_BOUNDS = (0.3, 3.0)     # search range for the temperature scalar
DEFAULT_TEMPERATURE = _env_float("CLOUD_TEMPERATURE", 1.0)  # fallback when no artifact

# Inference defaults (used by predict/CLI/web)
TTA_DEFAULT_RUNS = _env_int("CLOUD_TTA", 5)   # averaging augmented views stabilizes predictions
# Kept as a switch for loading the seed ensemble; a single --model path overrides it.
CONFIDENCE_ENSEMBLE = os.getenv("CLOUD_ENSEMBLE", "1") == "1"

# ============================================
# 🔍 PREDICTION SETTINGS (FIXED)
# ============================================
PREDICTION_THRESHOLD = _env_float("CLOUD_THRESHOLD", 0.5)

# Uncertainty band: predictions with confidence below this are flagged as uncertain
UNCERTAINTY_LOW_CONF = 0.70

# FIXED: Match training class indices exactly
# score > 0.5 → class 1 → "other"
# score < 0.5 → class 0 → "cumulonimbus"
PREDICTION_LABELS = {
    0: "Cumulonimbus Cloud ⛈️",
    1: "Other Cloud ☁️"
}

SHOW_CONFIDENCE = True
CONFIDENCE_DECIMALS = 2

# ============================================
# 🧪 EVALUATION / REPORTING
# ============================================
PLOT_CONFUSION_MATRIX = True
EVAL_THRESHOLDS_GRID = 91          # points searched in optimal-threshold sweep

# ============================================
# 💻 HARDWARE
# ============================================
USE_GPU = os.getenv("CLOUD_USE_GPU", "0") == "1"
TF_CPP_MIN_LOG_LEVEL = '2'
TF_ENABLE_ONEDNN_OPTS = '0'

# ============================================
# 🧹 LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_TO_FILE = True
LOG_FILE_PATH = LOG_DIR / "training.log"
