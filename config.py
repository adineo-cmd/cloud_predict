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
MODEL_PATH = MODEL_DIR / "cloud_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_cloud_model.keras"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for directory in [MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# 🖼️ IMAGE & DATA SETTINGS
# ============================================
IMG_SIZE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
BATCH_SIZE = _env_int("CLOUD_BATCH_SIZE", 16)
VALIDATION_SPLIT = _env_float("CLOUD_VAL_SPLIT", 0.2)
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
CLASS_MODE = "binary"

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
BASE_MODEL_NAME = "MobileNetV2"
BASE_MODEL_WEIGHTS = "imagenet"
INCLUDE_TOP = False
BASE_MODEL_TRAINABLE = False

HEAD_LAYERS = [
    {"type": "GlobalAveragePooling2D", "params": {}},
    {"type": "Dense", "params": {"units": 128, "activation": "relu"}},
    {"type": "Dropout", "params": {"rate": 0.5}},
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
METRICS = ["accuracy", "precision", "recall"]

# Two-phase fine-tuning (used by the CLI `train` command)
FINETUNE_ENABLED = os.getenv("CLOUD_FINETUNE", "1") == "1"
FINETUNE_FRACTION_UNFROZEN = 0.10   # top 10% of base layers
FINETUNE_LR_DIVISOR = 10            # fine-tune at lr / 10
LR_REDUCE_ON_PLATEAU = {"factor": 0.5, "patience": 2, "min_lr": 1e-6}
SEED = _env_int("CLOUD_SEED", 42)

# ============================================
# 🛡️ CALLBACKS
# ============================================
EARLY_STOPPING = {
    "enabled": True,
    "monitor": "val_loss",
    "patience": 3,
    "restore_best_weights": True,
    "mode": "min"
}

MODEL_CHECKPOINT = {
    "enabled": True,
    "monitor": "val_accuracy",
    "save_best_only": True,
    "mode": "max",
    "save_weights_only": False,
    "verbose": 1
}

TENSORBOARD = {
    "enabled": True,
    "log_dir": str(LOG_DIR / "tensorboard"),
    "histogram_freq": 0,
    "write_graph": True,
    "write_images": False
}

# ============================================
# 🎯 CONFIDENCE CALIBRATION (temperature scaling)
# ============================================
# Post-training, a scalar temperature T is fitted on the validation split to make
# reported probabilities trustworthy AND sharper: T < 1 increases confidence on
# clear cases. Disable with CLOUD_CALIBRATE=0 to fall back to raw sigmoid scores.
CALIBRATION_ENABLED = os.getenv("CLOUD_CALIBRATE", "1") == "1"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
TEMPERATURE_BOUNDS = (0.3, 3.0)     # search range for the temperature scalar
DEFAULT_TEMPERATURE = _env_float("CLOUD_TEMPERATURE", 1.0)  # fallback when no artifact

# Confidence-boosting ensemble inference defaults (used by predict/CLI/web)
TTA_DEFAULT_RUNS = _env_int("CLOUD_TTA", 5)   # averaging augmented views raises stable confidence
CONFIDENCE_ENSEMBLE = os.getenv("CLOUD_ENSEMBLE", "1") == "1"  # use best + final model average

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