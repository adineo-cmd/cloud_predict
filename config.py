# config.py
# Centralized configuration for Cumulonimbus Cloud Classification Project
import os
from pathlib import Path

# ============================================
# 📁 PATHS & DIRECTORIES
# ============================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
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
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
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
LEARNING_RATE = 1e-3
EPOCHS = 15
LOSS_FUNCTION = "binary_crossentropy"
METRICS = ["accuracy", "precision", "recall"]

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
# 🔍 PREDICTION SETTINGS (FIXED)
# ============================================
PREDICTION_THRESHOLD = 0.5

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
# 💻 HARDWARE
# ============================================
USE_GPU = False
TF_CPP_MIN_LOG_LEVEL = '2'
TF_ENABLE_ONEDNN_OPTS = '0'

# ============================================
# 🧹 LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_TO_FILE = True
LOG_FILE_PATH = LOG_DIR / "training.log"