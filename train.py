# train.py
import os
import logging
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = os.getenv('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from config import *

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH) if LOG_TO_FILE else logging.NullHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_dataset_info(train_data, val_data):
    logger.info("\n" + "="*60)
    logger.info("📊 DATASET ANALYSIS")
    logger.info("="*60)
    logger.info(f"Class indices: {train_data.class_indices}")
    
    for name, label in [("Training", train_data), ("Validation", val_data)]:
        logger.info(f"\n{name} samples per class:")
        for class_name, idx in label.class_indices.items():
            count = np.sum(np.array(label.classes) == idx)
            pct = count / len(label.classes) * 100
            logger.info(f"  {class_name}: {count} images ({pct:.1f}%)")
    logger.info("="*60 + "\n")


def prepare_data():
    logger.info("Preparing data generators...")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT,
        **AUGMENTATION_CONFIG
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT
    )
    
    train_data = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="training",
        shuffle=SHUFFLE_TRAIN
    )
    
    val_data = val_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation",
        shuffle=SHUFFLE_VAL
    )
    
    print_dataset_info(train_data, val_data)
    return train_data, val_data


def build_model():
    logger.info(f"Building {BASE_MODEL_NAME} model...")
    
    base_model = MobileNetV2(
        weights=BASE_MODEL_WEIGHTS,
        include_top=INCLUDE_TOP,
        input_shape=INPUT_SHAPE
    )
    
    for layer in base_model.layers:
        layer.trainable = BASE_MODEL_TRAINABLE
    
    x = base_model.output
    for layer_config in HEAD_LAYERS:
        layer_type = getattr(tf.keras.layers, layer_config["type"])
        x = layer_type(**layer_config["params"])(x)
    
    predictions = Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    
    logger.info(f"Model built with {model.count_params():,} parameters")
    return model


def compute_class_weights(train_data):
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    return dict(enumerate(class_weights))


def setup_callbacks():
    callbacks = []
    
    if EARLY_STOPPING["enabled"]:
        es_config = {k: v for k, v in EARLY_STOPPING.items() if k != "enabled"}
        callbacks.append(EarlyStopping(**es_config))
        logger.info("✓ EarlyStopping enabled")
    
    if MODEL_CHECKPOINT["enabled"]:
        mc_config = {k: v for k, v in MODEL_CHECKPOINT.items() if k != "enabled"}
        callbacks.append(ModelCheckpoint(str(BEST_MODEL_PATH), **mc_config))
        logger.info(f"✓ ModelCheckpoint enabled: {BEST_MODEL_PATH}")
    
    if TENSORBOARD["enabled"]:
        tb_config = {k: v for k, v in TENSORBOARD.items() if k != "enabled"}
        callbacks.append(TensorBoard(**tb_config))
        logger.info(f"✓ TensorBoard logging: {TENSORBOARD['log_dir']}")
    
    return callbacks


def train():
    logger.info("🚀 Starting training pipeline...")
    
    train_data, val_data = prepare_data()
    model = build_model()
    class_weights = compute_class_weights(train_data)
    logger.info(f"Using class weights: {class_weights}")
    
    callbacks = setup_callbacks()
    
    logger.info(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    model.save(MODEL_PATH)
    logger.info(f"✅ Training complete! Model saved to {MODEL_PATH}")
    
    val_metrics = model.evaluate(val_data, verbose=0)
    logger.info(f"\n📈 Final Validation Metrics:")
    for metric_name, value in zip(model.metrics_names, val_metrics):
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return history


if __name__ == "__main__":
    train()