# cloud.py
# Cumulonimbus Cloud Classification - Complete Fixed Version
# All bugs fixed: label mapping, prediction logic, confidence calculation

import os
import logging
import numpy as np
import cv2
from pathlib import Path

# Suppress TensorFlow warnings BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# ⚙️ CONFIGURATION
# ============================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Paths
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cloud_model.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_cloud_model.keras")
LOG_DIR = "logs"

# Ensure directories exist
for directory in [MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Class mapping (alphabetical order from flow_from_directory)
# 'cumulonimbus' < 'other' alphabetically
CLASS_INDICES = {"cumulonimbus": 0, "other": 1}
PREDICTION_LABELS = {0: "Cumulonimbus Cloud ⛈️", 1: "Other Cloud ☁️"}
PREDICTION_THRESHOLD = 0.5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# 1. DATA PREPARATION
# ============================================
logger.info("Loading dataset...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Print dataset analysis
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

# ============================================
# 2. MODEL ARCHITECTURE
# ============================================
logger.info("Loading pretrained model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ============================================
# 3. COMPILATION WITH BETTER METRICS
# ============================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(name='precision'), Recall(name='recall')]
)

logger.info(f"Model built with {model.count_params():,} parameters")

# ============================================
# 4. CALLBACKS
# ============================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=os.path.join(LOG_DIR, "tensorboard"),
    histogram_freq=0,
    write_graph=True,
    write_images=False
)

# ============================================
# 5. CLASS WEIGHTS (Handle Imbalance)
# ============================================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
logger.info(f"Using class weights: {class_weight_dict}")

# ============================================
# 6. TRAINING
# ============================================
logger.info("Training model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, tensorboard],
    class_weight=class_weight_dict,
    verbose=1
)

# Save the final model (modern .keras format)
model.save(MODEL_PATH)
logger.info(f"✅ Model training complete! Saved to {MODEL_PATH}")

# Print final validation metrics
val_metrics = model.evaluate(val_data, verbose=0)
logger.info("\n📈 Final Validation Metrics:")
for metric_name, value in zip(model.metrics_names, val_metrics):
    logger.info(f"  {metric_name}: {value:.4f}")

# ============================================
# 7. TESTING WITH SAMPLE IMAGE
# ============================================
logger.info("\nTesting with sample image...")
test_image = "test.jpg"

if os.path.exists(test_image):
    # Load image
    img = cv2.imread(test_image)
    
    if img is None:
        logger.error(f"❌ Failed to load image: {test_image}")
    else:
        # CRITICAL FIX: Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Use the same preprocess_input function used during training
        img = preprocess_input(img)
        
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(img, verbose=0)
        score = float(prediction[0][0])  # Extract float value

        # FIXED: Correct prediction logic based on class indices
        # score > 0.5 → class 1 → "other"
        # score < 0.5 → class 0 → "cumulonimbus"
        predicted_class = 1 if score > PREDICTION_THRESHOLD else 0
        label = PREDICTION_LABELS[predicted_class]
        
        # Calculate confidence for the predicted class
        confidence = score if predicted_class == 1 else (1 - score)

        print("\n" + "="*60)
        print(f"🖼️  Image: {test_image}")
        print(f"📊 Confidence: {confidence:.2f} ({confidence*100:.2f}%)")
        print(f"🎯 Prediction: {label}")
        print(f"⚙️  Threshold: {PREDICTION_THRESHOLD}")
        print(f"📈 Raw Score: {score:.4f}")
        print("\n🔍 Class Mapping:")
        print(f"  Class 0: Cumulonimbus Cloud ⛈️")
        print(f"  Class 1: Other Cloud ☁️")
        print("="*60 + "\n")
else:
    logger.warning(f"⚠️  Add a test image named {test_image} to test prediction.")

# ============================================
# 8. PRINT TRAINING SUMMARY
# ============================================
print("\n" + "="*60)
print("📋 TRAINING SUMMARY")
print("="*60)
print(f"✅ Model saved: {MODEL_PATH}")
print(f"✅ Best model saved: {BEST_MODEL_PATH}")
print(f"✅ TensorBoard logs: {os.path.join(LOG_DIR, 'tensorboard')}")
print(f"\n🎯 To monitor training graphs, run:")
print(f"   tensorboard --logdir={os.path.join(LOG_DIR, 'tensorboard')}")
print(f"\n🔮 To test more images, run:")
print(f"   python cloud.py")
print("="*60 + "\n")