# debug_classes.py
# Run this to verify class mapping before training
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import *
from collections import Counter

print("="*60)
print("🔍 CLASS MAPPING DEBUG")
print("="*60)

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT
)

train_data = datagen.flow_from_directory(
    str(DATASET_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode="binary",
    subset="training",
    shuffle=False
)

print(f"\nClass Indices: {train_data.class_indices}")
print(f"Total Training Samples: {len(train_data.filenames)}")
print(f"\nDistribution:")
for class_name, idx in train_data.class_indices.items():
    count = sum(1 for c in train_data.classes if c == idx)
    print(f"  {class_name} (class {idx}): {count} images")

print(f"\nFirst 5 filenames and their labels:")
for i in range(min(5, len(train_data.filenames))):
    print(f"  {train_data.filenames[i]} → class {train_data.classes[i]}")

print("\n" + "="*60)
print("✅ Verify this matches PREDICTION_LABELS in config.py")
print("="*60)