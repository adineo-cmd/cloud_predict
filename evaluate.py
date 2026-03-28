# evaluate.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = os.getenv('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import *

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_validation_data():
    logger.info("Loading validation data...")
    
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT
    )
    
    val_data = datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation",
        shuffle=False
    )
    
    return val_data


def evaluate_model():
    if not MODEL_PATH.exists():
        logger.error(f"❌ Model not found: {MODEL_PATH}")
        return
    model = load_model(MODEL_PATH)
    logger.info(f"✓ Loaded model from {MODEL_PATH}")
    
    val_data = load_validation_data()
    
    logger.info("Running predictions on validation set...")
    predictions = model.predict(val_data, verbose=1)
    y_pred = (predictions > PREDICTION_THRESHOLD).astype(int).flatten()
    y_true = np.array(val_data.classes)
    
    class_names = list(val_data.class_indices.keys())
    
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔲 Confusion Matrix:")
    print(cm)
    print(f"\nLegend:")
    print(f"  [[TN, FP],")
    print(f"   [FN, TP]]")
    print(f"  Where Positive = 'other' (class 1)")
    
    accuracy = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='binary')
    print(f"\n📊 Summary Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    
    cumulonimbus_ratio = np.mean(y_pred)
    if cumulonimbus_ratio > 0.8 or cumulonimbus_ratio < 0.2:
        print(f"\n⚠️  WARNING: Model predicts 'other' for {cumulonimbus_ratio*100:.1f}% of samples!")
    
    print("="*70 + "\n")
    
    if PLOT_CONFUSION_MATRIX:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.tight_layout()
        
        output_path = RESULTS_DIR / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {output_path}")
        plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_true
    }


if __name__ == "__main__":
    evaluate_model()