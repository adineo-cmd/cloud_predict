# predict.py
import os
import sys
import logging
import cv2
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = os.getenv('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import *

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_model_safe():
    if not MODEL_PATH.exists():
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        logger.error("💡 Run 'python train.py' first to train a model!")
        return None
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        return load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def preprocess_image(image_path):
    if not Path(image_path).exists():
        logger.error(f"❌ Image not found: {image_path}")
        return None
    
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"❌ Failed to load image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_image(model, image_path, show_debug=False):
    logger.info(f"Processing: {image_path}")
    
    img = preprocess_image(image_path)
    if img is None:
        return
    
    prediction = model.predict(img, verbose=0)
    score = float(prediction[0][0])
    
    # FIXED: Standard binary classification logic
    # score > 0.5 → class 1 → "other"
    # score < 0.5 → class 0 → "cumulonimbus"
    predicted_class = 1 if score > PREDICTION_THRESHOLD else 0
    label = PREDICTION_LABELS.get(predicted_class, f"Class {predicted_class}")
    
    # Calculate confidence for the predicted class
    confidence = score if predicted_class == 1 else (1 - score)
    
    print("\n" + "="*60)
    print(f"🖼️  Image: {Path(image_path).name}")
    if SHOW_CONFIDENCE:
        print(f"📊 Confidence: {confidence:.{CONFIDENCE_DECIMALS}f} ({confidence*100:.{CONFIDENCE_DECIMALS}f}%)")
    print(f"🎯 Prediction: {label}")
    print(f"⚙️  Threshold: {PREDICTION_THRESHOLD}")
    
    if show_debug:
        print(f"\n🔍 DEBUG:")
        print(f"  Raw Score: {score:.6f}")
        print(f"  Predicted Class Index: {predicted_class}")
        print(f"  Class Mapping: {PREDICTION_LABELS}")
        print(f"  Input Shape: {img.shape}")
        print(f"  Pixel Range: [{img.min():.3f}, {img.max():.3f}]")
        
        print(f"\n📈 Threshold Analysis:")
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            result_class = 1 if score > thresh else 0
            result_label = PREDICTION_LABELS[result_class]
            marker = " ← CURRENT" if thresh == PREDICTION_THRESHOLD else ""
            print(f"  Threshold {thresh:.1f}: {result_label}{marker}")
    
    print("="*60 + "\n")
    
    return score, predicted_class


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    show_debug = "--debug" in sys.argv or "-d" in sys.argv
    
    model = load_model_safe()
    if model is None:
        return
    
    predict_image(model, image_path, show_debug=show_debug)


if __name__ == "__main__":
    main()