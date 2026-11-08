# app.py
# Professional web interface for the Cumulonimbus Cloud Classifier (Flask).
# Run:  python app.py   →  http://127.0.0.1:5000
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

from config import (MODEL_PATH, RESULTS_DIR, IMG_SIZE, PREDICTION_THRESHOLD,
                    PREDICTION_LABELS)
from utils import configure_tf_runtime, setup_logger
from cli.predict_command import (_load_keras_model, _tta_batch, compute_gradcam, overlay_gradcam,
                                 load_model_ensemble, ensemble_scores, resolve_calibration,
                                 resolve_backbone)
from cli.data import get_preprocess_input
from calibration import apply_calibration

app = Flask(__name__, static_folder="static", template_folder="templates")
logger = setup_logger("webapp", str(Path(__file__).resolve().parent / "logs" / "webapp.log"))

_model = None
_calibration = None


def get_model():
    """Seed-ensemble members (cached)."""
    global _model
    if _model is None:
        _model, paths = load_model_ensemble(None)
        logger.info(f"Model ensemble loaded from {[str(p) for p in paths]}")
    return _model


def get_calibration():
    global _calibration
    if _calibration is None:
        _calibration = resolve_calibration(None)
        logger.info(f"Calibration: T={_calibration[0]} platt={_calibration[1]}")
    return _calibration


@app.route("/")
def index():
    return render_template("index.html", version="2.1.0")


@app.route("/api/health")
def health():
    ok = MODEL_PATH.exists()
    return jsonify({
        "status": "ok" if ok else "model_missing",
        "model": str(MODEL_PATH) if ok else None,
        "device": configure_tf_runtime.__doc__ and "configured",
        "time": datetime.now().isoformat(timespec="seconds"),
    })


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in request"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    tta = int(request.args.get("tta", 5))
    want_cam = request.args.get("gradcam", "0") == "1"
    threshold = float(request.args.get("threshold", PREDICTION_THRESHOLD))

    buf = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 415

    t0 = time.perf_counter()
    models = get_model()
    temperature, platt = get_calibration()
    preprocess_input = get_preprocess_input(resolve_backbone())
    batch = _tta_batch(img_bgr, IMG_SIZE, preprocess_input, max(1, tta))
    raw_score, stacked = ensemble_scores(models, batch)
    agreement = float(np.mean((stacked > threshold).astype(int) == (raw_score > threshold)))
    score = float(apply_calibration(raw_score, temperature, platt))
    pred_class = 1 if score > threshold else 0
    # Honest confidence: the calibrated probability of the predicted class.
    confidence = score if pred_class == 1 else 1 - score
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    payload = {
        "file": file.filename,
        "label": PREDICTION_LABELS[pred_class],
        "class_index": pred_class,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 6),
        "uncalibrated_score": round(raw_score, 6),
        "temperature": temperature,
        "calibration_method": "platt" if platt else ("temperature" if temperature != 1.0 else "none"),
        "calibrated": bool(platt) or abs(temperature - 1.0) > 1e-9,
        "view_agreement": round(agreement, 3),
        "ensemble_models": len(models),
        "latency_ms": latency_ms,
    }

    if want_cam:
        try:
            cam = compute_gradcam(models[0], batch[:1])
            vis_dir = RESULTS_DIR / "web_vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            out = vis_dir / f"{Path(file.filename).stem}_{int(time.time())}.jpg"
            overlay_gradcam(img_bgr, cam, out)
            payload["visualization_url"] = f"/vis/{out.name}"
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")

    logger.info(f"predict ok: {payload['label']} conf={confidence:.3f} {latency_ms}ms")
    return jsonify(payload)


@app.route("/vis/<path:filename>")
def serve_vis(filename):
    return send_from_directory(RESULTS_DIR / "web_vis", filename)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cloud Classifier web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    configure_tf_runtime(use_gpu=False)
    app.run(host=args.host, port=args.port, debug=False)
