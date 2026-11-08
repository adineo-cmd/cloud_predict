# cli/predict_command.py
# Professional inference: batch/directory/glob inputs, TTA, Grad-CAM explainability,
# JSON output, optional saved visualizations.
import json
import glob as globlib
from pathlib import Path

import click
import numpy as np
import cv2

from errors import ModelNotFoundError, InvalidImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_inputs(patterns):
    files = []
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            files += [f for f in sorted(path.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        elif any(ch in str(p) for ch in "*?["):
            files += [Path(f) for f in sorted(globlib.glob(p))
                      if Path(f).suffix.lower() in IMAGE_EXTS]
        else:
            files.append(path)
    if not files:
        raise InvalidImageError(f"No images matched input(s): {patterns}")
    return files


def _load_keras_model(model_path):
    from config import MODEL_PATH
    from tensorflow.keras.models import load_model

    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(path)
    return load_model(path), path


def _preprocess(img_bgr, img_size, preprocess_input):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    return preprocess_input(rgb).astype("float32")


def _multi_crop(img_bgr, scale: float, dx: float = 0.0, dy: float = 0.0):
    """Center/scale crop with optional fractional offsets (clamped inside bounds)."""
    h, w = img_bgr.shape[:2]
    ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
    max_dx, max_dy = max(0, (w - cw) // 2), max(0, (h - ch) // 2)
    ox = int(round(dx * max_dx))
    oy = int(round(dy * max_dy))
    x0 = (w - cw) // 2 + ox
    y0 = (h - ch) // 2 + oy
    return img_bgr[y0:y0 + ch, x0:x0 + cw]


def _brightness(img_bgr, factor: float):
    return np.clip(img_bgr.astype("float32") * factor, 0, 255).astype("uint8")


# Deterministic view generators — each entry transforms a BGR image.
# Diverse views that average out model noise → more stable predictions.
_TTA_VIEWS = [
    lambda im: im,                                        # original
    lambda im: cv2.flip(im, 1),                           # horizontal flip
    lambda im: _multi_crop(im, 0.9),                      # center crop 90%
    lambda im: _multi_crop(im, 0.85, dx=-1.0, dy=0.0),    # left-biased crop
    lambda im: _multi_crop(im, 0.85, dx=1.0, dy=0.0),     # right-biased crop
    lambda im: _brightness(im, 1.15),                     # brighter sky
    lambda im: _brightness(im, 0.85),                     # darker / stormier
    lambda im: cv2.GaussianBlur(im, (0, 0), sigmaX=1.5),  # mild denoise
]


def _tta_batch(img_bgr, img_size, preprocess_input, runs):
    """Build a batch of deterministic augmented views for test-time augmentation.

    Uses up to len(_TTA_VIEWS) distinct geometric/photometric views; averaging
    their scores suppresses single-view noise. Note: for cumulonimbus detection
    the geometric crop views can discard diagnostic sky context (see paper).
    """
    runs = max(1, min(int(runs), len(_TTA_VIEWS)))
    batch = []
    for i in range(runs):
        view = _TTA_VIEWS[i](img_bgr)
        batch.append(_preprocess(view, img_size, preprocess_input))
    return np.stack(batch)


def load_model_ensemble(model_path=None):
    """Load the seed-ensemble members trained by `python -m cli train`.

    Members live in models/ensemble/member_*.keras and their probabilities are
    averaged at inference (a genuine variance reduction). Falls back to the
    single primary model when no ensemble directory exists. An explicit
    --model path overrides everything and loads just that model.
    Returns (models, paths).
    """
    from config import MODEL_PATH, ENSEMBLE_DIR, CONFIDENCE_ENSEMBLE

    if model_path:
        primary, ppath = _load_keras_model(model_path)
        return [primary], [ppath]

    models, paths = [], []
    if CONFIDENCE_ENSEMBLE:
        members = sorted(ENSEMBLE_DIR.glob("member_*.keras"))
        if members:
            try:
                from tensorflow.keras.models import load_model
                for m in members:
                    models.append(load_model(m))
                    paths.append(m)
            except Exception:
                models, paths = [], []  # fall back to primary below

    if not models:
        primary, ppath = _load_keras_model(None)
        models, paths = [primary], [ppath]
    return models, paths


def resolve_backbone():
    """Backbone of the loaded artifact (from ensemble manifest, else config)."""
    from config import BACKBONE, ENSEMBLE_MANIFEST_PATH
    try:
        import json
        manifest = json.loads(Path(ENSEMBLE_MANIFEST_PATH).read_text())
        return manifest.get("backbone", BACKBONE)
    except Exception:
        return BACKBONE


def ensemble_scores(models, batch):
    """Mean raw sigmoid score across all ensemble members."""
    per_model = [m.predict(batch, verbose=0).flatten() for m in models]
    stacked = np.stack(per_model)          # (n_models, n_views)
    return float(np.mean(stacked)), stacked


def resolve_calibration(calibrate: bool | None = True):
    """Return (temperature, platt) from the calibration artifact.

    Only artifacts fitted on the validation split (with enough samples) are
    trusted; anything else — notably temperatures fitted on training data,
    which inflate confidence — falls back to identity (T=1, no Platt).
    """
    from config import (CALIBRATION_ENABLED, CALIBRATION_PATH, DEFAULT_TEMPERATURE,
                        CALIBRATION_TRUSTED_FIT, CALIBRATION_MIN_SAMPLES)
    from calibration import load_calibration

    enabled = CALIBRATION_ENABLED if calibrate is None else bool(calibrate)
    if not enabled:
        return 1.0, None
    cal = load_calibration(CALIBRATION_PATH)
    if cal is None:
        return DEFAULT_TEMPERATURE, None
    if cal.get("fitted_on") != CALIBRATION_TRUSTED_FIT:
        return 1.0, None
    if int(cal.get("n_samples", 0)) < CALIBRATION_MIN_SAMPLES:
        return 1.0, None
    platt = cal.get("platt") if isinstance(cal.get("platt"), dict) else None
    return float(cal.get("temperature", DEFAULT_TEMPERATURE)), platt


def resolve_temperature(calibrate: bool | None = True):
    """Return the fitted temperature (1.0 disables scaling). Compat wrapper."""
    temperature, _ = resolve_calibration(calibrate)
    return temperature


def compute_gradcam(model, img_array, last_conv_name=None):
    """Grad-CAM heatmap for the given preprocessed single-image array (1,H,W,C)."""
    import tensorflow as tf

    if isinstance(model, (list, tuple)):
        model = model[0]  # ensemble: explain using the primary model

    # Pick the deepest true convolutional feature extractor layer.
    # Exclude top prediction convs (e.g. MobileNetV2 'Conv_1', 1x1x1) and
    # depthwise separable convs, which produce degenerate/unstable CAMs.
    def _is_good_conv(l):
        n = l.name.lower()
        if "depthwise" in n or "conv_" == n.rstrip("0123456789"):
            return False
        cfg = getattr(l, "get_config", lambda: {})()
        if cfg.get("filters") is None:      # depthwise conv has no filters
            return False
        s = l.output.shape
        return len(s) == 4 and None not in s[1:] and s[-1] >= 32

    conv_layers = [l for l in model.layers if _is_good_conv(l)]
    if not conv_layers:
        raise RuntimeError("No suitable Conv2D layer found for Grad-CAM.")
    target = model.get_layer(last_conv_name) if last_conv_name else conv_layers[-1]

    img = tf.cast(img_array, tf.float32)

    # Keras 3 functional models: build a sub-model once, then call it INSIDE
    # the tape so gradients are recorded against that watched invocation.
    layer_model = tf.keras.Model(model.inputs, [target.output, model.output])

    with tf.GradientTape() as tape:
        conv_out_full, preds = layer_model(img, training=False)
        loss = preds[0][0] if preds.shape[-1] == 1 else tf.reduce_max(preds[0])
        # Gradient wrt the conv activation must be requested FIRST: after the
        # first query a non-persistent tape is released (further queries
        # would return None on some TF builds).
        grads = tape.gradient(loss, conv_out_full)

    if grads is None:
        raise RuntimeError("Grad-CAM gradient is None; ensure a trainable Conv2D target exists.")
    weights = tf.reduce_mean(grads[0], axis=(0, 1))       # (C,)
    cam = tf.nn.relu(tf.reduce_sum(conv_out_full[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_gradcam(original_bgr, cam, out_path=None):
    orig_h, orig_w = original_bgr.shape[:2]
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (orig_w, orig_h))
    overlay = cv2.addWeighted(original_bgr, 0.6, heat, 0.4, 0)
    if out_path:
        cv2.imwrite(str(out_path), overlay)
    return overlay


def run_predict(images, model_path=None, threshold=None, tta_runs=None,
                gradcam=False, save_vis=False, as_json=False, debug=False,
                calibrate=None):
    from config import (IMG_SIZE, PREDICTION_THRESHOLD, PREDICTION_LABELS, SHOW_CONFIDENCE,
                        TTA_DEFAULT_RUNS, UNCERTAINTY_LOW_CONF, VISUALIZATIONS_DIR)
    from calibration import apply_calibration
    from cli.data import get_preprocess_input

    models, mpaths = load_model_ensemble(model_path)
    thr = threshold if threshold is not None else PREDICTION_THRESHOLD
    runs = tta_runs if tta_runs is not None else TTA_DEFAULT_RUNS
    temperature, platt = resolve_calibration(calibrate)
    preprocess_input = get_preprocess_input(resolve_backbone())

    files = _resolve_inputs(images)
    results = []
    failed = 0

    for f in files:
        img_bgr = cv2.imread(str(f))
        if img_bgr is None:
            click.secho(f"\u2717 Unreadable image skipped: {f}", fg="yellow", err=True)
            failed += 1
            continue

        batch = _tta_batch(img_bgr, IMG_SIZE, preprocess_input, max(1, runs))
        raw_score, stacked = ensemble_scores(models, batch)
        uncertainty = float(np.std(stacked))
        # Consensus across views/models, reported as its own metric (not folded
        # into confidence — that used to inflate the displayed number).
        agreement = float(np.mean((stacked > thr).astype(int) == (raw_score > thr)))

        score = float(apply_calibration(raw_score, temperature, platt))
        pred_class = 1 if score > thr else 0
        label = PREDICTION_LABELS.get(pred_class, f"Class {pred_class}")
        confidence = score if pred_class == 1 else 1 - score

        entry = {
            "image": str(f),
            "score": round(score, 6),
            "raw_score": round(raw_score, 6),
            "predicted_class": pred_class,
            "label": label,
            "confidence": round(confidence, 4),
            "threshold": thr,
            "temperature": temperature,
            "calibration_method": "platt" if platt else ("temperature" if temperature != 1.0 else "none"),
            "tta_runs": int(min(max(1, runs), len(_TTA_VIEWS))),
            "ensemble_models": len(models),
            "view_agreement": round(agreement, 4),
            "tta_std": round(uncertainty, 4),
            "calibrated": bool(platt) or abs(temperature - 1.0) > 1e-9,
        }

        if gradcam:
            try:
                cam = compute_gradcam(models[0], batch[:1])
                vis_path = None
                if save_vis:
                    vis_dir = VISUALIZATIONS_DIR / "gradcam"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / (f.stem + "_gradcam.jpg")
                overlay_gradcam(img_bgr, cam, vis_path)
                entry["gradcam_saved"] = str(vis_path) if vis_path else None
            except Exception as e:  # never fail prediction because of explanation
                click.secho(f"\u26a0 Grad-CAM failed for {f}: {e}", fg="yellow", err=True)

        results.append(entry)

        if not as_json:
            color = "magenta" if pred_class == 0 else "cyan"
            flag = " ~uncertain" if confidence < UNCERTAINTY_LOW_CONF else ""
            click.secho(f"{f.name:<38} \u2192 {label}{flag}", fg=color)
            if SHOW_CONFIDENCE:
                extra = ""
                if temperature != 1.0:
                    extra += f" T={temperature:.2f}"
                if runs > 1:
                    extra += f" agreement={agreement:.0%} tta_std={uncertainty:.4f}"
                click.echo(f"{'':<38}   confidence={confidence:.2%} score={score:.4f}"
                           f" (raw={raw_score:.4f}){extra}")
            if debug:
                click.echo(f"{'':<38}   models={[str(p) for p in mpaths]} "
                           f"shape={(IMG_SIZE, IMG_SIZE, 3)}")
                for t in [0.3, 0.5, 0.7]:
                    mark = " \u2190 CURRENT" if abs(t - thr) < 1e-9 else ""
                    click.echo(f"{'':<38}   threshold {t:.1f} \u2192 {PREDICTION_LABELS[1 if score > t else 0]}{mark}")

    if as_json:
        click.echo(json.dumps({"model": [str(p) for p in mpaths], "count": len(results),
                               "results": results}, indent=2))

    if save_vis and not gradcam:
        click.secho("\u2139 --save-vis has no effect without --gradcam", fg="yellow")

    return 0 if failed == 0 else 4
