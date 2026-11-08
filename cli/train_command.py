# cli/train_command.py
# Production-grade training: seeded stratified splits, configurable transfer-learning
# backbone, cosine LR schedules, label smoothing, two-phase fine-tuning, and a real
# seed ensemble (N members, averaged probabilities) with validation-only calibration.
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from utils import set_global_seed, configure_tf_runtime, save_metrics_csv, get_git_version
from errors import DatasetNotFoundError


def _validate_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        raise DatasetNotFoundError(f"Dataset directory not found: {dataset_dir}")
    classes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    if set(classes) != {"cumulonimbus", "other"}:
        raise DatasetNotFoundError(
            f"Expected class folders 'cumulonimbus' and 'other' under {dataset_dir}, found {classes}"
        )


def _build_model(input_shape, backbone_name: str):
    """Frozen transfer-learning backbone + configurable classification head."""
    from config import BACKBONES, BASE_MODEL_WEIGHTS, INCLUDE_TOP, HEAD_LAYERS, OUTPUT_UNITS, OUTPUT_ACTIVATION
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    ctors = {"efficientnet_v2_b0": EfficientNetV2B0, "mobilenet_v2": MobileNetV2}
    if backbone_name not in ctors:
        raise ValueError(f"Unknown backbone {backbone_name!r}; choose from {sorted(ctors)}")

    base_model = ctors[backbone_name](weights=BASE_MODEL_WEIGHTS, include_top=INCLUDE_TOP,
                                      input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    for layer_config in HEAD_LAYERS:
        x = getattr(tf.keras.layers, layer_config["type"])(**layer_config["params"])(x)
    predictions = Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)(x)
    return Model(inputs=base_model.input, outputs=predictions), base_model


def _compile(model, lr, decay_steps=None):
    """Compile with (optional cosine-decayed) Adam and smoothed binary CE."""
    from config import LOSS_FUNCTION, LABEL_SMOOTHING
    import tensorflow as tf

    lr_schedule = None
    if decay_steps:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=int(decay_steps))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule or lr)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING) \
        if LOSS_FUNCTION == "binary_crossentropy" else LOSS_FUNCTION
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])


def _train_member(member_idx: int, member_seed: int, epochs: int, batch_size: int,
                  learning_rate: float, do_finetune: bool, runtime: str):
    """Train one ensemble member end-to-end. Returns (path, val_metrics, history)."""
    from config import (IMG_SIZE, INPUT_SHAPE, BACKBONE, EPOCHS, MODEL_DIR,
                        ENSEMBLE_DIR, RESULTS_DIR, LOG_DIR,
                        EARLY_STOPPING, TENSORBOARD, USE_GPU,
                        FINETUNE_ENABLED, FINETUNE_FRACTION_UNFROZEN, FINETUNE_LR_DIVISOR)
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    from sklearn.utils.class_weight import compute_class_weight
    from cli.data import load_split

    set_global_seed(member_seed)
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    member_path = ENSEMBLE_DIR / f"member_{member_idx}.keras"

    train_gen = load_split("train", batch_size=batch_size, augment=True, seed=member_seed)
    val_gen = load_split("val", batch_size=batch_size, augment=False)
    counts = {c: int(np.sum(train_gen.classes == i)) for c, i in train_gen.class_indices.items()}
    click.echo(f"  Data: {len(train_gen.filenames)} train / {len(val_gen.filenames)} val {counts}")

    class_weights = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)))

    model, base_model = _build_model(INPUT_SHAPE, BACKBONE)
    steps_per_epoch = max(1, len(train_gen))
    _compile(model, learning_rate, decay_steps=steps_per_epoch * epochs)
    click.echo(f"  Architecture: {BACKBONE} + head | {model.count_params():,} params")

    callbacks = []
    if EARLY_STOPPING["enabled"]:
        callbacks.append(EarlyStopping(**{k: v for k, v in EARLY_STOPPING.items() if k != "enabled"}))
    callbacks.append(ModelCheckpoint(str(member_path), monitor="val_loss",
                                     save_best_only=True, mode="min", verbose=0))
    if TENSORBOARD["enabled"]:
        tb = dict(TENSORBOARD)
        tb["log_dir"] = str(LOG_DIR / "tensorboard" / f"member_{member_idx}")
        callbacks.append(TensorBoard(**{k: v for k, v in tb.items() if k != "enabled"}))

    # ---- Phase 1: frozen backbone ---------------------------------------------------
    t0 = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                        callbacks=callbacks, class_weight=class_weights, verbose=1)
    all_hist = dict(history.history)

    # ---- Phase 2: optional fine-tuning ----------------------------------------------
    if do_finetune and FINETUNE_ENABLED:
        click.secho("  ▶ Fine-tuning top base layers (BatchNorm frozen, reduced lr)...", fg="magenta")
        n_layers = len(base_model.layers)
        unfreeze_from = int(n_layers * (1 - FINETUNE_FRACTION_UNFROZEN))
        for layer in base_model.layers[unfreeze_from:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        ft_epochs = max(3, epochs // 3)
        _compile(model, learning_rate / FINETUNE_LR_DIVISOR,
                 decay_steps=steps_per_epoch * ft_epochs)
        ft_cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, mode="min"),
                 ModelCheckpoint(str(member_path), monitor="val_loss",
                                 save_best_only=True, mode="min", verbose=0)]
        ft_hist = model.fit(train_gen, validation_data=val_gen, epochs=ft_epochs,
                            callbacks=ft_cb, class_weight=class_weights, verbose=1)
        for k, v in ft_hist.history.items():
            all_hist[k] = all_hist.get(k, []) + v

    duration = time.time() - t0
    model.save(member_path)

    # Keras 3's evaluate() aggregates metrics under 'compile_metrics', so read the
    # per-name metrics from history at the best epoch (the one ModelCheckpoint kept,
    # matching EarlyStopping's restore_best_weights).
    val_metrics = {}
    if "val_loss" in all_hist and all_hist["val_loss"]:
        best_epoch = int(np.argmin(all_hist["val_loss"]))
        val_metrics = {k[len("val_"):]: float(vals[best_epoch])
                       for k, vals in all_hist.items() if k.startswith("val_")}
    csv_path = save_metrics_csv(type("H", (), {"history": all_hist})(),
                                RESULTS_DIR / f"training_history_member{member_idx}.csv")
    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()) or "n/a"
    click.secho(f"  ✓ Member {member_idx} done in {duration / 60:.1f} min — {metrics_str}",
                fg="green")
    return {"path": member_path, "seed": member_seed, "val_metrics": val_metrics,
            "duration_seconds": round(duration, 1), "history_csv": str(csv_path)}


def _resume_member(member_idx: int, member_seed: int, batch_size: int):
    """Reuse an existing trained member checkpoint instead of retraining it.

    Makes training idempotent across interruptions: a loadable member_N.keras
    is scored on the validation split and kept; a corrupt/partial file is
    deleted so the member retrains from scratch.
    """
    from config import ENSEMBLE_DIR
    from cli.data import load_split

    member_path = ENSEMBLE_DIR / f"member_{member_idx}.keras"
    if not member_path.exists():
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(member_path)
    except Exception:
        member_path.unlink(missing_ok=True)
        return None

    val_gen = load_split("val", batch_size=batch_size, augment=False)
    scores = np.clip(model.predict(val_gen, verbose=0).flatten(), 1e-6, 1 - 1e-6)
    labels = np.array(val_gen.classes, dtype=float)
    preds = (scores > 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
    val_metrics = {"loss": round(float(log_loss(labels, scores)), 4),
                   "accuracy": round(float(accuracy_score(labels, preds)), 4),
                   "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
                   "recall": round(float(recall_score(labels, preds, zero_division=0)), 4)}
    click.secho(f"  ✓ Member {member_idx} resumed from existing checkpoint "
                f"(val_acc={val_metrics['accuracy']:.4f})", fg="yellow")
    return {"path": member_path, "seed": member_seed, "val_metrics": val_metrics,
            "duration_seconds": 0.0, "history_csv": None}


def run_train(epochs=None, batch_size=None, learning_rate=None,
              finetune=None, seed=42, device=None):
    from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, RESULTS_DIR,
                        ENSEMBLE_SIZE, ENSEMBLE_SEED_STRIDE, ENSEMBLE_MANIFEST_PATH,
                        CALIBRATION_ENABLED, CALIBRATION_PATH, TEMPERATURE_BOUNDS,
                        CALIBRATION_TRUSTED_FIT, BACKBONE, IMG_SIZE, USE_GPU, DATASET_DIR,
                        FINETUNE_ENABLED)
    from cli.data import split_sizes

    set_global_seed(seed)
    use_gpu = (device == "gpu") if device else USE_GPU
    runtime = configure_tf_runtime(use_gpu=use_gpu)

    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE
    do_finetune = FINETUNE_ENABLED if finetune is None else finetune

    click.secho(f"▶ Training | device={runtime} backbone={BACKBONE} img={IMG_SIZE} "
                f"epochs={epochs} batch={batch_size} lr={learning_rate} "
                f"finetune={do_finetune} members={ENSEMBLE_SIZE} base_seed={seed}",
                fg="cyan", bold=True)

    _validate_dataset(Path(DATASET_DIR))
    sizes = split_sizes()
    click.echo(f"  Split: " + " | ".join(
        f"{s}: {sum(v.values())} ({v})" for s, v in sizes.items()))

    seeds = [seed + i * ENSEMBLE_SEED_STRIDE for i in range(max(1, ENSEMBLE_SIZE))]
    members = []
    for i, member_seed in enumerate(seeds):
        click.secho(f"▶ Ensemble member {i + 1}/{len(seeds)} (seed={member_seed})...",
                    fg="cyan", bold=True)
        resumed = _resume_member(i, member_seed, batch_size)
        members.append(resumed or _train_member(i, member_seed, epochs, batch_size,
                                                learning_rate, do_finetune, runtime))

    # ---- Copy the best member to the backward-compat primary model path -------------
    # Only for the main ensemble — baseline/isolated runs (custom CLOUD_ENSEMBLE_DIR)
    # must not overwrite the deployed model.
    from config import MODEL_DIR, ENSEMBLE_DIR as _ens_dir
    best = max(members, key=lambda m: m["val_metrics"].get("accuracy", 0.0))
    if Path(_ens_dir) == Path(MODEL_DIR / "ensemble"):
        shutil.copy2(best["path"], MODEL_PATH)
        click.echo(f"  Best member: {best['path'].name} (val_acc={best['val_metrics']['accuracy']:.4f}) "
                   f"→ {MODEL_PATH}")
    else:
        click.echo(f"  Best member: {best['path'].name} (val_acc={best['val_metrics']['accuracy']:.4f}) "
                   f"(isolated run — primary model untouched)")

    # ---- Calibration: fitted on VALIDATION split with ensemble scores ---------------
    from cli.data import load_split
    from cli.predict_command import load_model_ensemble
    from calibration import (fit_temperature, fit_platt, save_calibration,
                             confidence_report, nll_loss, _logit)

    temperature, cal_report, platt = 1.0, None, None
    val_gen = load_split("val", batch_size=batch_size, augment=False)
    val_labels = np.array(val_gen.classes, dtype=float)
    models, _ = load_model_ensemble()
    per_model = [m.predict(val_gen, verbose=0).flatten() for m in models]
    val_scores = np.clip(np.mean(np.stack(per_model), axis=0), 1e-6, 1 - 1e-6)

    if CALIBRATION_ENABLED:
        if Path(_ens_dir) != Path(MODEL_DIR / "ensemble"):
            click.secho("  • Isolated run — skipping calibration artifact (main calibration kept).",
                        fg="yellow")
        else:
            click.secho("▶ Fitting confidence calibration on validation split (ensemble scores)...",
                        fg="cyan")
            temperature = fit_temperature(val_scores, val_labels,
                                          grid_lo=TEMPERATURE_BOUNDS[0],
                                          grid_hi=TEMPERATURE_BOUNDS[1])
            platt = fit_platt(val_scores, val_labels)
            # Prefer whichever mapping minimizes NLL on the validation split.
            if platt:
                platt_probs = np.array([1 / (1 + np.exp(-(platt["a"] * _logit(s) + platt["b"])))
                                        for s in val_scores])
                if nll_loss(val_scores, val_labels, temperature) <= \
                        float(-np.mean(val_labels * np.log(np.clip(platt_probs, 1e-7, 1 - 1e-7))
                                       + (1 - val_labels) * np.log(np.clip(1 - platt_probs, 1e-7, 1 - 1e-7)))):
                    platt = None  # temperature wins
                else:
                    temperature = 1.0
            cal_report = confidence_report(val_scores, val_labels, temperature)
            save_calibration(CALIBRATION_PATH.parent, {
                "temperature": temperature,
                "platt": platt,
                "method": "platt" if platt else "temperature",
                "fitted_on": CALIBRATION_TRUSTED_FIT,
                "n_samples": int(len(val_scores)),
                "ece_before": cal_report["before_scaling"]["ece"],
                "ece_after": cal_report["after_scaling"]["ece"],
                "avg_confidence_correct_before": cal_report["before_scaling"]["avg_confidence_correct"],
                "avg_confidence_correct_after": cal_report["after_scaling"]["avg_confidence_correct"],
                "model": str(MODEL_PATH),
                "ensemble_size": len(models),
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            })
            method = f"Platt scaling (a={platt['a']:.3f}, b={platt['b']:.3f})" if platt \
                else f"temperature T={temperature:.3f}"
            click.secho(f"  ✓ Calibration saved to {CALIBRATION_PATH}: {method}", fg="green")
            click.echo(f"    ECE (temperature view): {cal_report['before_scaling']['ece']:.4f} → "
                       f"{cal_report['after_scaling']['ece']:.4f}")

    # ---- Manifests -------------------------------------------------------------------
    ensemble_manifest = {
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "git_version": get_git_version(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "split_sizes": sizes,
        "members": [{"path": str(m["path"]), "seed": m["seed"],
                     "val_metrics": {k: round(float(v), 4) for k, v in m["val_metrics"].items()}}
                    for m in members],
        "best_member": str(best["path"]),
        "calibration": {"method": "platt" if platt else "temperature",
                        "temperature": temperature, "platt": platt,
                        "fitted_on": CALIBRATION_TRUSTED_FIT},
    }
    from config import ENSEMBLE_MANIFEST_PATH
    ENSEMBLE_MANIFEST_PATH.write_text(json.dumps(ensemble_manifest, indent=2))

    manifest = {
        "model": BACKBONE,
        "trained_at": ensemble_manifest["trained_at"],
        "git_version": get_git_version(),
        "base_seed": seed,
        "device": runtime,
        "ensemble_size": len(members),
        "members": ensemble_manifest["members"],
        "best_val_accuracy": round(float(best["val_metrics"]["accuracy"]), 4),
        "calibration": {"temperature": temperature, "report": cal_report},
        "artifacts": {"primary_model": str(MODEL_PATH),
                      "ensemble_manifest": str(ENSEMBLE_MANIFEST_PATH),
                      "calibration": str(CALIBRATION_PATH) if cal_report else None},
    }
    manifest_path = RESULTS_DIR / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    click.secho(f"✓ Training complete — {len(members)} members | best val acc "
                f"{best['val_metrics']['accuracy']:.4f}", fg="green", bold=True)
    click.echo(f"  Ensemble  : {ENSEMBLE_MANIFEST_PATH}")
    click.echo(f"  Primary   : {MODEL_PATH}")
    if cal_report is not None:
        click.echo(f"  Calibrated: T={temperature:.3f} (fit on validation split)")
