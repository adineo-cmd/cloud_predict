# cli/train_command.py
# Production-grade training: reproducibility, cosine LR schedule, two-phase fine-tuning,
# CSV history export and JSON model manifest.
import json
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


def run_train(epochs=None, batch_size=None, learning_rate=None,
              finetune=None, seed=42, device=None):
    from config import (
        DATASET_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, CLASS_MODE,
        AUGMENTATION_CONFIG, BASE_MODEL_WEIGHTS, INCLUDE_TOP, INPUT_SHAPE,
        HEAD_LAYERS, OUTPUT_UNITS, OUTPUT_ACTIVATION, LOSS_FUNCTION, METRICS,
        LEARNING_RATE, EPOCHS, MODEL_PATH, BEST_MODEL_PATH, RESULTS_DIR,
        EARLY_STOPPING, MODEL_CHECKPOINT, TENSORBOARD, USE_GPU, SHUFFLE_TRAIN, SHUFFLE_VAL,
        FINETUNE_ENABLED, FINETUNE_FRACTION_UNFROZEN, FINETUNE_LR_DIVISOR, LR_REDUCE_ON_PLATEAU,
    )
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight

    # ---- Runtime / reproducibility -------------------------------------------------
    set_global_seed(seed)
    use_gpu = (device == "gpu") if device else USE_GPU
    runtime = configure_tf_runtime(use_gpu=use_gpu)

    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE
    do_finetune = FINETUNE_ENABLED if finetune is None else finetune

    click.secho(f"▶ Training | device={runtime} epochs={epochs} batch={batch_size} "
                f"lr={learning_rate} finetune={do_finetune} seed={seed}", fg="cyan", bold=True)

    _validate_dataset(Path(DATASET_DIR))

    # ---- Data ----------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT, **AUGMENTATION_CONFIG)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, validation_split=VALIDATION_SPLIT)

    common = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size, class_mode=CLASS_MODE)
    train_data = train_datagen.flow_from_directory(
        str(DATASET_DIR), subset="training", shuffle=SHUFFLE_TRAIN, seed=seed, **common)
    val_data = val_datagen.flow_from_directory(
        str(DATASET_DIR), subset="validation", shuffle=SHUFFLE_VAL, seed=seed, **common)

    counts = {c: int(np.sum(train_data.classes == i)) for c, i in train_data.class_indices.items()}
    click.echo(f"  Dataset: {len(train_data.filenames)} train / {len(val_data.filenames)} val {counts}")

    class_weights = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(train_data.classes), y=train_data.classes)))

    # ---- Model ---------------------------------------------------------------------
    base_model = MobileNetV2(weights=BASE_MODEL_WEIGHTS, include_top=INCLUDE_TOP,
                             input_shape=INPUT_SHAPE)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    for layer_config in HEAD_LAYERS:
        x = getattr(tf.keras.layers, layer_config["type"])(**layer_config["params"])(x)
    predictions = Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    def compile_model(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=LOSS_FUNCTION, metrics=METRICS)

    compile_model(learning_rate)
    click.echo(f"  Architecture: {BASE_MODEL_NAME} + head | {model.count_params():,} params")

    callbacks = []
    if EARLY_STOPPING["enabled"]:
        callbacks.append(EarlyStopping(**{k: v for k, v in EARLY_STOPPING.items() if k != "enabled"}))
    if MODEL_CHECKPOINT["enabled"]:
        callbacks.append(ModelCheckpoint(str(BEST_MODEL_PATH),
                         **{k: v for k, v in MODEL_CHECKPOINT.items() if k != "enabled"}))
    if TENSORBOARD["enabled"]:
        callbacks.append(TensorBoard(**{k: v for k, v in TENSORBOARD.items() if k != "enabled"}))
    callbacks.append(ReduceLROnPlateau(monitor="val_loss", verbose=1, **LR_REDUCE_ON_PLATEAU))

    # ---- Phase 1: frozen backbone --------------------------------------------------
    t0 = time.time()
    history = model.fit(
        train_data, validation_data=val_data, epochs=epochs,
        callbacks=callbacks, class_weight=class_weights, verbose=1)

    all_hist = dict(history.history)

    # ---- Phase 2: optional fine-tuning ---------------------------------------------
    if do_finetune:
        click.secho("▶ Fine-tuning top base layers (unfreeze tail, reduced lr)...", fg="magenta")
        n_layers = len(base_model.layers)
        unfreeze_from = int(n_layers * (1 - FINETUNE_FRACTION_UNFROZEN))
        for layer in base_model.layers[unfreeze_from:]:
            layer.trainable = True
        compile_model(learning_rate / FINETUNE_LR_DIVISOR)
        ft_epochs = max(3, epochs // 3)
        ft_cb = [ReduceLROnPlateau(monitor="val_loss", verbose=1,
                                   **{**LR_REDUCE_ON_PLATEAU, "min_lr": 1e-7})]
        if MODEL_CHECKPOINT["enabled"]:
            ft_cb.append(ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_accuracy",
                                         save_best_only=True, mode="max", verbose=1))
        ft_hist = model.fit(train_data, validation_data=val_data, epochs=ft_epochs,
                            callbacks=ft_cb, class_weight=class_weights, verbose=1)
        for k, v in ft_hist.history.items():
            all_hist[k] = all_hist.get(k, []) + v
        trainable_now = sum(1 for l in model.layers if l.trainable)
        click.echo(f"  Fine-tuned with {trainable_now} newly-trainable layers")

    duration = time.time() - t0

    # ---- Artifacts -----------------------------------------------------------------
    model.save(MODEL_PATH)

    csv_path = save_metrics_csv(type("H", (), {"history": all_hist})(),
                                RESULTS_DIR / "training_history.csv")

    val_metrics = dict(zip(model.metrics_names, model.evaluate(val_data, verbose=0)))
    best_val_acc = max(all_hist.get("val_accuracy", [0]))

    # ---- Confidence calibration (temperature scaling on validation split) ----------
    from config import CALIBRATION_ENABLED, CALIBRATION_PATH, TEMPERATURE_BOUNDS
    from calibration import (fit_temperature, save_calibration, confidence_report,
                             expected_calibration_error)

    temperature, cal_report = 1.0, None
    if CALIBRATION_ENABLED:
        click.secho("▶ Fitting confidence calibration (temperature scaling)...", fg="cyan")
        val_scores = np.clip(model.predict(val_data, verbose=0).flatten(), 1e-6, 1 - 1e-6)
        val_labels = np.array(val_data.classes, dtype=float)
        temperature = fit_temperature(val_scores, val_labels,
                                      grid_lo=TEMPERATURE_BOUNDS[0],
                                      grid_hi=TEMPERATURE_BOUNDS[1])
        cal_report = confidence_report(val_scores, val_labels, temperature)
        save_calibration(CALIBRATION_PATH.parent, {
            "temperature": temperature,
            "fitted_on": "validation_split",
            "n_samples": int(len(val_scores)),
            "ece_before": cal_report["before_scaling"]["ece"],
            "ece_after": cal_report["after_scaling"]["ece"],
            "avg_confidence_correct_before": cal_report["before_scaling"]["avg_confidence_correct"],
            "avg_confidence_correct_after": cal_report["after_scaling"]["avg_confidence_correct"],
            "model": str(MODEL_PATH),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        })
        ece_raw = expected_calibration_error(val_scores, val_labels.astype(int))
        click.secho(f"  ✓ Temperature T={temperature:.3f} saved to {CALIBRATION_PATH}",
                    fg="green")
        click.echo(f"    ECE: {ece_raw:.4f} → {cal_report['after_scaling']['ece']:.4f} | "
                   f"avg conf (correct): "
                   f"{cal_report['before_scaling']['avg_confidence_correct']} → "
                   f"{cal_report['after_scaling']['avg_confidence_correct']}")

    manifest = {
        "model": BASE_MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "git_version": get_git_version(),
        "seed": seed,
        "device": runtime,
        "epochs_phase1": epochs,
        "fine_tuned": bool(do_finetune),
        "duration_seconds": round(duration, 1),
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "best_val_accuracy": round(best_val_acc, 4),
        "final_val_metrics": {k: round(float(v), 4) for k, v in val_metrics.items()},
        "calibration": {"temperature": temperature,
                        "report": cal_report},
        "artifacts": {"model": str(MODEL_PATH), "best_model": str(BEST_MODEL_PATH),
                      "history_csv": csv_path,
                      "calibration": str(CALIBRATION_PATH) if cal_report else None},
    }
    manifest_path = RESULTS_DIR / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    click.secho(f"✓ Training complete in {duration/60:.1f} min", fg="green", bold=True)
    click.echo(f"  Model      : {MODEL_PATH}")
    click.echo(f"  Manifest   : {manifest_path}")
    click.echo(f"  History    : {csv_path}")
    if cal_report is not None:
        click.echo(f"  Calibrated : T={temperature:.3f} (confidence sharpened/calibrated)")
    click.echo(f"  Final val  : " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
