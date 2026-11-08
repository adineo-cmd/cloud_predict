# Cumulonimbus Cloud Classifier ⛈️

Binary ground-based cloud classification: **cumulonimbus vs. other clouds/skies**.
Transfer-learning ensemble (EfficientNetV2-B0 by default) with TTA, temperature
calibration fitted on held-out validation data, Grad-CAM explainability, a Flask
web UI and a one-command pipeline.

## Pipeline

```
python run.py              # setup → train → calibrate → evaluate → benchmark → demo → web
python run.py quick        # reuse existing model, straight to the web UI
python -m cli train        # train the seed ensemble only
python -m cli evaluate     # held-out TEST split metrics (precision/recall/F1/ROC)
python -m cli predict img.jpg --json --gradcam
python app.py --port 8080  # web UI
```

## What makes the metrics trustworthy

- **Real data**: your own photos plus the public
  [CCSN database](https://doi.org/10.7910/DVN/CADDPD) (2,543 ground-based cloud
  photos; cumulonimbus class → `cumulonimbus`, all other classes → `other`).
  Reproducible via `python scripts/download_ccsn.py` (zip cached under `data/`,
  images gitignored with a `ccsn_` prefix).
- **Seeded stratified 80/10/10 split** (`utils.stratified_split`, cached in
  `results/dataset_split.json`) with a **held-out test split** that is never used
  for training, checkpointing, threshold search or calibration. A couple of your
  own photos per class are always reserved for val/test.
- **Real seed ensemble**: `CLOUD_ENSEMBLE_SIZE` (default 3) members trained with
  different seeds; their probabilities are averaged at inference. (The old
  "ensemble" averaged a model with its own checkpoint copy.)
- **Honest confidence**: the displayed confidence is the calibrated probability
  of the predicted class — no agreement multiplier. Platt scaling is
  fitted on the **validation** split only; artifacts fitted on training data are
  rejected at inference.
- **Optimal threshold**: searched on validation, reported as its transfer to test.

## Results (held-out test split, 256 images)

| Configuration | Accuracy | Macro-F1 | ROC-AUC | ECE |
|---|---|---|---|---|
| MobileNetV2 (original recipe) | 90.62% | 57.51% | 0.833 | 0.017 |
| EfficientNetV2-B0 (1 seed) | 92.97% | 74.42% | 0.894 | 0.018 |
| **Proposed: ensemble ×3 + Platt** | **93.36%** | **77.46%** | **0.921** | **0.027** |

Full analysis in `paper/cloud_classifier_research_paper.pdf` (IEEE style).
Model weights are distributed on Hugging Face (not git).

## Configuration (env-overridable)

| Variable | Default | Meaning |
|---|---|---|
| `CLOUD_ENSEMBLE_SIZE` | `3` | number of ensemble members to train |
| `CLOUD_BACKBONE` | `efficientnet_v2_b0` | or `mobilenet_v2` |
| `CLOUD_IMG_SIZE` | `256` | input resolution |
| `CLOUD_EPOCHS` | `15` | phase-1 epochs (finetune = max(3, epochs/3)) |
| `CLOUD_BATCH_SIZE` | `32` | batch size |
| `CLOUD_TTA` | `5` | test-time augmentation views at inference |
| `CLOUD_THRESHOLD` | `0.5` | decision threshold (score > t → "other") |
| `CLOUD_SEED` | `42` | base seed (members use seed + 1000·i) |

## Layout

```
cli/            train | predict | evaluate | calibrate | history | benchmark
cli/data.py     split → Keras generators, backbone preprocessing
utils.py        seeding + seeded stratified split
calibration.py  Platt scaling / temperature scaling math + ECE
scripts/download_ccsn.py   CCSN download & merge into dataset/
scripts/eval_tta.py        per-configuration held-out evaluation
scripts/make_paper_assets.py / make_paper_docx.py   paper figure & paper generator
dataset/        cumulonimbus/ other/  (your photos + ccsn_*.jpg)
models/         ensemble/member_*.keras (gitignored), calibration.json
results/        evaluation_report.json, split cache, plots, visualizations
paper/          research paper (DOCX source of truth + PDF)
```

## License

MIT — see LICENSE. CCSN data has its own terms on Harvard Dataverse; cite
Zhang et al., *"CloudNet: Ground-Based Cloud Classification With Deep
Learning"*, GRL 2018 when using the merged dataset.
