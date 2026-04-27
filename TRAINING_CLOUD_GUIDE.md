# Training-First Cloud Guide

This guide is for your current priority:

- clean the repo down to model-training essentials
- move that smaller codebase to Colab or another cloud environment
- retrain and improve the OCR model from the terminal
- leave Django for later

## Recommended direction

Right now, treat this as a training repo, not an app repo.

The sentence model path is the most important one because your OCR pipeline uses:

- `django_backend/train_sentence_model.py`
- `django_backend/model.py`
- `django_backend/dataset.py`
- `django_backend/data_augmentation.py`
- `django_backend/char_map.py`
- `django_backend/post_processing.py`

## Important issue fixed before retraining

Your existing real-sentence split is not safe for evaluation:

- `train_real_sentences` and `val_real_sentences` overlap 100%

That means validation loss from that split is not trustworthy.

Use this new script instead:

- `django_backend/build_real_sentence_splits.py`

It creates non-overlapping train/validation/test splits from the corpus in one pass.

## Keep these files for a training-only repo

Keep:

- `django_backend/char_map.py`
- `django_backend/model.py`
- `django_backend/model_old.py`
- `django_backend/dataset.py`
- `django_backend/data_augmentation.py`
- `django_backend/generate_real_sentence_data.py`
- `django_backend/build_real_sentence_splits.py`
- `django_backend/train_sentence_model.py`
- `django_backend/train_transfer_learning.py`
- `django_backend/predict_cli.py`
- `django_backend/post_processing.py`
- `django_backend/requirements-train.txt`
- `django_backend/fonts/working_fonts.txt`
- the font files referenced in `working_fonts.txt`
- `django_backend/data/as-wiki-2021.txt`

Optional:

- `django_backend/checkpoints/best_model_fast.pth`
  Use this only if you want transfer learning.

## Leave these out for now

Do not carry these into the training-first cloud repo:

- `django_backend/.venv/`
- `frontend/`
- Django app files
- `frontend/node_modules/`
- training/debug screenshots
- old backup interface files
- local databases
- most `test_*.py`, `diagnose_*.py`, `analyze_*.py`

## Fastest clean workflow

### 1. Build a training bundle locally

Use:

```bash
python scripts/create_training_bundle.py
```

That creates a smaller folder called `training_bundle/` with the essentials only.

### 2. Push the bundle to GitHub

From inside `training_bundle/`:

```bash
git init
git add .
git commit -m "Initial training-only OCR repo"
```

Then connect your remote and push.

### 3. Open Colab

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone <your-training-repo-url>
%cd /content/<your-training-repo>/django_backend
!pip install -r requirements-train.txt
```

## Suggested folder strategy in Colab

Keep code in Git, but keep generated data and big checkpoints in Drive:

- `/content/drive/MyDrive/assamese_ocr_assets/data/`
- `/content/drive/MyDrive/assamese_ocr_assets/checkpoints/`

Then link them:

```bash
!ln -s /content/drive/MyDrive/assamese_ocr_assets/data /content/<your-training-repo>/django_backend/data
!ln -s /content/drive/MyDrive/assamese_ocr_assets/checkpoints /content/<your-training-repo>/django_backend/checkpoints
```

If `data/` already exists in the repo copy, remove or rename it first.

## Clean retraining flow

### Step 1. Build non-overlapping splits

```bash
!python build_real_sentence_splits.py \
  --input data/as-wiki-2021.txt \
  --train-output data/train_real_sentences \
  --val-output data/val_real_sentences \
  --test-output data/test_real_sentences \
  --train-count 15000 \
  --val-count 4000 \
  --test-count 1000 \
  --seed 42
```

### Step 2. Train from the terminal

```bash
!python train_sentence_model.py \
  --train-img-dir data/train_real_sentences/images \
  --train-label-file data/train_real_sentences/labels/labels.txt \
  --val-img-dir data/val_real_sentences/images \
  --val-label-file data/val_real_sentences/labels/labels.txt \
  --epochs 25 \
  --batch-size 32 \
  --best-checkpoint checkpoints/best_model_sentences.pth \
  --final-checkpoint checkpoints/final_model_sentences.pth
```

### Step 3. Predict from the terminal

```bash
!python predict_cli.py \
  --image data/test_real_sentences/images/sentence_000000.png \
  --checkpoint checkpoints/best_model_sentences.pth
```

## If you want transfer learning later

If you keep `best_model_fast.pth`, you can still use:

```bash
!python train_transfer_learning.py
```

But I would first get a clean non-overlapping sentence split working with `train_sentence_model.py`.

## Recommended next move

Use this order:

1. create `training_bundle/`
2. push only that bundle
3. generate clean splits in Colab
4. retrain sentence model
5. use `predict_cli.py` for terminal predictions
6. rebuild Django later around the improved checkpoint
