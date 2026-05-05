# 📊 How to Test Accuracy (Google Colab)

Since your data is on Google Drive, run evaluation in **Colab** where Drive is already mounted.

## Quick Setup — Add These Cells to Your Notebook

After your existing setup cells (mounting Drive, cloning repo, linking symlinks), add these cells:

---

### Cell 1: Install dependency

```python
!pip install -q editdistance
```

---

### Cell 2: Evaluate on Validation Set (raw model output)

```python
%cd /content/assamese-ocr-training/django_backend

!python evaluate.py \
    --img-dir data/val_real_sentences/images \
    --label-file data/val_real_sentences/labels/labels.txt \
    --checkpoint checkpoints/best_model_sentences.pth \
    --num-samples 10
```

---

### Cell 3: Evaluate WITH spell-check post-processing

```python
# Compare with vs without post-processing to see if spell checker helps
!python evaluate.py \
    --img-dir data/val_real_sentences/images \
    --label-file data/val_real_sentences/labels/labels.txt \
    --checkpoint checkpoints/best_model_sentences.pth \
    --post-process \
    --num-samples 10
```

---

### Cell 3.5: Evaluate WITH CTC Beam Search (Better Accuracy)

```python
# Use beam search decoding instead of greedy decoding
!python evaluate.py \
    --img-dir data/val_real_sentences/images \
    --label-file data/val_real_sentences/labels/labels.txt \
    --checkpoint checkpoints/best_model_sentences.pth \
    --beam-width 10 \
    --num-samples 10
```

---

### Cell 4: Evaluate on Test Set

```python
import os
test_labels = 'data/test_real_sentences/labels/labels.txt'

if os.path.exists(test_labels):
    !python evaluate.py \
        --img-dir data/test_real_sentences/images \
        --label-file {test_labels} \
        --checkpoint checkpoints/best_model_sentences.pth \
        --num-samples 10 \
        --output test_results.tsv
    print('\n✅ Detailed results saved to test_results.tsv')
else:
    print('❌ No test set found.')
    print('Generate one: python build_real_sentence_splits.py --test-count 1000')
```

---

### Cell 5: Single image prediction with ground truth

```python
# Quick check on one image
!python predict_cli.py \
    --image data/val_real_sentences/images/sentence_000000.png \
    --checkpoint checkpoints/best_model_sentences.pth \
    --ground-truth data/val_real_sentences/labels/sentence_000000.txt
```

---

### Cell 6: Save full results to Drive

```python
# Export detailed per-image results to Google Drive for later analysis
!python evaluate.py \
    --img-dir data/val_real_sentences/images \
    --label-file data/val_real_sentences/labels/labels.txt \
    --checkpoint checkpoints/best_model_sentences.pth \
    --output /content/drive/MyDrive/assamese_ocr_assets/val_results.tsv

print('Results saved to Drive!')
```

---

## What You'll See

The evaluation script outputs:

```
======================================================================
EVALUATION RESULTS
======================================================================
Checkpoint : checkpoints/best_model_sentences.pth
Dataset    : data/val_real_sentences/images (4000 samples)
Post-proc  : OFF
Time       : 45.2s (88.5 img/s)
----------------------------------------------------------------------
CER         : 0.1234  (12.34%)
WER         : 0.3456  (34.56%)
Exact Match : 0.2100  (21.00%)
======================================================================

--- Best 10 predictions ---
  ✅ [sentence_000042.png] CER=0.00%
     GT  : অসমীয়া ভাষা
     Pred: অসমীয়া ভাষা

--- Worst 10 predictions ---
  ❌ [sentence_001234.png] CER=85.71%
     GT  : দক্ষিণ-পূব এছিয়া
     Pred: দক্ষণ-পব এছযা
```

## Interpreting Results

| Metric | Good | Decent | Needs Work |
|--------|------|--------|------------|
| **CER** | < 5% | 5-15% | > 15% |
| **WER** | < 15% | 15-30% | > 30% |
| **Exact Match** | > 50% | 20-50% | < 20% |

> **Tip**: Run with and without `--post-process` to see if the spell checker is actually helping or hurting accuracy.
