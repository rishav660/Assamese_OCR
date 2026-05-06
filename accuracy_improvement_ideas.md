# 🎯 Assamese OCR — Accuracy Improvement Ideas

## Current Architecture Summary

| Component | What You Have |
|---|---|
| **Model** | CRNN: 7-layer CNN (with BatchNorm) → 3-layer BiLSTM (512 hidden) → Linear |
| **Loss** | CTC Loss (blank = last index) |
| **Decoding** | Greedy (argmax → collapse repeats & blanks) |
| **Post-processing** | Peter Norvig-style spell checker (edit distance 1 only, corpus = `as-wiki-2021.txt`) |
| **Data** | Synthetically rendered sentences from wiki corpus, 3 Noto Bengali fonts, 15K train / 4K val |
| **Augmentation** | Brightness, contrast, sharpness, rotation ±2°, blur, noise, erosion/dilation |
| **Image size** | Fixed 32×320 in dataset (but transform resizes to 32×512) |
| **Char map** | 121 characters (Assamese script + Latin + punctuation + digits) |

---

## 🔴 Critical Issues Found

> [!WARNING]
> **Image size mismatch**: [dataset.py](file:///c:/Users/diksh/OneDrive/Desktop/Assamese_OCR/django_backend/dataset.py#L108-L109) hard-codes resize to `32×320`, but [data_augmentation.py](file:///c:/Users/diksh/OneDrive/Desktop/Assamese_OCR/django_backend/data_augmentation.py#L100) resizes to `32×512`, and `predict_cli.py` also uses `32×512`. This means **training images are 320px wide but inference images are 512px wide** — a major train/test mismatch that directly hurts accuracy.

> [!WARNING]
> **Double resize**: In `dataset.py`, the transform is applied first (resizes to 512), then the image is converted back to PIL and resized again to 320. The augmentation → tensor → PIL → resize → tensor pipeline also destroys quality.

---

## 📊 Improvement Ideas (Prioritized by Impact/Effort)

### 1. 🔧 Fix the Image Size Mismatch (Critical — FREE Accuracy)

**Impact: 🔴 Very High | Effort: 🟢 Trivial**

The training pipeline resizes to `32×320` in `dataset.py` L108-109, but inference uses `32×512`. This means the model never sees images at the resolution it's tested on.

**Fix:** Make `dataset.py` use the same `32×512` as everything else, or better — make it configurable:

```python
# dataset.py — remove the hard-coded resize, let the transform handle it
# Delete lines 107-114 and rely solely on the transform pipeline
``` 

Also fix the double-transform issue: the transform already converts to tensor, then `__getitem__` converts back to PIL to resize again. This should be a single pipeline.

---

### 2. 🧠 Add a Language Model for Beam Search Decoding

**Impact: 🔴 Very High | Effort: 🟡 Medium**

Your current decoding is **greedy argmax**, which picks the single best character at each timestep independently. This ignores all linguistic context. A **CTC beam search with a language model** can dramatically improve accuracy (typically **5-15% CER improvement**).

**Two-level approach:**

#### a) Character-level N-gram LM (Simpler)
Build a character trigram/4-gram model from `as-wiki-2021.txt` and use it to rescore CTC beams:

```python
# language_model.py
import math
from collections import Counter, defaultdict

class CharNgramLM:
    """Character-level n-gram language model for Assamese"""
    
    def __init__(self, corpus_path, n=4, smoothing=0.01):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self._build(corpus_path)
    
    def _build(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                text = '<' + line.strip() + '>'  # BOS/EOS markers
                for i in range(len(text) - self.n + 1):
                    context = text[i:i+self.n-1]
                    char = text[i+self.n-1]
                    self.ngram_counts[context][char] += 1
                    self.context_counts[context] += 1
    
    def log_prob(self, char, context):
        """P(char | context) with add-k smoothing"""
        context = context[-(self.n-1):]
        count = self.ngram_counts[context][char]
        total = self.context_counts[context]
        vocab_size = len(set(c for counts in self.ngram_counts.values() for c in counts))
        prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
        return math.log(prob)
    
    def score_text(self, text):
        """Score an entire string"""
        text = '<' + text + '>'
        score = 0.0
        for i in range(self.n - 1, len(text)):
            context = text[i-self.n+1:i]
            score += self.log_prob(text[i], context)
        return score
```

#### b) Word-level N-gram LM (More Powerful)
Use [KenLM](https://github.com/kpu/kenlm) to train a word-level trigram model on the wiki corpus, then integrate with CTC beam search using [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode):

```python
# With pyctcdecode — drop-in replacement for greedy decode
from pyctcdecode import build_ctcdecoder

labels = [''] + [idx_to_char[i] for i in range(1, len(idx_to_char)+1)]  
decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path="assamese_lm.arpa",  # trained with KenLM
    alpha=0.5,  # LM weight
    beta=1.0,   # word insertion bonus
)

# At inference:
logits = model(image_tensor)  # (T, 1, C)
text = decoder.decode(logits.squeeze(1).cpu().numpy())
```

> [!TIP]
> `pyctcdecode` is the single highest-impact change you can make. It's used by most production OCR systems and Hugging Face's Wav2Vec2 for speech.

---

### 3. 📝 Upgrade the Spell Checker

**Impact: 🟡 Medium-High | Effort: 🟢 Low-Medium**

Your current spell checker has several limitations:

| Issue | Impact |
|---|---|
| Edit distance 1 only (edits2 is commented out) | Misses corrections that need 2 edits |
| No context awareness — each word corrected independently | Can "correct" a word into the wrong word |
| Norvig's `letters` string may be incomplete for Assamese | Missing conjuncts, nukta forms |
| No confidence gating — always tries to correct unknown words | Over-corrects proper nouns, rare valid words |

**Improvements:**

```python
# A) Enable edit distance 2 with pruning (not brute force)
def candidates(self, word):
    if word in self.words:
        return {word}
    ed1 = self.known(self.edits1(word))
    if ed1:
        return ed1
    # Only try edits2 for short words where it's tractable
    if len(word) <= 8:
        ed2 = self.known(self.edits2(word))
        if ed2:
            return ed2
    return {word}

# B) Add confidence gating — don't correct if the OCR output 
#    has a high character-level confidence
def correction_with_confidence(self, word, avg_char_confidence=1.0):
    if word in self.words:
        return word
    if avg_char_confidence > 0.95:
        return word  # OCR is very confident, don't override
    return self.correction(word)

# C) Add context-aware correction using bigrams
def correct_sentence_contextual(self, sentence):
    """Use word bigram probabilities to pick best correction"""
    words = sentence.split()
    corrected = []
    for i, word in enumerate(words):
        candidates = self.candidates(word)
        if len(candidates) <= 1:
            corrected.append(self.correction(word))
        else:
            # Score candidates by P(word) * P(word | prev_word)
            prev = corrected[-1] if corrected else '<S>'
            best = max(candidates, 
                      key=lambda c: self.P(c) * self.bigram_P(c, prev))
            corrected.append(best)
    return ' '.join(corrected)
```

---

### 4. 🏗️ Upgrade Model Architecture

**Impact: 🟡 Medium-High | Effort: 🟡 Medium**

Your CRNN is a solid baseline, but there are drop-in upgrades:

#### a) Add Attention (Replace LSTM → Transformer Encoder)
```python
class CRNN_Attention(nn.Module):
    def __init__(self, img_height, nn_classes, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.cnn = ...  # same CNN backbone
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Linear(d_model, nn_classes)
```

#### b) Use a Pretrained Backbone (ResNet-18 or EfficientNet)
Replace your hand-crafted CNN with a pretrained feature extractor. Even though it's trained on natural images, the low-level features (edges, curves) transfer well to text:

```python
import torchvision.models as models

class CRNN_Pretrained(nn.Module):
    def __init__(self, nn_classes):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Modify first conv for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove avg pool and FC
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        # Adaptive pool to collapse height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(512, nn_classes)
```

#### c) Use TrOCR / PARSeq (State-of-the-Art)
These are encoder-decoder Transformer models specifically designed for OCR. You can fine-tune them on Assamese:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Fine-tune on your Assamese data
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
```

> [!NOTE]
> TrOCR would require the most effort but could yield the highest accuracy. It's what production systems like Azure Document Intelligence use internally.

---

### 5. 📊 Scale Up Training Data

**Impact: 🟡 Medium-High | Effort: 🟢 Low**

You're training on **15K synthetic sentences** with only **3 fonts**. This is quite limited.

**Quick wins:**
- **More fonts**: Download 10-20 more Bengali/Assamese-compatible fonts from Google Fonts (Tiro Bangla, Anek Bangla, Noto Sans Bengali variations, Hind Siliguri, etc.)
- **More sentences**: Your wiki corpus is ~23 MB. You're only using 15K sentences. Increase to 50K-100K.
- **Vary rendering parameters**: Randomize font size (36-64px), line spacing, padding, and background color during generation.
- **Add real-world images**: If you have access to scanned Assamese documents, newspapers, or book pages, even 500-1000 real images with manual labels would significantly help.

```python
# In generate_real_sentence_data.py — add rendering variations
import random

def render_sentence_image(self, text):
    font_path = random.choice(self.font_paths)
    font_size = random.randint(36, 64)  # Vary size
    
    # Random background shade (not always pure white)
    bg_color = random.randint(230, 255)
    # Random text color (not always pure black)  
    text_color = random.randint(0, 40)
    
    img = Image.new('L', (img_width, img_height), color=bg_color)
    draw.text((x, y), text, fill=text_color, font=font)
```

---

### 6. 🔍 Better Augmentation

**Impact: 🟡 Medium | Effort: 🟢 Low**

Your augmentation is decent but missing some important OCR-specific transforms:

```python
# Add these to OCRAugmentation

# 8. Perspective transform (simulates camera angle)
if random.random() < self.p * 0.3:
    width, height = img.size
    coeffs = [1 + random.uniform(-0.05, 0.05) for _ in range(8)]
    img = img.transform((width, height), Image.PERSPECTIVE, coeffs)

# 9. Random padding/cropping (simulates imperfect bounding boxes)
if random.random() < self.p * 0.4:
    pad_top = random.randint(0, 5)
    pad_bottom = random.randint(0, 5)
    pad_left = random.randint(0, 10)
    pad_right = random.randint(0, 10)
    img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=255)

# 10. JPEG compression artifacts (simulates real-world images)
if random.random() < self.p * 0.3:
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=random.randint(40, 85))
    buffer.seek(0)
    img = Image.open(buffer).convert('L')

# 11. Random line/scratch artifacts
if random.random() < self.p * 0.1:
    draw = ImageDraw.Draw(img)
    x1 = random.randint(0, img.width)
    y1 = random.randint(0, img.height)
    x2 = random.randint(0, img.width)
    draw.line([(x1, y1), (x2, y1 + random.randint(-3, 3))], fill=128, width=1)
```

---

### 7. 🎯 CTC Beam Search (Without LM)

**Impact: 🟡 Medium | Effort: 🟢 Low**

Even **without** a language model, beam search is better than greedy decoding. It explores multiple hypotheses:

```python
import torch

def beam_search_decode(log_probs, beam_width=10, blank_idx=121):
    """
    CTC beam search decoding (no LM).
    log_probs: (T, C) tensor of log probabilities
    """
    T, C = log_probs.shape
    beams = [{'seq': [], 'log_p_blank': 0.0, 'log_p_non_blank': float('-inf')}]
    
    for t in range(T):
        new_beams = {}
        for beam in beams:
            for c in range(C):
                log_p = log_probs[t, c].item()
                # ... (standard CTC prefix beam search algorithm)
        beams = sorted(new_beams.values(), key=lambda b: ..., reverse=True)[:beam_width]
    
    # Return best sequence
    best = max(beams, key=lambda b: ...)
    return ''.join(idx_to_char.get(idx, '') for idx in best['seq'])
```

Or simply use the `ctcdecode` library:
```bash
pip install pyctcdecode
```

---

### 8. 📐 Dynamic Image Width (Aspect-Ratio Preserving)

**Impact: 🟡 Medium | Effort: 🟡 Medium**

Currently all images are squashed to a fixed width (320 or 512), regardless of the actual text length. Short words get stretched, long sentences get compressed. This hurts accuracy for both extremes.

**Fix:** Preserve aspect ratio by scaling height to 32 and letting width vary, then pad to the batch's max width:

```python
# In dataset.py
def resize_preserve_aspect(image, target_height=32, max_width=512):
    w, h = image.size
    ratio = target_height / h
    new_w = min(int(w * ratio), max_width)
    image = image.resize((new_w, target_height), Image.BILINEAR)
    
    # Pad to max_width with white
    if new_w < max_width:
        padded = Image.new('L', (max_width, target_height), 255)
        padded.paste(image, (0, 0))
        return padded
    return image
```

---

### 9. 🏋️ Training Strategy Improvements

**Impact: 🟡 Medium | Effort: 🟢 Low**

Several quick training tweaks that help:

| Tweak | What to Change |
|---|---|
| **Cosine annealing** | Replace `ReduceLROnPlateau` with `CosineAnnealingWarmRestarts` — smoother LR schedule |
| **Label smoothing for CTC** | Not directly supported, but you can add a small uniform noise to the log-softmax output |
| **Gradient accumulation** | Simulate larger batch sizes (64 or 128) by accumulating gradients over 2-4 steps |
| **Mixed precision (AMP)** | `torch.cuda.amp` → ~2x faster training, same accuracy, allows larger batches |
| **Curriculum learning** | Train on short sentences first (3-5 words), then gradually increase to full sentences |

```python
# Mixed precision training example
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels, input_lengths, target_lengths)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 10. 📏 Add Proper Evaluation Metrics

**Impact: 🟢 Indirect but Important | Effort: 🟢 Low**

You're currently tracking only **CTC loss**. You should also track:

- **CER** (Character Error Rate) — the standard OCR metric
- **WER** (Word Error Rate) — word-level accuracy
- **Exact match rate** — % of sentences decoded perfectly

```python
import editdistance

def compute_cer(predictions, targets):
    """Character Error Rate"""
    total_chars = 0
    total_errors = 0
    for pred, target in zip(predictions, targets):
        total_errors += editdistance.eval(pred, target)
        total_chars += len(target)
    return total_errors / total_chars if total_chars > 0 else 0.0

def compute_wer(predictions, targets):
    """Word Error Rate"""
    total_words = 0
    total_errors = 0
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_errors += editdistance.eval(pred_words, target_words)
        total_words += len(target_words)
    return total_errors / total_words if total_words > 0 else 0.0
```

---

## 📋 Priority Implementation Order

| Priority | Idea | Impact | Effort | Notes |
|:---:|---|:---:|:---:|---|
| **1** | Fix image size mismatch (§1) | 🔴 Very High | 🟢 5 min | **Do this first — it's a bug** |
| **2** | CTC beam search + char LM (§2a + §7) | 🔴 Very High | 🟡 1-2 days | Biggest single accuracy gain |
| **3** | Scale up training data & fonts (§5) | 🟡 High | 🟢 Few hours | More data = more robust |
| **4** | Upgrade spell checker (§3) | 🟡 Medium-High | 🟢 Half day | Context-aware corrections |
| **5** | Add CER/WER metrics (§10) | 🟢 Essential | 🟢 1 hour | You need this to measure everything else |
| **6** | Better augmentation (§6) | 🟡 Medium | 🟢 1-2 hours | Especially JPEG artifacts & perspective |
| **7** | Dynamic image width (§8) | 🟡 Medium | 🟡 Half day | Helps with variable-length text |
| **8** | Training tweaks (§9) | 🟡 Medium | 🟢 Few hours | AMP + cosine annealing |
| **9** | Attention / Transformer (§4a-b) | 🟡 Medium-High | 🟡 1-2 days | Architecture upgrade |
| **10** | TrOCR fine-tuning (§4c) | 🔴 Highest ceiling | 🔴 3-5 days | Nuclear option — state-of-the-art |

---

> [!IMPORTANT]
> **Start with items 1 and 5 (the bug fix + metrics).** Without proper metrics, you can't measure whether other changes help. Then move to beam search + LM (item 2), which is typically the single biggest accuracy boost for CTC-based OCR systems.

Which of these would you like me to implement first?
