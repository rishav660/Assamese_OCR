"""
OCR Evaluation Metrics — CER, WER, and Exact Match Rate

Usage:
    from metrics import compute_cer, compute_wer, compute_exact_match, OCRMetrics

    # Individual functions
    cer = compute_cer(["predicted"], ["ground truth"])
    wer = compute_wer(["predicted"], ["ground truth"])

    # Or use the accumulator for epoch-level tracking
    m = OCRMetrics()
    m.update(predictions=["pred1", "pred2"], targets=["gt1", "gt2"])
    print(m.summary())
"""


def _edit_distance(seq1, seq2):
    """
    Compute Levenshtein edit distance between two sequences.
    Works for both character lists and word lists.
    Pure-Python implementation (no external dependency required,
    but falls back to the `editdistance` C extension when available).
    """
    try:
        import editdistance
        return editdistance.eval(seq1, seq2)
    except ImportError:
        pass

    # Wagner-Fischer algorithm
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            prev, dp[j] = dp[j], min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + cost,     # substitution
            )
    return dp[m]


def compute_cer(predictions, targets):
    """
    Character Error Rate (CER).

    CER = (total character-level edit distance) / (total reference characters)

    Args:
        predictions: list of predicted strings
        targets:     list of ground-truth strings

    Returns:
        float — CER in [0, ∞) where 0 is perfect
    """
    total_errors = 0
    total_chars = 0
    for pred, target in zip(predictions, targets):
        total_errors += _edit_distance(list(pred), list(target))
        total_chars += len(target)
    return total_errors / total_chars if total_chars > 0 else 0.0


def compute_wer(predictions, targets):
    """
    Word Error Rate (WER).

    WER = (total word-level edit distance) / (total reference words)

    Args:
        predictions: list of predicted strings
        targets:     list of ground-truth strings

    Returns:
        float — WER in [0, ∞) where 0 is perfect
    """
    total_errors = 0
    total_words = 0
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_errors += _edit_distance(pred_words, target_words)
        total_words += len(target_words)
    return total_errors / total_words if total_words > 0 else 0.0


def compute_exact_match(predictions, targets):
    """
    Exact Match Rate — fraction of predictions that perfectly match the target.

    Args:
        predictions: list of predicted strings
        targets:     list of ground-truth strings

    Returns:
        float — ratio in [0, 1] where 1 is perfect
    """
    if not predictions:
        return 0.0
    matches = sum(1 for p, t in zip(predictions, targets) if p == t)
    return matches / len(predictions)


class OCRMetrics:
    """
    Accumulator for tracking CER / WER / Exact Match across batches.

    Example:
        metrics = OCRMetrics()
        for batch in val_loader:
            preds = decode(model(batch))
            targets = get_targets(batch)
            metrics.update(preds, targets)
        print(metrics.summary())
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._char_errors = 0
        self._char_total = 0
        self._word_errors = 0
        self._word_total = 0
        self._exact_matches = 0
        self._sample_count = 0

    def update(self, predictions, targets):
        """Add a batch of predictions and targets."""
        for pred, target in zip(predictions, targets):
            # Character level
            self._char_errors += _edit_distance(list(pred), list(target))
            self._char_total += len(target)

            # Word level
            pred_words = pred.split()
            target_words = target.split()
            self._word_errors += _edit_distance(pred_words, target_words)
            self._word_total += len(target_words)

            # Exact match
            if pred == target:
                self._exact_matches += 1
            self._sample_count += 1

    @property
    def cer(self):
        return self._char_errors / self._char_total if self._char_total > 0 else 0.0

    @property
    def wer(self):
        return self._word_errors / self._word_total if self._word_total > 0 else 0.0

    @property
    def exact_match(self):
        return self._exact_matches / self._sample_count if self._sample_count > 0 else 0.0

    def summary(self):
        return (
            f"CER: {self.cer:.4f} ({self.cer * 100:.2f}%) | "
            f"WER: {self.wer:.4f} ({self.wer * 100:.2f}%) | "
            f"Exact Match: {self.exact_match:.4f} ({self.exact_match * 100:.2f}%) | "
            f"Samples: {self._sample_count}"
        )

    def as_dict(self):
        return {
            "cer": self.cer,
            "wer": self.wer,
            "exact_match": self.exact_match,
            "samples": self._sample_count,
        }
