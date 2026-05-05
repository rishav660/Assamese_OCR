"""
CTC Beam Search Decoder (no language model).

Implements the CTC prefix beam search algorithm from:
  Hannun, "Sequence Modeling With CTC", Distill 2017.

Compared to greedy decoding (argmax at each timestep), beam search
explores multiple hypotheses simultaneously, which often produces
better results — especially for ambiguous characters.

Usage:
    from beam_search import ctc_beam_search, ctc_beam_search_batch

    # Single sequence: log_probs shape (T, C)
    text = ctc_beam_search(log_probs, idx_to_char, blank_idx=121, beam_width=10)

    # Batch: log_probs shape (T, B, C)
    texts = ctc_beam_search_batch(log_probs, input_lengths, idx_to_char, blank_idx=121)
"""

import math
from collections import defaultdict


def _log_sum_exp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search(log_probs, idx_to_char, blank_idx, beam_width=10):
    """
    CTC prefix beam search for a single sequence.

    Args:
        log_probs:   (T, C) tensor or numpy array of log probabilities
                     (output of log_softmax)
        idx_to_char: dict mapping index -> character
        blank_idx:   index of the CTC blank token
        beam_width:  number of beams to keep at each timestep

    Returns:
        str — best decoded string
    """
    T, C = log_probs.shape
    NEG_INF = float("-inf")

    # Each beam is keyed by its label prefix (tuple of ints).
    # Values: (log_p_blank, log_p_non_blank)
    #   log_p_blank     = log P(prefix, ends with blank)
    #   log_p_non_blank = log P(prefix, ends with non-blank)
    beams = {(): (0.0, NEG_INF)}  # start: empty prefix, blank prob = 1

    for t in range(T):
        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

        # Prune to top beam_width beams by total probability
        sorted_beams = sorted(
            beams.items(),
            key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
            reverse=True,
        )[:beam_width]

        for prefix, (p_b, p_nb) in sorted_beams:
            # Total log probability of this prefix
            p_total = _log_sum_exp(p_b, p_nb)

            for c in range(C):
                log_p = log_probs[t, c].item() if hasattr(log_probs, 'item') else float(log_probs[t, c])

                if c == blank_idx:
                    # Blank extends the prefix without changing it
                    old_b, old_nb = new_beams[prefix]
                    new_beams[prefix] = (
                        _log_sum_exp(old_b, p_total + log_p),
                        old_nb,
                    )
                else:
                    # Non-blank character
                    end_t = prefix[-1] if prefix else None

                    if c == end_t:
                        # Same character as last in prefix:
                        # - Can extend only via the blank path (collapse prevention)
                        # - Or start a new repeated char via blank path
                        new_prefix = prefix + (c,)

                        # Extend same char (must come after blank)
                        old_b, old_nb = new_beams[prefix]
                        new_beams[prefix] = (
                            old_b,
                            _log_sum_exp(old_nb, p_nb + log_p),
                        )

                        # New repeated char (from blank path)
                        old_b2, old_nb2 = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b2,
                            _log_sum_exp(old_nb2, p_b + log_p),
                        )
                    else:
                        # Different character — extend prefix
                        new_prefix = prefix + (c,)
                        old_b, old_nb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            old_b,
                            _log_sum_exp(old_nb, p_total + log_p),
                        )

        beams = dict(new_beams)

    # Find the best beam
    best_prefix = max(
        beams.items(),
        key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
    )[0]

    return "".join(idx_to_char.get(idx, "") for idx in best_prefix)


def ctc_beam_search_batch(log_probs, input_lengths, idx_to_char, blank_idx,
                          beam_width=10):
    """
    CTC beam search for a batch of sequences.

    Args:
        log_probs:      (T, B, C) tensor of log probabilities
        input_lengths:  (B,) tensor of sequence lengths per sample
        idx_to_char:    dict mapping index -> character
        blank_idx:      index of the CTC blank token
        beam_width:     number of beams

    Returns:
        list[str] — decoded strings for each sample in the batch
    """
    B = log_probs.size(1)
    results = []
    for i in range(B):
        seq_len = input_lengths[i].item()
        single_log_probs = log_probs[:seq_len, i, :]  # (T_i, C)
        text = ctc_beam_search(single_log_probs, idx_to_char, blank_idx,
                               beam_width=beam_width)
        results.append(text)
    return results


def greedy_decode(log_probs, idx_to_char, blank_idx):
    """
    Standard CTC greedy decode for a single sequence (for comparison).

    Args:
        log_probs:   (T, C) tensor of log probabilities
        idx_to_char: dict mapping index -> character
        blank_idx:   index of the CTC blank token

    Returns:
        str — decoded string
    """
    import torch
    indices = torch.argmax(log_probs, dim=1)
    prev = -1
    chars = []
    for idx in indices:
        idx = idx.item()
        if idx != prev and idx != blank_idx:
            ch = idx_to_char.get(idx, "")
            if ch:
                chars.append(ch)
        prev = idx
    return "".join(chars)
