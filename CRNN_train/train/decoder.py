"""
decoder.py — Beam Search CTC decoder and greedy fallback.
"""
import torch
import numpy as np
from collections import defaultdict
from config import BLANK_IDX, IDX2CHAR


# ─── Greedy decoder ───────────────────────────────────────────────────────────

def greedy_decode(log_probs: torch.Tensor,
                  idx2char: dict = IDX2CHAR,
                  blank_idx: int = BLANK_IDX) -> list[str]:
    """
    Args:
        log_probs: [T, B, num_classes]  (log-softmax output)
    Returns:
        list of decoded strings, length B
    """
    probs = log_probs.exp()                          # [T, B, C]
    best  = probs.argmax(dim=2)                      # [T, B]
    best  = best.permute(1, 0).cpu().numpy()         # [B, T]

    results = []
    for seq in best:
        chars = []
        prev = None
        for idx in seq:
            if idx != prev:
                if idx != blank_idx:
                    chars.append(idx2char.get(int(idx), ''))
            prev = idx
        results.append(''.join(chars))
    return results


# ─── Beam Search decoder ──────────────────────────────────────────────────────

def beam_search_decode(log_probs: torch.Tensor,
                       beam_width: int = 5,
                       blank_idx:  int = BLANK_IDX,
                       idx2char:   dict = IDX2CHAR) -> list[str]:
    """
    CTC Beam Search decoding (no language model — pure acoustic).

    Args:
        log_probs : [T, B, num_classes]  (log-softmax)
        beam_width: k  (number of beams to keep)
        blank_idx : CTC blank token index
        idx2char  : mapping from index to character

    Returns:
        list[str] — best decoded plates, one per batch element
    """
    probs = log_probs.exp().cpu().numpy()  # [T, B, C]
    T, B, C = probs.shape

    results = []
    for b in range(B):
        p = probs[:, b, :]   # [T, C]
        decoded = _beam_search_single(p, beam_width, blank_idx, idx2char)
        results.append(decoded)
    return results


def _beam_search_single(probs: np.ndarray,
                         beam_width: int,
                         blank_idx: int,
                         idx2char: dict) -> str:
    """
    Beam search for a single sequence.
    probs: [T, C]  (probabilities, NOT log)
    """
    T, C = probs.shape

    # Beam state: dict { prefix_tuple: (prob_blank, prob_non_blank) }
    # prob_blank    = cumulative prob that the beam ends with blank
    # prob_non_blank= cumulative prob that the beam ends with non-blank
    NEG_INF = -1e30

    # Initialise
    beams = {(): (1.0, 0.0)}   # empty prefix

    for t in range(T):
        new_beams = defaultdict(lambda: (0.0, 0.0))

        # Prune to top beam_width
        beams = _prune(beams, beam_width)

        for prefix, (pb, pnb) in beams.items():
            for c in range(C):
                p_c = float(probs[t, c])
                if c == blank_idx:
                    # Extend with blank: prefix stays the same
                    old_pb, old_pnb = new_beams[prefix]
                    new_beams[prefix] = (old_pb + (pb + pnb) * p_c, old_pnb)
                else:
                    # Extend with non-blank character c
                    new_prefix = prefix + (c,)
                    old_pb, old_pnb = new_beams[new_prefix]

                    if prefix and prefix[-1] == c:
                        # Same char as last in prefix: only blank can connect
                        new_beams[new_prefix] = (old_pb, old_pnb + pb * p_c)
                    else:
                        new_beams[new_prefix] = (old_pb, old_pnb + (pb + pnb) * p_c)

        beams = dict(new_beams)

    # Best beam
    best_prefix = max(beams, key=lambda p: sum(beams[p]))
    return ''.join(idx2char.get(c, '') for c in best_prefix
                   if c != blank_idx)


def _prune(beams: dict, beam_width: int) -> dict:
    """Keep top-k beams by total probability."""
    scored = sorted(beams.items(), key=lambda kv: sum(kv[1]), reverse=True)
    return dict(scored[:beam_width])
