"""
Beam Search Decoder for CTC output.
Implements CTC-compatible beam search decoding.
"""

import torch
import numpy as np
from collections import defaultdict


def beam_search_decode(log_probs, beam_width=5, blank_idx=0):
    """
    CTC Beam Search Decoder.

    Args:
        log_probs: (T, num_classes) log probabilities for a single sample
        beam_width: number of beams to keep
        blank_idx: index of CTC blank token (last class)

    Returns:
        best_sequence: list of character indices (without blanks/duplicates)
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()

    T, C = log_probs.shape
    blank_idx = C - 1  # CTC blank is the last class

    # Initialize beams: (prefix, last_char) -> log_probability
    # prefix is a tuple of character indices (without blanks)
    beams = {((), None): 0.0}

    for t in range(T):
        new_beams = defaultdict(lambda: float('-inf'))

        for (prefix, last_char), log_prob in beams.items():
            # For each class at this timestep
            top_k = min(beam_width * 2, C)
            top_indices = np.argsort(log_probs[t])[-top_k:]

            for c in top_indices:
                c_log_prob = log_probs[t, c]
                new_log_prob = log_prob + c_log_prob

                if c == blank_idx:
                    # Blank: prefix stays the same
                    key = (prefix, None)
                    new_beams[key] = np.logaddexp(new_beams[key], new_log_prob)
                elif c == last_char:
                    # Same char as last non-blank: collapse (don't extend)
                    key = (prefix, c)
                    new_beams[key] = np.logaddexp(new_beams[key], new_log_prob)
                else:
                    # New character: extend prefix
                    new_prefix = prefix + (c,)
                    key = (new_prefix, c)
                    new_beams[key] = np.logaddexp(new_beams[key], new_log_prob)

        # Prune to top beam_width beams
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
        beams = dict(sorted_beams[:beam_width])

    # Return best sequence
    best_beam = max(beams.items(), key=lambda x: x[1])
    best_prefix = best_beam[0][0]
    return list(best_prefix)


def beam_search_decode_batch(log_probs_batch, beam_width=5):
    """
    Batch beam search decoding.

    Args:
        log_probs_batch: (T, B, C) log probabilities
        beam_width: beam width

    Returns:
        list of decoded sequences (list of list of ints)
    """
    if isinstance(log_probs_batch, torch.Tensor):
        log_probs_batch = log_probs_batch.detach().cpu().numpy()

    T, B, C = log_probs_batch.shape
    results = []
    for b in range(B):
        seq = beam_search_decode(log_probs_batch[:, b, :], beam_width=beam_width)
        results.append(seq)
    return results


def greedy_decode(log_probs, blank_idx=None):
    """
    Greedy CTC decoder (fallback).

    Args:
        log_probs: (T, num_classes) log probabilities

    Returns:
        decoded sequence (list of ints)
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()

    T, C = log_probs.shape
    if blank_idx is None:
        blank_idx = C - 1

    best_path = np.argmax(log_probs, axis=1)

    # Remove consecutive duplicates and blanks
    decoded = []
    prev = -1
    for idx in best_path:
        if idx != blank_idx and idx != prev:
            decoded.append(idx)
        prev = idx

    return decoded
