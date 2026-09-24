"""
Controlled Context Contamination — three-degree-of-freedom model.

Corruption is defined by a single contiguous block in the context,
positioned and scaled by three independent knobs:

    recency          r in [0, 1]   0 = block touches the end of context
                                    (closest to the forecast boundary)
                                    1 = block touches the start of context
    span             s in (0, 1]   fraction of context length in the block
    distinguishability d in [0, 1] how far the block's values are pushed
                                    from the true signal (0 = untouched,
                                    1 = fully replaced / max noise)

This replaces the old (type, mode, lambda) parameterization. "type" is
now just which corruption *mechanism* fills the block (additive_noise,
signal_mixing, segment_replacement); recency/span jointly replace the
old early/middle/recent/random modes with a continuous position.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContaminationParams:
    recency: float            # 0..1, 0 = nearest to prediction step
    span: float                # 0..1, fraction of context length
    distinguishability: float  # 0..1, corruption intensity in the block
    mechanism: str = "signal_mixing"  # additive_noise | signal_mixing | segment_replacement


def block_bounds(context_length: int, recency: float, span: float) -> tuple[int, int]:
    """
    Compute [start, end) indices of the contaminated block.

    recency=0 anchors the block so it ends exactly at context_length
    (touches the last observed timestep, nearest the forecast horizon).
    recency=1 anchors the block so it starts exactly at 0.
    Intermediate recency linearly interpolates the anchor between
    these two extremes, sliding the block back through the context
    while keeping its length fixed.
    """
    if not (0.0 <= recency <= 1.0):
        raise ValueError(f"recency must be in [0,1], got {recency}")
    if not (0.0 < span <= 1.0):
        raise ValueError(f"span must be in (0,1], got {span}")

    k = max(1, round(span * context_length))
    slack = context_length - k  # room the block has to slide
    end = context_length - round(recency * slack)
    start = end - k
    start = max(0, start)
    end = min(context_length, start + k)
    return start, end


def apply_contamination(
    context: np.ndarray,
    donor_context: np.ndarray,
    params: ContaminationParams,
    rng: np.random.Generator,
    noise_scale: float | None = None,
) -> np.ndarray:
    """
    Apply corruption to a single univariate context [L] inside the block
    defined by (recency, span). Returns a new array; input is untouched.
    """
    L = len(context)
    if donor_context.shape != context.shape:
        raise ValueError("donor_context must match context shape")

    start, end = block_bounds(L, params.recency, params.span)
    x_tilde = context.copy()
    d = params.distinguishability

    if d <= 0.0 or start >= end:
        return x_tilde

    block = context[start:end]
    donor_block = donor_context[start:end]

    if params.mechanism == "additive_noise":
        scale = noise_scale if noise_scale is not None else (np.std(block) + 1e-8)
        eps = rng.standard_normal(size=block.shape)
        x_tilde[start:end] = block + d * scale * eps

    elif params.mechanism == "signal_mixing":
        x_tilde[start:end] = (1.0 - d) * block + d * donor_block

    elif params.mechanism == "segment_replacement":
        # d controls what fraction of the block's positions get fully
        # swapped for donor values (d=1 -> entire block replaced).
        k = end - start
        n_replace = round(d * k)
        if n_replace > 0:
            idx = rng.choice(k, size=n_replace, replace=False)
            sub = block.copy()
            sub[idx] = donor_block[idx]
            x_tilde[start:end] = sub
    else:
        raise ValueError(f"unknown mechanism: {params.mechanism}")

    return x_tilde


def apply_contamination_batch(
    contexts: np.ndarray,
    donor_contexts: np.ndarray,
    params: ContaminationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """contexts, donor_contexts: [B, L] -> contaminated [B, L]"""
    out = np.empty_like(contexts)
    for i in range(contexts.shape[0]):
        out[i] = apply_contamination(contexts[i], donor_contexts[i], params, rng)
    return out


def contamination_mask(context_length: int, recency: float, span: float) -> np.ndarray:
    """Boolean [L] mask, True where the block touches (used for plotting)."""
    start, end = block_bounds(context_length, recency, span)
    mask = np.zeros(context_length, dtype=bool)
    mask[start:end] = True
    return mask


def choose_donor_indices(num_windows: int, rng: np.random.Generator) -> np.ndarray:
    """donor_indices[i] != i, fixed for a given rng state so all models see the same donors."""
    donor_indices = np.empty(num_windows, dtype=np.int64)
    for i in range(num_windows):
        j = rng.integers(0, num_windows - 1)
        donor_indices[i] = j if j < i else j + 1
    return donor_indices
