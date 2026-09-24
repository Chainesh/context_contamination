"""
Diagnostic plots for the recency/span/distinguishability contamination
model. These answer one question: how much of the context is actually
being touched, and where. Two plots only — no result plots live here,
those belong in the main evaluation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

from contamination import ContaminationParams, apply_contamination, contamination_mask


def plot_contamination_coverage(
    context_length: int,
    span_values: list[float],
    recency_steps: int = 60,
    save_path: str | None = None,
):
    """
    Coverage heatmap: x = position in context, y = recency (swept
    continuously from 0 to 1), color = whether that position falls
    inside the corrupted block. One panel per span value.

    This is the primary "how much context is contaminated" plot: it
    shows the full recency sweep at once instead of stepping through
    frames.
    """
    recencies = np.linspace(0.0, 1.0, recency_steps)
    n_panels = len(span_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2), squeeze=False)
    axes = axes[0]

    for ax, span in zip(axes, span_values):
        coverage = np.zeros((recency_steps, context_length))
        for i, r in enumerate(recencies):
            coverage[i] = contamination_mask(context_length, r, span)

        im = ax.imshow(
            coverage,
            aspect="auto",
            origin="lower",
            extent=[0, context_length, 0.0, 1.0],
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        ax.axvline(context_length, color="tab:red", linewidth=1.2, linestyle="--")
        ax.set_title(f"span = {span:g}")
        ax.set_xlabel("context position (0 = oldest, L = forecast boundary)")

    axes[0].set_ylabel("recency  (0 = touches forecast boundary, 1 = touches start)")
    fig.suptitle(f"Contamination block coverage across recency sweep  (L={context_length})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_contamination_example(
    context: np.ndarray,
    donor_context: np.ndarray,
    recency: float,
    span: float,
    distinguishability: float,
    mechanism: str,
    rng: np.random.Generator,
    save_path: str | None = None,
):
    """
    Single-window sanity check: clean vs contaminated context overlaid,
    with the corrupted block shaded. Confirms the math matches intent
    for a specific (recency, span, distinguishability) setting.
    """
    params = ContaminationParams(
        recency=recency, span=span, distinguishability=distinguishability, mechanism=mechanism
    )
    contaminated = apply_contamination(context, donor_context, params, rng)
    mask = contamination_mask(len(context), recency, span)
    idx = np.arange(len(context))

    fig, ax = plt.subplots(figsize=(9, 4))
    if mask.any():
        start = idx[mask][0]
        end = idx[mask][-1] + 1
        ax.axvspan(start, end, color="tab:orange", alpha=0.15, label="contaminated block")

    ax.plot(idx, context, color="tab:blue", linewidth=1.4, label="clean context")
    ax.plot(idx, contaminated, color="tab:red", linewidth=1.2, linestyle="--", label="contaminated context")
    ax.axvline(len(context), color="black", linewidth=1, linestyle=":", label="forecast boundary")

    ax.set_title(
        f"{mechanism}  |  recency={recency:g}  span={span:g}  distinguishability={distinguishability:g}"
    )
    ax.set_xlabel("context position")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
