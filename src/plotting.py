import matplotlib.pyplot as plt


def plot_degradation_vs_distinguishability(df, mechanism, span, primary_metric="mae", save_path=None):
    sub = df[(df["mechanism"] == mechanism) & (df["span"] == span)]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (model, recency), group in sub.groupby(["model", "recency"]):
        group = group.sort_values("distinguishability")
        ax.plot(group["distinguishability"], group[primary_metric], marker="o", markersize=3,
                label=f"{model}  recency={recency:g}")

    ax.set_xlabel("distinguishability")
    ax.set_ylabel(primary_metric.upper())
    ax.set_title(f"{mechanism}  |  span={span:g}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_recency_sensitivity(df, mechanism, span, primary_metric="mae", save_path=None):
    sub = df[(df["mechanism"] == mechanism) & (df["span"] == span)]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, group in sub.groupby("model"):
        agg = group.groupby("recency")["relative_degradation"].max().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=model)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("recency  (0 = touches forecast boundary, 1 = touches start of context)")
    ax.set_ylabel(f"max relative degradation ({primary_metric})")
    ax.set_title(f"Recency sensitivity  |  {mechanism}  span={span:g}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_distinguishability_star_comparison(cct_df, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cct_df = cct_df.copy()
    cct_df["setting"] = cct_df["mechanism"] + " r=" + cct_df["recency"].astype(str) + " s=" + cct_df["span"].astype(str)

    models = cct_df["model"].unique()
    settings = cct_df["setting"].unique()
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        sub = cct_df[cct_df["model"] == model].set_index("setting").reindex(settings)
        x = range(len(settings))
        ax.bar([xi + i * width for xi in x], sub["distinguishability_star"], width=width, label=model)

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in range(len(settings))])
    ax.set_xticklabels(settings, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("distinguishability*  (contamination tolerance)")
    ax.set_title("Context contamination tolerance by setting")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
