"""Figures for the low-N benchmark.

House style: log-x learning curves, mean over seeds with a +/- 1 s.d. band,
one colour per feature set held constant across every panel in the repo so
that curves can be compared by eye between figures.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES

COLORS = {
    "random": "#9aa0a6",
    "aac": "#7fb3d5",
    "biophys": "#48a999",
    "cheap": "#1f4e79",
    "esm2": "#e06c2b",
    "esm2+cheap": "#7d3c98",
}
LABELS = {
    "random": "random control",
    "aac": "length + AA composition",
    "biophys": "biophysical descriptors",
    "cheap": "cheap (all descriptors)",
    "esm2": "ESM-2 650M (frozen)",
    "esm2+cheap": "ESM-2 + cheap",
}
HEAD_LABELS = {"linear": "linear (logistic / ridge)", "mlp": "MLP (64)", "gbm": "gradient boosting"}
SPLIT_LABELS = {"random": "random split", "cluster": "cluster-held-out split"}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


KEYS = ["dataset", "target", "split", "features", "head", "n_request"]


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- s.d. over seeds, keyed on the *requested* training-set size.

    Under a cluster-held-out split the "all" pool differs slightly between
    seeds, so the realised sizes are averaged into a single x position.
    """
    g = df.groupby(KEYS)
    out = g.agg(mean=(metric, "mean"), std=(metric, "std"), count=(metric, "count"),
                n_train=("n_train", "mean")).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def best_head_envelope(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (target, split, feature set, N), keep the head with the best mean.

    Applied identically to every feature set, so comparisons stay fair, but
    it is an optimistic envelope and is labelled as such wherever it is used.
    """
    idx = agg.groupby(["dataset", "target", "split", "features", "n_request"])["mean"].idxmax()
    return agg.loc[idx].reset_index(drop=True)


def _ticks(sub: pd.DataFrame) -> list[int]:
    """One tick per requested training-set size.

    Cluster-held-out splits realise a slightly different "all" pool per seed,
    so nearly-coincident positions are collapsed to a single label.
    """
    vals = sorted({int(round(v)) for v in sub.n_train.dropna()})
    out: list[int] = []
    for v in vals:
        if out and v < out[-1] * 1.12:
            out[-1] = max(out[-1], v)
            continue
        out.append(v)
    return out


def _curve(ax, sub: pd.DataFrame, chance=None):
    for feat, d in sub.groupby("features"):
        d = d.sort_values("n_train")
        c = COLORS.get(feat, "k")
        ax.plot(d.n_train, d["mean"], marker="o", ms=3.5, lw=1.6, color=c, label=LABELS.get(feat, feat))
        ax.fill_between(
            d.n_train, d["mean"] - d["std"], d["mean"] + d["std"], color=c, alpha=0.13, lw=0
        )
    ax.set_xscale("log")
    ax.set_xlabel("training antibodies (N)")
    ticks = _ticks(sub)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=7.5)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if chance is not None:
        ax.axhline(chance, color="k", ls=":", lw=1, zorder=0)


def _seed_note(df: pd.DataFrame) -> str:
    n = df["count"].max() if "count" in df and len(df) else np.nan
    n = 0 if not np.isfinite(n) else int(n)
    return f"mean $\\pm$ 1 s.d. over {n} resampled splits"


def _feature_legend(fig, ncol=6, y=0.5):
    handles = [
        plt.Line2D([], [], color=COLORS[f], marker="o", ms=4, lw=1.8, label=LABELS[f])
        for f in COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8,
               bbox_to_anchor=(0.5, y))


def chen_figure(agg: pd.DataFrame, metric: str = "auc", path=None):
    sub = agg[agg.dataset.str.startswith("SAbDab")]
    heads = ["linear", "mlp", "gbm"]
    splits = ["random", "cluster"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True, sharex=True)
    for r, sp in enumerate(splits):
        for c, hd in enumerate(heads):
            ax = axes[r, c]
            # NB: sub.head is the DataFrame method, not the column -- index by name.
            _curve(ax, sub[(sub["split"] == sp) & (sub["head"] == hd)], chance=0.5)
            if r == 0:
                ax.set_title(HEAD_LABELS[hd], fontsize=9.5)
                ax.set_xlabel("")
            if c == 0:
                ax.set_ylabel(f"{SPLIT_LABELS[sp]}\ntest ROC-AUC")
    fig.suptitle(
        "SAbDab_Chen developability: how test ROC-AUC grows with training-set size\n"
        f"{_seed_note(sub)}; dotted line is chance",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    _feature_legend(fig, ncol=6, y=0.0)
    return _save(fig, path or FIGURES / "fig1_chen_learning_curves.png")


def tap_figure(agg: pd.DataFrame, targets, metric="spearman", path=None, envelope=True):
    sub = best_head_envelope(agg) if envelope else agg
    sub = sub[sub.dataset == "TAP"]
    fig, axes = plt.subplots(len(targets), 2, figsize=(8.6, 2.05 * len(targets)), sharex=True)
    for r, tgt in enumerate(targets):
        for c, sp in enumerate(["random", "cluster"]):
            ax = axes[r, c]
            _curve(ax, sub[(sub.target == tgt) & (sub.split == sp)], chance=0.0)
            ax.set_ylim(-0.55, 1.02)
            if r == 0:
                ax.set_title(SPLIT_LABELS[sp], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{tgt}\nSpearman $\\rho$")
            if r < len(targets) - 1:
                ax.set_xlabel("")
    tail = " (best of 3 heads per point)" if envelope else ""
    fig.suptitle(
        f"TAP developability metrics: test Spearman $\\rho$ vs. training-set size{tail}\n"
        f"{_seed_note(sub)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    _feature_legend(fig, ncol=3, y=0.0)
    return _save(fig, path or FIGURES / "fig2_tap_learning_curves.png")


def gap_figure(gap: pd.DataFrame, path=None):
    """Random-split score vs. cluster-split score at the largest N."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = [f"{r.target}" for r in gap.itertuples()]
    ypos = np.arange(len(gap))[::-1]
    ax.hlines(ypos, gap.cluster, gap.random_, color="#c9ccd1", lw=3, zorder=1)
    ax.scatter(gap.random_, ypos, s=46, color="#e06c2b", zorder=3, label="random split")
    ax.scatter(gap.cluster, ypos, s=46, color="#1f4e79", zorder=3, label="cluster-held-out split")
    for y, r in zip(ypos, gap.itertuples()):
        ax.text(
            max(r.random_, r.cluster) + 0.02, y, f"$\\Delta$ {r.random_ - r.cluster:+.2f}",
            va="center", fontsize=8, color="#444",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("test score at the largest training-set size (AUC for Chen, Spearman for TAP)")
    ax.set_xlim(-0.18, 1.20)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        "What a random split buys you that generalisation does not\n"
        "best feature set x head per target, largest N", fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, path or FIGURES / "fig3_random_vs_cluster.png")


def crossover_figure(cross: pd.DataFrame, path=None):
    """Where frozen ESM-2 starts to beat the cheap descriptors, by target."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    piv = cross.pivot(index="target", columns="split", values="delta")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order]
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv["random"], w, color="#e06c2b", label="random split")
    ax.bar(x + w / 2, piv["cluster"], w, color="#1f4e79", label="cluster-held-out split")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("ESM-2 score  $-$  cheap-descriptor score\n(at the largest N)")
    ax.legend(fontsize=8)
    ax.set_title("Does a 650M-parameter protein language model beat Biopython one-liners?", fontsize=11)
    fig.tight_layout()
    return _save(fig, path or FIGURES / "fig4_esm_vs_cheap.png")


def sample_efficiency_figure(eff: pd.DataFrame, path=None):
    """How many antibodies each feature set needs before it is worth acting on.

    A matrix rather than a bar chart: the values span two orders of magnitude
    and a good many cells are "never reached at any N we measured", which a
    log-scale bar chart renders as an invisible zero.
    """
    feats = [f for f in COLORS if f in set(eff.features)]
    targets = list(dict.fromkeys(eff.target))
    m = np.full((len(targets), len(feats)), np.nan)
    for r in eff.itertuples():
        if r.target in targets and r.features in feats:
            m[targets.index(r.target), feats.index(r.features)] = r.n_required

    with np.errstate(invalid="ignore"):
        logm = np.log10(m)
    fig, ax = plt.subplots(figsize=(1.35 * len(feats) + 3.0, 0.62 * len(targets) + 2.4))
    ax.set_facecolor("#eceff1")
    im = ax.imshow(logm, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(logm) if np.isfinite(logm).any() else 0,
                   vmax=np.nanmax(logm) if np.isfinite(logm).any() else 1)
    lo, hi = im.get_clim()
    for i in range(len(targets)):
        for j in range(len(feats)):
            v = m[i, j]
            if np.isnan(v):
                ax.text(j, i, "never", ha="center", va="center", fontsize=7.5, color="#78868f")
            else:
                dark = (logm[i, j] - lo) / max(hi - lo, 1e-9) > 0.62
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=8.5, color="white" if dark else "#20262a")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([LABELS[f] for f in feats], rotation=22, ha="right", fontsize=8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8.5)
    ax.set_xticks(np.arange(len(feats) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(targets) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_title(
        "Training antibodies needed before the model is worth acting on\n"
        "cluster-held-out split; ROC-AUC $\\geq$ 0.70 for developability, "
        "Spearman $\\rho \\geq$ 0.50 for the TAP metrics",
        fontsize=10.5, pad=12,
    )
    fig.tight_layout()
    return _save(fig, path or FIGURES / "fig5_sample_efficiency.png")


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def scaling_figure(agg_env: pd.DataFrame, dataset: str, targets, path=None):
    """Measured points plus the fitted power law, extended past the data.

    The extrapolated region is dashed and shaded to keep it visually distinct
    from anything that was actually measured.
    """
    from .scaling import fit_curve, power_law

    sub = agg_env[(agg_env.dataset == dataset) & (agg_env.split == "cluster")]
    targets = [t for t in targets if t in set(sub.target)]
    if not targets:
        print(f"skipping scaling figure: no cluster-split rows for {dataset}")
        return None
    fig, axes = plt.subplots(1, len(targets), figsize=(4.9 * len(targets), 3.9), squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        g_all = sub[sub.target == tgt]
        n_max = g_all.n_train.max()
        grid = np.logspace(np.log10(25), np.log10(n_max * 12), 200)
        for feat in ["cheap", "esm2", "esm2+cheap"]:
            g = g_all[g_all.features == feat].sort_values("n_train")
            if g.empty:
                continue
            c = COLORS[feat]
            ax.plot(g.n_train, g["mean"], "o", ms=4, color=c, label=LABELS[feat])
            params, r2 = fit_curve(g.n_train.to_numpy(), g["mean"].to_numpy())
            if params is None:
                continue
            ax.plot(grid[grid <= n_max], power_law(grid[grid <= n_max], *params), "-", lw=1.5, color=c)
            ax.plot(grid[grid >= n_max], power_law(grid[grid >= n_max], *params), "--", lw=1.3, color=c, alpha=0.8)
        ax.axvspan(n_max, grid[-1], color="#000000", alpha=0.045, lw=0)
        lo, hi = ax.get_ylim()
        ax.text(n_max * 1.25, hi - 0.03 * (hi - lo), "extrapolated",
                fontsize=7.5, color="#777", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("training antibodies (N)")
        ax.set_title(tgt, fontsize=9.5)
    axes[0][0].set_ylabel("test score (cluster-held-out split)")
    axes[0][0].legend(loc="lower right", fontsize=7.4)
    fig.suptitle(
        "How many more antibodies would it take?  Saturating power-law fit, extended past the data",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path or FIGURES / "fig6_scaling_extrapolation.png")
