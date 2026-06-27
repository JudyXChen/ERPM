"""Plots: Sobol bars/heatmaps and the Morris mu*-sigma scatter."""

import pathlib

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _stack(results, key, n_rows, qois):
    """Row(parameter/group) x column(QoI) matrix for one index key."""
    return np.array([[results[q][key][i] for q in qois] for i in range(n_rows)])


def _sobol_heatmap(rows, qois, panels, out_path, suptitle):
    """Shared renderer: one viridis panel per (label, matrix, vmax)."""
    plt = _mpl()
    n = len(rows)
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(2.0 * len(qois) * len(panels) + 2, 0.42 * n + 2.2),
        squeeze=False)
    for ax, (lab, M, vmax) in zip(axes[0], panels):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(qois)))
        ax.set_xticklabels(qois, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(lab, fontsize=10)
        for r in range(n):
            for c in range(len(qois)):
                v = M[r, c]
                if not np.isfinite(v):
                    continue
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v / max(vmax, 1e-9) > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grouped_sobol(results, out_dir):
    """S1/ST grouped-bar chart per QoI."""
    plt = _mpl()
    out_dir = pathlib.Path(out_dir)
    qois = list(results)
    fig, axes = plt.subplots(1, len(qois), figsize=(4.6 * len(qois), 4.6),
                             squeeze=False)
    for ax, qoi in zip(axes[0], qois):
        r = results[qoi]
        order = np.argsort(r["ST"])[::-1]
        g = [r["groups"][i] for i in order]
        x = np.arange(len(g))
        ax.bar(x - 0.2, [r["S1"][i] for i in order], 0.4, label="S1 (main)",
               color="#2c7fb8")
        ax.bar(x + 0.2, [r["ST"][i] for i in order], 0.4, label="ST (total)",
               color="#d7301f")
        ax.set_xticks(x)
        ax.set_xticklabels(g, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sobol index")
        ax.set_title(qoi)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "grouped_sobol.png", dpi=140)
    plt.close(fig)


def plot_grouped_sobol_heatmap(results, out_dir, title=""):
    """Mechanism x QoI heatmaps: ST (total), S1 (main), ST-S1 (interaction)."""
    qois = list(results)
    groups = results[qois[0]]["groups"]
    ST = _stack(results, "ST", len(groups), qois)
    S1 = _stack(results, "S1", len(groups), qois)
    IX = np.clip(ST - S1, 0.0, None)  # interaction share; negatives are noise
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [groups[i] for i in order]
    ST, S1, IX = ST[order], S1[order], IX[order]
    panels = [
        ("Sobol ST (total)", ST, max(1.0, np.nanmax(ST))),
        ("Sobol S1 (main)", S1, max(1.0, np.nanmax(S1))),
        ("ST - S1 (interaction)", IX, max(1.0, np.nanmax(IX))),
    ]
    sup = "Grouped Sobol: mechanism attribution" + (f" ({title})" if title else "")
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / "grouped_sobol_heatmap.png", sup)


def plot_within_sobol_heatmap(results, names, out_dir, title=""):
    """Parameter x QoI heatmaps: ST (total) and S1 (main)."""
    qois = list(results)
    ST = _stack(results, "ST", len(names), qois)
    S1 = _stack(results, "S1", len(names), qois)
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [names[i] for i in order]
    ST, S1 = ST[order], S1[order]
    panels = [
        ("Sobol ST (total)", ST, max(1.0, np.nanmax(ST))),
        ("Sobol S1 (main)", S1, max(1.0, np.nanmax(S1))),
    ]
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / "within_sobol_heatmap.png",
                   f"Within-group sensitivity: {title}")


def plot_morris(results, out_dir):
    """mu*-sigma scatter per QoI, labelling the most influential parameters."""
    plt = _mpl()
    out_dir = pathlib.Path(out_dir)
    qois = list(results)
    fig, axes = plt.subplots(1, len(qois), figsize=(5.2 * len(qois), 4.8),
                             squeeze=False)
    for ax, qoi in zip(axes[0], qois):
        r = results[qoi]
        mu, sig, names = r["mu_star"], r["sigma"], r["names"]
        ax.scatter(mu, sig, s=18, color="#2c7fb8")
        for i in np.argsort(mu)[::-1][:8]:
            ax.annotate(names[i], (mu[i], sig[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mu*  (overall influence)")
        ax.set_ylabel("sigma  (nonlinearity / interaction)")
        ax.set_title(qoi)
    fig.tight_layout()
    fig.savefig(out_dir / "morris_mu_sigma.png", dpi=140)
    plt.close(fig)
