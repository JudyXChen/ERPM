"""Plots for the sensitivity results: grouped-Sobol bars and Morris mu*-sigma."""

import pathlib

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_grouped_sobol(results, out_dir):
    plt = _mpl()
    out_dir = pathlib.Path(out_dir)
    qois = list(results)
    fig, axes = plt.subplots(1, len(qois), figsize=(4.6 * len(qois), 4.6),
                             squeeze=False)
    for ax, qoi in zip(axes[0], qois):
        r = results[qoi]
        groups = r["groups"]
        order = np.argsort(r["ST"])[::-1]
        g = [groups[i] for i in order]
        s1 = [r["S1"][i] for i in order]
        st = [r["ST"][i] for i in order]
        x = np.arange(len(g))
        ax.bar(x - 0.2, s1, 0.4, label="S1 (main)", color="#2c7fb8")
        ax.bar(x + 0.2, st, 0.4, label="ST (total)", color="#d7301f")
        ax.set_xticks(x)
        ax.set_xticklabels(g, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sobol index")
        ax.set_title(qoi)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "grouped_sobol.png", dpi=140)
    plt.close(fig)


def plot_within_sobol_heatmap(results, names, out_dir, title=""):
    """Parameter x QoI heatmaps for a within-group run.

    Three panels share the parameter (row) axis:
      - Sobol ST : total-order importance (unsigned, 0..1).
      - Sobol S1 : first-order / main effect (the rest is interaction).
      - PRCC     : signed direction (Leung-style; diverging colormap),
                   '*' marks the cells with |PRCC| significant at p<0.05.
    Rows are ordered by mean ST so the dominant levers sit at the top.
    """
    plt = _mpl()
    out_dir = pathlib.Path(out_dir)
    qois = list(results)

    ST = np.array([[results[q]["ST"][i] for q in qois]
                   for i in range(len(names))])
    S1 = np.array([[results[q]["S1"][i] for q in qois]
                   for i in range(len(names))])

    def _prcc(key):
        return np.array([
            [(results[q].get(key) or [np.nan] * len(names))[i] for q in qois]
            for i in range(len(names))])

    PR = _prcc("prcc")
    PV = _prcc("prcc_pval")

    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [names[i] for i in order]
    ST, S1, PR, PV = ST[order], S1[order], PR[order], PV[order]

    panels = [
        ("Sobol ST (total)", ST, "viridis", 0.0, max(1.0, np.nanmax(ST)), False),
        ("Sobol S1 (main)", S1, "viridis", 0.0, max(1.0, np.nanmax(S1)), False),
        ("PRCC (direction)", PR, "RdBu_r", -1.0, 1.0, True),
    ]
    n = len(rows)
    fig, axes = plt.subplots(
        1, 3, figsize=(2.0 * len(qois) * 3 + 2, 0.42 * n + 2.2),
        squeeze=False)
    for ax, (lab, M, cmap, vmin, vmax, signed) in zip(axes[0], panels):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
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
                star = "*" if signed and np.isfinite(PV[r, c]) \
                    and PV[r, c] < 0.05 else ""
                txt = f"{v:.2f}{star}"
                shade = abs(v) / max(vmax, 1e-9) if signed else v / max(vmax, 1e-9)
                ax.text(c, r, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if shade > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Within-group sensitivity: {title}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "within_sobol_heatmap.png", dpi=140)
    plt.close(fig)


def plot_morris(results, out_dir):
    plt = _mpl()
    out_dir = pathlib.Path(out_dir)
    qois = list(results)
    fig, axes = plt.subplots(1, len(qois), figsize=(5.2 * len(qois), 4.8),
                             squeeze=False)
    for ax, qoi in zip(axes[0], qois):
        r = results[qoi]
        mu, sig, names = r["mu_star"], r["sigma"], r["names"]
        ax.scatter(mu, sig, s=18, color="#2c7fb8")
        # label the most influential handful
        order = np.argsort(mu)[::-1][:8]
        for i in order:
            ax.annotate(names[i], (mu[i], sig[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mu*  (overall influence)")
        ax.set_ylabel("sigma  (nonlinearity / interaction)")
        ax.set_title(qoi)
    fig.tight_layout()
    fig.savefig(out_dir / "morris_mu_sigma.png", dpi=140)
    plt.close(fig)
