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
