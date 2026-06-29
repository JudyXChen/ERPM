"""Plots: Sobol bars/heatmaps and the Morris mu*-sigma scatter."""

import pathlib

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Display labels (mathtext): Ca_cyto subscripted, Greek tau, per-pathway Q.
_QOI_LABELS = {
    # stimulus response
    "ca_cyto_peak": r"$Ca_{cyto}$ peak",
    "ca_cyto_ttp": r"$Ca_{cyto}$ $t_{peak}$",
    "ca_cyto_fwhm": r"$Ca_{cyto}$ FWHM",
    "ca_cyto_tau_decay": r"$Ca_{cyto}$ $\tau_{decay}$",
    "ca_cyto_auc": r"$Ca_{cyto}$ AUC",
    "ca_cyto_plateau": r"$Ca_{cyto}$ plateau",
    "ip3r_peak_open": "IP3R open",
    "ryr_peak_open": "RyR open",
    "q_ipr": r"$Q_{IP3R}$",
    "q_ryr": r"$Q_{RyR}$",
    "q_serca": r"$Q_{SERCA}$",
    "q_soce": r"$Q_{SOCE}$",
    "cicr_gain": "CICR gain",
    # resting / free-run
    "abs_drain": "|drain|",
    "tau_drain": r"$\tau_{drain}$",
    "ryr_open": "RyR open",
    "ip3r_open": "IP3R open",
    "caer_frac_end": r"$Ca_{ER}$ frac",
    "ca_c_end": r"$Ca_{cyto}$ end",
}


def _qoi_label(name):
    return _QOI_LABELS.get(name, name)


def _stack(results, key, n_rows, qois):
    """Row(parameter/group) x column(QoI) matrix for one index key."""
    return np.array([[results[q][key][i] for q in qois] for i in range(n_rows)])


def _sobol_heatmap(rows, qois, panels, out_path):
    """Shared renderer: one viridis panel per (label, matrix, vmax)."""
    plt = _mpl()
    n = len(rows)
    labels = [_qoi_label(q) for q in qois]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(2.0 * len(qois) * len(panels) + 2, 0.42 * n + 2.2),
        squeeze=False)
    for ax, (lab, M, vmax) in zip(axes[0], panels):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(qois)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
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
    fig.tight_layout()
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
        ax.bar(x - 0.2, [r["S1"][i] for i in order], 0.4, label="S1",
               color="#2c7fb8")
        ax.bar(x + 0.2, [r["ST"][i] for i in order], 0.4, label="ST",
               color="#d7301f")
        ax.set_xticks(x)
        ax.set_xticklabels(g, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sobol index")
        ax.set_title(_qoi_label(qoi))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "grouped_sobol.png", dpi=140)
    plt.close(fig)


def plot_grouped_sobol_heatmap(results, out_dir, title=""):
    """Mechanism x QoI heatmaps: Sobol ST and Sobol S1."""
    qois = list(results)
    groups = results[qois[0]]["groups"]
    ST = _stack(results, "ST", len(groups), qois)
    S1 = _stack(results, "S1", len(groups), qois)
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [groups[i] for i in order]
    ST, S1 = ST[order], S1[order]
    panels = [
        ("Sobol ST", ST, max(1.0, np.nanmax(ST))),
        ("Sobol S1", S1, max(1.0, np.nanmax(S1))),
    ]
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / "grouped_sobol_heatmap.png")


def plot_within_sobol_heatmap(results, names, out_dir, title=""):
    """Parameter x QoI heatmaps: Sobol ST and Sobol S1."""
    qois = list(results)
    ST = _stack(results, "ST", len(names), qois)
    S1 = _stack(results, "S1", len(names), qois)
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [names[i] for i in order]
    ST, S1 = ST[order], S1[order]
    panels = [
        ("Sobol ST", ST, max(1.0, np.nanmax(ST))),
        ("Sobol S1", S1, max(1.0, np.nanmax(S1))),
    ]
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / "within_sobol_heatmap.png")


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
        ax.set_xlabel(r"$\mu^*$  (overall influence)")
        ax.set_ylabel(r"$\sigma$  (nonlinearity / interaction)")
        ax.set_title(_qoi_label(qoi))
    fig.tight_layout()
    fig.savefig(out_dir / "morris_mu_sigma.png", dpi=140)
    plt.close(fig)
