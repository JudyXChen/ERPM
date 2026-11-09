"""Plots: Sobol bars/heatmaps and the Morris mu*-sigma scatter.
"""

import argparse
import json
import pathlib
import re

from scripts import plot_style

import numpy as np

from .evaluate import CA_QOI_NAMES, FLUX_QOI_NAMES

# QoI families rendered as separate grouped figures (suffix -> member QoIs).
_QOI_FAMILIES = (("calcium", CA_QOI_NAMES), ("flux", FLUX_QOI_NAMES))


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style.apply_style(plt)
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
    "abs_drain": "ER drain rate",
    "tau_drain": "Time to empty ER",
    "ryr_open": "RyR open",
    "ip3r_open": "IP3R open",
    "caer_frac_end": "ER Ca remaining",
    "ca_c_end": "Final cytosolic Ca",
}


def _qoi_label(name):
    return _QOI_LABELS.get(name, name)


def _stack(results, key, n_rows, qois):
    """Row(parameter/group) x column(QoI) matrix for one index key."""
    return np.array([[results[q][key][i] for q in qois] for i in range(n_rows)])


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _sobol_heatmap(rows, qois, panels, out_path, cell_w=0.62):
    """Fixed-size pages of at most six rows and three QoIs; never shrink fonts.

    cell_w is retained for compatibility; physical panel size is standardized.
    """
    _mpl()
    for lab, matrix, vmax in panels:
        for r0 in range(0, len(rows), 6):
            for c0 in range(0, len(qois), 3):
                fig, ax = plot_style.figure()
                values = matrix[r0:r0+6, c0:c0+3]
                im = ax.imshow(values, aspect="auto", cmap=plot_style.blue_colormap(),
                               vmin=0, vmax=vmax)
                ax.set_xticks(range(values.shape[1]),
                              [_qoi_label(q) for q in qois[c0:c0+3]],
                              rotation=45, ha="right")
                ax.set_yticks(range(values.shape[0]), rows[r0:r0+6])
                for r in range(values.shape[0]):
                    for c in range(values.shape[1]):
                        value = values[r, c]
                        if np.isfinite(value):
                            ax.text(c, r, f"{value:.2f}", ha="center", va="center",
                                    color="white" if value / max(vmax, 1e-9) > 0.65
                                    else plot_style.COLORS["default"])
                colorbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.05)
                colorbar.set_label(lab)
                suffix = f"{_slug(lab)}_r{r0//6+1}_c{c0//3+1}"
                plot_style.save(fig, plot_style.panel_path(out_path, suffix))


def plot_grouped_sobol(results, out_dir, panel_w=3.2, name="grouped_sobol.png"):
    """One horizontal S1/ST grouped-bar panel per QoI, with a shared legend."""
    _mpl()
    out_dir = pathlib.Path(out_dir)
    for qoi, r in results.items():
        fig, ax = plot_style.figure()
        order = np.argsort(r["ST"])[::-1]
        groups = [r["groups"][i] for i in order]
        y = np.arange(len(groups))
        ax.barh(y - 0.2, [r["S1"][i] for i in order], 0.4, label="S1",
                color=plot_style.shade("blue"))
        ax.barh(y + 0.2, [r["ST"][i] for i in order], 0.4, label="ST",
                color=plot_style.shade("purple"))
        ax.set_yticks(y, groups)
        ax.invert_yaxis()
        ax.set_xlabel("Sobol index")
        plot_style.save(fig, plot_style.panel_path(out_dir / name, qoi),
                        legend_name="sobol_legend.png", ncol=2)


def plot_grouped_sobol_heatmap(results, out_dir, title="", cell_w=0.62,
                               name="grouped_sobol_heatmap.png"):
    """Mechanism x QoI heatmaps: Sobol ST and Sobol S1."""
    qois = list(results)
    groups = results[qois[0]]["groups"]
    ST = _stack(results, "ST", len(groups), qois)
    S1 = _stack(results, "S1", len(groups), qois)
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [groups[i] for i in order]
    ST, S1 = ST[order], S1[order]
    # Sobol indices are bounded by 1; clip the colour scale there so noisy
    # >1 estimates (the printed value still shows) don't wash out the panel.
    panels = [
        ("Sobol ST", ST, 1.0),
        ("Sobol S1", S1, 1.0),
        ("ST - S1 (interaction)", np.clip(ST - S1, 0.0, None), 1.0),
    ]
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / name, cell_w=cell_w)


def plot_grouped(results, out_dir, *, cell_w=0.62, bar=True):
    """Render grouped Sobol as one figure per QoI family (calcium / flux), so
    no single heatmap is too wide. QoIs outside those families (e.g. resting
    runs) fall back to one combined figure with the legacy name."""
    families = [(tag, {q: results[q] for q in names if q in results})
                for tag, names in _QOI_FAMILIES]
    families = [(tag, sub) for tag, sub in families if sub]
    covered = {q for _, sub in families for q in sub}
    other = {q: results[q] for q in results if q not in covered}
    if other:
        families.append(("other", other))

    single = len(families) == 1  # resting run -> keep the unsuffixed filename
    for tag, sub in families:
        sfx = "" if single else f"_{tag}"
        plot_grouped_sobol_heatmap(sub, out_dir, cell_w=cell_w,
                                   name=f"grouped_sobol_heatmap{sfx}.png")
        if bar:
            plot_grouped_sobol(sub, out_dir, name=f"grouped_sobol{sfx}.png")
        print(f"  [{tag}] {len(sub)} QoIs -> grouped_sobol_heatmap{sfx}_*.png")


def plot_within_sobol_heatmap(results, names, out_dir, title=""):
    """Parameter x QoI heatmaps: Sobol ST and Sobol S1."""
    qois = list(results)
    ST = _stack(results, "ST", len(names), qois)
    S1 = _stack(results, "S1", len(names), qois)
    order = np.argsort(np.nanmean(ST, axis=1))[::-1]
    rows = [names[i] for i in order]
    ST, S1 = ST[order], S1[order]
    panels = [
        ("Sobol ST", ST, 1.0),
        ("Sobol S1", S1, 1.0),
    ]
    _sobol_heatmap(rows, qois, panels,
                   pathlib.Path(out_dir) / "within_sobol_heatmap.png")


def plot_morris(results, out_dir):
    """One mu*-sigma panel per QoI; numbered top-five markers avoid text overlap."""
    _mpl()
    out_dir = pathlib.Path(out_dir)
    for qoi, r in results.items():
        fig, ax = plot_style.figure()
        mu, sig, names = np.array(r["mu_star"]), np.array(r["sigma"]), r["names"]
        ax.scatter(mu, sig, s=plot_style.MARKER_SIZE, color=plot_style.shade("blue", "light"))
        for rank, i in enumerate(np.argsort(mu)[::-1][:5], 1):
            ax.scatter(mu[i], sig[i], marker=f"${rank}$", s=81,
                       color=plot_style.shade("blue", "dark"), label=names[i])
        ax.margins(0.12)
        ax.set(xlabel=r"$\mu^*$", ylabel=r"$\sigma$")
        plot_style.save(fig, out_dir / f"morris_mu_sigma_{qoi}.png")


def main(argv=None):
    """Re-render grouped figures from a saved sobol_indices.json, no model run."""
    ap = argparse.ArgumentParser(
        description="Re-render grouped Sobol figures from a saved run.")
    ap.add_argument("path", help="grouped_sobol dir or sobol_indices.json")
    ap.add_argument("--qois", nargs="+", default=None,
                    help="render ONE figure with these QoIs (default: split "
                         "into calcium + flux figures)")
    ap.add_argument("--tag", default="custom",
                    help="filename suffix for the --qois figure")
    ap.add_argument("--cell-w", type=float, default=0.62,
                    help="deprecated compatibility flag; panels are fixed at 3 × 3 inches")
    ap.add_argument("--no-bar", action="store_true",
                    help="skip the S1/ST bar chart, heatmaps only")
    a = ap.parse_args(argv)

    path = pathlib.Path(a.path)
    json_path = path / "sobol_indices.json" if path.is_dir() else path
    out_dir = json_path.parent
    with open(json_path) as fh:
        results = json.load(fh)

    if a.qois:
        missing = [q for q in a.qois if q not in results]
        if missing:
            ap.error(f"unknown QoI(s): {missing}; available: {list(results)}")
        sub = {q: results[q] for q in a.qois}
        plot_grouped_sobol_heatmap(sub, out_dir, cell_w=a.cell_w,
                                   name=f"grouped_sobol_heatmap_{a.tag}.png")
        if not a.no_bar:
            plot_grouped_sobol(sub, out_dir, name=f"grouped_sobol_{a.tag}.png")
        print(f"  [{a.tag}] {len(sub)} QoIs -> grouped_sobol_heatmap_{a.tag}_*.png")
    else:
        plot_grouped(results, out_dir, cell_w=a.cell_w, bar=not a.no_bar)
    print(f"-> {out_dir}/")


if __name__ == "__main__":
    main()
