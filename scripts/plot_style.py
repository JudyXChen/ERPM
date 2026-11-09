"""Canonical ERPM plotting style and PNG export. See doc/PLOTTING.md."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIGSIZE = (3, 3)
DPI = 300
FONT_SIZE = 9
LINEWIDTH = 1.6
MARKER_SIZE = 22
FAMILIES = {
    "blue": {"light": "#B3D2E6", "main": "#4F93BF", "dark": "#2E6F9D"},
    "purple": {"light": "#D7B9E2", "main": "#AF72C3", "dark": "#844A9D"},
    "orange": {"light": "#F2CDA9", "main": "#CC762B", "dark": "#9A4E16"},
}


def shade(family, tone="main"):
    return FAMILIES[family][tone]


def colors(n):
    if not 0 <= n <= 6:
        raise ValueError("Use at most six colors; split panels or add line styles")
    return [shade(f, t) for t in ("main", "dark") for f in FAMILIES][:n]


COLORS = {
    "cyto": shade("blue"), "er": shade("purple"),
    "ecs": shade("orange"), "junction": shade("purple", "dark"),
    "voltage": shade("orange", "dark"), "default": "#333333",
    "reference": "#777777",
}
SPECIES_COMPARTMENT = {
    **dict.fromkeys(("Ca_c", "Ca_c_ions", "IP3", "PLC_free", "PLC_Ca",
                     "PLC_Ca_PIP2", "Bm_free", "Bm_Ca", "Bf_free", "Bf_Ca"), "cyto"),
    "Ca_ER": "er", "Glu": "ecs", "V_m": "voltage",
}
PATHWAY_COLORS = {
    "SERCA": shade("blue"), "SERCA binding": shade("blue"),
    "IP3R": shade("orange"), "RyR": shade("purple"),
    "ER leak": shade("blue", "dark"), "PM influx": shade("blue", "dark"),
    "PM efflux": shade("orange", "dark"), "ER exchange": shade("purple"),
    "buffering": shade("orange"), "Ca decay": shade("purple", "dark"),
    "net": "#333333", "recorded change": "#333333", "flux sum": "#777777",
}
# SOCE is the hue, RyR is the line style, in every comparison panel.
CONDITIONS = {
    "ryr_on_soce_off": ("RyR on, SOCE off", shade("blue"), "-"),
    "ryr_on_soce_on": ("RyR on, SOCE on", shade("purple"), "-"),
    "ryr_off_soce_off": ("RyR off, SOCE off", shade("blue"), "--"),
    "ryr_off_soce_on": ("RyR off, SOCE on", shade("purple"), "--"),
}
EXPORTED = []  # Used by the recorded-data replot command to verify its outputs.


def color_for(name):
    return COLORS[SPECIES_COMPARTMENT.get(name, "default")]


def blue_colormap():
    return LinearSegmentedColormap.from_list(
        "erpm_blue", ["#FFFFFF", shade("blue", "light"), shade("blue"), shade("blue", "dark")])


def apply_style(plt=plt):
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "figure.figsize": FIGSIZE, "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE, "axes.linewidth": 1.0,
        "lines.linewidth": LINEWIDTH, "lines.markersize": 4.5,
        "axes.prop_cycle": plt.cycler(color=colors(6)),
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def format_axis(ax):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")


def figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    format_axis(ax)
    return fig, ax


def panel_path(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}.png")


def save_legend(handles, labels, path, *, ncol=1):
    """Export one transparent legend; callers share a name for identical groups."""
    path = Path(path).with_suffix(".png")
    if path.parent.name != "fig":
        path = path.parent / "fig" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE)
    legend = fig.legend(handles, labels, loc="center", frameon=False, ncol=ncol)
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches=box.expanded(1.03, 1.05))
    plt.close(fig)
    return path


def save(fig, path, *, legend_name="auto", ncol=1):
    """Save an exact 900×900 clean panel and a viewing copy with an outer legend.

    Reuse legend_name across panels with identical labels, colors and styles.
    Colorbars and scientific annotations remain part of the clean panel.
    """
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = path.parent / "fig" / path.name
    clean.parent.mkdir(parents=True, exist_ok=True)
    handles, labels = [], []
    for ax in fig.axes:
        hs, ls = ax.get_legend_handles_labels()
        for handle, label in zip(hs, ls):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title("")
    for legend in list(fig.legends):
        legend.remove()
    fig.set_size_inches(FIGSIZE)
    fig.tight_layout(pad=0.7)
    fig.savefig(clean, dpi=DPI)
    legend_path = None
    if len(labels) > 1:
        # Figure-relative anchoring keeps long legends clear of axis labels.
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0),
                   frameon=False, ncol=ncol)
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
        if legend_name is not None:
            name = f"{path.stem}_legend.png" if legend_name == "auto" else legend_name
            legend_path = save_legend(handles, labels, path.parent / name, ncol=ncol)
    else:
        fig.savefig(path, dpi=DPI)
    EXPORTED.append({"view": str(path), "panel": str(clean),
                     "legend": str(legend_path) if legend_path else None})
    plt.close(fig)
    return path
