FIGSIZE = (5.0, 4.0)
DPI = 300

COLORS = {
    "cyto": "#6A3D9A",      # purple
    "er": "#C7A6E8",        # light purple
    "ecs": "#9ED17E",       # light green
    "voltage": "#8B1A1A",   # dark red
    "default": "#333333",
}

SPECIES_COMPARTMENT = {
    "Ca_c": "cyto",
    "Ca_c_ions": "cyto",
    "IP3": "cyto",
    "PLC_free": "cyto",
    "PLC_Ca": "cyto",
    "PLC_Ca_PIP2": "cyto",
    "Bm_free": "cyto",
    "Bm_Ca": "cyto",
    "Bf_free": "cyto",
    "Bf_Ca": "cyto",
    "Ca_ER": "er",
    "Glu": "ecs",
    "V_m": "voltage",
}


def color_for(name):
    """Return the publication color for a species/stimulus name."""
    return COLORS[SPECIES_COMPARTMENT.get(name, "default")]


def apply_style(plt):
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.4,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
