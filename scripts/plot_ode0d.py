"""Plot an ode0d traces.csv
"""

import csv
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/erpm_mplconfig")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.parameters import default_parameters
from scripts import plot_style


UM = 1.0e6
plot_style.apply_style(plt)


def load(csv_path):
    with open(csv_path) as fh:
        r = csv.reader(fh)
        header = next(r)
        data = np.array([[float(x) for x in row] for row in r])
    return header, {h: data[:, i] for i, h in enumerate(header)}


def plot_one(out_dir, t_ms, y, filename, ylabel, color, scale=1.0):
    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    ax.plot(t_ms, y * scale, color=color)
    ax.set(xlabel="t (ms)", ylabel=ylabel)
    plot_style.format_axis(ax)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_stimulus(out_dir, t_ms, vm, glu):
    """V_m and glutamate as two stacked subplots in stimulus.png."""
    w, h = plot_style.FIGSIZE
    fig, (ax_g, ax_v) = plt.subplots(
        2, 1, sharex=True, figsize=(w, 1.6 * h))
    ax_g.plot(t_ms, glu * UM, color=plot_style.color_for("Glu"))
    ax_g.set(ylabel="Glu (µM)")
    ax_v.plot(t_ms, vm, color=plot_style.color_for("V_m"))
    ax_v.set(xlabel="t (ms)", ylabel="V_m (mV)")
    for ax in (ax_g, ax_v):
        plot_style.format_axis(ax)
    fig.tight_layout()
    path = out_dir / "stimulus.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _open_channel_series(col):
    p = default_parameters()
    specs = [
        ("AMPAR", float(p.N_AMPAR), ["AO"]),
        ("NMDAR", float(p.N_NMDAR), ["R"]),
        ("VSCC", float(p.N_VSCC), ["V4"]),
        ("IP3R", float(p.N_IP3R), ["IPR_110"]),
        (
            "RyR",
            float(p.N_RyR),
            [
                f"R_O_{ja}_{ji}"
                for ja in range(5)
                for ji in range(5)
            ],
        ),
    ]
    series = {}
    for channel, n_total, states in specs:
        present = [state for state in states if state in col]
        if not present:
            continue
        open_fraction = np.zeros_like(col[present[0]])
        for state in present:
            open_fraction += col[state]
        series[channel] = n_total * open_fraction
    if "P_O_STIM1" in col:
        open_fraction = col["P_O_STIM1"]
        series["Orai1"] = float(p.N_Orai1) * open_fraction
    return series


def write_open_channel_tables(out_dir, col):
    series = _open_channel_series(col)
    if not series:
        return []

    paths = []
    channels = list(series)
    summary_path = out_dir / "open_channels_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "start", "min", "max", "end"])
        for channel in channels:
            y = series[channel]
            w.writerow([channel, y[0], y.min(), y.max(), y[-1]])
    paths.append(summary_path)
    return paths


def ca_ion_count(ca_molar):
    p = default_parameters()
    return ca_molar * float(p.N_A) * float(p.V_spine_L)


def write_ca_ions_summary(out_dir, col):
    if "Ca_c" not in col:
        return []
    ca_ions = ca_ion_count(col["Ca_c"])
    path = out_dir / "ca_ions_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "start", "min", "max", "end"])
        w.writerow([
            "Ca_c",
            ca_ions[0],
            ca_ions.min(),
            ca_ions.max(),
            ca_ions[-1],
        ])
    return [path]


def _time_window(col, t_min_ms, t_max_ms):
    t_ms = col["t"] * 1e3
    mask = np.ones_like(t_ms, dtype=bool)
    if t_min_ms is not None:
        mask &= t_ms >= t_min_ms
    if t_max_ms is not None:
        mask &= t_ms <= t_max_ms
    return t_ms[mask], mask


def main(out_dir, t_min_ms=0.0, t_max_ms=50.0):
    out_dir = pathlib.Path(out_dir)
    _header, col = load(out_dir / "traces.csv")
    t_ms, plot_mask = _time_window(col, t_min_ms, t_max_ms)

    written = []
    if "V_m" in col and "Glu" in col:
        written.append(
            plot_stimulus(out_dir, t_ms, col["V_m"][plot_mask],
                          col["Glu"][plot_mask])
        )
    if "Ca_c" in col:
        ca_c = col["Ca_c"][plot_mask]
        written.append(
            plot_one(out_dir, t_ms, ca_c, "Ca_cyto.png",
                     r"$[Ca^{2+}]_{cyto}$ (µM)",
                     color=plot_style.color_for("Ca_c"),
                     scale=UM)
        )
        ca_c_max = np.max(ca_c)
        if ca_c_max > 0.0:
            written.append(
                plot_one(out_dir, t_ms, ca_c / ca_c_max,
                         "Ca_cyto_norm.png",
                         r"Normalized $[Ca^{2+}]_{cyto}$",
                         color=plot_style.color_for("Ca_c")))
    if "Ca_ER" in col:
        written.append(
            plot_one(out_dir, t_ms, col["Ca_ER"][plot_mask], "Ca_ER.png",
                     r"$[Ca^{2+}]_{ER}$ (µM)",
                     color=plot_style.color_for("Ca_ER"),
                     scale=UM)
        )

    written.extend(write_open_channel_tables(out_dir, col))
    # written.extend(write_ca_ions_summary(out_dir, col))
    print("wrote " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default="results/ode0d")
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=50.0)
    args = parser.parse_args()
    main(args.out_dir, t_min_ms=args.t_min_ms, t_max_ms=args.t_max_ms)
