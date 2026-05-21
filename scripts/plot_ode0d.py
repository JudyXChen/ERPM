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
    return series


def write_open_channel_tables(out_dir, col):
    series = _open_channel_series(col)
    if not series:
        return []

    paths = []
    channels = list(series)
    counts_path = out_dir / "open_channels.csv"
    with open(counts_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + channels)
        for i, t in enumerate(col["t"]):
            w.writerow([t] + [series[channel][i] for channel in channels])
    paths.append(counts_path)

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
    if "V_m" in col:
        written.append(
            plot_one(out_dir, t_ms, col["V_m"][plot_mask], "Stimulus_Vm.png",
                     "V_m (mV)", color=plot_style.color_for("V_m"))
        )
    if "Glu" in col:
        written.append(
            plot_one(out_dir, t_ms, col["Glu"][plot_mask], "glu_conc.png",
                     "Glu (µM)", color=plot_style.color_for("Glu"), scale=UM)
        )
    if "Ca_c" in col:
        written.append(
            plot_one(out_dir, t_ms, col["Ca_c"][plot_mask], "Ca_cyto.png",
                     r"$[Ca^{2+}]_{cyto}$ (µM)",
                     color=plot_style.color_for("Ca_c"),
                     scale=UM)
        )
        written.append(
            plot_one(out_dir, t_ms, ca_ion_count(col["Ca_c"][plot_mask]),
                     "Ca_cyto_ions.png", r"$Ca^{2+}_{cyto}$ ions",
                     color=plot_style.color_for("Ca_c_ions"))
        )
    if "Ca_ER" in col:
        written.append(
            plot_one(out_dir, t_ms, col["Ca_ER"][plot_mask], "Ca_ER.png",
                     r"$[Ca^{2+}]_{ER}$ (µM)",
                     color=plot_style.color_for("Ca_ER"),
                     scale=UM)
        )

    written.extend(write_open_channel_tables(out_dir, col))
    written.extend(write_ca_ions_summary(out_dir, col))
    print("wrote " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default="results/ode0d")
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=50.0)
    args = parser.parse_args()
    main(args.out_dir, t_min_ms=args.t_min_ms, t_max_ms=args.t_max_ms)
