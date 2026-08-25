"""Plot the resting-IP3 target and two-dimensional mGluR audit."""

import argparse
import csv
import os
import pathlib
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/erpm_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_style


BLUE = "#437bbe"
GREEN = "#4eb748"
ORANGE = "#d17a22"
GRAY = "#555555"


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_both(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output} and {output.with_suffix('.pdf')}")


def plot_target(input_dir, output):
    rows = [row for row in read_rows(input_dir / "ip3_target_curve.csv")
            if row["protocol"] == "target_curve"]
    rows.sort(key=lambda row: float(row["IP3_setpoint_nM"]))
    selected = next(row for row in rows if row["selected"].lower() == "true")
    target_ip3 = float(selected["IP3_setpoint_nM"])

    x = np.array([float(row["IP3_setpoint_nM"]) for row in rows])
    ca_er = np.array([float(row["final_Ca_ER_uM"]) for row in rows])
    serca = np.array([float(row["SERCA_uptake_ions_s"]) for row in rows])
    ip3r = np.array([float(row["IP3R_release_ions_s"]) for row in rows])

    plot_style.apply_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    ax = axes[0]
    ax.plot(x, ca_er, marker="o", markersize=4.5, color=GREEN)
    ax.axhline(150.0, color=GRAY, ls="--", lw=1.1)
    ax.axvline(target_ip3, color=GRAY, ls=":", lw=1.1)
    ax.scatter([target_ip3], [float(selected["final_Ca_ER_uM"])],
               s=42, zorder=4, color=ORANGE)
    ax.annotate(
        f"{target_ip3:.2f} nM",
        (target_ip3, 150.0),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set(xlabel="IP3 setpoint (nM)", ylabel=r"Equilibrium $Ca_{ER}$ (µM)")
    plot_style.format_axis(ax)

    ax = axes[1]
    ax.plot(x, serca, marker="o", markersize=4.2, color=BLUE,
            label="SERCA uptake")
    ax.plot(x, ip3r, marker="s", markersize=4.0, color=ORANGE, ls="--",
            markerfacecolor="white", markeredgewidth=1.2,
            label="IP3R release")
    ax.axvline(target_ip3, color=GRAY, ls=":", lw=1.1)
    ax.set(xlabel="IP3 setpoint (nM)", ylabel="Equilibrium ER flux (ions/s)")
    ax.legend(frameon=False, fontsize=9)
    plot_style.format_axis(ax)

    fig.tight_layout(w_pad=2.6)
    save_both(fig, output)


def value_matrix(rows, row_factors, column_factors, field):
    lookup = {
        (float(row["mGluR_total_factor"]), float(row["N_mGluR_factor"])):
        float(row[field])
        for row in rows
    }
    return np.array([
        [lookup[(row_factor, column_factor)]
         for column_factor in column_factors]
        for row_factor in row_factors
    ])


def color_limits(values):
    low, high = float(np.min(values)), float(np.max(values))
    center = float(np.mean(values))
    if high - low <= max(abs(center) * 1e-8, 1e-6):
        padding = max(abs(low) * 1e-6, 1e-6)
        return low - padding, high + padding, True
    return low, high, False


def plot_mglur(input_dir, output):
    resting = read_rows(input_dir / "mglur_resting_sweep.csv")
    control = read_rows(input_dir / "mglur_glutamate_control.csv")
    row_factors = sorted({float(row["mGluR_total_factor"]) for row in resting})
    column_factors = sorted({float(row["N_mGluR_factor"]) for row in resting})
    panels = (
        (resting, "final_IP3_nM", "Resting IP3 (nM)"),
        (resting, "final_Ca_ER_uM", r"Resting $Ca_{ER}$ (µM)"),
        (control, "peak_IP3_increase_nM", "Peak ΔIP3 (nM)"),
        (control, "integrated_mGluR_IP3_nM", "Integrated mGluR IP3 (nM)"),
    )

    plot_style.apply_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.7))
    for ax, (rows, field, colorbar_label) in zip(axes.flat, panels):
        values = value_matrix(rows, row_factors, column_factors, field)
        low, high, uniform = color_limits(values)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis",
                          vmin=low, vmax=high)
        ax.set_xticks(range(len(column_factors)),
                      [f"{value:g}" for value in column_factors])
        ax.set_yticks(range(len(row_factors)),
                      [f"{value:g}" for value in row_factors])
        ax.set_xlabel(r"$N_{mGluR}$ factor")
        ax.set_ylabel("mGluR total factor")
        ax.tick_params(direction="out", length=3)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
        colorbar.set_label(colorbar_label, fontsize=11)
        colorbar.ax.tick_params(labelsize=9)
        if uniform:
            center = float(np.mean(values))
            colorbar.set_ticks([center], labels=[f"{center:.4g}"])

    fig.tight_layout(w_pad=1.7, h_pad=2.0)
    save_both(fig, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("results/ip3r_resting_analysis"))
    parser.add_argument(
        "--target-out",
        type=pathlib.Path,
        default=pathlib.Path("results/ip3r_resting_analysis/ip3_target.png"),
    )
    parser.add_argument(
        "--mglur-out",
        type=pathlib.Path,
        default=pathlib.Path("results/ip3r_resting_analysis/mglur_sweep.png"),
    )
    args = parser.parse_args(argv)
    plot_target(args.input, args.target_out)
    plot_mglur(args.input, args.mglur_out)


if __name__ == "__main__":
    main()
