"""Plot ER and cytosolic calcium fluxes from the 0D model."""

import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import stimulus
from model.ode0d import (
    _make_voltage, _solve_with_pulses, build_system, equilibrate_state, rhs,
)
from model.parameters import default_parameters
from scripts.plot_style import apply_style, format_axis
from scripts import plot_style


ER_REACTIONS = {
    "S_y2_y1": "SERCA", "S_y1_y0": "SERCA",
    "S_leak": "ER leak", "RyR_CICR": "RyR", "IP3R_flux": "IP3R",
}
CYTO_REACTIONS = {
    "V_Ca_entry": "PM influx", "N_Ca_entry": "PM influx",
    "SOCE_Ca_entry": "PM influx", "P_leak": "PM influx",
    "N_leak": "PM influx", "P_bind": "PM efflux", "N_bind": "PM efflux",
    "S_leak": "ER exchange", "RyR_CICR": "ER exchange",
    "IP3R_flux": "ER exchange", "S_x_x1": "SERCA binding",
    "S_x1_x2": "SERCA binding", "Bm_bind": "buffering",
    "Bf_bind": "buffering", "PLC_bindCa": "buffering",
    "R9": "buffering", "R11": "buffering",
    "Ca_lumped_decay": "Ca decay",
}


def _net(item, state, voltage):
    _name, forward, f_order, reverse, r_order, _stoich = item
    value = forward(*(state[i] for i in f_order), voltage) if forward else 0.0
    if reverse:
        value -= reverse(*(state[i] for i in r_order), voltage)
    return float(value)


def flux_series(p, names, compiled, t, y, voltage):
    """Return grouped contributions in ions/s for each compartment."""
    index = {name: i for i, name in enumerate(names)}
    groups = {"ER": {}, "cytosol": {}}
    targets = (
        ("ER", index["Ca_ER"], float(p.V_ER) * 1e-15, ER_REACTIONS),
        ("cytosol", index["Ca_c"], float(p.V_spine_L), CYTO_REACTIONS),
    )
    for compartment, target, volume, mapping in targets:
        for item in compiled:
            name, *_rest, stoich = item
            if target not in stoich:
                continue
            label = mapping.get(name)
            if compartment == "cytosol" and name.startswith(("RyR_act_", "RyR_inact_")):
                label = "buffering"
            if label is None:
                continue
            values = np.array([
                stoich[target] * _net(item, y[:, k], voltage(t[k]))
                * float(p.N_A) * volume
                for k in range(t.size)
            ])
            groups[compartment][label] = (
                groups[compartment].get(label, np.zeros_like(t)) + values)
    for values in groups.values():
        values["net"] = sum(values.values(), np.zeros_like(t))
    return groups


def plot_er_flux(path, t, er_flux, *, marker_time=None):
    """Plot ER fluxes; positive values add calcium to the ER."""
    apply_style(plt)
    fig, axis = plot_style.figure()
    for name in ("SERCA", "IP3R", "ER leak", "RyR"):
        if name in er_flux:
            axis.plot(t, er_flux[name], label=name, color=plot_style.PATHWAY_COLORS[name])
    axis.plot(t, er_flux["net"], color=plot_style.PATHWAY_COLORS["net"], linestyle="--", label="net")
    axis.axhline(0.0, color=plot_style.COLORS["reference"], linewidth=0.7)
    if marker_time is not None:
        axis.axvline(marker_time, color=plot_style.COLORS["reference"], linestyle=":")
    axis.set(xlabel="t (s)", ylabel="ER calcium flux (ions/s)")
    axis.legend(frameon=False, ncol=2)
    format_axis(axis)
    fig.tight_layout()
    plot_style.save(fig, path)


def cumulative(values, time):
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(time))
    return result


def plot_budget(path, p, names, t, y, flux):
    """Plot the large ER and cytosol cumulative calcium budget."""
    apply_style(plt)
    index = {name: i for i, name in enumerate(names)}
    counts = {
        "ER": y[index["Ca_ER"]] * p.N_A * p.V_ER * 1e-15,
        "cytosol": y[index["Ca_c"]] * p.N_A * p.V_spine_L,
    }
    plot_budget_counts(path, t, counts, flux)


def plot_budget_counts(path, t, counts, flux):
    """Render supplied recorded counts/fluxes without consulting model defaults."""
    apply_style(plt)
    order = {
        "ER": ("SERCA", "ER leak", "IP3R", "RyR"),
        "cytosol": ("PM influx", "PM efflux", "ER exchange",
                     "SERCA binding", "buffering", "Ca decay"),
    }
    for compartment in ("ER", "cytosol"):
        fig, axis = plot_style.figure()
        for label in order[compartment]:
            if label in flux[compartment]:
                axis.plot(t, cumulative(flux[compartment][label], t),
                          color=plot_style.PATHWAY_COLORS[label], label=label)
        measured = counts[compartment] - counts[compartment][0]
        summed = cumulative(flux[compartment]["net"], t)
        axis.plot(t, measured, color=plot_style.PATHWAY_COLORS["recorded change"],
                  label="recorded change")
        axis.plot(t, summed, color=plot_style.PATHWAY_COLORS["flux sum"],
                  linestyle="--", label="flux sum")
        axis.axhline(0.0, color=plot_style.COLORS["reference"], linewidth=0.7)
        axis.set(xlabel="t (s)", ylabel=f"Ca change in {compartment} (ions)")
        plot_style.save(fig, plot_style.panel_path(path, compartment))


def write_flux_data(path, p, names, t, y, flux):
    index = {name: i for i, name in enumerate(names)}
    columns = {"time_s": t}
    for compartment in ("ER", "cytosol"):
        prefix = "ER" if compartment == "ER" else "cyto"
        for label, values in flux[compartment].items():
            key = f"{prefix}_{label.replace(' ', '_')}_ions_s"
            columns[key] = values
        volume = p.V_ER * 1e-15 if compartment == "ER" else p.V_spine_L
        calcium = "Ca_ER" if compartment == "ER" else "Ca_c"
        count = y[index[calcium]] * p.N_A * volume
        columns[f"{prefix}_recorded_change_ions"] = count - count[0]
        columns[f"{prefix}_flux_sum_ions"] = cumulative(
            flux[compartment]["net"], t)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def run_trajectory(p, *, plc="new", v_rest=-65.0, final_t=2.0,
                   n_points=2000, glu_molecules=0.0, equilibrate_first=False,
                   stim_voltage=False, include_soce=False, ryr_off=False,
                   ip3r_off=False):
    names, y0, compiled, glu_index = build_system(
        p, plc=plc, include_soce=include_soce, ryr_off=ryr_off,
        ip3r_off=ip3r_off)
    rest_voltage = _make_voltage(v_rest)
    if glu_index is not None:
        y0[glu_index] = 0.0
    if equilibrate_first:
        equilibrium = equilibrate_state(y0, compiled, rest_voltage, names=names)
        if not equilibrium.converged:
            raise RuntimeError("resting state did not converge")
        y0 = equilibrium.state
    if glu_molecules and glu_index is not None:
        y0[glu_index] = stimulus.molecules_to_conc(glu_molecules, p)
    voltage = (_make_voltage(stimulus.voltage(times=[0.0], V_rest=v_rest))
               if stim_voltage else rest_voltage)
    t = np.linspace(0.0, final_t, n_points)
    solve = _solve_with_pulses if stim_voltage else solve_ivp
    kwargs = {"extra_boundaries": stimulus.voltage_onsets([0.0])} \
        if stim_voltage else {}
    solution = solve(
        rhs, (0.0, final_t), y0, method="BDF", t_eval=t,
        args=(compiled, voltage), rtol=1e-9, atol=1e-14, **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    return names, compiled, voltage, solution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plc", choices=("old", "new"), default="new")
    parser.add_argument("--final-t", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--v-rest", type=float, default=-65.0)
    parser.add_argument("--glu-molecules", type=float, default=0.0)
    parser.add_argument("--stim", action="store_true")
    parser.add_argument("--equilibrate-first", "--equilibrate",
                        action="store_true")
    parser.add_argument("--soce", action="store_true")
    parser.add_argument("--ryroff", action="store_true")
    parser.add_argument("--ip3r-off", action="store_true")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/Flux/standalone/er_flux.png"))
    parser.add_argument("--data-out", type=pathlib.Path)
    args = parser.parse_args(argv)

    p = default_parameters()
    names, compiled, voltage, solution = run_trajectory(
        p, plc=args.plc, v_rest=args.v_rest, final_t=args.final_t,
        n_points=args.n_points, glu_molecules=args.glu_molecules,
        equilibrate_first=args.equilibrate_first, stim_voltage=args.stim,
        include_soce=args.soce, ryr_off=args.ryroff,
        ip3r_off=args.ip3r_off)
    flux = flux_series(p, names, compiled, solution.t, solution.y, voltage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_out = args.data_out or args.out.with_suffix(".csv")
    data_out.parent.mkdir(parents=True, exist_ok=True)
    plot_budget(args.out, p, names, solution.t, solution.y, flux)
    write_flux_data(data_out, p, names, solution.t, solution.y, flux)
    print(f"Saved panels {args.out.stem}_ER.png / {args.out.stem}_cytosol.png and {data_out}; "
          f"final ER net flux = {flux['ER']['net'][-1]:+.3e} ions/s")


if __name__ == "__main__":
    main()
