"""Run the two fixed validation protocols for the mGluR-Gq-PLC model."""

import argparse
import csv
import json
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
from scripts.er_flux_decomp import (
    flux_series, plot_budget, plot_er_flux, write_flux_data,
)
from scripts.io_utils import read_columns, write_csv
from scripts.plot_style import apply_style
from scripts import plot_style


PERTURB_TIME = 100.0
RECOVERY_END = 1200.0
RECOVERY_WINDOW = 600.0
IP3_STEP_NM = 5.0
PULSE_END = 100.0
RTOL = 1e-8
ATOL = 1e-14


def reaction_rate(item, state, voltage=-65.0):
    _name, forward, f_order, reverse, r_order, _stoich = item
    value = forward(*(state[i] for i in f_order), voltage) if forward else 0.0
    if reverse:
        value -= reverse(*(state[i] for i in r_order), voltage)
    return float(value)


def pool_error(y, index, pool):
    total = y[[index[name] for name in pool]].sum(axis=0)
    return float(np.max(np.abs(total - total[0])) / max(abs(total[0]), 1.0))


def recovery_protocol(rest, index, compiled, voltage):
    first = solve_ivp(
        rhs, (0.0, PERTURB_TIME), rest, method="BDF",
        t_eval=np.linspace(0.0, PERTURB_TIME, 501),
        args=(compiled, voltage), rtol=RTOL, atol=ATOL)
    if not first.success:
        raise RuntimeError(first.message)
    perturbed = first.y[:, -1].copy()
    perturbed[index["IP3"]] += IP3_STEP_NM * 1e-9
    second = solve_ivp(
        rhs, (PERTURB_TIME, RECOVERY_END), perturbed, method="BDF",
        t_eval=np.linspace(PERTURB_TIME, RECOVERY_END, 2201),
        args=(compiled, voltage), rtol=RTOL, atol=ATOL)
    if not second.success:
        raise RuntimeError(second.message)
    return np.r_[first.t, second.t], np.c_[first.y, second.y], first, second


def stimulus_protocol(p, rest, index, compiled):
    initial = rest.copy()
    initial[index["Glu"]] = stimulus.molecules_to_conc(500.0, p)
    voltage = _make_voltage(stimulus.voltage(times=[0.0], V_rest=-65.0))
    times = np.unique(np.r_[
        np.linspace(0.0, 0.1, 501),
        np.linspace(0.15, PULSE_END, 1998),
    ])
    solution = _solve_with_pulses(
        rhs, (0.0, PULSE_END), initial, method="BDF", t_eval=times,
        args=(compiled, voltage),
        extra_boundaries=stimulus.voltage_onsets([0.0]),
        rtol=RTOL, atol=ATOL)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution, voltage


def plot_traces(out, recovery_t, recovery_y, pulse, index, activity, active_gq):
    recovery = {"time_s": recovery_t,
                "IP3_nM": recovery_y[index["IP3"]] * 1e9,
                "Ca_ER_uM": recovery_y[index["Ca_ER"]] * 1e6}
    stimulus_data = {"time_s": pulse.t,
                     "mGluR_R1": pulse.y[index["mGluR_Glu"]],
                     "mGluR_R2": pulse.y[index["mGluR_Glu2"]],
                     "Gq_active_molecules_um2": active_gq,
                     "IP3_nM": pulse.y[index["IP3"]] * 1e9,
                     "Ca_c_nM": pulse.y[index["Ca_c"]] * 1e9,
                     "Ca_ER_uM": pulse.y[index["Ca_ER"]] * 1e6}
    plot_trace_columns(out, recovery, stimulus_data)


def plot_trace_columns(out, recovery, stimulus_data):
    """Shared renderer for newly computed and recorded trace columns."""
    apply_style(plt)
    specs = (
        ("ip3", "IP3_nM", "IP3 (nM)", plot_style.color_for("IP3")),
        ("er", "Ca_ER_uM", r"$[Ca^{2+}]_{ER}$ (µM)", plot_style.color_for("Ca_ER")),
    )
    for suffix, key, label, color in specs:
        fig, ax = plot_style.figure()
        ax.plot(recovery["time_s"], recovery[key], color=color)
        ax.axvline(PERTURB_TIME, color=plot_style.COLORS["reference"], linestyle=":")
        ax.set(xlabel="t (s)", ylabel=label)
        plot_style.save(fig, out / f"recovery_{suffix}.png")
    fig, ax = plot_style.figure()
    for key, color in zip(("mGluR_R1", "mGluR_R2"), plot_style.colors(2)):
        ax.plot(stimulus_data["time_s"], stimulus_data[key], color=color,
                label=key.removeprefix("mGluR_"))
    ax.set(xlabel="t (s)", ylabel="Bound receptor fraction")
    plot_style.save(fig, out / "stimulus_receptor.png")
    for suffix, key, label, color in (*specs,
            ("gq", "Gq_active_molecules_um2", "Active Gq (molecules/µm²)", plot_style.shade("purple")),
            ("cyto", "Ca_c_nM", r"$[Ca^{2+}]_{cyto}$ (nM)", plot_style.color_for("Ca_c"))):
        fig, ax = plot_style.figure()
        ax.plot(stimulus_data["time_s"], stimulus_data[key], color=color)
        ax.set(xlabel="t (s)", ylabel=label)
        plot_style.save(fig, out / f"stimulus_{suffix}.png")


def write_model_tables(out, p):
    parameters = [
        ("k_mG_f", p.k_mG_f, "M^-1 s^-1", "original mGluR first binding"),
        ("k_mG_b", p.k_mG_b, "s^-1", "original mGluR first unbinding"),
        ("k_GG_f", p.k_GG_f, "M^-1 s^-1", "original mGluR second binding"),
        ("k_GG_b", p.k_GG_b, "s^-1", "original mGluR second unbinding"),
        ("plc_gq_eta", p.plc_gq_eta, "1", "equal R1/R2 efficacy assumption"),
        ("plc_gq_k_act", p.plc_gq_k_act, "s^-1", "22.2*(N_mGluR/A_PM)/Gq_total capacity mapping"),
        ("plc_gq_total_density", p.plc_gq_total_density, "molecules/um2", "10000 plus Gq released from receptor complexes"),
        ("plc_gq_plc_ca_fraction", p.plc_gq_plc_ca_fraction, "1", "successful resting-state mapping"),
    ]
    units = {
        7: ("s^-1", "s^-1"),
        8: ("um2 molecule^-1 s^-1", "s^-1"),
        9: ("M^-1 s^-1", "s^-1"),
        10: ("um2 molecule^-1 s^-1", "s^-1"),
        11: ("M^-1 s^-1", "s^-1"),
        12: ("s^-1", "um2 molecule^-1 s^-1"),
        13: ("s^-1", "um2 molecule^-1 s^-1"),
    }
    for number, (forward_unit, reverse_unit) in units.items():
        parameters.extend([
            (f"plc_gq_kf{number}", getattr(p, f"plc_gq_kf{number}"),
             forward_unit, "Calizo/Cooling VSMC"),
            (f"plc_gq_kr{number}", getattr(p, f"plc_gq_kr{number}"),
             reverse_unit, "Calizo/Cooling VSMC"),
        ])
    parameters.extend([
        ("plc_gq_kcat14", p.plc_gq_kcat14, "s^-1", "Calizo/Cooling activated catalysis"),
        ("plc_gq_km14", p.plc_gq_km14, "molecules/um2", "Calizo/Cooling"),
        ("plc_gq_kcat15", p.plc_gq_kcat15, "s^-1", "Calizo/Cooling basal catalysis"),
        ("plc_gq_km15", p.plc_gq_km15, "molecules/um2", "Calizo/Cooling"),
        ("plc_gq_pip2_density", p.plc_gq_pip2_density, "molecules/um2", "clamped Calizo/Cooling reservoir"),
        ("plc_gq_r15_scale", p.plc_gq_r15_scale, "1", "basal 20 nM IP3 fit after source removal"),
        ("plc_gq_k16", p.plc_gq_k16, "s^-1", "Calizo/Cooling R16 decay"),
        ("k_S_leak_gq", p.k_S_leak_gq, "s^-1", "150 uM ER-rest calibration"),
    ])
    with (out / "parameters.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "units", "provenance"])
        writer.writerows(parameters)

    reactions = [
        ("mG_bind1", "R0 + Glu <-> R1", "k_mG_f*R0*Glu - k_mG_b*R1"),
        ("mG_bind2", "R1 + Glu <-> R2", "k_GG_f*R1*Glu - k_GG_b*R2"),
        ("Gq_activation", "Gq_GDP -> Gq_GTP", "k_act*(R1 + eta*R2)/R_total*Gq_GDP"),
        ("R7", "Gq_GTP -> Gq_GDP", "kf7*Gq_GTP"),
        ("R8", "Gq_GTP + PLC <-> PLC_GqGTP", "kf8*Gq_GTP*PLC - kr8*PLC_GqGTP"),
        ("R9", "Ca + PLC <-> PLC_Ca", "kf9*Ca*PLC - kr9*PLC_Ca"),
        ("R10", "Gq_GTP + PLC_Ca <-> PLC_Ca_GqGTP", "kf10*Gq_GTP*PLC_Ca - kr10*PLC_Ca_GqGTP"),
        ("R11", "Ca + PLC_GqGTP <-> PLC_Ca_GqGTP", "kf11*Ca*PLC_GqGTP - kr11*PLC_Ca_GqGTP"),
        ("R12", "PLC_Ca_GqGTP -> Gq_GDP + PLC_Ca", "kf12*PLC_Ca_GqGTP"),
        ("R13", "PLC_GqGTP -> Gq_GDP + PLC", "kf13*PLC_GqGTP"),
        ("R14", "PIP2(clamped) -> IP3", "kcat14*PLC_Ca_GqGTP*PIP2/(Km14+PIP2)"),
        ("R15", "PIP2(clamped) -> IP3", "r15_scale*kcat15*PLC_Ca*PIP2/(Km15+PIP2)"),
        ("R16", "IP3 -> empty", "k16*IP3"),
    ]
    with (out / "reaction_mapping.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction", "stoichiometry", "rate"])
        writer.writerows(reactions)


def write_report(out, summary):
    rest = summary["resting_equilibrium"]
    recovery = summary["recovery"]
    pulse = summary["stimulus"]
    text = f"""# mGluR-Gq-PLC result

The cleaned `new` pathway combines the original two-step glutamate-mGluR kinetics with one effective receptor-driven Gq activation step and the Calizo/Cooling R7-R16 downstream branch. R16 is decay only; there is no reference IP3 source. The legacy pathway remains available as `--plc old`.

## Equations and scaling

Receptor occupancy is `a_R = (R1 + eta R2)/R_total`, with `eta = 1`, and activation is `v_act = k_act a_R Gq_GDP`. This changes only Gq-GDP to Gq-GTP; it does not consume receptor or glutamate. R14 and R15 use the published Michaelis-Menten forms listed in `reaction_mapping.csv`; R15 supplies basal IP3 and R16 removes it as `k16 IP3`.

`_surface_rx` is only a small constructor that assigns each reaction to the plasma membrane. It does not merge reactions: reversible receptor binding, Gq cycling, PLC binding, and catalysis remain separate `Reaction` objects. The redundant `mG_cat1` and `mG_cat2` parallel unbinding reactions are not part of this pathway. Bulk Glu consumption uses `N_mGluR/(N_A V_spine)`; surface Ca binding and IP3 production use `A_PM/(N_A V_spine)` to convert molecules/µm²/s into M/s.

## Validation

Two full validation protocols were run after a derivative-converged resting preparation (RyR and SOCE off, -65 mV). Equilibrium was reached at {rest['time_s']:.0f} s: IP3 {rest['IP3_nM']:.4f} nM, cytosolic calcium {rest['Ca_c_nM']:.4f} nM, and ER calcium {rest['Ca_ER_uM']:.4f} µM. The final ER fluxes were SERCA {rest['SERCA_ions_s']:.4f}, IP3R {rest['IP3R_ions_s']:.4f}, leak {rest['ER_leak_ions_s']:.4f}, and net {rest['ER_net_ions_s']:+.3e} ions/s.

After deleting the R16 reference source, `r15_scale` was increased fourfold to 0.521119220 so basal R15 production (2.5 nM/s) balances R16 decay at 20 nM. The existing Gq-specific ER leak, 0.101713805 s^-1, then retained the 150 µM ER target without further fitting.

The +5 nM IP3 perturbation recovered within {recovery['recovery_time_s']:.1f} s and remained within 0.1 nM IP3 and 0.5 µM ER calcium through 1200 s. Final ER calcium was {recovery['final_ER_uM']:.4f} µM.

One 500-molecule glutamate pulse with the existing EPSP+bAP stimulus gave bound receptor, active Gq, and IP3 responses. Peak IP3 was {pulse['peak_IP3_nM']:.3f} nM at {pulse['time_to_peak_IP3_s']:.3f} s; cytosolic calcium peaked at {pulse['peak_Ca_c_nM']:.3f} nM; ER calcium fell to {pulse['minimum_Ca_ER_uM']:.3f} µM and returned to {pulse['final_Ca_ER_uM']:.3f} µM by 100 s. This confirms the qualitative coupling and calcium response; stimulated rates were not refit and numerical agreement with the deleted experimental receptor model is not claimed.

## Limits

`k_act = 1.291678304 s^-1` was computed as `22.2 s^-1 * (N_mGluR/A_PM) / Gq_total`, preserving the former R5 maximum activation capacity at the present receptor and Gq abundance. It is not the R5 rate constant. Its calibration and receptor-number sensitivity remain unresolved. The restored ligand-binding constants are neuronal-model values, but their validity for this hybrid coupling has not been independently established. Stable resting ER calcium does not validate dendritic kinetics.

Validation used the repository's 0D ODE assembly. The spatial SMART builder selects the same reaction registrar, but no spatial solve was added to this limited run. No scaling defect was found in the inspected assembly; unit tests verify the surface-to-bulk factors.

Run `python scripts/plc_gq_analysis.py` to regenerate the RyR-off analysis, including its two-panel ER/cytosol decomposition. Regenerate the separate RyR-on files with `python scripts/er_flux_decomp.py --plc new --equilibrate-first --stim --glu-molecules 500 --final-t 100 --n-points 2499 --out results/Flux/plc_gq_historical_ryr_on/er_cyto_flux_decomp_ryr_on.png --data-out results/Flux/plc_gq_historical_ryr_on/er_cyto_flux_decomp_ryr_on.csv`.
"""
    report = ROOT / "doc/PLC_Gq_validation_report.md"
    report.write_text(text)


def _budget_variants(p):
    """Comparison sets applied in memory; defaults in parameters.py are untouched."""
    return {
        "previous": {"k_mG_b": 7.85, "plc_gq_kcat14": 10.0, "plc_gq_k16": 0.5},
        "current": {},
        "k_act_x3": {"plc_gq_k_act": p.plc_gq_k_act * 3},
        "k_act_x3_k16_2": {"plc_gq_k_act": p.plc_gq_k_act * 3, "plc_gq_k16": 2.0},
    }


PAIRING_DELAYS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
BUDGET_LABELS = {"previous": "Previous", "current": "Current", "k_act_x3": "k_act x3",
                 "k_act_x3_k16_2": "k_act x3, k16 = 2"}


def _half_width(t, y):
    level = y[0] + 0.5 * (y.max() - y[0])
    above = np.flatnonzero(y >= level)
    return float(t[above[-1]] - t[above[0]])


def _budget_trace(p, names, compiled, rest, *, glutamate, spike_time, end=40.0):
    index = {name: i for i, name in enumerate(names)}
    y = rest.copy()
    y[index["Glu"]] = stimulus.molecules_to_conc(500, p) if glutamate else 0.0
    if spike_time is None:
        voltage, boundaries = _make_voltage(-65.0), ()
        t = np.unique(np.r_[np.linspace(0, .1, 201), np.linspace(.1, end, 1996)])
    else:
        voltage = _make_voltage(stimulus.voltage(times=[spike_time], V_rest=-65.0))
        boundaries = stimulus.voltage_onsets([spike_time])
        t = np.unique(np.r_[np.linspace(0, end, 2001),
                            np.linspace(spike_time, spike_time + .1, 501),
                            np.linspace(spike_time + .1, spike_time + 1, 451)])
    sol = _solve_with_pulses(rhs, (0, end), y, method="BDF", t_eval=t, args=(compiled, voltage),
                             extra_boundaries=boundaries, rtol=1e-8, atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return t, sol.y, index


def _budget_columns(p, t, y, index):
    pip2 = p.plc_gq_pip2_density
    scale = p.A_PM / (p.N_A * p.V_spine_L) * 1e9
    r14 = p.plc_gq_kcat14 * y[index["PLC_Ca_GqGTP"]] * pip2 / (p.plc_gq_km14 + pip2) * scale
    r15 = (p.plc_gq_r15_scale * p.plc_gq_kcat15 * y[index["PLC_Ca"]] * pip2
           / (p.plc_gq_km15 + pip2) * scale)
    ip3 = y[index["IP3"]] * 1e9
    return {"time_s": t, "R14_nM_s": r14, "R15_nM_s": r15, "production_nM_s": r14 + r15,
            "decay_nM_s": p.plc_gq_k16 * ip3, "IP3_nM": ip3,
            "IP3R_Popen": y[index["IPR_110"]] ** 3,
            "occupancy": y[index["mGluR_Glu"]] + y[index["mGluR_Glu2"]],
            "Gq_GTP": y[index["Gq_GTP"]], "PLC_Ca": y[index["PLC_Ca"]],
            "PLC_Ca_GqGTP": y[index["PLC_Ca_GqGTP"]],
            "Ca_c_nM": y[index["Ca_c"]] * 1e9, "Ca_ER_uM": y[index["Ca_ER"]] * 1e6}


def _budget_variant(key, overrides, out):
    from scripts.calibrate_rest import fit
    values, names, compiled, eq, p = fit(overrides, tol=1e-5)
    # Each protocol starts from this variant's own fitted rest.
    glu_only = _budget_columns(p, *_budget_trace(p, names, compiled, eq.state,
                                                 glutamate=True, spike_time=None))
    full = _budget_columns(p, *_budget_trace(p, names, compiled, eq.state,
                                             glutamate=True, spike_time=0.0))
    write_csv(out / f"{key}_glu_only.csv", glu_only)
    write_csv(out / f"{key}_glu_spike.csv", full)
    t, po = glu_only["time_s"], glu_only["IP3R_Popen"]
    prod, ip3, er = glu_only["production_nM_s"], glu_only["IP3_nM"], full["Ca_ER_uM"]
    peak = int(po.argmax())
    low = int(er.argmin())
    half_er = np.flatnonzero((full["time_s"] > full["time_s"][low])
                             & (er >= er[low] + .5 * (er[0] - er[low])))
    row = {"overrides": overrides, "fitted": {k: float(v) for k, v in values.items()},
           "IP3_rest_nM": float(ip3[0]), "IP3_peak_nM": float(ip3.max()),
           "IP3_peak_s": float(t[ip3.argmax()]), "IP3R_peak": float(po[peak]),
           "IP3R_peak_s": float(t[peak]), "IP3R_half_width_s": _half_width(t, po),
           "IP3R_10pct_rise_s": float(t[np.flatnonzero(po >= po[0] + .1 * (po[peak] - po[0]))[0]]),
           "production_peak_nM_s": float(prod.max()), "production_peak_s": float(t[prod.argmax()]),
           "production_half_width_s": _half_width(t, prod),
           "R15_rest_nM_s": float(glu_only["R15_nM_s"][0]),
           "R15_min_nM_s": float(glu_only["R15_nM_s"].min()),
           "Gq_peak": float(glu_only["Gq_GTP"].max()),
           "Gq_half_width_s": _half_width(t, glu_only["Gq_GTP"]),
           "full_stimulus": {"ER_min_uM": float(er[low]), "ER_min_s": float(full["time_s"][low]),
                             "ER_half_recovery_s": float(full["time_s"][half_er[0]] - full["time_s"][low])
                             if len(half_er) else None,
                             "ER_at_5s_uM": float(np.interp(5.0, full["time_s"], er)),
                             "IP3_peak_nM": float(full["IP3_nM"].max())}}
    return key, row


def _pairing_window(key, overrides):
    """RyR off: IP3R release in the 0.3 s after a spike, against glutamate and spike alone."""
    from scripts.calibrate_rest import fit
    values = fit(overrides, tol=1e-5)[0]
    p = default_parameters()
    p.Glu_init = 0.0
    for name, value in values.items():
        setattr(p, name, value)
    names, initial, compiled, _ = build_system(p, ryr_off=True)
    rest = equilibrate_state(initial, compiled, _make_voltage(-65.0), names=names,
                             max_time_s=4000).state

    def release(t, y, index, start):
        window = (t >= start) & (t <= start + .3)
        flux = (p.k_IP3R_flux * y[index["IPR_110"]] ** 3
                * (y[index["Ca_ER"]] - y[index["Ca_c"]]) * p.N_IP3R)
        return float(np.trapezoid(flux[window], t[window]))

    glu = _budget_trace(p, names, compiled, rest, glutamate=True, spike_time=None)
    spike = release(*_budget_trace(p, names, compiled, rest, glutamate=False, spike_time=0.0), 0.0)
    rows = []
    for delay in PAIRING_DELAYS:
        paired = release(*_budget_trace(p, names, compiled, rest, glutamate=True, spike_time=delay), delay)
        alone = release(*glu, delay)
        rows.append({"delay_s": delay, "paired_ions": paired, "glutamate_ions": alone,
                     "spike_ions": spike, "synergy_ions": paired - alone - spike,
                     "ratio": paired / (alone + spike)})
    return key, rows


def plot_ip3_budget(out):
    plot_style.apply_style(plt)
    summary = json.loads((out / "ip3_budget_summary.json").read_text())
    keys = list(summary["variants"])
    colors = plot_style.colors(len(keys))
    data = {key: read_columns(out / f"{key}_glu_only.csv") for key in keys}
    for field, ylabel, name in (("IP3_nM", "IP3 (nM)", "ip3"),
                                ("IP3R_Popen", "IP3R open probability", "ip3r_popen"),
                                ("production_nM_s", "IP3 production (nM/s)", "production"),
                                ("Gq_GTP", "Gq-GTP (molecules/µm²)", "gq_gtp")):
        fig, ax = plot_style.figure()
        for key, color in zip(keys, colors):
            d = data[key]
            mask = d["time_s"] <= 5
            ax.plot(d["time_s"][mask], d[field][mask], color=color, label=BUDGET_LABELS[key])
        ax.set(xlabel="Time after glutamate (s)", ylabel=ylabel, xlim=(0, 5))
        plot_style.save(fig, out / f"{name}_5s.png", legend_name="variants_legend.png")
    for key in keys:
        d = data[key]
        mask = d["time_s"] <= 5
        fig, ax = plot_style.figure()
        for field, label, color in zip(("R14_nM_s", "R15_nM_s", "decay_nM_s"),
                                       ("R14 (PLC-Ca-Gq)", "R15 (PLC-Ca)", "Decay"),
                                       plot_style.colors(3)):
            ax.plot(d["time_s"][mask], d[field][mask], color=color, label=label)
        ax.set(xlabel="Time after glutamate (s)", ylabel="IP3 rate (nM/s)", xlim=(0, 5))
        plot_style.save(fig, out / f"budget_{key}.png", legend_name="budget_legend.png")
    fig, ax = plot_style.figure()
    for key, color in zip(summary["pairing"], plot_style.colors(len(summary["pairing"]))):
        rows = summary["pairing"][key]
        ax.plot([r["delay_s"] for r in rows], [r["synergy_ions"] for r in rows], marker="o",
                markersize=3, color=color, label=BUDGET_LABELS[key])
    ax.axhline(0, color=plot_style.COLORS["reference"], linestyle=":", linewidth=1)
    ax.set(xlabel="Spike delay after glutamate (s)", ylabel="Synergistic IP3R release (ions)")
    plot_style.save(fig, out / "pairing_window.png", legend_name="pairing_legend.png")


def ip3_budget(out):
    from joblib import Parallel, delayed
    out.mkdir(parents=True, exist_ok=True)
    variants = _budget_variants(default_parameters())
    rows = Parallel(n_jobs=4)(delayed(_budget_variant)(key, value, out)
                              for key, value in variants.items())
    windows = Parallel(n_jobs=4)(delayed(_pairing_window)(key, value) for key, value in variants.items())
    summary = {"protocol": "500 glutamate molecules; RyR on, SOCE off; rest refit per variant",
               "variants": dict(rows), "pairing": dict(windows)}
    (out / "ip3_budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_ip3_budget(out)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/plc_gq"))
    parser.add_argument("--flux-out", type=pathlib.Path,
                        default=pathlib.Path("results/Flux/plc_gq_validation"))
    parser.add_argument("--ip3-budget", action="store_true",
                        help="trace R14/R15/decay and compare parameter sets, then exit")
    parser.add_argument("--budget-out", type=pathlib.Path,
                        default=ROOT / "results/plc_gq/ip3_budget")
    args = parser.parse_args(argv)
    if args.ip3_budget:
        ip3_budget(args.budget_out)
        return
    args.flux_out.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    p = default_parameters()
    p.Glu_init = 0.0
    names, initial, compiled, _ = build_system(p, ryr_off=True, plc="new")
    index = {name: i for i, name in enumerate(names)}
    rest_voltage = _make_voltage(-65.0)
    equilibrium = equilibrate_state(
        initial, compiled, rest_voltage, names=names, rtol=RTOL, atol=ATOL)
    if not equilibrium.converged:
        raise RuntimeError("resting state did not converge")
    rest = equilibrium.state

    recovery_t, recovery_y, first, second = recovery_protocol(
        rest, index, compiled, rest_voltage)
    pulse, pulse_voltage = stimulus_protocol(p, rest, index, compiled)

    r1 = pulse.y[index["mGluR_Glu"]]
    r2 = pulse.y[index["mGluR_Glu2"]]
    activity = (r1 + p.plc_gq_eta * r2) / p.mGluR_total
    active_gq = (pulse.y[index["Gq_GTP"]]
                 + pulse.y[index["PLC_GqGTP"]]
                 + pulse.y[index["PLC_Ca_GqGTP"]])

    write_csv(args.out / "recovery_trace.csv", {
        "time_s": recovery_t,
        "IP3_nM": recovery_y[index["IP3"]] * 1e9,
        "Ca_c_nM": recovery_y[index["Ca_c"]] * 1e9,
        "Ca_ER_uM": recovery_y[index["Ca_ER"]] * 1e6,
    })
    write_csv(args.out / "stimulus_trace.csv", {
        "time_s": pulse.t,
        "V_m_mV": [pulse_voltage(t) for t in pulse.t],
        "Glu_uM": pulse.y[index["Glu"]] * 1e6,
        "mGluR_R0": pulse.y[index["mGluR_free"]],
        "mGluR_R1": r1, "mGluR_R2": r2,
        "receptor_activity": activity,
        "Gq_active_molecules_um2": active_gq,
        "IP3_nM": pulse.y[index["IP3"]] * 1e9,
        "Ca_c_nM": pulse.y[index["Ca_c"]] * 1e9,
        "Ca_ER_uM": pulse.y[index["Ca_ER"]] * 1e6,
    })

    er_flux = flux_series(
        p, names, compiled, recovery_t, recovery_y, rest_voltage)["ER"]
    write_csv(args.flux_out / "er_flux_trace.csv", {
        "time_s": recovery_t,
        **{name.replace(" ", "_") + "_ions_s": value
           for name, value in er_flux.items()},
    })
    plot_er_flux(
        args.flux_out / "er_flux_balance.png", recovery_t, er_flux,
        marker_time=PERTURB_TIME)
    pulse_flux = flux_series(
        p, names, compiled, pulse.t, pulse.y, pulse_voltage)
    plot_budget(
        args.flux_out / "er_cyto_flux_decomp.png", p, names,
        pulse.t, pulse.y, pulse_flux)
    write_flux_data(
        args.flux_out / "er_cyto_flux_decomp.csv", p, names,
        pulse.t, pulse.y, pulse_flux)
    plot_traces(
        args.out, recovery_t, recovery_y, pulse, index, activity, active_gq)

    final_er = {name: values[-1] for name, values in er_flux.items()}
    post_ip3 = second.y[index["IP3"]] * 1e9
    post_er = second.y[index["Ca_ER"]] * 1e6
    recovered = ((abs(post_ip3 - rest[index["IP3"]] * 1e9) <= 0.1)
                 & (abs(post_er - rest[index["Ca_ER"]] * 1e6) <= 0.5))
    maintained = np.logical_and.accumulate(recovered[::-1])[::-1]
    candidates = np.flatnonzero(
        maintained & (second.t <= PERTURB_TIME + RECOVERY_WINDOW))
    recovery_time = (float(second.t[candidates[0]] - PERTURB_TIME)
                     if candidates.size else None)
    if recovery_time is None:
        raise RuntimeError("IP3/ER recovery was not maintained within 600 s")

    ip3_index = index["IP3"]
    reactions = {item[0]: item for item in compiled}
    ip3_flux = {
        name: reactions[name][-1].get(ip3_index, 0.0)
        * reaction_rate(reactions[name], rest) * 1e9
        for name in ("R14", "R15", "R16_IP3_decay")
    }
    receptor_pool = ("mGluR_free", "mGluR_Glu", "mGluR_Glu2")
    gq_pool = ("Gq_GDP", "Gq_GTP", "PLC_GqGTP", "PLC_Ca_GqGTP")
    plc_pool = ("PLC", "PLC_GqGTP", "PLC_Ca", "PLC_Ca_GqGTP")
    pulse_ip3 = pulse.y[index["IP3"]] * 1e9
    pulse_ca = pulse.y[index["Ca_c"]] * 1e9
    pulse_er = pulse.y[index["Ca_ER"]] * 1e6
    summary = {
        "validation_protocols": 2,
        "solver": {"method": "BDF", "rtol": RTOL, "atol": ATOL},
        "resting_equilibrium": {
            "converged": equilibrium.converged,
            "time_s": equilibrium.time_s,
            "Ca_relative_rate_s-1": equilibrium.ca_relative_rate_s,
            "largest_other_derivative": equilibrium.other_rate,
            "IP3_nM": rest[index["IP3"]] * 1e9,
            "Ca_c_nM": rest[index["Ca_c"]] * 1e9,
            "Ca_ER_uM": rest[index["Ca_ER"]] * 1e6,
            "R14_IP3_nM_s": ip3_flux["R14"],
            "R15_IP3_nM_s": ip3_flux["R15"],
            "R16_decay_nM_s": ip3_flux["R16_IP3_decay"],
            "IP3_net_nM_s": sum(ip3_flux.values()),
            "SERCA_ions_s": final_er.get("SERCA", 0.0),
            "IP3R_ions_s": final_er.get("IP3R", 0.0),
            "ER_leak_ions_s": final_er.get("ER leak", 0.0),
            "ER_net_ions_s": final_er["net"],
        },
        "recovery": {
            "IP3_step_nM": IP3_STEP_NM,
            "recovery_window_s": RECOVERY_WINDOW,
            "recovery_time_s": recovery_time,
            "pre_step_max_IP3_drift_nM": float(np.max(abs(
                first.y[index["IP3"]] * 1e9 - rest[index["IP3"]] * 1e9))),
            "pre_step_max_ER_drift_uM": float(np.max(abs(
                first.y[index["Ca_ER"]] * 1e6 - rest[index["Ca_ER"]] * 1e6))),
            "final_IP3_nM": float(post_ip3[-1]),
            "final_ER_uM": float(post_er[-1]),
        },
        "stimulus": {
            "glutamate_molecules": 500.0,
            "EPSP_bAP": True,
            "peak_receptor_activity": float(activity.max()),
            "peak_Gq_active_molecules_um2": float(active_gq.max()),
            "peak_IP3_nM": float(pulse_ip3.max()),
            "time_to_peak_IP3_s": float(pulse.t[pulse_ip3.argmax()]),
            "peak_Ca_c_nM": float(pulse_ca.max()),
            "minimum_Ca_ER_uM": float(pulse_er.min()),
            "final_Ca_ER_uM": float(pulse_er[-1]),
            "qualitative_pathway_response": bool(
                activity.max() > 0 and active_gq.max() > 0
                and pulse_ip3.max() > rest[index["IP3"]] * 1e9
                and pulse_ca.max() > rest[index["Ca_c"]] * 1e9
                and pulse_er.min() < rest[index["Ca_ER"]] * 1e6),
            "receptor_conservation_error": pool_error(
                pulse.y, index, receptor_pool),
            "Gq_conservation_error": pool_error(pulse.y, index, gq_pool),
            "PLC_conservation_error": pool_error(pulse.y, index, plc_pool),
            "minimum_state": float(pulse.y.min()),
        },
    }
    with (args.out / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_model_tables(args.out, p)
    write_report(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
