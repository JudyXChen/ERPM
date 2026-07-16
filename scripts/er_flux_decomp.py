"""Ca2+ compartment budget for the 0D ODE model
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import stimulus
from model.ode0d import build_system, rhs, _make_voltage
from model.parameters import default_parameters


# ---- reaction -> budget-category maps ------------------------------------
def cat_cyto(nm):
    if nm in ("V_Ca_entry", "N_Ca_entry", "SOCE_Ca_entry", "P_leak", "N_leak"):
        return "PM influx (VSCC+NMDAR+SOCE+leaks)"
    if nm in ("P_bind", "N_bind"):
        return "PM efflux (PMCA+NCX)"
    if nm in ("S_leak", "RyR_CICR", "IP3R_flux"):
        return "ER release (leak+RyR+IP3R)"
    if nm in ("S_x_x1", "S_x1_x2"):
        return "ER uptake (SERCA)"
    if nm in ("Bm_bind", "Bf_bind", "PLC_bindCa"):
        return "buffering (Bm+Bf+PLC)"
    if nm.startswith("RyR_act") or nm.startswith("RyR_inact") \
            or nm.startswith("IP3R_J"):
        return "channel-site Ca binding"
    if nm == "Ca_lumped_decay":
        return "Ca→∅ decay"
    return None


def cat_er(nm):
    if nm in ("S_y2_y1", "S_y1_y0"):
        return "SERCA uptake"
    if nm == "S_leak":
        return "SERCA leak"
    if nm == "RyR_CICR":
        return "RyR"
    if nm == "IP3R_flux":
        return "IP3R"
    return None


CYTO_COLORS = {
    "PM influx (VSCC+NMDAR+SOCE+leaks)": "C0",
    "PM efflux (PMCA+NCX)": "C3",
    "ER release (leak+RyR+IP3R)": "C2",
    "ER uptake (SERCA)": "C1",
    "buffering (Bm+Bf+PLC)": "C6",
    "channel-site Ca binding": "C9",
    "Ca→∅ decay": "C8",
}
ER_COLORS = {
    "SERCA uptake": "C1",
    "SERCA leak": "C2",
    "RyR": "C4",
    "IP3R": "C5",
}


def cumtrapz(y, x):
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def run_trajectory(p, *, include_soce, v_rest, final_t, n_points,
                   glu_molecules, equilibrate_first, stim_voltage=False):
    names, y0, compiled, glu_idx = build_system(p, include_soce=include_soce)
    idx = {n: i for i, n in enumerate(names)}
    # Equilibration always uses the resting clamp; the run uses the EPSP+bAP
    # transient when --stim is set so glutamate/depolarisation raise Ca_c.
    vfun_rest = _make_voltage(stimulus.hold_voltage(v_rest))
    vfun = (_make_voltage(stimulus.voltage(times=[0.0], V_rest=v_rest))
            if stim_voltage else vfun_rest)
    y0 = y0.copy()
    if glu_idx is not None:
        y0[glu_idx] = 0.0

    if equilibrate_first:
        eq = solve_ivp(rhs, (0.0, 500.0), y0, method="BDF",
                       args=(compiled, vfun_rest), t_eval=[500.0],
                       rtol=1e-9, atol=1e-14)
        y0 = np.asarray(eq.y)[:, -1]

    if glu_molecules and glu_idx is not None:
        y0[glu_idx] = stimulus.molecules_to_conc(glu_molecules, p)

    t_eval = np.linspace(0.0, final_t, n_points)
    sol = solve_ivp(rhs, (0.0, final_t), y0, method="BDF",
                    args=(compiled, vfun), t_eval=t_eval,
                    rtol=1e-9, atol=1e-14)
    if not sol.success:
        print("WARNING solve_ivp:", sol.message)
    return names, idx, compiled, vfun, sol.t, sol.y


def contributions(compiled, target_idx, Y, Vm):
    """ions/s contribution of each reaction to species `target_idx` vs time
    (BEFORE the volume factor -- returns d[species]/dt in M/s per reaction)."""
    out = {}
    for nm, ff, fo, rf, ro, st in compiled:
        if target_idx not in st:
            continue
        net = np.zeros(Y.shape[1])
        if ff is not None:
            net = net + ff(*(Y[i] for i in fo), Vm)
        if rf is not None:
            net = net - rf(*(Y[i] for i in ro), Vm)
        out[nm] = st[target_idx] * net   # M/s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-t", type=float, default=2.0)
    ap.add_argument("--n-points", type=int, default=16000)
    ap.add_argument("--v-rest", type=float, default=-65.0)
    ap.add_argument("--no-soce", action="store_true")
    ap.add_argument("--glu-molecules", type=float, default=0.0,
                    help="single bolus at t=0 (0 = no stimulus)")
    ap.add_argument("--stim", action="store_true",
                    help="drive the run with the EPSP+bAP voltage transient "
                         "(depolarises PM so NMDAR/VSCC raise Ca_c)")
    ap.add_argument("--equilibrate", action="store_true",
                    help="start from the 500 s equilibrated state instead of "
                         "the default initial condition")
    ap.add_argument("--out", default="results/Flux/er_cyto_ca_budget_ode.png")
    a = ap.parse_args(argv)

    include_soce = not a.no_soce
    p = default_parameters()
    p.N_IP3R = 0.0
    p.V_spine_L = float(p.V_spine) * 1e-15

    _names, idx, compiled, vfun, t, Y = run_trajectory(
        p, include_soce=include_soce, v_rest=a.v_rest, final_t=a.final_t,
        n_points=a.n_points, glu_molecules=a.glu_molecules,
        equilibrate_first=a.equilibrate, stim_voltage=a.stim)

    Vm = np.array([vfun(tk) for tk in t])
    NA = float(p.N_A)
    V_cyto_L = float(p.V_spine_L)
    V_er_L = float(p.V_ER) * 1e-15
    ca_c, ca_er = idx["Ca_c"], idx["Ca_ER"]

    # per-reaction dY/dt (M/s) -> ions/s in the relevant compartment
    cyto_MPS = contributions(compiled, ca_c, Y, Vm)
    er_MPS = contributions(compiled, ca_er, Y, Vm)
    cyto_ions = {nm: v * NA * V_cyto_L for nm, v in cyto_MPS.items()}
    er_ions = {nm: v * NA * V_er_L for nm, v in er_MPS.items()}

    # group by category
    def group(ions, catfn):
        g = {}
        for nm, v in ions.items():
            c = catfn(nm)
            if c is None:
                continue
            g[c] = g.get(c, np.zeros_like(t)) + v
        return g

    cyto_g = group(cyto_ions, cat_cyto)
    er_g = group(er_ions, cat_er)

    N_cyto = Y[ca_c] * NA * V_cyto_L
    N_er = Y[ca_er] * NA * V_er_L

    # ---- figure -----------------------------------------------------------
    tag = ("glut bolus" if a.glu_molecules else "no stim")
    start = "equilibrated" if a.equilibrate else "default IC"
    fig, (axE, axC) = plt.subplots(1, 2, figsize=(14, 6))

    # ER lumen
    er_sum = np.zeros_like(t)
    for cat in ("SERCA uptake", "SERCA leak", "RyR", "IP3R"):
        if cat not in er_g:
            continue
        cum = cumtrapz(er_g[cat], t)
        er_sum = er_sum + cum
        axE.plot(t, cum, color=ER_COLORS[cat], lw=1.7, label=cat)
    er_meas = N_er - N_er[0]
    axE.plot(t, er_meas, color="k", lw=2.3, label="Recorded Ca change")
    axE.plot(t, er_sum, color="0.5", lw=1.0, ls="--", label="sum of fluxes")
    axE.axhline(0, color="k", lw=0.6, alpha=0.4)
    axE.set_title(f"ER lumen Ca  ({tag}, {start}, IP3R off, "
                  f"SOCE {'on' if include_soce else 'off'})")
    axE.set_xlabel("time (s)")
    axE.set_ylabel("cumulative Ca into compartment (ions)")
    axE.legend(fontsize=8, loc="best")
    axE.margins(x=0)

    # cytosol
    cy_sum = np.zeros_like(t)
    order = ["PM influx (VSCC+NMDAR+SOCE+leaks)", "PM efflux (PMCA+NCX)",
             "ER release (leak+RyR+IP3R)", "ER uptake (SERCA)",
             "buffering (Bm+Bf+PLC)", "channel-site Ca binding", "Ca→∅ decay"]
    for cat in order:
        if cat not in cyto_g:
            continue
        cum = cumtrapz(cyto_g[cat], t)
        cy_sum = cy_sum + cum
        axC.plot(t, cum, color=CYTO_COLORS[cat], lw=1.7, label=cat)
    cy_meas = N_cyto - N_cyto[0]
    axC.plot(t, cy_meas, color="k", lw=2.3, label="Recorded Ca change")
    axC.plot(t, cy_sum, color="0.5", lw=1.0, ls="--", label="sum of fluxes")
    axC.axhline(0, color="k", lw=0.6, alpha=0.4)
    axC.set_title("Cytosol Ca")
    axC.set_xlabel("time (s)")
    axC.set_ylabel("cumulative Ca into compartment (ions)")
    axC.legend(fontsize=8, loc="best")
    axC.margins(x=0)

    print(f"Scenario: {tag}, {start}, IP3R off, "
          f"SOCE {'on' if include_soce else 'off'}, t=0..{a.final_t}s")
    print(f"V_cyto={V_cyto_L*1e15:.4f} um^3  V_ER={V_er_L*1e15:.4f} um^3")
    print(f"ER:   N start={N_er[0]:.1f}  end={N_er[-1]:.1f} ions "
          f"(Δ={er_meas[-1]:+.1f})")
    print(f"  closure max|Σ-meas| = {np.abs(er_sum-er_meas).max():.3e} ions")
    print(f"Cyto: N start={N_cyto[0]:.1f}  end={N_cyto[-1]:.1f} ions "
          f"(Δ={cy_meas[-1]:+.1f})")
    print(f"  closure max|Σ-meas| = {np.abs(cy_sum-cy_meas).max():.3e} ions")
    print("\nCumulative ER throughput over window (ions):")
    for cat in ("SERCA uptake", "SERCA leak", "RyR", "IP3R"):
        if cat in er_g:
            print(f"  {cat:14s} {cumtrapz(er_g[cat], t)[-1]:+.1f}")
    print("Cumulative cytosol throughput over window (ions):")
    for cat in order:
        if cat in cyto_g:
            print(f"  {cat:34s} {cumtrapz(cyto_g[cat], t)[-1]:+.1f}")
    print("PM influx breakdown (ions):")
    for nm in ("SOCE_Ca_entry", "P_leak", "N_leak", "V_Ca_entry", "N_Ca_entry"):
        if nm in cyto_ions:
            print(f"  {nm:16s} {cumtrapz(cyto_ions[nm], t)[-1]:+.1f}")
    for nm in ("P_bind", "N_bind"):
        if nm in cyto_ions:
            print(f"  {nm:16s} {cumtrapz(cyto_ions[nm], t)[-1]:+.1f}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
