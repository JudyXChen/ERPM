"""Resting steady-state, ER-flux, and intracellular Ca mass-balance audit.

This is a diagnostic driver: it does not modify the model.  It runs a compact
set of mechanism knockouts at clamped resting voltage and writes CSV results to
results/steady_state by default.
"""

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.ode0d import _make_voltage, build_system, rhs
from model.parameters import default_parameters


VARIANTS = (
    ("A_full_soce", True, {}, set(), False, False),
    ("A_full_no_soce", False, {}, set(), False, False),
    ("B_vgcc_off", True, {"N_VSCC": 0.0}, set(), False, False),
    ("C_soce_off", False, {}, set(), False, False),
    ("D_pmca_off", True, {}, {"P_bind", "P_translocate", "P_leak"}, False, False),
    ("E_ncx_off", True, {}, {"N_bind", "N_translocate", "N_leak"}, False, False),
    ("F_pmca_ncx_off", True, {}, {
        "P_bind", "P_translocate", "P_leak",
        "N_bind", "N_translocate", "N_leak",
    }, False, False),
    ("G_decay_off", True, {"k_d": 0.0}, set(), False, False),
    # Remove uptake/extrusion cycles but retain their phenomenological inward leaks.
    ("H_no_outward", True, {"k_d": 0.0}, {
        "P_bind", "P_translocate", "N_bind", "N_translocate",
    }, False, False),
    # No exchange with ECM and no phenomenological decay.
    ("I_closed", False, {"k_d": 0.0}, {
        "P_bind", "P_translocate", "P_leak",
        "N_bind", "N_translocate", "N_leak",
        "V_Ca_entry", "N_Ca_entry", "Ca_lumped_decay",
    }, False, False),
    ("J_ryr_off", True, {}, set(), True, False),
    ("J_ip3r_off", True, {}, set(), False, True),
    ("J_ryr_ip3r_off", True, {}, set(), True, True),
    ("K_er_leak_off", True, {"k_S_leak": 0.0}, set(), False, False),
)


def reaction_net(item, y, vm):
    """Return the signed forward-minus-reverse rate of one compiled reaction."""
    _name, ff, fo, rf, ro, _st = item
    net = ff(*(y[i] for i in fo), vm) if ff is not None else 0.0
    if rf is not None:
        net -= rf(*(y[i] for i in ro), vm)
    return float(net)


def ca_inventory_weights(names, p):
    """Ca atoms represented by one unit of each state variable."""
    w = np.zeros(len(names), dtype=float)
    idx = {name: i for i, name in enumerate(names)}
    n_a = float(p.N_A)
    v_c = float(p.V_spine_L)
    v_er = float(p.V_ER) * 1e-15

    w[idx["Ca_c"]] = n_a * v_c
    w[idx["Ca_ER"]] = n_a * v_er
    for name in ("Bm_Ca", "Bf_Ca"):
        if name in idx:
            w[idx[name]] = n_a * v_c
    for name in ("PLC_Ca", "PLC_Ca_PIP2"):
        if name in idx:
            w[idx[name]] = float(p.N_PLC)
    if "PMCA1" in idx:
        w[idx["PMCA1"]] = float(p.N_PMCA)
    if "NCX1" in idx:
        w[idx["NCX1"]] = float(p.N_NCX)

    serca_ca = {"Sx1": 1, "Sx2": 2, "Sy2": 2, "Sy1": 1}
    for name, count in serca_ca.items():
        if name in idx:
            w[idx[name]] = count * float(p.N_SERCA)

    for name, i in idx.items():
        match = re.fullmatch(r"R_(?:C|O|CI|OI)_(\d)_(\d)", name)
        if match:
            w[i] = (int(match.group(1)) + int(match.group(2))) * float(p.N_RyR)
        # IP3R occupancy is deliberately not counted: in the current reaction
        # definitions Ca_c controls the gating rates but is not a stoichiometric
        # reactant, so changing IPR state does not remove free Ca from the model.
    return w


def er_fluxes(compiled, y, vm, idx, p):
    """Signed contributions to ER Ca content, positive into ER, in ions/s."""
    out = {"SERCA": 0.0, "ER_leak": 0.0, "RyR": 0.0, "IP3R": 0.0}
    caer = idx["Ca_ER"]
    scale = float(p.N_A) * float(p.V_ER) * 1e-15
    for item in compiled:
        name, *_rest, st = item
        if caer not in st:
            continue
        if name == "S_leak":
            group = "ER_leak"
        elif name.startswith("S_"):
            group = "SERCA"
        elif name.startswith("RyR"):
            group = "RyR"
        elif name.startswith("IP3R"):
            group = "IP3R"
        else:
            continue
        out[group] += st[caer] * reaction_net(item, y, vm) * scale
    out["ER_net"] = sum(out.values())
    return out


def boundary_fluxes(compiled, y, vm, idx, p):
    """Positive ECM entry and positive outward/sink rates, in ions/s."""
    by_name = {item[0]: item for item in compiled}
    cac = idx["Ca_c"]
    cscale = float(p.N_A) * float(p.V_spine_L)

    def ca_contribution(name):
        item = by_name.get(name)
        if item is None or cac not in item[-1]:
            return 0.0
        return item[-1][cac] * reaction_net(item, y, vm) * cscale

    entry_names = ("V_Ca_entry", "SOCE_Ca_entry", "N_Ca_entry", "P_leak", "N_leak")
    out = {name: ca_contribution(name) for name in entry_names}
    out["PMCA_out"] = (reaction_net(by_name["P_translocate"], y, vm)
                       * float(p.N_PMCA)) if "P_translocate" in by_name else 0.0
    out["NCX_out"] = (reaction_net(by_name["N_translocate"], y, vm)
                      * float(p.N_NCX)) if "N_translocate" in by_name else 0.0
    out["decay_out"] = -ca_contribution("Ca_lumped_decay")
    out["entry_total"] = sum(out[name] for name in entry_names)
    out["out_total"] = out["PMCA_out"] + out["NCX_out"] + out["decay_out"]
    out["boundary_net"] = out["entry_total"] - out["out_total"]
    return out


def integrate_to_steady(names, y0, compiled, vfun, *, chunk, max_t, ca_tol,
                        state_tol):
    idx = {name: i for i, name in enumerate(names)}
    y = np.asarray(y0, dtype=float).copy()
    t = 0.0
    consecutive = 0
    trace_t = [t]
    trace_y = [y.copy()]
    reached = False
    ca_resid = state_resid = np.inf

    while t < max_t:
        t1 = min(t + chunk, max_t)
        t_eval = None
        if t == 0.0:
            early = np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, t1])
            t_eval = np.unique(early[(early > t) & (early <= t1)])
        sol = solve_ivp(rhs, (t, t1), y, method="BDF",
                        args=(compiled, vfun), t_eval=t_eval,
                        rtol=1e-8, atol=1e-14)
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y[:, -1].copy()
        t = t1
        trace_t.extend(float(value) for value in sol.t)
        trace_y.extend(sol.y[:, k].copy() for k in range(sol.y.shape[1]))
        dy = rhs(t, y, compiled, vfun)
        ca_resid = max(
            abs(dy[idx["Ca_c"]]) / max(abs(y[idx["Ca_c"]]), 1e-12),
            abs(dy[idx["Ca_ER"]]) / max(abs(y[idx["Ca_ER"]]), 1e-12),
        )
        non_ca = np.ones(len(y), dtype=bool)
        non_ca[[idx["Ca_c"], idx["Ca_ER"]]] = False
        state_resid = float(np.max(np.abs(dy[non_ca])))
        if ca_resid < ca_tol and state_resid < state_tol:
            consecutive += 1
            if consecutive >= 2:
                reached = True
                break
        else:
            consecutive = 0
    return y, t, reached, ca_resid, state_resid, np.asarray(trace_t), np.asarray(trace_y)


def run_variant(spec, args, out_dir):
    name, include_soce, overrides, excluded, ryr_off, ip3r_off = spec
    p = default_parameters()
    for key, value in overrides.items():
        setattr(p, key, float(value))
    p.Glu_init = 0.0
    names, y0, compiled, _ = build_system(
        p, include_soce=include_soce, ryr_off=ryr_off, ip3r_off=ip3r_off)
    compiled = [item for item in compiled if item[0] not in excluded]
    idx = {state: i for i, state in enumerate(names)}
    vfun = _make_voltage(args.v_rest)
    vm = float(args.v_rest)
    weights = ca_inventory_weights(names, p)

    initial_er = er_fluxes(compiled, y0, vm, idx, p)
    initial_pm = boundary_fluxes(compiled, y0, vm, idx, p)
    inv0 = float(weights @ y0)
    deriv0 = float(weights @ rhs(0.0, y0, compiled, vfun))
    y, t, reached, ca_resid, state_resid, tt, yy = integrate_to_steady(
        names, y0, compiled, vfun, chunk=args.chunk, max_t=args.max_t,
        ca_tol=args.ca_tol, state_tol=args.state_tol)
    final_er = er_fluxes(compiled, y, vm, idx, p)
    final_pm = boundary_fluxes(compiled, y, vm, idx, p)
    inv1 = float(weights @ y)
    deriv1 = float(weights @ rhs(t, y, compiled, vfun))
    one_s_index = int(np.argmin(np.abs(tt - 1.0)))
    one_s_er = er_fluxes(compiled, yy[one_s_index], vm, idx, p)
    one_s_pm = boundary_fluxes(compiled, yy[one_s_index], vm, idx, p)

    with (out_dir / f"trace_{name}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("time_s", "Ca_c_M", "Ca_ER_M", "Ca_inventory_ions"))
        for k, tk in enumerate(tt):
            writer.writerow((tk, yy[k, idx["Ca_c"]], yy[k, idx["Ca_ER"]],
                             float(weights @ yy[k])))

    row = {
        "variant": name,
        "converged": reached,
        "time_s": t,
        "Ca_c_nM": y[idx["Ca_c"]] * 1e9,
        "Ca_ER_uM": y[idx["Ca_ER"]] * 1e6,
        "ca_relative_residual_s-1": ca_resid,
        "state_absolute_residual_s-1": state_resid,
        "inventory_initial_ions": inv0,
        "inventory_final_ions": inv1,
        "inventory_change_ions": inv1 - inv0,
        "inventory_relative_change": (inv1 - inv0) / max(abs(inv0), 1.0),
        "inventory_derivative_initial_ions_s": deriv0,
        "inventory_derivative_final_ions_s": deriv1,
    }
    row.update({f"ER_initial_{k}_ions_s": v for k, v in initial_er.items()})
    row.update({f"ER_1s_{k}_ions_s": v for k, v in one_s_er.items()})
    row.update({f"ER_final_{k}_ions_s": v for k, v in final_er.items()})
    row.update({f"PM_initial_{k}_ions_s": v for k, v in initial_pm.items()})
    row.update({f"PM_1s_{k}_ions_s": v for k, v in one_s_pm.items()})
    row.update({f"PM_final_{k}_ions_s": v for k, v in final_pm.items()})
    print(f"{name:20s} steady={str(reached):5s} t={t:6.0f}s "
          f"Ca_c={row['Ca_c_nM']:9.2f} nM Ca_ER={row['Ca_ER_uM']:10.2f} uM "
          f"dN/dt={deriv1: .3e} ions/s")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/steady_state")
    parser.add_argument("--v-rest", type=float, default=-65.0)
    parser.add_argument("--chunk", type=float, default=50.0)
    parser.add_argument("--max-t", type=float, default=2000.0)
    parser.add_argument("--ca-tol", type=float, default=1e-6)
    parser.add_argument("--state-tol", type=float, default=1e-9)
    parser.add_argument("--variants", nargs="*", help="optional variant names")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [v for v in VARIANTS if not args.variants or v[0] in args.variants]
    unknown = set(args.variants or ()) - {v[0] for v in VARIANTS}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    rows = [run_variant(spec, args, out_dir) for spec in selected]
    with (out_dir / "variant_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "run_config.json").open("w") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True)
    print(f"Saved {len(rows)} variants to {out_dir}")


if __name__ == "__main__":
    main()
