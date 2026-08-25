"""Find the resting IP3 setpoint for 150 uM ER Ca and audit mGluR scaling.

This is an analysis-only driver for the well-mixed model.  RyR and SOCE are
disabled, glutamate is absent during equilibration, and voltage is clamped.
The generated CSV files are deliberately tidy: one row per setpoint or per
pair of mGluR scaling factors.
"""

import argparse
import csv
import json
import pathlib
import subprocess
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import stimulus
from model.ode0d import _make_voltage, build_system, rhs
from model.parameters import default_parameters


DEFAULT_FACTORS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
TARGET_CA_ER_UM = 150.0


def reaction_net(item, y, vm):
    """Return an unscaled forward-minus-reverse reaction rate."""
    _name, forward, forward_order, reverse, reverse_order, _stoich = item
    value = forward(*(y[i] for i in forward_order), vm) if forward else 0.0
    if reverse:
        value -= reverse(*(y[i] for i in reverse_order), vm)
    return float(value)


def contribution(item, y, vm, state_index):
    """Return one reaction's contribution to a state derivative."""
    if item is None or state_index not in item[-1]:
        return 0.0
    return item[-1][state_index] * reaction_net(item, y, vm)


def integrate_to_equilibrium(names, y0, compiled, vfun, args):
    """Integrate in chunks until two consecutive derivative checks pass."""
    index = {name: i for i, name in enumerate(names)}
    y = np.asarray(y0, dtype=float).copy()
    t = 0.0
    consecutive = 0
    solver_success = True
    message = ""
    ca_residual = state_residual = np.inf

    while t < args.max_equilibration_s:
        next_t = min(t + args.convergence_chunk_s, args.max_equilibration_s)
        solution = solve_ivp(
            rhs,
            (t, next_t),
            y,
            method="BDF",
            args=(compiled, vfun),
            t_eval=[next_t],
            rtol=args.solver_rtol,
            atol=args.solver_atol,
        )
        if not solution.success:
            solver_success = False
            message = solution.message
            break
        y = solution.y[:, -1].copy()
        t = next_t
        derivative = rhs(t, y, compiled, vfun)
        ca_residual = max(
            abs(derivative[index[name]]) / max(abs(y[index[name]]), 1e-30)
            for name in ("Ca_c", "Ca_ER")
        )
        non_calcium = np.ones(len(y), dtype=bool)
        non_calcium[[index["Ca_c"], index["Ca_ER"]]] = False
        state_residual = float(np.max(np.abs(derivative[non_calcium])))
        if (ca_residual < args.ca_relative_tolerance
                and state_residual < args.state_absolute_tolerance):
            consecutive += 1
            if consecutive == 2:
                break
        else:
            consecutive = 0

    return {
        "state": y,
        "time_s": t,
        "solver_success": solver_success,
        "solver_message": message,
        "converged": solver_success and consecutive >= 2,
        "consecutive_checks": consecutive,
        "ca_relative_residual_s-1": float(ca_residual),
        "state_absolute_residual_s-1": float(state_residual),
    }


def equilibrium_fluxes(compiled, y, vm, index, p):
    """Return signed ER fluxes and IP3 production rates at one state."""
    by_name = {item[0]: item for item in compiled}
    ca_er = index["Ca_ER"]
    ip3 = index["IP3"]
    er_scale = float(p.N_A) * float(p.V_ER) * 1e-15

    serca_molar_s = sum(
        contribution(by_name.get(name), y, vm, ca_er)
        for name in ("S_y2_y1", "S_y1_y0")
    )
    ip3r_molar_s = contribution(by_name.get("IP3R_flux"), y, vm, ca_er)
    er_net_molar_s = sum(
        contribution(item, y, vm, ca_er) for item in compiled
    )

    def production_nm_s(name):
        return contribution(by_name.get(name), y, vm, ip3) * 1e9

    cat1 = production_nm_s("mG_cat1")
    cat2 = production_nm_s("mG_cat2")
    return {
        "SERCA_uptake_ions_s": serca_molar_s * er_scale,
        "IP3R_release_ions_s": -ip3r_molar_s * er_scale,
        "ER_net_ions_s": er_net_molar_s * er_scale,
        "PLC_IP3_production_nM_s": production_nm_s("PLC_cleave"),
        "mG_cat1_IP3_production_nM_s": cat1,
        "mG_cat2_IP3_production_nM_s": cat2,
        "mGluR_IP3_production_nM_s": cat1 + cat2,
    }


def target_run(setpoint_nm, args, *, initial_ip3_nm=None,
               initial_ca_er_um=None, protocol="target_curve"):
    p = default_parameters()
    p.Glu_init = 0.0
    p.IP3_basal = float(setpoint_nm) * 1e-9
    p.PLC_cleave_activity = 0.0
    names, y0, compiled, _ = build_system(p, include_soce=False, ryr_off=True)
    index = {name: i for i, name in enumerate(names)}
    if initial_ip3_nm is not None:
        y0[index["IP3"]] = float(initial_ip3_nm) * 1e-9
    if initial_ca_er_um is not None:
        y0[index["Ca_ER"]] = float(initial_ca_er_um) * 1e-6
    result = integrate_to_equilibrium(
        names, y0, compiled, _make_voltage(args.voltage_mV), args)
    y = result.pop("state")
    derivative = rhs(result["time_s"], y, compiled,
                     _make_voltage(args.voltage_mV))
    row = {
        "protocol": protocol,
        "IP3_setpoint_nM": float(setpoint_nm),
        "initial_IP3_nM": (float(setpoint_nm) if initial_ip3_nm is None
                            else float(initial_ip3_nm)),
        "initial_Ca_ER_uM": (float(p.Ca_ER_init) * 1e6
                              if initial_ca_er_um is None
                              else float(initial_ca_er_um)),
        "selected": False,
        **result,
        "final_IP3_nM": y[index["IP3"]] * 1e9,
        "final_Ca_c_nM": y[index["Ca_c"]] * 1e9,
        "final_Ca_ER_uM": y[index["Ca_ER"]] * 1e6,
        "IP3_residual_nM_s": derivative[index["IP3"]] * 1e9,
        **equilibrium_fluxes(compiled, y, args.voltage_mV, index, p),
    }
    return row


def solve_ip3_target(args):
    low, high = sorted(args.target_bracket_nM)
    cache = {}

    def evaluate(setpoint_nm):
        key = float(setpoint_nm)
        if key not in cache:
            row = target_run(key, args)
            if not row["solver_success"] or not row["converged"]:
                raise RuntimeError(
                    f"target run at {key:g} nM did not converge: "
                    f"{row['solver_message']}")
            cache[key] = row
        return cache[key]

    grid = np.linspace(low, high, args.target_curve_points)
    values = []
    for setpoint_nm in grid:
        row = evaluate(setpoint_nm)
        values.append(row["final_Ca_ER_uM"] - TARGET_CA_ER_UM)
    crossing = None
    for left, right, f_left, f_right in zip(
            grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0 or f_left * f_right <= 0.0:
            crossing = (float(left), float(right))
            break
    if crossing is None:
        raise RuntimeError(
            f"Ca_ER={TARGET_CA_ER_UM:g} uM is not bracketed by "
            f"{low:g}-{high:g} nM IP3")

    root = root_scalar(
        lambda value: evaluate(value)["final_Ca_ER_uM"] - TARGET_CA_ER_UM,
        bracket=crossing,
        method="brentq",
        xtol=args.root_tolerance_nM,
    )
    if not root.converged:
        raise RuntimeError("IP3 target root solver did not converge")
    selected = evaluate(root.root)
    selected["selected"] = True
    if abs(selected["final_Ca_ER_uM"] - TARGET_CA_ER_UM) > 0.5:
        raise AssertionError("selected target misses 150 +/- 0.5 uM Ca_ER")
    flux_scale = max(abs(selected["SERCA_uptake_ions_s"]), 1.0)
    if abs(selected["ER_net_ions_s"]) > 1e-5 * flux_scale:
        raise AssertionError("selected target has a non-negligible net ER flux")

    rows = sorted(cache.values(), key=lambda row: row["IP3_setpoint_nM"])
    for initial_ip3 in (10.0, 300.0):
        for initial_ca_er in (50.0, 300.0):
            check = target_run(
                root.root,
                args,
                initial_ip3_nm=initial_ip3,
                initial_ca_er_um=initial_ca_er,
                protocol="initial_condition_check",
            )
            if (not check["converged"]
                    or abs(check["final_Ca_ER_uM"] - TARGET_CA_ER_UM) > 0.5
                    or abs(check["final_IP3_nM"] - root.root) > 0.01):
                raise AssertionError("target is sensitive to tested initial conditions")
            rows.append(check)
    return float(root.root), rows


def resting_mglur_run(total_factor, number_factor, args):
    p = default_parameters()
    nominal_total = float(p.mGluR_total)
    nominal_number = float(p.N_mGluR)
    p.mGluR_total = nominal_total * float(total_factor)
    p.N_mGluR = nominal_number * float(number_factor)
    p.PLC_cleave_activity = 1.0
    p.Glu_init = 0.0
    names, y0, compiled, _ = build_system(p, include_soce=False, ryr_off=True)
    index = {name: i for i, name in enumerate(names)}
    vfun = _make_voltage(args.voltage_mV)
    result = integrate_to_equilibrium(names, y0, compiled, vfun, args)
    y = result.pop("state")
    derivative = rhs(result["time_s"], y, compiled, vfun)
    row = {
        "mGluR_total_factor": float(total_factor),
        "N_mGluR_factor": float(number_factor),
        "mGluR_total": float(p.mGluR_total),
        "N_mGluR": float(p.N_mGluR),
        **result,
        "final_IP3_nM": y[index["IP3"]] * 1e9,
        "final_Ca_c_nM": y[index["Ca_c"]] * 1e9,
        "final_Ca_ER_uM": y[index["Ca_ER"]] * 1e6,
        "IP3_residual_nM_s": derivative[index["IP3"]] * 1e9,
        **equilibrium_fluxes(compiled, y, args.voltage_mV, index, p),
    }
    return row, (p, names, index, compiled, y)


def positive_control_row(rest_row, rest_context, args):
    p, names, index, compiled, resting_state = rest_context
    initial = resting_state.copy()
    initial[index["Glu"]] = stimulus.molecules_to_conc(
        args.glutamate_molecules, p)
    times = np.linspace(0.0, args.glutamate_duration_s,
                        args.glutamate_time_points)
    vfun = _make_voltage(args.voltage_mV)
    solution = solve_ivp(
        rhs,
        (0.0, args.glutamate_duration_s),
        initial,
        method="BDF",
        args=(compiled, vfun),
        t_eval=times,
        rtol=args.solver_rtol,
        atol=args.solver_atol,
    )
    by_name = {item[0]: item for item in compiled}
    ip3_index = index["IP3"]

    def production_trace(name):
        item = by_name.get(name)
        if item is None or ip3_index not in item[-1]:
            return np.zeros(solution.t.size)
        return np.array([
            contribution(item, solution.y[:, k], args.voltage_mV, ip3_index)
            for k in range(solution.t.size)
        ])

    cat1 = production_trace("mG_cat1")
    cat2 = production_trace("mG_cat2")
    receptor_rate = cat1 + cat2
    integrated = float(np.trapezoid(receptor_rate, solution.t) * 1e9)
    baseline_ip3 = resting_state[ip3_index]
    return {
        "mGluR_total_factor": rest_row["mGluR_total_factor"],
        "N_mGluR_factor": rest_row["N_mGluR_factor"],
        "mGluR_total": rest_row["mGluR_total"],
        "N_mGluR": rest_row["N_mGluR"],
        "glutamate_molecules": float(args.glutamate_molecules),
        "duration_s": float(args.glutamate_duration_s),
        "solver_success": bool(solution.success),
        "solver_message": solution.message,
        "rest_converged": rest_row["converged"],
        "rest_ca_relative_residual_s-1": rest_row[
            "ca_relative_residual_s-1"],
        "rest_state_absolute_residual_s-1": rest_row[
            "state_absolute_residual_s-1"],
        "resting_IP3_nM": baseline_ip3 * 1e9,
        "peak_IP3_nM": np.max(solution.y[ip3_index]) * 1e9,
        "peak_IP3_increase_nM": (
            np.max(solution.y[ip3_index]) - baseline_ip3) * 1e9,
        "integrated_mG_cat1_IP3_nM": float(np.trapezoid(cat1, solution.t) * 1e9),
        "integrated_mG_cat2_IP3_nM": float(np.trapezoid(cat2, solution.t) * 1e9),
        "integrated_mGluR_IP3_nM": integrated,
        "peak_mGluR_IP3_production_nM_s": float(np.max(receptor_rate) * 1e9),
    }


def run_mglur_sweep(args):
    resting_rows = []
    control_rows = []
    for total_factor in args.sweep_factors:
        for number_factor in args.sweep_factors:
            row, context = resting_mglur_run(total_factor, number_factor, args)
            if not row["solver_success"] or not row["converged"]:
                raise RuntimeError(
                    "mGluR resting sweep failed to converge at "
                    f"({total_factor:g}, {number_factor:g})")
            resting_rows.append(row)
            control = positive_control_row(row, context, args)
            if not control["solver_success"]:
                raise RuntimeError(
                    "mGluR positive control failed at "
                    f"({total_factor:g}, {number_factor:g}): "
                    f"{control['solver_message']}")
            control_rows.append(control)

    if max(abs(row["mGluR_IP3_production_nM_s"])
           for row in resting_rows) > 1e-10:
        raise AssertionError("mGluR produced IP3 at zero glutamate")
    for field, tolerance in (("final_IP3_nM", 1e-6),
                             ("final_Ca_ER_uM", 1e-6)):
        if np.ptp([row[field] for row in resting_rows]) > tolerance:
            raise AssertionError(f"resting {field} depends on mGluR scaling")
    for row in control_rows:
        knockout = (row["mGluR_total_factor"] == 0.0
                    or row["N_mGluR_factor"] == 0.0)
        if knockout and abs(row["integrated_mGluR_IP3_nM"]) > 1e-10:
            raise AssertionError("an mGluR knockout produced receptor-derived IP3")
        if not knockout and (row["integrated_mGluR_IP3_nM"] <= 0.0
                             or row["peak_IP3_increase_nM"] <= 0.0):
            raise AssertionError("glutamate did not activate an intact mGluR branch")
    return resting_rows, control_rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows, concentration_fields, residual_fields):
    """Validate the finite/nonnegative and convergence data promised by CSVs."""
    for row in rows:
        if not row["solver_success"]:
            raise AssertionError("CSV row records solver failure")
        for field in concentration_fields:
            value = float(row[field])
            if not np.isfinite(value) or value < 0.0:
                raise AssertionError(f"invalid concentration {field}={value}")
        for field in residual_fields:
            value = float(row[field])
            if not np.isfinite(value) or value < 0.0:
                raise AssertionError(f"invalid residual {field}={value}")


def git_snapshot():
    def command(*args):
        result = subprocess.run(
            args, cwd=ROOT, check=False, text=True, capture_output=True)
        return result.stdout.strip()

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--short").splitlines(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/ip3r_resting_analysis"))
    parser.add_argument("--voltage", "--v-rest", dest="voltage_mV",
                        type=float, default=-65.0, help="voltage clamp (mV)")
    parser.add_argument("--convergence-chunk", dest="convergence_chunk_s",
                        type=float, default=50.0, help="chunk duration (s)")
    parser.add_argument("--max-equilibration", dest="max_equilibration_s",
                        type=float, default=2000.0, help="maximum rest run (s)")
    parser.add_argument("--ca-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--state-absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--solver-rtol", type=float, default=1e-8)
    parser.add_argument("--solver-atol", type=float, default=1e-14)
    parser.add_argument("--target-bracket", dest="target_bracket_nM", nargs=2,
                        type=float, default=(60.0, 80.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--target-curve-points", type=int, default=9)
    parser.add_argument("--root-tolerance-nM", type=float, default=1e-4)
    parser.add_argument("--glutamate-duration", dest="glutamate_duration_s",
                        type=float, default=2.0)
    parser.add_argument("--glutamate-molecules", type=float, default=500.0)
    parser.add_argument("--glutamate-time-points", type=int, default=401)
    parser.add_argument("--sweep-factors", nargs="+", type=float,
                        default=DEFAULT_FACTORS)
    args = parser.parse_args(argv)
    if args.target_curve_points < 2 or args.glutamate_time_points < 2:
        parser.error("target and glutamate time grids require at least 2 points")
    if args.convergence_chunk_s <= 0 or args.max_equilibration_s <= 0:
        parser.error("equilibration durations must be positive")
    if args.glutamate_duration_s <= 0 or args.glutamate_molecules < 0:
        parser.error("glutamate duration must be positive and molecules nonnegative")
    if len(set(args.sweep_factors)) != len(args.sweep_factors):
        parser.error("sweep factors must be unique")
    if any(value < 0 for value in args.sweep_factors):
        parser.error("sweep factors must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    selected_target_nm, target_rows = solve_ip3_target(args)
    resting_rows, control_rows = run_mglur_sweep(args)

    validate_rows(
        target_rows,
        ("final_IP3_nM", "final_Ca_c_nM", "final_Ca_ER_uM"),
        ("ca_relative_residual_s-1", "state_absolute_residual_s-1"),
    )
    validate_rows(
        resting_rows,
        ("final_IP3_nM", "final_Ca_c_nM", "final_Ca_ER_uM"),
        ("ca_relative_residual_s-1", "state_absolute_residual_s-1"),
    )
    validate_rows(
        control_rows,
        ("resting_IP3_nM", "peak_IP3_nM", "peak_IP3_increase_nM",
         "integrated_mGluR_IP3_nM"),
        ("rest_ca_relative_residual_s-1",
         "rest_state_absolute_residual_s-1"),
    )

    write_csv(args.out / "ip3_target_curve.csv", target_rows)
    write_csv(args.out / "mglur_resting_sweep.csv", resting_rows)
    write_csv(args.out / "mglur_glutamate_control.csv", control_rows)

    nominal = default_parameters()
    selected_row = next(row for row in target_rows if row["selected"])
    snapshot = {
        "selected_target": {
            "IP3_setpoint_nM": selected_target_nm,
            "final_Ca_ER_uM": selected_row["final_Ca_ER_uM"],
            "final_Ca_c_nM": selected_row["final_Ca_c_nM"],
            "SERCA_uptake_ions_s": selected_row["SERCA_uptake_ions_s"],
            "IP3R_release_ions_s": selected_row["IP3R_release_ions_s"],
            "ER_net_ions_s": selected_row["ER_net_ions_s"],
        },
        "protocol": {
            "RyR": False,
            "SOCE": False,
            "resting_glutamate": False,
            "target_PLC_cleave_activity": 0.0,
            "sweep_PLC_cleave_activity": 1.0,
            "sweep_IP3_basal_nM": float(nominal.IP3_basal) * 1e9,
        },
        "cli": {key: (str(value) if isinstance(value, pathlib.Path) else value)
                for key, value in vars(args).items()},
        "nominal_parameters": vars(nominal),
        "git": git_snapshot(),
    }
    with (args.out / "summary.json").open("w") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)

    print(
        f"Selected IP3 setpoint: {selected_target_nm:.5f} nM -> "
        f"Ca_ER={selected_row['final_Ca_ER_uM']:.5f} uM; "
        f"SERCA={selected_row['SERCA_uptake_ions_s']:.5f}, "
        f"IP3R={selected_row['IP3R_release_ions_s']:.5f} ions/s")
    print(f"Wrote {len(target_rows)} target rows and "
          f"{len(resting_rows)} x 2 mGluR rows to {args.out}")


if __name__ == "__main__":
    main()
