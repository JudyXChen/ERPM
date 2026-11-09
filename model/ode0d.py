"""Run the well-mixed 0D ER-PM model and save traces.csv."""

import argparse
import csv
import pathlib
import sys
import types

import numpy as np
import sympy as sym
from scipy.integrate import solve_ivp
from sympy.parsing.sympy_parser import parse_expr

from .parameters import balanced_r15_scale, default_parameters


def _install_smart_stub():
    if "smart.model_assembly" in sys.modules:
        return

    class _Unit:
        def __mul__(self, _other):
            return self

        def __rmul__(self, _other):
            return self

        def __truediv__(self, _other):
            return self

        def __rtruediv__(self, _other):
            return self

        def __pow__(self, _power):
            return self

    class Parameter:
        def __init__(self, name, value, *_args, **_kwargs):
            self.name = name
            self.value = value

    class Species:
        def __init__(
            self, name, initial_condition, *_args, **_kwargs
        ):
            self.name = name
            self.initial_condition = initial_condition

    class Reaction:
        def __init__(
            self,
            name,
            lhs,
            rhs,
            *,
            param_map=None,
            species_map=None,
            eqn_f_str="",
            eqn_r_str="",
            flux_scaling=None,
            **_kwargs,
        ):
            self.name = name
            self.lhs = lhs
            self.rhs = rhs
            self.param_map = param_map or {}
            self.species_map = species_map or {}
            self.eqn_f_str = eqn_f_str
            self.eqn_r_str = eqn_r_str
            self.flux_scaling = flux_scaling or {}

    smart_mod = types.ModuleType("smart")
    model_assembly_mod = types.ModuleType("smart.model_assembly")
    model_assembly_mod.Parameter = Parameter
    model_assembly_mod.Species = Species
    model_assembly_mod.Reaction = Reaction

    units_mod = types.ModuleType("smart.units")
    unit = types.SimpleNamespace(
        second=_Unit(),
        micrometer=_Unit(),
        millivolt=_Unit(),
        dimensionless=_Unit(),
    )
    units_mod.unit = unit

    smart_mod.model_assembly = model_assembly_mod
    smart_mod.units = units_mod
    sys.modules["smart"] = smart_mod
    sys.modules["smart.model_assembly"] = model_assembly_mod
    sys.modules["smart.units"] = units_mod


try:
    import smart.model_assembly  # noqa: F401
except ModuleNotFoundError:
    _install_smart_stub()

from .reactions import ip3r, registrars_for, ryr
from . import stimulus

VM = sym.Symbol("V_m")


def _shared_params(p):
    return {
        "V_spine": float(p.V_spine),
        "V_ER": float(p.V_ER),
        "V_spine_L": float(p.V_spine_L),
        "Ca_ext": float(p.Ca_ext),
        "F_const": float(p.F_const),
        "q_e": float(p.q_e),
        "N_A": float(p.N_A),
        "RT_F": float(p.RT_F),
        "k_glu_clearance": float(p.glu_clearance_rate),
    }


def build_system(p, *, include_soce=False, ip3r_subunits=False, ryr_off=False,
                 ip3r_off=False, plc="new"):
    from smart.model_assembly import Reaction  # container objects only

    registrars = registrars_for(plc=plc, include_soce=include_soce)
    if ryr_off:
        # Drop the RyR block entirely: its ~100 states and CICR flux vanish, so
        # the ER membrane sees only SERCA (pump+leak) and IP3R.
        registrars = tuple(r for r in registrars if r is not ryr.register)
    if ip3r_off:
        # Drop the IP3R block entirely: the ER membrane sees only SERCA (pump+leak)
        # and, unless also disabled, RyR. Used for pure-SERCA refill experiments.
        registrars = tuple(r for r in registrars
                           if r not in (ip3r.register, ip3r.register_subunits))
    if ip3r_subunits:
        registrars = tuple(ip3r.register_subunits if r is ip3r.register else r
                           for r in registrars)
    species_init, master, reactions = {}, _shared_params(p), []
    for reg in registrars:
        slist, plist, rlist = reg(p)
        for s in slist:
            species_init.setdefault(s.name, float(s.initial_condition))
        for par in plist:
            master[par.name] = float(par.value)
        reactions.extend(rlist)

    reactions.append(
        Reaction("glu_clearance", ["Glu"], [],
                 param_map={"k": "k_glu_clearance"},
                 eqn_f_str="k*Glu", species_map={"Glu": "Glu"})
    )

    names = list(species_init)
    idx = {n: i for i, n in enumerate(names)}
    y0 = np.array([species_init[n] for n in names], dtype=float)

    def compile_expr(s, r):
        if not s:
            return None, ()
        expr = parse_expr(s)
        subs, used = {}, set()
        for fsym in expr.free_symbols:
            nm = str(fsym)
            if nm in r.param_map:
                tgt = r.param_map[nm]
            elif nm in (r.species_map or {}):
                tgt = r.species_map[nm]
            else:
                tgt = nm
            if tgt == "V_m":
                subs[fsym] = VM
            elif tgt in idx:
                subs[fsym] = sym.Symbol(f"Y{idx[tgt]}")
                used.add(idx[tgt])
            elif tgt in master:
                subs[fsym] = sym.Float(master[tgt])
            else:
                raise KeyError(
                    f"{r.name}: symbol {nm!r}->{tgt!r} is neither species "
                    f"nor parameter")
        expr = expr.xreplace(subs)
        order = sorted(used)
        args = [sym.Symbol(f"Y{i}") for i in order] + [VM]
        return sym.lambdify(args, expr, "numpy"), tuple(order)

    compiled = []
    for r in reactions:
        ff, fo = compile_expr(getattr(r, "eqn_f_str", ""), r)
        rf, ro = compile_expr(getattr(r, "eqn_r_str", ""), r)
        st = {}
        for nm in r.lhs:
            st[nm] = st.get(nm, 0.0) - 1.0
        for nm in r.rhs:
            st[nm] = st.get(nm, 0.0) + 1.0
        fs = r.flux_scaling or {}
        for nm in list(st):
            sc = fs.get(nm)
            if sc is not None:
                st[nm] *= float(sc)
        st = {idx[nm]: v for nm, v in st.items() if nm in idx}
        if st and (ff or rf):
            compiled.append((r.name, ff, fo, rf, ro, st))

    return names, y0, compiled, idx.get("Glu")


def _make_voltage(voltage):
    if isinstance(voltage, sym.Expr):
        f = sym.lambdify(sym.Symbol("t"), voltage, "numpy")
        return lambda t: float(f(t))
    v = float(voltage)
    return lambda t: v


def rhs(t, y, compiled, vfun, source_fun=None, source_idx=None):
    Vm = vfun(t)
    d = np.zeros_like(y)
    for _nm, ff, fo, rf, ro, st in compiled:
        net = 0.0
        if ff is not None:
            net += ff(*(y[i] for i in fo), Vm)
        if rf is not None:
            net -= rf(*(y[i] for i in ro), Vm)
        if net:
            for i, s in st.items():
                d[i] += s * net
    if source_fun is not None and source_idx is not None:
        d[source_idx] += source_fun(t)
    return d


def equilibrate_state(y0, compiled, vfun, *, names=(), chunk_s=50.0,
                      max_time_s=2000.0, rtol=1e-9, atol=1e-14):
    """Integrate at rest until calcium and all other derivatives settle."""
    y = np.asarray(y0, dtype=float).copy()
    ca = [i for i, name in enumerate(names) if name in {"Ca_c", "Ca_ER"}]
    other = [i for i in range(y.size) if i not in ca]
    stable = 0
    elapsed = 0.0
    while elapsed < max_time_s:
        end = min(elapsed + chunk_s, max_time_s)
        sol = solve_ivp(
            rhs, (elapsed, end), y, method="BDF", t_eval=[end],
            args=(compiled, vfun), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"equilibration failed: {sol.message}")
        y = sol.y[:, -1]
        deriv = rhs(end, y, compiled, vfun)
        ca_rel = max((abs(deriv[i]) / max(abs(y[i]), 1e-12)
                      for i in ca), default=0.0)
        other_abs = max((abs(deriv[i]) for i in other), default=0.0)
        stable = stable + 1 if ca_rel < 1e-6 and other_abs < 1e-9 else 0
        elapsed = end
        if stable == 2:
            return types.SimpleNamespace(
                state=y, time_s=elapsed, derivative=deriv,
                ca_relative_rate_s=ca_rel, other_rate=other_abs,
                converged=True)
    return types.SimpleNamespace(
        state=y, time_s=elapsed, derivative=deriv,
        ca_relative_rate_s=ca_rel, other_rate=other_abs, converged=False)


def _solve_with_pulses(rhs_fun, t_span, y0, *, method, t_eval, args,
                       pulse_times=(), pulse_amount=0.0, glu_idx=None,
                       reset_pulse=True,
                       extra_boundaries=(), rtol=1e-6, atol=1e-12):
    final_t = float(t_span[1])
    pulses = sorted(
        float(t) for t in pulse_times
        if 0.0 < float(t) < final_t
    )
    pulse_set = set(pulses)
    cuts = pulses + [
        float(t) for t in extra_boundaries
        if 0.0 < float(t) < final_t
    ]
    boundaries = sorted(set(cuts)) + [final_t]
    t0 = float(t_span[0])
    y = np.array(y0, dtype=float)
    all_t, all_y = [], []
    success = True
    messages = []

    for boundary in boundaries:
        mask = (t_eval >= t0) & (t_eval <= boundary)
        if all_t:
            mask &= t_eval > t0
        seg_eval = t_eval[mask]
        is_pulse_boundary = boundary in pulse_set
        if boundary == t0:
            if is_pulse_boundary and glu_idx is not None:
                y[glu_idx] = (
                    pulse_amount if reset_pulse else y[glu_idx] + pulse_amount
                )
            continue
        sol = solve_ivp(
            rhs_fun, (t0, boundary), y, method=method, t_eval=seg_eval,
            args=args, rtol=rtol, atol=atol, dense_output=True,
        )
        success = success and sol.success
        if not sol.success:
            messages.append(sol.message)
        sol_t = np.asarray(sol.t, dtype=float)
        sol_y = np.asarray(sol.y, dtype=float)
        if sol_t.size:
            all_t.append(sol_t)
            all_y.append(sol_y)
            y = sol.sol(boundary).copy()
        else:
            fallback = solve_ivp(
                rhs_fun, (t0, boundary), y, method=method, t_eval=[boundary],
                args=args, rtol=rtol, atol=atol,
            )
            success = success and fallback.success
            if not fallback.success:
                messages.append(fallback.message)
                break
            y = np.asarray(fallback.y, dtype=float)[:, -1]
        if is_pulse_boundary and boundary < final_t and glu_idx is not None:
            y[glu_idx] = (
                pulse_amount if reset_pulse else y[glu_idx] + pulse_amount
            )
        t0 = boundary

    return types.SimpleNamespace(
        t=np.concatenate(all_t) if all_t else np.array([], dtype=float),
        y=np.concatenate(all_y, axis=1) if all_y else np.empty((len(y0), 0)),
        success=success,
        message="; ".join(messages),
    )


def run(glu_freq=1.0, glu_pulses=1, glu_molecules=stimulus.GLUT_BOLUS_MOLECULES,
        v_rest=-65.0, clamp_mV=None, final_t=0.5,
        out="results/ode0d", n_points=200, include_soce=False,
        t_stim=0.0, param_overrides=None,
        ip3r_subunits=False, ryr_off=False, plc="new",
        equilibrate_first=False):
    p = default_parameters()

    # Apply any parameter overrides, keeping V_spine_L (liters) in sync with
    # V_spine (um^3) so the N/(N_A*V) scalings hold.
    if param_overrides:
        for k, v in param_overrides.items():
            setattr(p, k, float(v))
        p.V_spine_L = float(p.V_spine) * 1e-15
        if "plc_gq_r15_scale" not in param_overrides:
            p.plc_gq_r15_scale = balanced_r15_scale(p)

    # --- glutamate: a single train of identical pulse resets starting at
    # t_stim -- glu_pulses releases at glu_freq.
    pulse_times = sorted(
        stimulus.release_times(glu_freq, glu_pulses, t_stim=t_stim))
    if not pulse_times:
        pulse_amount = 0.0
    else:
        pulse_amount = stimulus.molecules_to_conc(glu_molecules, p)
    # A release at t=0 is set as the initial condition; later releases (and a
    # delayed first release at t_stim>0) are applied as resets during the solve.
    first_at_zero = bool(pulse_times) and abs(pulse_times[0]) < 1e-12
    p.Glu_init = pulse_amount if first_at_zero else 0.0

    # --- voltage: each glutamate release evokes an EPSP+bAP, unless we clamp
    if clamp_mV is None:
        volt = stimulus.voltage(times=pulse_times, V_rest=v_rest)
    else:
        volt = stimulus.hold_voltage(clamp_mV)

    names, y0, compiled, _ = build_system(p, include_soce=include_soce,
                                          ip3r_subunits=ip3r_subunits,
                                          ryr_off=ryr_off, plc=plc)
    vfun = _make_voltage(volt)
    glu_idx = names.index("Glu") if "Glu" in names else None

    equilibrium = None
    if equilibrate_first:
        if glu_idx is not None:
            y0[glu_idx] = 0.0
        rest_v = clamp_mV if clamp_mV is not None else v_rest
        max_equilibration_s = 2000.0
        equilibrium = equilibrate_state(
            y0, compiled, _make_voltage(rest_v), names=names,
            max_time_s=max_equilibration_s)
        if not equilibrium.converged:
            raise RuntimeError(
                f"resting state did not converge by {max_equilibration_s:g} s")
        y0 = equilibrium.state.copy()
        if first_at_zero and glu_idx is not None:
            y0[glu_idx] = pulse_amount

    # Restart the integrator at each sharp V_m onset (EPSP and bAP) so it does
    # not step over the kinks; no transient onsets when V_m is clamped.
    onsets = () if clamp_mV is not None else stimulus.voltage_onsets(pulse_times)

    t_eval = np.linspace(0.0, final_t, n_points)
    sol = _solve_with_pulses(
        rhs, (0.0, final_t), y0, method="BDF", t_eval=t_eval,
        args=(compiled, vfun),
        pulse_times=pulse_times, extra_boundaries=onsets,
        pulse_amount=pulse_amount, glu_idx=glu_idx, reset_pulse=True,
    )
    if not sol.success:
        print("WARNING solve_ivp:", sol.message)

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "traces.csv"
    # Record the prescribed V_m alongside the state so the stimulus and
    # response can be plotted on a common time grid.
    vm = np.array([vfun(tk) for tk in sol.t])
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "V_m"] + names)
        for k in range(sol.t.size):
            w.writerow([sol.t[k], vm[k]] + list(sol.y[:, k]))

    print(f"\n0D reference: {sol.t.size} pts x {len(names)} species "
          f"-> {csv_path}  (solve_ivp BDF, success={sol.success})")
    v_desc = (f"clamp {clamp_mV} mV" if clamp_mV is not None
              else f"Leung-style transient at {len(pulse_times)} release(s) "
                   f"(V_rest={v_rest} mV)")
    print(f"Stimulus: Glu_pulse={pulse_amount:.3e} M, "
          f"n_releases={len(pulse_times)} @ {glu_freq} Hz, V_m={v_desc}")
    print(f"SOCE: {'enabled' if include_soce else 'disabled'}")
    print(f"RyR:  {'DISABLED (--ryroff)' if ryr_off else 'enabled'}")
    print(f"PLC:  {plc}")
    if equilibrium:
        print(f"Equilibrated: {equilibrium.time_s:g} s, "
              f"Ca relative rate={equilibrium.ca_relative_rate_s:.2e} s^-1")
    for sp in ("Glu", "Ca_c", "Ca_ER", "IP3", "P_O_STIM1"):
        if sp in names:
            tr = sol.y[names.index(sp)]
            print(f"  {sp:10s} start={tr[0]:.4e} min={tr.min():.4e} "
                  f"max={tr.max():.4e} end={tr[-1]:.4e}")
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="0D reference integrator.")
    ap.add_argument("--glu-pulses", type=int, default=1, dest="glu_pulses",
                    help="number of glutamate boluses (0 = no glutamate, "
                         "1 = single bolus, N = a train)")
    ap.add_argument("--glu-freq", type=float, default=1.0, dest="glu_freq",
                    help="release frequency (Hz) for multi-pulse trains")
    ap.add_argument("--glu-molecules", type=float, default=500.0,
                    dest="glu_molecules",
                    help="molecules per bolus, converted to spine "
                         "concentration (default 500)")
    ap.add_argument("--v-rest", type=float, default=-65.0, dest="v_rest",
                    help="resting potential (mV); each release adds an "
                         "EPSP+bAP on top")
    ap.add_argument("--clamp", type=float, default=None,
                    help="use a constant V_m clamp (mV) instead of the "
                         "release-evoked transient")
    ap.add_argument("--final-t", type=float, default=0.5, dest="final_t")
    ap.add_argument("--n-points", type=int, default=200, dest="n_points")
    ap.add_argument("--out", default="results/ode0d")
    ap.add_argument("--soce", action="store_true",
                    help="include STIM/Orai1 SOCE with N_Orai1 from parameters")
    ap.add_argument("--t-stim", type=float, default=0.0, dest="t_stim",
                    help="time (s) of the first glutamate release (default 0)")
    ap.add_argument("--ip3r-subunits", action="store_true",
                    dest="ip3r_subunits",
                    help="use the explicit 3-subunit IP3R (24 states) instead "
                         "of the cubed single-subunit P_open")
    ap.add_argument("--ryroff", action="store_true", dest="ryr_off",
                    help="disable the RyR block entirely (no RyR states, no "
                         "CICR flux); ER membrane keeps only SERCA and IP3R")
    ap.add_argument("--plc", choices=("old", "new"), default="new",
                    help="PLC pathway implementation (default: new)")
    ap.add_argument("--equilibrate-first", action="store_true",
                    help="converge the no-stimulus resting state before the run")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY=VAL",
                    help="override a model parameter, e.g. --set N_Orai1=50 "
                         "(repeatable)")
    a = ap.parse_args(argv)
    overrides = {}
    for kv in a.overrides:
        key, sep, val = kv.partition("=")
        if not sep:
            ap.error(f"--set expects KEY=VAL, got {kv!r}")
        overrides[key.strip()] = float(val)
    run(glu_freq=a.glu_freq, glu_pulses=a.glu_pulses,
        glu_molecules=a.glu_molecules,
        v_rest=a.v_rest, clamp_mV=a.clamp, final_t=a.final_t, out=a.out,
        n_points=a.n_points, include_soce=a.soce,
        t_stim=a.t_stim, param_overrides=overrides,
        ip3r_subunits=a.ip3r_subunits, ryr_off=a.ryr_off, plc=a.plc,
        equilibrate_first=a.equilibrate_first)


if __name__ == "__main__":
    main()
