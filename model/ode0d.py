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

from .parameters import default_parameters


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

from .reactions import BASE_REGISTRARS, SOCE_REGISTRARS
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


def build_system(p, *, include_soce=False):
    from smart.model_assembly import Reaction  # container objects only

    registrars = SOCE_REGISTRARS if include_soce else BASE_REGISTRARS
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
            args=args, rtol=rtol, atol=atol,
        )
        success = success and sol.success
        if not sol.success:
            messages.append(sol.message)
        sol_t = np.asarray(sol.t, dtype=float)
        sol_y = np.asarray(sol.y, dtype=float)
        if sol_t.size:
            all_t.append(sol_t)
            all_y.append(sol_y)
            y = sol_y[:, -1].copy()
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
        glu_conc=None, v_rest=-65.0, clamp_mV=None, final_t=0.5,
        out="results/ode0d", n_points=200, method="BDF", include_soce=False,
        glu_trains=1, train_interval=1.0):
    p = default_parameters()

    # --- glutamate: one or more trains of identical pulse resets.
    # Each train is glu_pulses at glu_freq; train k starts at k*train_interval.
    pulse_times = []
    for k in range(int(glu_trains)):
        pulse_times.extend(
            stimulus.release_times(glu_freq, glu_pulses,
                                   t_stim=k * train_interval))
    pulse_times = sorted(pulse_times)
    if not pulse_times:
        pulse_amount = 0.0
    elif glu_conc is not None:
        pulse_amount = float(glu_conc)
    else:
        pulse_amount = stimulus.molecules_to_conc(glu_molecules, p)
    p.Glu_init = pulse_amount if pulse_times else 0.0

    # --- voltage: each glutamate release evokes an EPSP+bAP, unless we clamp
    if clamp_mV is None:
        volt = stimulus.voltage(times=pulse_times, V_rest=v_rest)
    else:
        volt = stimulus.hold_voltage(clamp_mV)

    names, y0, compiled, _ = build_system(p, include_soce=include_soce)
    vfun = _make_voltage(volt)
    glu_idx = names.index("Glu") if "Glu" in names else None

    # Restart the integrator at each sharp V_m onset (EPSP and bAP) so it does
    # not step over the kinks; no transient onsets when V_m is clamped.
    onsets = () if clamp_mV is not None else stimulus.voltage_onsets(pulse_times)

    t_eval = np.linspace(0.0, final_t, n_points)
    sol = _solve_with_pulses(
        rhs, (0.0, final_t), y0, method=method, t_eval=t_eval,
        args=(compiled, vfun),
        pulse_times=pulse_times[1:], extra_boundaries=onsets,
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
          f"-> {csv_path}  (solve_ivp {method}, success={sol.success})")
    v_desc = (f"clamp {clamp_mV} mV" if clamp_mV is not None
              else f"Leung-style transient at {len(pulse_times)} release(s) "
                   f"(V_rest={v_rest} mV)")
    print(f"Stimulus: Glu_pulse={pulse_amount:.3e} M, "
          f"n_releases={len(pulse_times)} @ {glu_freq} Hz, V_m={v_desc}")
    print(f"SOCE: {'enabled' if include_soce else 'disabled'}")
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
    ap.add_argument("--glu-conc", type=float, default=None, dest="glu_conc",
                    help="concentration (M) per bolus; overrides "
                         "--glu-molecules")
    ap.add_argument("--v-rest", type=float, default=-65.0, dest="v_rest",
                    help="resting potential (mV); each release adds an "
                         "EPSP+bAP on top")
    ap.add_argument("--clamp", type=float, default=None,
                    help="use a constant V_m clamp (mV) instead of the "
                         "release-evoked transient")
    ap.add_argument("--final-t", type=float, default=0.5, dest="final_t")
    ap.add_argument("--n-points", type=int, default=200, dest="n_points")
    ap.add_argument("--method", default="BDF")
    ap.add_argument("--out", default="results/ode0d")
    ap.add_argument("--soce", action="store_true",
                    help="include STIM/Orai1 SOCE with N_Orai1 from parameters")
    ap.add_argument("--glu-trains", type=int, default=1, dest="glu_trains",
                    help="number of pulse trains (each is --glu-pulses at "
                         "--glu-freq)")
    ap.add_argument("--train-interval", type=float, default=1.0,
                    dest="train_interval",
                    help="seconds between the start of successive trains")
    a = ap.parse_args(argv)
    run(glu_freq=a.glu_freq, glu_pulses=a.glu_pulses,
        glu_molecules=a.glu_molecules, glu_conc=a.glu_conc,
        v_rest=a.v_rest, clamp_mV=a.clamp, final_t=a.final_t, out=a.out,
        n_points=a.n_points, method=a.method, include_soce=a.soce,
        glu_trains=a.glu_trains, train_interval=a.train_interval)


if __name__ == "__main__":
    main()
