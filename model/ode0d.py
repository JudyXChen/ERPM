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
    """Provide the small SMART API subset needed to assemble pure 0D ODEs."""
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

from .reactions import REGISTRARS
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
        "k_glu_clearance": float(getattr(p, "glu_clearance_rate", 0.0)),
    }


def build_system(p):
    from smart.model_assembly import Reaction  # container objects only

    species_init, master, reactions = {}, _shared_params(p), []
    for reg in REGISTRARS:
        slist, plist, rlist = reg(p)
        for s in slist:
            species_init.setdefault(s.name, float(s.initial_condition))
        for par in plist:
            master[par.name] = float(par.value)
        reactions.extend(rlist)

    # ode_system.build_model adds this one outside the registrars.
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


def rhs(t, y, compiled, vfun):
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
    return d


def run(glu="glut_release", voltage=-70.0, final_t=0.5, out="results/ode0d",
        n_points=200, method="BDF"):
    p = default_parameters()
    volt = stimulus.apply(p, glu=glu, voltage=voltage)
    names, y0, compiled, _ = build_system(p)
    vfun = _make_voltage(volt)

    t_eval = np.linspace(0.0, final_t, n_points)
    sol = solve_ivp(rhs, (0.0, final_t), y0, method=method, t_eval=t_eval,
                     args=(compiled, vfun), rtol=1e-6, atol=1e-12)
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
    v_desc = "bap_transient" if isinstance(volt, sym.Expr) else f"{voltage} mV"
    print(f"Stimulus: Glu_init={p.Glu_init:.3e} M, V_m={v_desc}")
    for sp in ("Glu", "Ca_c", "Ca_ER", "IP3", "P_O_STIM1", "P_O_STIM2"):
        if sp in names:
            tr = sol.y[names.index(sp)]
            print(f"  {sp:10s} start={tr[0]:.4e} min={tr.min():.4e} "
                  f"max={tr.max():.4e} end={tr[-1]:.4e}")
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="0D reference integrator.")
    ap.add_argument("--glu", default="glut_release")
    ap.add_argument("--voltage", type=float, default=-70.0)
    ap.add_argument("--bap", action="store_true",
                    help="use the causal V_rest+BPAP+EPSP transient instead "
                         "of the constant --voltage clamp")
    ap.add_argument("--final-t", type=float, default=0.5, dest="final_t")
    ap.add_argument("--n-points", type=int, default=200, dest="n_points")
    ap.add_argument("--method", default="BDF")
    ap.add_argument("--out", default="results/ode0d")
    a = ap.parse_args(argv)
    glu = None if a.glu == "none" else a.glu
    # bAP transient (sympy expr in t) overrides the scalar --voltage clamp.
    voltage = (
        stimulus.bap_voltage_expr()
        if a.bap else a.voltage
    )
    run(glu=glu, voltage=voltage, final_t=a.final_t, out=a.out,
        n_points=a.n_points, method=a.method)


if __name__ == "__main__":
    main()
