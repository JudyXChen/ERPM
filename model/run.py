import argparse
import csv
import pathlib
import sys

sys.setrecursionlimit(100_000)

from .ode_system import build_model
from .parameters import default_parameters
from . import stimulus


def _scalar(species):

    try:
        import numpy as np

        vec = np.asarray(species.u["u"].vector().get_local())
        dm = species.dof_map
        dm = dm() if callable(dm) else dm
        return float(vec[np.asarray(dm)].mean())
    except Exception:
        return float("nan")


def run(glu="glut_release", v_rest=-65.0, clamp_mV=None, final_t=0.5,
        out="results/hm_constant", initial_dt=1e-4, outer_radius=0.25,
        inner_radius=0.1, h_edge=0.05):
    p = default_parameters()

    # glutamate: a single bolus delivered as the initial Glu concentration
    if glu == "glut_release":
        p.Glu_init = stimulus.molecules_to_conc(
            stimulus.GLUT_BOLUS_MOLECULES, p)
    elif glu in (None, "none"):
        p.Glu_init = 0.0
    else:
        p.Glu_init = float(glu)

    # voltage: always-on V_rest+EPSP+BPAP transient, or a constant clamp
    if clamp_mV is None:
        voltage_expr = stimulus.voltage(V_rest=v_rest)
    else:
        voltage_expr = stimulus.hold_voltage(clamp_mV)

    m, _ = build_model(
        p, voltage_expr=voltage_expr, final_t=final_t, initial_dt=initial_dt,
        outer_radius=outer_radius, inner_radius=inner_radius, h_edge=h_edge,
    )

    names = [n for n, _ in m.sc.items]
    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "traces.csv"

    rows = []

    def snapshot():
        rows.append([float(m.t)] + [_scalar(m.sc[n]) for n in names])

    snapshot()
    while True:
        m.monolithic_solve()
        snapshot()
        if m.t >= m.final_t:
            break

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + names)
        w.writerows(rows)

    # Summary on the species we care about for a first look.
    def trace(name):
        i = names.index(name)
        return [r[i + 1] for r in rows]

    v_desc = (f"clamp {clamp_mV} mV" if clamp_mV is not None
              else f"transient (V_rest={v_rest} mV)")
    print(f"\nWrote {len(rows)} timepoints x {len(names)} species -> {csv_path}")
    print(f"Stimulus: Glu_init = {p.Glu_init:.3e} M, V_m = {v_desc}")
    for sp, lo_is_drop in (("Ca_c", False), ("Ca_ER", True)):
        if sp in names:
            tr = trace(sp)
            print(
                f"  {sp:6s}: start={tr[0]:.4e}  "
                f"min={min(tr):.4e}  max={max(tr):.4e}  end={tr[-1]:.4e} M"
            )
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run ODE model under a stimulus.")
    ap.add_argument("--glu", default="glut_release",
                    help="'glut_release', a concentration in M, or 'none'")
    ap.add_argument("--v-rest", type=float, default=-65.0, dest="v_rest",
                    help="resting potential (mV) of the always-on "
                         "V_rest+EPSP+BPAP transient")
    ap.add_argument("--clamp", type=float, default=None,
                    help="constant V_m clamp (mV) instead of the always-on "
                         "transient")
    ap.add_argument("--final-t", type=float, default=0.5, dest="final_t")
    ap.add_argument("--initial-dt", type=float, default=1e-4, dest="initial_dt")
    ap.add_argument("--out", default="results/hm_constant")
    ap.add_argument("--h-edge", type=float, default=0.05, dest="h_edge",
                    help="mesh edge size; 0.05 proven, >~0.1 breaks assembly")
    ap.add_argument("--outer-radius", type=float, default=0.25,
                    dest="outer_radius")
    ap.add_argument("--inner-radius", type=float, default=0.1,
                    dest="inner_radius")
    a = ap.parse_args(argv)
    glu = None if a.glu == "none" else a.glu
    run(glu=glu, v_rest=a.v_rest, clamp_mV=a.clamp, final_t=a.final_t,
        out=a.out, initial_dt=a.initial_dt, h_edge=a.h_edge,
        outer_radius=a.outer_radius, inner_radius=a.inner_radius)


if __name__ == "__main__":
    main()
