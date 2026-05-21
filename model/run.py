"""Entry point: run the well-mixed ODE model under a stimulus and save traces.

Run inside the SMART Docker container (see README), from the repo root::

    python -m model.run --glu glut_release --voltage -70 --final-t 0.5 \
        --out results/hm_constant

This first protocol is the HM-consistent **glutamate-only** test at a
constant voltage clamp (ODE_MODEL_PLAN.md default): a glutamate bolus is
released at t=0 with first-order clearance, V_m held constant. With V_m at
rest the NMDAR Mg2+ block is largely intact, so the expected Ca2+ response
is modest -- this validates the chemistry before adding the bAP transient.

Outputs: ``<out>/traces.csv`` (time + spine-mean of every species) and a
printed summary (resting vs peak Ca_c, ER Ca2+ drop).
"""

import argparse
import csv
import pathlib
import sys

# The 100-state RyR expansion (~240 reactions; cf. doc/CLAUDE.md) makes the
# Cyto Ca_c residual a very deep UFL Sum tree; FEniCS form compilation
# recurses through ufl_shape and overflows Python's default 1000 limit.
sys.setrecursionlimit(100_000)

from .ode_system import build_model
from .parameters import default_parameters
from . import stimulus


def _scalar(species):
    """Spine-mean of ONE species.

    `species.u["u"]` is the whole *compartment* mixed function (size =
    n_species_on_compartment x n_vertices); a plain .mean() averages across
    all species on that compartment (the per-compartment artifact bug).
    `species.dof_map` gives this species' DOF indices within that vector;
    well-mixed -> the sub-field is ~uniform so its mean is exact.
    """
    try:
        import numpy as np

        vec = np.asarray(species.u["u"].vector().get_local())
        dm = species.dof_map
        dm = dm() if callable(dm) else dm
        return float(vec[np.asarray(dm)].mean())
    except Exception:
        return float("nan")


def run(glu="glut_release", voltage=-70.0, final_t=0.5, out="results/hm_constant",
        initial_dt=1e-4, outer_radius=0.25, inner_radius=0.1, h_edge=0.05):
    p = default_parameters()
    voltage_expr = stimulus.apply(p, glu=glu, voltage=voltage)

    # Step-1 physics is 0D (every species D=0), but FEniCS still needs the
    # nested-sphere mesh resolved enough for the mixed-dimensional
    # cross-compartment coupling blocks: too coarse (h_edge >~ 0.1 with
    # inner_radius 0.1) -> PETSc MatSetValuesLocal out-of-range. h_edge=0.05
    # is the proven-working resolution. The ~15 min cost is form
    # compilation (mesh-independent), so coarsening barely helps anyway.
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

    print(f"\nWrote {len(rows)} timepoints x {len(names)} species -> {csv_path}")
    print(f"Stimulus: Glu_init = {p.Glu_init:.3e} M, V_m = {voltage} (mV)")
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
    ap.add_argument("--voltage", type=float, default=-70.0,
                    help="constant V_m clamp in mV")
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
    run(glu=glu, voltage=a.voltage, final_t=a.final_t, out=a.out,
        initial_dt=a.initial_dt, h_edge=a.h_edge,
        outer_radius=a.outer_radius, inner_radius=a.inner_radius)


if __name__ == "__main__":
    main()
