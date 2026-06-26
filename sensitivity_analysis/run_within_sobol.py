"""Within-group Sobol + PRCC: which *individual* parameter is the lever.

The grouped Sobol (run_sobol) said *which mechanism* controls the resting ER
drain (RyR, setpoints, SERCA-leak) and the multi-second outcome (SOCE).  This
runner drills inside chosen mechanism(s): it varies only those groups'
parameters, each as its own factor, with every other model parameter pinned at
nominal, and reports two complementary per-parameter measures:

  * Sobol S1 / ST  -- variance-based importance (unsigned; handles the
    non-monotonic drain<->refill competition correctly).
  * PRCC           -- signed direction (Leung et al. 2023 Fig. 2 style): which
    way to move each lever.  Computed from the same samples, no extra runs.

Both are rendered as parameter x QoI heatmaps.

Recommended runs
----------------
Drain levers (answers: fewer RyR channels? lower k_RyR? setpoints? leak?):
    conda run -n erpm python -m sensitivity_analysis.run_within_sobol \
        --groups ryr setpoints serca_leak --n 1024

SOCE floor (which SOCE parameter sets the multi-second ER level):
    conda run -n erpm python -m sensitivity_analysis.run_within_sobol \
        --groups soce --soce --trajectory --t-max 3.0 --n 512 \
        --out results/sensitivity/within_soce

Smoke test:
    conda run -n erpm python -m sensitivity_analysis.run_within_sobol \
        --groups ryr --n 16
"""

import argparse
import json
import pathlib

import numpy as np
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyze

from .core import build_problem, run_matrix, clean_for_analysis
from .prcc import prcc_matrix

# Resting QoIs are always available; trajectory QoIs need --trajectory.
RESTING_QOIS = ("abs_drain", "tau_drain", "ryr_open", "ip3r_open")
TRAJECTORY_QOIS = ("caer_frac_end", "ca_c_end")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["ryr", "setpoints", "serca_leak"],
                    help="mechanism group(s) to drill into (per-parameter)")
    ap.add_argument("--n", type=int, default=1024,
                    help="Sobol base sample N (evals = N*(P+2), P=#params)")
    ap.add_argument("--soce", action="store_true",
                    help="enable the SOCE module (required if --groups includes soce)")
    ap.add_argument("--trajectory", action="store_true",
                    help="also compute free-run QoIs (caer_frac_end / ca_c_end)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--t-max", type=float, default=1.0, dest="t_max",
                    help="free-integration window (s) for trajectory QoIs")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/sensitivity/within_sobol")
    a = ap.parse_args(argv)

    # Per-parameter (grouped=False) but restricted to the chosen mechanism(s).
    problem = build_problem(include_soce=a.soce, grouped=False,
                            only_groups=a.groups)
    P = problem["num_vars"]
    X = sobol_sampler.sample(problem, a.n, calc_second_order=False, seed=a.seed)
    print(f"Within-group Sobol: groups={a.groups}, {P} params, "
          f"N={a.n} -> {X.shape[0]} evaluations")

    Y = run_matrix(problem, X, include_soce=a.soce,
                   trajectory=a.trajectory, n_jobs=a.n_jobs, t_max=a.t_max)

    report = list(RESTING_QOIS)
    if a.trajectory:
        report += list(TRAJECTORY_QOIS)

    names = problem["names"]
    # PRCC uses the raw (log10) samples; rank correlation is transform-invariant.
    prcc = prcc_matrix(X, Y, report)

    results = {}
    for qoi in report:
        col, n_fail = clean_for_analysis(Y[qoi])
        Si = sobol_analyze.analyze(problem, col, calc_second_order=False,
                                   print_to_console=False)
        results[qoi] = {
            "names": names,
            "S1": [float(x) for x in Si["S1"]],
            "ST": [float(x) for x in Si["ST"]],
            "S1_conf": [float(x) for x in Si["S1_conf"]],
            "ST_conf": [float(x) for x in Si["ST_conf"]],
            "prcc": prcc.get(qoi, {}).get("rho"),
            "prcc_pval": prcc.get(qoi, {}).get("pval"),
            "n_failed": n_fail,
        }
        _print_table(qoi, names, Si, results[qoi]["prcc"], n_fail, X.shape[0])

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"groups": a.groups, "names": names, "n": a.n,
            "n_eval": int(X.shape[0]), "soce": a.soce,
            "trajectory": a.trajectory, "t_max": a.t_max}
    with open(out_dir / "within_sobol_indices.json", "w") as fh:
        json.dump({"meta": meta, "qois": results}, fh, indent=2)
    np.savez(out_dir / "samples.npz", X=X,
             **{f"Y_{k}": v for k, v in Y.items()})
    try:
        from .plots import plot_within_sobol_heatmap
        plot_within_sobol_heatmap(results, names, out_dir,
                                  title=" + ".join(a.groups))
        print(f"\nPlots + data -> {out_dir}/")
    except Exception as exc:  # plotting is optional
        print(f"\nData -> {out_dir}/  (plot skipped: {exc})")
    return results


def _print_table(qoi, names, Si, prcc, n_fail, n_eval, top=12):
    st = np.asarray(Si["ST"])
    order = np.argsort(st)[::-1][:top]
    print("\n" + "=" * 64)
    print(f"QoI: {qoi}   (ST-ranked; {n_fail}/{n_eval} evals failed)")
    print("=" * 64)
    print(f"  {'param':<16}{'S1':>9}{'ST':>9}{'PRCC':>9}")
    for i in order:
        pc = prcc[i] if prcc is not None else float("nan")
        print(f"  {names[i]:<16}{Si['S1'][i]:>9.3f}{Si['ST'][i]:>9.3f}{pc:>9.3f}")


if __name__ == "__main__":
    main()
