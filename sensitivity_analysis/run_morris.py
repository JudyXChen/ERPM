"""Morris elementary-effects screen over all parameters individually.

A cheap global screen that ranks every parameter by mu_star (overall influence)
and sigma (nonlinearity / interaction). 

Run:
    conda run -n erpm python -m sensitivity_analysis.run_morris --r 20
"""

import argparse
import json
import pathlib

import numpy as np
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyze

from .core import build_problem, run_matrix, clean_for_analysis

REPORT_QOIS = ("abs_drain", "tau_drain")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=int, default=20,
                    help="number of Morris trajectories (evals = r*(P+1))")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--soce", action="store_true")
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--t-max", type=float, default=1.0, dest="t_max",
                    help="free-integration window (s) for trajectory QoIs")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/sensitivity/morris")
    a = ap.parse_args(argv)

    # per-parameter screen -> no groups
    problem = build_problem(include_soce=a.soce, grouped=False)
    X = morris_sampler.sample(problem, a.r, num_levels=a.levels, seed=a.seed)
    print(f"Morris: {problem['num_vars']} params, r={a.r} "
          f"-> {X.shape[0]} evaluations")

    Y = run_matrix(problem, X, include_soce=a.soce,
                   trajectory=a.trajectory, n_jobs=a.n_jobs, t_max=a.t_max)

    report = REPORT_QOIS + (("caer_frac_end",) if a.trajectory else ())
    results = {}
    for qoi in report:
        col, n_fail = clean_for_analysis(Y[qoi])
        Si = morris_analyze.analyze(problem, X, col, num_levels=a.levels,
                                    print_to_console=False)
        results[qoi] = {
            "names": list(Si["names"]),
            "mu_star": [float(x) for x in Si["mu_star"]],
            "sigma": [float(x) for x in Si["sigma"]],
            "n_failed": n_fail,
        }
        _print_table(qoi, Si, n_fail, X.shape[0])

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "morris_indices.json", "w") as fh:
        json.dump(results, fh, indent=2)
    try:
        from .plots import plot_morris
        plot_morris(results, out_dir)
        print(f"\nPlots + data -> {out_dir}/")
    except Exception as exc:
        print(f"\nData -> {out_dir}/  (plot skipped: {exc})")
    return results


def _print_table(qoi, Si, n_fail, n_eval, top=15):
    mu = np.asarray(Si["mu_star"])
    order = np.argsort(mu)[::-1][:top]
    print("\n" + "=" * 56)
    print(f"QoI: {qoi}   (top {top} by mu*; {n_fail}/{n_eval} failed)")
    print("=" * 56)
    print(f"  {'param':<16}{'mu_star':>14}{'sigma':>14}")
    for i in order:
        print(f"  {Si['names'][i]:<16}{Si['mu_star'][i]:>14.3e}"
              f"{Si['sigma'][i]:>14.3e}")


if __name__ == "__main__":
    main()
