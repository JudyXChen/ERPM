"""Grouped Sobol analysis of the resting ER balance (mechanism attribution).

Each reaction module (SERCA, RyR, IP3R, buffers, ...) is one Sobol factor, so
the result reads "X% of the variance in the resting ER drain is controlled by
SERCA, Y% by RyR, Z% by their interaction".  First/total-order only
(calc_second_order=False) to keep the sample count affordable.

Run (smoke test):
    conda run -n erpm python -m sensitivity_analysis.run_sobol --n 16
Run (full, ~50 min / 8 cores):
    conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024
"""

import argparse
import json
import pathlib

import numpy as np
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyze

from .core import build_problem, run_matrix, clean_for_analysis
from .evaluate import QOI_NAMES

# QoIs worth attributing.  net/abs drain and the trajectory outcomes are the
# headline resting-stability metrics.
REPORT_QOIS = ("abs_drain", "tau_drain", "caer_frac_end", "ca_c_end")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1024,
                    help="Sobol base sample N (evals = N*(G+2))")
    ap.add_argument("--soce", action="store_true")
    ap.add_argument("--trajectory", action="store_true",
                    help="also compute free-run QoIs (slower; needed for "
                         "caer_frac_end / t_half / ca_c_end)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--t-max", type=float, default=1.0, dest="t_max",
                    help="free-integration window (s) for trajectory QoIs; "
                         "set to your simulation horizon (e.g. 3.0)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/sensitivity/grouped_sobol")
    a = ap.parse_args(argv)

    problem = build_problem(include_soce=a.soce, grouped=True)
    n_groups = len(dict.fromkeys(problem["groups"]))
    X = sobol_sampler.sample(problem, a.n, calc_second_order=False,
                             seed=a.seed)
    print(f"Grouped Sobol: {n_groups} groups, {problem['num_vars']} params, "
          f"N={a.n} -> {X.shape[0]} evaluations")

    Y = run_matrix(problem, X, include_soce=a.soce,
                   trajectory=a.trajectory, n_jobs=a.n_jobs, t_max=a.t_max)

    report = [q for q in REPORT_QOIS
              if a.trajectory or q in ("abs_drain", "tau_drain")]
    results = {}
    for qoi in report:
        col, n_fail = clean_for_analysis(Y[qoi])
        Si = sobol_analyze.analyze(problem, col, calc_second_order=False,
                                   print_to_console=False)
        groups = list(dict.fromkeys(problem["groups"]))
        results[qoi] = {
            "groups": groups,
            "S1": [float(x) for x in Si["S1"]],
            "ST": [float(x) for x in Si["ST"]],
            "S1_conf": [float(x) for x in Si["S1_conf"]],
            "ST_conf": [float(x) for x in Si["ST_conf"]],
            "n_failed": n_fail,
        }
        _print_table(qoi, groups, Si, n_fail, X.shape[0])

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sobol_indices.json", "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(out_dir / "samples.npz", X=X, **{f"Y_{k}": v for k, v in Y.items()})
    try:
        from .plots import plot_grouped_sobol
        plot_grouped_sobol(results, out_dir)
        print(f"\nPlots + data -> {out_dir}/")
    except Exception as exc:  # plotting is optional
        print(f"\nData -> {out_dir}/  (plot skipped: {exc})")
    return results


def _print_table(qoi, groups, Si, n_fail, n_eval):
    order = np.argsort(Si["ST"])[::-1]
    print("\n" + "=" * 64)
    print(f"QoI: {qoi}   (ST-ranked; {n_fail}/{n_eval} evals failed)")
    print("=" * 64)
    print(f"  {'mechanism':<16}{'S1':>10}{'ST':>10}{'ST-S1(interact)':>18}")
    for i in order:
        s1, st = Si["S1"][i], Si["ST"][i]
        print(f"  {groups[i]:<16}{s1:>10.3f}{st:>10.3f}{st - s1:>18.3f}")


if __name__ == "__main__":
    main()
