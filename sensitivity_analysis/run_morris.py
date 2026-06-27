"""Morris elementary-effects screen over all parameters individually.

A cheap global screen ranking each parameter by mu_star (overall influence)
and sigma (nonlinearity / interaction); pre-screen for the Sobol runs.

Run:
    python -m sensitivity_analysis.run_morris --r 20
"""

import argparse

import numpy as np
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyze

from .core import (build_problem, run_matrix, clean_for_analysis,
                   add_common_args, save_run)

REPORT_QOIS = ("abs_drain", "tau_drain")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=int, default=20,
                    help="number of Morris trajectories (evals = r*(P+1))")
    ap.add_argument("--levels", type=int, default=4)
    add_common_args(ap, default_out="results/sensitivity/morris")
    a = ap.parse_args(argv)

    problem = build_problem(grouped=False)  # per-parameter screen
    X = morris_sampler.sample(problem, a.r, num_levels=a.levels, seed=a.seed)
    print(f"Morris: {problem['num_vars']} params, r={a.r} "
          f"-> {X.shape[0]} evaluations")

    Y = run_matrix(problem, X, trajectory=a.trajectory,
                   n_jobs=a.n_jobs, t_max=a.t_max)

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

    def plot(out):
        from .plots import plot_morris
        plot_morris(results, out)

    save_run(a.out, "morris_indices.json", results, plot=plot)
    return results


def _print_table(qoi, Si, n_fail, n_eval, top=15):
    order = np.argsort(Si["mu_star"])[::-1][:top]
    print("\n" + "=" * 56)
    print(f"QoI: {qoi}   (top {top} by mu*; {n_fail}/{n_eval} failed)")
    print("=" * 56)
    print(f"  {'param':<16}{'mu_star':>14}{'sigma':>14}")
    for i in order:
        print(f"  {Si['names'][i]:<16}{Si['mu_star'][i]:>14.3e}"
              f"{Si['sigma'][i]:>14.3e}")


if __name__ == "__main__":
    main()
