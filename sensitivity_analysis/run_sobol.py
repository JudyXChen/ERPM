"""Sobol sensitivity of the resting ER balance (grouped or within-group).

Default (--groups all): grouped Sobol, one index per mechanism -- "X% of the
variance in the resting drain is controlled by RyR, Y% by setpoints, ...".
Name groups instead (--groups ryr setpoints serca_leak): drill inside them,
one per-parameter index each with the rest of the model pinned at nominal.
Both report Sobol S1 / ST, rendered as a heatmap.

--stimulus switches the QoIs from the resting ER balance to the single-pulse
stimulus response (Ca_cyto transient features + per-pathway Ca flux), so the
heatmap attributes each response feature to a mechanism (or parameter).

Run:
    python -m sensitivity_analysis.run_sobol --n 1024
    python -m sensitivity_analysis.run_sobol --stimulus --n 1024
    python -m sensitivity_analysis.run_sobol --groups ryr setpoints serca_leak --n 1024
"""

import argparse

import numpy as np
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyze

from .core import (build_problem, run_matrix, clean_for_analysis,
                   add_common_args, save_run)
from .evaluate import STIM_QOI_NAMES

# Resting QoIs are always available; trajectory/stimulus QoIs need their flag.
GROUPED_QOIS = ("abs_drain", "tau_drain")
WITHIN_QOIS = ("abs_drain", "tau_drain", "ryr_open", "ip3r_open")
TRAJECTORY_QOIS = ("caer_frac_end", "ca_c_end")
STIM_QOIS = STIM_QOI_NAMES


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["all"],
                    help="'all' = grouped mechanism Sobol; or name group(s) to "
                         "drill into per-parameter")
    ap.add_argument("--n", type=int, default=1024,
                    help="Sobol base sample N (evals = N*(K+2))")
    add_common_args(ap, default_out=None)
    a = ap.parse_args(argv)

    grouped = a.groups == ["all"]
    problem = build_problem(grouped=grouped,
                            only_groups=None if grouped else a.groups)
    keys = (list(dict.fromkeys(problem["groups"])) if grouped
            else problem["names"])
    X = sobol_sampler.sample(problem, a.n, calc_second_order=False, seed=a.seed)
    label = "grouped" if grouped else " + ".join(a.groups)
    print(f"Sobol ({label}): {problem['num_vars']} params, N={a.n} "
          f"-> {X.shape[0]} evaluations")

    Y = run_matrix(problem, X, trajectory=a.trajectory, stim=a.stimulus,
                   n_jobs=a.n_jobs, t_max=a.t_max, stim_window=a.stim_window)

    # --stimulus drives the stimulus-response QoIs; otherwise report the
    # resting/free-run ER-balance QoIs.
    if a.stimulus:
        report = list(STIM_QOIS)
    else:
        report = list(GROUPED_QOIS if grouped else WITHIN_QOIS)
        if a.trajectory:
            report += list(TRAJECTORY_QOIS)
    keyname = "groups" if grouped else "names"

    results = {}
    for qoi in report:
        col, n_fail = clean_for_analysis(Y[qoi])
        Si = sobol_analyze.analyze(problem, col, calc_second_order=False,
                                   print_to_console=False)
        results[qoi] = {
            keyname: keys,
            "S1": [float(x) for x in Si["S1"]],
            "ST": [float(x) for x in Si["ST"]],
            "S1_conf": [float(x) for x in Si["S1_conf"]],
            "ST_conf": [float(x) for x in Si["ST_conf"]],
            "n_failed": n_fail,
        }
        _print_table(qoi, keys, Si, n_fail, X.shape[0])

    if grouped:
        out_dir = a.out or "results/sensitivity/grouped_sobol"
        json_name = "sobol_indices.json"

        def plot(out):
            from .plots import plot_grouped_sobol, plot_grouped_sobol_heatmap
            plot_grouped_sobol(results, out)
            plot_grouped_sobol_heatmap(results, out)
    else:
        out_dir = a.out or f"results/sensitivity/within_{'_'.join(a.groups)}"
        json_name = "within_sobol_indices.json"

        def plot(out):
            from .plots import plot_within_sobol_heatmap
            plot_within_sobol_heatmap(results, keys, out,
                                      title=" + ".join(a.groups))

    save_run(out_dir, json_name, results, X=X, Y=Y, plot=plot)
    return results


def _print_table(qoi, keys, Si, n_fail, n_eval, top=15):
    order = np.argsort(Si["ST"])[::-1][:top]
    print("\n" + "=" * 60)
    print(f"QoI: {qoi}   (ST-ranked; {n_fail}/{n_eval} failed)")
    print("=" * 60)
    print(f"  {'name':<16}{'S1':>9}{'ST':>9}{'ST-S1':>9}")
    for i in order:
        s1, st = Si["S1"][i], Si["ST"][i]
        print(f"  {keys[i]:<16}{s1:>9.3f}{st:>9.3f}{st - s1:>9.3f}")


if __name__ == "__main__":
    main()
