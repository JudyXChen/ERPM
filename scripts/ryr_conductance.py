"""Steady-state RyR open probability with every O(ja, ji) state conducting.

Run with the ERPM Python environment. This diagnostic reads the current RyR
reaction registry and parameters; it does not alter model kinetics or the
model's flux observable. Q uses the column-vector convention dp/dt = Q @ p.
Calcium concentrations are clamped independently at each point, not evolved.

Example:
    MPLCONFIGDIR=/tmp/erpm-mpl python -B scripts/ryr_conductance.py
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
import sympy as sym
from scipy.linalg import expm, solve

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the project's lightweight reaction containers when SMART is absent.
from model import ode0d  # noqa: F401, E402
from model.parameters import default_parameters  # noqa: E402
from model.reactions import ryr  # noqa: E402
from scripts.plot_style import apply_style  # noqa: E402
from scripts import plot_style


def build_generator(parameters, *, original_gate=False):
    """Compile all receptor transition rates from the registered reactions."""
    species, registered_parameters, reactions = ryr.register(parameters)
    if original_gate:
        # Historical bypass is available only for isolated-channel comparison.
        for reaction in reactions:
            if reaction.name.startswith("RyR_CIOI_"):
                source = next(iter(reaction.lhs))
                reaction.eqn_f_str = f"alpha*{source}"
    names = [item.name for item in species]
    index = {name: i for i, name in enumerate(names)}
    values = {item.name: float(item.value) for item in registered_parameters}
    ca_c, ca_er = sym.symbols("Ca_c Ca_ER")
    sources, destinations, rates = [], [], []
    excluded = []
    for reaction in reactions:
        left = [name for name in reaction.lhs if name in index]
        right = [name for name in reaction.rhs if name in index]
        if not left and not right:
            excluded.append(reaction.name)  # Ion transport is not state gating.
            continue
        if len(left) != 1 or len(right) != 1:
            raise ValueError(f"Expected one receptor on each side: {reaction.name}")
        substitutions = {
            sym.Symbol(alias): values[name]
            for alias, name in reaction.param_map.items()
        }
        for expression, source, destination in (
            (reaction.eqn_f_str, left[0], right[0]),
            (reaction.eqn_r_str, right[0], left[0]),
        ):
            if not expression:
                continue
            expr = sym.sympify(expression).subs(substitutions)
            state = sym.Symbol(source)
            coefficient = sym.diff(expr, state)
            if sym.simplify(expr - state * coefficient) != 0:
                raise ValueError(f"Nonlinear receptor transition: {reaction.name}")
            if coefficient.free_symbols - {ca_c, ca_er}:
                raise ValueError(f"Unresolved rate symbols: {reaction.name}")
            sources.append(index[source])
            destinations.append(index[destination])
            rates.append(coefficient)
    sources, destinations = np.asarray(sources), np.asarray(destinations)
    rate_function = sym.lambdify((ca_c, ca_er), rates, "numpy")

    def generator(cytosol_m, er_m):
        rate_values = np.asarray(rate_function(cytosol_m, er_m), dtype=float)
        if not np.isfinite(rate_values).all() or np.any(rate_values < 0):
            raise ValueError("Transition rates must be finite and nonnegative")
        q = np.zeros((len(names), len(names)))
        np.add.at(q, (destinations, sources), rate_values)
        np.add.at(q, (sources, sources), -rate_values)
        return q

    return names, generator, len(rates), excluded


def stationary_distribution(q):
    """Solve Qp=0 after replacing one equation by sum(p)=1."""
    system = q.copy()
    system[-1, :] = 1.0
    target = np.zeros(len(q))
    target[-1] = 1.0
    probability = solve(system, target)
    if not np.isfinite(probability).all() or probability.min() < -1e-12:
        raise RuntimeError("Invalid stationary probability distribution")
    # Remove only negative floating-point roundoff; retain small positive states.
    probability = np.maximum(probability, 0.0)
    probability /= probability.sum()
    scale = max(1.0, np.linalg.norm(q, ord=1))
    if np.linalg.norm(q @ probability, ord=np.inf) / scale > 1e-12:
        raise RuntimeError("Stationary residual exceeds tolerance")
    return probability


def plot_conductance(path, calcium, p_open, *, ca_er, resting_po=None):
    """Render recorded gating values without recomputing the model."""
    apply_style()
    fig, ax = plot_style.figure()
    ax.semilogx(calcium, p_open, color=plot_style.PATHWAY_COLORS["RyR"])
    ax.set(xlabel=r"Cytosolic Ca$^{2+}$ (µM)", ylabel=r"Open probability, $P_O$",
           xlim=(min(calcium), max(calcium)), ylim=(0, max(0.05, max(p_open)*1.15)))
    ax.text(0.04, 0.95, rf"ER Ca$^{{2+}}$ = {ca_er:g} µM" + "\nAll 25 O states conduct",
            transform=ax.transAxes, va="top")
    if resting_po is not None and min(calcium) <= 0.1 <= max(calcium):
        ax.plot(0.1, resting_po, "o", color=plot_style.PATHWAY_COLORS["RyR"])
        ax.annotate(f"0.1 µM: {resting_po:.5f}", (0.1, resting_po),
                    xytext=(8, 8), textcoords="offset points")
    plot_style.save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-min", type=float, default=0.001, help="Cytosolic minimum (uM)")
    parser.add_argument("--ca-max", type=float, default=1000.0, help="Cytosolic maximum (uM)")
    parser.add_argument("--points", type=int, default=601)
    parser.add_argument("--ca-er", type=float, default=150.0, help="Clamped ER calcium (uM)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "results/ryr_conductance")
    args = parser.parse_args()
    if not 0 < args.ca_min < args.ca_max or args.points < 2 or args.ca_er < 0:
        parser.error("Require 0 < ca-min < ca-max, points >= 2, and ca-er >= 0")

    parameters = default_parameters()
    names, generator, transition_count, excluded = build_generator(parameters)
    open_indices = [i for i, name in enumerate(names) if name.startswith("R_O_")]
    if len(names) != 100 or len(open_indices) != 25:
        raise RuntimeError("Expected 100 receptor states and 25 conducting O states")
    sample_ca = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 150.0, 1000.0]
    calcium = np.unique(np.r_[
        np.geomspace(args.ca_min, args.ca_max, args.points),
        [value for value in sample_ca if args.ca_min <= value <= args.ca_max],
    ])
    er_m = args.ca_er * 1e-6
    occupancies = []
    residuals, column_errors, marginal_errors = [], [], []
    activation_counts = np.array([int(name.split("_")[-2]) for name in names])
    # Independent check: all macrostates share the same activation binding rates.
    from math import comb
    for cytosol_um in calcium:
        q = generator(cytosol_um * 1e-6, er_m)
        probability = stationary_distribution(q)
        occupancies.append(probability)
        residuals.append(float(np.max(np.abs(q @ probability))))
        column_errors.append(float(np.max(np.abs(q.sum(axis=0)))))
        fraction = parameters.k_AB * cytosol_um * 1e-6
        fraction /= fraction + parameters.k_AU
        expected = np.array([comb(4, j) * fraction**j * (1-fraction)**(4-j) for j in range(5)])
        observed = np.array([probability[activation_counts == j].sum() for j in range(5)])
        marginal_errors.append(float(np.max(np.abs(observed - expected))))
    occupancies = np.asarray(occupancies)
    p_open = occupancies[:, open_indices].sum(axis=1)
    if max(marginal_errors) > 1e-10:
        raise RuntimeError("Activation occupancy failed the independent binomial check")

    # Check relaxation from C00 against the stationary solution at three clamps.
    initial = np.zeros(len(names))
    initial[names.index("R_C_0_0")] = 1.0
    relaxation_errors = {}
    for cytosol_um in (0.1, 1.0, 10.0):
        q = generator(cytosol_um * 1e-6, er_m)
        error = np.max(np.abs(expm(q * 100.0) @ initial - stationary_distribution(q)))
        relaxation_errors[str(cytosol_um)] = float(error)
        if error > 1e-8:
            raise RuntimeError("Time-domain relaxation does not match stationary solution")

    samples = []
    for cytosol_um in sample_ca:
        probability = stationary_distribution(generator(cytosol_um * 1e-6, er_m))
        samples.append({"Ca_c_uM": cytosol_um, "P_open": float(probability[open_indices].sum())})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "ryr_conductance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Ca_c_uM", "Ca_ER_uM", "P_open", "k_RyR_times_P_open_M_inv_s_inv", *names])
        for ca, po, probability in zip(calcium, p_open, occupancies):
            writer.writerow([ca, args.ca_er, po, parameters.k_RyR * po, *probability])

    plot_conductance(args.out_dir / "ryr_conductance.png", calcium, p_open,
                     ca_er=args.ca_er, resting_po=next(
                         row["P_open"] for row in samples if row["Ca_c_uM"] == 0.1))

    metadata = {
        "method": "Qp=0; sum(p)=1; Q[destination, source] is a transition rate in s^-1",
        "conducting_states": [names[i] for i in open_indices],
        "state_count": len(names), "directed_transition_count": transition_count,
        "excluded_transport_reactions": excluded,
        "Ca_ER_uM": args.ca_er, "Ca_c_range_uM": [args.ca_min, args.ca_max],
        "sweep_points": len(calcium),
        "note": "P_open is normalized conductance for equal-conductance O states. ER calcium does not enter these gating rates; it enters the separate release driving force.",
        "parameters": {key: getattr(parameters, key) for key in (
            "k_AB", "k_AU", "k_IC", "k_IO", "k_IU", "alpha_RyR", "beta_RyR",
            "delta_prime", "delta_RyR", "gamma_RyR", "k_RyR")},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / "model/reactions/ryr.py", ROOT / "model/parameters.py", pathlib.Path(__file__))},
        "checks": {"max_abs_Qp_per_s": max(residuals), "max_abs_column_sum_per_s": max(column_errors),
                   "max_abs_probability_sum_error": float(np.max(np.abs(occupancies.sum(axis=1)-1))),
                   "min_probability": float(occupancies.min()),
                   "max_activation_binomial_error": max(marginal_errors),
                   "relaxation_100s_max_probability_errors": relaxation_errors},
        "sample_values": samples,
    }
    (args.out_dir / "ryr_conductance_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.out_dir), "sample_values": samples, "checks": metadata["checks"]}, indent=2))


if __name__ == "__main__":
    main()
