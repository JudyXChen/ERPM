# sensitivity_analysis

Global sensitivity analysis of the well-mixed **0D** ER-PM model
(`model.ode0d`), focused on the **resting baseline** (no stimulus): which
mechanism controls the resting ER Ca²⁺ balance and the ER-drain stability.

It reuses `analysis.er_depletion` for the resting QoIs (gating is
pre-equilibrated with Ca_c/Ca_ER clamped, then the resting flux balance is read
off by reaction), so the SA measures exactly the same quantities as the existing
one-at-a-time sweeps — only globally and grouped.

## Design

- **Grouping by mechanism.** Parameters are grouped into reaction modules
  (SERCA-uptake, SERCA-leak, RyR, IP3R, PMCA/NCX, buffers, IP3-basal, geometry,
  setpoints, SOCE). Edit `config.GROUPS` to re-cut. A *grouped* Sobol run then
  yields one index per mechanism.
- **Log-uniform sampling.** Each parameter is drawn over `[nominal/f, nominal*f]`
  in log10 space (`config.FACTORS`: ×/÷2 for geometry, ×/÷3 for setpoints, ×/÷10
  for rates/counts). `config.RANGE_OVERRIDES` sets absolute bounds per parameter.
- **QoIs** (`evaluate.QOI_NAMES`): resting `net_drain` / `abs_drain`,
  `tau_drain`, `ip3r_open`, `ryr_open` (cheap, equilibration only); plus
  `caer_frac_end`, `t_half`, `ca_c_end` from a free integration (`--trajectory`).
- **Failure handling:** stiff/extreme draws return NaN and are mean-filled
  before SALib analysis; the failed count is reported per QoI.

## Methods (three complementary analyses)

1. **Morris screen** (`run_morris`) — cheap all-parameter elementary-effects
   ranking (`mu*`, `sigma`). Validates the grouping and picks drill-down targets.
2. **Grouped Sobol** (`run_sobol`) — variance attribution per mechanism
   (`S1`, `ST`; `ST-S1` = interaction). The centerpiece.
3. **Within-group Sobol + PRCC** (`run_within_sobol`) — drill inside the winning
   mechanism(s): vary only their parameters (each its own factor, everything
   else pinned at nominal) and report, per parameter, the **Sobol** S1/ST
   (unsigned importance) **and PRCC** (signed direction, Leung et al. 2023
   Fig. 2 style). Rendered as parameter×QoI heatmaps. `--groups` picks the
   mechanisms; `prcc.py` computes the partial rank correlations from the *same*
   samples (no extra model runs).

Sobol is the importance measure of record here: the resting balance is a
non-monotonic, interaction-heavy refill↔drain competition with CICR feedback,
where PRCC's monotonicity assumption fails. PRCC is included only as a *signed
direction hint* (which way to move a lever), to be read alongside — not instead
of — the Sobol magnitude.

## Usage

```bash
# cheap screen (~5 min): rank all parameters
conda run -n erpm python -m sensitivity_analysis.run_morris --r 20

# grouped Sobol, mechanism attribution (~46 min / 8 cores)
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024

# add free-run QoIs (caer_frac_end / t_half / ca_c_end)
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 512 --trajectory

# include SOCE as a tenth group
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024 --soce

# within-group: which RyR / setpoint / leak parameter is the drain lever
conda run -n erpm python -m sensitivity_analysis.run_within_sobol \
    --groups ryr setpoints serca_leak --n 1024

# within-group SOCE: which SOCE parameter sets the multi-second ER floor
conda run -n erpm python -m sensitivity_analysis.run_within_sobol \
    --groups soce --soce --trajectory --t-max 3.0 --n 512 \
    --out results/sensitivity/within_soce
```

Cost: resting eval ≈ 0.24 s/eval on 8 cores. Sobol evals = `N*(G+2)`
(G = #groups); Morris evals = `r*(P+1)` (P = #parameters).

Outputs (`results/sensitivity/...`): index tables (stdout), `*_indices.json`,
`samples.npz`, and bar / mu*–sigma plots.
