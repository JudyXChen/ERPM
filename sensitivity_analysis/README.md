# sensitivity_analysis

Global sensitivity analysis of the well-mixed **0D** ER-PM model
(`model.ode0d`), focused on the **resting baseline** (no stimulus): which
mechanism controls the resting ER Ca²⁺ balance and the ER-drain stability.

Gating is pre-equilibrated with Ca_c/Ca_ER clamped and the resting flux balance
is read off by reaction, so the SA measures the same quantities as the manual
one-at-a-time sweeps — only globally and grouped.

## Design

- **Grouping by mechanism.** Parameters are grouped into reaction modules
  (SERCA-uptake, SERCA-leak, RyR, IP3R, PMCA/NCX, buffers, IP3-basal, geometry,
  setpoints, SOCE). Edit `config.GROUPS` to re-cut. A *grouped* Sobol run yields
  one index per mechanism.
- **Log-uniform sampling.** Each parameter is drawn over `[nominal/f, nominal*f]`
  in log10 space (`config.FACTORS`: ×/÷2 geometry, ×/÷3 setpoints, ×/÷10
  rates/counts). `config.RANGE_OVERRIDES` sets absolute bounds per parameter.
- **QoIs** (`evaluate.QOI_NAMES`): resting `net_drain`/`abs_drain`, `tau_drain`,
  `ip3r_open`, `ryr_open`; plus `caer_frac_end`,
  `t_half`, `ca_c_end` from a free integration (`--trajectory`).
- **Failure handling:** stiff/extreme draws return NaN and are mean-filled
  before SALib analysis; the failed count is reported per QoI.

## Methods

1. **Morris screen** (`run_morris`) — cheap all-parameter elementary-effects
   ranking (`mu*`, `sigma`). 
2. **Sobol** (`run_sobol`). With no `--groups` it runs
   *grouped* Sobol (variance attribution per mechanism: `S1`, `ST`; `ST-S1` =
   interaction). Naming `--groups` switches to a *within-group* run: vary only
   those mechanisms' parameters (rest pinned at nominal) for per-parameter
   `S1`/`ST`. Both render parameter×QoI heatmaps.

## Usage

```bash
# Morris screen : rank all parameters
conda run -n erpm python -m sensitivity_analysis.run_morris --r 20

# grouped Sobol, mechanism attribution
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024

# add free-run QoIs (caer_frac_end / ca_c_end)
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 512 --trajectory

# within-group: RyR / setpoint / leak parameter
conda run -n erpm python -m sensitivity_analysis.run_sobol \
    --groups ryr setpoints serca_leak --n 1024

# within-group SOCE: 
conda run -n erpm python -m sensitivity_analysis.run_sobol \
    --groups soce --trajectory --t-max 3.0 --n 512 \
    --out results/sensitivity/within_soce
```

Outputs (`results/sensitivity/...`): index tables (stdout), `*_indices.json`,
`samples.npz`, and heatmap / bar / mu*–sigma plots.