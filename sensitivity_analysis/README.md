# sensitivity_analysis

Global sensitivity analysis of the well-mixed **0D** ER-PM model
(`model.ode0d`). Two QoI families: the **resting baseline** (no stimulus) —
which mechanism controls the resting ER Ca²⁺ balance and the ER-drain
stability — and the **stimulus response** (`--stimulus`) — which mechanism
controls the single-pulse cytosolic-Ca transient and the per-pathway Ca flux.

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
- **QoIs** (`evaluate.QOI_NAMES`):
  - *resting* — `net_drain`/`abs_drain`, `tau_drain`, `ip3r_open`, `ryr_open`;
    plus `caer_frac_end`, `t_half`, `ca_c_end` from a free integration
    (`--trajectory`).
  - *stimulus response* (`--stimulus`, `STIM_QOI_NAMES`) — a single glutamate
    pulse (EPSP+bAP) from the equilibrated resting state, over a 50 ms window
    (`--stim-window`). The evoked Ca_cyto transient rises to a peak (~3 ms),
    decays, then levels off, so the features are: `ca_cyto_peak`, `ca_cyto_ttp`,
    `ca_cyto_fwhm`, `ca_cyto_tau_decay` (fit of the fall toward the plateau),
    `ca_cyto_auc`, `ca_cyto_plateau`; plus per-pathway Ca delivered to the
    cytosol — `ip3r_peak_open`, `ryr_peak_open`, `q_ipr`, `q_ryr`, `q_serca`,
    `q_soce`, and `cicr_gain` (ER-release share of total Ca influx).
- **Failure handling:** stiff/extreme draws return NaN and are mean-filled
  before SALib analysis; the failed count is reported per QoI.

## Methods

1. **Morris screen** (`run_morris`) — cheap all-parameter elementary-effects
   ranking (`mu*`, `sigma`). 
2. **Sobol** (`run_sobol`). With no `--groups` it runs
   *grouped* Sobol (variance attribution per mechanism: `S1`, `ST`). Naming
   `--groups` switches to a *within-group* run: vary only those mechanisms'
   parameters (rest pinned at nominal) for per-parameter `S1`/`ST`. Both render
   mechanism/parameter×QoI heatmaps (Sobol ST and Sobol S1).

## Usage

```bash
# Morris screen : rank all parameters
conda run -n erpm python -m sensitivity_analysis.run_morris --r 20

# grouped Sobol, resting ER-balance attribution
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024

# stimulus-response QoIs (single pulse, 50 ms window) -- the server default
conda run -n erpm python -m sensitivity_analysis.run_sobol --n 1024 --stimulus

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