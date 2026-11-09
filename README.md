# ERPM

```bash
conda env create -f environment.yml
conda activate erpm
```

## Run the model

```bash
python -m model.ode0d --soce --eqm --final-t 60 --out results/ode0d
python scripts/plot_ode0d.py results/ode0d
```

This runs one condition (RyR on, SOCE on) and matches the same condition in
`run_conditions.py` (Ca_c peak 8.2 µM, ER minimum 20.1 µM).

- **Use `--eqm`.** The default initial state is not a steady state. Without it,
  `Ca_ER` falls from 150 to about 110 µM over 8 s with no stimulus, and that drift
  hides the real response.
- **Output times are chosen automatically**, using the same grid as
  `run_conditions.py`: every 10 µs for 100 ms after each release (resolving the
  cytosolic peak and the ~50 ms ER minimum), then 2 ms to 10 s, 20 ms to 60 s and
  0.1 s after. A 60 s run writes ~17,500 rows (~58 MB).

To regenerate the full four-condition result set, including the
figures:

```bash
python scripts/run_conditions.py --tag sercaBartol_soceFast
```

This reruns and overwrites that folder, which holds the current defaults. Use a new
`--tag` to keep the saved run. The other folders in `results/stimulus/` used earlier
SERCA or SOCE parameters; `results/README.md` lists what each one used.

Rest differs slightly by condition: Ca_c 100 nM, Ca_ER 150 µM and IP3 50 nM with
SOCE off; Ca_ER 155.5 µM and Ca_c 108.6 nM with SOCE on.

### `model/ode0d.py` flags

| Flag | Default | Meaning |
|---|---|---|
| `--glu-pulses N` | `1` | glutamate boluses: `0` none, `1` single, `N` a train |
| `--glu-freq HZ` | `1.0` | release frequency (Hz) within a pulse train |
| `--glu-molecules N` | `500` | molecules per bolus (→ spine concentration) |
| `--t-stim S` | `0.0` | time (s) of the first glutamate release |
| `--clamp MV` | *(off)* | hold V_m at a constant clamp instead of the transient |
| `--soce [on\|off]` | `off` | STIM1/Orai1 SOCE; bare `--soce` means on |
| `--ryr [on\|off]` | `on` | RyR block; `--ryr off` removes its states and CICR flux |
| `--plc {old,new}` | `new` | select the Gq or legacy PLC pathway |
| `--eqm` | off | equilibrate at zero stimulus before the run |
| `--final-t S` | `0.5` | integration length (s) |
| `--out DIR` | `results/ode0d` | writes `traces.csv` here |
| `--set KEY=VAL` | — | override any model parameter, repeatable (e.g. `--set N_Orai1=50`) |

The solver is fixed to `BDF`. Glutamate delivery is a single train of
`--glu-pulses` releases at `--glu-freq`, starting at `--t-stim`. V_rest is fixed at
−65 mV (`model.stimulus.V_REST`); each release adds an EPSP+bAP on top. IP3R gating
uses one 8-state subunit with P_open = `IPR_110`³.

## Scripts

| Script | Use it to | Output |
|---|---|---|
| `run_conditions.py` | Run the stimulus in all four RyR/SOCE conditions, or redraw them with `--plot-only` | `results/stimulus/TAG`, `results/Flux/TAG` |
| `plot_ode0d.py` | Plot one `traces.csv` from `model.ode0d` | PNGs next to the CSV |
| `er_flux_decomp.py` | Plot one ER and cytosol calcium budget | `results/Flux/standalone` |
| `calibrate_rest.py` | Refit the resting state after changing sources, channels or geometry | `results/rest_calibration/summary.json` |
| `rest_diagnostics.py` | Test refill, SERCA, IP3, leak, buffer and VSCC assumptions at rest | `results/rest_calibration` |
| `plc_gq_analysis.py` | Validate the mGluR-Gq-PLC pathway | `results/plc_gq`, `results/Flux/plc_gq_validation`, `doc/PLC_Gq_validation_report.md` |
| `ryr_conductance.py` | Compute steady-state RyR open probability against clamped calcium | `results/ryr_conductance` |
| `replot_results.py` | Redraw every supported figure from saved data, without simulating | PNGs in place, `results/plot_manifest.json` |
| `plot_style.py` | Shared plotting style and export (imported only) | — |
| `io_utils.py` | Shared CSV helpers (imported only) | — |

### `run_conditions.py`

```bash
python scripts/run_conditions.py --tag my_run
python scripts/run_conditions.py --tag my_run --plot-only
python scripts/run_conditions.py --calibrate
python scripts/run_conditions.py --tag sercaBartol_soceFast_6pulses_0.2Hz --glu-pulses 6 --glu-freq 0.2
```

| Flag | Meaning |
|---|---|
| `--tag TAG` | output folder name (default `ryr_soce_repair`) |
| `--k-stim-um X` | override `K_STIM1` in µM |
| `--glu-pulses N` / `--glu-freq HZ` | release train (default one release) |
| `--plot-only` | redraw from saved CSVs |
| `--calibrate` | print the ER leak that gives 150 µM at rest |

`summary.json` records the protocol, rest, the ER minimum, ER before and after each
release, and the time to 50, 90, 95 and 99% recovery from the minimum after the last
release. Each release is 500 glutamate molecules plus an EPSP+bAP, the first at t = 0.
Runs simulate 300 s and stop with an error if the flux budget does not close. Trains
also write `ca_cyto_<T>s.png`, covering the whole train. Figure names are listed in
`results/README.md`.

### `plot_ode0d.py`

`python scripts/plot_ode0d.py [out_dir] [flags]` writes `ca_cyto.png`,
`ca_cyto_norm.png`, `ca_er.png`, `stimulus_glu.png`, `stimulus_voltage.png` and
`open_channels_summary.csv`.

| Flag | Default | Meaning |
|---|---|---|
| `out_dir` (positional) | `results/ode0d` | directory holding `traces.csv` |
| `--t-min S` / `--t-max S` | `0` / end of run | plotted time window, in seconds |

The time axis is in ms for windows shorter than 1 s and in s otherwise. All figures
follow this rule (`plot_style.time_unit`): fast cytosolic panels (100 ms) in ms, ER,
receptor and IP3 panels (seconds) in s.

### `er_flux_decomp.py`

Plots cumulative ER and cytosolic calcium budgets as separate panels. `--out`
supplies the filename prefix, and numerical flux data use the same prefix as a CSV.

| Flag | Default | Meaning |
|---|---|---|
| `--final-t S` | `2.0` | integration length; sampling is automatic, as in `model.ode0d` |
| `--glu-molecules` | `0.0` | single bolus at t=0 (`0` = no stimulus) |
| `--stim` | off | drive with the EPSP+bAP voltage transient |
| `--plc {old,new}` | `new` | choose the PLC pathway |
| `--eqm` | off | equilibrate the resting state before the run |
| `--soce [on\|off]` / `--ryr [on\|off]` / `--ip3r [on\|off]` | `off` / `on` / `on` | include or remove each block |
| `--out`, `--data-out` | `results/Flux/standalone/budget.png` | output paths |

### Resting calibration

Refit after changing resting sources, channels or geometry, then copy the fitted
values into `model/parameters.py`. The R15 scale (`plc_gq_r15_scale`) is derived from
`IP3_basal` and `plc_gq_k16`, so resting IP3 follows `IP3_basal`. Changing resting IP3
still shifts ER calcium, so refit `k_S_leak_gq` afterwards.

```bash
python scripts/calibrate_rest.py
python scripts/rest_diagnostics.py                # refill, SERCA, stimulus, IP3 directions
python scripts/rest_diagnostics.py --leak-buffer  # leak, buffer and k_d changes
python scripts/rest_diagnostics.py --vscc-sweep   # VSCC counts, optionally: --vscc-sweep 5 7 14
```

### mGluR-Gq-PLC pathway

The default `new` pathway keeps the original mGluR binding kinetics and couples
receptor occupancy to the Calizo/Cooling Gq-PLC reactions. Basal R15 production
balances R16 decay. The legacy pathway remains available with `--plc old`, where
`IP3_basal` is balanced by `IP3_basal_source` and `IP3_decay`.

```bash
python -m model.ode0d --eqm --ryr off --final-t 2
python scripts/plc_gq_analysis.py
python scripts/er_flux_decomp.py --eqm --ryr off
```

`python scripts/plc_gq_analysis.py --ip3-budget` traces IP3 production (R14, R15) against
decay, compares parameter sets and tests the synapse-spike pairing window. The sets are
applied in memory; `model/parameters.py` is not changed. Output goes to
`results/plc_gq/ip3_budget`.

## Plotting

All plotters use [scripts/plot_style.py](scripts/plot_style.py): 3 × 3 inch panels,
300 dpi, 9 pt Arial/sans-serif, and a shared palette. Viewing plots include legends.
Clean panels and transparent legends are saved in each result folder's `fig/`.

```bash
python scripts/replot_results.py
```

This reads saved CSV and JSON files only and checks they are unchanged. It rewrites
PNGs in place, so `--results DIR` lets you try it on a copy. Add
`--archive-superseded` to move replaced figures into `results/_legacy_plots/`.

## Tests

```bash
python -m unittest discover -s tests
```

| File | Covers |
|---|---|
| `tests/test_model.py` | rest calibration, flux balance, solver restarts, RyR gating, Gq-PLC rates and conservation, legacy PLC regression |
| `tests/test_plot_style.py` | panel size, legend export and colorbar handling |

## Notes

These live in `doc/`, which is gitignored, so the links only work locally.

- [ER refill compared with published data](doc/er_refill_literature_261031.md)
- [IP3 cascade compared with neuronal data](doc/ip3_cascade_review_261101.md)
- [Rest, SERCA and IP3 calibration](doc/rest_ip3_calibration_261030.md)
- [Kinetic assumptions](doc/model_review_261029.md)
- [Plotting rules](doc/PLOTTING.md)
- [Reaction scheme](doc/reactions.tex); implemented rates are in `model/parameters.py`
- Removed scripts are kept in `doc/retired_scripts/`

The former SERCA calibration scripts and `sensitivity_analysis.run_ryr_ip3r` no
longer exist, so their historical outputs cannot be regenerated.

Older notes may name these scripts:

| Old script | Now |
|---|---|
| `validate_er_repair.py` | renamed `run_conditions.py` |
| `leak_buffer_diagnosis.py` | `rest_diagnostics.py --leak-buffer` |
| `vscc_calibration.py` | `rest_diagnostics.py --vscc-sweep` |
| `audit_er_review.py` | recovery times now in `run_conditions.py`; removed |
| `ip3r_resting_analysis.py`, `ip3r_activation_audit.py` and their plotters | removed; targeted the old 20 nM IP3 rest and legacy pathway |
| `steady_state_mass_balance.py` | removed; covered by `calibrate_rest.py` and `tests/test_model.py` |
| `compare_plc_ip3r.py`, `plot_short_timescale.py` | removed; one-off comparisons whose outputs no longer exist |
