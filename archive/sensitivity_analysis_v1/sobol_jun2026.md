> **Historical (June 2026).** Run on an older model; `results/sensitivity/` no longer exists and rankings need rerunning before use. Current state: [rest calibration](rest_ip3_calibration_261030.md).

# Global Sensitivity Analysis of the 0D ER–PM Model (Sobol)

**Question.** Over a multi-second simulation with no stimulus, which physiological
mechanism controls (a) the resting drain of endoplasmic reticulum (ER) calcium and
(b) where calcium settles after several seconds? This identifies which parameters
to tune to obtain a stable resting baseline.

## Abbreviations

- **Sobol method** — a variance-based global sensitivity analysis: it attributes
  the variance of an output to each input (or group of inputs).
- **First-order Sobol index (S1)** — the fraction of output variance explained by a
  mechanism *acting on its own* (its main effect).
- **Total-order Sobol index (ST)** — the fraction of output variance a mechanism is
  involved in *including all its interactions* with other mechanisms. ST ≥ S1; the
  difference **ST − S1** is the interaction share.
- **Quantity of Interest (QoI)** — a scalar output being analyzed.
- **Confidence interval (CI)** — the 95% bootstrap uncertainty on an index.
- **ER** — endoplasmic reticulum; **PM** — plasma membrane.
- **SOCE** — store-operated calcium entry (the STIM1/Orai1 module): refills the ER
  when the store depletes. **STIM1** — stromal interaction molecule 1 (the ER
  calcium sensor); **Orai1** — the plasma-membrane SOCE channel.
- **RyR** — ryanodine receptor (ER calcium-release channel).
- **IP3R** — inositol-1,4,5-trisphosphate receptor (ER calcium-release channel).
- **SERCA** — sarco/endoplasmic reticulum calcium ATPase (pumps calcium into the ER).
- **PMCA** — plasma-membrane calcium ATPase; **NCX** — sodium–calcium exchanger
  (both extrude cytosolic calcium across the plasma membrane).

## Method

- **Model.** Well-mixed 0D reduction of the ER–PM calcium model (`model.ode0d`),
  with the SOCE module enabled.
- **What varied.** 68 parameters, drawn log-uniformly over one order of magnitude
  (×/÷10 for rates and channel counts; tighter for geometry and setpoints), and
  assigned to 10 mechanism **groups** so each Sobol index refers to a mechanism
  rather than an individual rate constant.
- **Groups.** geometry, SERCA-uptake, SERCA-leak, IP3R, RyR, PMCA/NCX, buffers,
  resting-IP3 production, setpoints (resting calcium levels), SOCE.
- **Quantities of interest.**
  - `abs_drain` — magnitude of the net resting ER calcium flux at the full-ER state
    (how strongly it drains).
  - `tau_drain` — characteristic drain time, ER calcium divided by the net flux
    (how fast it drains).
  - `caer_frac_end` — fraction of ER calcium remaining after a 3 s free run.
  - `ca_c_end` — cytosolic calcium at 3 s.
- **Why Sobol (not partial-rank correlation).** The resting balance is a
  non-monotonic, interaction-heavy competition between calcium release and refill
  with calcium-induced calcium release feedback; rank-correlation methods assume
  monotonicity and would mislead. Sobol makes no such assumption and quantifies
  interactions directly.
- **Run.** Base sample N = 1024 → 12,288 model evaluations, 3 s trajectory window,
  0 failures. ~65 min on 16 cores.

## Results

Indices are total-order (ST) ranked; ± is the 95% confidence interval. The
per-mechanism first-order and total-order indices for all four quantities of
interest were shown in Figure 1 (deleted).

***Figure 1.*** *Grouped Sobol indices per mechanism, one panel per quantity of
interest. Blue = first-order index (S1, main effect); red = total-order index
(ST, including interactions). Mechanisms are ordered by ST within each panel.
Left to right: `abs_drain` and `tau_drain` (the resting drain, dominated by RyR /
setpoints / SERCA-leak) versus `caer_frac_end` and `ca_c_end` (the 3 s outcome,
dominated by SOCE). The figure was deleted with the obsolete results.*

### `abs_drain` — strength of the resting ER drain
| Mechanism | S1 | ST | Interaction (ST−S1) |
|---|---|---|---|
| RyR | 0.599 ± 0.413 | **0.857 ± 0.471** | 0.26 |
| setpoints | 0.256 ± 0.340 | **0.681 ± 0.381** | 0.43 |
| geometry | 0.074 | 0.134 | 0.06 |
| IP3R | 0.002 | 0.051 | 0.05 |
| resting-IP3 | −0.026 | 0.009 | — |

### `tau_drain` — how fast the ER drains
| Mechanism | S1 | ST | Interaction (ST−S1) |
|---|---|---|---|
| RyR | 0.126 | **0.575 ± 0.479** | 0.45 |
| SERCA-leak | 0.055 | **0.430 ± 0.369** | 0.38 |
| IP3R | 0.097 | 0.293 ± 0.204 | 0.20 |
| geometry | 0.033 | 0.145 | 0.11 |
| setpoints | −0.006 | 0.073 | 0.08 |

### `caer_frac_end` — ER calcium remaining after 3 s
| Mechanism | S1 | ST | Interaction (ST−S1) |
|---|---|---|---|
| SOCE | 0.037 | **0.941 ± 0.332** | 0.90 |
| setpoints | −0.007 | 0.257 | 0.27 |
| buffers | −0.011 | 0.230 ± 0.062 | 0.24 |
| PMCA/NCX | −0.003 | 0.062 | 0.07 |
| RyR | −0.003 | 0.019 | 0.02 |

### `ca_c_end` — cytosolic calcium after 3 s
| Mechanism | S1 | ST | Interaction (ST−S1) |
|---|---|---|---|
| SOCE | 0.265 | **0.800 ± 0.718** | 0.54 |
| buffers | 0.004 | 0.229 ± 0.226 | 0.23 |
| PMCA/NCX | 0.005 | 0.107 | 0.10 |
| RyR | −0.001 | 0.028 | 0.03 |
| geometry | −0.001 | 0.016 | 0.02 |

## Interpretation

**The system splits into two regimes** (visible as the left-pair vs. right-pair of
panels in **Figure 1**).

1. **What drains the ER** (`abs_drain`, `tau_drain`) is governed by **RyR**, the
   **resting setpoints**, **SERCA-leak**, and **IP3R**. RyR is the single largest
   driver of the drain magnitude (ST = 0.86), consistent with the direct flux
   decomposition, where RyR is the biggest resting drain term. The resting
   setpoints are a close second (ST = 0.68) and act mostly through interactions:
   the resting cytosolic calcium level sets the channel open probabilities, so it
   modulates both RyR and IP3R release indirectly. SERCA-leak controls the
   *timescale* of the drain (ST = 0.43 on `tau_drain`).

2. **Where calcium settles after seconds** (`caer_frac_end`, `ca_c_end`) is
   governed almost entirely by **SOCE**, with cytosolic calcium shaped by
   **buffers** and **PMCA/NCX**.

**SOCE is a pure rescue mechanism — confirmed, not assumed.** SOCE has a
total-order index of **0.000** on both drain QoIs (it is inactive while the ER is
full, because STIM1 only engages once the store depletes) yet **0.94** on the 3 s
ER-survival QoI. Its effect is almost entirely interaction (first-order 0.04,
total-order 0.94): the refill it delivers depends on how far the ER drained (set by
RyR/IP3R) and on what the plasma-membrane extrusion and buffering leave in the
cytosol. SOCE therefore does not stop the drain — it sets the floor the ER settles
to over seconds. This is the clearest validation of the SOCE module's role and only
appears with SOCE enabled and a multi-second window.

**Why IP3R looks small in Sobol despite being a large drain term.** Sobol measures
variance *over the sampled parameter range*, not magnitude at the nominal point.
The resting IP3R open fraction is governed by cytosolic calcium and IP3 (the
setpoints and resting-IP3 groups), not by its own binding-rate constants within
their sampled range — so the variance IP3R "spends" is carried by the setpoints
group, which is why setpoints rank so high.

## Levers to stabilize the multi-second ER baseline

Ranked by the analysis:

1. **RyR** — the largest driver of the resting drain. Reducing the resting RyR
   open probability attacks the drain at its biggest source.
2. **Setpoints** — the resting calcium levels strongly modulate the drain through
   channel gating; confirm they are pinned to the intended physiological values.
3. **SERCA-leak (`k_S_leak`)** — sets the drain timescale; calibrate so pump and
   leak balance at the resting ER calcium level.
4. **SOCE** — will not stop the drain but sets the steady-state floor; a
   sufficiently strong SOCE can hold the ER at a depleted-but-stable nonzero level
   rather than letting it empty.

## Caveats

- **Wide confidence intervals.** For example, RyR on `abs_drain` is 0.86 ± 0.47.
  The qualitative conclusions are robust — the SOCE gaps are very large, and RyR
  and setpoints are clearly the top two drain drivers — but RyR and setpoints
  **cannot be cleanly ranked against each other** (their intervals overlap). The
  wide intervals are expected: the QoIs are heavy-tailed because parameters span
  orders of magnitude, and the system is strongly interaction-driven (total-order
  indices exceed first-order almost everywhere).
- **Indices are mechanism-level.** "RyR" and "SOCE" are groups; identifying the
  specific responsible parameter requires a within-group analysis (treating each
  parameter of the winning mechanism as its own factor).

## Next steps

1. **Within-group Sobol on RyR** (and setpoints) — find which individual RyR
   parameter controls the resting drain.
2. **Within-group Sobol on SOCE** — find which SOCE parameter sets the
   multi-second floor.
3. **Direct tuning experiment** — lower the resting RyR open probability (or adjust
   the setpoints), integrate to 3 s, and check whether the ER baseline holds.

*Data: `results/sensitivity/grouped_sobol/` — `sobol_indices.json` (indices and
intervals), `samples.npz` (raw inputs/outputs), `grouped_sobol.png` (bar charts).*
