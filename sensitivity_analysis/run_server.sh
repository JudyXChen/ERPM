#!/usr/bin/env bash
#
#   conda env create -f environment.yml   # one time
#   conda activate erpm
#   bash sensitivity_analysis/run_server.sh
#
# Tunables via environment variables (defaults in brackets):
#   NJOBS   [-1]     parallel workers (-1 = all cores)
#   N       [1024]   Sobol base sample N  (evals = N*(K+2))
#   STIMWIN [0.050]  stimulus response window (s) after the glutamate pulse
#   TMAX    [3.0]    trajectory free-integration window (s) (morris/trajectory)
#   STAGE   [sobol]  'smoke', 'sobol', 'morris', 'within', or 'both'
#                    ('smoke' = tiny N=4 end-to-end check, ~30 s, then exit)
#                    ('within' = per-parameter Sobol inside WGROUPS)
#   WGROUPS ["ryr setpoints serca_leak"]  groups for STAGE=within
#   OUT     [results/sensitivity]  output root
# QoIs are the single-pulse stimulus response (Ca_cyto transient + per-pathway
# Ca flux); SOCE is always enabled.

set -euo pipefail

NJOBS="${NJOBS:--1}"
N="${N:-1024}"
STIMWIN="${STIMWIN:-0.050}"
TMAX="${TMAX:-3.0}"
STAGE="${STAGE:-sobol}"
WGROUPS="${WGROUPS:-ryr setpoints serca_leak}"
OUT="${OUT:-results/sensitivity}"

# Run from the repo root so `python -m sensitivity_analysis...` resolves.
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "host=$(hostname)  cores=$(nproc 2>/dev/null || echo '?')  njobs=$NJOBS"
echo "stage=$STAGE  N=$N  stim_window=$STIMWIN  out=$OUT"
echo "----------------------------------------------------------------"

if [ "$STAGE" = "smoke" ]; then
  echo "[smoke] tiny end-to-end check (N=4): env + parallelism + SALib + stimulus"
  "$PY" -m sensitivity_analysis.run_sobol \
      --n 4 --stimulus --stim-window "$STIMWIN" \
      --n-jobs "$NJOBS" --out "$OUT/_smoke"
  rm -rf "$OUT/_smoke"
  echo "smoke PASS -> environment is good; run STAGE=sobol (or both) for the real thing"
  exit 0
fi

if [ "$STAGE" = "morris" ] || [ "$STAGE" = "both" ]; then
  echo "[morris] global screen (all parameters)"
  "$PY" -m sensitivity_analysis.run_morris \
      --r 20 --stimulus --stim-window "$STIMWIN" \
      --n-jobs "$NJOBS" --out "$OUT/morris"
fi

if [ "$STAGE" = "sobol" ] || [ "$STAGE" = "both" ]; then
  echo "[sobol] grouped Sobol (stimulus-response mechanism attribution)"
  "$PY" -m sensitivity_analysis.run_sobol \
      --n "$N" --stimulus --stim-window "$STIMWIN" \
      --n-jobs "$NJOBS" --out "$OUT/grouped_sobol"
fi

if [ "$STAGE" = "within" ]; then
  WTAG="$(echo "$WGROUPS" | tr ' ' '_')"
  echo "[within] per-parameter Sobol over: $WGROUPS"
  "$PY" -m sensitivity_analysis.run_sobol \
      --groups $WGROUPS --n "$N" --stimulus --stim-window "$STIMWIN" \
      --n-jobs "$NJOBS" --out "$OUT/within_${WTAG}"
fi

echo "----------------------------------------------------------------"
echo "done -> $OUT/"
