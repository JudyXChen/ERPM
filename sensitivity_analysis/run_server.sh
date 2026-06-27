#!/usr/bin/env bash
#
#   conda env create -f environment.yml   # one time
#   conda activate erpm
#   bash sensitivity_analysis/run_server.sh
#
# Tunables via environment variables (defaults in brackets):
#   NJOBS   [8]      parallel workers (-1 = all cores)
#   N       [1024]   Sobol base sample N  (evals = N*(K+2))
#   TMAX    [3.0]    trajectory free-integration window (s)
#   STAGE   [sobol]  'smoke', 'sobol', 'morris', 'within', or 'both'
#                    ('smoke' = tiny N=4 end-to-end check, ~30 s, then exit)
#                    ('within' = per-parameter Sobol inside WGROUPS)
#   WGROUPS ["ryr setpoints serca_leak"]  groups for STAGE=within
#   OUT     [results/sensitivity]  output root
# SOCE is always enabled.

set -euo pipefail

NJOBS="${NJOBS:-8}"
N="${N:-1024}"
TMAX="${TMAX:-3.0}"
STAGE="${STAGE:-sobol}"
WGROUPS="${WGROUPS:-ryr setpoints serca_leak}"
OUT="${OUT:-results/sensitivity}"

# Run from the repo root so `python -m sensitivity_analysis...` resolves.
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "host=$(hostname)  cores=$(nproc 2>/dev/null || echo '?')  njobs=$NJOBS"
echo "stage=$STAGE  N=$N  t_max=$TMAX  out=$OUT"
echo "----------------------------------------------------------------"

if [ "$STAGE" = "smoke" ]; then
  echo "[smoke] tiny end-to-end check (N=4): env + parallelism + SALib + trajectory"
  "$PY" -m sensitivity_analysis.run_sobol \
      --n 4 --trajectory --t-max "$TMAX" \
      --n-jobs "$NJOBS" --out "$OUT/_smoke"
  rm -rf "$OUT/_smoke"
  echo "smoke PASS -> environment is good; run STAGE=sobol (or both) for the real thing"
  exit 0
fi

if [ "$STAGE" = "morris" ] || [ "$STAGE" = "both" ]; then
  echo "[morris] global screen (all parameters)"
  "$PY" -m sensitivity_analysis.run_morris \
      --r 20 --trajectory --t-max "$TMAX" \
      --n-jobs "$NJOBS" --out "$OUT/morris"
fi

if [ "$STAGE" = "sobol" ] || [ "$STAGE" = "both" ]; then
  echo "[sobol] grouped Sobol (mechanism attribution)"
  "$PY" -m sensitivity_analysis.run_sobol \
      --n "$N" --trajectory --t-max "$TMAX" \
      --n-jobs "$NJOBS" --out "$OUT/grouped_sobol"
fi

if [ "$STAGE" = "within" ]; then
  WTAG="$(echo "$WGROUPS" | tr ' ' '_')"
  echo "[within] per-parameter Sobol over: $WGROUPS"
  "$PY" -m sensitivity_analysis.run_sobol \
      --groups $WGROUPS --n "$N" --trajectory --t-max "$TMAX" \
      --n-jobs "$NJOBS" --out "$OUT/within_${WTAG}"
fi

echo "----------------------------------------------------------------"
echo "done -> $OUT/"
