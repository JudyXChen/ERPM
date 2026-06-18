#!/usr/bin/env bash
# Standard 0D runs + plots.
#   glut       : single 500-molecule bolus, plotted 0-100 ms
#   glut_50hz  : 50 Hz x6 burst (1 uM/pulse, 100 ms), plotted 0-300 ms
# Toggle SOCE by adding/removing the --soce flag.
set -e

# 1. glut, SOCE on
conda run -n erpm python -m model.ode0d --soce --glu-pulses 1 --glu-molecules 500 \
    --final-t 0.1 --n-points 1000 --out results/glut_SOCE20
conda run -n erpm python scripts/plot_ode0d.py results/glut_SOCE20 --t-min-ms 0 --t-max-ms 100

# 2. glut, SOCE off
conda run -n erpm python -m model.ode0d --glu-pulses 1 --glu-molecules 500 \
    --final-t 0.1 --n-points 1000 --out results/glut_SOCEoff
conda run -n erpm python scripts/plot_ode0d.py results/glut_SOCEoff --t-min-ms 0 --t-max-ms 100

# 3. glut_10hz, SOCE on
conda run -n erpm python -m model.ode0d --soce --glu-pulses 5 --glu-freq 10 --glu-conc 1e-6 \
    --final-t 1 --n-points 5000 --out results/glut_10hz_SOCE20
conda run -n erpm python scripts/plot_ode0d.py results/glut_10hz_SOCE20 --t-min-ms 0 --t-max-ms 1000

# 4. glut_10hz, SOCE off
conda run -n erpm python -m model.ode0d --glu-pulses 5 --glu-freq 10 --glu-conc 1e-6 \
    --final-t 1 --n-points 5000 --out results/glut_10hz_SOCEoff
conda run -n erpm python scripts/plot_ode0d.py results/glut_10hz_SOCEoff --t-min-ms 0 --t-max-ms 1000