# ERPM

## Run

```bash
conda env create -f environment.yml
conda activate erpm
python -m model.ode0d --soce --final-t 1 --out results
python scripts/plot_ode0d.py results
```