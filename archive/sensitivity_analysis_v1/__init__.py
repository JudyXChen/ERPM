"""Global sensitivity analysis for the well-mixed 0D ER-PM model (SOCE always on).

    python -m sensitivity_analysis.run_morris    # all-parameter Morris screen
    python -m sensitivity_analysis.run_sobol     # grouped / within-group Sobol

Edit config.GROUPS to re-cut the parameter grouping.
"""

from .config import GROUPS, FACTORS
from .evaluate import QOI_NAMES, evaluate

__all__ = ["GROUPS", "FACTORS", "QOI_NAMES", "evaluate"]
