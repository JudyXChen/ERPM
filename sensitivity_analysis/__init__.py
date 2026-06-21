"""Global sensitivity analysis for the well-mixed 0D ER-PM model.

Entry points:
    python -m sensitivity_analysis.run_morris   # cheap all-parameter screen
    python -m sensitivity_analysis.run_sobol    # grouped Sobol (mechanism)

See config.GROUPS to re-cut the parameter grouping.
"""

from .config import GROUPS, FACTORS, active_groups
from .evaluate import QOI_NAMES, evaluate

__all__ = ["GROUPS", "FACTORS", "active_groups", "QOI_NAMES", "evaluate"]
