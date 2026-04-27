"""
TTV-BLS: Transit Timing Variation-aware Box Least Squares

A Python package for detecting transiting exoplanets in the presence of
Transit Timing Variations (TTVs).

Based on: Kalogerakos & West (2026), "Robustness of Transit Detection to
Small-Amplitude Transit Timing Variations: A TTV-Aware BLS Study"

Main components:
- core: TTV-BLS algorithm implementation
- lightcurve: Synthetic light curve generation with TTVs
- utils: Utility functions for analysis and I/O
"""

from .core import (
    distort_timebase,
    transit_search,
    ttv_bls_search,
    ttv_grid_search,
    optimal_sample_periods,
    compute_snr_ratio,
)

from .lightcurve import (
    create_lightcurve,
    compute_transit_template,
    get_transit_duration,
    compute_a_ttv_over_t14,
    set_random_seed,
)

from .utils import (
    setup_logging,
    save_results,
    load_results,
    load_config,
    compute_sde_statistics,
    compute_improvement_statistics,
    find_critical_threshold,
)

__version__ = "1.0.0"
__author__ = "S. Kalogerakos"

# Critical threshold constant (A_TTV / T_14)
CRITICAL_THRESHOLD = 0.5  # Lower bound
CRITICAL_THRESHOLD_UPPER = 0.7  # Upper bound
