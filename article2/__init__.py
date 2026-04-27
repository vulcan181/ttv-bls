"""
TTV Correction Module for Article 2

This module provides tools for correcting Transit Timing Variations (TTVs)
in exoplanet light curves, extending the methods from Article 1.

Modules:
    sinusoidal: Basic sinusoidal TTV correction (from Article 1)
    chopping: Synodic chopping correction for near-resonance systems
    composite: Combined sinusoidal + chopping correction
    blind_search: Grid search optimization for unknown TTV parameters
"""

from .sinusoidal import sinusoidal_correction, distort_timebase
from .chopping import chopping_correction, compute_synodic_period
from .composite import composite_ttv_correction
from .blind_search import blind_ttv_search, adaptive_grid_search

__all__ = [
    'sinusoidal_correction',
    'distort_timebase',
    'chopping_correction',
    'compute_synodic_period',
    'composite_ttv_correction',
    'blind_ttv_search',
    'adaptive_grid_search',
]
