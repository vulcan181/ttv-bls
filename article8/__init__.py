"""
Article 8: Photometric Mass Estimation via TTV Harmonic Analysis

This package provides tools for estimating companion masses from TTV time series
using the Deck & Agol (2015) chopping formula and Lithwick et al. (2012)
near-resonant TTV formula.
"""

from .laplace_coefficients import (
    laplace_coefficient,
    chopping_coefficient_A1,
    get_alpha_from_period_ratio,
    get_lithwick_coefficients
)

from .ttv_models import (
    compute_resonance_parameter,
    compute_super_period,
    compute_synodic_period,
    lithwick_ttv_amplitude,
    chopping_ttv_amplitude,
    generate_lithwick_ttv,
    generate_chopping_ttv,
    generate_combined_ttv,
    mass_from_chopping_amplitude,
    mass_from_lithwick_amplitude
)

from .harmonic_decomposition import (
    lomb_scargle_periodogram,
    find_significant_peaks,
    fit_sinusoid,
    extract_ttv_harmonics
)

__version__ = "0.1.0"
__author__ = "Stamatis Kalogerakos"
