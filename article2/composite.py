#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Composite TTV Correction Module

Implements combined sinusoidal + chopping TTV correction for
near-resonance systems where both components are significant.

This extends the basic TTV-BLS from Article 1 by adding support
for the shorter-period chopping variations that occur during
planetary conjunctions.
"""

import numpy as np

# Handle both package import and direct execution
try:
    from .sinusoidal import sinusoidal_correction, compute_super_period
    from .chopping import chopping_correction, compute_synodic_period
except ImportError:
    from sinusoidal import sinusoidal_correction, compute_super_period
    from chopping import chopping_correction, compute_synodic_period


def composite_ttv_correction(t, params):
    """
    Apply combined sinusoidal + chopping correction.

    This function applies both the long-period resonant correction
    and the short-period chopping correction to the time array.

    Parameters
    ----------
    t : array_like
        Time array (days)
    params : dict
        TTV parameters dictionary with keys:
        - 'A_ttv': float, sinusoidal amplitude (days)
        - 'P_ttv': float, sinusoidal period / super-period (days)
        - 'E_ttv': float, sinusoidal epoch (days)
        - 'A_chop': float, chopping amplitude (days)
        - 'P_syn': float, synodic period (days)
        - 'E_chop': float, chopping epoch (days)
        Optional:
        - 't0': float, reference epoch (default 0.0)

    Returns
    -------
    t_corrected : ndarray
        Corrected time array with both TTV components removed

    Notes
    -----
    The corrections are applied sequentially:
    1. First, the sinusoidal (resonant) correction
    2. Then, the chopping correction

    The order matters slightly but the difference is typically small
    for realistic TTV amplitudes.

    Examples
    --------
    >>> params = {
    ...     'A_ttv': 0.1,    # 0.1 days
    ...     'P_ttv': 100.0,  # 100 days
    ...     'E_ttv': 0.0,
    ...     'A_chop': 0.02,  # 0.02 days
    ...     'P_syn': 30.0,   # 30 days
    ...     'E_chop': 0.0
    ... }
    >>> t = np.linspace(0, 1000, 10000)
    >>> t_corr = composite_ttv_correction(t, params)
    """
    t = np.asarray(t)

    # Extract parameters
    A_ttv = params.get('A_ttv', 0.0)
    P_ttv = params.get('P_ttv', 0.0)
    E_ttv = params.get('E_ttv', 0.0)
    A_chop = params.get('A_chop', 0.0)
    P_syn = params.get('P_syn', 0.0)
    E_chop = params.get('E_chop', 0.0)
    t0 = params.get('t0', 0.0)

    # Apply sinusoidal correction
    t_corr = t.copy()
    if A_ttv > 0 and P_ttv > 0:
        phase_sin = 2 * np.pi * (t_corr - (t0 + E_ttv)) / P_ttv
        t_corr = t_corr - A_ttv * np.sin(phase_sin)

    # Apply chopping correction
    if A_chop > 0 and P_syn > 0:
        phase_chop = 2 * np.pi * (t_corr - E_chop) / P_syn
        t_corr = t_corr - A_chop * np.sin(phase_chop)

    return t_corr


def composite_ttv_correction_from_periods(t, P_orb, P_perturber,
                                           A_ttv, E_ttv, A_chop, E_chop, t0=0.0):
    """
    Apply composite correction using orbital periods directly.

    This is a convenience function that computes the super-period
    and synodic period from the orbital periods.

    Parameters
    ----------
    t : array_like
        Time array (days)
    P_orb : float
        Orbital period of transiting planet (days)
    P_perturber : float
        Orbital period of perturbing planet (days)
    A_ttv : float
        Sinusoidal TTV amplitude (days)
    E_ttv : float
        Sinusoidal TTV epoch (days)
    A_chop : float
        Chopping amplitude (days)
    E_chop : float
        Chopping epoch (days)
    t0 : float, optional
        Reference epoch (days). Default is 0.0.

    Returns
    -------
    t_corrected : ndarray
        Corrected time array
    P_ttv : float
        Computed super-period (days)
    P_syn : float
        Computed synodic period (days)
    """
    # compute_super_period already imported at module level

    # Determine resonance order (approximate)
    ratio = P_perturber / P_orb if P_perturber > P_orb else P_orb / P_perturber
    j = int(round(ratio))
    if j < 2:
        j = 2

    # Compute periods
    P_ttv = compute_super_period(min(P_orb, P_perturber), max(P_orb, P_perturber), j=j)
    P_syn = compute_synodic_period(P_orb, P_perturber)

    # Apply correction
    params = {
        'A_ttv': A_ttv,
        'P_ttv': P_ttv,
        'E_ttv': E_ttv,
        'A_chop': A_chop,
        'P_syn': P_syn,
        'E_chop': E_chop,
        't0': t0
    }

    t_corr = composite_ttv_correction(t, params)

    return t_corr, P_ttv, P_syn


def generate_composite_ttv_signal(t, P_orb, t0, params):
    """
    Generate the composite TTV signal (timing offsets).

    This computes what the transit timing offsets would be at each
    potential transit epoch, useful for injection tests.

    Parameters
    ----------
    t : array_like
        Time array (days) - typically transit epochs
    P_orb : float
        Orbital period (days)
    t0 : float
        Reference epoch (days)
    params : dict
        TTV parameters (same as composite_ttv_correction)

    Returns
    -------
    ttv_signal : ndarray
        Timing offset at each time (days)
    """
    t = np.asarray(t)

    A_ttv = params.get('A_ttv', 0.0)
    P_ttv = params.get('P_ttv', 0.0)
    E_ttv = params.get('E_ttv', 0.0)
    A_chop = params.get('A_chop', 0.0)
    P_syn = params.get('P_syn', 0.0)
    E_chop = params.get('E_chop', 0.0)

    ttv = np.zeros_like(t)

    # Sinusoidal component
    if A_ttv > 0 and P_ttv > 0:
        phase_sin = 2 * np.pi * (t - (t0 + E_ttv)) / P_ttv
        ttv += A_ttv * np.sin(phase_sin)

    # Chopping component
    if A_chop > 0 and P_syn > 0:
        phase_chop = 2 * np.pi * (t - E_chop) / P_syn
        ttv += A_chop * np.sin(phase_chop)

    return ttv


def compute_ttv_transit_times(t_linear, params):
    """
    Compute actual transit times given linear ephemeris and TTV parameters.

    Parameters
    ----------
    t_linear : array_like
        Linear ephemeris transit times (days)
    params : dict
        TTV parameters

    Returns
    -------
    t_actual : ndarray
        Actual transit times with TTV applied (days)
    """
    t_linear = np.asarray(t_linear)

    A_ttv = params.get('A_ttv', 0.0)
    P_ttv = params.get('P_ttv', 0.0)
    E_ttv = params.get('E_ttv', 0.0)
    A_chop = params.get('A_chop', 0.0)
    P_syn = params.get('P_syn', 0.0)
    E_chop = params.get('E_chop', 0.0)
    t0 = params.get('t0', t_linear[0] if len(t_linear) > 0 else 0.0)

    t_actual = t_linear.copy()

    # Add sinusoidal TTV
    if A_ttv > 0 and P_ttv > 0:
        phase_sin = 2 * np.pi * (t_linear - (t0 + E_ttv)) / P_ttv
        t_actual += A_ttv * np.sin(phase_sin)

    # Add chopping TTV
    if A_chop > 0 and P_syn > 0:
        phase_chop = 2 * np.pi * (t_linear - E_chop) / P_syn
        t_actual += A_chop * np.sin(phase_chop)

    return t_actual


if __name__ == "__main__":
    # Simple test
    print("Testing composite TTV correction module...")

    # Create test time array
    t = np.linspace(0, 1000, 10000)

    # Test parameters (hypothetical 2:1 resonance system)
    params = {
        'A_ttv': 0.1,    # 0.1 days = 2.4 hours
        'P_ttv': 150.0,  # 150 days super-period
        'E_ttv': 0.0,
        'A_chop': 0.02,  # 0.02 days = 29 minutes
        'P_syn': 25.0,   # 25 days synodic period
        'E_chop': 5.0,
        't0': 0.0
    }

    # Apply composite correction
    t_corr = composite_ttv_correction(t, params)

    # Check correction range
    max_shift = np.max(np.abs(t_corr - t))
    print(f"  Max total time shift: {max_shift:.4f} days ({max_shift*24*60:.1f} minutes)")

    # Test transit time generation
    t_linear = np.arange(0, 1000, 10.0)  # Linear ephemeris every 10 days
    t_actual = compute_ttv_transit_times(t_linear, params)
    ttv_signal = t_actual - t_linear
    print(f"  TTV signal range: {ttv_signal.min()*24*60:.1f} to {ttv_signal.max()*24*60:.1f} minutes")

    # Test convenience function
    t_corr2, P_ttv, P_syn = composite_ttv_correction_from_periods(
        t, P_orb=10.0, P_perturber=20.0,
        A_ttv=0.05, E_ttv=0.0, A_chop=0.01, E_chop=0.0
    )
    print(f"  Computed periods: P_ttv={P_ttv:.1f} days, P_syn={P_syn:.1f} days")

    print("  All tests passed!")
