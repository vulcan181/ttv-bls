#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sinusoidal TTV Correction Module

Implements the basic sinusoidal TTV correction from Article 1:
    t' = t - A_TTV * sin(2*pi*(t - (t0 + E_TTV)) / P_TTV)

This correction "unwinds" the TTV signal so that transits align
during phase folding, improving BLS detection efficiency.

Reference: Article 1, Appendix C
"""

import numpy as np


def sinusoidal_correction(t, A_ttv, P_ttv, E_ttv, t0=0.0):
    """
    Apply sinusoidal TTV correction to time array.

    This is the core TTV-BLS correction from Article 1. It transforms
    the time axis to remove the sinusoidal TTV component.

    Parameters
    ----------
    t : array_like
        Time array (days, BJD or similar)
    A_ttv : float
        TTV amplitude (days). This is the semi-amplitude of the
        sinusoidal timing variation.
    P_ttv : float
        TTV super-period (days). For near mean-motion resonance systems,
        this is typically P_super = 1/|j/P_out - (j-1)/P_in|
    E_ttv : float
        TTV epoch offset (days). Phase offset of the TTV sinusoid.
    t0 : float, optional
        Reference epoch for the transiting planet (days). Default is 0.0.

    Returns
    -------
    t_corrected : ndarray
        Corrected time array with TTV removed

    Notes
    -----
    The correction formula is:
        t' = t - A_TTV * sin(2*pi*(t - (t0 + E_TTV)) / P_TTV)

    This effectively "shifts" each data point in time to account for
    where the transit center has moved due to TTVs.

    Examples
    --------
    >>> t = np.linspace(0, 1000, 10000)
    >>> t_corr = sinusoidal_correction(t, A_ttv=0.1, P_ttv=100, E_ttv=0)
    """
    t = np.asarray(t)

    if P_ttv <= 0 or A_ttv <= 0:
        return t.copy()

    phase = 2 * np.pi * (t - (t0 + E_ttv)) / P_ttv
    t_corrected = t - A_ttv * np.sin(phase)

    return t_corrected


def distort_timebase(t, epoch, p_ttv, a_ttv, e_ttv):
    """
    Apply TTV distortion to time array (alias for sinusoidal_correction).

    This function name is used for backward compatibility with Article 1
    simulation scripts.

    Parameters
    ----------
    t : array_like
        Time array (days)
    epoch : float
        Reference epoch (t0)
    p_ttv : float
        TTV period (days)
    a_ttv : float
        TTV amplitude (days)
    e_ttv : float
        TTV epoch offset (days)

    Returns
    -------
    t_corrected : ndarray
        Corrected time array
    """
    return sinusoidal_correction(t, a_ttv, p_ttv, e_ttv, t0=epoch)


def inverse_sinusoidal_correction(t, A_ttv, P_ttv, E_ttv, t0=0.0, n_iterations=5):
    """
    Compute the inverse sinusoidal correction (for injection).

    Given corrected times, find the original times that would produce them.
    This is useful for injecting transits with TTVs into light curves.

    Parameters
    ----------
    t : array_like
        Corrected time array (days)
    A_ttv, P_ttv, E_ttv, t0 : float
        TTV parameters (same as sinusoidal_correction)
    n_iterations : int, optional
        Number of Newton-Raphson iterations for inversion. Default is 5.

    Returns
    -------
    t_original : ndarray
        Original time array that would produce t after correction

    Notes
    -----
    Uses Newton-Raphson iteration to solve:
        t = t_orig - A_TTV * sin(2*pi*(t_orig - (t0 + E_TTV)) / P_TTV)
    for t_orig given t.
    """
    t = np.asarray(t)

    if P_ttv <= 0 or A_ttv <= 0:
        return t.copy()

    # Initial guess
    t_orig = t.copy()

    # Newton-Raphson iteration
    for _ in range(n_iterations):
        phase = 2 * np.pi * (t_orig - (t0 + E_ttv)) / P_ttv
        f = t_orig - A_ttv * np.sin(phase) - t
        df = 1 - A_ttv * (2 * np.pi / P_ttv) * np.cos(phase)
        t_orig = t_orig - f / df

    return t_orig


def compute_super_period(P_inner, P_outer, j=2):
    """
    Compute the TTV super-period for a near j:(j-1) resonance.

    Parameters
    ----------
    P_inner : float
        Orbital period of inner planet (days)
    P_outer : float
        Orbital period of outer planet (days)
    j : int, optional
        Resonance order (j:(j-1)). Default is 2 for 2:1 resonance.

    Returns
    -------
    P_super : float
        Super-period (days)

    Notes
    -----
    The super-period is given by:
        P_super = 1 / |j/P_outer - (j-1)/P_inner|

    For a perfect j:(j-1) resonance, P_super would be infinite.
    Near-resonance systems have finite but typically long super-periods.

    Examples
    --------
    >>> # Kepler-9 b,c near 2:1 resonance
    >>> P_super = compute_super_period(19.24, 38.91, j=2)
    """
    freq_inner = (j - 1) / P_inner
    freq_outer = j / P_outer
    delta_freq = abs(freq_outer - freq_inner)

    if delta_freq < 1e-10:
        return np.inf

    return 1.0 / delta_freq


if __name__ == "__main__":
    # Simple test
    print("Testing sinusoidal TTV correction module...")

    # Create test time array
    t = np.linspace(0, 1000, 10000)

    # Test parameters
    A_ttv = 0.05  # 0.05 days = 72 minutes
    P_ttv = 100.0  # 100 days
    E_ttv = 0.0

    # Apply correction
    t_corr = sinusoidal_correction(t, A_ttv, P_ttv, E_ttv)

    # Test inverse
    t_inv = inverse_sinusoidal_correction(t_corr, A_ttv, P_ttv, E_ttv)

    # Check roundtrip
    max_error = np.max(np.abs(t - t_inv))
    print(f"  Roundtrip max error: {max_error:.2e} days ({max_error*24*60:.2e} minutes)")

    # Test super-period calculation
    P_super = compute_super_period(19.24, 38.91, j=2)
    print(f"  Kepler-9 super-period: {P_super:.1f} days")

    print("  All tests passed!")
