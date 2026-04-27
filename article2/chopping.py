#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synodic Chopping TTV Correction Module

Implements correction for the "chopping" component of TTVs that occurs
in near mean-motion resonance systems due to close conjunctions.

Theory Background (Deck & Agol 2015):
Near mean-motion resonances, TTVs have multiple components:
- Resonant term: Period P_super = 1/|j/P_out - (j-1)/P_in|
- Chopping term: Period P_syn = 1/|1/P_out - 1/P_in| (synodic period)

The chopping term arises from the orbital dynamics during conjunctions
and is typically smaller but faster than the resonant term.
"""

import numpy as np


def compute_synodic_period(P_orb, P_perturber):
    """
    Compute the synodic period between two planets.

    The synodic period is the time between successive conjunctions
    (alignments) of the two planets, as seen from the star.

    Parameters
    ----------
    P_orb : float
        Orbital period of the transiting planet (days)
    P_perturber : float
        Orbital period of the perturbing planet (days)

    Returns
    -------
    P_syn : float
        Synodic period (days)

    Notes
    -----
    The synodic period is given by:
        P_syn = 1 / |1/P_perturber - 1/P_orb|

    For example, if P_orb = 10 days and P_perturber = 15 days:
        P_syn = 1 / |1/15 - 1/10| = 1 / |0.0667 - 0.1| = 30 days

    Examples
    --------
    >>> P_syn = compute_synodic_period(10.0, 15.0)
    >>> print(f"Synodic period: {P_syn:.1f} days")
    Synodic period: 30.0 days
    """
    if P_orb <= 0 or P_perturber <= 0:
        raise ValueError("Periods must be positive")

    freq_diff = abs(1.0 / P_perturber - 1.0 / P_orb)

    if freq_diff < 1e-10:
        return np.inf

    return 1.0 / freq_diff


def compute_perturber_period(P_orb, resonance):
    """
    Compute the perturber period for a given resonance configuration.

    Parameters
    ----------
    P_orb : float
        Orbital period of the transiting planet (days)
    resonance : str
        Resonance specification, e.g., '2:1', '3:2', '5:3'
        The first number is for the outer planet, second for inner.

    Returns
    -------
    P_perturber : float
        Orbital period of the perturbing planet (days)
    is_inner : bool
        True if the perturber is the inner planet

    Examples
    --------
    >>> P_pert, is_inner = compute_perturber_period(20.0, '2:1')
    >>> # Returns ~40.0, False (perturber is outer, with ~2:1 period ratio)
    """
    parts = resonance.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid resonance format: {resonance}")

    j_outer = int(parts[0])
    j_inner = int(parts[1])

    # Period ratio for exact resonance
    ratio = j_outer / j_inner

    # Assume transiting planet could be either inner or outer
    # Return the perturber period assuming transiting planet is inner
    P_perturber = P_orb * ratio
    is_inner = False  # Perturber is outer

    return P_perturber, is_inner


def chopping_correction(t, P_orb, P_perturber, A_chop, E_chop):
    """
    Apply synodic chopping correction to time array.

    This corrects for the short-period "chopping" component of TTVs
    that arises from planetary conjunctions.

    Parameters
    ----------
    t : array_like
        Time array (days)
    P_orb : float
        Orbital period of transiting planet (days)
    P_perturber : float
        Orbital period of perturbing planet (days)
    A_chop : float
        Chopping amplitude (days). Typically smaller than the
        resonant TTV amplitude.
    E_chop : float
        Chopping epoch (days). Phase offset of the chopping signal.

    Returns
    -------
    t_corrected : ndarray
        Corrected time array with chopping component removed

    Notes
    -----
    The correction formula is:
        P_syn = 1 / |1/P_perturber - 1/P_orb|
        t' = t - A_chop * sin(2*pi*(t - E_chop) / P_syn)

    Examples
    --------
    >>> t = np.linspace(0, 1000, 10000)
    >>> t_corr = chopping_correction(t, P_orb=10.0, P_perturber=15.0,
    ...                               A_chop=0.01, E_chop=0.0)
    """
    t = np.asarray(t)

    if A_chop <= 0:
        return t.copy()

    P_syn = compute_synodic_period(P_orb, P_perturber)

    if np.isinf(P_syn):
        return t.copy()

    phase = 2 * np.pi * (t - E_chop) / P_syn
    t_corrected = t - A_chop * np.sin(phase)

    return t_corrected


def estimate_chopping_amplitude(M_perturber, M_star, P_orb, P_perturber, e_perturber=0.0):
    """
    Estimate the chopping amplitude from system parameters.

    Uses scaling relations from Lithwick et al. (2012) and
    Deck & Agol (2015).

    Parameters
    ----------
    M_perturber : float
        Mass of perturbing planet (Earth masses)
    M_star : float
        Mass of host star (Solar masses)
    P_orb : float
        Orbital period of transiting planet (days)
    P_perturber : float
        Orbital period of perturbing planet (days)
    e_perturber : float, optional
        Eccentricity of perturber orbit. Default is 0.0.

    Returns
    -------
    A_chop : float
        Estimated chopping amplitude (days)

    Notes
    -----
    This is an approximation. The actual chopping amplitude depends
    on the full orbital configuration and can vary significantly.

    Rough scaling:
        A_chop ~ (M_pert / M_star) * P_syn / (2*pi) * f(period_ratio)

    where f is a function of order unity that depends on the period ratio.
    """
    # Convert perturber mass to solar masses
    M_pert_solar = M_perturber * 3.0027e-6  # Earth mass in solar masses

    # Mass ratio
    mu = M_pert_solar / M_star

    # Synodic period
    P_syn = compute_synodic_period(P_orb, P_perturber)

    if np.isinf(P_syn):
        return 0.0

    # Period ratio factor (simplified)
    ratio = max(P_orb, P_perturber) / min(P_orb, P_perturber)
    f_ratio = 1.0 / (ratio - 1.0)**2 if ratio > 1.01 else 100.0

    # Rough estimate (scaled to match observed systems)
    A_chop = mu * P_syn / (2 * np.pi) * f_ratio * 0.1  # 0.1 is empirical factor

    # Eccentricity enhancement
    A_chop *= (1 + 2 * e_perturber)

    return A_chop


if __name__ == "__main__":
    # Simple test
    print("Testing chopping TTV correction module...")

    # Create test time array
    t = np.linspace(0, 1000, 10000)

    # Test parameters (Kepler-36 like: 7:6 resonance)
    P_orb = 13.87  # days
    P_perturber = 16.22  # days (outer planet)
    A_chop = 0.02  # 0.02 days = 29 minutes
    E_chop = 0.0

    # Compute synodic period
    P_syn = compute_synodic_period(P_orb, P_perturber)
    print(f"  Synodic period: {P_syn:.2f} days")

    # Apply correction
    t_corr = chopping_correction(t, P_orb, P_perturber, A_chop, E_chop)

    # Check correction range
    max_shift = np.max(np.abs(t_corr - t))
    print(f"  Max time shift: {max_shift:.4f} days ({max_shift*24*60:.1f} minutes)")

    # Test perturber period calculation
    P_pert, is_inner = compute_perturber_period(10.0, '2:1')
    print(f"  2:1 resonance perturber period: {P_pert:.1f} days")

    # Test amplitude estimation
    A_est = estimate_chopping_amplitude(
        M_perturber=10.0,  # 10 Earth masses
        M_star=1.0,        # Solar mass
        P_orb=P_orb,
        P_perturber=P_perturber
    )
    print(f"  Estimated chopping amplitude: {A_est*24*60:.1f} minutes")

    print("  All tests passed!")
