"""
Synthetic light curve generation with TTVs.

This module provides functions to create realistic transit light curves
with sinusoidal Transit Timing Variations using the BATMAN package.
"""

import numpy as np
from math import sin, pi, sqrt, fabs
from numpy.random import default_rng

try:
    import batman
except ImportError:
    raise ImportError("batman-package required. Install with: pip install batman-package")


# Default random number generator
_rng = default_rng()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    global _rng
    _rng = default_rng(seed)


def compute_transit_template(cadence, period, r_planet, a_rs=15.0, inc=90.0,
                             limb_dark_coeffs=(0.1, 0.3)):
    """
    Compute a transit template using BATMAN.

    Parameters
    ----------
    cadence : float
        Observation cadence (seconds)
    period : float
        Orbital period (days)
    r_planet : float
        Planet radius in units of stellar radii (Rp/Rs)
    a_rs : float, optional
        Semi-major axis in units of stellar radii. Default 15.0
    inc : float, optional
        Orbital inclination (degrees). Default 90.0
    limb_dark_coeffs : tuple, optional
        Quadratic limb darkening coefficients [u1, u2]. Default (0.1, 0.3)

    Returns
    -------
    tuple
        (template_t, template_flux, transit_duration)
        - template_t: time stamps relative to mid-transit (days)
        - template_flux: flux values (deviation from 1.0, so negative during transit)
        - transit_duration: total transit duration T14 (days)
    """
    params = batman.TransitParams()
    params.t0 = 0.0  # time of inferior conjunction
    params.per = period
    params.rp = r_planet
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(limb_dark_coeffs)
    params.limb_dark = "quadratic"

    # Compute template over one orbit centred on transit
    t = np.arange(-period * 0.5, period * 0.5, cadence / 86400.0)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params) - 1.0  # Deviation from 1.0

    # Extract non-zero portion (transit only)
    nzidx = np.where(flux != 0.0)[0]
    if len(nzidx) == 0:
        # No transit visible (planet too small or grazing)
        return np.array([0.0]), np.array([0.0]), 0.0

    nz_start = max(0, min(nzidx) - 3)
    nz_end = min(len(flux), max(nzidx) + 4)

    template_t = t[nz_start:nz_end]
    template_flux = flux[nz_start:nz_end]
    transit_duration = t[nz_end - 1] - t[nz_start]

    return template_t, template_flux, transit_duration


def get_transit_duration(period, r_planet, a_rs=15.0, inc=90.0):
    """
    Compute transit duration T14 analytically.

    Parameters
    ----------
    period : float
        Orbital period (days)
    r_planet : float
        Planet radius in units of stellar radii (Rp/Rs)
    a_rs : float, optional
        Semi-major axis in units of stellar radii. Default 15.0
    inc : float, optional
        Orbital inclination (degrees). Default 90.0

    Returns
    -------
    float
        Transit duration T14 (days)
    """
    inc_rad = np.radians(inc)
    b = a_rs * np.cos(inc_rad)  # Impact parameter

    # Check if transit occurs
    if b >= 1.0 + r_planet:
        return 0.0

    # T14 formula (circular orbit)
    sin_term = np.sqrt((1 + r_planet) ** 2 - b ** 2) / a_rs
    if sin_term > 1.0:
        sin_term = 1.0
    t14 = (period / np.pi) * np.arcsin(sin_term)

    return t14


def create_lightcurve(cadence, duration, period, epoch, a_ttv=0.0, p_ttv=0.0,
                      e_ttv=0.0, count_rate=0.0, r_planet=0.1, a_rs=15.0,
                      inc=90.0, limb_dark_coeffs=(0.1, 0.3), seed=None):
    """
    Create a synthetic light curve with optional TTVs and noise.

    Parameters
    ----------
    cadence : float
        Observation cadence (seconds)
    duration : float
        Total observation duration (days)
    period : float
        Orbital period (days)
    epoch : float
        Time of first transit (days from start)
    a_ttv : float, optional
        TTV amplitude (days). Default 0.0
    p_ttv : float, optional
        TTV period (days). Default 0.0
    e_ttv : float, optional
        TTV epoch offset (days). Default 0.0
    count_rate : float, optional
        Source count rate (photons/s). If >0, adds Poisson noise. Default 0.0
    r_planet : float, optional
        Planet radius (Rp/Rs). Default 0.1
    a_rs : float, optional
        Semi-major axis (a/Rs). Default 15.0
    inc : float, optional
        Orbital inclination (degrees). Default 90.0
    limb_dark_coeffs : tuple, optional
        Limb darkening coefficients. Default (0.1, 0.3)
    seed : int, optional
        Random seed for noise. Default None

    Returns
    -------
    tuple
        (t, flux, flux_err, transit_duration)
        - t: time stamps (days)
        - flux: flux values (deviation from 1.0)
        - flux_err: flux uncertainties
        - transit_duration: T14 (days)
    """
    if seed is not None:
        rng = default_rng(seed)
    else:
        rng = _rng

    # Sanity check
    if p_ttv > 0 and a_ttv > 0:
        assert fabs(e_ttv) < p_ttv, "TTV epoch must be in range -p_ttv < e_ttv < +p_ttv"

    # Compute transit template
    template_t, template_flux, transit_duration = compute_transit_template(
        cadence, period, r_planet, a_rs, inc, limb_dark_coeffs
    )

    # Create time array
    t = np.arange(0.0, duration, cadence / 86400.0, dtype=np.float32)
    flux = np.zeros_like(t)

    # Interpolate transits at computed mid-times
    n = 0
    while True:
        tmid = epoch + n * period
        if p_ttv > 0.0 and a_ttv > 0.0:
            tmid += a_ttv * sin(2 * pi * (n * period - e_ttv) / p_ttv)
        if tmid > duration + transit_duration:
            break
        flux += np.interp(t - tmid, template_t, template_flux, left=0.0, right=0.0)
        n += 1

    # Add noise if requested
    if count_rate > 0.0:
        sigma = 1.0 / sqrt(count_rate * cadence)
        flux = rng.normal(flux, sigma).astype(np.float32)
        flux_err = np.full_like(flux, sigma, dtype=np.float32)
    else:
        flux_err = np.zeros_like(flux, dtype=np.float32)

    return t, flux, flux_err, transit_duration


def create_lightcurve_target_snr(cadence, duration, period, epoch, target_bls_sde,
                                 a_ttv=0.0, p_ttv=0.0, e_ttv=0.0, r_planet=0.1,
                                 a_rs=15.0, inc=90.0, limb_dark_coeffs=(0.1, 0.3),
                                 seed=None, max_iterations=20, tolerance=0.5):
    """
    Create light curve with noise level calibrated to achieve target BLS SDE.

    This is useful for Study 2 (marginal detection) where we need light curves
    at the detection boundary.

    Parameters
    ----------
    cadence : float
        Observation cadence (seconds)
    duration : float
        Total observation duration (days)
    period : float
        Orbital period (days)
    epoch : float
        Time of first transit (days from start)
    target_bls_sde : float
        Target SDE for standard BLS (without TTV correction)
    a_ttv, p_ttv, e_ttv : float
        TTV parameters
    r_planet : float
        Planet radius (Rp/Rs)
    a_rs : float
        Semi-major axis (a/Rs)
    inc : float
        Orbital inclination (degrees)
    limb_dark_coeffs : tuple
        Limb darkening coefficients
    seed : int, optional
        Random seed
    max_iterations : int
        Maximum iterations for count rate search
    tolerance : float
        Acceptable SDE deviation from target

    Returns
    -------
    tuple
        (t, flux, flux_err, transit_duration, count_rate, achieved_sde)
    """
    from .core import transit_search

    # Start with initial guess based on transit depth and target SNR
    # SDE ~ SNR * sqrt(N_transits) approximately
    n_transits = int(duration / period)
    depth_approx = r_planet ** 2
    initial_count_rate = (depth_approx * target_bls_sde / sqrt(n_transits)) ** (-2) / cadence

    # Binary search for correct count rate
    count_rate_low = initial_count_rate * 0.01
    count_rate_high = initial_count_rate * 100

    for iteration in range(max_iterations):
        count_rate = sqrt(count_rate_low * count_rate_high)  # Geometric mean

        t, flux, flux_err, t14 = create_lightcurve(
            cadence, duration, period, epoch, a_ttv, p_ttv, e_ttv,
            count_rate, r_planet, a_rs, inc, limb_dark_coeffs, seed
        )

        # Run standard BLS (no TTV correction)
        res = transit_search(t, flux, flux_err, period * 0.5, period * 2.0)
        achieved_sde = np.max(res['sde'])

        if abs(achieved_sde - target_bls_sde) < tolerance:
            return t, flux, flux_err, t14, count_rate, achieved_sde

        # Adjust search bounds
        if achieved_sde > target_bls_sde:
            count_rate_high = count_rate  # Need more noise (lower count rate)
        else:
            count_rate_low = count_rate  # Need less noise (higher count rate)

    # Return best attempt
    return t, flux, flux_err, t14, count_rate, achieved_sde


def compute_a_ttv_over_t14(a_ttv, transit_duration):
    """
    Compute the dimensionless TTV amplitude ratio.

    Parameters
    ----------
    a_ttv : float
        TTV amplitude (days)
    transit_duration : float
        Transit duration T14 (days)

    Returns
    -------
    float
        A_TTV / T_14 ratio
    """
    if transit_duration > 0:
        return a_ttv / transit_duration
    return 0.0
