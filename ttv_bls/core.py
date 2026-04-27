"""
Core TTV-BLS algorithm implementation.

This module implements:
- Standard BLS (Box Least Squares) transit search
- TTV-BLS with time-base distortion
- Grid search over TTV parameter space

Based on the algorithm described in Kalogerakos & West (2026).
"""

import numpy as np
import numpy.lib.recfunctions as rfn
from math import sqrt, pi, ceil
import multiprocessing as mp
from itertools import starmap


def optimal_sample_periods(min_period, max_period, obs_duration, bin_width):
    """
    Generate optimally-spaced sample periods for BLS search.

    The spacing ensures that frequency resolution is matched to the
    bin width, avoiding over-sampling at long periods.

    Parameters
    ----------
    min_period : float
        Minimum trial period (days)
    max_period : float
        Maximum trial period (days)
    obs_duration : float
        Total observation duration (days)
    bin_width : float
        Phase bin width (days)

    Returns
    -------
    np.ndarray
        Array of sample periods
    """
    sample_periods = []
    period = min_period
    while period < max_period:
        sample_periods.append(period)
        delta_freq = bin_width / (period * obs_duration)
        period = 1.0 / (-delta_freq + 1.0 / period)
    return np.array(sample_periods)


def distort_timebase(t, epoch, p_ttv, a_ttv, e_ttv=0.0):
    """
    Apply TTV correction to time stamps.

    Transforms time stamps according to:
        t' = t - A_TTV * sin(2*pi*(t - (epoch + E_TTV)) / P_TTV)

    This "unwinds" sinusoidal TTVs so transits align during phase folding.

    Parameters
    ----------
    t : np.ndarray
        Original time stamps (days)
    epoch : float
        Reference epoch for TTV (days)
    p_ttv : float
        TTV period (days)
    a_ttv : float
        TTV amplitude (days)
    e_ttv : float, optional
        TTV epoch offset (days). Default 0.0

    Returns
    -------
    np.ndarray
        Corrected time stamps
    """
    if p_ttv > 0.0 and a_ttv > 0.0:
        return t - a_ttv * np.sin(2 * pi * (t - (epoch + e_ttv)) / p_ttv)
    return t


def transit_search(tstamp, flux, flux_err, min_period=None, max_period=None,
                   sample_periods=None, bin_width=45.0, min_box_width=2,
                   max_box_width=5):
    """
    Perform BLS transit search on a light curve.

    Parameters
    ----------
    tstamp : np.ndarray
        Time stamps (days)
    flux : np.ndarray
        Flux values (normalised, transit depth negative)
    flux_err : np.ndarray
        Flux uncertainties
    min_period : float, optional
        Minimum trial period (days)
    max_period : float, optional
        Maximum trial period (days)
    sample_periods : np.ndarray, optional
        Pre-computed sample periods (overrides min/max_period)
    bin_width : float, optional
        Phase bin width in minutes. Default 45.0
    min_box_width : int, optional
        Minimum transit width in bins. Default 2
    max_box_width : int, optional
        Maximum transit width in bins. Default 5

    Returns
    -------
    np.ndarray
        Structured array with fields:
        - period: trial period
        - delta_chisq: delta chi-squared statistic
        - epoch: best-fit epoch
        - depth: transit depth
        - depth_err: depth uncertainty
        - num_pts_in_transit: points in transit
        - width: transit width (days)
        - sde: Signal Detection Efficiency
    """
    t0 = np.min(tstamp)
    bin_width_days = bin_width / (24 * 60)  # Convert minutes to days
    obs_duration = np.max(tstamp) - np.min(tstamp)

    # Determine sample periods
    assert (min_period is not None and max_period is not None and sample_periods is None) \
        or sample_periods is not None, \
        "Either sample_periods or both min_period and max_period must be specified"

    if sample_periods is None:
        sample_periods = optimal_sample_periods(min_period, max_period, obs_duration, bin_width_days)

    # Compute weights
    weight = 1.0 / (flux_err * flux_err)
    wflux = flux * weight
    wflux2 = (flux * flux) * weight
    T = np.sum(weight)

    pdgram = []
    for period in sample_periods:
        num_bins = int((period / bin_width_days) + 0.5)
        real_bin_width = period / num_bins
        bin_idx = np.floor(np.mod((tstamp - t0) / period, 1.0) * num_bins).astype(np.int32)

        # Compute partial sums for each bin
        pstat = np.zeros((4, num_bins), dtype=np.float32)
        pstat[0, :] = np.bincount(bin_idx, wflux, num_bins)
        pstat[1, :] = np.bincount(bin_idx, wflux2, num_bins)
        pstat[2, :] = np.bincount(bin_idx, weight, num_bins)
        pstat[3, :] = np.bincount(bin_idx, minlength=num_bins)

        # Extend for wrap-around
        pstat = np.concatenate((pstat, pstat[:, :max_box_width]), axis=1)

        # Stack shifted copies
        pstat = np.stack([np.roll(pstat, -shft, axis=1)
                          for shft in range(max_box_width + 1)], axis=2)

        # Accumulate partial sums
        sstat = np.cumsum(pstat, axis=-1)[:, :, min_box_width:]

        # Compute delta-chisq
        S = sstat[0, :, :]
        Q = sstat[1, :, :]
        R = sstat[2, :, :]
        N = sstat[3, :, :]
        denom = R * (T - R)
        delta_chisq = np.where((N > 2) * (denom != 0.0), -(S * S) * T / denom, 0.0)

        # Find most significant detection
        min_idx = np.unravel_index(np.argmin(delta_chisq), delta_chisq.shape)
        delta_chisq_best = delta_chisq[min_idx]
        epoch_best = (min_idx[0] + (min_idx[1] + min_box_width) * 0.5) * real_bin_width
        depth = S[min_idx] * T / (R[min_idx] * (T - R[min_idx])) if R[min_idx] * (T - R[min_idx]) > 0 else 0.0
        depth_err = sqrt(T / (R[min_idx] * (T - R[min_idx]))) if R[min_idx] * (T - R[min_idx]) > 0 else 0.0
        num_pts = N[min_idx]
        width = (min_idx[1] + min_box_width) * real_bin_width

        pdgram.append((period, delta_chisq_best, epoch_best + t0, depth, depth_err, num_pts, width))

    # Pack results
    res = np.array(pdgram, dtype=[
        ('period', 'f8'), ('delta_chisq', 'f4'), ('epoch', 'f4'), ('depth', 'f4'),
        ('depth_err', 'f4'), ('num_pts_in_transit', 'i4'), ('width', 'f4')
    ])

    # Compute SDE
    median = np.median(res['delta_chisq'])
    rms = np.median(np.abs(res['delta_chisq'] - median)) * 1.48
    sde = (median - res['delta_chisq']) / rms if rms > 0 else np.zeros(len(res))
    res = rfn.append_fields(res, 'sde', sde, dtypes=('f4',), usemask=False)

    return res


def ttv_bls_search(t, flux, flux_err, p_ttv, a_ttv, e_ttv, min_period, max_period,
                   bin_width=45.0):
    """
    Perform BLS search with TTV correction.

    Parameters
    ----------
    t : np.ndarray
        Time stamps (days)
    flux : np.ndarray
        Flux values
    flux_err : np.ndarray
        Flux uncertainties
    p_ttv : float
        TTV period (days)
    a_ttv : float
        TTV amplitude (days)
    e_ttv : float
        TTV epoch offset (days)
    min_period : float
        Minimum trial period (days)
    max_period : float
        Maximum trial period (days)
    bin_width : float, optional
        Phase bin width in minutes. Default 45.0

    Returns
    -------
    np.ndarray
        Best detection from BLS search (single row of structured array)
    """
    t_prime = distort_timebase(t, 0.0, p_ttv, a_ttv, e_ttv)
    res = transit_search(t_prime, flux, flux_err, min_period, max_period, bin_width=bin_width)
    best_idx = np.argmax(res['sde'])
    return res[best_idx]


def _ttv_bls_worker(args):
    """Worker function for parallel TTV-BLS search."""
    p_ttv, a_ttv, e_ttv, min_period, max_period, bin_width = args
    # Access global variables set by initializer
    t_prime = distort_timebase(_t_global, 0.0, p_ttv, a_ttv, e_ttv)
    res = transit_search(t_prime, _flux_global, _flux_err_global, min_period, max_period,
                         bin_width=bin_width)
    best_idx = np.argmax(res['sde'])
    return res[best_idx]


def _init_worker(t, flux, flux_err):
    """Initialize worker with shared data."""
    global _t_global, _flux_global, _flux_err_global
    _t_global = t
    _flux_global = flux
    _flux_err_global = flux_err


def ttv_grid_search(t, flux, flux_err, min_p_ttv, max_p_ttv, n_p_ttv,
                    min_a_ttv, max_a_ttv, n_a_ttv, min_e_ttv=0.0, max_e_ttv=0.0,
                    n_e_ttv=1, min_period=0.6, max_period=75.0, bin_width=45.0,
                    parallel=True, n_workers=None):
    """
    Perform grid search over TTV parameter space.

    Parameters
    ----------
    t : np.ndarray
        Time stamps (days)
    flux : np.ndarray
        Flux values
    flux_err : np.ndarray
        Flux uncertainties
    min_p_ttv, max_p_ttv : float
        TTV period range (days)
    n_p_ttv : int
        Number of P_TTV grid points
    min_a_ttv, max_a_ttv : float
        TTV amplitude range (days)
    n_a_ttv : int
        Number of A_TTV grid points
    min_e_ttv, max_e_ttv : float, optional
        TTV epoch range (days). Default 0.0
    n_e_ttv : int, optional
        Number of E_TTV grid points. Default 1
    min_period, max_period : float, optional
        Trial period range for BLS (days)
    bin_width : float, optional
        Phase bin width in minutes. Default 45.0
    parallel : bool, optional
        Use parallel processing. Default True
    n_workers : int, optional
        Number of worker processes. Default None (use all CPUs)

    Returns
    -------
    np.ndarray
        3D structured array of shape (n_a_ttv, n_p_ttv, n_e_ttv)
        with best BLS result at each grid point
    """
    # Generate grid points
    p_ttv_samples = np.linspace(min_p_ttv, max_p_ttv, n_p_ttv, dtype=np.float32)
    a_ttv_samples = np.linspace(min_a_ttv, max_a_ttv, n_a_ttv, dtype=np.float32)
    e_ttv_samples = np.linspace(min_e_ttv, max_e_ttv, n_e_ttv, dtype=np.float32)

    # Create argument tuples for all grid points
    samples = []
    for a_ttv in a_ttv_samples:
        for p_ttv in p_ttv_samples:
            for e_ttv in e_ttv_samples:
                samples.append((p_ttv, a_ttv, e_ttv, min_period, max_period, bin_width))

    # Run search
    if parallel:
        if n_workers is None:
            n_workers = mp.cpu_count()
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(t, flux, flux_err)) as pool:
            res = list(pool.map(_ttv_bls_worker, samples, chunksize=ceil(len(samples) / n_workers)))
    else:
        _init_worker(t, flux, flux_err)
        res = [_ttv_bls_worker(s) for s in samples]

    # Convert to structured array and reshape
    res = np.array(res)

    # Add TTV parameters to results
    ttv_params = np.array([(s[1], s[0], s[2]) for s in samples],
                          dtype=[('a_ttv', 'f4'), ('p_ttv', 'f4'), ('e_ttv', 'f4')])
    res = rfn.merge_arrays([res, ttv_params], flatten=True, usemask=False)

    return res.reshape((n_a_ttv, n_p_ttv, n_e_ttv))


def compute_snr_ratio(sde_corrected, sde_standard):
    """
    Compute SNR improvement ratio.

    Parameters
    ----------
    sde_corrected : float
        SDE with TTV correction
    sde_standard : float
        SDE without TTV correction

    Returns
    -------
    float
        Improvement ratio (>1 means TTV-BLS is better)
    """
    if sde_standard > 0:
        return sde_corrected / sde_standard
    return np.inf if sde_corrected > 0 else 1.0
