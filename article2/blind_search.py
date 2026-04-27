#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blind TTV Search Module

Implements grid search optimization over TTV parameters to find
the best correction when TTV parameters are unknown.

This is essential for discovering new planets in systems where
TTVs are suspected but not yet characterized.
"""

import numpy as np
from math import sqrt
import time
import numpy.lib.recfunctions as rfn


def optimal_sample_periods(min_period, max_period, obs_duration, bin_width=45.0):
    """
    Generate optimally spaced trial periods for BLS.

    Uses frequency spacing that ensures adequate sampling without
    redundant computation.

    Parameters
    ----------
    min_period : float
        Minimum period to search (days)
    max_period : float
        Maximum period to search (days)
    obs_duration : float
        Total observation baseline (days)
    bin_width : float, optional
        Phase bin width in minutes. Default is 45.

    Returns
    -------
    periods : ndarray
        Array of trial periods (days)
    """
    bin_width_days = bin_width / (24 * 60)
    sample_periods = []
    p = min_period
    while p < max_period:
        sample_periods.append(p)
        delta_freq = bin_width_days / (p * obs_duration)
        p = 1.0 / (-delta_freq + 1.0 / p)
    return np.array(sample_periods)


def transit_search(tstamp, flux, flux_err, min_period=0.5, max_period=100.0,
                   bin_width=45.0, min_box_width=2, max_box_width=5):
    """
    Perform BLS transit search.

    Parameters
    ----------
    tstamp : array_like
        Time stamps (days)
    flux : array_like
        Flux values (relative)
    flux_err : array_like
        Flux uncertainties
    min_period, max_period : float
        Period search range (days)
    bin_width : float
        Phase bin width in minutes
    min_box_width, max_box_width : int
        Box width range in bins

    Returns
    -------
    results : recarray
        Structured array with columns:
        period, delta_chisq, epoch, depth, depth_err, num_pts_in_transit, width, sde
    """
    tstamp = np.asarray(tstamp)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)

    t0 = np.min(tstamp)
    bin_width_days = bin_width / (24 * 60)
    obs_duration = np.max(tstamp) - np.min(tstamp)
    sample_periods = optimal_sample_periods(min_period, max_period, obs_duration, bin_width)

    weight = 1.0 / (flux_err * flux_err)
    wflux = flux * weight
    T = np.sum(weight)

    pdgram = []
    for p in sample_periods:
        num_bins = int((p / bin_width_days) + 0.5)
        if num_bins < max_box_width + 1:
            continue
        real_bin_width = p / num_bins
        bin_idx = np.floor(np.mod((tstamp - t0) / p, 1.0) * num_bins).astype(np.int32)

        pstat = np.zeros((4, num_bins), dtype=np.float32)
        pstat[0, :] = np.bincount(bin_idx, wflux, num_bins)
        pstat[1, :] = np.bincount(bin_idx, (flux * flux) * weight, num_bins)
        pstat[2, :] = np.bincount(bin_idx, weight, num_bins)
        pstat[3, :] = np.bincount(bin_idx, minlength=num_bins)

        pstat = np.concatenate((pstat, pstat[:, :max_box_width]), axis=1)
        pstat = np.stack([np.roll(pstat, -shft, axis=1) for shft in range(max_box_width + 1)], axis=2)
        sstat = np.cumsum(pstat, axis=-1)[:, :, min_box_width:]

        S, R, N = sstat[0], sstat[2], sstat[3]
        denom = R * (T - R)
        delta_chisq = np.where((N > 2) * (denom != 0.0), -(S * S) * T / denom, 0.0)

        min_idx = np.unravel_index(np.argmin(delta_chisq), delta_chisq.shape)
        delta_chisq_val = delta_chisq[min_idx]
        epoch_val = (min_idx[0] + (min_idx[1] + min_box_width) * 0.5) * real_bin_width
        depth = S[min_idx] * T / (R[min_idx] * (T - R[min_idx])) if denom[min_idx] != 0 else 0
        depth_err = sqrt(T / (R[min_idx] * (T - R[min_idx]))) if denom[min_idx] != 0 else 0
        width = (min_idx[1] + min_box_width) * real_bin_width

        pdgram.append((p, delta_chisq_val, epoch_val + t0, depth, depth_err, N[min_idx], width))

    if len(pdgram) == 0:
        return None

    res = np.array(pdgram, dtype=[('period', 'f8'), ('delta_chisq', 'f4'), ('epoch', 'f4'),
                                   ('depth', 'f4'), ('depth_err', 'f4'),
                                   ('num_pts_in_transit', 'i4'), ('width', 'f4')])

    median = np.median(res['delta_chisq'])
    rms = np.median(np.abs(res['delta_chisq'] - median)) * 1.48
    sde = (median - res['delta_chisq']) / rms if rms > 0 else np.zeros(len(res))
    res = rfn.append_fields(res, 'sde', sde, dtypes=('f4'), usemask=False)

    return res


def distort_timebase(t, epoch, p_ttv, a_ttv, e_ttv):
    """Apply TTV correction to time array."""
    t = np.asarray(t)
    if p_ttv <= 0 or a_ttv <= 0:
        return t.copy()
    return t - a_ttv * np.sin(2 * np.pi * (t - (epoch + e_ttv)) / p_ttv)


def blind_ttv_search(t, flux, flux_err, A_grid, P_grid, E_grid='auto',
                     min_period=0.5, max_period=100.0, verbose=False):
    """
    Grid search over TTV parameters to find optimal correction.

    Parameters
    ----------
    t : array_like
        Time array (days)
    flux : array_like
        Flux values
    flux_err : array_like
        Flux uncertainties
    A_grid : array_like
        TTV amplitude values to test (days)
    P_grid : array_like
        TTV period values to test (days)
    E_grid : array_like or 'auto'
        TTV epoch values to test (days). If 'auto', uses 10 values
        per period spanning one TTV cycle.
    min_period, max_period : float
        Orbital period search range (days)
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        {
            'best_A': float,
            'best_P': float,
            'best_E': float,
            'best_SDE': float,
            'best_orbital_period': float,
            'all_results': list of dicts,
            'computation_time': float
        }
    """
    t = np.asarray(t)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)

    A_grid = np.atleast_1d(A_grid)
    P_grid = np.atleast_1d(P_grid)

    start_time = time.time()

    best_sde = -np.inf
    best_params = {'A': 0, 'P': 0, 'E': 0}
    best_orbital_period = 0

    all_results = []
    total_trials = 0

    for A in A_grid:
        for P in P_grid:
            # Determine E grid
            if E_grid == 'auto':
                E_values = np.linspace(0, P, 10, endpoint=False)
            else:
                E_values = np.atleast_1d(E_grid)

            for E in E_values:
                total_trials += 1

                # Apply TTV correction
                t_corr = distort_timebase(t, 0.0, P, A, E)

                # Run BLS
                res = transit_search(t_corr, flux, flux_err, min_period, max_period)

                if res is None:
                    continue

                # Get best SDE
                best_idx = np.argmax(res['sde'])
                sde = res['sde'][best_idx]
                orbital_period = res['period'][best_idx]

                all_results.append({
                    'A_ttv': A, 'P_ttv': P, 'E_ttv': E,
                    'SDE': float(sde), 'orbital_period': float(orbital_period)
                })

                if sde > best_sde:
                    best_sde = sde
                    best_params = {'A': A, 'P': P, 'E': E}
                    best_orbital_period = orbital_period

    computation_time = time.time() - start_time

    if verbose:
        print(f"Blind TTV search completed:")
        print(f"  Trials: {total_trials}")
        print(f"  Best SDE: {best_sde:.2f}")
        print(f"  Best A_TTV: {best_params['A']*24*60:.1f} min")
        print(f"  Best P_TTV: {best_params['P']:.1f} days")
        print(f"  Best orbital period: {best_orbital_period:.4f} days")
        print(f"  Time: {computation_time:.1f} s")

    return {
        'best_A': best_params['A'],
        'best_P': best_params['P'],
        'best_E': best_params['E'],
        'best_SDE': float(best_sde),
        'best_orbital_period': float(best_orbital_period),
        'all_results': all_results,
        'computation_time': computation_time,
        'n_trials': total_trials
    }


def adaptive_grid_search(t, flux, flux_err, A_range=(0.01, 0.5), P_range=(20, 300),
                         min_period=0.5, max_period=100.0,
                         n_iterations=3, refinement_factor=3, verbose=False):
    """
    Adaptive coarse-to-fine grid search.

    Starts with a coarse grid, then refines around the best solution.

    Parameters
    ----------
    t, flux, flux_err : array_like
        Light curve data
    A_range : tuple
        (min, max) TTV amplitude range (days)
    P_range : tuple
        (min, max) TTV period range (days)
    min_period, max_period : float
        Orbital period search range (days)
    n_iterations : int
        Number of refinement iterations
    refinement_factor : float
        Factor by which to shrink search range each iteration
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Same as blind_ttv_search, plus 'iterations' list
    """
    A_min, A_max = A_range
    P_min, P_max = P_range

    iterations = []
    best_overall = None

    for iteration in range(n_iterations):
        # Grid density depends on iteration
        if iteration == 0:
            n_A, n_P = 5, 10
        else:
            n_A, n_P = 5, 5

        A_grid = np.linspace(A_min, A_max, n_A)
        P_grid = np.linspace(P_min, P_max, n_P)

        if verbose:
            print(f"Iteration {iteration + 1}/{n_iterations}:")
            print(f"  A range: [{A_min*24*60:.1f}, {A_max*24*60:.1f}] min")
            print(f"  P range: [{P_min:.1f}, {P_max:.1f}] days")

        # Run search
        result = blind_ttv_search(t, flux, flux_err, A_grid, P_grid,
                                  min_period=min_period, max_period=max_period,
                                  verbose=False)

        iterations.append({
            'iteration': iteration,
            'A_range': (A_min, A_max),
            'P_range': (P_min, P_max),
            'best_SDE': result['best_SDE'],
            'best_A': result['best_A'],
            'best_P': result['best_P'],
        })

        if best_overall is None or result['best_SDE'] > best_overall['best_SDE']:
            best_overall = result

        # Refine range around best
        A_center = result['best_A']
        P_center = result['best_P']

        A_width = (A_max - A_min) / refinement_factor
        P_width = (P_max - P_min) / refinement_factor

        A_min = max(A_range[0], A_center - A_width / 2)
        A_max = min(A_range[1], A_center + A_width / 2)
        P_min = max(P_range[0], P_center - P_width / 2)
        P_max = min(P_range[1], P_center + P_width / 2)

        if verbose:
            print(f"  Best SDE: {result['best_SDE']:.2f}")

    best_overall['iterations'] = iterations

    if verbose:
        print(f"\nFinal result:")
        print(f"  Best SDE: {best_overall['best_SDE']:.2f}")
        print(f"  Best A_TTV: {best_overall['best_A']*24*60:.1f} min")
        print(f"  Best P_TTV: {best_overall['best_P']:.1f} days")

    return best_overall


def compute_sde_landscape(t, flux, flux_err, A_grid, P_grid,
                          min_period=0.5, max_period=100.0):
    """
    Compute full SDE landscape over A_TTV and P_TTV grid.

    Useful for visualization and understanding the parameter space.

    Parameters
    ----------
    t, flux, flux_err : array_like
        Light curve data
    A_grid, P_grid : array_like
        Parameter grids
    min_period, max_period : float
        Orbital period search range

    Returns
    -------
    landscape : dict
        {
            'A_grid': ndarray,
            'P_grid': ndarray,
            'SDE': 2D ndarray (shape: len(A_grid) x len(P_grid)),
            'orbital_period': 2D ndarray
        }
    """
    A_grid = np.atleast_1d(A_grid)
    P_grid = np.atleast_1d(P_grid)

    SDE = np.zeros((len(A_grid), len(P_grid)))
    orbital_period = np.zeros((len(A_grid), len(P_grid)))

    for i, A in enumerate(A_grid):
        for j, P in enumerate(P_grid):
            # Use E=0 for landscape (could average over E)
            t_corr = distort_timebase(t, 0.0, P, A, 0.0)
            res = transit_search(t_corr, flux, flux_err, min_period, max_period)

            if res is not None:
                best_idx = np.argmax(res['sde'])
                SDE[i, j] = res['sde'][best_idx]
                orbital_period[i, j] = res['period'][best_idx]

    return {
        'A_grid': A_grid,
        'P_grid': P_grid,
        'SDE': SDE,
        'orbital_period': orbital_period
    }


if __name__ == "__main__":
    # Simple test
    print("Testing blind TTV search module...")

    # Create synthetic light curve with TTV
    np.random.seed(42)

    # Parameters
    duration = 150.0  # days
    cadence = 600.0   # seconds
    period = 5.0      # days
    depth = 0.01      # relative
    A_ttv_true = 0.05  # days
    P_ttv_true = 80.0  # days

    # Generate time array
    t = np.arange(0, duration, cadence / 86400.0)
    n_pts = len(t)

    # Generate flux with transits and TTVs
    flux = np.zeros(n_pts)
    n_transit = 0
    epoch = 1.0
    while True:
        t_mid = epoch + n_transit * period
        t_mid += A_ttv_true * np.sin(2 * np.pi * n_transit * period / P_ttv_true)
        if t_mid > duration:
            break
        in_transit = np.abs(t - t_mid) < 0.05
        flux[in_transit] = -depth
        n_transit += 1

    # Add noise
    noise_level = 0.002
    flux += np.random.normal(0, noise_level, n_pts)
    flux_err = np.full(n_pts, noise_level)

    print(f"  Created light curve with {n_transit} transits")

    # Test coarse search
    print("\n  Testing coarse grid search...")
    A_grid = np.array([0.02, 0.04, 0.06, 0.08])
    P_grid = np.array([40, 60, 80, 100, 120])

    result = blind_ttv_search(t, flux, flux_err, A_grid, P_grid, verbose=True)

    print(f"\n  True A_TTV: {A_ttv_true*24*60:.1f} min, Found: {result['best_A']*24*60:.1f} min")
    print(f"  True P_TTV: {P_ttv_true:.1f} days, Found: {result['best_P']:.1f} days")

    # Test adaptive search
    print("\n  Testing adaptive search...")
    result_adaptive = adaptive_grid_search(
        t, flux, flux_err,
        A_range=(0.01, 0.1), P_range=(30, 150),
        n_iterations=2, verbose=True
    )

    print("\n  All tests passed!")
