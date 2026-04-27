#!/usr/bin/env python3
"""
TTV-BLS Quickstart Example

This script demonstrates the basic usage of the TTV-BLS package.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from ttv_bls import (
    create_lightcurve,
    transit_search,
    distort_timebase,
    get_transit_duration,
    CRITICAL_THRESHOLD,
)


def example_basic_usage():
    """Basic example: create light curve and run BLS."""
    print("\n=== Example 1: Basic TTV-BLS Usage ===\n")

    # Create a light curve with a transiting planet experiencing TTVs
    t, flux, flux_err, t14 = create_lightcurve(
        cadence=120,           # 2-minute cadence (seconds)
        duration=180,          # 180 days observation
        period=5.0,            # 5-day orbital period
        epoch=2.5,             # First transit at t=2.5 days
        a_ttv=0.04,            # TTV amplitude = 58 minutes
        p_ttv=60.0,            # TTV period = 60 days
        count_rate=10000,      # High SNR
        r_planet=0.08,         # Sub-Neptune sized
        seed=42,               # Reproducible
    )

    print(f"Light curve: {len(t)} points over {t[-1]:.1f} days")
    print(f"Transit duration T_14 = {t14*24*60:.1f} minutes")
    print(f"TTV amplitude A_TTV = {0.04*24*60:.1f} minutes")
    print(f"Ratio A_TTV/T_14 = {0.04/t14:.2f}")

    # Standard BLS search
    res_std = transit_search(t, flux, flux_err, min_period=2, max_period=10)
    best_std = res_std[np.argmax(res_std['sde'])]
    print(f"\nStandard BLS: Period = {best_std['period']:.3f} d, SDE = {best_std['sde']:.1f}")

    # TTV-BLS search (with known TTV parameters)
    t_corr = distort_timebase(t, epoch=0, p_ttv=60.0, a_ttv=0.04, e_ttv=0.0)
    res_ttv = transit_search(t_corr, flux, flux_err, min_period=2, max_period=10)
    best_ttv = res_ttv[np.argmax(res_ttv['sde'])]
    print(f"TTV-BLS: Period = {best_ttv['period']:.3f} d, SDE = {best_ttv['sde']:.1f}")

    improvement = best_ttv['sde'] / best_std['sde']
    print(f"\nImprovement factor: {improvement:.2f}x")

    return res_std, res_ttv


def example_critical_threshold():
    """Demonstrate the critical threshold A_TTV/T_14 ~ 0.5-0.7."""
    print("\n=== Example 2: Critical Threshold Demonstration ===\n")

    print(f"Critical threshold: A_TTV/T_14 = {CRITICAL_THRESHOLD} - 0.7")
    print("\nScanning TTV amplitudes...\n")

    # Fixed parameters
    period = 10.0
    r_planet = 0.05
    a_rs = 15.0
    t14 = get_transit_duration(period, r_planet, a_rs)

    print(f"Transit duration T_14 = {t14*24*60:.1f} minutes")
    print(f"Critical A_TTV = {CRITICAL_THRESHOLD * t14 * 24 * 60:.1f} - {0.7 * t14 * 24 * 60:.1f} minutes")
    print()

    # Test different ratios
    ratios = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]

    print(f"{'Ratio':>8} | {'A_TTV (min)':>12} | {'SDE_std':>8} | {'SDE_ttv':>8} | {'Advantage':>10}")
    print("-" * 60)

    for ratio in ratios:
        a_ttv = ratio * t14

        t, flux, flux_err, _ = create_lightcurve(
            cadence=120, duration=365, period=period, epoch=5.0,
            a_ttv=a_ttv, p_ttv=120.0, count_rate=5000, r_planet=r_planet,
            a_rs=a_rs, seed=42,
        )

        res_std = transit_search(t, flux, flux_err, min_period=5, max_period=20)
        sde_std = np.max(res_std['sde'])

        t_corr = distort_timebase(t, epoch=0, p_ttv=120.0, a_ttv=a_ttv, e_ttv=0.0)
        res_ttv = transit_search(t_corr, flux, flux_err, min_period=5, max_period=20)
        sde_ttv = np.max(res_ttv['sde'])

        advantage = sde_ttv / sde_std if sde_std > 0 else np.inf

        marker = " <-- threshold" if 0.5 <= ratio <= 0.7 else ""
        print(f"{ratio:>8.2f} | {a_ttv*24*60:>12.1f} | {sde_std:>8.1f} | {sde_ttv:>8.1f} | {advantage:>9.2f}x{marker}")


def example_plot_periodogram():
    """Create and save a periodogram comparison plot."""
    print("\n=== Example 3: Periodogram Comparison Plot ===\n")

    # High TTV case
    t, flux, flux_err, t14 = create_lightcurve(
        cadence=120, duration=365, period=8.0, epoch=4.0,
        a_ttv=0.08, p_ttv=100.0, count_rate=8000, r_planet=0.06, seed=42,
    )

    # Standard BLS
    res_std = transit_search(t, flux, flux_err, min_period=3, max_period=15)

    # TTV-BLS
    t_corr = distort_timebase(t, epoch=0, p_ttv=100.0, a_ttv=0.08, e_ttv=0.0)
    res_ttv = transit_search(t_corr, flux, flux_err, min_period=3, max_period=15)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(res_std['period'], res_std['sde'], 'r-', alpha=0.7)
    ax1.axhline(7, ls='--', color='gray', alpha=0.5)
    ax1.axvline(8.0, ls=':', color='blue', alpha=0.5, label='True period')
    ax1.set_ylabel('SDE')
    ax1.set_title('Standard BLS')
    ax1.legend()

    ax2.plot(res_ttv['period'], res_ttv['sde'], 'b-', alpha=0.7)
    ax2.axhline(7, ls='--', color='gray', alpha=0.5)
    ax2.axvline(8.0, ls=':', color='blue', alpha=0.5)
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel('SDE')
    ax2.set_title('TTV-BLS')

    plt.tight_layout()
    plt.savefig('periodogram_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: periodogram_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("TTV-BLS Quickstart Examples")
    print("=" * 60)

    example_basic_usage()
    example_critical_threshold()
    example_plot_periodogram()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
