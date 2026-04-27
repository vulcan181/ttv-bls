#!/usr/bin/env python3
"""
Quick test script for Article I TTV-BLS demonstration.

This script demonstrates the key finding: TTV-BLS provides significant
advantage when A_TTV / T_14 > 0.5-0.7.
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
    compute_a_ttv_over_t14,
)


def main():
    print("=" * 60)
    print("TTV-BLS Quick Test - Article I Demonstration")
    print("=" * 60)

    # Parameters
    cadence = 120  # 2-minute cadence (seconds)
    duration = 365  # 1 year (days)
    period = 10.0  # Orbital period (days)
    epoch = 5.0
    r_planet = 0.05  # Neptune-sized
    a_rs = 15.0
    p_ttv = 120.0  # TTV period (days)
    count_rate = 5000  # Moderate SNR

    # Get transit duration
    t14 = get_transit_duration(period, r_planet, a_rs)
    print(f"\nTransit duration T_14 = {t14*24*60:.1f} minutes")

    # Test different TTV amplitudes
    a_ttv_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]  # days

    results = []

    print("\nRunning TTV amplitude sweep...")
    print("-" * 60)

    for a_ttv in a_ttv_values:
        ratio = compute_a_ttv_over_t14(a_ttv, t14)

        # Create light curve with TTV
        t, flux, flux_err, _ = create_lightcurve(
            cadence=cadence,
            duration=duration,
            period=period,
            epoch=epoch,
            a_ttv=a_ttv,
            p_ttv=p_ttv,
            e_ttv=0.0,
            count_rate=count_rate,
            r_planet=r_planet,
            a_rs=a_rs,
            seed=42,
        )

        # Standard BLS
        res_std = transit_search(t, flux, flux_err, min_period=5, max_period=20)
        sde_std = np.max(res_std['sde'])

        # TTV-BLS (with correct parameters)
        t_corr = distort_timebase(t, epoch=0, p_ttv=p_ttv, a_ttv=a_ttv, e_ttv=0.0)
        res_ttv = transit_search(t_corr, flux, flux_err, min_period=5, max_period=20)
        sde_ttv = np.max(res_ttv['sde'])

        advantage = sde_ttv / sde_std if sde_std > 0 else np.inf

        results.append({
            'a_ttv': a_ttv,
            'ratio': ratio,
            'sde_std': sde_std,
            'sde_ttv': sde_ttv,
            'advantage': advantage,
        })

        print(f"A_TTV = {a_ttv*24*60:5.1f} min | A_TTV/T_14 = {ratio:.2f} | "
              f"SDE_std = {sde_std:5.1f} | SDE_ttv = {sde_ttv:5.1f} | "
              f"Advantage = {advantage:.2f}x")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ratios = [r['ratio'] for r in results]
    sde_std = [r['sde_std'] for r in results]
    sde_ttv = [r['sde_ttv'] for r in results]
    advantages = [r['advantage'] for r in results]

    # Left panel: SDE vs A_TTV/T_14
    ax1.plot(ratios, sde_std, 'o-', label='Standard BLS', color='red')
    ax1.plot(ratios, sde_ttv, 's-', label='TTV-BLS', color='blue')
    ax1.axhline(7, ls='--', color='gray', label='Detection threshold (SDE=7)')
    ax1.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Critical threshold')
    ax1.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax1.set_ylabel('SDE', fontsize=12)
    ax1.legend()
    ax1.set_title('Detection Significance vs TTV Amplitude')
    ax1.grid(True, alpha=0.3)

    # Right panel: Advantage factor
    ax2.plot(ratios, advantages, 'o-', color='green', markersize=8)
    ax2.axhline(1, ls='--', color='gray', label='Parity')
    ax2.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Critical threshold')
    ax2.set_xlabel(r'$A_{\rm TTV} / T_{14}$', fontsize=12)
    ax2.set_ylabel('Advantage Factor (TTV-BLS / Standard)', fontsize=12)
    ax2.legend()
    ax2.set_title('TTV-BLS Advantage Factor')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: quick_test_results.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCritical threshold region: A_TTV/T_14 = 0.5 - 0.7")
    print(f"Transit duration T_14 = {t14*24*60:.1f} minutes")
    print(f"Critical A_TTV range = {0.5*t14*24*60:.1f} - {0.7*t14*24*60:.1f} minutes")

    # Find where advantage > 2
    for r in results:
        if r['advantage'] > 2:
            print(f"\nFirst significant advantage (>2x) at A_TTV/T_14 = {r['ratio']:.2f}")
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
