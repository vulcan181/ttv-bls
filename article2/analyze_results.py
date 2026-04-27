#!/usr/bin/env python3
"""
Analyze Article 2 simulation results and generate figures.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article2_simulations")
OUTPUT_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article2/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (8, 6)


def load_all_results(study_name):
    """Load all results from a study."""
    results = []
    pattern = str(BASE_DIR / study_name / "results_*" / "results_summary.json")
    for filepath in glob.glob(pattern):
        with open(filepath) as f:
            results.append(json.load(f))
    return results


def analyze_nonsinusoidal():
    """Analyze non-sinusoidal TTV study results."""
    results = load_all_results("nonsinusoidal_study")
    print(f"\n=== Non-Sinusoidal Study ({len(results)} results) ===")

    for r in results:
        print(f"  {r['config']['resonance']:5s} A_TTV={r['config']['A_ttv']:.2f} "
              f"chop_frac={r['config']['A_chop_fraction']:.1f}: "
              f"BLS SDE={r['sde_bls']:.1f}, Sin SDE={r['sde_sin']:.1f}, Comp SDE={r['sde_comp']:.1f} "
              f"| Imp(sin)={r['improvement_sin_vs_none']:.1f}% Imp(comp)={r['improvement_comp_vs_none']:.1f}%")

    return results


def analyze_injection_recovery():
    """Analyze injection-recovery study results."""
    results = load_all_results("injection_recovery")
    print(f"\n=== Injection-Recovery Study ({len(results)} results) ===")

    for r in results:
        print(f"  {r['config']['target_name']:12s} TTV={r['config']['ttv_label']:6s} "
              f"A/T14={r['A_ttv_over_T14']:.2f}: "
              f"BLS SDE={r['sde_bls']:.1f}, TTV SDE={r['sde_ttv']:.1f} "
              f"| Improvement={r['improvement_percent']:.1f}%")

    return results


def analyze_plato_population():
    """Analyze PLATO population study results."""
    results = load_all_results("plato_population")
    print(f"\n=== PLATO Population Study ({len(results)} results) ===")

    # Aggregate across chunks
    total_planets = sum(r['total_planets'] for r in results)
    total_bls = sum(r['detectable_bls'] for r in results)
    total_ttv = sum(r['detectable_ttv'] for r in results)
    total_recovered = sum(r['recovered_by_ttv'] for r in results)

    print(f"  Total planets: {total_planets}")
    print(f"  Detectable by BLS: {total_bls} ({100*total_bls/total_planets:.1f}%)")
    print(f"  Detectable by TTV-BLS: {total_ttv} ({100*total_ttv/total_planets:.1f}%)")
    print(f"  Recovered by TTV-BLS (missed by BLS): {total_recovered}")
    print(f"  Additional detection rate: +{100*total_recovered/total_bls:.2f}%")

    # By planet type
    print("\n  By planet type:")
    types = ['hot_jupiter', 'warm_jupiter', 'hot_neptune', 'warm_neptune', 'super_earth', 'earth']
    type_stats = {t: {'total': 0, 'bls': 0, 'ttv': 0, 'recovered': 0} for t in types}

    for r in results:
        for t in types:
            if t in r['by_type']:
                type_stats[t]['total'] += r['by_type'][t]['total']
                type_stats[t]['bls'] += r['by_type'][t]['detectable_bls']
                type_stats[t]['ttv'] += r['by_type'][t]['detectable_ttv']
                type_stats[t]['recovered'] += r['by_type'][t]['recovered_by_ttv']

    for t in types:
        s = type_stats[t]
        if s['total'] > 0:
            print(f"    {t:15s}: {s['total']:4d} planets, BLS={s['bls']:4d}, TTV={s['ttv']:4d}, "
                  f"recovered={s['recovered']:3d}")

    return results, type_stats


def plot_injection_recovery(results):
    """Create injection-recovery improvement figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    a_over_t14 = [r['A_ttv_over_T14'] for r in results]
    improvement = [r['improvement_percent'] for r in results]
    targets = [r['config']['target_name'] for r in results]

    # Color by target
    colors = {'Kepler-9': 'blue', 'Kepler-10': 'red'}
    c = [colors.get(t, 'gray') for t in targets]

    ax.scatter(a_over_t14, improvement, c=c, s=100, alpha=0.7, edgecolors='black')

    # Add critical threshold region
    ax.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Critical threshold (Article 1)')

    # Add labels
    ax.set_xlabel(r'$A_{\mathrm{TTV}} / T_{14}$', fontsize=12)
    ax.set_ylabel('SDE Improvement (%)', fontsize=12)
    ax.set_title('Injection-Recovery: TTV-BLS Improvement', fontsize=14)

    # Add legend for targets
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Kepler-9 (high var)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Kepler-10 (control)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 300)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_injection_recovery.pdf', dpi=300)
    plt.savefig(OUTPUT_DIR / 'fig_injection_recovery.png', dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'fig_injection_recovery.pdf'}")
    plt.close()


def plot_nonsinusoidal(results):
    """Create non-sinusoidal correction figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Improvement vs A_TTV for different chopping fractions
    ax1 = axes[0]

    # Group by chopping fraction
    chop_0 = [r for r in results if r['config']['A_chop_fraction'] == 0.0]
    chop_03 = [r for r in results if r['config']['A_chop_fraction'] == 0.3]

    a_ttv_0 = [r['config']['A_ttv'] for r in chop_0]
    imp_0 = [r['improvement_sin_vs_none'] for r in chop_0]

    a_ttv_03 = [r['config']['A_ttv'] for r in chop_03]
    imp_03 = [r['improvement_comp_vs_none'] for r in chop_03]

    ax1.plot(a_ttv_0, imp_0, 'o-', label='Sinusoidal only', markersize=8)
    ax1.plot(a_ttv_03, imp_03, 's--', label='Sinusoidal + Chopping (30%)', markersize=8)

    ax1.set_xlabel(r'$A_{\mathrm{TTV}}$ (days)', fontsize=12)
    ax1.set_ylabel('SDE Improvement (%)', fontsize=12)
    ax1.set_title('TTV Correction Benefit', fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 0.2)

    # Right: Additional benefit from chopping
    ax2 = axes[1]

    # Calculate additional improvement from composite vs sinusoidal
    for r in chop_03:
        a_ttv = r['config']['A_ttv']
        chop_benefit = r['improvement_comp_vs_sin']
        resonance = r['config']['resonance']
        color = 'blue' if resonance == '2:1' else 'orange'
        ax2.bar(f"{a_ttv:.2f}\n({resonance})", chop_benefit, color=color, alpha=0.7)

    ax2.set_xlabel(r'$A_{\mathrm{TTV}}$ (days) / Resonance', fontsize=12)
    ax2.set_ylabel('Additional Improvement from Chopping (%)', fontsize=12)
    ax2.set_title('Chopping Correction Benefit', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_nonsinusoidal.pdf', dpi=300)
    plt.savefig(OUTPUT_DIR / 'fig_nonsinusoidal.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'fig_nonsinusoidal.pdf'}")
    plt.close()


def plot_plato_yield(results, type_stats):
    """Create PLATO yield prediction figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: By planet type
    ax1 = axes[0]
    types = ['super_earth', 'warm_neptune', 'hot_neptune', 'warm_jupiter', 'hot_jupiter']
    type_labels = ['Super-Earth', 'Warm Neptune', 'Hot Neptune', 'Warm Jupiter', 'Hot Jupiter']

    x = np.arange(len(types))
    width = 0.35

    bls_counts = [type_stats[t]['bls'] for t in types]
    ttv_counts = [type_stats[t]['ttv'] for t in types]

    ax1.bar(x - width/2, bls_counts, width, label='BLS', color='steelblue')
    ax1.bar(x + width/2, ttv_counts, width, label='TTV-BLS', color='coral')

    ax1.set_xlabel('Planet Type', fontsize=12)
    ax1.set_ylabel('Detectable Planets', fontsize=12)
    ax1.set_title('PLATO 2-Year Detection Yield by Type', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(type_labels, rotation=45, ha='right')
    ax1.legend()

    # Right: Recovered planets
    ax2 = axes[1]
    recovered = [type_stats[t]['recovered'] for t in types]
    colors = ['green' if r > 0 else 'gray' for r in recovered]

    ax2.bar(type_labels, recovered, color=colors, alpha=0.7)
    ax2.set_xlabel('Planet Type', fontsize=12)
    ax2.set_ylabel('Additional Planets Recovered by TTV-BLS', fontsize=12)
    ax2.set_title('Planets Missed by BLS, Recovered by TTV-BLS', fontsize=14)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_plato_yield.pdf', dpi=300)
    plt.savefig(OUTPUT_DIR / 'fig_plato_yield.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'fig_plato_yield.pdf'}")
    plt.close()


def main():
    print("=" * 60)
    print("Article 2 Results Analysis")
    print("=" * 60)

    # Analyze each study
    nonsine_results = analyze_nonsinusoidal()
    inject_results = analyze_injection_recovery()
    plato_results, plato_type_stats = analyze_plato_population()

    # Generate figures
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    if inject_results:
        plot_injection_recovery(inject_results)

    if nonsine_results:
        plot_nonsinusoidal(nonsine_results)

    if plato_results:
        plot_plato_yield(plato_results, plato_type_stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
