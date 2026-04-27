#!/usr/bin/env python3
"""
Create empirical truncation figure from actual Article 5 simulation data.

Uses CONSOLIDATED_RESULTS.json which has injection-recovery results across
different periods (250-550 days) corresponding to different transit counts
in a 4-year baseline.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_simulation_data():
    """Load the consolidated results from Article 5 simulations."""
    results_path = Path(__file__).parent.parent.parent.parent / \
        'simulations/melendo/article5_simulations/CONSOLIDATED_RESULTS.json'

    if not results_path.exists():
        # Try alternative path
        results_path = Path('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article5_simulations/CONSOLIDATED_RESULTS.json')

    with open(results_path) as f:
        data = json.load(f)

    return data['injection_recovery']['individual']


def period_to_transits(period_days, baseline_years=4):
    """Convert orbital period to expected number of transits."""
    baseline_days = baseline_years * 365.25
    return baseline_days / period_days


def analyze_by_transit_count(results):
    """Group results by transit count and A_TTV/T14 ratio."""

    # Define bins for A_TTV/T14
    ratio_bins = [
        (0, 0.3, 'low'),
        (0.3, 0.6, 'medium'),
        (0.6, 1.2, 'high'),
        (1.2, 2.0, 'very_high')
    ]

    # Group by period (proxy for transit count)
    period_groups = {}
    for r in results:
        period = r['period']
        ratio = r['a_ttv_t14']

        if period not in period_groups:
            period_groups[period] = []
        period_groups[period].append(r)

    # Calculate statistics by period and ratio bin
    stats = []
    for period, group in sorted(period_groups.items()):
        n_transits = period_to_transits(period)

        for low, high, label in ratio_bins:
            subset = [r for r in group if low <= r['a_ttv_t14'] < high]
            if subset:
                improvements = [r['improvement'] for r in subset]
                bls_comp = [r['bls_completeness'] for r in subset]
                ttv_comp = [r['ttv_completeness'] for r in subset]

                stats.append({
                    'period': period,
                    'n_transits': n_transits,
                    'ratio_label': label,
                    'ratio_low': low,
                    'ratio_high': high,
                    'mean_improvement': np.mean(improvements),
                    'std_improvement': np.std(improvements),
                    'mean_bls_comp': np.mean(bls_comp),
                    'mean_ttv_comp': np.mean(ttv_comp),
                    'n_samples': len(subset)
                })

    return stats


def create_empirical_figure(stats, output_dir='.'):
    """Create empirical truncation figure from simulation data."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract unique values
    periods = sorted(set(s['period'] for s in stats))
    n_transits_list = [period_to_transits(p) for p in periods]

    # Panel A: Improvement vs transit count for different A_TTV/T14 ratios
    ax1 = axes[0]

    ratio_labels = ['medium', 'high', 'very_high']
    colors = {'medium': 'blue', 'high': 'orange', 'very_high': 'red'}
    markers = {'medium': 'o', 'high': 's', 'very_high': '^'}
    labels = {'medium': '$A_{TTV}/T_{14}$ = 0.3-0.6',
              'high': '$A_{TTV}/T_{14}$ = 0.6-1.2',
              'very_high': '$A_{TTV}/T_{14}$ > 1.2'}

    for ratio_label in ratio_labels:
        subset = [s for s in stats if s['ratio_label'] == ratio_label]
        if subset:
            x = [s['n_transits'] for s in subset]
            y = [s['mean_improvement'] for s in subset]
            yerr = [s['std_improvement'] / np.sqrt(s['n_samples']) for s in subset]

            ax1.errorbar(x, y, yerr=yerr, fmt=markers[ratio_label]+'-',
                        color=colors[ratio_label], linewidth=2, markersize=8,
                        capsize=3, label=labels[ratio_label])

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvspan(2, 4, alpha=0.15, color='green', label='HZ regime')

    ax1.set_xlabel('Number of Transits (4-year baseline)', fontsize=12)
    ax1.set_ylabel('Completeness Improvement (%)', fontsize=12)
    ax1.set_title('(a) TTV-BLS Improvement vs Transit Count\n(Empirical from Simulations)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(2, 7)
    ax1.grid(True, alpha=0.3)

    # Panel B: Average completeness comparison (high A_TTV/T14 only)
    ax2 = axes[1]

    high_ratio = [s for s in stats if s['ratio_label'] in ['high', 'very_high']]

    # Group by period for cleaner plotting
    by_period = {}
    for s in high_ratio:
        p = s['period']
        if p not in by_period:
            by_period[p] = {'bls': [], 'ttv': [], 'n_tr': s['n_transits']}
        by_period[p]['bls'].append(s['mean_bls_comp'])
        by_period[p]['ttv'].append(s['mean_ttv_comp'])

    x_vals = []
    bls_vals = []
    ttv_vals = []
    for p in sorted(by_period.keys()):
        x_vals.append(by_period[p]['n_tr'])
        bls_vals.append(np.mean(by_period[p]['bls']))
        ttv_vals.append(np.mean(by_period[p]['ttv']))

    ax2.plot(x_vals, bls_vals, 'o-', color='blue', linewidth=2, markersize=8,
             label='Standard BLS')
    ax2.plot(x_vals, ttv_vals, 's-', color='orange', linewidth=2, markersize=8,
             label='TTV-BLS')
    ax2.fill_between(x_vals, bls_vals, ttv_vals, alpha=0.2, color='orange')
    ax2.axvspan(2, 4, alpha=0.15, color='green', label='HZ regime')

    ax2.set_xlabel('Number of Transits (4-year baseline)', fontsize=12)
    ax2.set_ylabel('Detection Completeness (%)', fontsize=12)
    ax2.set_title('(b) Completeness at High $A_{TTV}/T_{14}$ (> 0.6)\n(Empirical from Simulations)',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(2, 7)
    ax2.set_ylim(0, max(max(bls_vals), max(ttv_vals)) * 1.2 + 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(f'{output_dir}/truncation_empirical.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/truncation_empirical.pdf', bbox_inches='tight')
    print(f"Saved empirical figure to {output_dir}/truncation_empirical.png/pdf")

    plt.close()

    return stats


def print_summary(stats):
    """Print summary statistics for the paper."""
    print("\n=== Empirical Truncation Analysis ===\n")

    # Few-transit regime (N < 4)
    few_transit = [s for s in stats if s['n_transits'] < 4 and s['ratio_label'] in ['high', 'very_high']]
    # Many-transit regime (N >= 5)
    many_transit = [s for s in stats if s['n_transits'] >= 5 and s['ratio_label'] in ['high', 'very_high']]

    if few_transit:
        avg_few = np.mean([s['mean_improvement'] for s in few_transit])
        print(f"Few-transit regime (N < 4, A_TTV/T14 > 0.6):")
        print(f"  Average improvement: {avg_few:+.1f}%")
        print(f"  N samples: {sum(s['n_samples'] for s in few_transit)}")

    if many_transit:
        avg_many = np.mean([s['mean_improvement'] for s in many_transit])
        print(f"\nMany-transit regime (N >= 5, A_TTV/T14 > 0.6):")
        print(f"  Average improvement: {avg_many:+.1f}%")
        print(f"  N samples: {sum(s['n_samples'] for s in many_transit)}")

    if few_transit and many_transit:
        ratio = avg_few / avg_many if avg_many != 0 else float('inf')
        print(f"\nRatio (few/many): {ratio:.1f}x")

    print("\n=== By Period (Transit Count) ===")
    print(f"{'Period':>8} {'N_tr':>6} {'Ratio':>12} {'BLS%':>8} {'TTV%':>8} {'Imp%':>8} {'N':>4}")
    print("-" * 60)

    for s in sorted(stats, key=lambda x: (x['period'], x['ratio_low'])):
        print(f"{s['period']:>8} {s['n_transits']:>6.1f} {s['ratio_label']:>12} "
              f"{s['mean_bls_comp']:>8.1f} {s['mean_ttv_comp']:>8.1f} "
              f"{s['mean_improvement']:>+8.1f} {s['n_samples']:>4}")


if __name__ == '__main__':
    print("Loading simulation data...")
    results = load_simulation_data()
    print(f"Loaded {len(results)} individual results")

    print("\nAnalyzing by transit count...")
    stats = analyze_by_transit_count(results)

    print_summary(stats)

    print("\nCreating figure...")
    create_empirical_figure(stats, '.')
