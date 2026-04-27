#!/usr/bin/env python3
"""
Create the key figure showing mass-eccentricity degeneracy validation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.switch_backend('Agg')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

def load_results(results_dir):
    """Load all results."""
    results_dir = Path(results_dir)
    results = []
    for f in sorted(results_dir.glob("config_*_results.json")):
        with open(f, 'r') as fp:
            results.append(json.load(fp))
    return results


def theoretical_bias(e, Delta=0.05, f=-1.19):
    """
    Compute theoretical mass overestimate from Lithwick formula.

    Assumes Z_free ~ e (simplified) and random phase.
    """
    # Z_free magnitude scales with e
    Z_free_mag = e * 0.5  # Rough scaling

    # The term in Lithwick formula
    # |term| = |-f - 3*Z_free/(2*Delta)|
    # For random phase, average |term| ≈ sqrt(f² + (3*Z_free/(2*Delta))²)

    extra_term = 3 * Z_free_mag / (2 * Delta)
    term_mag = np.sqrt(f**2 + extra_term**2)

    return term_mag / abs(f)


def main():
    # Load results
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / 'injection_recovery'
    output_dir = base_dir / 'results' / 'analysis'

    results = load_results(results_dir)

    # Aggregate by eccentricity
    by_ecc = defaultdict(list)
    for r in results:
        if r['mass_ratio_median'] is not None:
            e = r['config']['e2']
            by_ecc[e].append(r['mass_ratio_median'])

    eccs = sorted(by_ecc.keys())
    medians = [np.median(by_ecc[e]) for e in eccs]
    q25 = [np.percentile(by_ecc[e], 25) for e in eccs]
    q75 = [np.percentile(by_ecc[e], 75) for e in eccs]

    # Theoretical prediction
    e_theory = np.linspace(0, 0.25, 100)
    bias_theory = [theoretical_bias(e) for e in e_theory]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel a: Mass ratio vs eccentricity
    ax = axes[0]
    ax.fill_between(eccs, q25, q75, alpha=0.3, color='steelblue', label='IQR')
    ax.plot(eccs, medians, 'o-', color='steelblue', markersize=10, linewidth=2, label='Simulation median')
    ax.plot(e_theory, bias_theory, '--', color='darkred', linewidth=2, label='Lithwick theory')
    ax.axhline(y=1, color='green', linestyle=':', linewidth=2, label='Perfect recovery')
    ax.axhline(y=2, color='orange', linestyle=':', alpha=0.7)
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7)

    ax.set_xlabel('Companion Eccentricity $e_2$')
    ax.set_ylabel('Recovered / True Mass Ratio')
    ax.set_title('(a) Mass-Eccentricity Degeneracy Validation')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    ax.text(0.15, 8, 'Lithwick+2012\nprediction:\nBias $\\propto$ 1/|$\\Delta$|',
            fontsize=10, ha='center', style='italic')

    # Panel b: Detection and recovery rates by mass
    ax = axes[1]

    by_mass = defaultdict(lambda: {'det': [], 'rec': []})
    for r in results:
        m = r['config']['m2_earth']
        by_mass[m]['det'].append(r['detection_rate'])
        by_mass[m]['rec'].append(r['recovery_rate_factor2'])

    masses = sorted(by_mass.keys())
    det_rates = [np.mean(by_mass[m]['det']) * 100 for m in masses]
    rec_rates = [np.mean(by_mass[m]['rec']) * 100 for m in masses]

    x = np.arange(len(masses))
    width = 0.35

    bars1 = ax.bar(x - width/2, det_rates, width, label='TTV Detection', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, rec_rates, width, label='Mass Recovery (×2)', color='coral', alpha=0.8)

    ax.set_xlabel('Companion Mass ($M_\\oplus$)')
    ax.set_ylabel('Rate (%)')
    ax.set_title('(b) Detection Requires Massive Companions')
    ax.set_xticks(x)
    ax.set_xticklabels(masses)
    ax.legend()
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, det_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, rec_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'key_figure_degeneracy.png', dpi=150)
    plt.savefig(output_dir / 'key_figure_degeneracy.pdf')
    plt.close()

    print(f"Saved key figure to {output_dir / 'key_figure_degeneracy.png'}")

    # Create additional figure: N_transits threshold
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate by (mass, n_transits)
    by_mass_ntr = defaultdict(lambda: defaultdict(list))
    for r in results:
        m = r['config']['m2_earth']
        ntr = r['config']['n_transits']
        by_mass_ntr[m][ntr].append(r['detection_rate'])

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(masses)))

    for m, color in zip(masses, colors):
        ntrs = sorted(by_mass_ntr[m].keys())
        rates = [np.mean(by_mass_ntr[m][n]) * 100 for n in ntrs]
        ax.plot(ntrs, rates, 'o-', color=color, linewidth=2, markersize=8, label=f'{m} $M_\\oplus$')

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Number of Transits')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('TTV Detection Rate: The 50-Transit Threshold')
    ax.legend(title='Companion Mass')
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3)

    # Shade the "good" region
    ax.axvspan(50, 210, alpha=0.1, color='green')
    ax.text(130, 70, 'Detectable\nregime', fontsize=12, ha='center', color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_dir / 'ntransits_threshold.png', dpi=150)
    plt.savefig(output_dir / 'ntransits_threshold.pdf')
    plt.close()

    print(f"Saved N_transits threshold figure")


if __name__ == "__main__":
    main()
