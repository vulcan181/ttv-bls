#!/usr/bin/env python3
"""
Create figures for Article 5 based on feedback requirements.

Updated 2026-01-28 to add:
- Binomial confidence intervals
- N per cell annotations
- Statistical annotations per feedback items
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats


def wilson_ci(k, n, alpha=0.05):
    """
    Compute Wilson score binomial confidence interval.

    Parameters:
    -----------
    k : int - number of successes
    n : int - number of trials
    alpha : float - significance level (default 0.05 for 95% CI)

    Returns:
    --------
    (lower, upper) : tuple of CI bounds
    """
    if n == 0:
        return (0, 0)
    p = k / n
    z = stats.norm.ppf(1 - alpha/2)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))

# Set up matplotlib for publication - INCREASED FONT SIZES for two-column readability
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

# Load consolidated results
results_path = Path('/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article5_simulations/CONSOLIDATED_RESULTS.json')
with open(results_path) as f:
    data = json.load(f)

# Extract injection-recovery data
ir_data = data['injection_recovery']['individual']

# Parse into arrays
a_ttv_t14 = np.array([d['a_ttv_t14'] for d in ir_data])
bls_comp = np.array([d['bls_completeness'] for d in ir_data])
ttv_comp = np.array([d['ttv_completeness'] for d in ir_data])
improvement = np.array([d['improvement'] for d in ir_data])
periods = np.array([d['period'] for d in ir_data])
radii = np.array([d['radius'] for d in ir_data])

# Calculate number of transits (approximate for 4-year baseline)
n_transits = np.floor(4 * 365.25 / periods)

# ============================================================================
# Figure A1: Decision Rule Figure
# ============================================================================
def create_decision_rule_figure():
    """Create the headline 'decision rule' figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the critical threshold band (0.5-0.7)
    ax.axvspan(0.5, 0.7, alpha=0.3, color='orange', label='Critical threshold band')

    # Plot individual points
    scatter = ax.scatter(a_ttv_t14, improvement,
                        c=n_transits, cmap='viridis',
                        s=50, alpha=0.7, edgecolors='k', linewidths=0.5)

    # Add colorbar for number of transits
    cbar = plt.colorbar(scatter, ax=ax, label='Number of transits')

    # Bin the data and plot trend with N annotations
    bins = [0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.2, 2.0]
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_ns = []

    for i in range(len(bins)-1):
        mask = (a_ttv_t14 >= bins[i]) & (a_ttv_t14 < bins[i+1])
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(improvement[mask]))
            bin_stds.append(np.std(improvement[mask]))
            bin_ns.append(n_bin)

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
               fmt='s-', color='red', markersize=10, linewidth=2,
               capsize=5, label='Binned mean ± std', zorder=10)

    # Add N labels to bins
    for x, y, n in zip(bin_centers, bin_means, bin_ns):
        ax.annotate(f'N={n}', (x, y+8), ha='center', fontsize=8, color='darkred')

    # Add horizontal line at 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Add decision rule text
    ax.text(0.95, 0.95,
            r'$A_{\mathrm{TTV}}/T_{14} \gtrsim 0.5$: Use TTV-BLS' + '\n' +
            r'$A_{\mathrm{TTV}}/T_{14} \lesssim 0.5$: Standard BLS sufficient',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(r'$A_{\mathrm{TTV}} / T_{14}$')
    ax.set_ylabel('Completeness Improvement (percentage points)')
    ax.set_title('TTV-BLS Decision Rule: When to Apply TTV Correction')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-15, 60)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_rule.png', dpi=300, bbox_inches='tight')
    plt.savefig('decision_rule.pdf', dpi=300, bbox_inches='tight')
    print("Saved: decision_rule.png/pdf")
    plt.close()

# ============================================================================
# Figure A2a: Heatmap of completeness improvement
# ============================================================================
def create_heatmap_figure():
    """Create heatmap showing improvement across (Period, Radius) for high TTV."""
    # Filter for high TTV cases (A_TTV/T14 > 0.5)
    high_ttv_mask = a_ttv_t14 > 0.5

    # Get unique periods and radii
    unique_periods = sorted(set(periods))
    unique_radii = sorted(set(radii))

    # Create improvement matrix and count matrix
    improvement_matrix = np.zeros((len(unique_radii), len(unique_periods)))
    count_matrix = np.zeros((len(unique_radii), len(unique_periods)), dtype=int)
    # Also track BLS and TTV completeness for CI calculation
    bls_sum = np.zeros((len(unique_radii), len(unique_periods)))
    ttv_sum = np.zeros((len(unique_radii), len(unique_periods)))

    for i, p in enumerate(periods):
        if not high_ttv_mask[i]:
            continue
        pi = unique_periods.index(p)
        ri = unique_radii.index(radii[i])
        improvement_matrix[ri, pi] += improvement[i]
        count_matrix[ri, pi] += 1
        bls_sum[ri, pi] += bls_comp[i]
        ttv_sum[ri, pi] += ttv_comp[i]

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        improvement_matrix = np.where(count_matrix > 0,
                                      improvement_matrix / count_matrix,
                                      np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(improvement_matrix, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=-10, vmax=50)

    ax.set_xticks(range(len(unique_periods)))
    ax.set_xticklabels([f'{int(p)}' for p in unique_periods])
    ax.set_yticks(range(len(unique_radii)))
    ax.set_yticklabels([f'{r:.3f}' for r in unique_radii])

    ax.set_xlabel('Orbital Period (days)')
    ax.set_ylabel(r'$R_p/R_\star$')
    ax.set_title(r'Completeness Improvement for $A_{\mathrm{TTV}}/T_{14} > 0.5$')

    cbar = plt.colorbar(im, ax=ax, label='Improvement (pp)')

    # Add values and N per cell annotations
    for i in range(len(unique_radii)):
        for j in range(len(unique_periods)):
            val = improvement_matrix[i, j]
            n_cell = count_matrix[i, j]
            if not np.isnan(val) and n_cell > 0:
                color = 'white' if val > 25 or val < 0 else 'black'
                # Show improvement value and N
                ax.text(j, i, f'{val:.0f}\n(N={n_cell})', ha='center', va='center',
                       color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig('improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('improvement_heatmap.pdf', dpi=300, bbox_inches='tight')
    print("Saved: improvement_heatmap.png/pdf")
    plt.close()

# ============================================================================
# Figure A2b: Scatter by number of transits
# ============================================================================
def create_scatter_by_transits():
    """Create scatter plot of improvement vs A_TTV/T14 colored by N_transits."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade critical band
    ax.axvspan(0.5, 0.7, alpha=0.2, color='orange')

    # Create colormap for discrete transit numbers
    unique_n = sorted(set(n_transits))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_n)))

    for i, n in enumerate(unique_n):
        mask = n_transits == n
        ax.scatter(a_ttv_t14[mask], improvement[mask],
                  c=[colors[i]], s=40, alpha=0.7,
                  label=f'N={int(n)} transits', edgecolors='k', linewidths=0.3)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$A_{\mathrm{TTV}} / T_{14}$')
    ax.set_ylabel('Completeness Improvement (pp)')
    ax.set_title('Improvement by Number of Observable Transits')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scatter_by_transits.png', dpi=300, bbox_inches='tight')
    plt.savefig('scatter_by_transits.pdf', dpi=300, bbox_inches='tight')
    print("Saved: scatter_by_transits.png/pdf")
    plt.close()

# ============================================================================
# Run all figure generation
# ============================================================================
if __name__ == '__main__':
    print("Creating feedback figures...")
    create_decision_rule_figure()
    create_heatmap_figure()
    create_scatter_by_transits()
    print("Done!")
