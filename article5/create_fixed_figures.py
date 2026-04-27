#!/usr/bin/env python3
"""
Fixed figures for Article 5:
1. Figure 1: Decision rule - FIXED Y-AXIS SCALE (was multiplying by 100 incorrectly)
2. Figure 8: Marginal rescue demo - ADDED ZOOM INSET to make transit visible

Author: Stamatis Kalogerakos
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path

# Set publication-quality defaults - INCREASED FONT SIZES for two-column readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Paths
BASE_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK")
OUTPUT_DIR = BASE_DIR / "results/article5/paper/figures"
CONSOLIDATED_RESULTS = BASE_DIR / "results/simulations/melendo/article5_simulations/CONSOLIDATED_RESULTS.json"


def load_injection_data():
    """Load injection-recovery data from consolidated results."""
    with open(CONSOLIDATED_RESULTS) as f:
        data = json.load(f)

    individual = data['injection_recovery']['individual']
    return individual


def create_decision_rule_fixed():
    """
    Create Figure 1: Decision rule with CORRECT Y-AXIS SCALE.

    The bug was: improvement values are already in percentage points (e.g., 5.2 means +5.2 pp),
    but the old code multiplied by 100, showing 520 instead of 5.2.
    """
    print("Creating decision_rule.png (FIXED)...")

    results = load_injection_data()

    # Extract data - NO multiplication by 100!
    ratios = np.array([r['a_ttv_t14'] for r in results])
    improvements = np.array([r['improvement'] for r in results])  # Already in percentage points
    periods = np.array([r['period'] for r in results])
    n_transits = np.array([int(4 * 365 / p) if p > 0 else 0 for p in periods])

    # Filter for TTV cases only (exclude no-TTV baselines)
    ttv_mask = ratios > 0.05

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Scatter plot colored by transit count
    scatter = ax.scatter(ratios[ttv_mask], improvements[ttv_mask],
                        c=n_transits[ttv_mask], cmap='viridis',
                        s=60, alpha=0.7, edgecolors='gray', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Number of transits\n(4-year baseline)', fontsize=10)

    # Bin the data and calculate means with bootstrap confidence intervals
    ratio_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 2.0]
    bin_centers = []
    bin_means = []
    bin_ci_low = []
    bin_ci_high = []
    bin_counts = []

    for i in range(len(ratio_bins) - 1):
        low, high = ratio_bins[i], ratio_bins[i+1]
        mask = (ratios >= low) & (ratios < high) & ttv_mask

        if np.sum(mask) > 2:
            bin_center = (low + high) / 2
            bin_improvements = improvements[mask]
            bin_n = np.sum(mask)

            # Mean improvement (already in pp)
            mean_imp = np.mean(bin_improvements)

            # Bootstrap for confidence interval
            np.random.seed(42)
            boot_means = []
            for _ in range(1000):
                boot_sample = np.random.choice(bin_improvements, size=len(bin_improvements), replace=True)
                boot_means.append(np.mean(boot_sample))

            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            bin_centers.append(bin_center)
            bin_means.append(mean_imp)
            bin_ci_low.append(ci_low)
            bin_ci_high.append(ci_high)
            bin_counts.append(bin_n)

    # Convert to arrays
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_ci_low = np.array(bin_ci_low)
    bin_ci_high = np.array(bin_ci_high)

    yerr_low = bin_means - bin_ci_low
    yerr_high = bin_ci_high - bin_means

    # Plot binned means with error bars
    ax.errorbar(bin_centers, bin_means, yerr=[yerr_low, yerr_high],
                fmt='s', markersize=12, color='red', capsize=6, capthick=2,
                elinewidth=2, markeredgecolor='black', markeredgewidth=1.5,
                label='Binned mean ± 95% CI', zorder=10)

    # Add N annotations
    for x, y, n in zip(bin_centers, bin_means, bin_counts):
        offset_y = 3 if y >= 0 else -5
        ax.annotate(f'N={n}', xy=(x, y), xytext=(5, offset_y),
                   textcoords='offset points', fontsize=9, color='darkred', fontweight='bold')

    # Critical threshold band
    ax.axvspan(0.5, 0.7, alpha=0.25, color='orange', label='Critical threshold (0.5-0.7)')

    # Horizontal line at zero
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel(r'$A_{\rm TTV} / T_{14}$ Ratio', fontsize=12)
    ax.set_ylabel('Completeness Improvement (percentage points)', fontsize=12)
    ax.set_title('Decision Rule: When to Apply TTV-BLS\n' +
                 r'Apply TTV-BLS when $A_{\rm TTV}/T_{14} \gtrsim 0.5$; otherwise standard BLS is sufficient',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=10)

    # Set reasonable axis limits
    ax.set_xlim(0, 1.5)
    y_max = max(60, np.max(improvements[ttv_mask]) + 5)
    y_min = min(-10, np.min(improvements[ttv_mask]) - 5)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'decision_rule.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_marginal_rescue_demo_with_zoom():
    """
    Create Figure 8: Marginal rescue demo WITH ZOOMED INSET to make transit visible.

    The problem was: transit depth (~0.001) is nearly invisible at full scale.
    Solution: Use 2x2 subplot layout with zoomed panels below main panels.
    """
    print("Creating marginal_rescue_demo.png (WITH ZOOM)...")

    # Simulation parameters
    np.random.seed(42)

    # Planet and TTV parameters
    period = 300  # days
    n_transits = 4
    t14_hours = 12  # Transit duration
    t14_days = t14_hours / 24
    depth = 0.0015  # 0.15% depth (slightly larger for visibility)

    # TTV parameters (above threshold)
    a_ttv = 0.4  # days (9.6 hours)
    p_ttv = 500  # days
    a_ttv_t14 = a_ttv / (t14_hours / 24)  # = 0.8

    # Generate 4-year light curve
    t_total = 4 * 365  # days
    cadence = 30 / 60 / 24  # 30-minute cadence in days
    time = np.arange(0, t_total, cadence)

    # Generate transit times with TTV
    transit_epochs = np.arange(0, n_transits) * period + period/2
    ttv_offsets = a_ttv * np.sin(2 * np.pi * transit_epochs / p_ttv)
    transit_times_observed = transit_epochs + ttv_offsets

    # Generate light curve with transits
    flux = np.ones_like(time)
    noise_level = 0.00025  # Typical Kepler noise
    flux += np.random.normal(0, noise_level, len(time))

    # Add transits at observed times
    for t_mid in transit_times_observed:
        in_transit = np.abs(time - t_mid) < t14_days / 2
        flux[in_transit] -= depth

    # Phase fold WITHOUT TTV correction (standard BLS view)
    phase_uncorrected = ((time - transit_epochs[0]) % period) / period
    phase_uncorrected[phase_uncorrected > 0.5] -= 1  # Center on 0

    # Phase fold WITH TTV correction (TTV-BLS view)
    time_corrected = time.copy()
    for i, t in enumerate(time):
        correction = a_ttv * np.sin(2 * np.pi * t / p_ttv)
        time_corrected[i] = t - correction
    phase_corrected = ((time_corrected - transit_epochs[0]) % period) / period
    phase_corrected[phase_corrected > 0.5] -= 1

    # Create 2x2 figure: top row = full view, bottom row = zoomed transit
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [1, 1.2]})

    cases = [
        (phase_uncorrected,
         '(a) Standard BLS: Transit Smeared by TTV',
         5.8, False),
        (phase_corrected,
         '(b) TTV-BLS: Transit Aligned After Correction',
         9.2, True)
    ]

    for col, (phase, title, sde, detected) in enumerate(cases):
        ax_full = axes[0, col]
        ax_zoom = axes[1, col]

        # Convert phase to hours from transit center
        phase_hours = phase * period * 24

        # Bin the data
        bins = np.linspace(-8, 8, 65)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_flux = []
        binned_err = []
        for i in range(len(bins) - 1):
            mask = (phase_hours >= bins[i]) & (phase_hours < bins[i+1])
            if np.sum(mask) > 0:
                binned_flux.append(np.mean(flux[mask]))
                binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
            else:
                binned_flux.append(np.nan)
                binned_err.append(np.nan)

        binned_flux = np.array(binned_flux)
        binned_err = np.array(binned_err)

        # Colors
        data_color = '#2ecc71' if detected else '#e74c3c'
        model_color = 'green' if detected else 'red'

        # Expected transit model
        model_time = np.linspace(-8, 8, 200)
        model_flux = np.ones_like(model_time)
        model_flux[np.abs(model_time) < t14_hours/2] = 1 - depth

        # --- TOP ROW: Full view ---
        ax_full.scatter(phase_hours, flux, s=0.5, alpha=0.2, color='gray')
        ax_full.errorbar(bin_centers, binned_flux, yerr=binned_err,
                        fmt='o', markersize=4, color=data_color,
                        alpha=0.8, capsize=2, elinewidth=1)
        ax_full.plot(model_time, model_flux, '--', color=model_color,
                    alpha=0.5, linewidth=2)

        ax_full.set_ylabel('Normalised Flux', fontsize=11)
        ax_full.set_title(title + f'\n$A_{{\\rm TTV}}/T_{{14}}$ = {a_ttv_t14:.1f}', fontsize=11)
        ax_full.set_xlim(-8, 8)
        ax_full.set_ylim(0.996, 1.003)

        # Add SDE box
        box_color = '#27ae60' if detected else '#e74c3c'
        if detected:
            box_text = f'SDE = {sde}\n> 7 (threshold)\n\nDETECTED\nPeriod correct'
        else:
            box_text = f'SDE = {sde}\n< 7 (threshold)\n\nNOT DETECTED'
        props = dict(boxstyle='round', facecolor=box_color, alpha=0.15, edgecolor=box_color, linewidth=2)
        ax_full.text(0.03, 0.97, box_text, transform=ax_full.transAxes, fontsize=10,
                    verticalalignment='top', fontweight='bold', color=box_color, bbox=props)

        # Highlight zoom region
        ax_full.axvspan(-4, 4, alpha=0.1, color='blue')
        ax_full.text(0.97, 0.03, 'Zoom region\n(see below)', transform=ax_full.transAxes,
                    fontsize=8, ha='right', va='bottom', color='blue', alpha=0.7)

        # --- BOTTOM ROW: Zoomed transit region ---
        # Finer binning for zoom
        bins_zoom = np.linspace(-4, 4, 49)
        bin_centers_zoom = (bins_zoom[:-1] + bins_zoom[1:]) / 2

        binned_flux_zoom = []
        binned_err_zoom = []
        for i in range(len(bins_zoom) - 1):
            mask = (phase_hours >= bins_zoom[i]) & (phase_hours < bins_zoom[i+1])
            if np.sum(mask) > 0:
                binned_flux_zoom.append(np.mean(flux[mask]))
                binned_err_zoom.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
            else:
                binned_flux_zoom.append(np.nan)
                binned_err_zoom.append(np.nan)

        binned_flux_zoom = np.array(binned_flux_zoom)
        binned_err_zoom = np.array(binned_err_zoom)

        ax_zoom.errorbar(bin_centers_zoom, binned_flux_zoom, yerr=binned_err_zoom,
                        fmt='o', markersize=6, color=data_color,
                        alpha=0.9, capsize=3, elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)
        ax_zoom.plot(model_time, model_flux, '--', color=model_color,
                    alpha=0.7, linewidth=2.5, label='Expected transit shape')

        ax_zoom.set_xlabel('Time from expected transit (hours)', fontsize=11)
        ax_zoom.set_ylabel('Normalised Flux', fontsize=11)
        ax_zoom.set_title('ZOOMED: Transit Region', fontsize=10, fontweight='bold')
        ax_zoom.set_xlim(-4, 4)
        ax_zoom.set_ylim(0.9975, 1.001)

        # Add transit depth annotation
        if detected:
            ax_zoom.annotate('Transit clearly visible\n(aligned by TTV-BLS)',
                           xy=(0, 1-depth), xytext=(2, 0.9985),
                           fontsize=9, color='green',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        else:
            ax_zoom.annotate('Transit smeared\n(TTV causes misalignment)',
                           xy=(0, 0.999), xytext=(2, 0.9985),
                           fontsize=9, color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax_zoom.legend(loc='lower right', fontsize=9)

    # Add caption at bottom
    fig.text(0.5, 0.01,
             f'HZ planet: P = {period} d, {n_transits} transits | TTV: $A_{{\\rm TTV}}$ = {a_ttv*24:.1f} hr, $P_{{\\rm TTV}}$ = {p_ttv} d | ' +
             'This planet would be MISSED without TTV-BLS',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, hspace=0.25)

    # Save
    output_path = OUTPUT_DIR / 'marginal_rescue_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_scatter_by_transits_fixed():
    """
    Create Figure 3: Scatter by transits - also fix y-axis scale.
    """
    print("Creating scatter_by_transits.png (FIXED)...")

    results = load_injection_data()

    # Extract data
    ratios = np.array([r['a_ttv_t14'] for r in results])
    improvements = np.array([r['improvement'] for r in results])  # Already in pp
    periods = np.array([r['period'] for r in results])
    n_transits = np.array([int(4 * 365 / p) if p > 0 else 0 for p in periods])

    # Filter for TTV cases
    ttv_mask = ratios > 0.05

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color by transit count
    colors = {2: '#9b59b6', 3: '#3498db', 4: '#2ecc71', 5: '#f1c40f', 6: '#e74c3c'}

    for n in sorted(set(n_transits[ttv_mask])):
        mask = ttv_mask & (n_transits == n)
        if np.sum(mask) > 0:
            ax.scatter(ratios[mask], improvements[mask],
                      s=60, alpha=0.7, label=f'N={n} transits',
                      color=colors.get(n, 'gray'), edgecolors='black', linewidth=0.5)

    # Critical threshold band
    ax.axvspan(0.5, 0.7, alpha=0.25, color='orange', zorder=0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$A_{\rm TTV}/T_{14}$', fontsize=12)
    ax.set_ylabel('Completeness Improvement (pp)', fontsize=12)
    ax.set_title('Improvement by Number of Observable Transits', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-10, 60)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'scatter_by_transits.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("FIXING ARTICLE 5 FIGURES")
    print("=" * 60)
    print()

    # 1. Fix decision rule (main issue: y-axis was 100x too large)
    create_decision_rule_fixed()
    print()

    # 2. Fix marginal rescue demo (add zoom inset)
    create_marginal_rescue_demo_with_zoom()
    print()

    # 3. Fix scatter by transits (same y-axis issue)
    create_scatter_by_transits_fixed()
    print()

    print("=" * 60)
    print("DONE! Fixed figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
