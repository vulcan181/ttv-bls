#!/usr/bin/env python3
"""
Red Noise Validation Tests for Article VI (Article 7)

Option B: Compute β factor directly from O-C time series using time-averaging.
Tests whether consecutive transit times have correlated errors.

Author: Stamatis Kalogerakos
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "red_noise_results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_oc_timeseries(name):
    """Load O-C time series from JSON file."""
    # Try exact match first
    filename = FIGURES_DIR / f"oc_timeseries_{name.replace(' ', '_')}.json"
    if filename.exists():
        with open(filename) as f:
            return json.load(f)

    # Try without spaces
    filename = FIGURES_DIR / f"oc_timeseries_{name.replace(' ', '')}.json"
    if filename.exists():
        with open(filename) as f:
            return json.load(f)

    return None


def compute_beta_from_oc(oc_values, epochs):
    """
    Compute β factor from O-C time series using time-averaging.

    For white noise: RMS of binned O-C decreases as 1/√N
    For red (correlated) noise: it decreases slower
    β = measured_RMS(N) / expected_RMS(N)

    We bin CONSECUTIVE transits to test for temporal correlation.
    """
    oc = np.array(oc_values)
    ep = np.array(epochs)

    # Sort by epoch
    sort_idx = np.argsort(ep)
    oc = oc[sort_idx]
    ep = ep[sort_idx]

    # Find consecutive epoch groups (where epochs differ by 1)
    # This is important because non-consecutive transits shouldn't be binned together
    consecutive_groups = []
    current_group = [0]

    for i in range(1, len(ep)):
        if ep[i] - ep[i-1] == 1:
            current_group.append(i)
        else:
            if len(current_group) >= 2:
                consecutive_groups.append(current_group)
            current_group = [i]

    if len(current_group) >= 2:
        consecutive_groups.append(current_group)

    # Unbinned RMS (baseline)
    sigma_1 = np.std(oc)

    # Compute β for different bin sizes
    bin_sizes = [2, 3, 4, 5]
    beta_values = []
    diagnostics = {
        'sigma_1': sigma_1,
        'n_oc': len(oc),
        'n_consecutive_groups': len(consecutive_groups),
        'bin_results': []
    }

    for N in bin_sizes:
        # Collect binned values from consecutive groups
        binned_values = []

        for group in consecutive_groups:
            group_oc = oc[group]
            # Bin in groups of N
            n_bins = len(group_oc) // N
            for i in range(n_bins):
                bin_mean = np.mean(group_oc[i*N:(i+1)*N])
                binned_values.append(bin_mean)

        if len(binned_values) < 3:
            continue

        # Measured RMS of binned values
        sigma_N = np.std(binned_values)

        # Expected RMS for white noise
        sigma_white = sigma_1 / np.sqrt(N)

        # β factor
        beta_N = sigma_N / sigma_white if sigma_white > 0 else 1.0

        beta_values.append(beta_N)
        diagnostics['bin_results'].append({
            'N': N,
            'n_bins': len(binned_values),
            'sigma_N': sigma_N,
            'sigma_white': sigma_white,
            'beta_N': beta_N
        })

    # Final β = max(1.0, median of β values)
    if len(beta_values) > 0:
        beta = max(1.0, np.median(beta_values))
    else:
        beta = 1.0

    diagnostics['beta'] = beta
    diagnostics['beta_values'] = beta_values

    return beta, diagnostics


def analyze_candidate(name, data):
    """Analyze one candidate for red noise."""
    oc = data['oc_minutes']
    epochs = data['epochs']
    oc_err = data.get('oc_err_minutes', [0.5] * len(oc))

    # Compute β
    beta, diagnostics = compute_beta_from_oc(oc, epochs)

    # Original σ_TTV (from paper methodology)
    # Paper uses fixed expected_noise = 0.5 minutes (conservative baseline)
    oc_rms = np.std(oc)
    expected_noise = 0.5  # minutes - matches run_diagnostic.py
    sigma_ttv_original = oc_rms / expected_noise

    # β-corrected σ_TTV
    # If red noise present, the effective noise floor is higher
    corrected_noise = expected_noise * beta
    sigma_ttv_corrected = oc_rms / corrected_noise

    # Also store median per-transit uncertainty for reference
    median_sigma_t = np.median(oc_err)

    return {
        'name': name,
        'n_transits': len(oc),
        'oc_rms_min': oc_rms,
        'expected_noise_min': expected_noise,
        'median_sigma_t_min': median_sigma_t,
        'beta': beta,
        'sigma_ttv_original': sigma_ttv_original,
        'sigma_ttv_corrected': sigma_ttv_corrected,
        'still_significant': sigma_ttv_corrected > 2.0,
        'diagnostics': diagnostics
    }


def create_beta_diagnostic_plot(result, output_dir):
    """Create diagnostic plot showing σ(N) vs N."""
    name = result['name']
    diag = result['diagnostics']

    if len(diag.get('bin_results', [])) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    Ns = [r['N'] for r in diag['bin_results']]
    sigma_Ns = [r['sigma_N'] for r in diag['bin_results']]
    sigma_whites = [r['sigma_white'] for r in diag['bin_results']]

    # Plot measured σ_N
    ax.scatter(Ns, sigma_Ns, s=100, c='blue', label='Measured σ(N)', zorder=3)
    ax.plot(Ns, sigma_Ns, 'b-', alpha=0.5)

    # Plot expected white noise
    N_line = np.linspace(1, max(Ns) + 1, 50)
    sigma_1 = diag['sigma_1']
    ax.plot(N_line, sigma_1 / np.sqrt(N_line), 'r--', lw=2,
            label=f'White noise: σ₁/√N (σ₁={sigma_1:.2f} min)')

    ax.set_xlabel('Bin size N (consecutive transits)', fontsize=12)
    ax.set_ylabel('RMS of binned O-C (minutes)', fontsize=12)
    ax.set_title(f'{name}: β = {result["beta"]:.2f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max(Ns) + 0.5)
    ax.set_ylim(0, sigma_1 * 1.5)

    # Add text annotation
    textstr = f'N_transits = {result["n_transits"]}\nσ_TTV: {result["sigma_ttv_original"]:.1f} → {result["sigma_ttv_corrected"]:.1f}'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    figpath = output_dir / f"beta_diagnostic_{name.replace(' ', '_')}.png"
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {figpath.name}")


def create_summary_plot(results, output_dir):
    """Create summary bar chart of original vs β-corrected σ_TTV."""
    if len(results) == 0:
        return

    # Sort by original σ_TTV
    results = sorted(results, key=lambda x: -x['sigma_ttv_original'])

    names = [r['name'].replace(' b', '') for r in results]
    original = [r['sigma_ttv_original'] for r in results]
    corrected = [r['sigma_ttv_corrected'] for r in results]
    betas = [r['beta'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, original, width, label='Original σ_TTV', color='steelblue')
    bars2 = ax.bar(x + width/2, corrected, width, label='β-corrected σ_TTV', color='coral')

    ax.axhline(2, color='red', ls='--', lw=2, label='Detection threshold (σ=2)')

    # Add β values as text
    for i, (bar, beta) in enumerate(zip(bars2, betas)):
        ax.annotate(f'β={beta:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    ax.set_xlabel('System', fontsize=12)
    ax.set_ylabel('σ_TTV', fontsize=12)
    ax.set_title('TTV Significance: Original vs β-Corrected (Red Noise Test)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    figpath = output_dir / "beta_summary.png"
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {figpath}")


def main():
    print("=" * 70)
    print("RED NOISE VALIDATION: β FACTOR FROM O-C TIME SERIES")
    print("=" * 70)
    print()

    # Find all O-C time series files
    oc_files = list(FIGURES_DIR.glob("oc_timeseries_*.json"))
    print(f"Found {len(oc_files)} O-C time series files")

    if len(oc_files) == 0:
        print("ERROR: No O-C time series files found in figures/")
        return

    # Analyze each candidate
    results = []

    print("\nAnalyzing candidates:")
    print("-" * 50)

    for oc_file in sorted(oc_files):
        with open(oc_file) as f:
            data = json.load(f)

        name = data['name']
        print(f"\n{name}:")
        print(f"  N_transits = {len(data['oc_minutes'])}")
        print(f"  O-C RMS = {np.std(data['oc_minutes']):.2f} min")

        result = analyze_candidate(name, data)
        results.append(result)

        print(f"  β = {result['beta']:.2f}")
        print(f"  σ_TTV: {result['sigma_ttv_original']:.1f} → {result['sigma_ttv_corrected']:.1f} "
              f"{'✓' if result['still_significant'] else '✗'}")

    # Create diagnostic plots
    print("\n" + "-" * 50)
    print("Creating diagnostic plots...")
    for result in results:
        create_beta_diagnostic_plot(result, RESULTS_DIR)

    # Create summary plot
    create_summary_plot(results, RESULTS_DIR)

    # Save results table
    df = pd.DataFrame([{
        'System': r['name'],
        'N_transits': r['n_transits'],
        'O-C_RMS_min': round(r['oc_rms_min'], 2),
        'beta': round(r['beta'], 2),
        'sigma_TTV_original': round(r['sigma_ttv_original'], 1),
        'sigma_TTV_corrected': round(r['sigma_ttv_corrected'], 1),
        'Still_significant': r['still_significant'],
    } for r in results])

    csv_path = RESULTS_DIR / "beta_factor_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Print table
    print("\n" + "=" * 70)
    print("β FACTOR RESULTS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_survive = sum(1 for r in results if r['still_significant'])
    n_total = len(results)
    median_beta = np.median([r['beta'] for r in results])

    print(f"\n1. SURVIVABILITY: {n_survive}/{n_total} candidates remain significant (σ_TTV > 2)")
    print(f"\n2. MEDIAN β = {median_beta:.2f}")

    if median_beta > 1.5:
        print("   → Significant red noise detected in consecutive transits")
    elif median_beta > 1.2:
        print("   → Modest red noise detected")
    else:
        print("   → Little to no red noise (timing errors approximately white)")

    print(f"\n3. CONCLUSION:")
    if n_survive == n_total:
        print("   The red noise analysis SUPPORTS all companion candidate detections.")
    elif n_survive >= n_total * 0.8:
        print("   The red noise analysis STRONGLY SUPPORTS the companion candidate detections.")
    elif n_survive >= n_total * 0.5:
        print("   The red noise analysis PARTIALLY SUPPORTS the companion candidate detections.")
    else:
        print("   The red noise analysis DOES NOT SUPPORT the companion candidate detections.")

    # List survivors
    print(f"\n4. SURVIVING CANDIDATES ({n_survive}):")
    for r in sorted(results, key=lambda x: -x['sigma_ttv_corrected']):
        if r['still_significant']:
            print(f"   {r['name']:15s} σ_TTV = {r['sigma_ttv_corrected']:.1f} (β={r['beta']:.2f})")

    # Save full results JSON
    full_results = {
        'results': results,
        'summary': {
            'n_survive': n_survive,
            'n_total': n_total,
            'median_beta': float(median_beta),
        }
    }

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    json_path = RESULTS_DIR / "beta_factor_full_results.json"
    with open(json_path, 'w') as f:
        json.dump(convert_numpy(full_results), f, indent=2)
    print(f"\nFull results saved to {json_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
