#!/usr/bin/env python3
"""
Analyze Set A: TTV Amplitude Distributions from N-body integrations

Creates empirical TTV amplitude lookup tables as function of:
- Period ratio (P_outer/P_inner)
- Mass ratio (M_outer/M_inner)
- Eccentricity
- Resonance proximity (delta_res)

Output: TTV lookup table for use in monotransit/duo-transit period inference
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/simulations/melendo/article6_simulations/set_a_ttv_amplitude_distributions/results")
OUTPUT_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article6/results")
FIGURES_DIR = Path("/mnt/c/users/stamatis/Dropbox/Personal2/WARWICK/results/article6/paper/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_all_results():
    """Load all Set A results into a DataFrame"""
    results = []
    result_files = list(RESULTS_DIR.glob("result_*.json"))
    print(f"Loading {len(result_files)} result files...")

    for f in result_files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            # Extract key fields
            config = data.get('config', {})
            results.append({
                'run_name': data.get('run_name'),
                'period_ratio': config.get('period_ratio'),
                'mass_ratio': config.get('mass_ratio'),
                'eccentricity': config.get('eccentricity', 0),
                'delta_res': config.get('delta_res'),
                'P_inner': config.get('P_inner'),
                'P_outer': config.get('P_outer'),
                'M_inner_earth': config.get('M_inner_earth'),
                'M_outer_earth': config.get('M_outer_earth'),
                'attv': data.get('attv'),  # TTV amplitude in days
                'attv_minutes': data.get('attv_minutes'),
                'pttv': data.get('pttv'),  # TTV period
                'n_transits': data.get('n_transits'),
                'stable': data.get('stable', True),
                'energy_error': data.get('energy_error'),
                'oc_rms': data.get('oc_rms'),
                'oc_max': data.get('oc_max'),
                'analysis_status': data.get('analysis_status'),
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} results")
    return df

def compute_statistics(df):
    """Compute summary statistics for the TTV amplitude distributions"""
    # Filter stable simulations with successful analysis
    stable = df[(df['stable'] == True) & (df['analysis_status'] == 'success')]
    print(f"Stable simulations: {len(stable)}/{len(df)}")

    # Get unique parameter values
    period_ratios = sorted(stable['period_ratio'].unique())
    mass_ratios = sorted(stable['mass_ratio'].unique())
    delta_res_values = sorted(stable['delta_res'].unique())

    print(f"Period ratios: {len(period_ratios)} values from {min(period_ratios):.2f} to {max(period_ratios):.2f}")
    print(f"Mass ratios: {len(mass_ratios)} values from {min(mass_ratios):.4f} to {max(mass_ratios):.4f}")
    print(f"Delta res: {len(delta_res_values)} values from {min(delta_res_values):.4f} to {max(delta_res_values):.4f}")

    # Overall statistics
    stats = {
        'n_total': len(df),
        'n_stable': len(stable),
        'stability_fraction': len(stable) / len(df) if len(df) > 0 else 0,
        'attv_mean_days': stable['attv'].mean(),
        'attv_median_days': stable['attv'].median(),
        'attv_std_days': stable['attv'].std(),
        'attv_min_days': stable['attv'].min(),
        'attv_max_days': stable['attv'].max(),
        'attv_mean_minutes': stable['attv_minutes'].mean(),
        'attv_median_minutes': stable['attv_minutes'].median(),
        'pttv_mean_days': stable['pttv'].mean(),
        'pttv_median_days': stable['pttv'].median(),
    }

    return stats, stable

def create_lookup_table(df):
    """Create TTV amplitude lookup table grouped by parameters"""
    # Group by period ratio, mass ratio, delta_res
    groups = df.groupby(['period_ratio', 'mass_ratio', 'delta_res']).agg({
        'attv': ['mean', 'std', 'min', 'max', 'count'],
        'attv_minutes': ['mean', 'std'],
        'pttv': ['mean', 'std'],
        'stable': 'sum',
    }).reset_index()

    # Flatten column names
    groups.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in groups.columns]

    return groups

def plot_attv_vs_period_ratio(df, output_path):
    """Plot TTV amplitude vs period ratio"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter by mass ratio
    mass_ratios = sorted(df['mass_ratio'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(mass_ratios)))

    for mr, color in zip(mass_ratios, colors):
        subset = df[df['mass_ratio'] == mr]
        axes[0].scatter(subset['period_ratio'], subset['attv_minutes'],
                       c=[color], alpha=0.5, s=10, label=f'MR={mr:.2f}')

    axes[0].set_xlabel('Period Ratio (P_outer/P_inner)')
    axes[0].set_ylabel('TTV Amplitude (minutes)')
    axes[0].set_title('TTV Amplitude vs Period Ratio')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=6, ncol=2)

    # Right: median by period ratio
    median_by_pr = df.groupby('period_ratio')['attv_minutes'].median()
    axes[1].plot(median_by_pr.index, median_by_pr.values, 'ko-')
    axes[1].set_xlabel('Period Ratio (P_outer/P_inner)')
    axes[1].set_ylabel('Median TTV Amplitude (minutes)')
    axes[1].set_title('Median TTV vs Period Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_resonance_enhancement(df, output_path):
    """Plot TTV amplitude enhancement near resonances"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by delta_res and period_ratio
    for pr in sorted(df['period_ratio'].unique())[:5]:  # First 5 period ratios
        subset = df[df['period_ratio'] == pr]
        grouped = subset.groupby('delta_res')['attv_minutes'].median()
        ax.plot(grouped.index, grouped.values, 'o-', label=f'PR={pr:.2f}')

    ax.set_xlabel('Delta from Resonance')
    ax.set_ylabel('Median TTV Amplitude (minutes)')
    ax.set_title('TTV Enhancement Near Mean-Motion Resonances')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("="*60)
    print("Article 6 - Set A Analysis: TTV Amplitude Distributions")
    print("="*60)

    # Load results
    df = load_all_results()

    # Compute statistics
    stats, stable_df = compute_statistics(df)

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Create lookup table
    lookup = create_lookup_table(stable_df)

    # Save outputs
    lookup.to_csv(OUTPUT_DIR / "ttv_amplitude_lookup_table.csv", index=False)
    print(f"\nSaved lookup table: {OUTPUT_DIR / 'ttv_amplitude_lookup_table.csv'}")

    with open(OUTPUT_DIR / "set_a_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {OUTPUT_DIR / 'set_a_statistics.json'}")

    # Save full data for further analysis
    stable_df.to_csv(OUTPUT_DIR / "set_a_full_results.csv", index=False)
    print(f"Saved full results: {OUTPUT_DIR / 'set_a_full_results.csv'}")

    # Generate figures
    plot_attv_vs_period_ratio(stable_df, FIGURES_DIR / "fig_attv_vs_period_ratio.png")
    plot_resonance_enhancement(stable_df, FIGURES_DIR / "fig_resonance_enhancement.png")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return stats, stable_df, lookup

if __name__ == "__main__":
    stats, df, lookup = main()
