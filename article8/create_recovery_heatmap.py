#!/usr/bin/env python3
"""
Create recovery rate heatmap from existing injection-recovery results.

Analyzes the 240,000 simulation results and produces a heatmap showing
recovery rate (within factor 2) in (N_transits, sigma_t) space.

Output: paper/figures/fig_recovery_heatmap.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Load all result files from the results directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for result_file in results_path.glob("config_*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def aggregate_by_n_tr_sigma_t(results, target_mass=30.0, target_e=0.0):
    """
    Aggregate results by (N_transits, sigma_t) for a specific mass and eccentricity.

    Returns dict mapping (n_tr, sigma_t) -> recovery_rate
    """
    aggregated = defaultdict(lambda: {'n_configs': 0, 'recovery_sum': 0})

    for r in results:
        config = r.get('config', {})
        m2 = config.get('m2_earth')
        e2 = config.get('e2')
        n_tr = config.get('n_transits')
        sigma_t = config.get('sigma_t')

        # Filter by mass and eccentricity
        if m2 is None or e2 is None:
            continue
        if abs(m2 - target_mass) > 0.1:
            continue
        if abs(e2 - target_e) > 0.01:
            continue

        key = (n_tr, sigma_t)
        aggregated[key]['n_configs'] += 1
        aggregated[key]['recovery_sum'] += r.get('recovery_rate_factor2', 0)

    # Compute average recovery rates
    recovery_map = {}
    for key, data in aggregated.items():
        if data['n_configs'] > 0:
            recovery_map[key] = data['recovery_sum'] / data['n_configs']

    return recovery_map


def create_heatmap(results, output_path, target_mass=30.0, target_e=0.0):
    """Create recovery rate heatmap figure."""

    recovery_map = aggregate_by_n_tr_sigma_t(results, target_mass, target_e)

    if not recovery_map:
        print(f"No data found for mass={target_mass}, e={target_e}")
        return

    # Extract unique values
    n_transits_vals = sorted(set(k[0] for k in recovery_map.keys()))
    sigma_t_vals = sorted(set(k[1] for k in recovery_map.keys()))

    print(f"N_transits values: {n_transits_vals}")
    print(f"sigma_t values: {sigma_t_vals}")

    # Create 2D array
    recovery_grid = np.full((len(sigma_t_vals), len(n_transits_vals)), np.nan)

    for i, sigma_t in enumerate(sigma_t_vals):
        for j, n_tr in enumerate(n_transits_vals):
            if (n_tr, sigma_t) in recovery_map:
                recovery_grid[i, j] = recovery_map[(n_tr, sigma_t)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot heatmap
    im = ax.imshow(
        recovery_grid * 100,  # Convert to percentage
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0, vmax=50,
        extent=[0, len(n_transits_vals), 0, len(sigma_t_vals)]
    )

    # Add contours
    X, Y = np.meshgrid(np.arange(len(n_transits_vals)) + 0.5,
                       np.arange(len(sigma_t_vals)) + 0.5)
    contours = ax.contour(X, Y, recovery_grid * 100,
                          levels=[25, 50, 75],
                          colors=['white', 'yellow', 'red'],
                          linewidths=2)
    ax.clabel(contours, inline=True, fontsize=12, fmt='%d%%')

    # Configure axes
    ax.set_xticks(np.arange(len(n_transits_vals)) + 0.5)
    ax.set_xticklabels(n_transits_vals, fontsize=12)
    ax.set_yticks(np.arange(len(sigma_t_vals)) + 0.5)
    ax.set_yticklabels(sigma_t_vals, fontsize=12)

    ax.set_xlabel('Number of Transits', fontsize=14)
    ax.set_ylabel(r'Timing Precision $\sigma_t$ (min)', fontsize=14)
    ax.set_title(f'Mass Recovery Rate (within factor 2)\n'
                 f'$m_2 = {target_mass}$ $M_\\oplus$, $e = {target_e}$',
                 fontsize=14)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Recovery Rate (%)')
    cbar.ax.tick_params(labelsize=12)

    # Add annotations for key regions
    ax.annotate('Noise\nLimited', xy=(0.5, 4.5), fontsize=12, color='white',
                ha='center', va='center', weight='bold')
    ax.annotate('PLATO\nRegime', xy=(4.5, 0.5), fontsize=12, color='black',
                ha='center', va='center', weight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    # Also save PNG version
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved PNG to {png_path}")

    plt.close(fig)

    return recovery_map


def create_multi_mass_heatmap(results, output_path):
    """Create multi-panel heatmap showing all masses."""

    masses = [1, 3, 10, 30, 100]
    target_e = 0.0

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for ax, mass in zip(axes, masses):
        recovery_map = aggregate_by_n_tr_sigma_t(results, mass, target_e)

        if not recovery_map:
            ax.set_title(f'{mass} $M_\\oplus$\nNo data')
            continue

        # Extract unique values
        n_transits_vals = sorted(set(k[0] for k in recovery_map.keys()))
        sigma_t_vals = sorted(set(k[1] for k in recovery_map.keys()))

        # Create 2D array
        recovery_grid = np.full((len(sigma_t_vals), len(n_transits_vals)), np.nan)

        for i, sigma_t in enumerate(sigma_t_vals):
            for j, n_tr in enumerate(n_transits_vals):
                if (n_tr, sigma_t) in recovery_map:
                    recovery_grid[i, j] = recovery_map[(n_tr, sigma_t)]

        # Plot heatmap
        im = ax.imshow(
            recovery_grid * 100,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=0, vmax=50,
            extent=[0, len(n_transits_vals), 0, len(sigma_t_vals)]
        )

        # Add contours
        X, Y = np.meshgrid(np.arange(len(n_transits_vals)) + 0.5,
                           np.arange(len(sigma_t_vals)) + 0.5)
        if not np.all(np.isnan(recovery_grid)):
            contours = ax.contour(X, Y, recovery_grid * 100,
                                  levels=[25, 50],
                                  colors=['white', 'yellow'],
                                  linewidths=1.5)

        # Configure axes
        ax.set_xticks(np.arange(len(n_transits_vals)) + 0.5)
        ax.set_xticklabels(n_transits_vals, fontsize=10, rotation=45)
        ax.set_yticks(np.arange(len(sigma_t_vals)) + 0.5)
        ax.set_yticklabels(sigma_t_vals, fontsize=10)

        ax.set_xlabel('$N_{\\rm tr}$', fontsize=12)
        ax.set_title(f'{mass} $M_\\oplus$', fontsize=12)

    axes[0].set_ylabel(r'$\sigma_t$ (min)', fontsize=12)

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Recovery Rate (%)')

    fig.suptitle('Mass Recovery Rate (within factor 2) at $e = 0$', fontsize=14, y=1.02)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-panel figure to {output_path}")

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')

    plt.close(fig)


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "injection_recovery"
    figures_dir = script_dir / "paper" / "figures"

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")

    if len(results) == 0:
        print("No results found. Checking alternative locations...")
        # Try the configs directory structure
        alt_results_dir = script_dir / "results"
        for subdir in alt_results_dir.iterdir():
            if subdir.is_dir():
                results.extend(load_results(subdir))
        print(f"Found {len(results)} results in alternative locations")

    if len(results) == 0:
        print("\nNo results found. Please ensure simulation results are available.")
        print("Expected location: results/injection_recovery/config_*_results.json")
        return

    # Create single-mass heatmap (30 M_Earth, e=0)
    print("\nCreating single-mass heatmap (30 M_Earth)...")
    create_heatmap(
        results,
        figures_dir / "fig_recovery_heatmap.pdf",
        target_mass=30.0,
        target_e=0.0
    )

    # Create multi-mass panel figure
    print("\nCreating multi-mass heatmap...")
    create_multi_mass_heatmap(
        results,
        figures_dir / "fig_recovery_heatmap_all_masses.pdf"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
