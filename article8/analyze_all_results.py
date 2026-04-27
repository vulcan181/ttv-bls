#!/usr/bin/env python3
"""
Comprehensive analysis of all Article 8 simulation results.

Generates:
1. Extended degeneracy table (e=0.0 to e=0.30)
2. Chopping contour map figure
3. Recovery heatmap figure
4. Bias calibration analysis
5. Second-order resonance comparison
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Load all result files from a directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for result_file in sorted(results_path.glob("config_*_results.json")):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def analyze_eccentricity_extension(results_dir, output_file):
    """Analyze eccentricity extension results and print degeneracy table."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    # Group by eccentricity
    by_ecc = defaultdict(list)
    for r in results:
        config = r.get('config', {})
        e2 = config.get('e2')
        if e2 is not None:
            mass_ratio = r.get('mass_ratio_median')
            if mass_ratio is not None:
                by_ecc[e2].append(mass_ratio)

    print("\n=== ECCENTRICITY EXTENSION RESULTS ===")
    print("Eccentricity | Mass Ratio (median) | Std | N")
    print("-" * 50)

    with open(output_file, 'w') as f:
        f.write("# Eccentricity Extension Results\n")
        f.write("# e, mass_ratio_median, mass_ratio_std, n_configs\n")

        for e in sorted(by_ecc.keys()):
            ratios = by_ecc[e]
            median = np.median(ratios)
            std = np.std(ratios)
            print(f"e = {e:.2f}    |  {median:.2f} +/- {std:.2f}  | {std:.2f} | {len(ratios)}")
            f.write(f"{e:.2f}, {median:.2f}, {std:.2f}, {len(ratios)}\n")

    print(f"\nResults saved to {output_file}")


def create_chopping_contour(results_dir, output_file):
    """Create chopping detectability contour map."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    # Group by mass
    masses = [1, 3, 10, 30, 100]
    n_transits_vals = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    sigma_t_vals = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for ax, target_mass in zip(axes, masses):
        # Build detection rate grid
        detection_grid = np.full((len(sigma_t_vals), len(n_transits_vals)), np.nan)

        for r in results:
            config = r.get('config', {})
            m2 = config.get('m2_earth')
            n_tr = config.get('n_transits')
            sigma_t = config.get('sigma_t')

            if m2 != target_mass:
                continue

            try:
                i = sigma_t_vals.index(sigma_t)
                j = n_transits_vals.index(n_tr)
                detection_grid[i, j] = r.get('detection_rate', 0)
            except ValueError:
                continue

        # Plot
        im = ax.imshow(
            detection_grid * 100,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=0, vmax=100,
            extent=[0, len(n_transits_vals), 0, len(sigma_t_vals)]
        )

        # Add contours
        if not np.all(np.isnan(detection_grid)):
            X, Y = np.meshgrid(np.arange(len(n_transits_vals)) + 0.5,
                               np.arange(len(sigma_t_vals)) + 0.5)
            try:
                contours = ax.contour(X, Y, detection_grid * 100,
                                      levels=[25, 50, 75],
                                      colors=['white', 'yellow', 'red'],
                                      linewidths=1.5)
                ax.clabel(contours, inline=True, fontsize=10, fmt='%d%%')
            except:
                pass

        ax.set_xticks(np.arange(len(n_transits_vals)) + 0.5)
        ax.set_xticklabels(n_transits_vals, fontsize=9, rotation=45)
        ax.set_yticks(np.arange(len(sigma_t_vals)) + 0.5)
        ax.set_yticklabels(sigma_t_vals, fontsize=9)
        ax.set_xlabel('$N_{\\rm tr}$', fontsize=12)
        ax.set_title(f'{target_mass} $M_\\oplus$', fontsize=12)

    axes[0].set_ylabel('$\\sigma_t$ (min)', fontsize=12)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Detection Rate (%)')

    fig.suptitle('Chopping Signal Detectability (2:1 resonance, e=0)', fontsize=14, y=1.02)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chopping contour saved to {output_file}")


def create_recovery_heatmap(results_dir, output_file, target_mass=30, target_e=0.0):
    """Create recovery rate heatmap for a specific mass and eccentricity."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    n_transits_vals = sorted(set(r['config']['n_transits'] for r in results if 'config' in r))
    sigma_t_vals = sorted(set(r['config']['sigma_t'] for r in results if 'config' in r))

    recovery_grid = np.full((len(sigma_t_vals), len(n_transits_vals)), np.nan)

    for r in results:
        config = r.get('config', {})
        m2 = config.get('m2_earth')
        e2 = config.get('e2')
        n_tr = config.get('n_transits')
        sigma_t = config.get('sigma_t')

        if m2 != target_mass or abs(e2 - target_e) > 0.01:
            continue

        try:
            i = sigma_t_vals.index(sigma_t)
            j = n_transits_vals.index(n_tr)
            recovery_grid[i, j] = r.get('recovery_rate_factor2', 0)
        except ValueError:
            continue

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        recovery_grid * 100,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0, vmax=50,
        extent=[0, len(n_transits_vals), 0, len(sigma_t_vals)]
    )

    # Add contours
    if not np.all(np.isnan(recovery_grid)):
        X, Y = np.meshgrid(np.arange(len(n_transits_vals)) + 0.5,
                           np.arange(len(sigma_t_vals)) + 0.5)
        try:
            contours = ax.contour(X, Y, recovery_grid * 100,
                                  levels=[10, 25, 50],
                                  colors=['white', 'yellow', 'red'],
                                  linewidths=2)
            ax.clabel(contours, inline=True, fontsize=12, fmt='%d%%')
        except:
            pass

    ax.set_xticks(np.arange(len(n_transits_vals)) + 0.5)
    ax.set_xticklabels(n_transits_vals, fontsize=12)
    ax.set_yticks(np.arange(len(sigma_t_vals)) + 0.5)
    ax.set_yticklabels(sigma_t_vals, fontsize=12)

    ax.set_xlabel('Number of Transits', fontsize=14)
    ax.set_ylabel('Timing Precision $\\sigma_t$ (min)', fontsize=14)
    ax.set_title(f'Mass Recovery Rate (within factor 2)\n$m_2 = {target_mass}$ $M_\\oplus$, $e = {target_e}$', fontsize=14)

    cbar = fig.colorbar(im, ax=ax, label='Recovery Rate (%)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Recovery heatmap saved to {output_file}")


def analyze_bias_calibration(results_dir, output_file):
    """Analyze systematic bias at e=0."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print("\n=== BIAS CALIBRATION RESULTS (e=0, m2=30 M_Earth) ===")
    print("Period Ratio | N_tr | sigma_t | Mass Ratio | Detection")
    print("-" * 60)

    # Group by period ratio
    by_pr = defaultdict(list)

    for r in results:
        config = r.get('config', {})
        pr = config.get('P2_over_P1')
        n_tr = config.get('n_transits')
        sigma_t = config.get('sigma_t')
        mass_ratio = r.get('mass_ratio_median')
        det_rate = r.get('detection_rate', 0)

        if mass_ratio is not None:
            by_pr[pr].append({
                'n_tr': n_tr,
                'sigma_t': sigma_t,
                'mass_ratio': mass_ratio,
                'det_rate': det_rate
            })

    with open(output_file, 'w') as f:
        f.write("# Bias Calibration Results\n")
        f.write("# P2/P1, n_transits, sigma_t, mass_ratio, detection_rate\n")

        for pr in sorted(by_pr.keys()):
            for entry in sorted(by_pr[pr], key=lambda x: (x['n_tr'], x['sigma_t'])):
                print(f"{pr:.2f}        | {entry['n_tr']:3d}  | {entry['sigma_t']:.1f}     | {entry['mass_ratio']:.2f}       | {entry['det_rate']*100:.0f}%")
                f.write(f"{pr:.2f}, {entry['n_tr']}, {entry['sigma_t']}, {entry['mass_ratio']:.3f}, {entry['det_rate']:.3f}\n")

    # Summary
    all_ratios = [e['mass_ratio'] for entries in by_pr.values() for e in entries]
    print(f"\nOverall bias at e=0: {np.median(all_ratios):.2f} +/- {np.std(all_ratios):.2f}")
    print(f"Results saved to {output_file}")


def analyze_second_order(results_dir, output_file):
    """Compare first-order vs second-order resonance performance."""
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print("\n=== SECOND-ORDER RESONANCE RESULTS ===")

    # Group by period ratio
    by_pr = defaultdict(lambda: {'det': [], 'rec': []})

    for r in results:
        config = r.get('config', {})
        pr = config.get('P2_over_P1')
        det_rate = r.get('detection_rate', 0)
        rec_rate = r.get('recovery_rate_factor2', 0)

        by_pr[pr]['det'].append(det_rate)
        by_pr[pr]['rec'].append(rec_rate)

    print("Period Ratio | Resonance | Detection | Recovery (x2)")
    print("-" * 55)

    resonance_names = {1.40: '7:5', 1.67: '5:3', 3.05: '3:1'}

    with open(output_file, 'w') as f:
        f.write("# Second-Order Resonance Results\n")
        f.write("# P2/P1, resonance, detection_rate, recovery_rate\n")

        for pr in sorted(by_pr.keys()):
            det_mean = np.mean(by_pr[pr]['det'])
            rec_mean = np.mean(by_pr[pr]['rec'])
            res_name = resonance_names.get(pr, '?')
            print(f"{pr:.2f}        | {res_name:5s}     | {det_mean*100:5.1f}%    | {rec_mean*100:5.1f}%")
            f.write(f"{pr:.2f}, {res_name}, {det_mean:.3f}, {rec_mean:.3f}\n")

    print(f"\nResults saved to {output_file}")


def main():
    base_dir = Path(__file__).parent.parent
    sim_dir = base_dir / "simulations"
    figures_dir = base_dir / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARTICLE 8 COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # 1. Eccentricity extension
    print("\n[1/5] Analyzing eccentricity extension...")
    analyze_eccentricity_extension(
        sim_dir / "eccentricity_extension" / "results",
        base_dir / "eccentricity_extension_summary.txt"
    )

    # 2. Chopping contour map
    print("\n[2/5] Creating chopping contour map...")
    create_chopping_contour(
        sim_dir / "chopping_contour" / "results",
        str(figures_dir / "fig_chopping_contour.pdf")
    )

    # 3. Recovery heatmap
    print("\n[3/5] Creating recovery heatmap...")
    create_recovery_heatmap(
        sim_dir / "chopping_contour" / "results",
        str(figures_dir / "fig_recovery_heatmap.pdf"),
        target_mass=30,
        target_e=0.0
    )

    # 4. Bias calibration
    print("\n[4/5] Analyzing bias calibration...")
    analyze_bias_calibration(
        sim_dir / "bias_calibration" / "results",
        base_dir / "bias_calibration_summary.txt"
    )

    # 5. Second-order resonances
    print("\n[5/5] Analyzing second-order resonances...")
    analyze_second_order(
        sim_dir / "second_order_resonances" / "results",
        base_dir / "second_order_summary.txt"
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_dir}")
    print(f"Summary files saved to: {base_dir}")


if __name__ == "__main__":
    main()
