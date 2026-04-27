#!/usr/bin/env python3
"""
Analyze blended regime cases from existing simulation results.

Identifies configurations where the super-period and synodic period
are within 20% of each other, causing ambiguous peak classification.

This analysis requires NO new simulations - it reanalyzes existing results.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def compute_super_period(P1, P2_over_P1, j_R):
    """Compute the super-period for a given resonance."""
    P2 = P1 * P2_over_P1
    return abs(j_R / P2 - (j_R - 1) / P1) ** (-1)


def compute_synodic_period(P1, P2_over_P1):
    """Compute the synodic period."""
    P2 = P1 * P2_over_P1
    return abs(1/P1 - 1/P2) ** (-1)


def determine_j_R(P2_over_P1):
    """Determine the nearest first-order resonance."""
    j_R = round(P2_over_P1 / (P2_over_P1 - 1))
    return max(2, min(7, j_R))


def is_blended(P2_over_P1, P1=5.0, threshold=0.2):
    """
    Check if a period ratio leads to blended super/synodic periods.

    Returns True if |P_super - P_syn| / P_super < threshold
    """
    j_R = determine_j_R(P2_over_P1)
    P_super = compute_super_period(P1, P2_over_P1, j_R)
    P_syn = compute_synodic_period(P1, P2_over_P1)

    relative_diff = abs(P_super - P_syn) / P_super
    return relative_diff < threshold


def load_results(results_dir):
    """Load all result files from the results directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for result_file in results_path.glob("config_*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    return results


def analyze_blended_regime(results):
    """
    Analyze detection and recovery rates separately for blended vs non-blended cases.
    """
    blended_stats = defaultdict(lambda: {'n': 0, 'detection_sum': 0, 'recovery_sum': 0})
    non_blended_stats = defaultdict(lambda: {'n': 0, 'detection_sum': 0, 'recovery_sum': 0})

    for r in results:
        config = r.get('config', {})
        P2_over_P1 = config.get('P2_over_P1')
        e2 = config.get('e2')

        if P2_over_P1 is None or e2 is None:
            continue

        # Check if blended
        blended = is_blended(P2_over_P1)

        # Aggregate by eccentricity
        key = f"e={e2:.2f}"

        if blended:
            blended_stats[key]['n'] += 1
            blended_stats[key]['detection_sum'] += r.get('detection_rate', 0)
            blended_stats[key]['recovery_sum'] += r.get('recovery_rate_factor2', 0)
        else:
            non_blended_stats[key]['n'] += 1
            non_blended_stats[key]['detection_sum'] += r.get('detection_rate', 0)
            non_blended_stats[key]['recovery_sum'] += r.get('recovery_rate_factor2', 0)

    return blended_stats, non_blended_stats


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "results" / "injection_recovery"

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")

    if len(results) == 0:
        print("No results found. Checking alternative locations...")
        alt_results_dir = script_dir / "results"
        for subdir in alt_results_dir.iterdir():
            if subdir.is_dir():
                results.extend(load_results(subdir))
        print(f"Found {len(results)} results in alternative locations")

    if len(results) == 0:
        print("\nNo results found. Please ensure simulation results are available.")
        return

    # Analyze blended vs non-blended
    print("\nAnalyzing blended regime...")
    blended_stats, non_blended_stats = analyze_blended_regime(results)

    # Print results
    print("\n" + "="*60)
    print("BLENDED REGIME ANALYSIS")
    print("(Super-period and synodic period within 20%)")
    print("="*60)

    print("\n--- Blended Cases ---")
    for key in sorted(blended_stats.keys()):
        stats = blended_stats[key]
        if stats['n'] > 0:
            det_rate = stats['detection_sum'] / stats['n']
            rec_rate = stats['recovery_sum'] / stats['n']
            print(f"{key}: n={stats['n']:4d}, detection={det_rate*100:5.1f}%, recovery={rec_rate*100:5.1f}%")

    print("\n--- Non-Blended Cases ---")
    for key in sorted(non_blended_stats.keys()):
        stats = non_blended_stats[key]
        if stats['n'] > 0:
            det_rate = stats['detection_sum'] / stats['n']
            rec_rate = stats['recovery_sum'] / stats['n']
            print(f"{key}: n={stats['n']:4d}, detection={det_rate*100:5.1f}%, recovery={rec_rate*100:5.1f}%")

    # Summary comparison
    print("\n--- Summary ---")

    total_blended_n = sum(s['n'] for s in blended_stats.values())
    total_blended_det = sum(s['detection_sum'] for s in blended_stats.values())
    total_blended_rec = sum(s['recovery_sum'] for s in blended_stats.values())

    total_non_blended_n = sum(s['n'] for s in non_blended_stats.values())
    total_non_blended_det = sum(s['detection_sum'] for s in non_blended_stats.values())
    total_non_blended_rec = sum(s['recovery_sum'] for s in non_blended_stats.values())

    if total_blended_n > 0:
        print(f"Blended:     n={total_blended_n:4d}, "
              f"detection={100*total_blended_det/total_blended_n:5.1f}%, "
              f"recovery={100*total_blended_rec/total_blended_n:5.1f}%")

    if total_non_blended_n > 0:
        print(f"Non-blended: n={total_non_blended_n:4d}, "
              f"detection={100*total_non_blended_det/total_non_blended_n:5.1f}%, "
              f"recovery={100*total_non_blended_rec/total_non_blended_n:5.1f}%")

    # Which period ratios are blended?
    print("\n--- Period Ratio Classification ---")
    test_ratios = [1.52, 2.05, 2.53, 3.05, 4.05, 5.05]
    for pr in test_ratios:
        j_R = determine_j_R(pr)
        P_super = compute_super_period(5.0, pr, j_R)
        P_syn = compute_synodic_period(5.0, pr)
        rel_diff = abs(P_super - P_syn) / P_super
        status = "BLENDED" if is_blended(pr) else "distinct"
        print(f"P2/P1={pr:.2f}: j_R={j_R}, P_super={P_super:.1f}d, P_syn={P_syn:.1f}d, "
              f"diff={rel_diff*100:.1f}% [{status}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
