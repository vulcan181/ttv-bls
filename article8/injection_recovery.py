"""
Injection-recovery simulations for TTV mass estimation.

Tests the mass recovery pipeline across parameter space.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

# Add parent directory for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from laplace_coefficients import chopping_coefficient_A1, get_alpha_from_period_ratio
from ttv_models import (
    generate_combined_ttv,
    mass_from_chopping_amplitude,
    mass_from_lithwick_amplitude,
    compute_resonance_parameter,
    compute_super_period,
    compute_synodic_period,
    DAY_TO_MIN
)
from harmonic_decomposition import extract_ttv_harmonics


# Constants
M_SUN_KG = 1.989e30
M_EARTH_KG = 5.972e24


def run_single_simulation(
    m2_earth: float,
    P2_over_P1: float,
    e2: float,
    n_transits: int,
    sigma_t: float,
    P1: float = 5.0,
    M_star: float = 1.0,
    seed: int = None
) -> dict:
    """
    Run a single injection-recovery simulation.

    Args:
        m2_earth: Companion mass in Earth masses
        P2_over_P1: Period ratio P2/P1
        e2: Companion eccentricity
        n_transits: Number of observed transits
        sigma_t: Timing precision in minutes
        P1: Inner planet period in days
        M_star: Stellar mass in solar masses
        seed: Random seed for reproducibility

    Returns:
        dict with injection parameters and recovery results
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert mass to mass ratio
    mu2 = (m2_earth * M_EARTH_KG) / (M_star * M_SUN_KG)

    # Compute periods
    P2 = P1 * P2_over_P1

    # Determine resonance
    j_R = round(P2_over_P1 / (P2_over_P1 - 1))
    if j_R < 2:
        j_R = 2
    if j_R > 7:
        j_R = 7

    # Free eccentricity (simplified model)
    # Z_free magnitude scales with e2
    Z_free = e2 * np.exp(1j * np.random.uniform(0, 2*np.pi))

    # Generate transit times
    transit_numbers = np.arange(n_transits)
    times = transit_numbers * P1

    # Generate true TTV signal
    E_TTV = np.random.uniform(0, P1)
    phi_chop = np.random.uniform(0, 2*np.pi)

    true_ttv = generate_combined_ttv(
        transit_numbers, P1, P2, mu2, j_R,
        E_TTV=E_TTV, Z_free=Z_free, phi_chop=phi_chop
    )

    # Add noise
    noise = np.random.normal(0, sigma_t, n_transits)
    observed_oc = true_ttv + noise
    oc_errors = np.ones(n_transits) * sigma_t

    # Compute expected amplitudes for reference
    alpha = get_alpha_from_period_ratio(P2_over_P1)
    Delta = compute_resonance_parameter(P1, P2, j_R)
    P_super = compute_super_period(P1, P2, j_R)
    P_syn = compute_synodic_period(P1, P2)

    # True amplitudes
    from ttv_models import lithwick_ttv_amplitude, chopping_ttv_amplitude
    true_A_lithwick = lithwick_ttv_amplitude(P1, mu2, j_R, Delta, Z_free)
    true_A_chop = chopping_ttv_amplitude(P1, mu2, alpha)

    # Extract harmonics
    baseline = times[-1] - times[0]
    min_period = max(5.0, P1 * 2)
    max_period = min(500.0, baseline / 2)

    try:
        result = extract_ttv_harmonics(
            times, observed_oc, oc_errors, P1,
            min_period=min_period, max_period=max_period
        )
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'injected': {
                'm2_earth': m2_earth,
                'P2_over_P1': P2_over_P1,
                'e2': e2,
                'n_transits': n_transits,
                'sigma_t': sigma_t
            }
        }

    # Analyze results
    recovered_mass = None
    recovery_method = None
    detection = False

    harmonics = result['harmonics']

    if len(harmonics) > 0:
        # Find best harmonic (strongest or best matching expected periods)
        best_harmonic = None
        best_score = 0

        for h in harmonics:
            # Score based on proximity to expected periods
            score = 0
            if h['classification'] == 'synodic':
                # Check if close to true synodic period
                if abs(h['period'] - P_syn) / P_syn < 0.2:
                    score = h['amplitude'] / h['amplitude_error'] + 5
            elif h['classification'] == 'super':
                # Check if close to true super-period
                if abs(h['period'] - P_super) / P_super < 0.2:
                    score = h['amplitude'] / h['amplitude_error'] + 3

            if score > best_score:
                best_score = score
                best_harmonic = h

        if best_harmonic is not None:
            detection = True

            if best_harmonic['classification'] == 'synodic':
                # Use chopping formula
                recovered_mass = mass_from_chopping_amplitude(
                    best_harmonic['amplitude'],
                    P1,
                    best_harmonic['implied_P2_over_P1'] or P2_over_P1,
                    M_star
                )
                recovery_method = 'chopping'
            elif best_harmonic['classification'] == 'super':
                # Use Lithwick formula
                implied_P2 = best_harmonic['implied_P2'] or P2
                implied_j_R = best_harmonic['j_R'] or j_R
                recovered_mass = mass_from_lithwick_amplitude(
                    best_harmonic['amplitude'],
                    P1, implied_P2, implied_j_R, M_star,
                    assume_low_e=True
                )
                recovery_method = 'lithwick'
            else:
                # Unknown classification - try Lithwick as fallback
                recovered_mass = mass_from_lithwick_amplitude(
                    best_harmonic['amplitude'],
                    P1, P2, j_R, M_star,
                    assume_low_e=True
                )
                recovery_method = 'lithwick_fallback'

    # Compute recovery metrics
    if recovered_mass is not None:
        mass_ratio = recovered_mass / m2_earth
        within_factor_2 = 0.5 <= mass_ratio <= 2.0
        within_factor_3 = 0.33 <= mass_ratio <= 3.0
    else:
        mass_ratio = None
        within_factor_2 = False
        within_factor_3 = False

    return {
        'success': True,
        'detection': detection,
        'injected': {
            'm2_earth': m2_earth,
            'P2_over_P1': P2_over_P1,
            'e2': e2,
            'n_transits': n_transits,
            'sigma_t': sigma_t,
            'true_A_lithwick': true_A_lithwick,
            'true_A_chop': true_A_chop,
            'P_super': P_super,
            'P_syn': P_syn,
            'j_R': j_R
        },
        'recovered': {
            'm2_earth': recovered_mass,
            'method': recovery_method,
            'n_harmonics': len(harmonics)
        },
        'metrics': {
            'mass_ratio': mass_ratio,
            'within_factor_2': within_factor_2,
            'within_factor_3': within_factor_3
        }
    }


def run_configuration(config: dict, n_realizations: int = 100) -> dict:
    """
    Run multiple realizations for a single parameter configuration.

    Args:
        config: dict with m2_earth, P2_over_P1, e2, n_transits, sigma_t
        n_realizations: Number of noise realizations

    Returns:
        dict with aggregated statistics
    """
    results = []

    for i in range(n_realizations):
        result = run_single_simulation(
            m2_earth=config['m2_earth'],
            P2_over_P1=config['P2_over_P1'],
            e2=config['e2'],
            n_transits=config['n_transits'],
            sigma_t=config['sigma_t'],
            seed=config.get('base_seed', 0) + i
        )
        results.append(result)

    # Aggregate statistics
    n_success = sum(1 for r in results if r.get('success', False))
    n_detection = sum(1 for r in results if r.get('detection', False))
    n_within_2 = sum(1 for r in results if r.get('metrics', {}).get('within_factor_2', False))
    n_within_3 = sum(1 for r in results if r.get('metrics', {}).get('within_factor_3', False))

    mass_ratios = [r['metrics']['mass_ratio'] for r in results
                   if r.get('metrics', {}).get('mass_ratio') is not None]

    return {
        'config': config,
        'n_realizations': n_realizations,
        'n_success': n_success,
        'detection_rate': n_detection / n_realizations if n_realizations > 0 else 0,
        'recovery_rate_factor2': n_within_2 / n_realizations if n_realizations > 0 else 0,
        'recovery_rate_factor3': n_within_3 / n_realizations if n_realizations > 0 else 0,
        'mass_ratio_median': np.median(mass_ratios) if mass_ratios else None,
        'mass_ratio_std': np.std(mass_ratios) if mass_ratios else None,
        'individual_results': results
    }


def main():
    """Main entry point for HPC execution."""
    if len(sys.argv) < 2:
        print("Usage: python injection_recovery.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    print(f"Loading config from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    n_realizations = config.get('n_realizations', 100)

    print(f"Running injection-recovery simulation")
    print(f"  m2 = {config['m2_earth']} M_Earth")
    print(f"  P2/P1 = {config['P2_over_P1']}")
    print(f"  e2 = {config['e2']}")
    print(f"  N_transits = {config['n_transits']}")
    print(f"  sigma_t = {config['sigma_t']} min")
    print(f"  N_realizations = {n_realizations}")
    print()

    start_time = time.time()
    result = run_configuration(config, n_realizations)
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Detection rate: {result['detection_rate']*100:.1f}%")
    print(f"Recovery rate (factor 2): {result['recovery_rate_factor2']*100:.1f}%")
    print(f"Recovery rate (factor 3): {result['recovery_rate_factor3']*100:.1f}%")

    # Save results
    output_file = config_file.replace('configs/', 'results/').replace('.json', '_results.json')
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Don't save individual results to keep file size manageable
    result_summary = {k: v for k, v in result.items() if k != 'individual_results'}

    with open(output_file, 'w') as f:
        json.dump(result_summary, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
