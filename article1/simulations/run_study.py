#!/usr/bin/env python
"""
Study 1: Phase Bin Width Sensitivity Simulation.

Run TTV-BLS simulations to measure how detection efficiency depends on
phase bin width and A_TTV/T_14 ratio.

Usage:
    python run_bin_width_study.py config.json [--n-workers N]
"""

import sys
import os

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

from common.ttv_bls_core import transit_search, distort_timebase
from common.lightcurve_generator import create_lightcurve, get_transit_duration
from common.utils import setup_logging, save_results, compute_improvement_statistics


def run_single_realization(args):
    """Run a single realization with given parameters."""
    (config, seed) = args

    # Create light curve with TTV
    t, flux, flux_err, t14 = create_lightcurve(
        cadence=config['cadence_sec'],
        duration=config['duration_days'],
        period=config['period_days'],
        epoch=config['epoch_days'],
        a_ttv=config['a_ttv_days'],
        p_ttv=config['p_ttv_days'],
        e_ttv=config['e_ttv_days'],
        count_rate=config['count_rate'],
        r_planet=config['r_planet'],
        a_rs=config['a_rs'],
        seed=seed
    )

    # Run standard BLS (no TTV correction)
    res_std = transit_search(
        t, flux, flux_err,
        min_period=config['min_period'],
        max_period=config['max_period'],
        bin_width=config['bin_width_min']
    )
    best_std = res_std[np.argmax(res_std['sde'])]
    sde_standard = best_std['sde']

    # Run TTV-BLS (with known TTV parameters)
    t_corrected = distort_timebase(
        t, 0.0,
        config['p_ttv_days'],
        config['a_ttv_days'],
        config['e_ttv_days']
    )
    res_corr = transit_search(
        t_corrected, flux, flux_err,
        min_period=config['min_period'],
        max_period=config['max_period'],
        bin_width=config['bin_width_min']
    )
    best_corr = res_corr[np.argmax(res_corr['sde'])]
    sde_corrected = best_corr['sde']

    # Compute metrics
    snr_ratio = sde_corrected / sde_standard if sde_standard > 0 else np.inf

    return {
        'seed': seed,
        'sde_standard': float(sde_standard),
        'sde_corrected': float(sde_corrected),
        'snr_ratio': float(snr_ratio),
        'period_std': float(best_std['period']),
        'period_corr': float(best_corr['period']),
        't14_actual': float(t14),
        'detected_std': sde_standard >= 7.0,
        'detected_corr': sde_corrected >= 7.0,
    }


def run_study(config_file, n_workers=None, output_dir=None):
    """Run the bin width study for a single configuration."""

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    run_id = config.get('run_id', 'unknown')

    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_dir / f"{run_id}.log"
    logger = setup_logging(f"study1_{run_id}", str(log_file))

    logger.info(f"Starting Study 1: Bin Width Sensitivity")
    logger.info(f"Configuration: {run_id}")
    logger.info(f"  bin_width: {config['bin_width_min']} min")
    logger.info(f"  A_TTV/T_14: {config['a_ttv_over_t14']}")
    logger.info(f"  T_14: {config['t14_hours']} hours")
    logger.info(f"  n_realizations: {config['n_realizations']}")

    # Generate seeds for each realization
    base_seed = config.get('random_seed_base', 42)
    seeds = [base_seed + i for i in range(config['n_realizations'])]

    # Prepare arguments for parallel execution
    args_list = [(config, seed) for seed in seeds]

    # Run simulations
    start_time = datetime.now()

    if n_workers is None:
        n_workers = mp.cpu_count()

    logger.info(f"Running {config['n_realizations']} realizations on {n_workers} workers...")

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(pool.map(run_single_realization, args_list))
    else:
        results = [run_single_realization(args) for args in args_list]

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")

    # Convert to structured array
    dtype = [
        ('seed', 'i4'),
        ('sde_standard', 'f4'),
        ('sde_corrected', 'f4'),
        ('snr_ratio', 'f4'),
        ('period_std', 'f4'),
        ('period_corr', 'f4'),
        ('t14_actual', 'f4'),
        ('detected_std', '?'),
        ('detected_corr', '?'),
    ]
    results_array = np.array([tuple(r[k] for k, _ in dtype) for r in results], dtype=dtype)

    # Compute summary statistics
    sde_std = results_array['sde_standard']
    sde_corr = results_array['sde_corrected']
    improvement_stats = compute_improvement_statistics(sde_std, sde_corr)

    logger.info(f"Results summary:")
    logger.info(f"  SDE (standard): {np.mean(sde_std):.2f} +/- {np.std(sde_std):.2f}")
    logger.info(f"  SDE (corrected): {np.mean(sde_corr):.2f} +/- {np.std(sde_corr):.2f}")
    logger.info(f"  SNR ratio (mean): {improvement_stats['ratio_mean']:.3f}")
    logger.info(f"  Detection rate (std): {improvement_stats['detection_rate_std']:.1%}")
    logger.info(f"  Detection rate (corr): {improvement_stats['detection_rate_corr']:.1%}")

    # Save results
    save_config = config.copy()
    save_config['improvement_stats'] = improvement_stats
    save_config['elapsed_seconds'] = elapsed

    npy_file, json_file = save_results(
        results_array, output_dir, run_id, save_config
    )

    logger.info(f"Results saved to: {npy_file}")

    return results_array, improvement_stats


def main():
    parser = argparse.ArgumentParser(
        description='Run Study 1: Bin Width Sensitivity')
    parser.add_argument('config', help='Path to configuration JSON file')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    run_study(args.config, args.n_workers, args.output_dir)


if __name__ == '__main__':
    main()
