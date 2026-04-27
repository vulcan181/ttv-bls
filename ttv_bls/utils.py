"""
Utility functions for TTV-BLS simulations.

This module provides:
- Result I/O (JSON, NumPy)
- Statistics computation
- Logging setup
- Configuration handling
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys


def setup_logging(name, log_file=None, level=logging.INFO):
    """
    Set up logging for simulation scripts.

    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file. If None, logs to stdout only
    level : int
        Logging level

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def save_results(results, output_dir, prefix, config=None):
    """
    Save simulation results to disk.

    Saves:
    - results.npy: NumPy binary with structured array
    - results_summary.json: Human-readable summary
    - config.json: Configuration used (if provided)

    Parameters
    ----------
    results : np.ndarray
        Structured array of results
    output_dir : str or Path
        Output directory
    prefix : str
        Filename prefix
    config : dict, optional
        Configuration dictionary to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results
    npy_file = output_dir / f"{prefix}_{timestamp}.npy"
    np.save(npy_file, results)

    # Save summary
    summary = compute_sde_statistics(results)
    summary['timestamp'] = timestamp
    summary['n_results'] = len(results.flatten()) if hasattr(results, 'flatten') else len(results)

    json_file = output_dir / f"{prefix}_{timestamp}_summary.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_serializer)

    # Save config
    if config is not None:
        config_file = output_dir / f"{prefix}_{timestamp}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=_json_serializer)

    return npy_file, json_file


def load_results(npy_file):
    """
    Load simulation results from NumPy file.

    Parameters
    ----------
    npy_file : str or Path
        Path to .npy file

    Returns
    -------
    np.ndarray
        Loaded results
    """
    return np.load(npy_file, allow_pickle=True)


def load_config(json_file):
    """
    Load configuration from JSON file.

    Parameters
    ----------
    json_file : str or Path
        Path to JSON config file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def compute_sde_statistics(results):
    """
    Compute summary statistics for SDE values.

    Parameters
    ----------
    results : np.ndarray
        Structured array with 'sde' field

    Returns
    -------
    dict
        Statistics dictionary
    """
    if hasattr(results, 'flatten'):
        sde = results['sde'].flatten()
    else:
        sde = np.array([r['sde'] for r in results])

    sde = sde[np.isfinite(sde)]

    if len(sde) == 0:
        return {'n_valid': 0}

    stats = {
        'n_valid': len(sde),
        'sde_mean': float(np.mean(sde)),
        'sde_median': float(np.median(sde)),
        'sde_std': float(np.std(sde)),
        'sde_min': float(np.min(sde)),
        'sde_max': float(np.max(sde)),
        'sde_q25': float(np.percentile(sde, 25)),
        'sde_q75': float(np.percentile(sde, 75)),
        'detection_rate_7': float(np.mean(sde >= 7.0)),
        'detection_rate_8': float(np.mean(sde >= 8.0)),
    }

    return stats


def compute_improvement_statistics(sde_standard, sde_corrected):
    """
    Compute improvement statistics for TTV-BLS vs standard BLS.

    Parameters
    ----------
    sde_standard : np.ndarray
        SDE values from standard BLS
    sde_corrected : np.ndarray
        SDE values from TTV-BLS

    Returns
    -------
    dict
        Improvement statistics
    """
    sde_standard = np.asarray(sde_standard).flatten()
    sde_corrected = np.asarray(sde_corrected).flatten()

    # Filter valid values
    mask = np.isfinite(sde_standard) & np.isfinite(sde_corrected) & (sde_standard > 0)
    sde_std = sde_standard[mask]
    sde_corr = sde_corrected[mask]

    if len(sde_std) == 0:
        return {'n_valid': 0}

    ratio = sde_corr / sde_std
    improvement = sde_corr - sde_std

    stats = {
        'n_valid': len(sde_std),
        'ratio_mean': float(np.mean(ratio)),
        'ratio_median': float(np.median(ratio)),
        'ratio_std': float(np.std(ratio)),
        'improvement_mean': float(np.mean(improvement)),
        'improvement_median': float(np.median(improvement)),
        'fraction_improved': float(np.mean(ratio > 1.0)),
        'detection_rate_std': float(np.mean(sde_std >= 7.0)),
        'detection_rate_corr': float(np.mean(sde_corr >= 7.0)),
        'detection_improvement': float(np.mean(sde_corr >= 7.0) - np.mean(sde_std >= 7.0)),
    }

    return stats


def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000,
                                   confidence=0.95, seed=None):
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic_func : callable
        Function to compute statistic (e.g., np.mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (point_estimate, ci_low, ci_high)
    """
    from numpy.random import default_rng

    rng = default_rng(seed)
    data = np.asarray(data).flatten()
    n = len(data)

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Confidence interval
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_low, ci_high


def find_critical_threshold(a_ttv_t14_values, improvement_ratios, threshold_ratio=1.5):
    """
    Find the critical A_TTV/T_14 threshold where improvement becomes significant.

    Parameters
    ----------
    a_ttv_t14_values : np.ndarray
        Array of A_TTV/T_14 values
    improvement_ratios : np.ndarray
        Corresponding improvement ratios (SDE_corr / SDE_std)
    threshold_ratio : float
        Ratio value defining "significant" improvement. Default 1.5

    Returns
    -------
    float
        Critical threshold value (or NaN if not found)
    """
    a_ttv_t14_values = np.asarray(a_ttv_t14_values)
    improvement_ratios = np.asarray(improvement_ratios)

    # Sort by A_TTV/T_14
    sort_idx = np.argsort(a_ttv_t14_values)
    x = a_ttv_t14_values[sort_idx]
    y = improvement_ratios[sort_idx]

    # Find first crossing above threshold
    above_threshold = y >= threshold_ratio
    if not np.any(above_threshold):
        return np.nan

    first_above = np.argmax(above_threshold)
    if first_above == 0:
        return x[0]

    # Linear interpolation to find crossing point
    x1, x2 = x[first_above - 1], x[first_above]
    y1, y2 = y[first_above - 1], y[first_above]

    if y2 == y1:
        return x1

    critical = x1 + (threshold_ratio - y1) * (x2 - x1) / (y2 - y1)
    return critical


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def generate_config_grid(base_config, param_grids):
    """
    Generate configuration grid from parameter ranges.

    Parameters
    ----------
    base_config : dict
        Base configuration with fixed parameters
    param_grids : dict
        Dictionary mapping parameter names to lists of values

    Yields
    ------
    dict
        Configuration for each grid point
    """
    import itertools

    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())

    for values in itertools.product(*param_values):
        config = base_config.copy()
        for name, value in zip(param_names, values):
            config[name] = value
        yield config


def create_run_id(config):
    """
    Create a unique run ID from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    str
        Run ID string
    """
    # Extract key parameters for ID
    parts = []

    if 'bin_width' in config:
        parts.append(f"bw{config['bin_width']}")
    if 'a_ttv_over_t14' in config:
        parts.append(f"at{config['a_ttv_over_t14']:.2f}")
    if 't14_hours' in config:
        parts.append(f"t{config['t14_hours']:.1f}")
    if 'target_sde' in config:
        parts.append(f"sde{config['target_sde']:.1f}")
    if 'p_ttv' in config:
        parts.append(f"p{config['p_ttv']}")

    if not parts:
        parts.append(datetime.now().strftime('%H%M%S'))

    return '_'.join(parts)
