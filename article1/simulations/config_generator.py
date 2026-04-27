#!/usr/bin/env python
"""
Generate configuration files for Study 1: Bin Width Sensitivity.

Creates JSON config files for each parameter combination in the grid.
"""

import json
from pathlib import Path
import itertools


def generate_configs():
    """Generate all configuration files for the bin width study."""

    # Output directory
    config_dir = Path(__file__).parent / 'configs'
    config_dir.mkdir(exist_ok=True)

    # MINIMAL parameter grid (5 configs) - just answer "why 45 min bins?"
    bin_widths = [15, 30, 45, 60, 90]  # 5 values
    a_ttv_over_t14_values = [0.6]  # 1 value at threshold
    t14_hours_values = [3.0]  # 1 typical value

    # Fixed parameters
    base_config = {
        'cadence_sec': 120.0,           # 2-minute cadence (TESS-like)
        'duration_days': 180.0,         # 6 months observation
        'period_days': 4.0,             # Orbital period
        'epoch_days': 1.0,              # First transit epoch
        'p_ttv_days': 60.0,             # TTV period
        'e_ttv_days': -20.0,            # TTV epoch offset
        'count_rate': 1e7,              # High SNR for clean measurement
        'r_planet': 0.1,                # 10% radius ratio (Jupiter-like)
        'a_rs': 15.0,                   # Semi-major axis ratio
        'n_realizations': 100,          # Number of noise realizations
        'min_period': 0.6,              # BLS search range
        'max_period': 75.0,
        'random_seed_base': 42,         # For reproducibility
    }

    configs = []

    for bin_width, a_ttv_t14, t14_hours in itertools.product(
            bin_widths, a_ttv_over_t14_values, t14_hours_values):

        # Convert t14 to days
        t14_days = t14_hours / 24.0

        # Compute a_ttv from ratio and t14
        a_ttv_days = a_ttv_t14 * t14_days

        config = base_config.copy()
        config['bin_width_min'] = bin_width
        config['a_ttv_over_t14'] = a_ttv_t14
        config['t14_hours'] = t14_hours
        config['t14_days'] = t14_days
        config['a_ttv_days'] = a_ttv_days

        # Create unique run ID
        run_id = f"bw{bin_width}_at{a_ttv_t14:.2f}_t{t14_hours:.1f}"
        config['run_id'] = run_id

        # Save config file
        config_file = config_dir / f"config_{run_id}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        configs.append(config)
        print(f"Created: {config_file.name}")

    print(f"\nGenerated {len(configs)} configuration files in {config_dir}")

    # Create master config list
    master_file = config_dir / 'all_configs.json'
    with open(master_file, 'w') as f:
        json.dump({'configs': [c['run_id'] for c in configs]}, f, indent=2)

    return configs


if __name__ == '__main__':
    generate_configs()
