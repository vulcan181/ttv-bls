# TTV-BLS: Transit Timing Variation-aware Box Least Squares

A Python package for detecting transiting exoplanets in the presence of Transit Timing Variations (TTVs).

## Overview

Standard transit detection algorithms like BLS (Box Least Squares) can fail when transits exhibit significant timing variations. TTV-BLS corrects for sinusoidal TTVs before phase-folding, recovering detection sensitivity.

**Key Finding**: The critical threshold where TTV-BLS becomes beneficial is when `A_TTV / T_14 > 0.5-0.7` (TTV amplitude relative to transit duration).

## Installation

```bash
git clone https://github.com/vulcan181/ttv-bls.git
cd ttv-bls
pip install -e .
```

## Quick Start

```python
import numpy as np
from ttv_bls import create_lightcurve, transit_search, distort_timebase

# Create synthetic light curve with TTV
t, flux, flux_err, t14 = create_lightcurve(
    cadence=120,           # 2-minute cadence (seconds)
    duration=365,          # 1 year observation (days)
    period=10.0,           # Orbital period (days)
    epoch=5.0,             # First transit time (days)
    a_ttv=0.05,            # TTV amplitude (days) = 72 minutes
    p_ttv=120.0,           # TTV period (days)
    count_rate=1000,       # Photon count rate
    r_planet=0.1,          # Planet radius (Rp/Rs)
)

# Standard BLS search (will have reduced sensitivity due to TTV)
result_std = transit_search(t, flux, flux_err, min_period=5, max_period=20)
print(f"Standard BLS: SDE = {np.max(result_std['sde']):.1f}")

# TTV-BLS search (corrects for TTV)
t_corrected = distort_timebase(t, epoch=0, p_ttv=120.0, a_ttv=0.05, e_ttv=0.0)
result_ttv = transit_search(t_corrected, flux, flux_err, min_period=5, max_period=20)
print(f"TTV-BLS: SDE = {np.max(result_ttv['sde']):.1f}")
```

## Package Structure

```
ttv-bls/
├── ttv_bls/              # Core library
│   ├── core.py           # TTV-BLS algorithm
│   ├── lightcurve.py     # Light curve generation
│   └── utils.py          # Utilities
├── article1/             # Paper I reproduction scripts
├── article2/             # Paper II reproduction scripts
├── article3/             # Paper III reproduction scripts
├── article5/             # Paper IV reproduction scripts
├── article6/             # Paper V reproduction scripts
├── article7/             # Paper VI reproduction scripts
├── article8/             # Paper VII reproduction scripts
└── examples/             # Tutorial notebooks
```

## Publications

This code accompanies the following publications:

1. **Article I** - TTV-BLS Algorithm: "Robustness of Transit Detection to Small-Amplitude Transit Timing Variations"
2. **Article II** - Real-Data Validation: "Real-Data Validation of TTV-Aware Transit Detection"
3. **Article III** - TTV-Net: "Physics-Informed Deep Learning for TTV-Robust Transit Detection"
4. **Article IV** - Habitable Zone: "Hidden Earths: TTV-Aware Detection of Habitable Zone Planets"
5. **Article V** - Bayesian Inference: "TTV-Informed Detection and Characterisation of Single and Duo-Transit Events"
6. **Article VI** - Companion Survey: "A Systematic Search for Hidden Companions in 100 TESS Systems"
7. **Article VII** - Mass Estimation: "When Does Photometric Mass Estimation Work?"

## Citation

If you use this code, please cite:

```bibtex
@phdthesis{Kalogerakos2026,
  author = {Kalogerakos, Stamatis},
  title = {TTV-Aware Exoplanet Transit Detection: Bias Quantification and Survey Completeness},
  school = {University of Warwick},
  year = {2026},
}
```

## License

MIT License - see LICENSE file.
