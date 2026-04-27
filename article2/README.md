# Article II: Real-Data Validation

**Title**: "Real-Data Validation of TTV-Aware Transit Detection: Application to Kepler and TESS Multi-Planet Systems with Predictions for PLATO"

## Key Results

- Validated TTV-BLS on 50+ Kepler/TESS systems with known TTVs
- Recovery improvement: +12% detection rate for high-TTV systems
- PLATO predictions: ~2,000 additional detections with TTV-BLS

## Reproducing Results

### 1. Download Kepler/TESS data
```bash
python download_data.py
```

### 2. Run validation
```bash
python run_validation.py
```

### 3. Generate figures
```bash
python create_figures.py
```

## Data Sources

- Kepler DR25 light curves (MAST)
- TESS 2-minute cadence (MAST)
- NASA Exoplanet Archive for TTV catalogs
