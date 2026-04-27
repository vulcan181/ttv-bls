# Article VI: Companion Survey

**Title**: "A Systematic Search for Hidden Companions in 100 TESS Planetary Systems Using the O-C versus Depth Diagnostic"

## Key Results

- Survey of 100 TESS systems for companion signatures
- 12 companion candidates (31.6% of analyzed sample)
- WASP-76 b companion independently confirmed
- 6 novel discoveries with RV follow-up predictions

## Reproducing Results

### 1. Download TESS light curves
```bash
python download_tess_data.py
```

### 2. Run O-C analysis
```bash
python run_oc_analysis.py
```

### 3. Red noise validation
```bash
python red_noise_validation.py
```

### 4. Generate figures
```bash
python create_figures.py
```

## Results

See the paper for the full companion candidate table and RV predictions.
