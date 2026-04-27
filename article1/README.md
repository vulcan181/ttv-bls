# Article I: TTV-BLS Algorithm

**Title**: "Robustness of Transit Detection to Small-Amplitude Transit Timing Variations: A TTV-Aware BLS Study"

## Key Results

- Critical threshold: A_TTV / T_14 ~ 0.5-0.7
- Below threshold: TTV-BLS offers minimal advantage
- Above threshold: Advantage factors of 3x to >20x

## Reproducing Results

### 1. Install dependencies
```bash
pip install -e ..
```

### 2. Run simulations
```bash
# Generate parameter grid
python simulations/config_generator.py

# Run simulation study (requires HPC for full grid)
python simulations/run_study.py configs/config_001.json

# Or run quick test locally
python run_quick_test.py
```

### 3. Generate figures
```bash
python create_figures.py
```

## Files

- `simulations/` - HPC simulation scripts
- `create_figures.py` - Figure generation
- `run_quick_test.py` - Quick local test

