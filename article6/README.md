# Article V: Bayesian Period Inference

**Title**: "TTV-Informed Detection and Characterisation of Single and Duo-Transit Events: Bayesian Period Inference with Dynamical Priors"

## Key Results

- Bayesian framework for monotransit/duo-transit period estimation
- TTV-informed priors reduce period ambiguity
- Application to TESS single-transit candidates

## Reproducing Results

### 1. Run N-body simulations (Set A)
```bash
python run_nbody_grid.py
```

### 2. Run recovery analysis (Set B)
```bash
python run_recovery_study.py
```

### 3. Generate figures
```bash
python create_figures.py
```

## Key Methods

- REBOUND N-body integration for TTV lookup tables
- Bayesian marginalization over orbital periods
- Alias discrimination for duo-transit events
