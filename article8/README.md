# Article VII: Mass Estimation

**Title**: "When Does Photometric Mass Estimation Work? A Comprehensive Assessment of TTV-Based Companion Characterization"

## Key Results

- 600,000+ injection-recovery simulations
- Chopping method accuracy assessment
- Bias calibration for resonance-dependent systematics
- PLATO predictions: ~31,000 photometric masses

## Reproducing Results

### 1. Run injection-recovery simulations
```bash
python run_injection_recovery.py
```

### 2. Analyze bias calibration
```bash
python analyze_bias.py
```

### 3. Generate figures
```bash
python create_figures.py
```

## Key Findings

- 2:1 resonance: unbiased (1.0x)
- 3:2 resonance: 2.0x systematic bias
- Eccentricity extends degeneracy significantly
- Decision flowchart for method selection
