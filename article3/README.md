# Article III: TTV-Net Deep Learning

**Title**: "TTV-Net: Physics-Informed Deep Learning for TTV-Robust Transit Detection"

## Key Results

- CNN architecture for TTV-robust transit detection
- Trained on 100,000+ synthetic light curves
- Achieves 95% detection rate where standard methods fail
- Physics-informed loss function incorporating TTV priors

## Reproducing Results

### 1. Generate training data
```bash
python generate_training_data.py
```

### 2. Train model
```bash
python train_ttvnet.py --epochs 100 --batch_size 64
```

### 3. Evaluate
```bash
python evaluate_model.py
```

## Requirements

Additional dependencies for deep learning:
```bash
pip install tensorflow>=2.10.0
```

## Model Architecture

- Input: Phase-folded light curve (512 bins)
- 4 convolutional layers with batch normalization
- Dense layers with dropout
- Output: Transit probability + TTV parameters
