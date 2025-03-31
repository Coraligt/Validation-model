# Semiconductor Device Leakage Detection Model

This project implements a deep learning solution for detecting leakage in semiconductor devices. The model architecture is adapted from a 1D-CNN design originally used for cardiac signal classification, now modified to classify semiconductor device characteristics.

## Project Overview

- **Task**: Binary classification of semiconductor devices as leaky (1) or non-leaky (0)
- **Input**: Time series data from semiconductor measurements (t, q values)
- **Model**: 1D Convolutional Neural Network with MLP classifier head
- **Performance Focus**: FB Score, Accuracy, Sensitivity, Specificity

## Project Structure

```
VALIDATION-MODEL/
├── Valid/                     # Validation data directory
│   ├── dataset_3/             # Original device CSV files
│   └── indices/               # Preprocessed data and indices
│       ├── preprocessed/      # Normalized data files
│       ├── test_indices.csv   # Test set indices
│       └── train_indices.csv  # Training set indices
├── models/                    # Trained model directory (created by train.py)
├── dataset.py                 # Dataset loading and preprocessing
├── model.py                   # Model architecture definition
├── swa.py                     # Stochastic Weight Averaging implementation
├── utils.py                   # Metrics and utility functions
├── train.py                   # Training script
├── test.py                    # Testing and evaluation script
├── main.py                    # Main pipeline script (simplified)
└── README.md                  # This file
```

## Data Format

- **Source Data**: ~30,000 CSV files representing individual semiconductor device measurements
- **Filename Format**: `dev####_Y.csv` where `####` is a device ID and `Y` is the label (0=non-leaky, 1=leaky)
- **File Content**: 1002 rows with columns for 't' (time), 'v' (voltage), 'q' (charge), 'i' (current)
- **Features Used**: For the current implementation, only `t` and `q` columns are used

## Model Architecture

The model uses a 1D-CNN + MLP architecture:
1. One 1D convolutional layer (filters=3, kernel_size=85, stride=32) 
2. Batch normalization
3. ReLU activation
4. Fully connected layers (20 → 10 → 2)
5. Output layer (2 classes: non-leaky/leaky)

## Usage

### Training

The training script assumes preprocessed data and indices already exist in the `Valid/indices` directory:

```bash
python train.py --epochs 50 --batch_size 32 --use_swa --device cuda:0
```

Optional arguments:
```
--data_dir        Directory containing preprocessed files (default: ./Valid/indices/preprocessed)
--indices_dir     Directory containing index files (default: ./Valid/indices)
--output_dir      Directory to save models (default: ./output/models)
--model_type      Model type (default: 'default', options: 'default', 'small')
--use_columns     Features to use (default: 't,q')
--learning_rate   Learning rate (default: 0.0002)
--scheduler       Learning rate scheduler (default: 'cosine', options: 'cosine', 'none')
--use_swa         Enable Stochastic Weight Averaging for improved generalization
--device          Computing device (default: 'cuda:0', options: 'cuda:0', 'cpu')
```

### Testing

Test a trained model on the validation set:

```bash
python test.py --model_path ./output/models/best_model_fb.pth --device cuda:0
```

Additional options:
```
--analyze_errors  Generate detailed analysis of prediction errors
--model_summary   Print model summary and architecture details
```

## Features

1. **Advanced Training Techniques**:
   - Cosine annealing learning rate scheduling
   - Stochastic Weight Averaging (SWA)
   - Signal-based data augmentation

2. **Evaluation Metrics**:
   - Standard metrics: Accuracy, F1 Score
   - Domain-specific metrics: FB Score, Sensitivity, Specificity
   - Visual analysis: Confusion matrix, ROC curve

3. **Model Variants**:
   - Default model: Matches the architecture of the original model
   - Small model: Reduced parameter version for faster training

## Results

After training, models and evaluation results will be saved in the output directory:

```
output/models/
  ├── best_model_fb.pth      # Best model by FB score
  ├── best_model_acc.pth     # Best model by accuracy
  ├── swa_model.pth          # SWA model (if enabled)
  ├── final_model.pth        # Final model
  ├── history.json           # Training history
  └── final_eval/            # Evaluation visualizations
```

## Implementation Details

### Data Preprocessing
- MinMax scaling for `t` and `q` features
- Device-level train/test splitting to prevent data leakage
- Padding/truncation of sequences to uniform length (1002 points)

### Data Augmentation
- Signal flipping (amplitude inversion)
- Gaussian noise addition
- Optional time reversal

### Training Configuration
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Batch size: 32 (default)
- Learning rate: 0.0002 (default)

### Evaluation
- Primary metric: FB score (F-beta with beta=2)
- Additional metrics: F1, Sensitivity, Specificity, BAC, Accuracy

## Requirements

- PyTorch (>= 1.7.0)
- NumPy
- Pandas
- Matplotlib
- tqdm
- scikit-learn

## Acknowledgments

This project adapts the model architecture from the TinyML Contest, originally designed for cardiac signal classification, and applies it to semiconductor device leakage detection.
