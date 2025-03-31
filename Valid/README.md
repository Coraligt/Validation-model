# Semiconductor Device Leakage Detection Model

This project implements a complete deep learning solution for detecting leakage in semiconductor devices. The model architecture is adapted from the original TinyML contest model, modified to fit the semiconductor data characteristics.

## Project Structure

```
├── dataset.py              # Dataset loading and processing
├── model.py                # Model architecture definition
├── swa.py                  # Stochastic Weight Averaging implementation
├── utils.py                # Evaluation metrics and utility functions
├── prepare_data.py         # Data preprocessing script
├── train.py                # Model training script
├── test.py                 # Model testing script
└── main.py                 # Complete pipeline script
```

## Installation

Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas matplotlib tqdm scikit-learn
```

## Data Format

Input data should be CSV files, each containing measurements from a semiconductor device, with the following format:
- Filename format: `dev####_Y.csv`, where `####` is the device ID and `Y` is the label (0=non-leaky, 1=leaky)
- Each file contains 1002 rows of data
- Columns are: `t` (time), `v` (voltage), `q` (charge), `i` (current)

## Model Architecture

The model uses a 1D-CNN + MLP architecture:
1. One 1D convolutional layer (kernel_size=85, stride=32)
2. Batch normalization
3. ReLU activation
4. Fully connected layers [20, 10]
5. Output layer (2 classes: non-leaky/leaky)

## Usage

### 1. Complete Pipeline

Run the complete pipeline (data preparation, training, and testing):

```bash
python main.py --data_dir ./dataset_3 --output_dir ./output
```

### 2. Data Preparation

Prepare and analyze data only:

```bash
python prepare_data.py --data_dir ./dataset_3 --output_dir ./preprocessed_data
```

### 3. Model Training

Train model using preprocessed data:

```bash
python train.py --data_dir ./preprocessed_data/preprocessed --indices_dir ./preprocessed_data --output_dir ./model_output
```

Enable Stochastic Weight Averaging (SWA):

```bash
python train.py --data_dir ./preprocessed_data/preprocessed --indices_dir ./preprocessed_data --output_dir ./model_output --use_swa
```

### 4. Model Testing

Test a trained model:

```bash
python test.py --data_dir ./preprocessed_data/preprocessed --indices_dir ./preprocessed_data --model_path ./model_output/best_model_fb.pth --output_dir ./test_results
```

Analyze prediction errors:

```bash
python test.py --data_dir ./preprocessed_data/preprocessed --indices_dir ./preprocessed_data --model_path ./model_output/best_model_fb.pth --output_dir ./test_results --analyze_errors
```

## Command Line Arguments

### Main Script (main.py)

```
--data_dir              Directory containing original CSV files
--output_dir            Directory to save output files
--use_columns           Columns to use as features (default: 't,q')
--seq_length            Sequence length (default: 1002)
--model_type            Model type ('default' or 'small')
--epochs                Number of training epochs (default: 50)
--batch_size            Batch size (default: 32)
--learning_rate         Learning rate (default: 0.0002)
--use_swa               Enable Stochastic Weight Averaging
--skip_prepare          Skip data preparation
--skip_train            Skip training
--skip_test             Skip testing
--device                Device (e.g. 'cuda:0' or 'cpu')
```

### Training Script (train.py)

```
--data_dir              Directory containing preprocessed CSV files
--indices_dir           Directory containing index files
--output_dir            Directory to save models
--use_columns           Columns to use as features
--seq_length            Sequence length
--model_type            Model type
--epochs                Number of training epochs
--batch_size            Batch size
--learning_rate         Learning rate
--min_lr                Minimum learning rate
--scheduler             Learning rate scheduler ('cosine' or 'none')
--use_time_reverse      Use time reversal data augmentation
--use_swa               Enable Stochastic Weight Averaging
--swa_start             Epoch to start SWA
--swa_freq              SWA update frequency
--swa_lr                SWA learning rate
--seed                  Random seed
--device                Device
--num_workers           Number of data loading workers
```

### Testing Script (test.py)

```
--data_dir              Directory containing preprocessed CSV files
--indices_dir           Directory containing index files
--output_dir            Directory to save test results
--model_path            Model path
--use_columns           Columns to use as features
--seq_length            Sequence length
--model_type            Model type
--batch_size            Batch size
--analyze_errors        Analyze prediction errors
--model_summary         Print model summary
--device                Device
--num_workers           Number of data loading workers
```

## Features

1. **Data Processing**:
   - Data normalization
   - Data augmentation (signal flipping, Gaussian noise, optional time reversal)
   - Device-level splitting (ensuring samples from the same device don't appear in both train and test sets)

2. **Training Techniques**:
   - Cosine annealing learning rate scheduling
   - Stochastic Weight Averaging (SWA)
   - Early stopping and best model saving

3. **Evaluation Metrics**:
   - Accuracy, F1 score, FB score
   - Sensitivity (recall), Specificity
   - Precision, Negative Predictive Value
   - ROC curve and AUC
   - Confusion matrix visualization

4. **Model Variants**:
   - Standard model (consistent with original TinyML contest model)
   - Small model (fewer parameters for resource-constrained scenarios)

## Results

After training, the following files will be saved in the output directory:

```
models/
  ├── best_model_fb.pth      # Best model by FB score
  ├── best_model_acc.pth     # Best model by accuracy
  ├── swa_model.pth          # SWA model (if enabled)
  ├── final_model.pth        # Final model
  └── history.json           # Training history

test_results/
  ├── confusion_matrix.png   # Confusion matrix visualization
  ├── roc_curve.png          # ROC curve
  ├── metrics.png            # Evaluation metrics chart
  ├── metrics.json           # Detailed evaluation metrics
  └── error_analysis/        # Error analysis (if enabled)
```

## Differences from Original Model

This project is based on the original TinyML contest model, with the following adjustments:

1. Migrated from TensorFlow to PyTorch
2. Adjusted input sequence length to 1002 (original was 1250)
3. Maintained core architecture (1D-CNN + MLP)
4. Implemented similar data augmentation strategies (signal flipping, noise addition)
5. Implemented Stochastic Weight Averaging (SWA) and cosine annealing scheduling
6. Added more detailed evaluation metrics and error analysis