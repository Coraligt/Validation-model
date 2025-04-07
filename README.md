# Semiconductor Leakage Detection

A deep learning-based system for detecting leakage in semiconductor devices using time-series data, adapted from the GATECH-EIC LAB's TinyML contest model of for IEGM classification.

## Overview

This project implements a 1D CNN architecture to detect leakage in semiconductor devices based on charge (q) time-series data. The model analyzes device characterization data to classify semiconductor devices as either leaky (1) or non-leaky (0), providing critical quality assurance for semiconductor manufacturing.

## Project Structure


```
Training/ Original model from EIC LAB fro IEGM classification
Train/ Original model of PyTorch version for IEGM classification
Valid/
├── __pycache__/
├── Code/
├── dataset_3/           # Original semiconductor data in CSV format
├── evaluation_results/  # Model evaluation output
├── indices/             # Train/validation/test splits
│   └── preprocessed/    # Normalized data
├── model_output/        # Trained models and results
│   ├── run_0/
│   ├── run_1/
│   └── ...
├── Preversion/
├── tuning_results/      # Hyperparameter tuning results
├── cosine_annealing.py  # Learning rate scheduler implementation
├── dataset.py           # Data loading and preprocessing utilities
├── evaluate.py          # Model evaluation script
├── hyperparameter_tuning.py  # Automated hyperparameter optimization
├── inference.py         # Model inference on new data
├── main.py              # Main entry point for training pipeline
├── model.py             # Neural network architecture definition
├── prepare_data.py      # Data preparation and preprocessing
├── swa.py               # Stochastic Weight Averaging implementation
├── train.py             # Model training script
├── trainselect.py       # Enhanced training with model selection
└── utils.py             # Utility functions for metrics calculation
```

## Environment Setup

### Option 1: GPU Environment 

Create a conda environment using the provided environment.yml file:

1. First, create an environment.yml file with the following content:

```yaml
name: semiconductor_env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch=1.12.1
  - torchvision
  - cudatoolkit=11.3
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - tqdm
  - pip
  - pip:
    - tensorboard
```

2. Create the environment:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate semiconductor_env
```

For PACE cluster users, you might need to use:

```bash
source activate semiconductor_env
```

If you encounter issues with modules, try loading necessary modules:

```bash
module load anaconda3
module load cuda/12.1  # or whatever version you need
```

### Option 2: CPU-only Environment

If you don't have a GPU, you can create a CPU-only environment:

```bash
conda create -n semiconductor_env python=3.8
conda activate semiconductor_env
conda install pytorch cpuonly -c pytorch
conda install -c conda-forge numpy pandas scikit-learn matplotlib tqdm
pip install tensorboard
```

### Option 3: Using pip (CPU or GPU)

```bash
# Create a virtual environment
python -m venv semiconductor_env
source semiconductor_env/bin/activate  # On Windows: semiconductor_env\Scripts\activate

# Install dependencies for CPU
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib tqdm tensorboard

# For GPU, install the appropriate PyTorch version from https://pytorch.org/get-started/locally/
```

## Data Preparation

Before training the model, you need to prepare your semiconductor device data:

1. Organize your CSV files in the following format:
   - Each device should have a CSV file named as `dev#_label.csv` where # is the device ID and label is 0 (non-leaky) or 1 (leaky)
   - Each CSV file should contain columns for time (t), voltage (v), charge (q), and current (i)

2. Run the data preparation script:

```bash
python Valid/prepare_data.py --data_dir ./Valid/dataset_3 --output_dir ./Valid/indices
```

This script will:
- Analyze the dataset characteristics
- Split the data into train/validation/test sets
- Normalize the data
- Create index files for training

## Training

The training pipeline includes several options:

### Basic Training

```bash
python Valid/main.py --data_dir ./Valid/dataset_3 --output_dir ./Valid --model_dir ./Valid/model_output
```

### Advanced Training with Multiple Runs and Model Selection

```bash
python Valid/trainselect.py --data_dir ./Valid/indices/preprocessed --indices_dir ./Valid/indices --output_dir ./Valid/model_output --num_runs 5 --use_swa
```

Important parameters:
- `--use_swa`: Enable Stochastic Weight Averaging
- `--do_augmentation`: Apply data augmentation
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--device`: Specify CPU or GPU (e.g., "cuda:0" or "cpu")

## Hyperparameter Tuning

To find optimal hyperparameters:

```bash
python Valid/hyperparameter_tuning.py --data_dir ./Valid/indices/preprocessed --indices_dir ./Valid/indices --output_dir ./Valid/tuning_results --n_iter 20
```

## Evaluation

Evaluate a trained model:

```bash
python Valid/evaluate.py --data_dir ./Valid/indices/preprocessed --indices_dir ./Valid/indices --model_path ./Valid/model_output/best_model_fb.pth --output_dir ./Valid/evaluation_results
```

## Inference

Run inference on new semiconductor device data:

```bash
python Valid/inference.py --model_path ./Valid/model_output/best_model_fb.pth --input_dir ./path/to/new/data --output_file predictions.csv
```

## Model Architecture

The model architecture consists of:
1. 1D convolutional layer with configurable filters (default: 3)
2. Batch normalization
3. ReLU activation
4. Flattening layer
5. Two fully connected layers (default sizes: 20 and 10) with dropout
6. Output layer for binary classification (leaky vs non-leaky)

## Performance Metrics

The model is evaluated using:
- Accuracy: Overall classification accuracy
- F-Beta Score (FB): Weighted F-score (β=2 by default) prioritizing recall
- Sensitivity: True positive rate
- Specificity: True negative rate
- PPV: Positive predictive value (precision)
- NPV: Negative predictive value

## Recommended Workflow

1. Prepare your semiconductor device data
2. Run hyperparameter tuning to find optimal configuration
3. Train multiple models with the best hyperparameters
4. Select the best model based on FB score
5. Evaluate the model on the test set
6. Deploy for inference on new devices

## License



## Acknowledgments

This project adapts the architecture from a TinyML contest model originally designed by GATECH EIC LAB for IEGM signal analysis to the domain of semiconductor device characterization.
