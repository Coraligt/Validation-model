# Semiconductor Leakage Detection

This project implements a deep learning model to detect leakage in semiconductor devices using time series data.

## Project Structure

```
semiconductor_leakage/
├── environment.yml             # Conda environment configuration
├── run_on_pace.sh              # Shell script for running on PACE cluster
├── prepare_data.py             # Data preparation script
├── dataset.py                  # Dataset and data loader implementations
├── model.py                    # Model architecture definition
├── utils.py                    # Utility functions
├── cosine_annealing.py         # Cosine annealing learning rate scheduler
├── swa.py                      # Stochastic Weight Averaging implementation
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference script
├── main.py                     # Main entry point
└── hyperparameter_tuning.py    # Hyperparameter tuning script
```

## Dataset Format

The dataset consists of CSV files with 1002 rows and 4 columns:
- `t`: Time
- `v`: Voltage
- `q`: Charge
- `i`: Current

File naming format: `dev#_label.csv` where `label` is 0 for non-leaky and 1 for leaky.

The current implementation uses only the `q` (charge) column for classification, similar to the original IEGM model.

## Running on PACE Cluster

1. Upload all code files to your PACE cluster directory
2. Make the run script executable:
   ```
   chmod +x run_on_pace.sh
   ```
3. Submit the job:
   ```
   qsub run_on_pace.sh
   ```

## Manual Execution Steps

If you prefer to run steps manually:

1. **Create environment**:
   ```
   conda env create -f environment.yml
   conda activate semiconductor_env
   ```

2. **Prepare data**:
   ```
   python prepare_data.py --data_dir ./dataset_3 --output_dir ./Valid
   ```

3. **Train model**:
   ```
   python train.py --data_dir ./Valid/indices/preprocessed \
                   --indices_dir ./Valid/indices \
                   --output_dir ./model_output \
                   --batch_size 64 \
                   --epochs 100 \
                   --learning_rate 0.0002 \
                   --device cuda:0 \
                   --use_swa
   ```

4. **Evaluate model**:
   ```
   python evaluate.py --data_dir ./Valid/indices/preprocessed \
                      --indices_dir ./Valid/indices \
                      --model_path ./model_output/best_model_fb.pth \
                      --output_dir ./evaluation_results \
                      --device cuda:0
   ```

5. **Run inference on new data**:
   ```
   python inference.py --model_path ./model_output/best_model_fb.pth \
                       --input_dir ./new_data \
                       --output_file ./predictions.csv \
                       --device cuda:0
   ```

## Hyperparameter Tuning

To find the best hyperparameters:

```
python hyperparameter_tuning.py --data_dir ./Valid/indices/preprocessed \
                                --indices_dir ./Valid/indices \
                                --output_dir ./tuning_results \
                                --n_iter 20 \
                                --device cuda:0
```

## Model Architecture

The model architecture is a 1D CNN based on the original IEGM model, with the following structure:
- 1D Convolutional layer (kernel size 85, stride 32)
- Batch Normalization
- ReLU activation
- Fully connected layers with ReLU activations and dropout
- Output layer with 2 units (binary classification)
