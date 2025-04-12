# Running Inference on Measured Semiconductor Data

This guide explains how to use your trained model to make predictions on your real measured semiconductor data, handling the differences between your training data (synthetic) and testing data (real measurements).

## Key Features

- Handles different numbers of data points through intelligent resampling
- Works with your specific CSV file format (no headers)
- Processes your summary file (`testdata_10k_summary.txt`) for labels
- Calculates comprehensive metrics and visualizations
- Supports batch processing for fast evaluation of ~300 files

## Prerequisites

- Your trained PyTorch model (`.pth` file)
- Directory with your measured data CSV files (containing time, voltage, charge, current columns)
- Summary file with labels in the format: `label,filename`
- Python environment with required dependencies (PyTorch, numpy, pandas, matplotlib)

## Basic Usage

```bash
python Valid/inference.py \
    --model_path ./Valid/model_output/run_2/best_model_fb.pth \
    --data_dir ./test_data_10kHz \
    --summary_file ./test_data_10kHz/testdata_10k_summary.txt \
    --output_dir ./measured_data_results \
    --create_samples
```

This will:

1. Load your trained model
2. Read all files listed in the summary file
3. Process each file by extracting the charge (q) data, normalizing it, and resampling to match the model's expected input length (1002 points)
4. Run inference in batches for efficiency
5. Calculate and visualize performance metrics
6. Create sample visualizations of correct and incorrect predictions

## Command Line Arguments

```
--model_path       Path to your trained model file (.pth)
--data_dir         Directory containing your measured data files
--summary_file     Path to your summary file with labels (testdata_10k_summary.txt)
--output_dir       Directory to save inference results
--batch_size       Batch size for inference (default: 32)
--feature_column   Column to use as feature (default: q)
--create_samples   Enable creation of sample visualizations
--num_samples      Number of samples to visualize per category (default: 5)
--device           Device to use for inference (cuda or cpu)
--no_cuda          Disable CUDA even if available
```

## Output Files

The script creates several output files in the specified output directory:

- `inference_results.csv`: CSV with predictions for each file
- `metrics.txt`: Text file with performance metrics
- `confusion_matrix.png`: Visualization of the confusion matrix
- `confidence_distribution.png`: Histogram of prediction confidences
- `confidence_by_correctness.png`: Comparison of confidence for correct vs. incorrect predictions
- `sample_visualizations/`: Directory with sample visualizations (if `--create_samples` is specified)

## Notes on Data Format

The script handles your specific data format:

- CSV files with no headers
- Expected columns in order: time (t), voltage (v), charge (q), current (i)
- The script extracts the charge (q) column (usually the 3rd column) for inference
- Each file is individually normalized before resampling
- File naming format: `dev#_frequency_voltage.csv` (the script only cares about the internal data)

## Handling Frequency Differences

Since your measured data might have different frequencies than your training data:

- Files with more than 1002 points are downsampled using interpolation
- Files with fewer than 1002 points are upsampled using interpolation
- This ensures the model receives input of the expected size while preserving signal characteristics

## Example: Running with Multiple GPUs or Splitting Files

If you have many files or a large model, you can split processing:

```bash
# Process first 150 files on GPU 0
python inference_real_data.py \
    --model_path ./model_output/best_model_fb.pth \
    --data_dir ./measured_data_directory \
    --summary_file ./first_half_summary.txt \
    --output_dir ./results_part1 \
    --device cuda:0

# Process remaining files on GPU 1
python inference_real_data.py \
    --model_path ./model_output/best_model_fb.pth \
    --data_dir ./measured_data_directory \
    --summary_file ./second_half_summary.txt \
    --output_dir ./results_part2 \
    --device cuda:1
```

## Troubleshooting

### Missing Files

If files in your summary don't exist in the data directory, the script will skip them and print warnings.

### Different Column Orders

The script assumes columns are in order: time, voltage, charge, current. If your columns are in a different order, modify the `preprocess_file` function.

### Memory Issues

If you encounter memory issues, reduce the batch size:

```bash
python inference_real_data.py --model_path ./your_model.pth --data_dir ./your_data --summary_file ./summary.txt --batch_size 8
```
