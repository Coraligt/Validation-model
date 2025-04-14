#!/bin/bash

echo "Step 1: Preparing data..."
python prepare_data.py \
    --data_dir ./dataset_3 \
    --output_dir ./indices \
    --val_ratio 0.20 \
    --test_ratio 0.10 \

# 2. Train baseline model
echo "Step 2: Training baseline model..."
echo "Running baseline model training (TinyML contest multi-layer CNN architecture)"
python Valid/trainselect.py \
    --data_dir ./indices/preprocessed \
    --indices_dir ./indices \
    --output_dir ./model_output/baseline \
    --model_type baseline \
    --do_augmentation False \
    --use_swa False \
    --epochs 60 \
    --batch_size 32 \
    --device "cuda:0"

# 3. Train improved model with multiple runs to find best configuration
echo "Step 3: Training improved model with multiple runs..."
python trainselect.py \
    --data_dir ./indices/preprocessed \
    --indices_dir ./indices \
    --output_dir ./model_output/improved \
    --model_type improved \
    --do_augmentation \
    --use_swa \
    --swa_start 10 \
    --swa_freq 5 \
    --swa_lr 0.0001 \
    --num_runs 5 \
    --random_hyperparams \
    --epochs 60 \
    --device cuda:0

# 4. Evaluate both models
echo "Step 4: Evaluating baseline model..."
python evaluate.py \
    --data_dir ./indices/preprocessed \
    --indices_dir ./indices \
    --model_path ./model_output/baseline/run_0/best_model_fb.pth \
    --output_dir ./evaluation_results/baseline \
    --device cuda:0

echo "Step 5: Evaluating improved model..."
python evaluate.py \
    --data_dir ./indices/preprocessed \
    --indices_dir ./indices \
    --model_path ./model_output/improved/best_model_fb.pth \
    --output_dir ./evaluation_results/improved \
    --device cuda:0