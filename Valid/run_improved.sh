#!/bin/bash

echo "Running improved model training (Single-layer CNN architecture)"
python Valid/trainselect.py \
    --data_dir ./Valid/indices/preprocessed \
    --indices_dir ./Valid/indices \
    --output_dir ./Valid/model_output/improved \
    --model_type improved \
    --do_augmentation True \
    --use_swa True \
    --swa_start 10 \
    --swa_freq 5 \
    --swa_lr 0.0001 \
    --epochs 50 \
    --learning_rate 0.0002 \
    --batch_size 32 \
    --device "cuda:0"