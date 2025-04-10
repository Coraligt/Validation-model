#!/bin/bash

echo "Running baseline model training (TinyML contest multi-layer CNN architecture)"
python Valid/trainselect.py \
    --data_dir ./Valid/indices/preprocessed \
    --indices_dir ./Valid/indices \
    --output_dir ./Valid/model_output/baseline \
    --model_type baseline \
    --do_augmentation False \
    --use_swa False \
    --epochs 50 \
    --learning_rate 0.0002 \
    --batch_size 32 \
    --device "cuda:0"