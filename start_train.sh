#!/bin/bash

echo "================================"
echo "  Nano Simple Audio Model Training"
echo "================================"

export CUDA_VISIBLE_DEVICES=0

python -u train.py 2>&1 | tee training.log

echo ""
echo "Training completed! Logs saved in training.log"
