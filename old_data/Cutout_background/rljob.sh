#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J cutout_back_p2
#BSUB -n 1
#BSUB -W 23:45
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u s183742@student.dtu.dk

# Load modules
module swap python3/3.8.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg

echo "Running script..."
python3 regularizer_cutout_background.py

