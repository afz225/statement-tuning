#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 96:00:00

python3 -m st -m evaluate \
            --cache_dir "/scratch/afz225/.cache" \
            --spc 10000 \
            --experiment_name "10000-SPC-large" \
            --n_runs 1 \
            --n_shots "0" \
            --model "roberta-large"
