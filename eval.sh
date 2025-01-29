#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 96:00:00

# python3 -m st -m evaluate \
#             --cache_dir "/scratch/afz225/.cache" \
#             --spc 50000 \
#             --experiment_name "50000-SPC" \
#             --n_shots full,1000,500,200,32,0 \
#             --model "roberta-base"

python3 -m st -m evaluate \
            --cache_dir "/scratch/afz225/.cache" \
            --experiment_name "samsum,wikilingua-EXCLUDE" \
            --exclude "samsum,wikilingua" \
            --n_shots "0" \
            --max_epochs 15 \
            --patience 5 \
            --n_runs 5 \
            --spc 7143 \
            --model "roberta-base" 
