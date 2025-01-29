#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 96:00:00

declare -a spc_list=( 50000 )
SHOTS="full,1000,500,200,32,0"
MODEL="roberta-base"

for SPC in "${spc_list[@]}"
do
    EXPERIMENT_NAME="${SPC}-SPC"
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --n_runs 2 \
            --spc $SPC \
            --max_epochs 6 \
            --patience 3 \
            --skip_eval \
            --experiment_name $EXPERIMENT_NAME \
            --n_shots $SHOTS \
            --model $MODEL 
done