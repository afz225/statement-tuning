#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 96:00:00

declare -a spc_list=( 40000 )
SHOTS="full,1000,500,200,32,0"
MODEL="roberta-large"

for SPC in "${spc_list[@]}"
do
    EXPERIMENT_NAME="${SPC}-SPC-large"
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --n_runs 1 \
            --max_epochs 8 \
            --spc $SPC \
            --batch_size 8 \
            --patience 5 \
            --experiment_name $EXPERIMENT_NAME \
            --n_shots $SHOTS \
            --skip_eval \
            --model $MODEL 
done