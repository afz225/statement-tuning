#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00

declare -a spc_list=(1000 2000 3000 4000 5000 10000 20000 40000 50000)
SHOTS="full,1000,500,200,32,0"
MODEL="roberta-base"

for SPC in "${spc_list[@]}"
do
    EXPERIMENT_NAME="${SPC}-SPC"
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --spc $SPC \
            --experiment_name $EXPERIMENT_NAME \
            --n_shots $SHOTS \
            --model $MODEL 
done