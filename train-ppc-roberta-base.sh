#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 96:00:00
    

declare -a ppc_list=( 3 5 )
SHOTS="0"
MODEL="roberta-base"

for PPC in "${ppc_list[@]}"
do
    EXPERIMENT_NAME="${PPC}-PPC"
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --ppc $PPC \
            --experiment_name $EXPERIMENT_NAME \
            --n_shots $SHOTS \
            --model $MODEL 
done