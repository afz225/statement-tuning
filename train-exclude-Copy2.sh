#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00

exclude_list=("samsum,wikilingua" "offensive" "massive" "dpr" "yelp_polarity" "mintaka,squad,qasc,sciq,race")
spc_list=(6250 7143 7692 8333 9091 10000 20000)

for ((i=0; i<${#exclude_list[@]}; i++))
do
    EXCLUDE=$(IFS=,; echo "${exclude_list[*]:0:$i}")
    SPC=${spc_list[$i]}
    SHOTS="0"
    MODEL="roberta-base"
    EXPERIMENT_NAME="${EXCLUDE}-EXCLUDE"
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --experiment_name $EXPERIMENT_NAME \
            --exclude $EXCLUDE \
            --n_shots $SHOTS \
            --model $MODEL 
done