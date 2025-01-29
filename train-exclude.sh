#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00

exclude_list=("samsum,wikilingua" "offensive" "massive")
spc_list=( 0 7143 7692 8333 )

# for ((i=0; i<${#exclude_list[@]}; i++))
# do
#     EXCLUDE=$(IFS=,; echo "${exclude_list[*]:0:$i}")
#     SPC=${spc_list[$i]}
#     SHOTS="0"
#     MODEL="roberta-base"
#     EXPERIMENT_NAME="${EXCLUDE}-EXCLUDE"
#     python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name $EXPERIMENT_NAME \
#             --exclude $EXCLUDE \
#             --n_shots $SHOTS \
#             --spc $SPC \
#             --model $MODEL 
# done

for ((i=1; i<${#exclude_list[@]}+1; i++))
do
    EXCLUDE=$(IFS=,; echo "${exclude_list[*]:0:$i}")
    SPC=${spc_list[$i]}
    SHOTS="0"
    MODEL="roberta-base"
    EXPERIMENT_NAME="${EXCLUDE}-EXCLUDE"
    echo $EXCLUDE
    echo $SPC
    echo $EXPERIMENT_NAME
    python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --experiment_name $EXPERIMENT_NAME \
            --exclude $EXCLUDE \
            --max_epochs 15 \
            --n_runs 5 \
            --patience 5 \
            --n_shots $SHOTS \
            --spc $SPC \
            --model $MODEL 
done

# for ((i=4; i<${#exclude_list[@]}+1; i++))
# do
#     EXCLUDE=$(IFS=,; echo "${exclude_list[*]:0:$i}")
#     SPC=${spc_list[$i]}
#     SHOTS="0"
#     MODEL="roberta-base"
#     EXPERIMENT_NAME="${EXCLUDE}-EXCLUDE"
#     echo $EXCLUDE
#     echo $SPC
#     echo $EXPERIMENT_NAME
#     python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name $EXPERIMENT_NAME \
#             --exclude $EXCLUDE \
#             --n_shots $SHOTS \
#             --max_epochs 8 \
#             --patience 5 \
#             --spc $SPC \
#             --model $MODEL 
# done

# for ((i=1; i<${#exclude_list[@]}+1; i++))
# do
#     EXCLUDE=$(IFS=,; echo "${exclude_list[*]:0:$i}")
#     SPC=${spc_list[$i]}
#     SHOTS="0"
#     MODEL="roberta-base"
#     EXPERIMENT_NAME="${EXCLUDE}-EXCLUDE"
#     echo $EXCLUDE
#     echo $SPC
#     echo $EXPERIMENT_NAME
#     python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name $EXPERIMENT_NAME \
#             --exclude $EXCLUDE \
#             --n_shots $SHOTS \
#             --spc $SPC \
#             --model $MODEL 
# done