#!/bin/bash

# conda init 

eval "$(conda shell.bash hook)"

conda activate st

export HF_HOME="/share03/afz225/.cache"
export TRANSFORMERS_CACHE="/share03/afz225/.cache"

cd /share03/afz225/statement-tuning

gpu=0
for i in {1..5}:
do
    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "None-EXCLUDE-SPC-5000" \
    #             --n_shots "0" \
    #             --n_runs 1 \
    #             --skip_eval \
    #             --max_epochs 20 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base"&

    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "samsum,wikilingua-EXCLUDE-SPC-5000" \
    #             --exclude "samsum,wikilingua" \
    #             --n_shots "0" \
    #             --n_runs 1 \
    #             --max_epochs 20 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base" \
    #             --skip_eval&

    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "samsum,wikilingua,offensive-EXCLUDE-SPC-5000" \
    #             --exclude "samsum,wikilingua,offensive" \
    #             --n_shots "0" \
    #             --max_epochs 20 \
    #             --n_runs 1 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base" \
    #             --skip_eval&

    export CUDA_VISIBLE_DEVICES=$gpu
    python3 -m st -m train \
                --cache_dir "/share03/afz225/.cache" \
                --experiment_name "samsum,wikilingua,offensive,massive-EXCLUDE-SPC-5000" \
                --exclude "samsum,wikilingua,offensive,massive" \
                --n_shots "0" \
                --max_epochs 20 \
                --n_runs 1 \
                --patience 8 \
                --spc 5000 \
                --model "roberta-base" \
                --skip_eval&

    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "samsum,wikilingua,offensive,massive,dpr-EXCLUDE-SPC-5000" \
    #             --exclude "samsum,wikilingua,offensive,massive,dpr" \
    #             --n_shots "0" \
    #             --max_epochs 20 \
    #             --n_runs 1 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base" \
    #             --skip_eval&

    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity-EXCLUDE-SPC-5000" \
    #             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity" \
    #             --n_shots "0" \
    #             --max_epochs 20 \
    #             --n_runs 1 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base" \
    #             --skip_eval&

    # export CUDA_VISIBLE_DEVICES=$gpu
    # python3 -m st -m train \
    #             --cache_dir "/share03/afz225/.cache" \
    #             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race-EXCLUDE-SPC-5000" \
    #             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race" \
    #             --n_shots "0" \
    #             --n_runs 1 \
    #             --max_epochs 20 \
    #             --patience 8 \
    #             --spc 5000 \
    #             --model "roberta-base" \
    #             --skip_eval&

    gpu=$((gpu+1))
    if [ $gpu -eq 8 ]
    then ## sleep for 1.5 minutes
        wait
        gpu=0
    fi
done

wait