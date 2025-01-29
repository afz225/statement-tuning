#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00


python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --experiment_name "None-EXCLUDE-SPC-5000" \
            --n_shots "0" \
            --n_runs 4 \
            --max_epochs 20 \
            --patience 8 \
            --spc 5000 \
            --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua" \
#             --n_shots "0" \
#             --n_runs 4 \
#             --max_epochs 20 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua,offensive" \
#             --n_shots "0" \
#             --max_epochs 20 \
#             --n_runs 4 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua,offensive,massive" \
#             --n_shots "0" \
#             --max_epochs 20 \
#             --n_runs 4 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr" \
#             --n_shots "0" \
#             --max_epochs 20 \
#             --n_runs 4 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity" \
#             --n_shots "0" \
#             --n_runs 4 \
#             --max_epochs 20 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race-EXCLUDE-SPC-5000" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race" \
#             --n_shots "0" \
#             --n_runs 4 \
#             --max_epochs 20 \
#             --patience 8 \
#             --spc 5000 \
#             --model "roberta-base" 