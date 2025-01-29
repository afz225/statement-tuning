#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -t 72:00:00


python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --experiment_name "samsum,wikilingua-EXCLUDE" \
            --exclude "samsum,wikilingua" \
            --n_shots "0" \
            --max_epochs 20 \
            --patience 8 \
            --n_runs 5 \
            --spc 7143 \
            --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive-EXCLUDE" \
#             --exclude "samsum,wikilingua,offensive" \
#             --n_shots "0" \
#             --max_epochs 15 \
#             --n_runs 5 \
#             --patience 5 \
#             --spc 7692 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive-EXCLUDE" \
#             --exclude "samsum,wikilingua,offensive,massive" \
#             --n_shots "0" \
#             --max_epochs 15 \
#             --n_runs 5 \
#             --patience 5 \
#             --spc 8333 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr-EXCLUDE" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr" \
#             --n_shots "0" \
#             --max_epochs 15 \
#             --n_runs 5 \
#             --patience 5 \
#             --spc 9091 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity-EXCLUDE" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity" \
#             --n_shots "0" \
#             --n_runs 5 \
#             --max_epochs 15 \
#             --patience 5 \
#             --spc 15000 \
#             --model "roberta-base" 

# python3 -m st -m train \
#             --cache_dir "/scratch/afz225/.cache" \
#             --experiment_name "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race-EXCLUDE" \
#             --exclude "samsum,wikilingua,offensive,massive,dpr,yelp_polarity,mintaka,squad,qasc,sciq,race" \
#             --n_shots "0" \
#             --n_runs 5 \
#             --max_epochs 15 \
#             --patience 5 \
#             --spc 20000 \
#             --model "roberta-base" 