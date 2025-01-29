#!/bin/bash

# conda init 

eval "$(conda shell.bash hook)"

conda activate st

export HF_HOME="/share03/afz225/.cache"
export TRANSFORMERS_CACHE="/share03/afz225/.cache"

cd /share03/afz225/statement-tuning

python3 -m st -m evaluate \
            --cache_dir "/share03/afz225/.cache" \
            --spc 10000 \
            --experiment_name "10000-SPC-large" \
            --n_runs 5 \
            --shuffle \
            --n_shots "0" \
            --model "roberta-large"
