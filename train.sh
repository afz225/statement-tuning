#!/bin/bash

python3 -m st -m train \
            --cache_dir "/scratch/afz225/.cache" \
            --exclude "qqp,winogrande,piqa,mnli,snli,mintaka,squad,yelp_polarity,wikilingua,squad,offensive,massive,dpr,qasc,sciq,samsum," \