#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.venvs

python src/data_prep/dstc_base_data_prep.py