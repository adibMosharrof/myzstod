#!/bin/bash -l
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs

cd /mounts/u-amo-d1/adibm-data/projects/ZSToD

accelerate launch --mixed_precision=no --num_processes=1  playground/t5_tod.py