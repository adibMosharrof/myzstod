#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
time python src/inference.py