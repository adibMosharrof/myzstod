#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
time python src/trainer.py --config-name simple_tod_trainer_full 