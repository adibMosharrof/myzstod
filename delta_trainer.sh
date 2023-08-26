#! /bin/bash

#SBATCH --time=2-00:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=gt_trainer # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH --partition=gpuA40x4 # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm_out/%j.err # Error file for this job.
#SBATCH -o slurm_out/%j.out # Output file for this job.
#SBATCH -A bbyl-delta-gpu # Project allocation account name (REQUIRED)
#SBATCH --gpus-per-node=1
#SBATCH --constraint='projects'

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda  activate ./envs
# time python src/trainer.py --config-name delta_trainer_lite
time python src/trainer.py --config-name delta_arithmetic_trainer
