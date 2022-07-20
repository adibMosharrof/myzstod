#! /bin/bash

#SBATCH --time=45:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=my_test_job # Job name
#SBATCH --ntasks=1 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH --partition=V4V16_SKY32M192_L # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err # Error file for this job.
#SBATCH -o slurm-%j.out # Output file for this job.
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:1

source activate ./envs
time python src/trainer.py --config-name simple_tod_trainer_full_lcc