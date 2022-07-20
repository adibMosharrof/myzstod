#! /bin/bash

#SBATCH --time=45:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=my_test_job # Job name
#SBATCH --ntasks=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH --partition=V4V32_SKY32M192_L # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm_out/%j.err # Error file for this job.
#SBATCH -o slurm_out/%j.out # Output file for this job.
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:4
module load ccs/Miniconda3
source activate ./envs
time python src/trainer.py --config-name lcc_simple_tod_trainer train_batch_size=5
