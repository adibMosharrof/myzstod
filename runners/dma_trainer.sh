#!/bin/bash -l


if [[ "$1" == "a44" ]]
then
    config_name='da40_4_multi_adapter'
    num_gpus=4
    partition='gpuA40x4'
elif [[ "$1" == "a18" ]]
then
    config_name='da100_8_multi_adapter'
    num_gpus=8
    partition='gpuA100x8'
else
    config_name='delta_trainer'
    partition='gpuA40x4'
    num_gpus=2
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --mem=220g
#SBATCH --time=2-00:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=gt_trainer # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH -e slurm_out/%j.err # Error file for this job.
#SBATCH -o slurm_out/%j.out # Output file for this job.
#SBATCH -A bbyl-delta-gpu # Project allocation account name (REQUIRED)
#SBATCH --constraint='projects'
#SBATCH --partition=$partition # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --gpu-bind=closest

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda  activate ./envs
time deepspeed --no_local_rank src/trainer.py --config-name $config_name 
#time python src/trainer.py --config-name $config_name
#time python src/trainer.py --config-name delta_arithmetic_trainer

exit 0
EOT
