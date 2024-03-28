#!/bin/bash -l


# partition='gpuA100x4'
# partition='gpuA100x4-interactive'
partition='gpuA40x4'
# partition='gpuA100x8'
num_gpus=2
# num_gpus=8
memory=200g
time='2-00:00:00'
# time='1:00:00'


d_folder=$(date +'%Y-%m-%d')
slurm_folder_base=slurm_out/tod
mkdir -p $slurm_folder_base/$d_folder
slurm_folder=$slurm_folder_base/$d_folder



sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time # Time limit for the job (REQUIRED).
#SBATCH --job-name=t5todplayground # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH -e $slurm_folder/%j.err # Error file for this job.
#SBATCH -o $slurm_folder/%j.out # Output file for this job.
#SBATCH -A bbyl-delta-gpu # Project allocation account name (REQUIRED)
#SBATCH --partition=$partition # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='projects'


source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
# conda  activate ./envs
# source ~/miniconda/etc/profile.d/conda.sh
#conda  activate /tmp/.venv
conda activate /scratch/bbyl/amosharrof/envs

cd /projects/bbyl/amosharrof/ZSToD

export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/delta-2022-03/apps/libaio/0.3.110-gcc-11.2.0-sht3clf/lib/
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export WANDB__SERVICE_WAIT=300
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda/lib

# accelerate launch --multi_gpu --num_processes=$num_gpus  playground/t5_tod.py
# accelerate launch --multi_gpu --mixed_precision=no --num_processes=$num_gpus  playground/t5_tod.py
# accelerate launch --multi_gpu --mixed_precision=bf16  --num_processes=$num_gpus  playground/t5_tod.py
# time accelerate launch --mixed_precision=bf16  --num_processes=$num_gpus  playground/t5_tod.py --config-name t5_inference
time accelerate launch --mixed_precision=bf16  --num_processes=$num_gpus  playground/t5_tod.py --config-name t5_trainer

exit 0
EOT
