#!/bin/bash -l



setting="$1"
interactive="i"
if [[ "$setting" == "$interactive" ]]; then
    partition='gpuA40x4-interactive'
    # partition='gpuA100x4-interactive'
    # partition='gpuA100x8-interactive'
    time='1:00:00'
    num_gpus=2
    num_gpus=4
else
    partition='gpuA40x4'
    # partition='gpuA100x4'
    # partition='gpuA100x8'
    # time='2-00:00:00'
    time='35:00:00'
    num_gpus=4
fi
memory=240g

project_root=/scratch/bbyl/amosharrof/ZSToD

d_folder=$(date +'%Y-%m-%d')
slurm_folder_base=slurm_out/pseudo_labels
mkdir -p $slurm_folder_base/$d_folder
slurm_folder=$slurm_folder_base/$d_folder



sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time # Time limit for the job (REQUIRED).
#SBATCH --job-name=pseudo # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH -e $slurm_folder/%j.err # Error file for this job.
#SBATCH -o $slurm_folder/%j.out # Output file for this job.
#SBATCH -A bbyl-delta-gpu # Project allocation account name (REQUIRED)
#SBATCH --partition=$partition # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'



source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /scratch/bbyl/amosharrof/envs

# cd $project_root
# python src/data_exploration/schema_pseudo_labels.py

# cd $project_root/data/dstc8-schema-guided-dialogue
# python -m sgd_x.generate_sgdx_dialogues

cd $project_root


export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/delta-2022-03/apps/libaio/0.3.110-gcc-11.2.0-sht3clf/lib/
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export WANDB__SERVICE_WAIT=300
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda/lib


time accelerate launch --mixed_precision=bf16  --num_processes=$num_gpus --main_process_port=29503  src/my_trainers/probing_trainer.py --config-name pseudo_trainer
# time accelerate launch  --num_processes=$num_gpus --main_process_port=29503  src/my_trainers/probing_trainer.py --config-name probing_trainer


exit 0
EOT
