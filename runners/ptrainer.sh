#!/bin/bash -l



partition='V4V32_CAS40M192_L'
partition='V4V32_SKY32M192_L'
partition='A2V80_ICE56M256_L'
num_gpus=2
time='3-00:00:00'

# memory=200g
memory=180g

project_root=/project/msi290_uksr/generative_tod

d_folder=$(date +'%Y-%m-%d')
slurm_folder_base=slurm_out/probing
mkdir -p $slurm_folder_base/$d_folder
slurm_folder=$slurm_folder_base/$d_folder



sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time # Time limit for the job (REQUIRED).
#SBATCH --job-name=t5todplayground # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH -e $slurm_folder/%j.err # Error file for this job.
#SBATCH -o $slurm_folder/%j.out # Output file for this job.
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --partition=$partition # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gres=gpu:$num_gpus



source ~/miniconda3/etc/profile.d/conda.sh
conda activate /project/msi290_uksr/generative_tod/envs

# cd $project_root
# python src/data_exploration/schema_pseudo_labels.py

# cd $project_root/data/dstc8-schema-guided-dialogue
# python -m sgd_x.generate_sgdx_dialogues

cd $project_root


export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export WANDB__SERVICE_WAIT=300


time accelerate launch --mixed_precision=bf16  --num_processes=$num_gpus --main_process_port=29503  src/my_trainers/probing_trainer.py --config-name probing_trainer
# time accelerate launch --mixed_precision=fp16  --num_processes=$num_gpus --main_process_port=29503  src/my_trainers/probing_trainer.py --config-name probing_trainer

exit 0
EOT
