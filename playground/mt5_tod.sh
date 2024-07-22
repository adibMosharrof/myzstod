#!/bin/bash -l

setting="$1"
interactive="i"
if [[ "$setting" == "$interactive" ]]; then
    partition='gpuA40x4-interactive'
    time='1:00:00'
    num_gpus=2
else
    partition='gpuA40x4'
    # partition='gpuA100x8'
    time='2-00:00:00'
    num_gpus=2
fi

memory=200g
num_nodes=2

d_folder=$(date +'%Y-%m-%d')
slurm_folder_base=slurm_out/tod
mkdir -p $slurm_folder_base/$d_folder
slurm_folder=$slurm_folder_base/$d_folder




sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time # Time limit for the job (REQUIRED).
#SBATCH --job-name=t5todplayground # Job name
#SBATCH --nodes=$num_nodes
#SBATCH --ntasks-per-node=$num_gpus # Number of cores for the job. Same as SBATCH -n 8
#SBATCH -e $slurm_folder/%j.err # Error file for this job.
#SBATCH -o $slurm_folder/%j.out # Output file for this job.
#SBATCH -A bbyl-delta-gpu # Project allocation account name (REQUIRED)
#SBATCH --partition=$partition # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /scratch/bbyl/amosharrof/envs

cd /scratch/bbyl/amosharrof/ZSToD

export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/delta-2022-03/apps/libaio/0.3.110-gcc-11.2.0-sht3clf/lib/
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export WANDB__SERVICE_WAIT=300
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda/lib

rm -f test.txt myhostfile .deepspeed_env
srun hostname | uniq > test.txt
MASTER_ADDR=$(head -n 1 test.txt)


python playground/create_hostfile.py --num_nodes $num_nodes --num_gpus $num_gpus 
# time accelerate launch --num_processes $((num_nodes*num_gpus)) --num_machines=$num_nodes --mixed_precision=bf16 --main_process_ip=$(head -n 1 test.txt) --main_process_port 29500 playground/t5_tod.py --config-name t5_trainer
time deepspeed --hostfile=myhostfile --master_addr=$(head -n 1 test.txt) --launcher SLURM playground/t5_tod.py --config-name t5_trainer



# python -u -m torch.distributed.run --nproc_per_node $num_gpus --nnodes $num_nodes --rdzv_endpoint $(head -n 1 test.txt):6000 --rdzv_backend c10d --max_restarts 0 --tee 3  playground/t5_tod.py --config-name t5_trainer --data-impl mmap --distributed-backend nccl

# exit 0
EOT
