#! /bin/bash

#SBATCH --time=3-00:00:00 # Time limit for the job (REQUIRED).
#SBATCH --job-name=gt_trainer # Job name
#SBATCH --ntasks=8 # Number of cores for the job. Same as SBATCH -n 8
#SBATCH --partition=V4V32_CAS40M192_L # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e task_slurm_out/%j.err # Error file for this job.
#SBATCH -o task_slurm_out/%j.out # Output file for this job.
#SBATCH -A gol_msi290_uksr # Project allocation account name (REQUIRED)
#SBATCH --gres=gpu:2

#module load ccs/Miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.venvs
time python src/trainer.py --config-name lcc_multi_adapter
# time python src/trainer.py --config-name lcc_arithmetic_trainer
# outdir=`python src/trainer.py --config-name lcc_simple_tod_trainer`
# echo $outdir
# cd ../google-research

# conda activate schema_guided_dst/envs
# python -m schema_guided_dst.evaluate \
# --dstc8_data_dir /project/msi290_uksr/generative_tod/data/dstc8-schema-guided-dialogue \
# --prediction_dir $outdir/reconstruct/ --eval_set test \
# --output_metric_file $outdir/sgd_results.json