#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
# outdir=`time python src/trainer.py --config-name simple_tod_trainer_full | tail -n 1`
# outdir=$(date +'%Y-%m-%d/%H-%M-%S')
time python -m torch.distributed.launch --use_env --nproc_per_node 1 src/trainer.py --config-name simple_tod_trainer
# echo $outdir
# cd ../google-research

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate schema_guided_dst/envs
# python -m schema_guided_dst.evaluate \
# --dstc8_data_dir /mounts/u-amo-d0/grad/adibm/projects/generative_tod/data/dstc8-schema-guided-dialogue \
# --prediction_dir $outdir/reconstruct/ --eval_set test \
# --output_metric_file $outdir/sgd_results.json