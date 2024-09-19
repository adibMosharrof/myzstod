#!/bin/bash -l
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
# conda activate ../quantiazation/envs/
# outdir=`time python src/trainer.py --config-name simple_tod_trainer_full | tail -n 1`
# outdir=$(date +'%Y-%m-%d/%H-%M-%S')
# time python -m torch.distributed.launch --use_env --nproc_per_node 1 src/trainer.py --config-name simple_tod_trainer
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0

config_name="simple_tod_trainer"
if [[ "$1" == "woz21" ]]
then
    config_name="multi_woz_21"
else
    config_name="simple_tod_trainer"
fi
# time python src/trainer.py --config-name multi_woz_21
# time python src/trainer.py --config-name $config_name

# deepspeed --no_local_rank src/trainer.py --config-name $config_name --deepspeed config/ds_config.json
deepspeed --no_local_rank src/trainer.py --config-name $config_name 

# echo $outdir
# cd ../google-research

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate schema_guided_dst/envs
# python -m schema_guided_dst.evaluate \
# --dstc8_data_dir /mounts/u-amo-d0/grad/adibm/projects/generative_tod/data/dstc8-schema-guided-dialogue \
# --prediction_dir $outdir/reconstruct/ --eval_set test \
# --output_metric_file $outdir/sgd_results.json