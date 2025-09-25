#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Pass_scienceqa_textvqa_vizwiz_gqa/global_step_211/actor/huggingface # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/rollout_sft/pass_dataset/geo/train-00000-of-00001.parquet@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.ref.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=Pass_scienceqa_textvqa_vizwiz_gqa_geo \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=10


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/mix_5_geo.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/pass_5_geo.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/Pass_scienceqa_textvqa_vizwiz_gqa_geo/global_step_30/actor

# /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Mix_scienceqa_textvqa/global_step_33/actor/huggingface