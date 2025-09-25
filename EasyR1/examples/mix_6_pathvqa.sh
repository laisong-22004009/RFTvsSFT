#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Pass_scienceqa_textvqa_vizwiz_gqa_geo/global_step_30/actor/huggingface # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/rollout_sft/pass_dataset/pathvqa/train-00000-of-00001.parquet@train \
    data.val_files=flaviagiammarino/path-vqa@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.ref.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=Pass_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    data.prompt_key=question \
    data.image_key=image \
    trainer.total_epochs=1


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/mix_6_pathvqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/pass_6_pathvqa.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/Mix_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_44/actor

