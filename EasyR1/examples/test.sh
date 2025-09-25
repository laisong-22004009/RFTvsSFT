set -x

#base
MODEL_PATH0=/scratch/esg8sdce/pre_trained_models/Qwen2.5-VL-7B-Instruct

#grpo
MODEL_PATH1=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface
#grpo_no_cot
MODEL_PATH2=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/NO-COT_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface
#grpo_no_kl
MODEL_PATH3=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/NOKL_scienceqa_textvqa_vizwiz_gqa/global_step_120/actor/huggingface
#sft
MODEL_PATH4=/scratch/esg8sdce/pre_trained_models/Qwen2.5-VL-7B-Instruct_scienceqa_textvqa_vizwiz_gqa_geo_path-vqa_clevr
#remax
MODEL_PATH5=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface
#rloo
MODEL_PATH6=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface
#joint_sft
MODEL_PATH7=/scratch/esg8sdce/pre_trained_models/Qwen2.5-VL-7B-Instruct_multitask

DATA_PATH0=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/MMMU/test-00000-of-00001.parquet@test
DATA_PATH1=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/MMLU-Pro/test-00000-of-00001.parquet@test
DATA_PATH2=lmms-lab/POPE@test


# # COT

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH0} \
# #     trainer.experiment_name=COT_base_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH0} \
# #     trainer.experiment_name=COT_base_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH0} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_base_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH1} \
# #     trainer.experiment_name=COT_grpo_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH1} \
# #     trainer.experiment_name=COT_grpo_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_grpo_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH2} \
# #     trainer.experiment_name=COT_grpo_no_cot_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH2} \
# #     trainer.experiment_name=COT_grpo_no_cot_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH2} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_grpo_no_cot_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH3} \
# #     trainer.experiment_name=COT_grpo_no_kl_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH3} \
# #     trainer.experiment_name=COT_grpo_no_kl_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH3} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_grpo_no_kl_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH4} \
# #     trainer.experiment_name=COT_sft_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH4} \
# #     trainer.experiment_name=COT_sft_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH4} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_sft_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35


# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH5} \
# #     trainer.experiment_name=COT_remax_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH5} \
# #     trainer.experiment_name=COT_remax_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH5} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_remax_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH6} \
# #     trainer.experiment_name=COT_rloo_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH6} \
# #     trainer.experiment_name=COT_rloo_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH6} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_rloo_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH7} \
# #     trainer.experiment_name=COT_joint_on_mmmu \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH7} \
# #     trainer.experiment_name=COT_joint_on_mmlu-pro \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH7} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=COT_joint_on_pope \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # NO COT
# # NO COT
# # NO COT
# # NO COT
# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH0} \
# #     trainer.experiment_name=NOCOT_base_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH0} \
# #     trainer.experiment_name=NOCOT_base_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH0} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_base_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH1} \
# #     trainer.experiment_name=NOCOT_grpo_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH1} \
# #     trainer.experiment_name=NOCOT_grpo_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_grpo_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH2} \
# #     trainer.experiment_name=NOCOT_grpo_no_cot_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH2} \
# #     trainer.experiment_name=NOCOT_grpo_no_cot_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH2} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_grpo_no_cot_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH3} \
# #     trainer.experiment_name=NOCOT_grpo_no_kl_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH3} \
# #     trainer.experiment_name=NOCOT_grpo_no_kl_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH3} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_grpo_no_kl_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH4} \
# #     trainer.experiment_name=NOCOT_sft_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH4} \
# #     trainer.experiment_name=NOCOT_sft_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH4} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_sft_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH5} \
# #     trainer.experiment_name=NOCOT_remax_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH5} \
# #     trainer.experiment_name=NOCOT_remax_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH5} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_remax_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1


# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH6} \
# #     trainer.experiment_name=NOCOT_rloo_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH6} \
# #     trainer.experiment_name=NOCOT_rloo_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH6} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_rloo_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1


# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH0}  \
# #     worker.actor.model.model_path=${MODEL_PATH7} \
# #     trainer.experiment_name=NOCOT_joint_on_mmmu \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1

# # sleep 35

# # python3 -m verl.trainer.main \
# #     config=examples/test.yaml \
# #     data.train_files=hiyouga/geometry3k@test \
# #     data.val_files=${DATA_PATH1}  \
# #     worker.actor.model.model_path=${MODEL_PATH7} \
# #     trainer.experiment_name=NOCOT_joint_on_mmlu-pro \
# #     data.format_prompt=./examples/format_prompt/no_cot.jinja \
# #     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
# #     trainer.n_gpus_per_node=8 \
# #     worker.actor.fsdp.torch_dtype=bf16 \
# #     trainer.total_epochs=1


# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=${DATA_PATH2} \
#     data.val_files=${DATA_PATH2}  \
#     worker.actor.model.model_path=${MODEL_PATH7} \
#     data.image_key=image \
#     data.prompt_key=question \
#     trainer.experiment_name=NOCOT_joint_on_pope \
#     data.format_prompt=./examples/format_prompt/no_cot.jinja \
#     worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1




##############################################################################################################





# ReMax
MODEL_PATH_0='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa/global_step_135/actor/huggingface'
MODEL_PATH_1='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz/global_step_80/actor/huggingface'
MODEL_PATH_2='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz_gqa/global_step_156/actor/huggingface'
MODEL_PATH_3='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz_gqa_geo/global_step_80/actor/huggingface'
MODEL_PATH_4='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_38/actor/huggingface'
MODEL_PATH_5='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/ReMax_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface'

#RLoo
MODEL_PATH_6='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa/global_step_135/actor/huggingface'
MODEL_PATH_7='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz/global_step_80/actor/huggingface'
MODEL_PATH_8='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz_gqa/global_step_156/actor/huggingface'
MODEL_PATH_9='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz_gqa_geo/global_step_40/actor/huggingface'
MODEL_PATH_10='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_38/actor/huggingface'
MODEL_PATH_11='/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/rloo_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor/huggingface'
# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH7} \
#     trainer.experiment_name=TEST_MIX_vizwiz_rl_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_0} \
#     trainer.experiment_name=TEST_remax_textvqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_1} \
#     trainer.experiment_name=TEST_remax_vizwiz_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_1} \
#     trainer.experiment_name=TEST_remax_vizwiz_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_2} \
#     trainer.experiment_name=TEST_remax_gqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_2} \
#     trainer.experiment_name=TEST_remax_gqa_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_2} \
#     trainer.experiment_name=TEST_remax_gqa_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_3} \
#     trainer.experiment_name=TEST_remax_geo_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_3} \
#     trainer.experiment_name=TEST_remax_geo_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_3} \
#     trainer.experiment_name=TEST_remax_geo_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH_3} \
#     trainer.experiment_name=TEST_remax_geo_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_4} \
#     trainer.experiment_name=TEST_remax_pathvqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_4} \
#     trainer.experiment_name=TEST_remax_pathvqa_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_4} \
#     trainer.experiment_name=TEST_remax_pathvqa_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH_4} \
#     trainer.experiment_name=TEST_remax_pathvqa_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test \
#     worker.actor.model.model_path=${MODEL_PATH_4} \
#     trainer.experiment_name=TEST_remax_pathvqa_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=flaviagiammarino/path-vqa@test \
#     data.val_files=flaviagiammarino/path-vqa@test \
#     worker.actor.model.model_path=${MODEL_PATH_5} \
#     trainer.experiment_name=TEST_remax_clevr_on_pathvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=question \
#     data.image_key=image \
#     trainer.total_epochs=1






# sleep 35






# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_6} \
#     trainer.experiment_name=TEST_rloo_textvqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_7} \
#     trainer.experiment_name=TEST_rloo_vizwiz_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_7} \
#     trainer.experiment_name=TEST_rloo_vizwiz_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_8} \
#     trainer.experiment_name=TEST_rloo_gqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_8} \
#     trainer.experiment_name=TEST_rloo_gqa_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_8} \
#     trainer.experiment_name=TEST_rloo_gqa_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_9} \
#     trainer.experiment_name=TEST_rloo_geo_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_9} \
#     trainer.experiment_name=TEST_rloo_geo_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_9} \
#     trainer.experiment_name=TEST_rloo_geo_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


python3 -m verl.trainer.main \
    config=examples/test.yaml \
    data.train_files=hiyouga/geometry3k@test \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH_9} \
    trainer.experiment_name=TEST_rloo_geo_on_gqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_10} \
#     trainer.experiment_name=TEST_rloo_pathvqa_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_10} \
#     trainer.experiment_name=TEST_rloo_pathvqa_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_10} \
#     trainer.experiment_name=TEST_rloo_pathvqa_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH_10} \
#     trainer.experiment_name=TEST_rloo_pathvqa_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test \
#     worker.actor.model.model_path=${MODEL_PATH_10} \
#     trainer.experiment_name=TEST_rloo_pathvqa_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=flaviagiammarino/path-vqa@test \
#     data.val_files=flaviagiammarino/path-vqa@test \
#     worker.actor.model.model_path=${MODEL_PATH_11} \
#     trainer.experiment_name=TEST_rloo_clevr_on_pathvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=question \
#     data.image_key=image \
#     trainer.total_epochs=1



















# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=flaviagiammarino/path-vqa@test  \
#     data.val_files=flaviagiammarino/path-vqa@test   \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_pathvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=question \
#     data.image_key=image \
#     trainer.total_epochs=1

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test  \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=TEST_simple_clevr \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=problem \
#     data.image_key=image \
#     trainer.total_epochs=1




# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=flaviagiammarino/path-vqa@test  \
#     data.val_files=flaviagiammarino/path-vqa@test   \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=remax_on_pathvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=question \
#     data.image_key=image \
#     trainer.total_epochs=1

# sleep 35


# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_scienceqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_textvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_vizwiz \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_gqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=hiyouga/geometry3k@test \
#     data.val_files=hiyouga/geometry3k@test  \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_geo \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     trainer.total_epochs=1

# sleep 35

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=flaviagiammarino/path-vqa@test  \
#     data.val_files=flaviagiammarino/path-vqa@test   \
#     worker.actor.model.model_path=${MODEL_PATH1} \
#     trainer.experiment_name=rloo_on_pathvqa \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=question \
#     data.image_key=image \
#     trainer.total_epochs=1

# python3 -m verl.trainer.main \
#     config=examples/test.yaml \
#     data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test  \
#     data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test  \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=TEST_simple_clevr \
#     trainer.n_gpus_per_node=8 \
#     worker.actor.fsdp.torch_dtype=bf16 \
#     data.prompt_key=problem \
#     data.image_key=image \
#     trainer.total_epochs=1







# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/test.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/test.log 2>&1 &

