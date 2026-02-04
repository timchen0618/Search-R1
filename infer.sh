export CUDA_VISIBLE_DEVICES=0,1,2,3
export data_name='musique'
export DATA_DIR=data/${data_name}
export N_GPUS=4

WAND_PROJECT='Search-R1'

export MAX_TURN=8
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export BASE_MODEL='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo'
export TOPK=3

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues


port=$1
retriever_url="http://127.0.0.1:$port/retrieve"
PYTHONUNBUFFERED=1 python3 -m verl.inferencer.inference \
    data.val_files=$DATA_DIR/test_base.parquet \
    data.val_data_num=null \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    max_turns=$MAX_TURN \
    retriever.url=$retriever_url \
    retriever.topk=$TOPK \
    2>&1 | tee $EXPERIMENT_NAME.log
