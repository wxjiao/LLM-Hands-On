#!/bin/bash
#set -x
#
source ~/.bashrc

# Previous env
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=eth1
#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_SL=3
#export NCCL_NET_GDR_READ=1


# RDMA env
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"  # port different from remote submit
#export RANK=$OMPI_COMM_WORLD_RANK
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
#export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE


Proj_path=[YOUR_WORKSPACE]
Code_path=${Proj_path}/mytrain/llms_scripts
Model_path=${Proj_path}/FinetuneModels/Meta-Llama-3-8B-sft
#

Data_path=${Proj_path}/Dataset/dpo_chat
Train_path=${Data_path}/dpo_train.json
Valid_path=${Data_path}/dpo_eval.json

#
Output_path=${Proj_path}/FinetuneModels/Meta-Llama-3-8B-sft-dpo
# original 16:=4 x 1 x 4

accelerate launch --config_file ${Code_path}/deepspeed_zero3.yaml \
    ${Code_path}/llms_dpo.py \
    --model_name_or_path ${Model_path} \
    --bf16 True \
    --train_file ${Train_path} \
    --eval_file ${Valid_path} \
    --output_dir ${Output_path} \
    --per_device_train_batch_size 1 \
    --max_prompt_length 512 \
    --max_length 1024 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --eval_steps 50 \
    --max_steps 2000 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --output_dir=${Output_path} \
    --warmup_steps 10 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
    --attn_implementation=flash_attention_2 \
    --gradient_checkpointing True

