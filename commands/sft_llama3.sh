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
Model_path=${Proj_path}/PreTrainModels/Meta-Llama-3-8B
#

Data_path=${Proj_path}/Dataset/sft_chat
Train_path=${Data_path}/joint_train.json
Valid_path=${Data_path}/joint_eval.json

#
Output_path=${Proj_path}/FinetuneModels/Meta-Llama-3-8B-sft
# original 128:=8 x 4 x 8; larger batch 512:=16 x 4 x 8

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${Code_path}/llms_sft.py \
    --deepspeed ${Code_path}/deepspeed_config.json \
    --model_name_or_path ${Model_path} \
    --train_file ${Train_path} \
    --validation_file ${Valid_path} \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 1800 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --block_size 2048 \
    --do_train \
    --evaluation_strategy "no" \
    --bf16 True \
    --streaming True \
    --accelerator_config ${Code_path}/accelerator_config.json \
    --dataloader_prefetch_factor 0 \
    --ddp_timeout 7200 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${Output_path} \
    |& tee ${Output_path}.train.log

