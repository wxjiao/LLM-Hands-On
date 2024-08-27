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
export MASTER_PORT="${MASTER_PORT:=29501}"
#export RANK=$OMPI_COMM_WORLD_RANK
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
#export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE


Proj_path=[YOUR_WORKSPACE]
Code_path=${Proj_path}/mytrain/llms_scripts
Model_path=${Proj_path}/PreTrainModels/Qwen2-0.5B
#
#
LR=2e-5
WARMUP=100
STEP=6000
SkipN=0

Data_path=${Proj_path}/Dataset/ptr_zhenjako
Train_path=${Data_path}/ptr_train.zhenjako.sh00.pc.shuf.json
Valid_path=${Data_path}/ptr_eval.json

Output_path=${Proj_path}/FinetuneModels/Qwen2-0.5B-zhenjako-ctr
# larger batch 512:=16 x 4 x 8; original 128

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${Code_path}/llms_ptr.py \
    --deepspeed ${Code_path}/deepspeed_config_zero2.json \
    --model_name_or_path ${Model_path} \
    --train_file ${Train_path} \
    --validation_file ${Valid_path} \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps ${STEP} \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0.1 \
    --warmup_steps ${WARMUP} \
    --lr_scheduler_type "constant" \
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
    --ignore_data_skip False \
    --skip_first_N_lines ${SkipN} \
    --output_dir ${Output_path}


#    --ignore_data_skip True  #use for continual training
#    --overwrite_output_dir True \
#    --eval_steps 1000 \
#    --warmup_ratio 0.03 \
#    --do_eval \
#    --resume_from_checkpoint True \
#    --validation_split_percentage 0 \
