#!/bin/bash
# author: mingding

# this is launched by srun
# command for this script: srun -N 2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=4 --job-name=slurm_example --partition=dev --time=00:10:00 --output=slurm_example.out --error=slurm_example.err ./single_launch.sh

# if SLURM defined, set by SLURM environment
module load cuda/11.7

export WORLD_SIZE=${SLURM_NTASKS:-1}
export RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    export MASTER_ADDR=localhost
    export MASTER_PORT=7878
else
    export MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
# generate a port at random
export LOCAL_RANK=${SLURM_LOCALID:-0}

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export HF_HOME=/nxchinamobile2/shared/official_pretrains/hf_home
export SAT_HOME=/nxchinamobile2/shared/official_pretrains/sat_home
#export LD_LIBRARY_PATH=/data/apps/source/nccl/build/lib/:$LD_LIBRARY_PATH

# python pseudo_training.py --world_size $WORLD_SIZE --rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --local_rank $LOCAL_RANK

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

echo ${main_dir}

train_data="/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/translate.jsonl"
gpt_options=" \
       --experiment-name finetune-chatglm2-6b \
       --model-parallel-size 1 \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       --max_source_length 200 \
       --max_target_length 200 \
       --train-data ${train_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --save-interval 100 \
       --eval-interval 100 \
       --save ./checkpoints \
       --split 98,1,1 \
       --eval-iters 1 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.00004 \
       --batch-size 16 \
       --skip-init \
       --fp16 \
       --block-size 128
"
  
python finetune_chatglm2.py ${gpt_options} --local_rank $LOCAL_RANK
