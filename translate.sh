#!/bin/bash
# author: mingding

# this is launched by srun
# command for this script: srun -N 2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=4 --job-name=slurm_example --partition=dev --time=00:10:00 --output=slurm_example.out --error=slurm_example.err ./single_launch.sh

# if SLURM defined, set by SLURM environment
module load cuda/11.7

WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
LOCAL_RANK=${SLURM_LOCALID:-0}

# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    export MASTER_ADDR=localhost
    export MASTER_PORT=7878
else
    export MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) + $RANK)
fi
# generate a port at random

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

gpt_options=" \
       --batch_size 1 \
       --max_length 300 \
       --num_workers 6 \
"

python scripts/translate_coyo.py ${gpt_options} --world_size $WORLD_SIZE --rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --local_rank $LOCAL_RANK
