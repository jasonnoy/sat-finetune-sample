#!/bin/bash
#SBATCH --output=ftcleaner_%j.out
#SBATCH --error=ftcleaner_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=dev
#SBATCH --gres=gpu:8
#SBATCH --export=ALL

srun train.sh 
echo "Done with job $SLURM_JOB_ID"

