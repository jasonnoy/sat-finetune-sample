#!/bin/bash
#SBATCH --job-name=translate_COYO
#SBATCH --output=translate_%j.out
#SBATCH --error=translate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude=g0001
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

srun translate.sh
echo "Done with job $SLURM_JOB_ID"

