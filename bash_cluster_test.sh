#!/bin/bash
#SBATCH -J test_slurm_gpu
#SBATCH -o output_%j.out
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -N 1
# maximum time limit : at least 1 minute ; format : "minutes" | "minutes:seconds" | "hours:minutes:seconds" | "days-hours" | "days-hours:minutes"
#SBATCH -t 30-0
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH -p GPU
#SBATCH -x calcul-gpu-lahc-2
#    SBATCH -D /home_expes/gt78520h
#

source /home_expes/tools/python/python3_gpu
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/home_expes/tools/cuda/cuda-8.0/lib64:$LD_LIBRARY_PATH

srun -N1 -n1 -c$SLURM_CPUS_PER_TASK --gres=gpu:1 python script.py  
