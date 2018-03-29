#!/bin/bash
#SBATCH -J test_slurm
#SBATCH -o Output/output_%j.txt
#SBATCH -D /home_expes/kt82128h/GridNet/Python_Files
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -N 1
# maximum time limit : at least 1 minute ; format : "minutes" | "minutes:seconds" | "hours:minutes:seconds" | "days-hours" | "days-hours:minutes"
#SBATCH -t 2:0
#SBATCH --mem=3G
#SBATCH -p SHORT
#SBATCH -x calcul-gpu-lahc-2

source /home_expes/tools/python/python3_cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/home_expes/tools/cuda/cuda-8.0/lib64:$LD_LIBRARY_PATH

srun -N1 -n1 -c$SLURM_CPUS_PER_TASK  python Main.py  
