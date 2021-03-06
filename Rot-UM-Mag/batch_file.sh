#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH -C gpu
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -A m1759
#SBATCH -q special

srun -N 1 python -u run_model.py &
srun -N 1 python -u run_model1.py &
wait
