#!/bin/bash
#SBATCH -A m2401_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --mail-user=matthewliska92@gmail.com
#SBATCH -t 01:00:00
#SBATCH -n 150
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load PrgEnv-gnu
module load gcc/9.3.0
module load cuda
ulimit -s unlimited

export SLURM_CPU_BIND="cores"
srun -n150 ./harm