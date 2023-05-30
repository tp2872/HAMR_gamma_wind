#!/bin/bash
#SBATCH -A AST178
#SBATCH -J HAMR_CART
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -S 8
#SBATCH --threads-per-core=1
#SBATCH -N 3

module load PrgEnv-cray
module load cray-mpich
module load craype-accel-amd-gfx90a
module load amd-mixed/5.4.3
export MPICH_GPU_SUPPORT_ENABLED=1

cd /lustre/orion/ast178/proj-shared/TILTED_CART

srun -n24 -c7 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest ./harm
