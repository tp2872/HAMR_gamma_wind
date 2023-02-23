#!/bin/bash
#SBATCH -A AST178
#SBATCH -J HAMR_TEST
#SBATCH -o %x-%j.out
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 1

module load PrgEnv-cray
module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm
export MPICH_GPU_SUPPORT_ENABLED=1

cd /autofs/nccs-svm1_home1/mliska/RHAMR_CUDA

srun -n1 --ntasks-per-node=1 ./harm
