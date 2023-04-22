#!/bin/bash
#SBATCH -A AST178
#SBATCH -J HAMR_TEST
#SBATCH -o %x-%j.out
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 100

module load PrgEnv-cray
module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
export MPICH_GPU_SUPPORT_ENABLED=1

cd /autofs/nccs-svm1_home1/mliska/RHAMR_CUDA

srun -N100 -c8 --ntasks-per-node=8 ./harm
