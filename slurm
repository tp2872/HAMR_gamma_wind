#!/bin/bash
#SBATCH -A m2401_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name="bhns_nuM1"
#SBATCH --mail-user=danat.issa@gmail.com
#SBATCH -t 06:00:00
#SBATCH -n 24
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -d singleton
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
module load PrgEnv-gnu
module load gcc/9.3.0
module load cudatoolkit
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -s unlimited
export SLURM_CPU_BIND="cores"

mkdir -p restart_dumps

if test -e rdumps0; then
    cp -r rdumps0 restart_dumps/rdumps0-$SLURM_JOB_ID
    cp -r rdumps1 restart_dumps/rdumps1-$SLURM_JOB_ID
fi

srun -n24 ./harm