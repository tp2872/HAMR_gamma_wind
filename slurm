#!/bin/bash -l
#SBATCH --job-name="THIN_GIBWA"
#SBATCH --account="s952"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthewliska92@gmail.com
#SBATCH --time=00:15:00
#SBATCH --nodes=200
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_RDMA_ENABLED_CUDA=1
module load cdt/20.08
module switch PrgEnv-cray PrgEnv-gnu
module load daint-gpu
module load craype-accel-nvidia60
cd /scratch/snx3000/mliska/test
srun ./harm
