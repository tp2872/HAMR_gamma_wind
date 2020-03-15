#!/bin/bash
#SBATCH --job-name="BART1024"
#SBATCH --output="hellompi.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --reservation=bgpu
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=40
#SBATCH --mail-user=matthewliska92@gmail.com
#SBATCH --export=ALL
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=v100

module purge
module load slurm
module load intel/compiler intel/mpi
module load cuda/10.1.243_418.87.00 

cd /mnt/sdceph/users/bripperda/HAMR_TEST/
make clean
make -j
EXPORT OMP_NUM_THREADS=10
srun -n 96 --cpus-per-task=10 --gres=gpu:4 --mpi=pmi2 ./harm