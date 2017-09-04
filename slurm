#!/bin/bash
### set the number of nodes
### set the number of PEs per node
### set the XE/XK feature
#PBS -q high
#PBS -l nodes=300:ppn=16:xk
### set the wallclock time
#PBS -l walltime=00:30:00
### set the job name
#PBS -N TDISKS25A93T45S_CUDA
### set the job stdout and stderr
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
### set email notification
#PBS -m bea
#PBS -M matthewliska92@gmail.com

module swap PrgEnv-cray PrgEnv-intel
module load cudatoolkit
export MPICH_NEMESIS_ASYNC_PROGRESS=1
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_RDMA_ENABLED_CUDA=1
cd /scratch/sciteam/liska/TDISKS25A93T45S
make clean
make -j
aprun -n 300 -N 1 -d 16 -cc none ./harm
aprun -n 299 -N 1 -d 16 -cc none ./harm