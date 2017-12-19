#!/bin/bash
### set the number of nodes
### set the number of PEs per node
### set the XE/XK feature
#PBS -q normal
#PBS -l nodes=250:ppn=16:xk
### set the wallclock time
#PBS -l walltime=00:25:00
### set the job name
#PBS -N TDISKS25A93T65S
### set the job stdout and stderr
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
### set email notification
#PBS -m bea
#PBS -M matthewliska92@gmail.com
#PBS -l flags=preemptee

module swap PrgEnv-cray PrgEnv-intel
module load cudatoolkit
module swap cray-mpich/7.5.0 cray-mpich/7.5.3
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH 
export MPICH_RDMA_ENABLED_CUDA=1

cd /scratch/sciteam/liska/NEW/TDISKS25A93T65S

make clean
make -j
aprun -n 250 -N 1 -d 16 -cc none ./harm
aprun -n 149 -N 1 -d 16 -cc none ./harm