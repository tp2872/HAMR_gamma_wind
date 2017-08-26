#!/bin/bash
### set the number of nodes
### set the number of PEs per node
### set the XE/XK feature
#PBS -q high
#PBS -l nodes=300:ppn=16:xk
### set the wallclock time
#PBS -l walltime=24:00:00
### set the job name
#PBS -N LTDISKS25A93T60
### set the job stdout and stderr
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
### set email notification
#PBS -m bea
#PBS -M matthewliska92@gmail.com

###module swap craype/2.5.4 craype/2.5.0
###module swap cray-mpich/7.3.3  cray-mpich/7.3.0
module swap PrgEnv-cray PrgEnv-intel
module load cudatoolkit
export OMP_NUM_THREADS=16
###qsub -I -l nodes=4:ppn=1:xk -l walltime=00:30:00 -q debug
cd /scratch/sciteam/liska/NEW/LTDISKS25A93T60
make clean
make -j
aprun -n 300 -N 1 -d 16 -cc none ./harm
aprun -n 299 -N 1 -d 16 -cc none ./harm