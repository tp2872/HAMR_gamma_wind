#!/bin/bash
# Begin LSF Directives
#BSUB -P PHY129
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -alloc_flags "smt1"
#BSUB -N danat.issa@gmail.com
#BSUB -J RunSim123
#BSUB -o RunSim123.%J
#BSUB -e RunSim123.%J

module load xl
module load cuda 
cd $PROJWORK/phy129/IC_test/harm2d
make clean
make -j
jsrun --smpiargs "-gpu" -n1 -a6 -g6 -c42 -bpacked:7 ./harm
