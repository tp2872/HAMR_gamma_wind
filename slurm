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
cd $PROJWORK/phy129/T65TOR
make clean
make -j
jsrun --smpiargs "-gpu" -n1 -a6 ./harm
