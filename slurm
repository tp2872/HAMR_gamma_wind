#!/bin/bash
# Begin LSF Directives
#BSUB -P PHY129
#BSUB -W 0:20
#BSUB -nnodes 600
#BSUB -alloc_flags "smt1"
#BSUB -N matthewliska92@gmail.com
#BSUB -J RunSim123
#BSUB -o RunSim123.%J
#BSUB -e RunSim123.%J

module load xl
module load cuda 
cd RHAMR
make clean
make -j
jsrun --smpiargs "-gpu" -n600 -a6 -g6 -c42 -bpacked:7 ./harm
