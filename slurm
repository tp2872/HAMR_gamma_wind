#!/bin/bash
# Begin LSF Directives
#BSUB -P PHY129
#BSUB -W 02:00
#BSUB -nnodes 7
#BSUB -alloc_flags "smt1"
#BSUB -N matthewliska92@gmail.com
#BSUB -J TILTED_CART
#BSUB -o TILTED_CART.%J
#BSUB -e TILTED_CART.%J

module load gcc
module load cuda/11.0.3
ulimit -s unlimited
cd $PROJWORK/phy129/TILTED_CART

jsrun --smpiargs="-gpu" -n7 -a6 -g6 -c42 -bpacked:7 ./harm
