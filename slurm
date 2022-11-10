#!/bin/bash
# Begin LSF Directives
#BSUB -P PHY129
#BSUB -W 12:00
#BSUB -nnodes 1
#BSUB -alloc_flags "smt1"
#BSUB -N matthewliska92@gmail.com
#BSUB -J HIP
#BSUB -o HIP.%J
#BSUB -e HIP.%J

module load xl
module load cuda/11.0.3
cd $PROJWORK/phy129/hiptest

jsrun --smpiargs "-gpu" -n1 -a1 -g1 -c42 -bpacked:42 ./harm
