#!/bin/bash -l
#SBATCH -A AST20011
#SBATCH --job-name="THIN_RAD_SANE"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthewliska92@gmail.com
#SBATCH -t 2-00:00:00
#SBATCH -p v100
#SBATCH -N 64 #Number of nodes
#SBATCH -n 256 # Number of cores

module load gcc
module load cuda
export OMP_NUM_THREADS=10
export MY_SPECTRUM_OPTIONS="--gpu"
cd /scratch/03284/tg825826/THIN_RAD_SANE

ibrun -np 256 ./harm

cd /scratch/03284/tg825826/PLASMOID1024_3
ibrun -np 256 ./harm
