#!/bin/bash
#SBATCH -J HARM3D1          # job name
#SBATCH -o HARMoutput.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 9              # total number of nodes requested
#SBATCH -n 9              # total number of mpi tasks requested
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -t 0:09:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=matthew.liska@student.uva.nl
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
ibrun ./harm              # run the MPI executable named a.out