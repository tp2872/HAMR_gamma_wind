################################################################################
# Sample script that shows how to
# Restore VisIt session with different sources
#
# Amit Chourasia, SDSC, UC San Diego
# Jan Jun, 2015
# CC-BY-SA
#
# Adapted for Tchekhovskoy PRAC by Mark Van Moer, NCSA, Summer 2017
# - assumes that each datafile is a separate timestep.
# 
# MVM: this is called within the job script as:
# visit -cli -nowin -quiet -s render.py
################################################################################
import sys, os
import random
from time import clock, time, sleep
from glob import glob

# for visit engine_par jobs (also, change to PBS TORQUE
#tasks = os.environ['SLURM_NTASKS']

# Change the following as needed
################################################################################
# VisIt session file location

dir = "/gpfs/alpine/phy129/proj-shared/T65TOR/reduced"
sessionfile = dir+"/visit/contoursummit.session"

# Image save information
# MVM: changing for testing
myOutputDir = dir+"/visit/"
myOutputFilename = "vr"
width = 2000 # 1280 Image resolution width in pixels
height = 2000 # 720 Image resolution width in pixels
################################################################################

start= int(sys.argv[1])
end=int(sys.argv[2])
step=int(sys.argv[3])
numtasks = 1
rank = 0

import socket
print("Starting job on {}.".format(socket.gethostname()))

#Count number of .vtk files
count=0
for i in range(0, (end - start) / step, step):
    i2 = start + i * step
    if (os.path.isfile(dir+"/visit/data%dn0.vtk" %i2)):
        count+=1
		
#Distribute among nodes
dumps_per_node=int(count/numtasks)
if(count%numtasks!=0):
    dumps_per_node+=1
count=0
	
#Generate .png files using session file
for i in range(0, (end - start) // step, step):
    i2 = start + i * step
    if (os.path.isfile(dir+"/visit/data%dn0.vtk" %i2)):
        count+=1
        if(1):
            datafile = dir+"/visit/data%dn0.vtk" %i2
            print 'rendering1 datafile: {}'.format(datafile)
            RestoreSessionWithDifferentSources(sessionfile, 0, datafile)
            print 'rendering2 datafile: {}'.format(datafile)

            # Set the save window attributes
            s = SaveWindowAttributes()
            s.outputToCurrentDirectory = 0   # do not automatically write to current dir
            s.outputDirectory = myOutputDir  # write images to this location
            s.family = 0                     # disable default file-naming
            s.fileName = myOutputFilename + '{:05d}'.format(i2) # setup output image filename
            s.saveTiled = 1                  # if tiling is used
            s.format = s.PNG                 # Use PNG as they are compressed and lossless
            s.width = width
            s.height = height
            s.SetResConstraint(0)
            s.SetSaveTiled(0)
            SetSaveWindowAttributes(s)

            DrawPlots()
            SaveWindow()
            DeleteAllPlots()
            CloseDatabase(datafile)

CloseComputeEngine()
print("Rendering completed on node %d" %rank)

sys.exit() # required by VisIt
