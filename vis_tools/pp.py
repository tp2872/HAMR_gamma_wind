# coding: utf-8
# python pp.py build_ext --inplace
# In[21]:
# from __future__ import division__future__ import division
# from IPython.display import display

import os, sys, gc
import shutil
sys.path.append("/gpfs/alpine/phy129/proj-shared/T65TOR/HAMR3/lib/python3.7/site-packages")

#import sympy as sym
# from sympy import *
import numpy as np
from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# add the current dir to the path
import inspect 	

this_script_full_path = inspect.stack()[0][1]
dirname = os.path.dirname(this_script_full_path)
sys.path.append(dirname)

import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import operator
import threading

from matplotlib.gridspec import GridSpec
from distutils.dir_util import copy_tree

# add amsmath to the preamble
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb,amsmath}"]
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

rc('text', usetex=False)
font = {'size': 40}
rc('font', **font)
rc('xtick', labelsize=70)
rc('ytick', labelsize=70)
# rc('xlabel', **int(f)ont)
# rc('ylabel', **int(f)ont)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cmr10'
mpl.rcParams['font.sans-serif'] = 'cmr10'
plt.rcParams['image.cmap'] = 'jet'
if mpl.get_backend() != "module://ipykernel.pylab.backend_inline":
    plt.switch_backend('agg')
	
# needed in Python 3 for the axes to use Computer Modern (cm) fonts
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
legend = {'fontsize': 40}
rc('legend', **legend)
axes = {'labelsize': 50}
rc('axes', **axes)

fontsize = 38
mytype = np.float32

from sympy.interactive import printing

printing.init_printing(use_latex=True)

# For ODE integration
from scipy.integrate import odeint
from scipy.interpolate import interp1d

np.seterr(divide='ignore')

def avg(v):
    return (0.5 * (v[1:] + v[:-1]))

def der(v):
    return ((v[1:] - v[:-1]))

def shrink(matrix, f):
    return matrix.reshape(f, matrix.shape[0] / f, f, matrix.shape[1] / f, f, matrix.shape[2] / f).sum(axis=0).sum(
        axis=1).sum(axis=2)

def myfloat(f, acc="float32"):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc == 1 or acc == "float32":
        return (np.float32(f))
    else:
        return (np.float64(f))

def rpar_new(dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, density_scale,REF_1,REF_2,REF_3, RAD_M1, RESISTIVE, TWO_T, P_NUM
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, NODE, TIMELEVEL,flag_restore,r1,r2,r3, export_raytracing_RAZIEH, interpolate_var, rank

    if (os.path.isfile("dumps%d/parameters" % dump)):
        fin = open("dumps%d/parameters" % dump, "rb")
    else:
        print("Rpar error!")

    t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    Dtd = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtl = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtr = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    dump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    rdump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    dt = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    failed = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    bs1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    startx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    tf = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    density_scale = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    for n in range(0,13):
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    if(trash >= 1000):
        P_NUM=1
        trash=trash-1000
    else:
        P_NUM=0
    if (trash >= 100):
        TWO_T = 1
        trash = trash - 100
    else:
        TWO_T = 0
    if (trash >= 10):
        RESISTIVE = 1
        trash = trash - 10
    else:
        RESISTIVE = 0
    if (trash >= 1):
        RAD_M1 = 1
        trash = trash - 1
    else:
        RAD_M1 = 0
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    #Set grid spacing
    _dx1=(np.log(Rout)-np.log(Rin))/(bs1*nb1)
    _dx2=2.0*1.0/(bs2*nb2)
    _dx3=2.0*np.pi/(bs3*nb3)

    nb = n_active_total
    rhor = 1 + (1 - a ** 2) ** 0.5

    NODE=np.copy(n_ord)
    TIMELEVEL=np.copy(n_ord)

    REF_1=1
    REF_2=1
    REF_3=1
    flag_restore = 0
    size = os.path.getsize("dumps%d/parameters" % dump)
    if(size>=66*4+3*n_active_total*4):
        n=0
        while n<n_active_total:
            n_ord[n]=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            NODE[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n=n+1
    elif(size >= 66 * 4 + 2 * n_active_total * 4):
        n = 0
        flag_restore=1
        while n < n_active_total:
            n_ord[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n = n + 1

    if(export_raytracing_RAZIEH==1 and (bs1%lowres1!=0 or bs2%lowres2!=0 or bs3%lowres3!=0 or ((lowres1 & (lowres1-1) == 0) and lowres1 != 0)!=1 or ((lowres2 & (lowres2-1) == 0) and lowres2 != 0)!=1 or ((lowres3 & (lowres3-1) == 0) and lowres3 != 0)!=1)):
        if(rank==0):
            print("For raytracing block size needs to be divisable by lowres!")
    if(export_raytracing_RAZIEH==1 and interpolate_var==0):
        if (rank == 0):
            print("Warning: Variable interpolation is highly recommended for raytracing!")
    fin.close()

def rpar_old(dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, density_scale,REF_1,REF_2,REF_3
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, RAD_M1, RESISTIVE, TWO_T, P_NUM
    global flag_restore
    flag_restore = 0
    temp_array=np.zeros((15),dtype=np.int32)

    if (os.path.isfile("dumps%d/parameters" % dump)):
        fin = open("dumps%d/parameters" % dump, "rb")
    else:
        print("Rpar error!")
    t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    Dtd = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtl = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtr = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    dump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    rdump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    dt = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    failed = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    bs1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    startx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]

    tf = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    density_scale = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]

    for i in range(0, 15):
        temp_array[i] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    nb = n_active_total
    rhor = 1 + (1 - a ** 2) ** 0.5

    gd1_temp = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    gd2_temp = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    for n in range(0, nmax):
        block[n, AMR_REFINED] = gd1_temp[n]
        block[n, AMR_ACTIVE] = gd2_temp[n]

    i = 0
    for n in range(0, nmax):
        if block[n, AMR_ACTIVE] == 1:
            n_ord[i] = n
            i += 1
    fin.close()
    RAD_M1=0
    RESISTIVE=0
    TWO_T=0
    P_NUM=0
    if((nb2==6) or nb2==12 or nb2==24 or nb2==48 or nb2==96):
        if(nb3<=2):
            if (rank==0):
                print("Derefinement near pole detected. Please make sure this is appropriate for the dataset!")
            REF_1 = 0
            REF_2 = 0
            if(bs3>1):
                REF_3 = 1
            else:
                REF_3 = 0
    else:
        REF_1 = 1
        REF_2 = 1
        if (bs3 > 1):
            REF_3 = 1
        else:
            REF_3=0
    if(nb3>2):
        REF_1=1
        REF_2=1
        REF_3=1

def rpar_write(dir, dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, density_scale, RAD_M1,RESISTIVE, TWO_T, P_NUM, NODE, TIMELEVEL
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp
    trash=0
    fin = open(dir+"/backup/dumps%d/parameters" % dump, "wb")

    t.tofile(fin)
    n_active.tofile(fin)
    n_active_total.tofile(fin)
    nstep.tofile(fin)
    Dtd.tofile(fin)
    Dtl.tofile(fin)
    Dtr.tofile(fin)
    dump_cnt.tofile(fin)
    rdump_cnt.tofile(fin)
    dt.tofile(fin)
    failed.tofile(fin)
    np.int32(bs1new).tofile(fin)
    np.int32(bs2new).tofile(fin)
    np.int32(bs3new).tofile(fin)
    nmax.tofile(fin)
    nb1.tofile(fin)
    nb2.tofile(fin)
    nb3.tofile(fin)
    startx1.tofile(fin)
    startx2.tofile(fin)
    startx3.tofile(fin)
    np.float64(_dx1).tofile(fin)
    np.float64(_dx2).tofile(fin)
    np.float64(_dx3).tofile(fin)
    tf.tofile(fin)
    a.tofile(fin)
    gam.tofile(fin)
    cour.tofile(fin)
    Rin.tofile(fin)
    Rout.tofile(fin)
    R0.tofile(fin)
    density_scale.tofile(fin)
    for n in range(0, 13):
        np.int32(trash).tofile(fin)
    trash=1*RAD_M1+10*RESISTIVE+100*TWO_T+1000*P_NUM
    np.int32(trash).tofile(fin)

    n=0
    while n < n_active_total:
        n_ord[n].tofile(fin)
        TIMELEVEL[n].tofile(fin)
        NODE[n].tofile(fin)
        n = n + 1
    fin.close()

#Reorders n_ord, TIMELEVEL and NODE
def restore_dump(dir,dump):
    global n_ord, NODE, TIMELEVEL, numtasks_local, n_active_total, rank

    #Find number of nodes
    numtasks_local = 0
    while (os.path.isfile(dir+"/dumps%d" % dump + "/new_dump%d"  % numtasks_local)):
        numtasks_local = numtasks_local + 1
    if(rank==0):
        print("Number of nodes: %d" %numtasks_local)

    #Allocate memory for node arrays
    n_ord_node=np.zeros((numtasks_local, np.int(n_active_total/numtasks_local*5)), dtype=np.int32, order='C')
    n_active_total_node=np.zeros((numtasks_local), dtype=np.int32, order='C')
    TIMELEVEL_node = np.zeros((numtasks_local, np.int(n_active_total/numtasks_local*5)), dtype=np.int32, order='C')

    #Get node number in NODE from z-curve
    get_NODE(dir, dump)

    #Order grid per node
    for n in range(0,n_active_total):
        n_ord_node[NODE[n]][n_active_total_node[NODE[n]]]=n_ord[n]
        TIMELEVEL_node[NODE[n]][n_active_total_node[NODE[n]]] = TIMELEVEL[n]
        n_active_total_node[NODE[n]]=n_active_total_node[NODE[n]]+1

    n2=0
    for i in range(0,numtasks_local):
        for n in range(0, n_active_total_node[i]):
            n_ord[n2]=n_ord_node[i][n]
            NODE[n2]=i
            TIMELEVEL[n2]=TIMELEVEL_node[i][n]
            n2=n2+1

#Calculates NODE for each block
def get_NODE(dir, dump):
    global n_active_total, NODE, TIMELEVEL, numtasks_local
    timelevel_cutoff=6
    MAX_WEIGHT=1

    n_active_total_t=np.zeros((timelevel_cutoff), dtype=np.int32, order='C')
    n_active_total_steps_t = np.zeros((timelevel_cutoff), dtype=np.int32, order='C')
    n_active_localsteps=np.zeros((numtasks_local), dtype=np.int32, order='C')
    n_ord_total_RM_t=np.zeros((n_active_total, timelevel_cutoff), dtype=np.int32, order='C')
    for i in range(0,timelevel_cutoff):
        n_active_total_t[i] = 0
        n_active_total_steps_t[i] = 0

    for n in range(0,n_active_total):
        tl = np.int(np.log(TIMELEVEL[n])/(np.log(2.0))+0.01)
        n_ord_total_RM_t[n_active_total_t[tl]][tl] = n
        n_active_total_steps_t[tl] = n_active_total_steps_t[tl]+2**timelevel_cutoff // TIMELEVEL[n]
        n_active_total_t[tl]=n_active_total_t[tl]+1

    for u in range(0,numtasks_local):
        n_active_localsteps[u] = 0
    increment = 0
    fillup_mode = 0
    u = 0
    sw = 0

    for i in range(0, timelevel_cutoff):
        if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] > n_active_localsteps[u % numtasks_local]):
            nr_timesteps = 2**timelevel_cutoff // TIMELEVEL[n_ord_total_RM_t[0][i]]
            increment = (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] - n_active_localsteps[u % numtasks_local]) // nr_timesteps
            fillup_mode = 1

        if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] == n_active_localsteps[u % numtasks_local]):
            rem = n_active_total_t[i] % (numtasks_local)
            increment = (n_active_total_t[i] - rem) // (numtasks_local)
            fillup_mode = 0
            sw = 1
        n = 0

        while (n < n_active_total_t[i]):
            nr_timesteps = 2**timelevel_cutoff // TIMELEVEL[n_ord_total_RM_t[n][i]]

            if (fillup_mode == 1):
                increment = (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] - n_active_localsteps[u % numtasks_local]) // nr_timesteps

            if (n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] == n_active_localsteps[u % numtasks_local]):
                rem = (n_active_total_t[i] - n) % (numtasks_local)
                increment = (n_active_total_t[i] - n - rem) // (numtasks_local)
                fillup_mode = 0
                sw = 1

            if (fillup_mode == 0 and ((n_active_total_t[i] - n) // (increment + 1)) == rem and (n_active_total_t[i] - n) % (increment + 1) == 0 and rem > 0):
                increment += 1
                sw = 1

            increment = min(increment, n_active_total_t[i] - n)

            for j in range(0,increment):
                NODE[n_ord_total_RM_t[n + j][i]] = (u % numtasks_local)
                n_active_localsteps[u % numtasks_local] = n_active_localsteps[u % numtasks_local] + nr_timesteps

            n = n + increment
            if (n_active_localsteps[u % numtasks_local] == n_active_localsteps[(u - 1 + numtasks_local) % numtasks_local] or sw == 1):
                sw = 0
                u = (u + 1) % numtasks_local

def rblock_new(dump):
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6
    global AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12
    global AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8
    global AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8
    global AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P
    global block, nmax, n_ord, AMR_TIMELEVEL

    AMR_ACTIVE = 0
    AMR_LEVEL = 1
    AMR_REFINED = 2
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    AMR_PARENT = 6
    AMR_CHILD1 = 7
    AMR_CHILD2 = 8
    AMR_CHILD3 = 9
    AMR_CHILD4 = 10
    AMR_CHILD5 = 11
    AMR_CHILD6 = 12
    AMR_CHILD7 = 13
    AMR_CHILD8 = 14
    AMR_NBR1 = 15
    AMR_NBR2 = 16
    AMR_NBR3 = 17
    AMR_NBR4 = 18
    AMR_NBR5 = 19
    AMR_NBR6 = 20
    AMR_NODE = 21
    AMR_POLE = 22
    AMR_GROUP = 23
    AMR_CORN1 = 24
    AMR_CORN2 = 25
    AMR_CORN3 = 26
    AMR_CORN4 = 27
    AMR_CORN5 = 28
    AMR_CORN6 = 29
    AMR_CORN7 = 30
    AMR_CORN8 = 31
    AMR_CORN9 = 32
    AMR_CORN10 = 33
    AMR_CORN11 = 34
    AMR_CORN12 = 35
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112  
    AMR_NBR1_3=113
    AMR_NBR1_4=114
    AMR_NBR1_7=115
    AMR_NBR1_8=116
    AMR_NBR2_1=117
    AMR_NBR2_2=118
    AMR_NBR2_3=119
    AMR_NBR2_4=120
    AMR_NBR3_1=121
    AMR_NBR3_2=122
    AMR_NBR3_5=123
    AMR_NBR3_6=124
    AMR_NBR4_5=125
    AMR_NBR4_6=126
    AMR_NBR4_7=127
    AMR_NBR4_8=128
    AMR_NBR5_1=129
    AMR_NBR5_3=130
    AMR_NBR5_5=131
    AMR_NBR5_7=132
    AMR_NBR6_2=133
    AMR_NBR6_4=134
    AMR_NBR6_6=135
    AMR_NBR6_8=136
    AMR_NBR1P=161
    AMR_NBR2P=162
    AMR_NBR3P=163
    AMR_NBR4P=164
    AMR_NBR5P=165
    AMR_NBR6P=166
    AMR_TIMELEVEL=36
    
    # Read in data for every block
    if (os.path.isfile("dumps%d/grid" % dump)):
        fin = open("dumps%d/grid" % dump, "rb")
        size = os.path.getsize("dumps%d/grid" % dump)
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = 36
    elif(os.path.isfile("gdumps/grid")):
        fin = open("gdumps/grid", "rb")
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
    else:
        print("Cannot find grid file!")

    # Allocate memory
    block = np.zeros((nmax, 200), dtype=np.int32, order='C')
    n_ord = np.zeros((nmax), dtype=np.int32, order='C')

    gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
    gd = gd.reshape((NV, nmax), order='F').T
    block[:,0:NV] = gd
    if(NV<170):
        block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
        block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
        block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

    i = 0
    if (os.path.isfile("dumps%d/grid" % dump)):
        for n in range(0, nmax):
            if block[n, AMR_ACTIVE] == 1:
                n_ord[i] = n
                i += 1

    fin.close()

def rgdump_new(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3
    import pp_c
    set_cart=0
    set_xc=0

    if((bs1%lowres1)!=0 or (bs2%lowres2)!=0 or (bs3%lowres3)!=0):
        print("Incompatible lowres settings in rgdump_new")

    bs1new = int(bs1 / lowres1)
    bs2new = int(bs2 / lowres2)
    bs3new = int(bs3 / lowres3)

    nx = bs1new * nb1
    ny = bs2new * nb2
    nz = bs3new * nb3

    # Allocate memory
    x1 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    x2 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    x3 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    r = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    h = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    ph = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')

    if axisym:
        gcov = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=mytype, order='C')
        gdet = np.zeros((nb, bs1new, bs2new, 1), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=mytype, order='C')
    else:
        gcov = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        gdet = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')

    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8 and bs3!=1):
        flag=1
    else:
        flag=0

    pp_c.rgdump_new(flag, dir, axisym, n_ord,lowres1,lowres2,lowres3,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet)

def rgdump_griddata(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym, interpolate_var
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3, REF_1, REF_2, REF_3
    global startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box, rank, gridsizex1, gridsizex2, gridsizex3, check_files
    import pp_c

    set_cart=0
    set_xc=0

    ACTIVE1 = np.max(block[n_ord, AMR_LEVEL1])*REF_1
    ACTIVE2 = np.max(block[n_ord, AMR_LEVEL2])*REF_2
    ACTIVE3 = np.max(block[n_ord, AMR_LEVEL3])*REF_3

    if ((int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1) % lowres1) != 0 or (int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2) % lowres2) != 0 or (int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3) % lowres3) != 0):
        print("Incompatible lowres settings in rgdump_griddata")

    gridsizex1 = int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1/lowres1)
    gridsizex2 = int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2/lowres2)
    gridsizex3 = int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3/lowres3)

    _dx1 = _dx1 * lowres1 * (1.0 / (1.0 + REF_1) ** ACTIVE1)
    _dx2 = _dx2 * lowres2 * (1.0 / (1.0 + REF_2) ** ACTIVE2)
    _dx3 = _dx3 * lowres3 * (1.0 / (1.0 + REF_3) ** ACTIVE3)

    #Calculate inner and outer boundaries of selection box after upscaling and downscaling; Assumes uniform grid x1=log(r) etc
    if(do_box==1):
        i_min = max(np.int32((np.log(r_min)-(startx1+0.5*_dx1)) / _dx1) + 1, 0)
        i_max = min(np.int32((np.log(r_max)-(startx1+0.5*_dx1)) / _dx1) + 1, gridsizex1)
        j_min=max(np.int32(((2.0/np.pi*(theta_min)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,0)
        j_max=min(np.int32(((2.0/np.pi*(theta_max)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,gridsizex2)
        z_min=max(np.int32((phi_min-(startx3+0.5*_dx3))/_dx3) + 1,0)
        z_max=min(np.int32((phi_max-(startx3+0.5*_dx3))/_dx3) + 1,gridsizex3)

        gridsizex1 = i_max-i_min
        gridsizex2 = j_max-j_min
        gridsizex3 = z_max-z_min

        if((j_max<j_min or i_max<i_min or z_max<z_min) and rank==0):
            print("Bad box selection")
    else:
        i_min=0
        i_max=gridsizex1
        j_min=0
        j_max=gridsizex2
        z_min=0
        z_max=gridsizex3

    nx = gridsizex1
    ny = gridsizex2
    nz = gridsizex3

    # Allocate memory
    x1 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x2 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    x3 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    r = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    h = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ph = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if axisym:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
    else:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(rank==0 and check_files==1):
        for n in range(0,n_active_total):
            if(os.path.isfile('gdumps/gdump%d' %n_ord[n])==0 or os.path.getsize('gdumps/gdump%d' %n_ord[n])!=(9*bs1*bs2*bs3+(bs1*bs2*49)*(axisym)+(bs1*bs2*bs3*49)*(axisym==0))*8):
                print("Gdump file %d doesn't exist" %n_ord[n])
    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8):
        flag=1
    else:
        flag=0

    pp_c.rgdump_griddata(flag, interpolate_var, dir, axisym, n_ord,lowres1, lowres2, lowres3 ,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet,block, nb1, nb2, nb3, REF_1, REF_2, REF_3, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]), startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH, i_min, i_max, j_min, j_max, z_min, z_max)

def rgdump_write(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3, REF_1, REF_2, REF_3
    import pp_c
    f1 = int(lowres1)
    f2 = int(lowres2)
    f3 = int(lowres3)
    pp_c.rgdump_write(0, dir +"/backup", axisym, n_ord,f1,f2,f3,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet)

def rdump_new(dir, dump):
    global rho, ug, uu,uu_rad, E_rad, TE, TI, photon_number, RAD_M1, RESISTIVE, TWO_T, P_NUM, B, E, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3
    import pp_c

    if ((int(bs1) % lowres1) != 0 or (int(bs2) % lowres2) != 0 or (int(bs3) % lowres3) != 0):
        print("Incompatible lowres settings in rdump_new")

    bs1new = int(bs1 / lowres1)
    bs2new = int(bs2 / lowres2)
    bs3new = int(bs3 / lowres3)
    nb2d = nb

    # Allocate memory
    rho = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    ug = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    uu = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    B = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    if(RAD_M1):
        E_rad = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        uu_rad = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    else:
        E_rad=ug
        uu_rad=uu

    if (RESISTIVE):
        E = np.zeros(4, (nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    else:
        E = B

    if(TWO_T):
        TE = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        TI = np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    else:
        TE=rho
        TI=rho

    if(P_NUM):
        photon_number=np.zeros((nb, bs1new, bs2new, bs3new), dtype=mytype, order='C')
    else:
        photon_number=rho

    if(os.path.isfile("dumps%d/new_dump" %dump)):
        flag=1
    else:
        flag=0
    pp_c.rdump_new(flag, RAD_M1, RESISTIVE, TWO_T, P_NUM, dir, dump, n_active_total, lowres1, lowres2, lowres3,nb,bs1,bs2,bs3, rho,ug, uu, B, E, E_rad, uu_rad, TE, TI, photon_number, gcov,gcon,axisym)

    _dx1 = _dx1 * lowres1
    _dx2 = _dx2 * lowres2
    _dx3 = _dx3 * lowres3

def rdump_griddata(dir, dump):
    global rho, ug, uu,uu_rad, E_rad, E,  TE, TI, photon_number, RAD_M1, RESISTIVE, TWO_T, P_NUM, B, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3, nb, nb1, nb2, nb3, REF_1, REF_2, REF_3, n_ord, interpolate_var, export_raytracing_GRTRANS,export_raytracing_RAZIEH, DISK_THICKNESS, a, gam, bsq, Rdot
    global startx1,startx2,startx3,_dx1,_dx2,_dx3,x1,x2,x3
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box, check_files
    import pp_c

    # Allocate memory
    rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    if(export_raytracing_RAZIEH):
        Rdot = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        Rdot = np.zeros((1, 1, 1, 1), dtype=mytype, order='C')
    bsq = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    if(RAD_M1):
        E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E_rad=np.copy(ug)
        uu_rad=np.copy(uu)

    if (RESISTIVE):
        E = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E = B

    if (TWO_T):
        TE = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        TI = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        TE = rho
        TI = rho

    if (P_NUM):
        photon_number = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        photon_number = rho
        
    if (os.path.isfile("dumps%d/new_dump" % dump)):
        flag = 1
    else:
        if (rank == 0 and check_files == 10):
            for count in range(0, 5400):
                if (os.path.isfile("dumps%d/new_dump%d" %(dump, count))==0):
                    print("Dump file %d in folder %d doesn't exist" %(count,dump))
        flag = 0

    pp_c.rdump_griddata(flag, interpolate_var, np.int32(RAD_M1),np.int32(RESISTIVE), TWO_T, P_NUM, dir, dump, n_active_total, lowres1, lowres2, lowres3, nb,bs1,bs2,bs3, rho,ug, uu, B, E, E_rad, uu_rad, TE, TI, photon_number, gcov,gcon,axisym,n_ord,block, nb1,nb2,nb3,REF_1, REF_2,REF_3, np.max(block[n_ord, AMR_LEVEL1]),np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]),export_raytracing_RAZIEH, DISK_THICKNESS, a, gam, Rdot, bsq, r, startx1,startx2,startx3,_dx1,_dx2,_dx3,x1,x2,x3, i_min, i_max, j_min, j_max, z_min, z_max)

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3

    if (do_box == 1):
        startx1 = startx1 + (i_min) * _dx1
        startx2 = startx2 + (j_min) * _dx2
        startx3 = startx3 + (z_min) * _dx3

    nb2d = nb
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1

def rdump_write(dir, dump):
    global rho, ug, uu, B, E, uu_rad, E_rad, TE, TI, photon_number, RAD_M1, RESISTIVE, TWO_T, P_NUM, gcov, gcov,axisym, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, export_visit
    import pp_c
    if (os.path.isdir(dir + "/backup/dumps%d" %dump) == 0):
        os.makedirs(dir + "/backup/dumps%d" %dump)
    pp_c.rdump_write(0, RAD_M1, RESISTIVE, TWO_T, P_NUM, dir+"/backup", dump, n_active_total, lowres1, lowres2, lowres3,nb,bs1,bs2,bs3, rho,ug, uu, B,E, E_rad, uu_rad, TE, TI, photon_number, gcov,gcon,axisym)

def downscale(dir, dump):
    rgdump_write(dir)
    rdump_write(dir, dump)
    rpar_write(dir,dump)
    if (os.path.isfile(dir + "/dumps%d/grid" % dump)==1):
        dest=open(dir + "/backup/dumps%d/grid" %dump, 'wb')
        shutil.copyfileobj(open(dir+'/dumps%d/grid'%dump, 'rb'), dest)
        dest.close()

#Execute after executing griddata
def rdiag_new(dump):
    global divb, fail1, fail2, lowres, bs1, bs2, bs3, bs1new, bs2new, bs3new, interpolate_var

    f1 = lowres1
    f2 = lowres2
    f3 = lowres3

    # Allocate memory
    divb = np.zeros((nb, int(bs1/f1), int(bs2/f2), int(bs3/f3)), dtype=mytype, order='C')
    fail1 = np.zeros((nb, int(bs1/f1), int(bs2/f2), int(bs3/f3)), dtype=mytype, order='C')
    fail2 = np.zeros((nb, int(bs1/f1), int(bs2/f2), int(bs3/f3)), dtype=mytype, order='C')

    for n in range(0, n_active_total):
        # read image
        fin = open("dumps%d/new_dumpdiag%d" % (dump, n_ord[n]), "rb")
        gd = np.fromfile(fin, dtype=mytype, count=3 * bs1 * bs2 * bs3, sep='')
        gd = gd.reshape((-1, bs1 * bs2 * bs3), order='F')
        gd = gd.reshape((-1, bs3, bs2, bs1), order='F')
        gd = myfloat(gd.transpose(0, 3, 2, 1))

        for i in range(int(bs1/f1)):
            for j in range(int(bs2/f2)):
                for k in range(int(bs3/f3)):
                    divb[n, i, j, k] = np.average(gd[0, i * f1:(i + 1) * f1, j * f2:(j + 1) * f2, k * f3:(k + 1) * f3])
                    fail1[n, i, j, k] = np.average(gd[1, i * f1:(i + 1) * f1, j * f2:(j + 1) * f2, k * f3:(k + 1) * f3])
                    fail2[n, i, j, k] = np.average(gd[2, i * f1:(i + 1) * f1, j * f2:(j + 1) * f2, k * f3:(k + 1) * f3])
        fin.close()

    grid_3D = np.zeros((1, bs1new, bs2new, bs3new), dtype=mytype, order='C')

    griddata_3D(divb, grid_3D,interpolate_var)
    divb = np.copy(grid_3D)
    griddata_3D(fail1, grid_3D,interpolate_var)
    fail1 = np.copy(grid_3D)
    griddata_3D(fail2, grid_3D,interpolate_var)
    fail2 = np.copy(grid_3D)

from scipy import ndimage
def griddata_3D(input, output, inter=1):
    global axisym, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, export_visit,block, n_ord, nb1, nb2, nb3
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    import pp_c

    pp_c.griddata3D(nb, bs1new, bs2new, bs3new, nb1,nb2,nb3, n_ord, block, input, output, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

def griddata_2D(input, output, inter=1):
    global axisym, nb2d, bs1, bs2, bs3, bs1new, bs2new, bs3new, lowres1, lowres2, lowres3, export_visit, block, n_ord, nb1, nb2, nb3
    global AMR_ACTIVE, AMR_LEVEL, AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    import pp_c

    pp_c.griddata2D(nb, bs1new, bs2new, bs3new, nb1,nb2,nb3, n_ord, block, input, output, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

def griddataall():
    global block, n_ord, rho, uu, uu_rad, E_rad, RAD_M1, RESISTIVE, TWO_T, P_NUM, ud, bu, bd, B, E, TE, TI, photon_number, ug, dxdxp, bsq, r, h, ph, nb, nb1, nb2, nb3, bs1, bs2, bs3, alpha, gcon, gcov, gdet, REF_1, REF_2, REF_3,interpolate_var
    global x1, x2, x3, ti, tj, tk, Rout,startx1,startx2,startx3,_dx1,_dx2,_dx3
    global gridsizex1, gridsizex2, gridsizex3
    global bs1new, bs2new, bs3new, lowres1,lowres2,lowres3, axisym
    ACTIVE1 = np.max(block[n_ord, AMR_LEVEL1])
    ACTIVE2 = np.max(block[n_ord, AMR_LEVEL2])
    ACTIVE3 = np.max(block[n_ord, AMR_LEVEL3])

    if(nb==1):
        print("Griddata cannot be executed with only 1 block")

    if(interpolate_var):
        print("Interpolation not supported in griddataall. Use rdump_griddata and rgdump_griddata!")

    bs1new = int(bs1 / lowres1)
    bs2new = int(bs2 / lowres2)
    bs3new = int(bs3 / lowres3)
    gridsizex1 = nb1 * (1 + REF_1) ** ACTIVE1 * bs1new
    gridsizex2 = nb2 * (1 + REF_2) ** ACTIVE2 * bs2new
    gridsizex3 = nb3 * (1 + REF_3) ** ACTIVE3 * bs3new

    grid_3D = np.zeros((4,1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    if axisym:
        grid_2D = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order='C')
    else:
        grid_2D = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')

    griddata_3D(rho, grid_3D[0],interpolate_var)
    rho = np.copy(grid_3D[0])
    griddata_3D(uu[0], grid_3D[0], interpolate_var)
    griddata_3D(uu[1], grid_3D[1], interpolate_var)
    griddata_3D(uu[2], grid_3D[2], interpolate_var)
    griddata_3D(uu[3], grid_3D[3], interpolate_var)
    uu=np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    uu[0] = np.copy(grid_3D[0])
    uu[1] = np.copy(grid_3D[1])
    uu[2] = np.copy(grid_3D[2])
    uu[3] = np.copy(grid_3D[3])
    griddata_3D(B[1], grid_3D[0],interpolate_var)
    griddata_3D(B[2], grid_3D[1],interpolate_var)
    griddata_3D(B[3], grid_3D[2], interpolate_var)
    B=np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    B[1] = np.copy(grid_3D[0])
    B[2] = np.copy(grid_3D[1])
    B[3] = np.copy(grid_3D[2])
    if(RAD_M1):
        griddata_3D(E_rad, grid_3D[0], interpolate_var)
        E_rad = np.copy(grid_3D[0])
        griddata_3D(uu_rad[0], grid_3D[0], interpolate_var)
        griddata_3D(uu_rad[1], grid_3D[1], interpolate_var)
        griddata_3D(uu_rad[2], grid_3D[2], interpolate_var)
        griddata_3D(uu_rad[3], grid_3D[3], interpolate_var)
        uu_rad = np.copy(grid_3D)
    if (RESISTIVE):
        griddata_3D(E, grid_3D[0], interpolate_var)
        E = np.copy(grid_3D[0])
    if(TWO_T):
        griddata_3D(TE, grid_3D[0], interpolate_var)
        TE = np.copy(grid_3D[0])
        griddata_3D(TI, grid_3D[0], interpolate_var)
        TI = np.copy(grid_3D[0])
    if(P_NUM):
        griddata_3D(photon_number, grid_3D[0], interpolate_var)
        TE = np.copy(grid_3D[0])

    griddata_3D(ug, grid_3D[0],interpolate_var)
    ug = np.copy(grid_3D[0])
    griddata_3D(x1, grid_3D[0], interpolate_var)
    x1 = np.copy(grid_3D[0])
    griddata_3D(x2, grid_3D[0], interpolate_var)
    x2 = np.copy(grid_3D[0])
    griddata_3D(x3, grid_3D[0], interpolate_var)
    x3 = np.copy(grid_3D[0])
    griddata_3D(r, grid_3D[0], interpolate_var)
    r = np.copy(grid_3D[0])
    griddata_3D(h, grid_3D[0], interpolate_var)
    h = np.copy(grid_3D[0])
    griddata_3D(ph, grid_3D[0], interpolate_var)
    ph = np.copy(grid_3D[0])

    griddata_2D(gdet, grid_2D[0,0], interpolate_var)
    gdet= np.copy(grid_2D[0,0])
    for i in range(0, 4):
        for j in range(0, 4):
            griddata_2D(dxdxp[i,j], grid_2D[i,j],interpolate_var)
    dxdxp = np.copy(grid_2D)
    for i in range(0, 4):
        for j in range(0, 4):
            griddata_2D(gcov[i, j], grid_2D[i, j], interpolate_var)
    gcov = np.copy(grid_2D)
    for i in range(0, 4):
        for j in range(0, 4):
            griddata_2D(gcon[i, j], grid_2D[i, j], interpolate_var)
    gcon = np.copy(grid_2D)

    ti=None
    tj=None
    tk=None

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3
    _dx1 = _dx1*(1.0/(1.0 + REF_1) ** ACTIVE1)
    _dx2 = _dx2*(1.0/(1.0 + REF_2) ** ACTIVE2)
    _dx3 = _dx3*(1.0/(1.0 + REF_3) ** ACTIVE3)
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1

def set_pole():
    global bsq, rho, ug, uu, uu_rad, E_rad, E, TE, TI, photon_number, RESISTIVE, TWO_T, P_NUM, RAD_M1, do_box

    ph[:, :, :, 0] = 0.0
    ph[:, :, :, bs3new - 1] = 0.0
    avg=0.5*(bsq[:, :, :, 0]+bsq[:, :, :, bs3new - 1])
    bsq[:, :, :, 0]=avg
    bsq[:, :, :, bs3new-1]=avg
    avg=0.5*(rho[:, :, :, 0]+rho[:, :, :, bs3new - 1])
    rho[:, :, :, 0]=avg
    rho[:, :, :, bs3new-1]=avg
    avg=0.5*(ug[:, :, :, 0]+ug[:, :, :, bs3new - 1])
    ug[:, :, :, 0]=avg
    ug[:, :, :, bs3new-1]=avg
    avg=0.5*(uu[:, :, :, :, 0]+uu[:, :, :, :, bs3new - 1])
    uu[:, :, :, :, 0]=avg
    uu[:, :, :, :, bs3new-1]=avg

    if(RAD_M1):
        avg = 0.5 * (E_rad[:, :, :, 0] + E_rad[:, :, :, bs3new - 1])
        E_rad[:, :, :, 0] = avg
        E_rad[:, :, :, bs3new - 1] = avg
        avg = 0.5 * (uu_rad[:, :, :, :, 0] + uu_rad[:, :, :, :, bs3new - 1])
        uu_rad[:, :, :, :, 0] = avg
        uu_rad[:, :, :, :, bs3new - 1] = avg

    if (RESISTIVE):
        avg = 0.5 * (E[:, :, :, 0] + E[:, :, :, bs3new - 1])
        E[:, :, :, 0] = avg
        E[:, :, :, bs3new - 1] = avg

    if (TWO_T):
        avg = 0.5 * (TE[:, :, :, 0] + TE[:, :, :, bs3new - 1])
        TE[:, :, :, 0] = avg
        TE[:, :, :, bs3new - 1] = avg
        avg = 0.5 * (TI[:, :, :, 0] + TI[:, :, :, bs3new - 1])
        TI[:, :, :, 0] = avg
        TI[:, :, :, bs3new - 1] = avg

    if (P_NUM):
        avg = 0.5 * (photon_number[:, :, :, 0] + photon_number[:, :, :, bs3new - 1])
        photon_number[:, :, :, 0] = avg
        photon_number[:, :, :, bs3new - 1] = avg

    for offset in range(0, np.int(bs3new / 2)):
        bsq[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (bsq[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + bsq[:, :, bs2new - 2, offset])
        bsq[:, :, bs2new - 1, offset] = bsq[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
        bsq[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (bsq[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + bsq[:, :, 1, offset])
        bsq[:, :, 0, offset] = bsq[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        rho[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (rho[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + rho[:, :, bs2new - 2, offset])
        rho[:, :, bs2new - 1, offset] = rho[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
        rho[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (rho[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + rho[:, :, 1, offset])
        rho[:, :, 0, offset] = rho[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        ug[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (ug[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + ug[:, :, bs2new - 2, offset])
        ug[:, :, bs2new - 1, offset] = ug[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
        ug[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (ug[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + ug[:, :, 1, offset])
        ug[:, :, 0, offset] = ug[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        uu[:, :, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (uu[:, :, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + uu[:, :, :, bs2new - 2, offset])
        uu[:, :, :, bs2new - 1, offset] = uu[:, :, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
        uu[:, :, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (uu[:, :, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + uu[:, :, :, 1, offset])
        uu[:, :, :, 0, offset] = uu[:, :, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        if (RAD_M1):
            E_rad[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (E_rad[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + E_rad[:, :, bs2new - 2, offset])
            E_rad[:, :, bs2new - 1, offset] = E_rad[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            E_rad[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (E_rad[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + E_rad[:, :, 1, offset])
            E_rad[:, :, 0, offset] = E_rad[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]
            uu_rad[:, :, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (uu_rad[:, :, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + uu_rad[:, :, :, bs2new - 2, offset])
            uu_rad[:, :, :, bs2new - 1, offset] = uu_rad[:, :, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            uu_rad[:, :, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (uu_rad[:, :, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + uu_rad[:, :, :, 1, offset])
            uu_rad[:, :, :, 0, offset] = uu_rad[:, :, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        if (RESISTIVE):
            E[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (E[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + E[:, :, bs2new - 2, offset])
            E[:, :, bs2new - 1, offset] = E[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            E[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (E[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + E[:, :, 1, offset])
            E[:, :, 0, offset] = E[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        if (TWO_T):
            TE[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (TE[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + TE[:, :, bs2new - 2, offset])
            TE[:, :, bs2new - 1, offset] = TE[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            TE[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (TE[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + TE[:, :, 1, offset])
            TE[:, :, 0, offset] = TE[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]
            TI[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (TI[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + TI[:, :, bs2new - 2, offset])
            TI[:, :, bs2new - 1, offset] = TI[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            TI[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (TI[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + TI[:, :, 1, offset])
            TI[:, :, 0, offset] = TI[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

        if(P_NUM):
            photon_number[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (photon_number[:, :, bs2new - 2, int(len(r[0, 0, 0, :]) * .5) + offset] + photon_number[:, :, bs2new - 2, offset])
            photon_number[:, :, bs2new - 1, offset] = photon_number[:, :, bs2new - 1, int(len(r[0, 0, 0, :]) * .5) + offset]
            photon_number[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset] = 0.5 * (photon_number[:, :, 1, int(len(r[0, 0, 0, :]) * .5) + offset] + photon_number[:, :, 1, offset])
            photon_number[:, :, 0, offset] = photon_number[:, :, 0, int(len(r[0, 0, 0, :]) * .5) + offset]

def mdot(a, b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k],
    where i,j,k are spatial indices and m,n are variable indices.
    """
    if (a.ndim == 3 and b.ndim == 3) or (a.ndim == 4 and b.ndim == 4):
        c = (a * b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
        # c = np.empty(np.amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)
        c = np.empty((4, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        for i in range(a.shape[0]):
            c[i, :, :, :] = (a[i, :, :, :, :] * b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
        # c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
        c = np.empty((4, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        # print c.shape
        for i in range(b.shape[1]):
            # print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i, :, :, :] = (a * b[:, i, :, :, :]).sum(0)
    elif a.ndim == 5 and b.ndim == 5:
        # c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
        c = np.empty((4, bs1new, bs2new, bs3new), dtype=mytype, order='C')
        # print c.shape
        for i in range(b.shape[1]):
            # print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i, :, :, :] = (a * b[:, i, :, :, :]).sum(0)
    return c

def mdot2(a, b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k],
    where i,j,k are spatial indices and m,n are variable indices.
    """
    if a.ndim == 4 and b.ndim == 3:
        # c = np.empty(np.amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)
        c = np.empty((4, bs1new, bs3new), dtype=mytype, order='C')
        for i in range(a.shape[0]):
            c[i, :, :] = (a[i, :, :, :] * b).sum(0)
    elif a.ndim == 3 and b.ndim == 4:
        # c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
        c = np.empty((4, bs1new, bs3new), dtype=mytype, order='C')
        # print c.shape
        for i in range(b.shape[1]):
            # print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i, :, :] = (a * b[:, i, :, :]).sum(0)
    return c

def psicalc(temp_tilt,temp_prec):
    global aphi, bs1new, bs2new, bs3new
    """
    Computes the field vector potential integrating from both poles to maintain accuracy.
    """

    B1_new=transform_scalar_tot(B[1],temp_tilt,temp_prec)
    aphi=np.zeros((nb,bs1new,bs2new,bs3new),dtype=np.float32)
    aphi2 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)
    daphi = ((gdet * B1_new) * _dx2*_dx3).sum(-1)
    aphi[:,:,:,0] = -daphi[:, :, ::-1].cumsum(axis=2)[:, :, ::-1]
    aphi[:,:,:,0] += 0.5 * daphi  # correction for half-cell shift between face and center in theta
    aphi2[:,:,:,0] = daphi[:, :, :].cumsum(axis=2)[:, :, :]
    aphi[:, :, :bs2new // 2] = aphi2[:, :, :bs2new // 2]
    for z in range(0,bs3new):
        aphi[:, :, :, z]=aphi[:,:,:,0]
    aphi_new=transform_scalar_tot(aphi, -temp_tilt,0)
    aphi_new=transform_scalar_tot(aphi_new,0,-temp_prec)
    aphi=aphi_new


def faraday_new():
    global fdd, fuu, omegaf1, omegaf2, omegaf1b, omegaf2b, rhoc, Bpol
    if 'fdd' in globals():
        del fdd
    if 'fuu' in globals():
        del fuu
    if 'omegaf1' in globals():
        del omegaf1
    if 'omemaf2' in globals():
        del omegaf2
    # these are native values according to HARM
    fdd = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=rho.dtype)
    # fdd[0,0]=0*gdet
    # fdd[1,1]=0*gdet
    # fdd[2,2]=0*gdet
    # fdd[3,3]=0*gdet
    fdd[0, 1] = gdet * (uu[2] * bu[3] - uu[3] * bu[2])  # f_tr
    fdd[1, 0] = -fdd[0, 1]
    fdd[0, 2] = gdet * (uu[3] * bu[1] - uu[1] * bu[3])  # f_th
    fdd[2, 0] = -fdd[0, 2]
    fdd[0, 3] = gdet * (uu[1] * bu[2] - uu[2] * bu[1])  # f_tp
    fdd[3, 0] = -fdd[0, 3]
    fdd[1, 3] = gdet * (uu[2] * bu[0] - uu[0] * bu[2])  # f_rp = gdet*B2
    fdd[3, 1] = -fdd[1, 3]
    fdd[2, 3] = gdet * (uu[0] * bu[1] - uu[1] * bu[0])  # f_hp = gdet*B1
    fdd[3, 2] = -fdd[2, 3]
    fdd[1, 2] = gdet * (uu[0] * bu[3] - uu[3] * bu[0])  # f_rh = gdet*B3
    fdd[2, 1] = -fdd[1, 2]
    #
    fuu = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=rho.dtype)
    # fuu[0,0]=0*gdet
    # fuu[1,1]=0*gdet
    # fuu[2,2]=0*gdet
    # fuu[3,3]=0*gdet
    fuu[0, 1] = -1 / gdet * (ud[2] * bd[3] - ud[3] * bd[2])  # f^tr
    fuu[1, 0] = -fuu[0, 1]
    fuu[0, 2] = -1 / gdet * (ud[3] * bd[1] - ud[1] * bd[3])  # f^th
    fuu[2, 0] = -fuu[0, 2]
    fuu[0, 3] = -1 / gdet * (ud[1] * bd[2] - ud[2] * bd[1])  # f^tp
    fuu[3, 0] = -fuu[0, 3]
    fuu[1, 3] = -1 / gdet * (ud[2] * bd[0] - ud[0] * bd[2])  # f^rp
    fuu[3, 1] = -fuu[1, 3]
    fuu[2, 3] = -1 / gdet * (ud[0] * bd[1] - ud[1] * bd[0])  # f^hp
    fuu[3, 2] = -fuu[2, 3]
    fuu[1, 2] = -1 / gdet * (ud[0] * bd[3] - ud[3] * bd[0])  # f^rh
    fuu[2, 1] = -fuu[1, 2]
    #
    # these 2 are equal in degen electrodynamics when d/dt=d/dphi->0
    omegaf1 = fdd[0, 1] / fdd[1, 3]  # = ftr/frp
    omegaf2 = fdd[0, 2] / fdd[2, 3]  # = fth/fhp
    #
    # from jon branch, 04/10/2012
    #
    # if 0:
    B1hat = B[1] * np.sqrt(gcov[1, 1])
    B2hat = B[2] * np.sqrt(gcov[2, 2])
    B3nonhat = B[3]
    v1hat = uu[1] * np.sqrt(gcov[1, 1]) / uu[0]
    v2hat = uu[2] * np.sqrt(gcov[2, 2]) / uu[0]
    v3nonhat = uu[3] / uu[0]
    #
    aB1hat = np.fabs(B1hat)
    aB2hat = np.fabs(B2hat)
    av1hat = np.fabs(v1hat)
    av2hat = np.fabs(v2hat)
    #
    vpol = np.sqrt(av1hat ** 2 + av2hat ** 2)
    Bpol = np.sqrt(aB1hat ** 2 + aB2hat ** 2)
    #
    # omegaf1b=(omegaf1*aB1hat+omegaf2*aB2hat)/(aB1hat+aB2hat)
    # E1hat=fdd[0,1]*np.sqrt(gn3[1,1])
    # E2hat=fdd[0,2]*np.sqrt(gn3[2,2])
    # Epabs=np.sqrt(E1hat**2+E2hat**2)
    # Bpabs=np.sqrt(aB1hat**2+aB2hat**2)+1E-15
    # omegaf2b=Epabs/Bpabs
    #
    # assume field swept back so omegaf is always larger than vphi (only true for outflow, so put in sign switch for inflow as relevant for disk near BH or even jet near BH)
    # GODMARK: These assume rotation about z-axis
    omegaf2b = np.fabs(v3nonhat) + np.sign(uu[1]) * (vpol / Bpol) * np.fabs(B3nonhat)
    #
    omegaf1b = v3nonhat - B3nonhat * (v1hat * B1hat + v2hat * B2hat) / (B1hat ** 2 + B2hat ** 2)
    #
    # charge
    #
    '''
    if 0:
        rhoc = np.zeros_like(rho)
        if nx>=2:
            rhoc[1:-1] += ((gdet*int(f)uu[0,1])[2:]-(gdet*int(f)uu[0,1])[:-2])/(2*_dx1)
        if ny>2:
            rhoc[:,1:-1] += ((gdet*int(f)uu[0,2])[:,2:]-(gdet*int(f)uu[0,2])[:,:-2])/(2*_dx2)
        if ny>=2 and nz > 1: #not sure if properly works for 2D XXX
            rhoc[:,0,:nz/2] += ((gdet*int(f)uu[0,2])[:,1,:nz/2]+(gdet*int(f)uu[0,2])[:,0,nz/2:])/(2*_dx2)
            rhoc[:,0,nz/2:] += ((gdet*int(f)uu[0,2])[:,1,nz/2:]+(gdet*int(f)uu[0,2])[:,0,:nz/2])/(2*_dx2)
        if nz>2:
            rhoc[:,:,1:-1] += ((gdet*int(f)uu[0,3])[:,:,2:]-(gdet*int(f)uu[0,3])[:,:,:-2])/(2*_dx3)
        if nz>=2:
            rhoc[:,:,0] += ((gdet*int(f)uu[0,3])[:,:,1]-(gdet*int(f)uu[0,3])[:,:,-1])/(2*_dx3)
            rhoc[:,:,-1] += ((gdet*int(f)uu[0,3])[:,:,0]-(gdet*int(f)uu[0,3])[:,:,-2])/(2*_dx3)
        rhoc /= gdet
    '''

def sph_to_cart(X, ph):
    X[1] = np.cos(ph)
    X[2] = np.sin(ph)
    X[3] = 0

# Rotate by angle tilt around y-axis, see wikipedia
def rotate_coord(X, tilt):
    X_tmp = np.copy(X)
    for i in range(1, 4):
        X_tmp[i] = X[i]

    X[1] = X_tmp[1] * np.cos(tilt) + X_tmp[3] * np.sin(tilt)
    X[2] = X_tmp[2]
    X[3] = -X_tmp[1] * np.sin(tilt) + X_tmp[3] * np.cos(tilt)

# Transform coordinates back to spherical
def cart_to_sph(X):
    theta = np.arccos(X[3])
    phi = np.arctan2(X[2], X[1])

    return theta, phi

def sph_to_cart2(X, h, ph):
    X[1] = np.sin(h) * np.cos(ph)
    X[2] = np.sin(h) * np.sin(ph)
    X[3] = np.cos(h)

def calc_scaleheight(tilt, prec, cutoff):
    global rho, gdet, bs1new, h, ph, H_over_R1, H_over_R2,h_new, ti, tj, tk
    X = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tilt_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.int32)
    H_over_R1 = np.zeros((nb, bs1new))
    h_avg = np.zeros((nb, bs1new, 1, 1))
    ph_new = np.copy(ph)
    h_new = np.copy(h)
    uu_proj = project_vector(uu)
    tilt_tmp[0, :, 0, 0] = tilt / 180.0 * np.pi
    prec_tmp[0, :, 0, 0] = prec / 360.0 * bs3new

    for i in range(0, bs1new):
        ph_new[0, i] = np.roll(ph[0, i], prec_tmp[0, i,0,0], axis=1)
    #ph_new[0]=ndimage.map_coordinates(ph_new[0], [[ti], [tj], [(tk-prec_tmp[0])%bs3new]], order=1, mode='nearest')
    sph_to_cart2(X, h_new, ph_new)
    rotate_coord(X, -tilt_tmp)
    h_new, ph_new = cart_to_sph(X)

    norm = (rho * (rho > cutoff) * gdet).sum(-1).sum(-1)
    h_avg[:, :, 0, 0] = (rho * (rho > cutoff) * gdet * h_new).sum(-1).sum(-1) / norm
    H_over_R1 = (rho * (rho > cutoff) * np.abs(h_new - np.pi / 2) * gdet).sum(-1).sum(-1) / norm
    cs_avg=(gdet*(rho>cutoff)*rho**2.0*np.sqrt(np.abs(2.0/np.pi*(gam-1)*ug/(rho+gam*ug)))).sum(-1).sum(-1)
    vrot_avg=(gdet*(rho>cutoff)*rho**2.0*uu_proj[3]/uu[0]).sum(-1).sum(-1)
    H_over_R2=cs_avg/vrot_avg

def set_tilted_arrays(tilt, prec):
    global phi_to_theta, phi_to_phi, theta_to_theta, theta_to_phi
    X = np.zeros((4, nb, bs1new, 1, bs3new), dtype=np.float32)
    X_tmp = np.copy(X)
    ph_old = ph[:, :, bs2new - 1:bs2new, :]
    tilt_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.int32)

    tilt_tmp[0, :, 0, 0] = tilt / 180.0 * np.pi
    prec_tmp[0, :, 0, 0] = prec / 360.0 * bs3new

    sph_to_cart(X, ph_old)
    rotate_coord(X, tilt_tmp)
    h_new, ph_new = cart_to_sph(X)

    X_tmp[1] = -np.sin(ph_old)
    X_tmp[2] = np.cos(ph_old)
    X_tmp[3] = 0.0
    rotate_coord(X_tmp, tilt_tmp)
    theta_to_phi = X_tmp[1] * np.cos(h_new) * np.cos(ph_new) + X_tmp[2] * np.cos(h_new) * np.sin(ph_new) - X_tmp[3] * np.sin(h_new)
    phi_to_phi = -X_tmp[1] * np.sin(ph_new) + X_tmp[2] * np.cos(ph_new)

    X_tmp[1] = 0.0
    X_tmp[2] = 0.0
    X_tmp[3] = -1
    rotate_coord(X_tmp, tilt_tmp)
    theta_to_theta = X_tmp[1] * np.cos(h_new) * np.cos(ph_new) + X_tmp[2] * np.cos(h_new) * np.sin(ph_new) - X_tmp[3] * np.sin(h_new)
    phi_to_theta = -X_tmp[1] * np.sin(ph_new) + X_tmp[2] * np.cos(ph_new)

    for i in range(0, bs1new):
        phi_to_phi[0, i] = np.roll(phi_to_phi[0, i], prec_tmp[0, i, 0, 0], axis=1)
        phi_to_theta[0, i] = np.roll(phi_to_theta[0, i], prec_tmp[0, i, 0, 0], axis=1)
        theta_to_theta[0, i] = np.roll(theta_to_theta[0, i], prec_tmp[0, i, 0, 0], axis=1)
        theta_to_phi[0, i] = np.roll(theta_to_phi[0, i], prec_tmp[0, i, 0, 0], axis=1)

    #phi_to_phi[0]=ndimage.map_coordinates(phi_to_phi[0], [[ti], [tj], [(tk-prec_tmp[0])%bs3new]], order=1, mode='nearest')
    #phi_to_theta[0]=ndimage.map_coordinates(phi_to_theta[0], [[ti], [tj], [(tk-prec_tmp[0])%bs3new]], order=1, mode='nearest')
    #theta_to_theta[0]=ndimage.map_coordinates(theta_to_theta[0], [[ti], [tj], [(tk-prec_tmp[0])%bs3new]], order=1, mode='nearest')
    #theta_to_phi[0]=ndimage.map_coordinates(theta_to_phi[0], [[ti], [tj], [(tk-prec_tmp[0])%bs3new]], order=1, mode='nearest')

def project_vector(vector):
    global phi_to_theta, phi_to_phi, theta_to_theta, theta_to_phi
    vector_proj = np.copy(vector)
    vector_proj[1] = vector[1] * np.sqrt(gcov[1, 1])
    vector_proj[2] = vector[2] * np.sqrt(gcov[2, 2]) * theta_to_theta + vector[3] * np.sqrt(gcov[3, 3]) * phi_to_theta
    vector_proj[3] = vector[2] * np.sqrt(gcov[2, 2]) * theta_to_phi + vector[3] * np.sqrt(gcov[3, 3]) * phi_to_phi

    return vector_proj

def project_vertical(input_var):
    global bs1new, bs2new, x2, bs3new, offset_x2, ti_p, tj_p, tk_p
    output_var = np.copy(input_var)

    # for i in range(0, bs1new):
    #    for z in range(0, bs3new):
    #        output_var[0, i, :, z] = np.roll(input_var[0, i, :, z], np.int32(offset_x2[0, i, 0, z]), axis=0)
    output_var[0] = ndimage.map_coordinates(input_var[0], [[ti_p], [(tj_p + offset_x2[0]) % bs2new], [tk_p]], order=1, mode='nearest')

    return output_var

def preset_project_vertical(var):
    global gdet, bs1new, bs2new, x2, bs3new, offset_x2, rho, ti_p, tj_p, tk_p
    x2_avg = np.zeros((nb, bs1new, 1, bs3new), dtype=np.float32)
    t1 = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    t2 = np.zeros((nb, 1, bs2new, 1), dtype=np.float32)
    t3 = np.zeros((nb, 1, 1, bs3new), dtype=np.float32)
    ti_p = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tj_p = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tk_p = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)

    t1[0, :, 0, 0] = np.arange(bs1new)
    t2[0, 0, :, 0] = np.arange(bs2new)
    t3[0, 0, 0, :] = np.arange(bs3new)

    ti_p[:, :, :, :] = t1
    tj_p[:, :, :, :] = t2
    tk_p[:, :, :, :] = t3

    norm = (var * gdet).sum(2)
    x2_avg[:, :, 0, :] = (var * gdet * x2).sum(2) / norm
    offset_x2 = (x2_avg - x2[0, 0, bs2new // 2, 0]) / _dx2

def misc_calc(calc_bu=1, calc_bsq=1,calc_eu = 0, calc_esq=0):
    global bu, eu, bsq, esq, bs1new,bs2new,bs3new,nb,uu,B,E,gcov, axisym, lum, Ldot, rad_avg
    import pp_c
    if(calc_bu==1):
        bu=np.copy(uu)
    else:
        bu=np.zeros((1, 1, 1, 1, 1), dtype=rho.dtype)
    if (calc_eu == 1):
        eu = np.copy(uu)
    else:
        eu = np.zeros((1, 1, 1, 1, 1), dtype=rho.dtype)
    if (calc_bsq == 1):
        bsq=np.copy(rho)
    else:
        bsq=np.zeros((1, 1, 1, 1), dtype=rho.dtype)
    if (calc_esq == 1):
        esq = np.copy(rho)
    else:
        esq = np.zeros((1, 1, 1, 1), dtype=rho.dtype)
    pp_c.misc_calc(bs1new, bs2new, bs3new, nb,axisym,uu, B,E, bu, eu, gcov, bsq, esq, calc_bu,calc_eu, calc_bsq, calc_esq)

def Tcalcud_new(kapa,nu):
    global gam
    bd_nu = (gcov[nu,:]*bu).sum(0)
    ud_nu = (gcov[nu,:]*uu).sum(0)
    Tud= bsq * uu[kapa] * ud_nu + 0.5 * bsq * (kapa==nu) - bu[kapa] * bd_nu +(rho + ug + (gam - 1) * ug) * uu[kapa] * ud_nu + (gam - 1) * ug * (kapa==nu)
    return Tud

# Matrix inversion
def invert_matrix():
    global dxdxp_inv, dxdr_inv, axisym

    dxdxp_inv = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')
    for i in range(0, bs1new):
        for j in range(0, bs2new):
            dxdxp_inv[:, :, 0, i, j, 0] = np.linalg.inv(dxdxp[:, :, 0, i, j, 0])

def sub_calc_jet_tot(var):
    global gdet, h, tilt_angle, bs2new
    JBH_cross_D = np.zeros((4), dtype=mytype, order='C')
    J_BH = np.zeros((4), dtype=mytype, order='C')
    rin = 10
    rout = 100
    lrho = np.log10(bsq * (rho ** -1))
    var[lrho < 0.5] = 0.0
    var[r < rin] = 0.0
    var[r > rout] = 0.0
    XX = np.sin(h) * np.cos(ph)
    YY = np.sin(h) * np.sin(ph)
    ZZ = np.cos(h)

    tilt = tilt_angle / 180 * 3.141592
    J_BH[1]=-np.sin(tilt)
    J_BH[2]=0
    J_BH[3]=np.cos(tilt)
    J_BH_length=np.sqrt(J_BH[1]*J_BH[1]+J_BH[2]*J_BH[2]+J_BH[3]*J_BH[3])

    var_flux_up_tot = np.zeros(3)
    var_flux_up_tot[0] = np.sum((XX[0, :, 0:int(bs2new // 2), :] * var[0, :, 0:int(bs2new // 2), :] * gdet[0, :, 0:int(bs2new // 2), :]))
    var_flux_up_tot[1] = np.sum((YY[0, :, 0:int(bs2new // 2), :] * var[0, :, 0:int(bs2new // 2), :] * gdet[0, :, 0:int(bs2new // 2), :]))
    var_flux_up_tot[2] = np.sum((ZZ[0, :, 0:int(bs2new // 2), :] * var[0, :, 0:int(bs2new // 2), :] * gdet[0, :, 0:int(bs2new // 2), :]))

    r_up = np.linalg.norm(var_flux_up_tot)
    JBH_cross_D[1] = J_BH[2] * var_flux_up_tot[2] - J_BH[3] * var_flux_up_tot[1]
    JBH_cross_D[2] = J_BH[3] * var_flux_up_tot[0] - J_BH[1] * var_flux_up_tot[2]
    JBH_cross_D[3] = J_BH[1] * var_flux_up_tot[1] - J_BH[2] * var_flux_up_tot[0]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    tilt_angle_jet = np.zeros(2)
    prec_angle_jet = np.zeros(2)
    tilt_angle_jet[0]=np.arccos(np.abs(var_flux_up_tot[0]*J_BH[1]+var_flux_up_tot[1]*J_BH[2]+var_flux_up_tot[2]*J_BH[3])/(J_BH_length*r_up))*180/3.14
    prec_angle_jet[0]=-np.arctan2(JBH_cross_D[1],JBH_cross_D[2])*180/3.14

    var_flux_down_tot = np.zeros(3)
    var_flux_down_tot[0] = np.sum((XX[0, :, int(bs2new // 2):int(bs2new), :] * var[0, :, int(bs2new // 2):int(bs2new), :] * gdet[0,:, int(bs2new // 2):int(bs2new), :]))
    var_flux_down_tot[1] = np.sum((YY[0, :, int(bs2new // 2):int(bs2new), :] * var[0, :, int(bs2new // 2):int(bs2new), :] * gdet[0,:, int(bs2new // 2):int(bs2new), :]))
    var_flux_down_tot[2] = np.sum((ZZ[0, :, int(bs2new // 2):int(bs2new), :] * var[0, :, int(bs2new // 2):int(bs2new), :] * gdet[0,:, int(bs2new // 2):int(bs2new), :]))

    r_down = np.linalg.norm(var_flux_down_tot)
    JBH_cross_D[1] = J_BH[2] * var_flux_down_tot[2] - J_BH[3] * var_flux_down_tot[1]
    JBH_cross_D[2] = J_BH[3] * var_flux_down_tot[0] - J_BH[1] * var_flux_down_tot[2]
    JBH_cross_D[3] = J_BH[1] * var_flux_down_tot[1] - J_BH[2] * var_flux_down_tot[0]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    tilt_angle_jet[1] = np.arccos(np.abs(var_flux_down_tot[0] * J_BH[1] + var_flux_down_tot[1] * J_BH[2] + var_flux_down_tot[2] * J_BH[3]) / (J_BH_length * r_down)) * 180 / 3.14
    prec_angle_jet[1] = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180 / 3.14

    return tilt_angle_jet, prec_angle_jet

def calc_jet_tot():
    global gdet, uu, bu, bsq, rho
    global r, h, ph, bs1new, bs2new, bs3new
    global tilt_angle_jet, prec_angle_jet

    var = np.copy(bsq)
    tilt_angle_jet, prec_angle_jet = sub_calc_jet_tot(var)

def sub_calc_jet(var):
    global tilt_angle,r
    global XX, YY, ZZ, gdet, h
    global angle_jet_var_up, angle_jet_var_down
    global sigma_Ju, gamma_Ju, E_Ju ,mass_Ju,temp_Ju
    global sigma_Jd, gamma_Jd, E_Jd, mass_Jd, temp_Jd
    lrho = np.log10(r**0.25*bsq * (rho ** -1))
    var[lrho < 0.5] = 0.0

    var_flux_cart_down = np.zeros((3, bs1new))
    var_flux_cart_up = np.zeros((3, bs1new))
    angle_jet_var_up = np.zeros((3, bs1new))
    angle_jet_var_down = np.zeros((3, bs1new))
    temp=np.zeros((3,bs1new,1,1))
    var_up = np.zeros((bs1new))
    var_down = np.zeros((bs1new))
    JBH_cross_D = np.zeros((4, nb, bs1new), dtype=mytype, order='C')
    J_BH = np.zeros((4, nb, bs1new), dtype=mytype, order='C')

    tilt = tilt_angle / 180.0 * 3.141592
    x = np.cos(-tilt) * XX - np.sin(-tilt) * ZZ
    y = YY
    z = np.sin(-tilt) * XX + np.cos(-tilt) * ZZ

    crit = np.logical_or(np.logical_and(r <= 25., bu[1] > 0.0), np.logical_and(r > 25., z > 0.0))
    var_u=np.copy(var)
    var_d=np.copy(var)
    var_u[crit<=0]=0.0
    var_d[crit>0]=0.0

    var_down = ((var_d * gdet)).sum(-1).sum(-1)
    var_up = ((var_u * gdet)).sum(-1).sum(-1)
    var_flux_cart_down[0] = ((XX * var_d * gdet)).sum(-1).sum(-1) / var_down
    var_flux_cart_up[0] = ((XX * var_u * gdet)).sum(-1).sum(-1) / var_up
    var_flux_cart_down[1] = ((YY * var_d * gdet)).sum(-1).sum(-1) / var_down
    var_flux_cart_up[1] = ((YY * var_u * gdet)).sum(-1).sum(-1) / var_up
    var_flux_cart_down[2] = ((ZZ * var_d * gdet)).sum(-1).sum(-1) / var_down
    var_flux_cart_up[2] = ((ZZ * var_u * gdet)).sum(-1).sum(-1) / var_up

    J_BH[1] = -np.sin(tilt)
    J_BH[2] = 0
    J_BH[3] = np.cos(tilt)
    J_BH_length = np.sqrt(J_BH[1] * J_BH[1] + J_BH[2] * J_BH[2] + J_BH[3] * J_BH[3])

    JBH_cross_D[1] = J_BH[2] * var_flux_cart_down[2] - J_BH[3] * var_flux_cart_down[1]
    JBH_cross_D[2] = J_BH[3] * var_flux_cart_down[0] - J_BH[1] * var_flux_cart_down[2]
    JBH_cross_D[3] = J_BH[1] * var_flux_cart_down[1] - J_BH[2] * var_flux_cart_down[0]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    rlength = np.sqrt(var_flux_cart_down[0, :] ** 2 + var_flux_cart_down[1, :] ** 2 + var_flux_cart_down[2, :] ** 2)
    angle_jet_var_down[0] = np.arccos(np.abs(var_flux_cart_down[0] * J_BH[1] + var_flux_cart_down[1] * J_BH[2] + var_flux_cart_down[2] * J_BH[3]) / rlength) * 180 / np.pi
    angle_jet_var_down[1] = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180 / 3.141592

    #Calculate opening angle jet
    temp[:,:,0,0]=var_flux_cart_down
    angle_jet_var_down[2] = (((XX[0] - temp[0]) ** 2 + (YY[0] - temp[1]) ** 2 + (ZZ[0] - temp[2]) ** 2) ** 0.5 * gdet[0] * (var_d > 0)[0]).sum(-1).sum(-1) / ((((var_d > 0) * gdet)[0]).sum(-1).sum(-1))
    angle_jet_var_down[2] = (3.0/2.0*angle_jet_var_down[2])/r[0,:,int(bs2new/2),0]/np.pi*180

    #Calculate misc quantaties upper jet
    kapa=1
    nu=0
    bd_nu = (gcov[nu, :] * bu).sum(0)
    ud_nu = (gcov[nu, :] * uu).sum(0)
    TudEM = bsq * uu[kapa] * ud_nu  - bu[kapa] * bd_nu
    TudMA = (rho + ug + (gam - 1) * ug) * uu[kapa] * ud_nu
    volumeu=((TudEM+TudMA)*(var_u!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    sigma_Ju=(TudEM*(var_u!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)/(TudMA*(var_u!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    gamma_Ju=((TudEM+TudMA)*uu[0]*np.sqrt(-1.0/gcon[0,0])*(var_u!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)/volumeu
    E_Ju=((TudEM+TudMA)*(var_u!=0.0)*gdet*_dx2*_dx3).sum(-1).sum(-1)
    mass_Ju=(rho*uu[1]*(var_u!=0.0)*gdet*_dx2*_dx3).sum(-1).sum(-1)
    temp_Ju = ((TudEM+TudMA)*ug/rho * (var_u != 0.0) * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1) / volumeu

    JBH_cross_D[1] = J_BH[2] * var_flux_cart_up[2] - J_BH[3] * var_flux_cart_up[1]
    JBH_cross_D[2] = J_BH[3] * var_flux_cart_up[0] - J_BH[1] * var_flux_cart_up[2]
    JBH_cross_D[3] = J_BH[1] * var_flux_cart_up[1] - J_BH[2] * var_flux_cart_up[0]
    JBH_cross_D_length = np.sqrt(JBH_cross_D[1] * JBH_cross_D[1] + JBH_cross_D[2] * JBH_cross_D[2] + JBH_cross_D[3] * JBH_cross_D[3])

    rlength = np.sqrt(var_flux_cart_up[0, :] ** 2 + var_flux_cart_up[1, :] ** 2 + var_flux_cart_up[2, :] ** 2)
    angle_jet_var_up[0] = np.arccos(np.abs(var_flux_cart_up[0] * J_BH[1] + var_flux_cart_up[1] * J_BH[2] + var_flux_cart_up[2] * J_BH[3]) / rlength) * 180 / 3.14
    angle_jet_var_up[1] = -np.arctan2(JBH_cross_D[1], JBH_cross_D[2]) * 180 / 3.141592

    #Calculate misc quantaties lower jet
    volumed=((TudEM+TudMA)*(var_d!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    sigma_Jd=(TudEM*(var_d!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)/(TudMA*(var_d!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    gamma_Jd=((TudEM+TudMA)*uu[0]*np.sqrt(-1.0/gcon[0,0])*(var_d!=0.0)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)/volumed
    E_Jd=((TudEM+TudMA)*(var_d!=0.0)*gdet*_dx2*_dx3).sum(-1).sum(-1)
    mass_Jd=(rho*uu[1]*(var_d!=0.0)*gdet*_dx2*_dx3).sum(-1).sum(-1)
    temp_Jd = ((TudEM+TudMA)*ug/rho * (var_d != 0.0) * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1) / volumed

    # Calculate opening angle jet
    temp[:, :, 0, 0] = var_flux_cart_up
    angle_jet_var_up[2] = (((XX[0] - temp[0]) ** 2 + (YY[0] - temp[1]) ** 2 + (ZZ[0] - temp[2]) ** 2) ** 0.5 * gdet[0] * (var_u > 0)[0]).sum(-1).sum(-1) / ((((var_u > 0) * gdet)[0]).sum(-1).sum(-1))
    angle_jet_var_up[2] = (3 / 2 * angle_jet_var_up[2]) / r[0, :,int(bs2new//2), 0]/np.pi*180

    return angle_jet_var_up, angle_jet_var_down, var_flux_cart_up, var_flux_cart_down

def calc_jet():
    global Tud, gdet, angle_jetEuu_up, angle_jetEuu_down
    global angle_jetEud_up, angle_jetEud_down
    global angle_jetpud_up, angle_jetpud_down
    global Euu_flux_cart_up, Euu_flux_cart_down
    global XX, YY, ZZ

    XX = (r * np.sin(h) * np.cos(ph))
    YY = (r * np.sin(h) * np.sin(ph))
    ZZ = (r * np.cos(h))

    angle_jetEuu_up = np.zeros((2, bs1new))
    angle_jetEuu_down = np.zeros((2, bs1new))
    Euucut = np.copy(bsq/rho)

    angle_jetEuu_up, angle_jetEuu_down, Euu_flux_cart_up, Euu_flux_cart_down = sub_calc_jet(Euucut)

# Calculate alpha viscosity parameter assuming no tilt
def calc_alpha(cutoff):
    global alpha_r,alpha_b, alpha_eff, gam, pitch_avg
    norm=(gdet*rho*(rho>cutoff)).sum(-1).sum(-1)
    fact=(gdet*rho*(rho>cutoff))
    v_avg1 = np.zeros((nb, bs1new, 1, 1))
    v_avg3 = np.zeros((nb, bs1new, 1, 1))
    bu_proj = project_vector(bu)
    uu_proj = project_vector(uu)
    ptot=(fact*((bsq / 2) + (gam - 1) * ug)).sum(-1).sum(-1)

    alpha_b = (fact*(bu_proj[1] * bu_proj[3])).sum(-1).sum(-1) / ptot

    v_avg1[:, :, 0, 0] = (fact * uu_proj[1]).sum(-1).sum(-1)/norm
    v_avg3[:, :, 0, 0] = (fact * uu_proj[3]).sum(-1).sum(-1) / norm
    alpha_r = (fact*(rho+bsq+gam*ug) * (uu_proj[1] - v_avg1) * (uu_proj[3] - v_avg3)).sum(-1).sum(-1) / ptot
    cs = np.sqrt(np.abs(gam * (gam - 1.0) * ug / (rho + ug + (gam - 1.0) * ug)))

    v_r = uu_proj[1]
    v_or = uu_proj[3]
    alpha_eff = (fact*v_r * v_or/uu[0]/uu[0]).sum(-1).sum(-1) / (fact*(cs ** 2)).sum(-1).sum(-1)

    pitch_avg = (fact * np.sqrt(bu_proj[1] * bu_proj[1] + bu_proj[2] * bu_proj[2])).sum(-1).sum(-1) / (fact * np.sqrt(bu_proj[3] * bu_proj[3])).sum(-1).sum(-1)

# Print total mass of disk in code units
def calc_Mtot():
    global Mtot
    Mtot = np.sum((rho * uu[0]) * _dx1 * _dx2 * _dx3 * gdet)

# Calculate precession period
def calc_PrecPeriod(angle_tilt):
    global gam,a,precperiod
    uu_proj = project_vector(uu)
    L = rho * r * uu_proj[3]
    tilt=np.zeros((nb,bs1new,1,1),dtype=np.float32)
    tilt[0,:,0,0]=(np.nan_to_num(angle_tilt)+0.1)/360.0*2.0*np.pi
    Z1 = 1.0 + (1.0 - a ** 2.0) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    Z2 = np.sqrt(3.0 * a ** 2.0 + Z1 ** 2.0)
    r_isco = (3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))
    L_tot = np.nan_to_num(L * gdet * _dx1 * _dx2 * _dx3 * np.sin(tilt)* (r > r_isco) * (r < 150)).sum(-1).sum(-1).sum(-1)
    vnod=1.0/(r**1.5+a)*(1.0-np.sqrt(1.0-4.0*a/r**1.5+3.0*a*a/r**2))
    tau_tot = np.nan_to_num(L * vnod * gdet * _dx1 * _dx2 * _dx3 * np.sin(tilt)* (r > r_isco) * (r < 150)).sum(-1).sum(-1).sum(-1)
    precperiod = 2 * np.pi * L_tot / tau_tot

# Calculate mass accretion rate as function of radius
def calc_Mdot():
    global Mdot
    Mdot = (-gdet * rho * uu[1] * _dx2 * _dx3).sum(-1).sum(-1)

def calc_profiles(cutoff):
    global pgas_avg, rho_avg, pb_avg, Q_avg1_1,Q_avg1_2,Q_avg1_3, Q_avg2_1,Q_avg2_2,Q_avg2_3
    calc_Q()
    norm1 = (gdet * rho * (rho > cutoff)).sum(-1).sum(-1)
    norm2 = (gdet * rho * (rho > cutoff)).sum(-1).sum(-1)
    norm3 = (gdet * np.sqrt(np.abs(rho * bsq)) * (rho > cutoff)).sum(-1).sum(-1)
    fact1 = gdet * rho * (rho > cutoff)
    fact2 = gdet * np.sqrt(np.abs(rho * bsq)) * (rho > cutoff)
    pgas_avg = (fact1 * (gam - 1.0) * ug).sum(-1).sum(-1) / norm1
    pb_avg = (fact1 * bsq / 2.0).sum(-1).sum(-1) / norm1
    rho_avg = (gdet * (rho > cutoff) * rho * rho)[:, :, :, :].sum(-1).sum(-1) / (gdet * rho* (rho > cutoff)).sum(-1).sum(-1)
    Q_avg1_1 = (fact1 * Q[1]).sum(-1).sum(-1) / norm2
    Q_avg1_2 = (fact1 * Q[2]).sum(-1).sum(-1) / norm2
    Q_avg1_3 = (fact1 * Q[3]).sum(-1).sum(-1) / norm2
    Q_avg2_1 = (fact2 * Q[1]).sum(-1).sum(-1) / norm3
    Q_avg2_2 = (fact2 * Q[2]).sum(-1).sum(-1) / norm3
    Q_avg2_3 = (fact2 * Q[3]).sum(-1).sum(-1) / norm3

def calc_lum():
    global lum
    p = (gam - 1.0) * ug
    lum = np.sum((rho ** 3.0 * p ** (-2.0) * np.exp(-0.2 * (rho ** 2.0 / (np.sqrt(bsq) * p ** 2.0)) ** (1.0 / 3.0)) * (h > np.pi / 3.0) * (h < 2.0 / 3.0 * np.pi) * (r < 50.) * gdet * _dx1 * _dx2 * _dx3))

def calc_rad_avg():
    global rad_avg
    rad_avg = (r * rho * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1).sum(-1) / ((rho * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1).sum(-1))

# Calculate energy accretion rate as function of radius
def calc_Edot():
    global Edot, Edotj
    temp=Tcalcud_new(1, 0)* gdet * _dx2 * _dx3
    Edot = (temp).sum(-1).sum(-1)
    Edotj = (temp*(bsq/rho>3)).sum(-1).sum(-1)

def calc_Ldot():
    global Ldot
    Ldot = (Tcalcud_new(1, 3)* gdet * _dx2 * _dx3).sum(-1).sum(-1)

# Calculate magnetic flux phibh as function of radius
def calc_phibh():
    global phibh
    phibh = 0.5 * (np.abs(gdet * B[1]) * _dx2 * _dx3).sum(-1).sum(-1)

# Calculate the Q resolution paramters and their average Q_avg in the disk
def calc_Q():
    global Q, Q_avg, lowres1, lowres2, lowres3
    Q = np.zeros((4, 1, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    Q_avg = np.zeros((4), dtype=np.float32, order='C')
    dx = np.zeros((4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')

    dx[1] = _dx1 / lowres1 * np.sqrt(gcov[1, 1, :, :, :, :])
    dx[2] = _dx2 / lowres2 * np.sqrt(gcov[2, 2, :, :, :, :])
    dx[3] = _dx3 / lowres3 * np.sqrt(gcov[3, 3, :, :, :, :])
    bu_proj = project_vector(bu)

    for dir in range(1, 4):
        alf_speed = np.sqrt(np.abs(bu_proj[dir] * bu_proj[dir]) / (rho + bsq + (gam) * ug))
        vrot = np.sqrt((uu[3] * uu[3] * gcov[3][3] + uu[2] * uu[2] * gcov[2][2] + uu[1] * uu[1] * gcov[1][1])) / uu[0]
        wavelength = 2 * 3.14 * alf_speed * r / vrot
        if (dir == 1):
            Q[dir] = wavelength / dx[dir]
        if (dir == 2):
            Q[dir] = wavelength / (dx[2]*theta_to_theta+dx[3]*np.abs(phi_to_theta))
        if (dir == 3):
            Q[dir] = wavelength / (dx[2]*np.abs(theta_to_phi) + dx[3] * phi_to_phi)
        Q[dir] = np.nan_to_num(Q[dir])

# Plot aspect ratio of jcell from the polar axis
def plot_aspect(jcell=50):
    aspect = _dx1 * dxdxp[1, 1, :, :, :, 0] / (r[:, :, :, 0] * (_dx2 * dxdxp[2, 2, :, :, :, 0]))
    for i in range_1(0, nb):
        plt.plot(r[i, :, jcell], aspect[i, :, jcell])
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.xlabel(r"$\log_{10}r/R_{g}$")
    plt.ylabel(r"$dz/dR$")
    plt.savefig("aspect_ratio.png", dpi=300)

# Print precession angle as function of radius
def plot_precangle():
    fig = plt.figure(figsize=(6, 6))
    for i in range(0, 1):
        plt.plot(r[i, :, 0, 0], angle_prec[i], color="blue", label=r"S25A93", lw=2)
    plt.xlim(0, 150)
    plt.ylim(0, 60)
    plt.xlabel(r"$r(R_{G})$", size=30)
    plt.ylabel(r"${\rm Precession\ angle\ } \gamma$", size=30)
    plt.savefig("GammavsR0.png", dpi=300)

# Calculate and plot surface density
def plot_SurfaceDensity():
    SD = (rho * np.sqrt(gcov[2, 2]) * _dx2).sum(3).sum(2)
    plt.plot(np.log10(r[0, :, bs2new // 2, 0]), np.log10(SD[0]))
    plt.xlim(0, 4)
    plt.ylim(0, 5)
    plt.xlabel(r"$\rm r(R_{G})$", size=30)
    plt.ylabel(r"\rm Surface density", size=30)
    plt.savefig("SD.png", dpi=300)

# Print tilt angle as function of radius
def plot_tiltangle():
    fig = plt.figure(figsize=(6, 6))
    for i in range(0, nb):
        plt.plot(r[i, :, 0, 0], angle_tilt[i], color="blue", label=r"S25A93", lw=2)
    plt.xlim(0, 150)
    plt.ylim(0, 45)
    plt.xlabel(r"$\rm r(R_{G})$", size=30)
    plt.ylabel(r"\rm Tilt $\alpha$", size=30)
    plt.savefig("TiltvsR0.png", dpi=300)

def get_longest_path_vertices(cs, index):
    maxlen = 0
    maxind = -1
    paths = cs.collections[0].get_paths()
    for i, p in enumerate(paths):
        lenp = len(p.vertices)
        if lenp > maxlen:
            maxlen = lenp
            maxind = index
    if maxind < 0:
        print("No paths found, using default one (0)")
        maxind = 0
    print(maxind)
    return cs.collections[0].get_paths()[maxind].vertices

# Precalculates the parameters along the jet's field lines
def precalc_jetparam():
    faraday_new()
    Tcalcud_new()
    global ci, cj, cr, cfitr, cbckeck, cbunching, cresult, cuu0, ch, ceps, comega, cmu, csigma, csigma1, csigma2, cbsq, cbsqorho, cbsqoug, crhooug, chm87, cBpol, cuupar
    import scipy.ndimage as ndimage
    nb2d = 1
    cs = [None] * nb2d
    v = [None] * nb2d
    cfitr = [None] * nb2d
    cbcheck = [None] * nb2d
    cbunching = [None] * nb2d
    ccurrent = [None] * nb2d
    cresult = [None] * nb2d
    ci = [None] * nb2d
    cj = [None] * nb2d
    cr = [None] * nb2d
    ceps = [None] * nb2d
    ch = [None] * nb2d
    cuu0 = [None] * nb2d
    comega = [None] * nb2d
    cmu = [None] * nb2d
    crho = [None] * nb2d
    cug = [None] * nb2d
    csigma = [None] * nb2d
    csigma1 = [None] * nb2d
    csigma2 = [None] * nb2d
    cbsq = [None] * nb2d
    cbsqorho = [None] * nb2d
    cbsqoug = [None] * nb2d
    crhooug = [None] * nb2d
    chm87 = [None] * nb2d
    cBpol = [None] * nb2d
    cuupar = [None] * nb2d
    # cs=plc_new(aphi,levels=(0.55*0.65*aphi.max(),),xcoord=ti, ycoord=tj,xy=0,colors="red")

    nr = 0  # number of radial lines
    cd = []
    cdi = []
    cdj = []
    vd = []
    Bpold = []
    cuu0d = []
    cugd = []
    chd = []
    for ri in range(0, nr):
        cd.append([])
        cdi.append([])
        cdj.append([])
        vd.append([])
        Bpold.append([])
        cuu0d.append([])
        cugd.append([])
        chd.append([])
        for i in range(0, nb2d):
            cd[ri].append(i + ri)
            cdi[ri].append(i + ri)
            cdj[ri].append(i + ri)
            vd[ri].append(i + ri)
            Bpold[ri].append(i + ri)
            cuu0d[ri].append(i + ri)
            cugd[ri].append(i + ri)
            chd[ri].append(i + ri)
    index = [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for ri in range(0, nr):
        cd[ri] = plc_new(np.log10(r), levels=(ri + 1.0,), colors="red", xcoord=ti, ycoord=tj, xy=0)

    for i in range(0, nb2d):
        if (tk[i, 0, 0, 0] == 0):

            for ri in range(0, nr):
                vd[ri][i] = get_longest_path_vertices(cd[ri][i], index[i])
                cdi[ri][i] = vd[ri][i][:, 0]
                cdj[ri][i] = vd[ri][i][:, 1]

                # Bpold[ri][i]=ndimage.map_coordinates(Bpol[i,:,:,0],np.array([cdi[ri],cdj[ri]]),order=1,mode="nearest")
                cuu0d[ri][i] = ndimage.map_coordinates((uu[0])[i, :, :, 0], np.array([cdi[ri], cdj[ri]]), order=1, mode="nearest")
                chd[ri][i] = ndimage.map_coordinates(h[i, :, :, 0], np.array([cdi[ri], cdj[ri]]), order=1,mode="nearest")
                cugd[ri][i] = ndimage.map_coordinates((ug)[i, :, :, 0], np.array([cdi[ri], cdj[ri]]), order=1,mode="nearest")

            k = plc_new(aphi, levels=(0.25 * 0.65 * aphi.max(),), xcoord=ti, i2=i, ycoord=tj, xy=0, colors="red")
            v[i] = get_longest_path_vertices(k, index[i])

            ci[i] = v[i][:, 0]
            cj[i] = v[i][:, 1]
            nu = 1.2
            # cfitr[i]=ndimage.map_coordinates(((Bpol/Bpol+(omegaf2*r*np.sin(h))**2)**0.5)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            # csigma1[i]=ndimage.map_coordinates((np.abs(mu/(omegaf2*r*np.sin(h))))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            # csigma2[i]=ndimage.map_coordinates((mu*h/3.5)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")

            # cbcheck[i]=ndimage.map_coordinates((Bpol**2/(bsq-Bpol**2))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            # cbcheck[i]=ndimage.map_coordinates(((bsq-Bpol**2)/rho)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cbunching[i] = ndimage.map_coordinates((3.14 * (r * np.sin(h)) ** 2 * Bpol / (3.14 * (r * np.sin(h)) ** 2 * Bpol)[i, 500, 0, 0])[i, :, :, 0],np.array([ci[i], cj[i]]), order=1, mode="nearest")
            # ccurrent[i]= ndimage.map_coordinates((np.sqrt(np.abs(B[3]*B[3]*gcov[3,3]+2*B[3]*B[1]*gcov[3,1]+2*B[3]*B[2]*gcov[3,2]+
            #                       2*B[3]*B[0]*gcov[3,0]))*r*np.sin(h))[i,:,:,0],np.array([[ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cresult[i] = ndimage.map_coordinates((sigma ** -0.5 * uu[0])[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            cr[i] = ndimage.map_coordinates(r[i, :, :, 0], np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]),order=1, mode="nearest")
            ceps[i] = ndimage.map_coordinates((rho * uu[1] / B[1])[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            ch[i] = ndimage.map_coordinates(h[i, :, :, 0], np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]),order=1, mode="nearest")
            cuu0[i] = ndimage.map_coordinates((uu[0])[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            comega[i] = ndimage.map_coordinates((omegaf2)[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            cmu[i] = ndimage.map_coordinates(mu[i, :, :, 0], np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]),order=1, mode="nearest")
            crho[i] = ndimage.map_coordinates(rho[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            crhooug[i] = ndimage.map_coordinates((rho / ug)[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            cug[i] = ndimage.map_coordinates(ug[i, :, :, 0], np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]),order=1, mode="nearest")
            csigma[i] = ndimage.map_coordinates(sigma[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            # cbsq[i]=ndimage.map_coordinates((bsq)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            # cbsqorho[i] = ndimage.map_coordinates((bsq/rho)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cbsqoug[i] = ndimage.map_coordinates((bsq / ug)[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            crhooug[i] = ndimage.map_coordinates((rho / ug)[i, :, :, 0],np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            chm87[i] = ndimage.map_coordinates((r ** (-0.42) / 3.8)[i, :, :, 0], np.array([ci[i] - ti[i, 0, 0, 0], cj[i] - tj[i, 0, 0, 0]]), order=1,mode="nearest")
            # cBpol=ndimage.map_coordinates(Bpol[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            # cuupar[i]=ndimage.map_coordinates((uu[1]*np.sqrt(gcov[1,1]))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            plt.plot(ci[i], cj[i], label=r"$\gamma$", color="blue", lw=1)

def plt_jetparam():
    global which
    nb2d = 1
    clen = [None] * nb2d
    ind = [None] * nb2d
    ind1 = [None] * nb2d
    ind2 = [None] * nb2d
    inds = [None] * nb2d
    indmax = [None] * nb2d
    indmax1 = [None] * nb2d
    indmax2 = [None] * nb2d

    whichpoles = [0, 1]
    lws = [3, 2]

    fig = plt.figure(figsize=(12, 8))
    plt.tick_params('both', length=5, width=2, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    firsttime = 1

    maxi = 0

    for i in range(0, nb2d):
        maxi = np.max(ci[i], maxi)
    for i in range(0, nb2d):
        if (tk[i, 0, 0, 0] == 0):

            clen[i] = len(cr[i])
            ind[i] = np.arange(clen[i])
            # indmax[i] = (np.where(ci[i] < maxi))[0][0]
            indmax1[i] = (np.where(ci[i][:clen[i] // 2] == np.max(ci[i][:clen[i] // 2])))[0][0]
            indmax2[i] = (np.where(ci[i][clen[i] // 2:] == np.max(ci[i][clen[i] // 2:])))[0][0] + clen[i] // 2
            ind1[i] = ind[i] < indmax1[i]
            ind2[i] = ind[i] > indmax2[i]
            inds[i] = [ind1[i], ind2[i]]

            for whichpole, lw in zip(whichpoles, lws):
                which = inds[i][whichpole]

                # plt.plot(cr[i][which],cbunching[i][which]/10000000,label=r"$a_{fp}$",color="cyan",lw=lw)
                # plt.plot(cr[i][which],current[i][which],label=r"$I$",color="purple",lw=lw)
                # plt.plot(cr[i][which],cresult[i][which],label=r"$\delta$",color="orange",lw=lw)
                plt.plot(cr[i][which], cuu0[i][which], label=r"$\gamma$", color="red", lw=lw)
                plt.plot(cr[i][which], csigma[i][which], label=r"$\sigma$", color="green", lw=lw)
                # plt.plot(cr[i][which],csigma1[i][which],label=r"$\sigma_{1}$",color="grey",lw=lw)
                # plt.plot(cr[i][which],cbsqorho[i][which],label=r"$b^2/\rho$",color="cyan",lw=lw)
                # plt.plot(cr[i][which],crhooug[i][which],label=r"$\rho/u_{g}$",color="purple",lw=lw)
                plt.plot(cr[i][which], cmu[i][which], label=r"$\mu$", color="blue", lw=lw)
                plt.plot(cr[i][which], 1000 * ceps[i][which], label=r"$\mu$", color="cyan", lw=lw)

                # plt.plot(cr[i][which],comega[i][which],label=r"$\omega$",color="cyan",lw=lw)
                # plt.plot(cr[i][which],10000*cug[i][which],label=r"ug",color="pink",lw=lw)
                # plt.plot(cr[i],cbsqorho[i],label=r"$10^4\rho$",color="magenta",lw=lw)
                # plt.plot(cr[i][which],cbcheck[i][which],label="Bpol",color="magenta",lw=lw)
                # plt.plot(cr[i][which],0.5*cr[which]**(-2.5*5/3),label=r"$90r^{-3/2}$",color="orange",lw=lw)
                plt.plot(cr[i][which], 2 * chm87[i][which], label=r"$\theta_{M87}$", color="purple", lw=lw)
                # plt.plot(cr[i][which],cBpol[i][which],label=r"$\theta_{M87}$",color="yellow",lw=lw)
                if whichpole == 0:
                    plt.plot(cr[i][which], ch[i][which] * cresult[i][which], label=r"$\gamma*\theta/\sigma^{0.5}$",
                             color="orange", lw=lw)
                    plt.plot(cr[i][which], ch[i][which], label=r"$\theta_{Matthew}$", color="black", lw=lw)
                    # plt.plot(cr[i][which],cmu[i][which]*ch[i][which]/3.84,label=r"$\sigma_{2}$",color="cyan",lw=lw)
                else:
                    plt.plot(cr[i][which], (np.pi - ch[i][which]) * cresult[i][which],
                             label=r"$\gamma*\theta/\sigma^{0.5}$", color="orange", lw=lw)
                    plt.plot(cr[i][which], np.pi - ch[i][which], label=r"$\theta$", color="black", lw=lw)
                    # plt.plot(cr[i][which],cmu[i][which]*(np.pi-ch[i][which])/3.84,label=r"$\sigma_{2}$",color="cyan",lw=lw)
                if firsttime == 1:
                    plt.legend(loc="upper right", frameon=False, ncol=4)
                    # plt.xlim(rhor,t+100)
                    plt.ylim(1e-3, 1e3)
                    axis_font = {'fontname': 'Arial', 'size': '24'}
                    plt.tick_params(axis='both', which='major', labelsize=24)
                    plt.tick_params(axis='both', which='minor', labelsize=24)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel(r"$\log_{10}(r/R_{g})$", fontsize=30)
                    plt.grid(b=1)
                firsttime = 0
    plt.savefig("evolution.png", dpi=300)

def plt_jetparam_trans():
    R = [None] * nr
    powexp = 2  # set to 10 for real plots to lower for debuggin
    for i in range(0, nb2d):
        if (tk[i, 0, 0, 0] == 0):
            for ri in range(3, 4):
                j = 0
                while cr[i][j] < 20000:
                    j += 1
                R[ri] = plt.scatter(np.sin(chd[ri][i]) / np.sin(ch[i][j]), np.log10(Bpold[ri][i]), color="red", lw=1)
    '''plt.legend((R[0], R[1], R[2], R[3], R[4]),
               (r"$r=$10^1$ R_{g}$",r"$r=$10^2$ R_{g}$",r"$r=$10^3$ R_{g}$", r"$r=$10^4$ R_{g}$",r"$r=$10^5$ R_{g}$"),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=16)'''
    plt.xlim(0, 0.99)
    plt.ylim(-5, -3)
    plt.xlabel(r"$\log_{10}R/R_{edge}$")
    plt.ylabel(r"$\log_{10}B_{p}$")
    plt.savefig("core.png", dpi=300)

#Sets kerr-schild coordinates
def set_uniform_grid():
    global x1, x2, x3, r, h, ph, bs1new, bs2new, bs3new, startx1, startx2, startx3, _dx1, _dx2, _dx3

    for i in range(0, bs1new):
        x1[:, i, :, :] = startx1 + (i+0.5) * _dx1
    for j in range(0, bs2new):
        x2[:, :, j, :] = startx2 + (j+0.5) * _dx2
    for z in range(0, bs3new):
        x3[:, :, :, z] = startx3 + (z+0.5) * _dx3

    r = np.exp(x1)
    h = (x2+1.0)/2.0*np.pi
    ph = x3

# Calculate uniform coordinates and Kerr-Schild metric for Ray-Tracing
def set_KS():
    global gcov_kerr, x1, x2, x3, r, h, ph, bs1new, bs2new, bs3new
    # Set covariant Kerr metric in double
    gcov_kerr = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32)

    set_uniform_grid()

    cth = np.cos(h)
    sth = np.sin(h)
    s2 = sth * sth
    rho2 = r * r + a * a * cth * cth

    gcov_kerr[0, 0, 0, :, :, 0] = (-1. + 2. * r / rho2)[0, :, :, 0]
    gcov_kerr[0, 1, 0, :, :, 0] = (2. * r / rho2)[0, :, :, 0]
    gcov_kerr[0, 3, 0, :, :, 0] = (-2. * a * r * s2 / rho2)[0, :, :, 0]
    gcov_kerr[1, 0, 0, :, :, 0] = gcov_kerr[0, 1, 0, :, :, 0]
    gcov_kerr[1, 1, 0, :, :, 0] =  (1. + 2. * r / rho2)[0, :, :, 0]
    gcov_kerr[1, 3, 0, :, :, 0] = (-a * s2 * (1. + 2. * r / rho2))[0, :, :, 0]
    gcov_kerr[2, 2, 0, :, :, 0] = rho2[0, :, :, 0]
    gcov_kerr[3, 0, 0, :, :, 0] = gcov_kerr[0, 3, 0, :, :, 0]
    gcov_kerr[3, 1, 0, :, :, 0] = gcov_kerr[1, 3, 0, :, :, 0]
    gcov_kerr[3, 3, 0, :, :, 0] = (s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2)))[0, :, :, 0]

    # Invert coviariant metric to get contravariant Kerr Schild metric
    #gcon_kerr = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float64)
    #for i in range(0, bs1new):
    #    for j in range(0, bs2new):
    #        gcon_kerr[:, :, 0, i, j, 0] = np.linalg.inv(gcov_kerr[:, :, 0, i, j, 0])

# Make file for raytracing
def dump_RT_BHOSS(dir, dump):
    global gcov, gcon,uu,bu,r,rho,ug, N1,bs1new,bs2new,bs3new
    # Set outputfile parameters
    thetain = 0.
    thetaout = np.pi
    phiin = 0.
    phiout = 2.0 * np.pi
    for i in range(0,bs1new):
        while (r[0, i, int(bs2new/2), 0] <100 or i == int(bs1new)-1):
            N1 = i
            break
    Rout = r[0, N1, 0, 0]
    rhoflr = (1 * 10 ** -6) * Rout ** (-2)
    pflr = (1 * 10 ** -7) * ((Rout ** -2) ** gam)
    metric = 1
    code = 2
    dim = 3

    # Allocate new arrays for output
    uukerr = np.copy(uu)

    # Set grid parameters alpha and beta in Kerr-Schild coordinates
    beta = np.zeros((4, nb, bs1new, bs2new, 1), dtype=np.float64)
    vkerr = uukerr / uukerr
    Bkerr = uukerr / uukerr

    # Transform to Kerr-Schild 4-velocity
    alpha = 1. / (-gcon[0, 0, 0]) ** 0.5
    beta[1:4, 0] = gcon[0, 1:4, 0] * alpha * alpha
    Bkerr[1:4, 0] = alpha * uu[0, 0] * bu[1:4, 0] - alpha * bu[0, 0] * uu[1:4, 0]
    vkerr[1:4, 0] = (beta[1:4, 0] + uu[1:4, 0] / uu[0, 0]) / alpha
    vkerr[:, 0] = mdot(dxdxp[:, :, 0], vkerr[:, 0])
    Bkerr[:, 0] = mdot(dxdxp[:, :, 0], Bkerr[:, 0])
    pressure=(gam-1.0)*ug

    # Start writing to binary
    if (1):
        import struct
        f = open(dir + "/RT/rt%d"%dump, "wb+")
        header = [N1, bs2new, bs3new]
        head = struct.pack('i' * 3, *header)
        f.write(head)
        header = [t, a, Rin, thetain, phiin, Rout, thetaout, phiout, rhoflr, pflr]
        head = struct.pack('d' * 10, *header)
        f.write(head)
        header = [metric, code, dim]
        head = struct.pack('i' * 3, *header)
        f.write(head)

        for z in range(0,bs3new):
            for j in range (0,bs2new):
                for i in range(0,N1):
                    data = [r[0,i,j,z],h[0,i,j,z],ph[0,i,j,z],rho[0,i,j,z],vkerr[1,0,i,j,z],vkerr[2,0,i,j,z],vkerr[3,0,i,j,z],pressure[0,i,j,z],Bkerr[1,0,i,j,z],Bkerr[2,0,i,j,z],Bkerr[3,0,i,j,z]]
                    s = struct.pack('f'*11, *data)
                    f.write(s)
        f.close()

def Tcalcuu():
    global Tuu, uu, bu, dxdr, dxdxp, bsq, rho, ug, gam, gcon
    Tuu = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float64, order='C')

    for kappa in np.arange(4):
        for nu in np.arange(4):
            Tuu[kappa, nu] = bsq * uu[kappa] * uu[nu] + (0.5 * bsq + (gam - 1) * ug) * gcon[kappa, nu] - bu[kappa] * bu[nu] + (rho + gam * ug) * uu[kappa] * uu[nu]

def mdot2(a, b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k],
    where i,j,k are spatial indices and m,n are variable indices.
    """
    if a.ndim == 4 and b.ndim == 3:
        # c = np.empty(np.amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)
        c = np.empty((4, bs1new, bs3new), dtype=mytype, order='C')
        for i in range(a.shape[0]):
            c[i, :, :] = (a[i, :, :, :] * b).sum(0)
    elif a.ndim == 3 and b.ndim == 4:
        # c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
        c = np.empty((4, bs1new, bs3new), dtype=mytype, order='C')
        # print c.shape
        for i in range(b.shape[1]):
            # print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i, :, :] = (a * b[:, i, :, :]).sum(0)
    return c


def calc_transformations(temp_tilt, temp_prec):
    global dxdxp, dxdxp_inv, dxdr, dxdr_inv, drtdr, drtdr_inv, r, h, ph, bs1new, bs2new, bs3new
    drtdr = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    drtdr_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdr = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdr_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdxp_inv = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')

    # Set tilt and precession angle in larger array
    tilt = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    tilt[0, :, 0, 0] = temp_tilt / 180.0 * np.pi
    prec[0, :, 0, 0] = temp_prec / 360.0 * 2.0 * np.pi

    # Transformation matrix from kerr-schild to modified kerr-schild
    for i in range(0, bs1new):
        for j in range(0, bs2new):
            dxdxp_inv[:, :, 0, i, j, 0] = np.linalg.inv(dxdxp[:, :, 0, i, j, 0])

    # Transformation matrix to Cartesian Kerr Schild from Spherical Kerr Schild
    dxdr[0, 0] = 1
    dxdr[0, 1] = 0
    dxdr[0, 2] = 0
    dxdr[0, 3] = 0
    dxdr[1, 0] = 0
    dxdr[1, 1] = (np.sin(h) * np.cos(ph))
    dxdr[1, 2] = (r * np.cos(h) * np.cos(ph))
    dxdr[1, 3] = (-r * np.sin(h) * np.sin(ph))
    dxdr[2, 0] = 0
    dxdr[2, 1] = (np.sin(h) * np.sin(ph))
    dxdr[2, 2] = (r * np.cos(h) * np.sin(ph))
    dxdr[2, 3] = (r * np.sin(h) * np.cos(ph))
    dxdr[3, 0] = 0
    dxdr[3, 1] = (np.cos(h))
    dxdr[3, 2] = (-r * np.sin(h))
    dxdr[3, 3] = 0

    # Set coordinates
    x0 = (r * np.sin(h) * np.cos(ph))
    y0 = (r * np.sin(h) * np.sin(ph))
    z0 = (r * np.cos(h))

    xt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.cos(tilt) - z0 * np.sin(tilt))
    yt = (y0 * np.cos(prec) + x0 * np.sin(prec))
    zt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.sin(tilt) + z0 * np.cos(tilt))

    rt = np.sqrt(xt * xt + yt * yt + zt * zt)
    ht = np.arccos(zt / rt)
    pht = np.arctan2(yt, xt)

    # Transformation matrix to Spherical Kerr Schild from Cartesian Kerr Schild
    for i in range(0, bs1new):
        for j in range(0, bs2new):
            for z in range(0, bs3new):
                dxdr_inv[:, :, 0, i, j, z] = np.linalg.inv(dxdr[:, :, 0, i, j, z])

    for i in range(0, bs1new):
        print(i)
        # Alloccate temporary arrays
        dxtdx = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')
        dxtdr = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')
        dxtdrt = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')
        dxtdrt_inv = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')

        # Transformation matrix to to tilted Cartesian Kerr Schild from Cartesian Kerr Schild
        dxtdx[0, 0] = 1
        dxtdx[0, 1] = 0
        dxtdx[0, 2] = 0
        dxtdx[0, 3] = 0
        dxtdx[1, 0] = 0
        dxtdx[1, 1] = (np.cos(tilt[:, i]) * np.cos(prec[:, i]))
        dxtdx[1, 2] = (-np.cos(tilt[:, i]) * np.sin(prec[:, i]))
        dxtdx[1, 3] = (-np.sin(tilt[:, i]))
        dxtdx[2, 0] = 0
        dxtdx[2, 1] = (np.sin(prec[:, i]))
        dxtdx[2, 2] = (np.cos(prec[:, i]))
        dxtdx[2, 3] = 0
        dxtdx[3, 0] = 0
        dxtdx[3, 1] = (np.sin(tilt[:, i]) * np.cos(prec[:, i]))
        dxtdx[3, 2] = (-np.sin(tilt[:, i]) * np.sin(prec[:, i]))
        dxtdx[3, 3] = (np.cos(tilt[:, i]))

        # Calculate transformation matrix from tilted Cartesian to tilted Kerr-Schild
        dxtdrt[0, 0] = 1
        dxtdrt[0, 1] = 0
        dxtdrt[0, 2] = 0
        dxtdrt[0, 3] = 0
        dxtdrt[1, 0] = 0
        dxtdrt[1, 1] = (np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 2] = (rt[:, i] * np.cos(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 3] = (-rt[:, i] * np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 0] = 0
        dxtdrt[2, 1] = (np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 2] = (rt[:, i] * np.cos(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 3] = (rt[:, i] * np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[3, 0] = 0
        dxtdrt[3, 1] = (np.cos(ht[:, i]))
        dxtdrt[3, 2] = (-rt[:, i] * np.sin(ht[:, i]))
        dxtdrt[3, 3] = 0

        temp=xt[:, i]**2.0+yt[:, i]**2.0
        temp2=np.sqrt(temp)*rt[:, i]
        dxtdrt_inv[0, 0] = 1
        dxtdrt_inv[0, 1] = 0
        dxtdrt_inv[0, 2] = 0
        dxtdrt_inv[0, 3] = 0
        dxtdrt_inv[1, 0] = 0
        dxtdrt_inv[1, 1] = xt[:, i]/rt[:, i]
        dxtdrt_inv[1, 2] = yt[:, i]/rt[:, i]
        dxtdrt_inv[1, 3] = zt[:, i]/rt[:, i]
        dxtdrt_inv[2, 0] = 0
        dxtdrt_inv[2, 1] = xt[:, i]*zt[:, i]/temp2
        dxtdrt_inv[2, 2] = yt[:, i]*zt[:, i]/temp2
        dxtdrt_inv[2, 3] = (-xt[:, i]**2-yt[:, i]**2)/temp2
        dxtdrt_inv[3, 0] = 0
        dxtdrt_inv[3, 1] = -yt[:, i]/temp
        dxtdrt_inv[3, 2] = xt[:, i]/temp
        dxtdrt_inv[3, 3] = 0

        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    dxtdr[i1, j1] = dxtdr[i1, j1] + dxtdx[i1, k] * dxdr[k, j1, :, i]

        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    drtdr[i1, j1, :, i] = drtdr[i1, j1, :, i] + dxtdrt_inv[i1, k] * dxtdr[k, j1]

        for j in range(0, bs2new):
            for z in range(0, bs3new):
                drtdr_inv[:, :, 0, i, j, z] = np.linalg.inv(drtdr[:, :, 0, i, j, z])

def calc_normal():
    global Normal_u, dxdr, dxdr_inv, dxdxp, dxdxp_inv, gcov_kerr, Tuu, L, Su
    Normal_u = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    xc = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float64, order='C')

    xc[0] = -1
    xc[1] = r * np.sin(h) * np.cos(ph)
    xc[2] = r * np.sin(h) * np.sin(ph)
    xc[3] = r * np.cos(h)

    Tcalcuu()
    Tuu_kerr = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')

    # Transform to kerr-schild
    for i1 in range(0, 4):
        for j1 in range(0, 4):
            Tuu_kerr[i1, j1] = 0.0
            for k in range(0, 4):
                for l in range(0, 4):
                    Tuu_kerr[i1, j1] = Tuu_kerr[i1, j1] + Tuu[k, l] * dxdxp[i1, k] * dxdxp[j1, l]

    # Transorm to cartesian kerr schild
    for i1 in range(0, 4):
        for j1 in range(0, 4):
            Tuu[i1, j1] = 0.0
            for k in range(0, 4):
                for l in range(0, 4):
                    Tuu[i1, j1] = Tuu[i1, j1] + Tuu_kerr[k, l] * dxdr[i1, k] * dxdr[j1, l]

    Normal_u[3] = ((xc[1] * Tuu[2, 0] - xc[2] * Tuu[1, 0]))
    Normal_u[2] = -((xc[1] * Tuu[3, 0] - xc[3] * Tuu[1, 0]))
    Normal_u[1] = ((xc[2] * Tuu[3, 0] - xc[3] * Tuu[2, 0]))

    # Normalize vector
    Normal_u[:, 0] = mdot(dxdr_inv[:, :, 0], Normal_u[:, 0])  # Transform to kerr-schild coordinates

def calc_transformations_new(temp_tilt, temp_prec):
    import pp_c
    '''
    Description:

    This is a modified version of 'calc_transformations', except here the matrix inversions
    are done analytically in C instead of numerically with np.linalg.inv.

    Here, Jacobians drtdr, dxdr, their inverses drtdr_inv and dxdxdr_inv, and dxdxp_inv are
    constructed given radial profiles of tilt and precession, such that the angular momentum
    unit vector is always parallel with the z' axis, and the x'-y' plane tracks the precession
    angle.

    '''
    global dxdxp, dxdxp_inv, dxdr, dxdr_inv, drtdr, drtdr_inv, r, h, ph, bs1new, bs2new, bs3new
    drtdr = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    drtdr_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdr = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdr_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    dxdxp_inv = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')

    # Set tilt and precession angle in larger array
    tilt = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    tilt[0, :, 0, 0] = temp_tilt / 180.0 * np.pi
    prec[0, :, 0, 0] = temp_prec / 360.0 * 2.0 * np.pi

    # Transformation matrix from kerr-schild to modified kerr-schild
    dxdxp_inv = pp_c.pointwise_invert_4x4(dxdxp, 1, bs1new, bs2new, 1)

    # Transformation matrix to Cartesian Kerr Schild from Spherical Kerr Schild
    dxdr[0, 0] = 1
    dxdr[0, 1] = 0
    dxdr[0, 2] = 0
    dxdr[0, 3] = 0
    dxdr[1, 0] = 0
    dxdr[1, 1] = (np.sin(h) * np.cos(ph))
    dxdr[1, 2] = (r * np.cos(h) * np.cos(ph))
    dxdr[1, 3] = (-r * np.sin(h) * np.sin(ph))
    dxdr[2, 0] = 0
    dxdr[2, 1] = (np.sin(h) * np.sin(ph))
    dxdr[2, 2] = (r * np.cos(h) * np.sin(ph))
    dxdr[2, 3] = (r * np.sin(h) * np.cos(ph))
    dxdr[3, 0] = 0
    dxdr[3, 1] = (np.cos(h))
    dxdr[3, 2] = (-r * np.sin(h))
    dxdr[3, 3] = 0

    # Set coordinates
    x0 = (r * np.sin(h) * np.cos(ph))
    y0 = (r * np.sin(h) * np.sin(ph))
    z0 = (r * np.cos(h))

    xt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.cos(tilt) - z0 * np.sin(tilt))
    yt = (y0 * np.cos(prec) + x0 * np.sin(prec))
    zt = ((x0 * np.cos(prec) - y0 * np.sin(prec)) * np.sin(tilt) + z0 * np.cos(tilt))

    rt = np.sqrt(xt * xt + yt * yt + zt * zt)
    ht = np.arccos(zt / (rt))
    pht = np.arctan2(yt, xt)

    # Transformation matrix to Spherical Kerr Schild from Cartesian Kerr Schild
    temp = x0 ** 2.0 + y0 ** 2.0
    temp2 = np.sqrt(temp) * r ** 2.0
    dxdr_inv[0, 0] = 1
    dxdr_inv[0, 1] = 0
    dxdr_inv[0, 2] = 0
    dxdr_inv[0, 3] = 0
    dxdr_inv[1, 0] = 0
    dxdr_inv[1, 1] = x0 / r
    dxdr_inv[1, 2] = y0 / r
    dxdr_inv[1, 3] = z0 / r
    dxdr_inv[2, 0] = 0
    dxdr_inv[2, 1] = x0 * z0 / temp2
    dxdr_inv[2, 2] = y0 * z0 / temp2
    dxdr_inv[2, 3] = (-x0 ** 2 - y0 ** 2) / temp2
    dxdr_inv[3, 0] = 0
    dxdr_inv[3, 1] = -y0 / temp
    dxdr_inv[3, 2] = x0 / temp
    dxdr_inv[3, 3] = 0
    # dxdr_inv = pp_c.pointwise_invert_4x4(dxdr,1,bs1new,bs2new,bs3new)

    for i in range(0, bs1new):
        # Alloccate temporary arrays
        dxtdx = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')
        dxtdr = np.zeros((4, 4, nb, bs2new, bs3new), dtype=np.float32, order='C')
        # NK: Added extra dim of 1 between nb and bs2new. Doesn't change anything, was just convenient
        # for inverting the matrix the same way as the other ndim=6 arrays.
        dxtdrt = np.zeros((4, 4, nb, 1, bs2new, bs3new), dtype=np.float32, order='C')
        dxtdrt_inv = np.zeros((4, 4, nb, 1, bs2new, bs3new), dtype=np.float32, order='C')

        # Transformation matrix to to tilted Cartesian Kerr Schild from Cartesian Kerr Schild
        dxtdx[0, 0] = 1
        dxtdx[0, 1] = 0
        dxtdx[0, 2] = 0
        dxtdx[0, 3] = 0
        dxtdx[1, 0] = 0
        dxtdx[1, 1] = (np.cos(tilt[:, i]) * np.cos(prec[:, i]))
        dxtdx[1, 2] = (-np.cos(tilt[:, i]) * np.sin(prec[:, i]))
        dxtdx[1, 3] = (-np.sin(tilt[:, i]))
        dxtdx[2, 0] = 0
        dxtdx[2, 1] = (np.sin(prec[:, i]))
        dxtdx[2, 2] = (np.cos(prec[:, i]))
        dxtdx[2, 3] = 0
        dxtdx[3, 0] = 0
        dxtdx[3, 1] = (np.sin(tilt[:, i]) * np.cos(prec[:, i]))
        dxtdx[3, 2] = (-np.sin(tilt[:, i]) * np.sin(prec[:, i]))
        dxtdx[3, 3] = (np.cos(tilt[:, i]))

        # Calculate transformation matrix from tilted Cartesian to tilted Kerr-Schild
        dxtdrt[0, 0, 0] = 1
        dxtdrt[0, 1, 0] = 0
        dxtdrt[0, 2, 0] = 0
        dxtdrt[0, 3, 0] = 0
        dxtdrt[1, 0, 0] = 0
        dxtdrt[1, 1, 0] = (np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 2, 0] = (rt[:, i] * np.cos(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[1, 3, 0] = (-rt[:, i] * np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 0, 0] = 0
        dxtdrt[2, 1, 0] = (np.sin(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 2, 0] = (rt[:, i] * np.cos(ht[:, i]) * np.sin(pht[:, i]))
        dxtdrt[2, 3, 0] = (rt[:, i] * np.sin(ht[:, i]) * np.cos(pht[:, i]))
        dxtdrt[3, 0, 0] = 0
        dxtdrt[3, 1, 0] = (np.cos(ht[:, i]))
        dxtdrt[3, 2, 0] = (-rt[:, i] * np.sin(ht[:, i]))
        dxtdrt[3, 3, 0] = 0

        temp = xt[:, i] ** 2.0 + yt[:, i] ** 2.0
        temp2 = np.sqrt(temp) * rt[:, i] ** 2.0
        dxtdrt_inv[0, 0] = 1
        dxtdrt_inv[0, 1] = 0
        dxtdrt_inv[0, 2] = 0
        dxtdrt_inv[0, 3] = 0
        dxtdrt_inv[1, 0] = 0
        dxtdrt_inv[1, 1] = xt[:, i] / rt[:, i]
        dxtdrt_inv[1, 2] = yt[:, i] / rt[:, i]
        dxtdrt_inv[1, 3] = zt[:, i] / rt[:, i]
        dxtdrt_inv[2, 0] = 0
        dxtdrt_inv[2, 1] = xt[:, i] * zt[:, i] / temp2
        dxtdrt_inv[2, 2] = yt[:, i] * zt[:, i] / temp2
        dxtdrt_inv[2, 3] = (-xt[:, i] ** 2 - yt[:, i] ** 2) / temp2
        dxtdrt_inv[3, 0] = 0
        dxtdrt_inv[3, 1] = -yt[:, i] / temp
        dxtdrt_inv[3, 2] = xt[:, i] / temp
        dxtdrt_inv[3, 3] = 0

        # dxtdrt_inv = pp_c.pointwise_invert_4x4(dxtdrt,1,1,bs2new,bs3new)

        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    dxtdr[i1, j1] = dxtdr[i1, j1] + dxtdx[i1, k] * dxdr[k, j1, :, i]

        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    drtdr[i1, j1, :, i] = drtdr[i1, j1, :, i] + dxtdrt_inv[i1, k, 0] * dxtdr[k, j1]

    drtdr_inv = pp_c.pointwise_invert_4x4(drtdr, 1, bs1new, bs2new, bs3new)

# Make file for raytracing
def dump_RT_RAZIEH(dir, dump, temp_tilt, temp_prec, advanced=1):
    global gcov, gcov_kerr, gcon, uu, bu, r, rho, ug, N1, bs1new, bs2new, bs3new, ug, rho, target_thickness, _dx1, _dx2, _dx3, Mdot
    global Normal_u, Rdot, Mdot, dxdr, dxdr_inv, dxdxp, dxdxp_inv, dxdxt, dxdxt_inv, a
    global drtdr, drtdr_inv, startx1, startx2, startx3, x1, x2, x3, r, h, ph, export_raytracing_RAZIEH, ph_temp, ph_proj
    global rho_proj, ug_proj, Normal_proj, vkerr_proj, uukerr_proj, h_proj, ph_proj, vkerr, gdet_t, gcov_t, gcov22_proj, gcov_kerr, j0, gdet_proj, gcov_proj, h_temp, source_temp, source_proj
    import pp_c
    # Find index for r=100
    for i in range(0, bs1new):
        while (r[0, i, int(bs2new / 2), 0] < 100 or i == int(bs1new) - 1):
            N1 = i
            break

    if (rank == 0):
        print("BS3NEW:", bs3new, "N1:", N1)

    # Calculate mass accretion rate
    calc_Mdot()

    # For extra safety at boundaries of grid where interpolation does not work, set coordinates manually assuming a uniform grid in log(r), theta and phi
    set_KS()

    # Calculate transformation matrices
    calc_transformations_new(temp_tilt, temp_prec)

    # Allocate new arrays for output
    beta = np.zeros((4, nb, bs1new, bs2new, 1), dtype=np.float32)
    vkerr = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)

    # Transform velocities to Kerr-Schild relative 4-velocity NOT 3 velocity
    alpha = 1. / (-gcon[0, 0, 0]) ** 0.5
    beta[1:4, 0] = gcon[0, 1:4, 0] * alpha * alpha
    vkerr[1:4, 0] = (beta[1:4, 0] * uu[0] + uu[1:4, 0])
    vkerr[:, 0] = mdot(dxdxp[:, :, 0], vkerr[:, 0])

    if (advanced == 1):
        # Transform metric to tilted Kerr-Schild coordinates
        gcov_t = np.zeros((nb, bs1new, bs2new, bs3new, 4, 4), dtype=np.float32)
        for i1 in range(0, 4):
            for j1 in range(0, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        gcov_t[:, :, :, :, i1, j1] = gcov_t[:, :, :, :, i1, j1] + gcov_kerr[k, l] * drtdr_inv[k, i1] * drtdr_inv[l, j1]

        # Calculate determinant tilted metric
        gdet_t = np.sqrt(-np.linalg.det(gcov_t))

    # Calculate normal vector to disk
    calc_normal()

    # Transform vectors to tilted coordinates
    if (advanced == 1):
        vkerr[:, 0] = mdot(drtdr[:, :, 0], vkerr[:, 0])
        Normal_u[:, 0] = mdot(drtdr[:, :, 0], Normal_u[:, 0])

    # Project stuff to tilted frame
    preset_transform_scalar(temp_tilt, temp_prec)
    rho_proj = transform_scalar(rho)
    ug_proj = transform_scalar(ug)
    source_proj = transform_scalar(Rdot)
    vkerr_proj = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    vkerr_proj[1] = transform_scalar(vkerr[1])
    vkerr_proj[2] = transform_scalar(vkerr[2])
    vkerr_proj[3] = transform_scalar(vkerr[3])
    Normal_proj = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    Normal_proj[1] = transform_scalar(Normal_u[1])
    Normal_proj[2] = transform_scalar(Normal_u[2])
    Normal_proj[3] = transform_scalar(Normal_u[3])
    gcov22_proj = transform_scalar(gcov_t[:, :, :, :, 2, 2])
    gdet_proj = transform_scalar(gdet_t)

    # Calculate normalization and set density filter
    filter = (h < (np.pi / 2.0 + 1.0)) * (h > (np.pi / 2.0 - 1.0))
    norm = ((rho_proj) * gdet_proj * filter).sum(2)

    # Average stuff in tilted frame
    rho_temp = (rho_proj * filter * _dx2 * (np.pi / 2.0) * np.sqrt(gcov22_proj)).sum(2)
    ug_temp = (ug_proj * filter * _dx2 * (np.pi / 2.0) * np.sqrt(gcov22_proj)).sum(2)
    source_temp = (source_proj * filter * 0.5 * _dx2 * (np.pi / 2.0) * np.sqrt(gcov22_proj)).sum(2)
    vkerr_temp = np.zeros((4, nb, bs1new, bs3new), dtype=np.float32)
    vkerr_temp = (rho_proj * filter * gdet_proj * vkerr_proj).sum(3) / norm
    Normal_temp = np.zeros((4, nb, bs1new, bs3new), dtype=np.float32)
    Normal_temp = (rho_proj * filter * gdet_proj * Normal_proj).sum(3) / norm

    # Set tilt and precession angle in larger array
    tilt = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    tilt[0, :, 0, 0] = -temp_tilt / 180.0 * np.pi
    prec[0, :, 0, 0] = -temp_prec / 360.0 * 2.0 * np.pi

    # Set projected coordinates
    xt = (r * np.sin(h) * np.cos(ph))
    yt = (r * np.sin(h) * np.sin(ph))
    zt = (r * np.cos(h))

    x = ((xt * np.cos(prec) - yt * np.sin(prec)) * np.cos(tilt) - zt * np.sin(tilt))
    y = (yt * np.cos(prec) + xt * np.sin(prec))
    z = ((xt * np.cos(prec) - yt * np.sin(prec)) * np.sin(tilt) + zt * np.cos(tilt))

    r_proj = np.sqrt(x * x + y * y + z * z)
    h_proj = np.arccos(z / r_proj)
    ph_proj = (np.arctan2(y, x) - prec) % (2.0 * np.pi)

    # Calculate index of midplane of disk in tilted frame
    j0 = ((rho_proj * filter * gdet_proj * x2).sum(2) / norm - (startx2 + 0.5 * _dx2)) / (_dx2)
    j0[:, :, :] = (0.0 - (startx2 + 0.5 * _dx2)) / (_dx2)

    if (advanced == 1):
        # Transform vectors to untilted coordinates
        drtdr_inv_proj = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
        for i1 in range(0, 4):
            for j1 in range(0, 4):
                drtdr_inv_proj[i1, j1] = transform_scalar(drtdr_inv[i1, j1])
        drtdr_inv_proj2 = np.zeros((4, 4, nb, bs1new, bs3new), dtype=np.float32)
        for i in range(0, bs1new):
            for z in range(0, bs3new):
                weight = 1.0 - (j0[0, i, z] - np.int32(j0[0, i, z]))
                drtdr_inv_proj2[:, :, :, i, z] = drtdr_inv_proj[:, :, :, i, np.int32(j0[0, i, z]), z] * weight + drtdr_inv_proj[:, :, :, i, (np.int32(j0[0, i, z]) + 1) % bs2new, z] * (1.0 - weight)
        vkerr_temp[:, 0] = mdot2(drtdr_inv_proj2[:, :, 0], vkerr_temp[:, 0])
        Normal_temp[:, 0] = mdot2(drtdr_inv_proj2[:, :, 0], Normal_temp[:, 0])

    # Calculate
    h_temp = np.zeros((nb, bs1new, bs3new), dtype=np.float32)
    ph_temp = np.zeros((nb, bs1new, bs3new), dtype=np.float32)
    for i in range(0, bs1new):
        for z in range(0, bs3new):
            weight = 1.0 - (j0[0, i, z] - np.int32(j0[0, i, z]))
            h_temp[0, i, z] = h_proj[0, i, np.int32(j0[0, i, z]), z] * weight + h_proj[0, i, (np.int32(j0[0, i, z]) + 1) % bs2new, z] * (1.0 - weight)
            ph_temp0 = ph_proj[0, i, np.int32(j0[0, i, z]), z]
            ph_temp1 = ph_proj[0, i, (np.int32(j0[0, i, z]) + 1) % bs2new, z]
            if (np.abs(ph_temp1 - ph_temp0) > np.pi):
                if (ph_temp0 > np.pi):
                    ph_temp1 = ph_temp1 + 2.0 * np.pi
                else:
                    ph_temp0 = ph_temp0 + 2.0 * np.pi
            ph_temp[0, i, z] = (ph_temp0 * weight + ph_temp1 * (1.0 - weight)) % (2.0 * np.pi)

    # Linear interpolation to non tilted frame
    z0 = 0
    for i in range(0, N1):
        for z in range(0, bs3new):
            while (1):
                if ((ph_temp[0, i, (z0) % bs3new]) < 0):
                    ph_temp[0, i, (z0) % bs3new] = ph_temp[0, i, (z0) % bs3new] + 2.0 * np.pi
                if ((ph_temp[0, i, (z0 + 1) % bs3new]) < 0):
                    ph_temp[0, i, (z0 + 1) % bs3new] = ph_temp[0, i, (z0 + 1) % bs3new] + 2.0 * np.pi
                if (ph_temp[0, i, (z0) % bs3new] > ph_temp[0, i, (z0 + 1) % bs3new]):
                    if (ph[0, i, 0, z] < np.pi):
                        ph_temp0 = ph_temp[0, i, (z0) % bs3new] - 2.0 * np.pi
                        ph_temp1 = ph_temp[0, i, (z0 + 1) % bs3new]
                    else:
                        ph_temp0 = ph_temp[0, i, (z0) % bs3new]
                        ph_temp1 = ph_temp[0, i, (z0 + 1) % bs3new] + 2.0 * np.pi
                else:
                    ph_temp0 = ph_temp[0, i, (z0) % bs3new]
                    ph_temp1 = ph_temp[0, i, (z0 + 1) % bs3new]

                if (ph_temp0 <= ph[0, i, 0, z] and ph_temp1 >= ph[0, i, 0, z]):
                    weight = 1.0 - (ph[0, i, 0, z] - ph_temp0) / (ph_temp1 - ph_temp0)
                    rho_proj[0, i, 0, z] = rho_temp[0, i, (z0) % bs3new] * weight + rho_temp[0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    ug_proj[0, i, 0, z] = ug_temp[0, i, (z0) % bs3new] * weight + ug_temp[0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    source_proj[0, i, 0, z] = source_temp[0, i, (z0) % bs3new] * weight + source_temp[0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    h_proj[0, i, 0, z] = h_temp[0, i, (z0) % bs3new] * weight + h_temp[0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    ph_proj[0, i, 0, z] = ph_temp0 * weight + ph_temp1 * (1.0 - weight)
                    vkerr_proj[:, 0, i, 0, z] = vkerr_temp[:, 0, i, (z0) % bs3new] * weight + vkerr_temp[:, 0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    Normal_proj[:, 0, i, 0, z] = Normal_temp[:, 0, i, (z0) % bs3new] * weight + Normal_temp[:, 0, i, (z0 + 1) % bs3new] * (1.0 - weight)
                    break
                else:
                    z0 = z0 + 1

    # Calculate kerr-metric for processed data coordinatesb n
    cth = np.cos(h_proj[0, 0:N1, 0:1, :])
    sth = np.sin(h_proj[0, 0:N1, 0:1, :])
    s2 = sth * sth
    radius = r[0:1, 0:N1, 0:1, :]
    rho2 = radius * radius + a * a * cth * cth
    gcov_kerr = np.zeros((4, 4, 1, N1, 1, bs3new), dtype=np.float32)
    gcov_kerr[0, 0] = (-1. + 2. * radius / rho2)
    gcov_kerr[0, 1] = (2. * radius / rho2)
    gcov_kerr[0, 3] = (-2. * a * radius * s2 / rho2)
    gcov_kerr[1, 0] = gcov_kerr[0, 1]
    gcov_kerr[1, 1] = (1. + 2. * radius / rho2)
    gcov_kerr[1, 3] = (-a * s2 * (1. + 2. * radius / rho2))
    gcov_kerr[2, 2] = rho2
    gcov_kerr[3, 0] = gcov_kerr[0, 3]
    gcov_kerr[3, 1] = gcov_kerr[1, 3]
    gcov_kerr[3, 3] = (s2 * (rho2 + a * a * s2 * (1. + 2. * radius / rho2)))

    # Invert coviariant metric to get contravariant Kerr Schild metric
    gcon_kerr = np.zeros((4, 4, 1, N1, 1, bs3new), dtype=np.float32)
    gcon_kerr = pp_c.pointwise_invert_4x4(gcov_kerr, 1, N1, 1, bs3new)

    # Convert velocity back to 4-velocity
    alpha = 1. / np.sqrt(-gcon_kerr[0, 0])
    beta = np.zeros((4, 1, N1, 1, bs3new), dtype=np.float64)
    beta[1:4] = gcon_kerr[0, 1:4] * alpha * alpha
    qsq = gcov_kerr[1, 1] * vkerr_proj[1, :, 0:N1, 0:1, :] * vkerr_proj[1, :, 0:N1, 0:1, :] + gcov_kerr[2, 2] * vkerr_proj[2, :, 0:N1, 0:1, :] * vkerr_proj[2, :, 0:N1, 0:1, :] + gcov_kerr[3, 3] * vkerr_proj[3, :, 0:N1, 0:1, :] * vkerr_proj[3, :, 0:N1, 0:1, :] + \
          2. * (gcov_kerr[1, 2] * vkerr_proj[1, :, 0:N1, 0:1, :] * vkerr_proj[2, :, 0:N1, 0:1, :] + gcov_kerr[1, 3] * vkerr_proj[1, :, 0:N1, 0:1, :] * vkerr_proj[3, :, 0:N1, 0:1, :] + gcov_kerr[2, 3] * vkerr_proj[2, :, 0:N1, 0:1, :] * vkerr_proj[3, :, 0:N1, 0:1, :])
    gamma = np.sqrt(1. + qsq)
    uukerr_proj = np.zeros((4, 1, N1, 1, bs3new), dtype=np.float32)
    uukerr_proj[0] = (gamma / alpha)
    uukerr_proj[1:4] = vkerr_proj[1:4, :, 0:N1, 0:1, :] - gamma * beta[1:4] / alpha

    # Start writing binary data
    import struct
    f = open(dir + "/RT/rt%d" % dump, "wb+")

    # Write header (2 integers)
    data = [int(N1), (bs3new)]
    s = struct.pack('i' * 2, *data)
    f.write(s)

    # Write data (15xfloat32)
    for i in range(0, N1):
        for z in range(0, bs3new):
            data = [t, Mdot[0, 5], temp_tilt[i], temp_prec[i], r[0, i, 0, z], ph_proj[0, i, 0, z], h_proj[0, i, 0, z],
                    rho_proj[0, i, 0, z], ug_proj[0, i, 0, z], uukerr_proj[0, 0, i, 0, z], uukerr_proj[1, 0, i, 0, z], uukerr_proj[2, 0, i, 0, z],
                    uukerr_proj[3, 0, i, 0, z], Normal_proj[1, 0, i, 0, z], Normal_proj[2, 0, i, 0, z], Normal_proj[3, 0, i, 0, z], source_proj[0, i, 0, z]]
            s = struct.pack('f' * 17, *data)
            f.write(s)
    f.close()

def merge_dump(dir):
    global n_ord, n_active_total
    os.chdir(dir)  # hamr
    destination = open('new_dump', 'wb')
    for i in glob.glob("dumpdiag*"):
        os.remove(i)
    length = len(os.listdir(dir))
    print("Length", length, "n_total", n_active_total, "n_ord[5]", n_ord[5], "dir", dir)

    for i in range(0, n_active_total):
        shutil.copyfileobj(open('dump%d' % n_ord[i], 'rb'), destination)
    destination.close()

def merge_dumps(dir):
    dumps = 0
    os.chdir(dir)  # hamr
    rblock_new()
    while (os.path.isfile(dir + "/dumps%d/parameters" % dumps)):
        dumps = dumps + 1
    if (rank == 0):
        print("nr_files", dumps)

    for i in range(0, dumps):
        if (i % numtasks == rank):
            os.chdir(dir)  # hamr
            rpar_new(i)
            merge_dump(dir + "/dumps%d" % i)


def backup_dump(dir1, dir2, dir3):
    global n_ord, n_active_total

    os.makedirs(dir2)
    os.chdir(dir2)  # hamr
    destination2 = open('parameters', 'wb')

    os.makedirs(dir3)
    os.chdir(dir3)  # hamr
    destination1 = open('new_dump', 'wb')
    destination3 = open('parameters', 'wb')

    os.chdir(dir1)  # hamr
    length = len(os.listdir(dir1))

    for i in range(0, n_active_total):
        os.chdir(dir2)  # hamr
        destination = open('dump%d' % n_ord[i], 'wb')
        os.chdir(dir1)  # hamr
        shutil.copyfileobj(open('dump%d' % n_ord[i], 'rb'), destination)
        destination.close()
    shutil.copyfileobj(open('new_dump', 'rb'), destination1)
    shutil.copyfileobj(open('parameters', 'rb'), destination2)
    shutil.copyfileobj(open('parameters', 'rb'), destination3)
    destination1.close()
    destination2.close()
    destination3.close()
    print("Length", length, "n_total", n_active_total, "n_ord[5]", n_ord[5], "dir", dir1)

def backup_dumps(dir1, dir2, dir3):
    dumps = 0
    os.chdir(dir1)  # hamr
    rblock_new()
    while (os.path.isfile(dir1 + "/dumps%d/parameters" % dumps)):
        dumps = dumps + 1
    if (rank == 0):
        print("nr_files", dumps)

    for i in range(0, dumps, 10):
        if ((i / 10) % numtasks == rank):
            os.chdir(dir1)  # hamr
            rpar_new(i)
            backup_dump(dir1 + "/dumps%d" % i, dir2 + "/dumps%d" % i, dir3 + "/dumps%d" % i)


import glob

def delete_dump(dir, start, end, stride):
    dumps = 0
    os.chdir(dir)  # hamr
    rblock_new()
    while (os.path.isfile(dir + "/dumps%d/parameters" % dumps)):
        dumps = dumps + 1
    if (rank == 0):
        print("nr_files", dumps)

    for i in range(start, end, stride):
        if (i % numtasks == rank):
            os.chdir(dir)  # hamr
            rpar_new(i)
            dir2 = dir + "/dumps%d" % i
            os.chdir(dir2)
            for j in glob.glob("dump*"):
                os.remove(j)
            os.chdir(dir)

def plc_cart(var, min, max, rmax, offset, name, label):
    global aphi, r, h, ph, print_fieldlines,notebook, do_box
    fig = plt.figure(figsize=(64, 32))

    X = r*np.sin(h)
    Y = r*np.cos(h)
    if(nb==1 and do_box==0):
        X[:,:,0]=0.0*X[:,:,0]
        X[:,:,bs2new-1]=0.0*X[:,:,bs2new-1]

    plotmax = int(10*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2)*plotmax:
            ilim = i
            break

    plt.subplot(1, 2, 1)
    plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
    res = plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    plt.ylabel(r"$z / R_g$", fontsize=90)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    #cb.ax.tick_params(labelsize=50)

    plt.subplot(1, 2, 2)
    plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 5, ymax=rmax * 5)
    res = plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1.0 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * 5, ymax=rmax * 5)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 5, ymax=rmax * 5)
        plc_new(aphi[:, 0:ilim], levels=np.arange(aphi[:, 0:ilim].min(), aphi[:, 0:ilim].max(), (aphi[:, 0:ilim].max()-aphi[:, 0:ilim].min())/20.0), cb=0,colors="black", isfilled=0, xcoord=-1.0 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * 5, ymax=rmax * 5)

    plt.xlabel(r"$x / R_g$", fontsize=90)
    #plt.ylabel(r"$z / R_g$", fontsize=60)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    #cb.ax.tick_params(labelsize=50)
    plt.savefig(name, dpi=100)
    if (notebook==0):
        plt.close('all')

def plc_new(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):  # plc
    global r, h, ph
    l = [None] * nb2d

    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 0)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    z = kwargs.pop('z', 0)

    if ax is None:
        ax = plt.gca()
    if isfilled:
        for i in range(0, nb):
            index_z_block=int((z-int((z/360))*360.0)/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contourf(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, extend='both',**kwargs)
    else:
        for i in range(0, nb):
            index_z_block=int(z/360.0*bs3new*nb3*(1+REF_3)**(block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block/bs3new)):
                offset=index_z_block-block[n_ord[i], AMR_COORD3]*bs3new
                res = ax.contour(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, linewidths=4, extend='both', **kwargs)
    if (cb == True):  # use color bar
        plt.colorbar(res, ax=ax)
    if xy:
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
    return res

def plc_cart_xy1(var, min, max, rmax, offset, transform, name, label):
    fig = plt.figure(figsize=(64, 32))

    X = np.multiply(r, np.sin(ph))
    Y = np.multiply(r, np.cos(ph))
    if(transform==1):
        var2 = transform_scalar(var)
        var2 = project_vertical(var2)
    else:
        var2=var
    plotmax = int(10*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2.0)*plotmax:
            ilim = i
            break

    plt.subplot(1, 2, 1)
    res = plc_new_xy(np.log10(var2)[:, 0:ilim], levels=np.arange(min, max, (max-min)/100.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    plt.ylabel(r"$y / R_g$", fontsize=90)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)

    plt.subplot(1, 2, 2)
    res = plc_new_xy(np.log10(var2)[:, 0:ilim], levels=np.arange(min, max, (max-min)/100.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 5, ymax=rmax * 5)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    #plt.ylabel(r"$y / R_g$", fontsize=60)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    plt.savefig(name, dpi=30)
    if (notebook == 0):
        plt.close('all')

def plc_new_xy(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):  # plc
    global r, h, ph, bs2new, notebook
    l = [None] * nb2d
    # xcoord = kwargs.pop('x1', None)
    # ycoord = kwargs.pop('x2', None)
    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 1)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    z = kwargs.pop('z', 0)
    if ax is None:
        ax = plt.gca()
    if (nb > 1):
        if isfilled:
            for i in range(0, nb):
                if block[n_ord[i], AMR_COORD2] == (nb2 * np.power(1 + REF_2, block[n_ord[i], AMR_LEVEL2])//2):
                    res = ax.contourf(xcoord[i, :, 0, :], ycoord[i, :, 0, :], myvar[i, :, 0, :], nc,extend='both', **kwargs)
        else:
            for i in range(0, nb):
                if block[n_ord[i], AMR_COORD2] == (nb2 * np.power(1 + REF_2, block[n_ord[i], AMR_LEVEL2])//2):
                    res = ax.contour(xcoord[i, :, 0, :], ycoord[i, :, 0, :], myvar[i, :, 0, :], nc,extend='both', **kwargs)
    else:
        if isfilled:
            res = ax.contourf(xcoord[0, :, int(bs2new // 2), :], ycoord[0, :, int(bs2new // 2), :],myvar[0, :, int(bs2new // 2), :], nc, extend='both', **kwargs)
        else:
            res = ax.contour(xcoord[0, :, int(bs2new // 2), :], ycoord[0, :, int(bs2new // 2), :], myvar[0, :, int(bs2new // 2), :],nc, extend='both', **kwargs)
    if (cb == True):  # use color bar
        plt.colorbar(res, ax=ax)
    if (xy == 1):
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
    return res

def plc_cart_grid(rmax=100, offset=0):
    global tj2, ti2, h2, r2, bs1new,bs2new,bs3new,notebook
    fig = plt.figure(figsize=(32, 32))
    h2 = np.zeros((nb, bs1new, bs2new + 2, bs3new), dtype=mytype, order='C')

    h2[0, :, 0, :] = -h[0, :, 0, :]
    h2[0, :, bs2new + 1, :] = 2 * np.pi - h[0, :, bs2new - 1, :]
    h2[0, :, 1:bs2new + 1, :] = h[0]

    r2 = np.zeros((nb, bs1new, bs2new + 2, bs3new), dtype=mytype, order='C')
    r2[0, :, 1:bs2new + 1, :] = r
    r2[0, :, 0, :] = r2[0, :, 1, :]
    r2[0, :, bs2new + 1, :] = r2[0, :, bs2new, :]

    ti2 = np.zeros((nb, bs1new, bs2new + 2, bs3new), dtype=mytype, order='C')
    ti2[0, :, 1:bs2new + 1, :] = ti
    ti2[0, :, 0, :] = ti2[0, :, 1, :]
    ti2[0, :, bs2new + 1, :] = ti2[0, :, bs2new, :]

    tj2 = np.zeros((nb, bs1new, bs2new + 2, bs3new), dtype=mytype, order='C')
    tj2[0, :, 1:bs2new + 1, :] = tj + 0.5
    tj2[0, :, 0, :] = -0.5
    tj2[0, :, bs2new + 1, :] = bs2new + 0.5

    X = np.multiply(r2, np.sin(h2))
    Y = np.multiply(r2, np.cos(h2))

    plotmax = int(rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > plotmax:
            ilim = i
            break

    plt.figure(figsize=(24, 24))
    plc_new((tj2)[0:ilim], levels=np.arange(0.0, 146.0, 4), cb=0, isfilled=0, xcoord=X[0:ilim], ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax, colors="black")
    res = plc_new((tj2)[0:ilim], levels=np.arange(0.0, 146.0, 4), cb=0, isfilled=0, xcoord=-1 * X[0:ilim],ycoord=Y[0:ilim], xy=1, z=int(len(r[0, 0, 0, :]) * .5) + offset, xmax=rmax, ymax=rmax, colors="black")
    plc_new((ti2 + 0.5)[0:ilim], levels=np.arange(-0.0, 144.0, 4), cb=0, isfilled=0, xcoord=X[0:ilim], ycoord=Y[0:ilim],xy=1, z=offset, xmax=rmax, ymax=rmax, colors="black")
    res = plc_new((ti2 + 0.5)[0:ilim], levels=np.arange(-0.0, 144.0, 4), cb=0, isfilled=0, xcoord=-1 * X[0:ilim],ycoord=Y[0:ilim], xy=1, z=int(len(r[0, 0, 0, :]) * .5) + offset, xmax=rmax, ymax=rmax, colors="black")
    plt.xlabel(r"$x / R_g$", fontsize=48)
    plt.ylabel(r"$y / R_g$", fontsize=48)
    plt.title(r"Grid structure$" % t, fontsize=60)
    plt.savefig("grid.png", dpi=300)
    if (notebook == 0):
        plt.close('all')

def plc_new_cart(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):  # plc
    global r, h, ph, nb
    fig = plt.figure(figsize=(32, 32))

    l = [None] * nb2d

    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 0)
    xmin = kwargs.pop('xmin', 10)
    ymin = kwargs.pop('ymin', 5)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    label = kwargs.pop('label', "test")
    name = kwargs.pop('name', "test")

    if ax is None:
        ax = plt.gca()
    if isfilled:
        for i in range(0, nb):
            index_z_block = int(0.5 * bs3new * nb3 * (1 + REF_3) ** (block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block / bs3new)):
                offset = index_z_block - block[n_ord[i], AMR_COORD3] * bs3new
                res = ax.contourf(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, extend='both', **kwargs)
    else:
        for i in range(0, nb):
            index_z_block = int(0.5 * bs3new * nb3 * (1 + REF_3) ** (block[n_ord[i], AMR_LEVEL3]))
            if (block[n_ord[i], AMR_COORD3] == int(index_z_block / bs3new)):
                offset = index_z_block - block[n_ord[i], AMR_COORD3] * bs3new
                res = ax.contour(xcoord[i, :, :, offset], ycoord[i, :, :, offset], myvar[i, :, :, offset], nc, linewidths=4, extend='both', **kwargs)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xlabel(r"$x / R_g$", fontsize=90)
    plt.ylabel(r"$y / R_g$", fontsize=90)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(res, cax=cax)

    plt.savefig(name, dpi=100)

def resample_cartesian(input, xin, xout, dx, yin, yout, dy, zin, zout, dz):
    global x, y, z, Nx, Ny, Nz, startx1, startx2, startx3, _dx1, _dx2, _dx3, h_cart, r_cart, ph_cart, ti, tj, tz

    # Create cartesian grid with inner and outer boundaries and spacing
    Nx = max(np.int32((xout - xin) / dx))
    Ny = max(np.int32((yout - yin) / dy))
    Nz = max(np.int32((zout - zin) / dz))

    x1 = np.zeros((1, Nx, 1, 1), dtype=np.float32)
    y1 = np.zeros((1, 1, Ny, 1), dtype=np.float32)
    z1 = np.zeros((1, 1, 1, Nz), dtype=np.float32)
    x = np.zeros((1, Nx, Ny, Nz), dtype=np.float32)
    y = np.zeros((1, Nx, Ny, Nz), dtype=np.float32)
    z = np.zeros((1, Nx, Ny, Nz), dtype=np.float32)
    x1[0, :, 0, 0] = xin + np.arange(Nx) * dx
    y1[0, 0, :, 0] = yin + np.arange(Ny) * dy
    z1[0, 0, 0, :] = zin + np.arange(Nz) * dz
    x[:, :, :, :] = x1
    y[:, :, :, :] = y1
    z[:, :, :, :] = z1

    # Convert to spherical coordinates
    r_cart = np.sqrt(x ** 2 + y ** 2 + z ** 2)[0]
    h_cart = np.arccos(z / r_cart)[0]
    ph_cart = (np.arctan2(y, x)[0]) % (2.0 * np.pi)

    # Check consistency of grid
    if (rank == -1):
        if (r.min() > r_cart.min()):
            print("Inner r boundary is too small")
        if (h.min() > h_cart.min()):
            print("Inner theta boundary is too small")
        if (ph.min() > ph_cart.min()):
            print("Inner phi boundary is too small")
        if (r.max() < r_cart.max()):
            print("Outer r boundary is too big")
        if (h.max() < h_cart.max()):
            print("Outer h boundary is too big")
        if (ph.max() < ph_cart.max()):
            print("Outer ph boundary is too big")

    ti = np.int32((np.log(r_cart) - (startx1 + 0.5 * _dx1)) / _dx1)
    tj = np.int32(((2.0 / np.pi * (h_cart) - 1.0) - (startx2 + 0.5 * _dx2)) / _dx2)
    tz = np.int32((ph_cart - (startx3 + 0.5 * _dx3)) / _dx3)
    tz[tz < 0] = 0
    tz[tz > bs3new - 1] = bs3new - 1

    output = np.zeros((1, Nx, Ny, Nz), dtype=np.float32)
    output[0] = ndimage.map_coordinates(input[0], [[ti], [tj], [tz]], order=1, mode='constant', cval=0.0)

    return output

def transform_scalar_tot(input, tilt, prec):
    preset_transform_scalar(tilt, prec)
    output=transform_scalar(input)
    return output

def preset_transform_scalar(tilt, prec):
    global ti,tj,tk, _dx2, _dx3
    X = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tilt_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    t1 = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    t2 = np.zeros((nb, 1, bs2new, 1), dtype=np.float32)
    t3 = np.zeros((nb, 1, 1, bs3new), dtype=np.float32)
    ti = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tj = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tk = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)

    t1[0, :, 0, 0] = np.arange(bs1new)
    t2[0, 0, :, 0] = np.arange(bs2new)
    t3[0, 0, 0, :] = np.arange(bs3new)

    ti[:, :, :, :] = t1
    tj[:, :, :, :] = t2
    tk[:, :, :, :] = t3

    tilt_tmp[0, :, 0, 0] = tilt / 180.0 * np.pi
    prec_tmp[0, :, 0, 0] = (prec / 360.0 * 2.0 * np.pi)

    sph_to_cart2(X, h, ph)
    rotate_coord(X, tilt_tmp)
    h_new, ph_new = cart_to_sph(X)
    ph_new = (ph_new + prec_tmp)
    tj = (((h_new[0] - h[0]) / (_dx2)*2.0/np.pi + tj)) % bs2new
    tk = (((ph_new[0] - ph[0]) / (_dx3) + tk)) % bs3new

def transform_scalar(input):
    global ti,tj,tk
    output = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)

    output[0] = ndimage.map_coordinates(input[0], [[ti], [tj], [tk]], order=1, mode='nearest')
    return output

def print_butterfly(f_but, radius,z):
    global bs1new, rho,ug,bu
    cell1 = 0
    cell2=0
    while (r[0, cell1, int(bs2new // 2), z] < 0.9*radius):
        cell1 += 1
    while (r[0, cell2, int(bs2new // 2), z] < 1.1*radius):
        cell2 += 1
    cell=int((cell1+cell2)*0.5)
    bu_proj = project_vector(bu)
    uu_proj = project_vector(uu)
    b_r = (bu_proj[1])[0, cell1:cell2, :, z].sum(0)
    b_theta = (bu_proj[2])[0,  cell1:cell2, :, z].sum(0)
    b_phi = (bu_proj[3])[0,  cell1:cell2, :, z].sum(0)
    u_r = (uu_proj[1])[0,  cell1:cell2, :, z].sum(0)
    u_theta = (uu_proj[2])[0,  cell1:cell2, :, z].sum(0)
    u_phi = (uu_proj[3])[0,  cell1:cell2, :, z].sum(0)
    rho_1 = (rho)[0,  cell1:cell2, :, z].sum(0)
    ug_1 = (ug)[0,  cell1:cell2, :, z].sum(0)
    bsq_1 = (bsq)[0,  cell1:cell2, :, z].sum(0)
    for g in range(0, bs2new):
        f_but.write("%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g \n" % (t, r[0, cell, bs2new // 2, 0], h[0,cell,g,0], rho_1[g], (gam-1)*ug_1[g],bsq_1[g], b_r[g], b_theta[g], b_phi[g], u_r[g], u_theta[g], u_phi[g]))

def dump_visit(dir, dump, radius):
    from numpy import mgrid, empty, sin, pi
    # from tvtk.api import tvtk, write_data
    from tvtk.api import tvtk, write_data;
    from tvtk.tvtk_access import tvtk
    global bu, uu, bsq, rho, bs1new, bs2new, bs3new, axisym

    ilim = 0
    while (r[0, ilim, 0, 0] < radius and (ilim<bs1new-1)):
        ilim += 1
    # visitrho = open(dir+"/visit/allrho%d.visit" % dump, 'w')
    visitdata = open(dir + "/visit/alldata%d.visit" % dump, 'w')
    # visitrho.write("!NBLOCKS %d\n" %nb)
    visitdata.write("!NBLOCKS %d\n" % nb)

    for n in range(0, nb):
        # The actual points.
        pts = empty(rho[n, 0:ilim].shape + (3,), dtype=float)
        pts[..., 0] = np.multiply(np.multiply(r[n, 0:ilim], np.cos(ph[n, 0:ilim])), np.sin(h[n, 0:ilim]))
        pts[..., 1] = np.multiply(np.multiply(r[n, 0:ilim], np.sin(ph[n, 0:ilim])), np.sin(h[n, 0:ilim]))
        pts[..., 2] = np.multiply(r[n, 0:ilim], np.cos(h[n, 0:ilim]))

        # We reorder the points, scalars and vectors so this is as per VTK's
        # requirement of x first, y next and z last.
        pts = pts.transpose(2, 1, 0, 3).copy()
        pts.shape = pts.size // 3, 3

        #rhobsqorho = np.abs(1.0/(100.0*rho))[n, 0:ilim]
        #bsqorho=np.abs(bsq/rho)[n, 0:ilim]
        #rhobsqorho[rhobsqorho>10000]=10000*rhobsqorho[rhobsqorho>10000]/rhobsqorho[rhobsqorho>10000]
        #rhobsqorho[bsqorho > 2.0]=10000 * bsqorho[bsqorho > 2.0]
        #rhobsqorho=np.log10(rhobsqorho)
        rhobsqorho=rho[n, 0:ilim]

        # Create the dataset.
        sg = tvtk.StructuredGrid(dimensions=rho[n, 0:ilim].shape, points=pts)
        scalars = rhobsqorho
        scalars = scalars.T.copy()
        sg.point_data.scalars = scalars.ravel()
        sg.point_data.scalars.name = "bsqorho"
        write_data(sg, dir + "/visit/data%dn%d.vtk" % (dump, n))
        visitdata.write(dir + "/visit/data%dn%d.vtk\n" % (dump, n))

def set_mpi(cluster):
    global comm, numtasks, rank,setmpi

    if (cluster == 1):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numtasks = comm.Get_size()
        rank = comm.Get_rank()
        setmpi=1
        print(numtasks, rank)
        if len(sys.argv) > 1:
            if sys.argv[1] == "build_ext":
                if (rank == 0):
                    setup(
                        cmdclass={'build_ext': build_ext},
                        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"], include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"], extra_link_args=["-O2 -fopenmp"])]
                    )
    else:
        numtasks = 1
        rank = 0
        setmpi=0
        if len(sys.argv) > 1:
            if sys.argv[1] == "build_ext":
                if (rank == 0):
                    setup(
                        cmdclass={'build_ext': build_ext},
                        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"], include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"], extra_link_args=["-O2 -fopenmp"])]
                    )

    if (setmpi == 1):
        comm.barrier()

def createRGrids(dir, dump, radius):
    global r, ph, rho,bsq, x1, x2, x3, n_active_total,REF_1,REF_2,REF_3,bs1new,bs2new,bs3new
    print("Visit with AMR not implemented yet!")

# There's probably away to write the .vtm directly through the VTK
# API, but I haven't figured that out yet.
def writeblocks(f, base, nblocks):
    for i in range(nblocks):
        f.write('    <Block index="{0}" name="block{0:04d}">\n'.format(i))
        f.write('      <Piece index="0">\n')
        f.write('        <DataSet index="0" file="{}.{:04d}.vtr">\n'.format(base, i))
        f.write('        </DataSet>\n')
        f.write('      </Piece>\n')
        f.write('    </Block>\n')

def writevtm(base, nblocks):
    with open('{}.vtm'.format(base), 'w') as f:
        f.write('<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt64" compressor="vtkZLibDataCompressor">\n')
        f.write('  <vtkMultiBlockDataSet>\n')
        writeblocks(f, base, nblocks)
        f.write('  </vtkMultiBlockDataSet>\n')
        f.write('</VTKFile>\n')

def cool_disk(target_thickness=0.03, rmax=100):
    global gam,ug,rho,r, Rdot
    if(target_thickness>0.01):
        r_photon=2.0*(1+np.cos(2.0/3.0*np.arccos(-a))) #photon orbit
        epsilon = ug /rho
        om_kepler = 1. / (r**1.5 + a)
        T_target = np.pi / 2. * (target_thickness * r * om_kepler)**2
        Y = (gam - 1.) * epsilon / T_target
        ll = om_kepler * ug * np.sqrt(Y - 1. + np.abs(Y - 1.))
        ud_0=gcov[0,0]*uu[0]+gcov[0,1]*uu[1]+gcov[0,2]*uu[2]+gcov[0,3]*uu[3]
        source = ud_0 * ll
        source[(bsq / rho >= 1.)]=0.0

        source[r>rmax]=0.0
        source[r<r_photon]=0.0
        #source_tot=np.sum(source*gdet*_dx1*_dx2*_dx3)
        Rdot=(source*gdet*_dx1*_dx2*_dx3)[0].sum(-1).sum(-1).cumsum(axis=0)
    else:
        Rdot =np.zeros(bs1new)

def calc_aux_disk():
    global Temp, tau_bf, tau_es

    # Set constaints
    MH_CGS = 1.673534e-24  # Mass hydrogen molecule
    MMW = 1.69  # Mean molecular weight
    BOLTZ_CGS = 1.3806504e-16  # Boltzmanns constant
    c = 3 * 10 ** 10  # cm/s
    G = 6.67 * 10 ** (-8)  # cm^3/g/s
    Msun = 2 * 10 ** 33  # g

    # Set black hole mass and EOS law
    Mbh = 10 * Msun  # g (assuming 10 Msun BH)
    GAMMA = 5.0 / 3.0

    # Calculate lenght and timescales
    length_scale = G * Mbh / c ** 2  # cm
    time_scale = length_scale / c  # s

    # Set desired Mdot compared to Eddington rate
    L_dot_edd = 1.3 * 10 ** 46 * Mbh / (10 ** 8 * Msun)  # g/s
    efficiency = 0.17  # Look up for a=0.9375 black hole
    M_dot_edd = (1.0 / efficiency) * L_dot_edd / c ** 2  # g/s
    Mdot_desired = 0.1 * M_dot_edd  # According to paper

    # Calculate mass and energy density scales
    rho_scale = (Mdot_desired) / (Mdot[0, 5] * (length_scale ** 3) / time_scale)
    ENERGY_DENSITY_SCALE = (rho_scale * c * c)

    # Calculate temperature
    dU_scaled = source_proj[0, :, 0, :] * rho_scale * c ** 2 * (length_scale) / time_scale  # erg/cm^2/s
    sigma = 5.67 * 10 ** (-5)  # erg/cm^2/s/K
    Temp = np.nan_to_num((dU_scaled / sigma) ** (0.25) + 0.001)  # Kelvin
    Temp = Temp.astype(np.float64)

    # Calculate both bound-free and scattering optical depth across disk
    Tg = MMW * MH_CGS * (GAMMA - 1.) * (ug_proj[0, :, 0, :] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * rho_proj[0, :, 0, :] * rho_scale)
    tau_bf = 3.0 * pow(10., 25.) * Tg ** (-3.5) * (rho_proj[0, :, 0, :] * rho_scale) ** 2
    tau_es = 0.4 * rho_proj[0, :, 0, :] * rho_scale

def calc_delta_cas():
    global bsq, rho, TE, TI, delta_cas, E_rad
    MU_E = 1
    MU_I = 1
    MH_CGS = 1.673534e-24
    ME_CGS = 9.1094e-28
    ratio = (TE * MU_E) / (TI * MU_I)
    c1 = 0.92
    c2 = 1.6 * ratio * ((TI * MU_I) > (TE * MU_E)) + 1.2 * ratio * ((TI * MU_I) <= (TE * MU_E))
    c3 = 18.0 - 5.0 * np.log10(ratio) * ((TI * MU_I) > (TE * MU_E)) + 18.0 * ratio * ((TI * MU_I) <= (TE * MU_E))

    beta = ((TI + TE) * rho) / (0.5 * bsq)
    fel = c1 * (c2 * c2 + np.power(beta, 2.0 + 0.2 * np.log10(ratio))) / (c3 * c3 + np.power(beta, 2.0 + 0.2 * np.log10(ratio))) * np.sqrt((MH_CGS / ME_CGS) * (MU_I * TI) / (MU_E * TE)) * np.exp(-1.0 / beta)

    # Calculate delta
    delta_cas = 1. / (1. + fel)


def calc_delta_rec():
    global bsq, rho, TE, TI, delta_rec, E_rad

    sigma_w = bsq / (rho + gam * ug)
    beta_i = ((TI) * rho) / (0.5 * bsq)
    beta_max = 1.0 / (4.0 * sigma_w)

    # Calculate delta
    delta_rec = 0.5 * np.exp((beta_i / beta_max - 1.0)) / (0.8 + np.sqrt(sigma_w))


def calc_RAD(MASS_DENSITY_SCALE):
    global Tg, Tr, Tr_g, kappa_bf, kappa_emmit, kappa_es, kappa_sy, M_EDD_RAT, RAD_M1, gam, rho, ug, bsq, cluster, R_G_CGS, Te, TE, Ti, TI
    global gamg, game, gami

    # Various constants
    ARAD = 7.5657e-15  # Radiation density constant
    MH_CGS = 1.673534e-24  # Mass hydrogen molecule
    MMW = 1.69  # Mean molecular weight
    Z_AB = (0.02)
    Y_AB = (0.28)
    X_AB = (0.70)
    MU_I = (4.0 / (4.0 * X_AB + Y_AB))
    MU_E = (2.0 / (1.0 + X_AB))
    MU_G = (4.0 / (6 * X_AB + Y_AB + 2.0))
    MASS_RATIO = (MH_CGS / ME_CGS)
    BOLTZ_CGS = 1.3806504e-16  # Boltzmanns constant
    THOMSON_CGS = 6.652e-25  # homson cross section
    PLANCK_CGS = 6.6260755e-27  # Planck's constant
    STEFAN_CGS = 5.67051e-5  ##Stefan-Boltzmann constant
    FINE_CGS = 7.29735308e-3
    ERM_CGS = 9.10938215e-28  # Electron rest mass
    E_CGS = 4.80320427e-10  # Elementary charge
    C_CGS = 2.99792458e10  # Speed of light
    M_SGRA_SOLAR = 1.0e1  # Solar masses
    M_SOLAR_CGS = 1.998e33  # Solar mass
    G_CGS = 6.67259e-8  # Gravitational constant
    Z_AB = 0.02
    X_AB = 0.7
    Ye = (1. + X_AB) / 2.0
    Y_AB = 0.28

    # Calculate gamma
    Theta_e = TE * (MU_E * MASS_RATIO)
    Theta_i = TI * MU_I

    if (1):
        game = gam
        gami = gam
    else:
        game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e)
        gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i)
    gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0))

    # Scaling from code units to cgs units
    R_G_CGS = (M_SGRA_SOLAR * M_SOLAR_CGS * G_CGS / (C_CGS * C_CGS))  # Gravitational radius
    R_GOC_CGS = (R_G_CGS / C_CGS)  # Light-crossing time
    ENERGY_DENSITY_SCALE = (MASS_DENSITY_SCALE * C_CGS * C_CGS)
    MAGNETIC_DENSITY_SCALE = ((MASS_DENSITY_SCALE ** 0.5) * C_CGS)
    PRESSURE_SCALE = (MASS_DENSITY_SCALE * C_CGS * C_CGS)
    CK_CGS = (8. * np.pi / (C_CGS * C_CGS * C_CGS * PLANCK_CGS * PLANCK_CGS * PLANCK_CGS))

    # Calculate Eddington ratio
    calc_Mdot()
    L_dot_edd = 1.3 * 10 ** 46 * M_SGRA_SOLAR / (10 ** 8)  # erg/s
    efficiency = 0.17
    M_dot_edd = (1.0 / efficiency) * L_dot_edd / C_CGS ** 2  # g/s
    Mdot_desired = 2000.0 * M_dot_edd
    Mdot_actual_cgs = Mdot[0, 5] * MASS_DENSITY_SCALE * (R_G_CGS ** 3) / R_GOC_CGS
    M_EDD_RAT = Mdot_actual_cgs / M_dot_edd

    # Calculate gas, radiation temperature and opacities
    ud_rad = gcov[:, 0] * uu[0] + gcov[:, 1] * uu[1] + gcov[:, 2] * uu[2] + gcov[:, 3] * uu[3]
    u_dot_urad = uu[0] * ud_rad[0] + uu[1] * ud_rad[1] + uu[2] * ud_rad[2] + uu[3] * ud_rad[3]
    urad_dot_urad = uu_rad[0] * ud_rad[0] + uu_rad[1] * ud_rad[1] + uu_rad[2] * ud_rad[2] + uu_rad[3] * ud_rad[3]
    Ehat = np.float64(ENERGY_DENSITY_SCALE * ((4. / 3.) * E_rad * u_dot_urad * u_dot_urad + (1. / 3.) * E_rad * urad_dot_urad))
    Tg = MMW * MH_CGS * (gam - 1.) * (ug * C_CGS * C_CGS) / (BOLTZ_CGS * rho)
    Tr = (Ehat * ENERGY_DENSITY_SCALE / ARAD) ** (0.25)
    if (TWO_T):
        Te = TE * MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS
        Ti = TI * MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS

    if (P_NUM):
        Nhat = np.float64(-photon_number * MASS_DENSITY_SCALE * u_dot_urad)
        Tr = Ehat / (BOLTZ_CGS * Nhat * (3. - 2.449724 * Nhat * Nhat * Nhat * Nhat / (CK_CGS * Ehat * Ehat * Ehat)))

    ne = rho * MASS_DENSITY_SCALE / (MU_E * MH_CGS)
    kappa_m = 0.1 * Z_AB;
    kappa_h = 1.1 * 10 ** (-25) * np.sqrt(Z_AB * rho * MASS_DENSITY_SCALE) * Tg ** (7.7)
    kappa_chianti = 4.0 * 10 ** (34) * rho * MASS_DENSITY_SCALE * (Z_AB / 0.02) * Ye * Tg ** (-1.7) * Tr ** (-3.0)
    kappa_bf = 3.0 * 10 ** (25) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * rho * MASS_DENSITY_SCALE * Tg ** (-0.5) * Tr ** (-3.0) * np.log(1. + 1.6 * (Tr / Tg))
    kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * rho * MASS_DENSITY_SCALE * Tg ** (-0.5) * Tr ** (-3.0) * np.log(1. + 1.6 * (Tr / Tg)) * (1. + 4.4 * 10 ** (-10.0) * Tg)
    kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff))

    kappa_m2 = 0.1 * Z_AB;
    kappa_h2 = 1.1 * 10 ** (-25) * np.sqrt(Z_AB * rho * MASS_DENSITY_SCALE) * Tg ** (7.7)
    kappa_chianti2 = 4.0 * 10 ** (34) * rho * MASS_DENSITY_SCALE * (Z_AB / 0.02) * Ye * Tg ** (-4.7)
    kappa_bf2 = 3.0 * 10 ** (25) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * rho * MASS_DENSITY_SCALE * Tg ** (-3.5) * np.log(1. + 1.6)
    kappa_ff2 = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * rho * MASS_DENSITY_SCALE * Tg ** (-3.5) * np.log(1. + 1.6) * (1. + 4.4 * 10 ** (-10.0) * Tg)

    kappa_sy2 = 1.59 * pow(10., -30.) * ne * 4. * np.pi * bsq * MAGNETIC_DENSITY_SCALE * MAGNETIC_DENSITY_SCALE / (Te * Te)
    kappa_emmit = 1. / (1. / (kappa_m2 + kappa_h2) + 1. / (kappa_chianti2 + kappa_bf2 + kappa_ff2))

    kappa_es = 0.2 * (1 + X_AB) * rho * MASS_DENSITY_SCALE
    
def calc_isco():
    global r_isco, a
    a=0.9375
    Z1=1.0+(1.0-a**2.0)**(1.0/3.0)*((1.0+a)**(1.0/3.0)+(1.0-a)**(1.0/3.0))
    Z2=np.sqrt(3.0*a**2.0+Z1**2.0)
    r_isco=(3.0+Z2-np.sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)))

#Does bookkeeping, ie how many lines are in the file and what do those lines represent (nr_dumps and radial bins)
def set_aux_rad(dir):
    global j1,j2,j_size,line_count
    f = open(dir+"/post_process_rad.txt", 'r')
    line=f.readline()
    j_size=1
    
    line=f.readline()
    line_list=line.split()
    t=myfloat(line_list[0])
    line_count=1

    while(1):
        line=f.readline()
        if(line==''):
            break
        line_list=line.split()
        t1=myfloat(line_list[0])
        if(t1==t):
            line_count=line_count+1         
        j_size=j_size+1    
    j_size=int(j_size/line_count)

    f.close()

def calc_rad(dir,m):
    global time,rad, Mdot,Edot,Rdot,Edotj,Ldot, alpha_r,alpha_b,alpha_eff,H_o_R_real,H_o_R_thermal, rho_avg,pgas_avg,pb_avg,pitch_avg, phibh 
    global angle_tilt_disk, angle_prec_disk,angle_tilt_corona, angle_prec_corona, angle_tilt_jet1,angle_prec_jet1, opening_jet1
    global Q1_1, Q1_2, Q1_3, Q2_1,Q2_2,Q2_3, angle_tilt_jet2,angle_prec_jet2, opening_jet2
    global tilt_dot,prec_dot, j_size,j1,j2, line_count
    global sigma_jet2, gamma_jet2, E_jet2, M_jet2, temp_jet2, sigma_jet1, gamma_jet1, E_jet1,M_jet1, temp_jet1

    f = open(dir+"/post_process_rad.txt", 'r')
    line=f.readline()    
    for j in range(0,j_size):
        for i in range(0,line_count):
            line=f.readline()       
            line_list=line.split()
            time[m, j, i]=myfloat(line_list[0])
            rad[m, j, i]=myfloat(line_list[1]) 
            phibh[m, j, i]=myfloat(line_list[2])  
            Mdot[m, j, i]=myfloat(line_list[3])   
            Edot[m, j, i]=myfloat(line_list[4])    
            Edotj[m, j, i]=myfloat(line_list[5])  
            Ldot[m, j, i]=myfloat(line_list[6])  
            alpha_r[m, j, i]=myfloat(line_list[7]) 
            alpha_b[m, j, i]=myfloat(line_list[8])
            alpha_eff[m, j, i]=myfloat(line_list[9])
            H_o_R_real[m, j, i]=myfloat(line_list[10])
            H_o_R_thermal[m, j, i]=np.sqrt(1)*myfloat(line_list[11])
            rho_avg[m, j, i]=myfloat(line_list[12])
            pgas_avg[m, j, i]=myfloat(line_list[13])
            pb_avg[m, j, i]=myfloat(line_list[14])
            Q1_1[m, j, i]=myfloat(line_list[15])
            Q1_2[m, j, i]=myfloat(line_list[16])
            Q1_3[m, j, i]=myfloat(line_list[17])
            Q2_1[m, j, i]=myfloat(line_list[18])
            Q2_2[m, j, i]=myfloat(line_list[19])
            Q2_3[m, j, i]=myfloat(line_list[20])
            pitch_avg[m, j, i]=myfloat(line_list[21])
            angle_tilt_disk[m, j, i]=myfloat(line_list[22])
            angle_prec_disk[m, j, i]=myfloat(line_list[23])
            angle_tilt_corona[m, j, i]=myfloat(line_list[24])
            angle_prec_corona[m, j, i]=myfloat(line_list[25])
            angle_tilt_jet1[m, j, i]=myfloat(line_list[26])
            angle_prec_jet1[m, j, i]=myfloat(line_list[27])
            opening_jet1[m, j, i]=myfloat(line_list[28])
            angle_tilt_jet2[m, j, i]=myfloat(line_list[29])
            angle_prec_jet2[m, j, i]=myfloat(line_list[30])
            opening_jet2[m, j, i]=myfloat(line_list[31])
            if(len(line_list)>=33):
                Rdot[m, j, i]=myfloat(line_list[32])   
            if(len(line_list)>=43):
                sigma_jet1[m, j, i]=myfloat(line_list[33])   
                gamma_jet1[m, j, i]=myfloat(line_list[34]) 
                E_jet1[m, j, i]=myfloat(line_list[35]) 
                M_jet1[m, j, i]=myfloat(line_list[36]) 
                temp_jet1[m, j, i]=myfloat(line_list[37]) 
                sigma_jet2[m, j, i]=myfloat(line_list[38])   
                gamma_jet2[m, j, i]=myfloat(line_list[39]) 
                E_jet2[m, j, i]=myfloat(line_list[40]) 
                M_jet2[m, j, i]=myfloat(line_list[41]) 
                temp_jet2[m, j, i]=myfloat(line_list[42]) 
                
    sort_array=np.argsort(time[m,:,0])
    time[m,:,:]=time[m,sort_array,:]
    phibh[m,:,:]=phibh[m,sort_array,:]
    Mdot[m,:,:]=Mdot[m,sort_array,:]
    Edot[m,:,:]=Edot[m,sort_array,:]
    Rdot[m,:,:]=Rdot[m,sort_array,:]
    Edotj[m,:,:]=Edotj[m,sort_array,:]
    Ldot[m,:,:]=Ldot[m,sort_array,:]
    alpha_r[m,:,:]=alpha_r[m,sort_array,:]
    alpha_b[m,:,:]=alpha_b[m,sort_array,:]
    alpha_eff[m,:,:]=alpha_eff[m,sort_array,:]
    H_o_R_real[m,:,:]=H_o_R_real[m,sort_array,:]
    H_o_R_thermal[m,:,:]=H_o_R_thermal[m,sort_array,:]
    rho_avg[m,:,:]=rho_avg[m,sort_array,:]
    pgas_avg[m,:,:]=pgas_avg[m,sort_array,:]
    pb_avg[m,:,:]=pb_avg[m,sort_array,:]
    Q1_1[m,:,:]=Q1_1[m,sort_array,:]
    Q1_2[m,:,:]=Q1_2[m,sort_array,:]
    Q1_3[m,:,:]=Q1_3[m,sort_array,:]
    Q2_1[m,:,:]=Q2_1[m,sort_array,:]
    Q2_2[m,:,:]=Q2_2[m,sort_array,:]
    Q2_3[m,:,:]=Q2_3[m,sort_array,:]
    pitch_avg[m,:,:]=pitch_avg[m,sort_array,:]
    angle_tilt_disk[m,:,:]=angle_tilt_disk[m,sort_array,:]
    angle_prec_disk[m,:,:]=angle_prec_disk[m,sort_array,:]
    angle_tilt_corona[m,:,:]=angle_tilt_corona[m,sort_array,:]
    angle_prec_corona[m,:,:]=angle_prec_corona[m,sort_array,:]
    angle_tilt_jet1[m,:,:]=angle_tilt_jet1[m,sort_array,:]
    angle_prec_jet1[m,:,:]=angle_prec_jet1[m,sort_array,:]
    opening_jet1[m,:,:]=opening_jet1[m,sort_array,:]
    angle_tilt_jet2[m,:,:]=angle_tilt_jet2[m,sort_array,:]
    angle_prec_jet2[m,:,:]=angle_prec_jet2[m,sort_array,:]
    opening_jet2[m,:,:]=opening_jet2[m,sort_array,:]
    sigma_jet1[m, :,:]=sigma_jet1[m, sort_array,:]
    gamma_jet1[m, :,:]=gamma_jet1[m, sort_array,:]
    E_jet1[m, :,:]=E_jet1[m, sort_array,:]
    M_jet1[m, :,:]=M_jet1[m, sort_array,:]
    temp_jet1[m, :,:]=temp_jet1[m, sort_array,:]
    sigma_jet2[m, :,:]=sigma_jet2[m, sort_array,:]
    gamma_jet2[m, :,:]=gamma_jet2[m, sort_array,:]
    E_jet2[m, :,:]=E_jet2[m, sort_array,:]
    M_jet2[m, :,:]=M_jet2[m, sort_array,:]
    temp_jet2[m, :,:]=temp_jet2[m, sort_array,:]
    f.close()

def calc_avg_rad(m, begin, end):
    global time,rad, Mdot,Edot,Rdot,Edotj,Ldot, alpha_r,alpha_b,alpha_eff,H_o_R_real,H_o_R_thermal, rho_avg,pgas_avg,pb_avg,pitch_avg, phibh 
    global angle_tilt_disk, angle_prec_disk,angle_tilt_corona, angle_prec_corona, angle_tilt_jet1,angle_prec_jet1, opening_jet1
    global Q1_1, Q1_2, Q1_3, Q2_1,Q2_2,Q2_3, angle_tilt_jet2,angle_prec_jet2, opening_jet2
    global sigma_jet2, gamma_jet2, E_jet2,M_jet2, temp_jet2, sigma_jet1, gamma_jet1, E_jet1,M_jet1, temp_jet1
    global avg_time,avg_rad, avg_Mdot,avg_Edot,avg_Rdot,avg_Edotj,avg_Ldot, avg_alpha_r,avg_alpha_b,avg_alpha_eff,avg_H_o_R_real,avg_H_o_R_thermal, avg_rho_avg,avg_pgas_avg,avg_pb_avg,avg_pitch_avg, avg_phibh 
    global avg_angle_tilt_disk, avg_angle_prec_disk,avg_angle_tilt_corona, avg_angle_prec_corona, avg_angle_tilt_jet1,avg_angle_prec_jet1, avg_opening_jet1
    global avg_Q1_1, avg_Q1_2, avg_Q1_3, avg_Q2_1,avg_Q2_2,avg_Q2_3, avg_angle_tilt_jet2,avg_angle_prec_jet2, avg_opening_jet2
    global avg_L_disk,avg_L_corona,avg_L_jet1,avg_L_jet2
    global tilt_dot,prec_dot, j1,j2, j_size,line_count
    global avg_sigma_jet2, avg_gamma_jet2, avg_E_jet2,avg_M_jet2, avg_temp_jet2, avg_sigma_jet1, avg_gamma_jet1, avg_E_jet1,avg_M_jet1, avg_temp_jet1

    
    j1=0
    j2=0
    j1_set=0
    j2_set=0
    for j in range(0,j_size):
        t1=time[m,j,0]     
        if(t1>begin and j1_set==0):
            j1=j
            j1_set=1
        if(t1>end and j2_set==0):
            j2=j+1
            j2_set=1
    if(j2_set==0):
        j2=j_size

    for j in range(j1,j2):
        for i in range(0,line_count):
            avg_time[m, i]+=time[m, j, i]/np.float(j2-j1)
            avg_rad[m, i]+=rad[m, j, i]/np.float(j2-j1)
            avg_phibh[m, i]+=phibh[m, j, i]/np.float(j2-j1) 
            avg_Mdot[m, i]+=Mdot[m, j, i]/np.float(j2-j1)  
            avg_Edot[m, i]+=Edot[m, j, i]/np.float(j2-j1)   
            avg_Rdot[m, i]+=Rdot[m, j, i]/np.float(j2-j1)
            avg_Edotj[m, i]+=Edotj[m, j, i]/np.float(j2-j1) 
            avg_Ldot[m, i]+=Ldot[m, j, i]/np.float(j2-j1) 
            avg_alpha_r[m, i]+=alpha_r[m, j, i]/np.float(j2-j1)
            avg_alpha_b[m, i]+=alpha_b[m, j, i]/np.float(j2-j1)
            avg_alpha_eff[m, i]+=alpha_eff[m, j, i]/np.float(j2-j1)
            avg_H_o_R_real[m, i]+=H_o_R_real[m, j, i]/np.float(j2-j1)
            avg_H_o_R_thermal[m, i]+=H_o_R_thermal[m, j, i]/np.float(j2-j1)
            avg_rho_avg[m, i]+=rho_avg[m, j, i]/np.float(j2-j1)
            avg_pgas_avg[m, i]+=pgas_avg[m, j, i]/np.float(j2-j1)
            avg_pb_avg[m, i]+=pb_avg[m, j, i]/np.float(j2-j1)
            avg_Q1_1[m, i]+=Q1_1[m, j, i]/np.float(j2-j1)
            avg_Q1_2[m, i]+=Q1_2[m, j, i]/np.float(j2-j1)
            avg_Q1_3[m, i]+=Q1_3[m, j, i]/np.float(j2-j1)
            avg_Q2_1[m, i]+=Q2_1[m, j, i]/np.float(j2-j1)
            avg_Q2_2[m, i]+=Q2_2[m, j, i]/np.float(j2-j1)
            avg_Q2_3[m, i]+=Q2_3[m, j, i]/np.float(j2-j1)
            avg_sigma_jet1[m, i]+=sigma_jet1[m, j, i]/np.float(j2-j1)
            avg_gamma_jet1[m, i]+=gamma_jet1[m, j, i]/np.float(j2-j1)
            avg_E_jet1[m, i]+=E_jet1[m, j, i]/np.float(j2-j1)
            avg_M_jet1[m, i]+=M_jet1[m, j, i]/np.float(j2-j1)
            avg_temp_jet1[m, i]+=temp_jet1[m, j, i]/np.float(j2-j1)
            avg_sigma_jet2[m, i]+=sigma_jet2[m, j, i]/np.float(j2-j1)
            avg_gamma_jet2[m, i]+=gamma_jet2[m, j, i]/np.float(j2-j1)
            avg_E_jet2[m, i]+=E_jet2[m, j, i]/np.float(j2-j1)
            avg_M_jet2[m, i]+=M_jet2[m, j, i]/np.float(j2-j1)
            avg_temp_jet2[m, i]+=temp_jet2[m, j, i]/np.float(j2-j1)
            avg_pitch_avg[m, i]+=pitch_avg[m, j, i]/np.float(j2-j1)
            avg_angle_tilt_disk[m, i]=angle_tilt_disk[m, j, i]/180.0*3.141592
            avg_angle_prec_disk[m, i]=angle_prec_disk[m, j, i]/180.0*3.141592
            avg_angle_tilt_corona[m, i]=angle_tilt_corona[m, j, i]/180.0*3.141592
            avg_angle_prec_corona[m, i]=angle_prec_corona[m, j, i]/180.0*3.141592
            avg_angle_tilt_jet1[m, i]=angle_tilt_jet1[m, j, i]/180.0*3.141592
            avg_angle_prec_jet1[m, i]=angle_prec_jet1[m, j, i]/180.0*3.141592
            avg_opening_jet1[m, i]=opening_jet1[m, j, i]
            avg_angle_tilt_jet2[m, i]=angle_tilt_jet2[m, j, i]/180.0*3.141592
            avg_angle_prec_jet2[m, i]=angle_prec_jet2[m, j, i]/180.0*3.141592
            avg_opening_jet2[m, i]=opening_jet2[m, j, i]

            avg_L_disk_r=np.sin(avg_angle_tilt_disk[m, i])
            avg_L_disk[1,m,i]+=np.cos(avg_angle_prec_disk[m,i])*avg_L_disk_r/np.float(j2-j1)
            avg_L_disk[2,m,i]+=np.sin(avg_angle_prec_disk[m,i])*avg_L_disk_r/np.float(j2-j1)
            avg_L_disk[3,m,i]+=np.cos(avg_angle_tilt_disk[m,i])/np.float(j2-j1)
            avg_L_corona_r=np.sin(avg_angle_tilt_corona[m,i])
            avg_L_corona[1,m,i]+=np.cos(avg_angle_prec_corona[m,i])*avg_L_corona_r/np.float(j2-j1)
            avg_L_corona[2,m,i]+=np.sin(avg_angle_prec_corona[m,i])*avg_L_corona_r/np.float(j2-j1)
            avg_L_corona[3,m,i]+=np.cos(avg_angle_tilt_corona[m,i])/np.float(j2-j1)
            avg_L_jet1_r=np.sin(avg_angle_tilt_jet1[m,i])
            avg_L_jet1[1,m,i]+=np.cos(avg_angle_prec_jet1[m,i])*avg_L_jet1_r/np.float(j2-j1)
            avg_L_jet1[2,m,i]+=np.sin(avg_angle_prec_jet1[m,i])*avg_L_jet1_r/np.float(j2-j1)
            avg_L_jet1[3,m,i]+=np.cos(avg_angle_tilt_jet1[m,i])/np.float(j2-j1)
            avg_L_jet2_r=np.sin(avg_angle_tilt_jet2[m,i])
            avg_L_jet2[1,m,i]+=np.cos(avg_angle_prec_jet2[m,i])*avg_L_jet2_r/np.float(j2-j1)
            avg_L_jet2[2,m,i]+=np.sin(avg_angle_prec_jet2[m,i])*avg_L_jet2_r/np.float(j2-j1)
            avg_L_jet2[3,m,i]+=np.cos(avg_angle_tilt_jet2[m,i])/np.float(j2-j1)
                
    for i in range(0,line_count):
        avg_angle_tilt_disk[m, i]=np.arccos(avg_L_disk[3,m,i])*180.0/3.14
        avg_angle_prec_disk[m, i]=np.arctan2(avg_L_disk[2,m,i],avg_L_disk[1,m,i])*180.0/3.14
        avg_angle_tilt_corona[m, i]=np.arccos(avg_L_corona[3,m,i])*180.0/3.14
        avg_angle_prec_corona[m, i]=np.arctan2(avg_L_corona[2,m,i],avg_L_corona[1,m,i])*180.0/3.14
        avg_angle_tilt_jet1[m, i]=np.arccos(avg_L_jet1[3,m,i])*180.0/3.14
        avg_angle_prec_jet1[m, i]=np.arctan2(avg_L_jet1[2,m,i],avg_L_jet1[1,m,i])*180.0/3.14
        avg_angle_tilt_jet2[m, i]=np.arccos(avg_L_jet2[3,m,i])*180.0/3.14
        avg_angle_prec_jet2[m, i]=np.arctan2(avg_L_jet2[2,m,i],avg_L_jet2[1,m,i])*180.0/3.14
    

def calc_sigma_rad(m):
    global sigma_time,sigma_rad,sigma_Mdot,sigma_Edot,sigma_Rdot,sigma_Edotj,sigma_Ldot, sigma_alpha_r,sigma_alpha_b,sigma_alpha_eff,sigma_H_o_R_real,sigma_H_o_R_thermal, sigma_rho_avg,sigma_pgas_avg,sigma_pb_avg,sigma_pitch_avg, sigma_phibh 
    global sigma_angle_tilt_disk, sigma_angle_prec_disk,sigma_angle_tilt_corona, sigma_angle_prec_corona, sigma_angle_tilt_jet1,sigma_angle_prec_jet1, sigma_opening_jet1
    global sigma_Q1_1, sigma_Q1_2, sigma_Q1_3, sigma_Q2_1,sigma_Q2_2,sigma_Q2_3, sigma_angle_tilt_jet2,sigma_angle_prec_jet2, sigma_opening_jet2
    global tilt_dot,prec_dot, i_size,j1,j2, j_size, line_count  
    global sigma_sigma_jet2, sigma_gamma_jet2, sigma_E_jet2,sigma_M_jet2, sigma_temp_jet2, sigma_sigma_jet1, sigma_gamma_jet1, sigma_E_jet1,sigma_M_jet1, sigma_temp_jet1
    
    for j in range(j1,j2):
        for i in range(0,line_count):
            sigma_time[m, i]+=((avg_time[m,i]-time[m,j,i])**2/np.float(j2-j1))
            sigma_rad[m, i]+=((avg_rad[m,i]-rad[m,j,i])**2/np.float(j2-j1))
            sigma_phibh[m, i]+=((avg_phibh[m,i]-phibh[m,j,i])**2/np.float(j2-j1)) 
            sigma_Mdot[m, i]+=((avg_Mdot[m,i]-Mdot[m,j,i])**2/np.float(j2-j1))   
            sigma_Edot[m, i]+=((avg_Edot[m,i]-Edot[m,j,i])**2/np.float(j2-j1))    
            sigma_Rdot[m, i]+=((avg_Rdot[m,i]-Rdot[m,j,i])**2/np.float(j2-j1))    
            sigma_Edotj[m, i]+=((avg_Edotj[m,i]-Edotj[m,j,i])**2/np.float(j2-j1))  
            sigma_Ldot[m, i]+=((avg_Ldot[m,i]-Ldot[m,j,i])**2/np.float(j2-j1))  
            sigma_alpha_r[m, i]+=((avg_alpha_r[m,i]-alpha_r[m,j,i])**2/np.float(j2-j1)) 
            sigma_alpha_b[m, i]+=((avg_alpha_b[m,i]-alpha_b[m,j,i])**2/np.float(j2-j1))
            sigma_alpha_eff[m, i]+=((avg_alpha_eff[m,i]-alpha_eff[m,j,i])**2/np.float(j2-j1))
            sigma_H_o_R_real[m, i]+=((avg_H_o_R_real[m,i]-H_o_R_real[m,j,i])**2/np.float(j2-j1))
            sigma_H_o_R_thermal[m, i]+=((avg_H_o_R_thermal[m,i]-H_o_R_thermal[m,j,i])**2/np.float(j2-j1))
            sigma_rho_avg[m, i]+=((avg_rho_avg[m,i]-rho_avg[m,j,i])**2/np.float(j2-j1))
            sigma_pgas_avg[m, i]+=((avg_pgas_avg[m,i]-pgas_avg[m,j,i])**2/np.float(j2-j1))
            sigma_pb_avg[m, i]+=((avg_pb_avg[m,i]-pb_avg[m,j,i])**2/np.float(j2-j1))
            sigma_Q1_1[m, i]+=((avg_Q1_1[m,i]-Q1_1[m,j,i])**2/np.float(j2-j1))
            sigma_Q1_2[m, i]+=((avg_Q1_2[m,i]-Q1_2[m,j,i])**2/np.float(j2-j1))
            sigma_Q1_3[m, i]+=((avg_Q1_3[m,i]-Q1_3[m,j,i])**2/np.float(j2-j1))
            sigma_Q2_1[m, i]+=((avg_Q2_1[m,i]-Q2_1[m,j,i])**2/np.float(j2-j1))
            sigma_Q2_2[m, i]+=((avg_Q2_2[m,i]-Q2_2[m,j,i])**2/np.float(j2-j1))
            sigma_Q2_3[m, i]+=((avg_Q2_3[m,i]-Q2_3[m,j,i])**2/np.float(j2-j1))
            sigma_sigma_jet1[m, i]+=((avg_sigma_jet1[m,i]-sigma_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_gamma_jet1[m, i]+=((avg_gamma_jet1[m,i]-gamma_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_E_jet1[m, i]+=((avg_E_jet1[m,i]-E_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_M_jet1[m, i]+=((avg_M_jet1[m,i]-M_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_temp_jet1[m, i]+=((avg_temp_jet1[m,i]-temp_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_sigma_jet2[m, i]+=((avg_sigma_jet2[m,i]-sigma_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_gamma_jet2[m, i]+=((avg_gamma_jet2[m,i]-gamma_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_E_jet2[m, i]+=((avg_E_jet2[m,i]-E_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_M_jet2[m, i]+=((avg_M_jet2[m,i]-M_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_temp_jet2[m, i]+=((avg_temp_jet2[m,i]-temp_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_pitch_avg[m, i]+=((avg_pitch_avg[m,i]-pitch_avg[m,j,i])**2/np.float(j2-j1))
            sigma_angle_tilt_disk[m, i]+=((avg_angle_tilt_disk[m,i]-angle_tilt_disk[m,j,i])**2/np.float(j2-j1))
            sigma_angle_prec_disk[m, i]+=((avg_angle_prec_disk[m,i]%360-angle_prec_disk[m,j,i]%360)**2/np.float(j2-j1))
            sigma_angle_tilt_corona[m, i]+=((avg_angle_tilt_corona[m,i]-angle_tilt_corona[m,j,i])**2/np.float(j2-j1))
            sigma_angle_prec_corona[m, i]+=((avg_angle_prec_corona[m,i]%360-angle_prec_corona[m,j,i]%360)**2/np.float(j2-j1))
            sigma_angle_tilt_jet1[m, i]+=((avg_angle_tilt_jet1[m,i]-angle_tilt_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_angle_prec_jet1[m, i]+=((avg_angle_prec_jet1[m,i]%360-angle_prec_jet1[m,j,i]%360)**2/np.float(j2-j1))
            sigma_opening_jet1[m, i]+=((avg_opening_jet1[m,i]-opening_jet1[m,j,i])**2/np.float(j2-j1))
            sigma_angle_tilt_jet2[m, i]+=((avg_angle_tilt_jet2[m,i]-angle_tilt_jet2[m,j,i])**2/np.float(j2-j1))
            sigma_angle_prec_jet2[m, i]+=((avg_angle_prec_jet2[m,i]%360-angle_prec_jet2[m,j,i]%360)**2/np.float(j2-j1))
            sigma_opening_jet2[m, i]+=((avg_opening_jet2[m,i]-opening_jet2[m,j,i])**2/np.float(j2-j1))
        
def alloc_mem_rad():
    global time,rad,Mdot,Edot,Rdot,Edotj,Ldot, alpha_r,alpha_b,alpha_eff,H_o_R_real,H_o_R_thermal, rho_avg,pgas_avg,pb_avg,pitch_avg, phibh 
    global angle_tilt_disk, angle_prec_disk,angle_tilt_corona, angle_prec_corona, angle_tilt_jet1,angle_prec_jet1, opening_jet1
    global Q1_1, Q1_2, Q1_3, Q2_1,Q2_2,Q2_3, angle_tilt_jet2,angle_prec_jet2, opening_jet2
    global sigma_jet2, gamma_jet2, E_jet2,M_jet2, temp_jet2, sigma_jet1, gamma_jet1, E_jet1,M_jet1, temp_jet1
    global avg_time,avg_rad,avg_Mdot,avg_Edot,avg_Rdot,avg_Edotj,avg_Ldot, avg_alpha_r,avg_alpha_b,avg_alpha_eff,avg_H_o_R_real,avg_H_o_R_thermal, avg_rho_avg,avg_pgas_avg,avg_pb_avg,avg_pitch_avg, avg_phibh 
    global avg_angle_tilt_disk, avg_angle_prec_disk,avg_angle_tilt_corona, avg_angle_prec_corona, avg_angle_tilt_jet1,avg_angle_prec_jet1, avg_opening_jet1
    global avg_Q1_1, avg_Q1_2, avg_Q1_3, avg_Q2_1,avg_Q2_2,avg_Q2_3, avg_angle_tilt_jet2,avg_angle_prec_jet2, avg_opening_jet2
    global avg_L_disk,avg_L_corona,avg_L_jet1,avg_L_jet2
    global avg_sigma_jet2, avg_gamma_jet2, avg_E_jet2,avg_M_jet2, avg_temp_jet2, avg_sigma_jet1, avg_gamma_jet1, avg_E_jet1,avg_M_jet1, avg_temp_jet1
    global sigma_sigma_jet2, sigma_gamma_jet2, sigma_E_jet2,sigma_M_jet2, sigma_temp_jet2, sigma_sigma_jet1, sigma_gamma_jet1, sigma_E_jet1,sigma_M_jet1, sigma_temp_jet1
    global sigma_time,sigma_rad,sigma_Mdot,sigma_Edot,sigma_Rdot,sigma_Edotj,sigma_Ldot, sigma_alpha_r,sigma_alpha_b,sigma_alpha_eff,sigma_H_o_R_real,sigma_H_o_R_thermal, sigma_rho_avg,sigma_pgas_avg,sigma_pb_avg,sigma_pitch_avg, sigma_phibh 
    global sigma_angle_tilt_disk, sigma_angle_prec_disk,sigma_angle_tilt_corona, sigma_angle_prec_corona, sigma_angle_tilt_jet1,sigma_angle_prec_jet1, sigma_opening_jet1
    global sigma_Q1_1, sigma_Q1_2, sigma_Q1_3, sigma_Q2_1,sigma_Q2_2,sigma_Q2_3, sigma_angle_tilt_jet2,sigma_angle_prec_jet2, sigma_opening_jet2
    global n_models,color,label
    
    color=[None]*n_models
    label=[None]*n_models

    i_size=960 #nr_points_in_x1
    j_size=22000 #nr_dumps
    time=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    for n in range(0,n_models):
        for j in range(j_size):
            time[n,j,0]=1000000+j
    rad=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    phibh=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Mdot=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Edot=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Rdot=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Edotj=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Ldot=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    alpha_r=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    alpha_b=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    alpha_eff=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    H_o_R_real=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    H_o_R_thermal=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    rho_avg=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    pgas_avg=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    pb_avg=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q1_1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q1_2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q1_3=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q2_1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q2_2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    Q2_3=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    pitch_avg=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_tilt_disk=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_prec_disk=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_tilt_corona=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_prec_corona=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_tilt_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_prec_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    opening_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    sigma_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    gamma_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    E_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    M_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    temp_jet1=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_tilt_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    angle_prec_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    opening_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    sigma_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    gamma_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    E_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    M_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    temp_jet2=np.zeros((n_models,j_size,i_size),dtype=mytype,order='F')
    
    avg_time=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_rad=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_phibh=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Mdot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Edot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Rdot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Edotj=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Ldot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_alpha_r=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_alpha_b=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_alpha_eff=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_H_o_R_real=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_H_o_R_thermal=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_rho_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_pgas_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_pb_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q1_1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q1_2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q1_3=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q2_1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q2_2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_Q2_3=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_pitch_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_L_disk=np.zeros((4,n_models,i_size),dtype=mytype,order='F')
    avg_L_corona=np.zeros((4,n_models,i_size),dtype=mytype,order='F')
    avg_L_jet1=np.zeros((4,n_models,i_size),dtype=mytype,order='F')
    avg_L_jet2=np.zeros((4,n_models,i_size),dtype=mytype,order='F')
    avg_angle_tilt_disk=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_prec_disk=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_tilt_corona=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_prec_corona=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_tilt_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_prec_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_opening_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_sigma_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_gamma_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_E_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_M_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_temp_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_tilt_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_angle_prec_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_opening_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_sigma_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_gamma_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_E_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_M_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    avg_temp_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    
    sigma_time=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_rad=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_phibh=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Mdot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Edot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Rdot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Edotj=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Ldot=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_alpha_r=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_alpha_b=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_alpha_eff=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_H_o_R_real=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_H_o_R_thermal=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_rho_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_pgas_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_pb_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q1_1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q1_2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q1_3=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q2_1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q2_2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_Q2_3=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_pitch_avg=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_tilt_disk=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_prec_disk=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_tilt_corona=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_prec_corona=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_tilt_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_prec_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_opening_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_sigma_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_gamma_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_E_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_M_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_temp_jet1=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_tilt_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_angle_prec_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_opening_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_sigma_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_gamma_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_E_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_M_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    sigma_temp_jet2=np.zeros((n_models,i_size),dtype=mytype,order='F')
    
#Does bookkeeping, ie how many lines are in the file and what do those lines represent (nr_dumps and radial bins)
def set_aux_time(dir):
    global j_size_t, color,label
    color=[None]*n_models
    label=[None]*n_models
    f = open(dir+"/post_process.txt", 'r')
    line=f.readline()
    j_size_t=0
    while(1):
        line=f.readline()
        if(line==''):
            break
        j_size_t=j_size_t+1
    f.close()

def calc_time(dir,m):
    global t, Mtot, t_Mdot,t_Edot,t_Edotj, t_Ldot, t_lum, t_prec_period, t_phibh, t_rad_avg,t_Rdot
    global t_angle_tilt_disk, t_angle_prec_disk, t_angle_tilt_corona, t_angle_prec_corona, t_angle_tilt_jet1,t_angle_prec_jet1, t_angle_tilt_jet2, t_angle_prec_jet2
    global j_size_t,pred_prec_angle
    f = open(dir+"/post_process.txt", 'r')
    line=f.readline()
    for j in range(0,j_size_t):
        line=f.readline()
        line_list=line.split()
        t[m,j]=myfloat(line_list[0])
        t_phibh[m,j]=line_list[1]
        t_Mdot[m,j]=line_list[2]
        t_Edot[m,j]=line_list[3]
        t_Edotj[m,j]=line_list[4]
        t_Ldot[m,j]=line_list[5]
        t_lum[m,j]=line_list[6]
        t_prec_period[m,j]=line_list[7]
        t_angle_tilt_disk[m,j]=line_list[8]
        t_angle_prec_disk[m,j]=line_list[9]
        t_angle_tilt_corona[m,j]=line_list[10]
        t_angle_prec_corona[m,j]=line_list[11]
        t_angle_tilt_jet1[m,j]=line_list[12]
        t_angle_prec_jet1[m,j]=line_list[13]
        t_angle_tilt_jet2[m,j]=line_list[14]
        t_angle_prec_jet2[m,j]=line_list[15]
        t_rad_avg[m,j]=line_list[16]
        if(len(line_list)==18):
            t_Rdot[m,j]=line_list[17]

    sort_array=np.argsort(t[m])
    t[m]=t[m,sort_array]
    t_phibh[m]=t_phibh[m,sort_array]
    t_Mdot[m]=t_Mdot[m,sort_array]
    t_Edot[m]=t_Edot[m,sort_array]
    t_Edotj[m]=t_Edotj[m,sort_array] 
    t_Ldot[m]=t_Ldot[m,sort_array] 
    t_lum[m]=t_lum[m,sort_array] 
    t_prec_period[m]=t_prec_period[m,sort_array] 
    t_angle_tilt_disk[m]=t_angle_tilt_disk[m,sort_array]
    t_angle_prec_disk[m]=t_angle_prec_disk[m,sort_array]
    t_angle_tilt_corona[m]=t_angle_tilt_corona[m,sort_array]
    t_angle_prec_corona[m]=t_angle_prec_corona[m,sort_array]
    t_angle_tilt_jet1[m]=t_angle_tilt_jet1[m,sort_array]
    t_angle_prec_jet1[m]=t_angle_prec_jet1[m,sort_array]
    t_angle_tilt_jet2[m]=t_angle_tilt_jet2[m,sort_array]
    t_angle_prec_jet2[m]=t_angle_prec_jet2[m,sort_array]
    t_rad_avg[m]=t_rad_avg[m,sort_array]
    t_Rdot[m]=t_Rdot[m,sort_array] 
    pred_prec_angle=np.copy(t_prec_period)*0.0
    for j in range(1,j_size_t):
        pred_prec_angle[m,j]=pred_prec_angle[m,j-1]+(t[m,j]-t[m,j-1])/t_prec_period[m,j]*360.0
    f.close()

def calc_sigma_time(dir,m, rmin, rmax):
    global t, Mtot, t_Mdot,t_Edot,t_Edotj, t_Ldot, t_lum, t_prec_period, t_phibh, t_rad_avg,t_Rdot
    global t_angle_tilt_disk, t_angle_prec_disk, t_angle_tilt_corona, t_angle_prec_corona, t_angle_tilt_jet1,t_angle_prec_jet1, t_angle_tilt_jet2, t_angle_prec_jet2
    global sigma_t_angle_tilt_disk, sigma_t_angle_prec_disk, sigma_t_angle_tilt_corona, sigma_t_angle_prec_corona, sigma_t_angle_tilt_jet1,sigma_t_angle_prec_jet1, sigma_t_angle_tilt_jet2, sigma_t_angle_prec_jet2
    global avg_rad
    global j_size, j_size_t
    i1=0
    i2=0
    i1_set=0
    i2_set=0
    #find radii i1,i2 for which you want to calculate sigma
    for i in range(0,line_count):
        if(rad[m,0,i]>rmin and i1_set==0):
            i1=i
            i1_set=1
        if(rad[m,0,i]>rmax and i2_set==0):
            i2=i
            i2_set=1
    if(i2_set==0):
        i2=line_count
    if(j_size!=j_size_t):
        print("Error j_size!=j_size_t")
    for j in range(0, j_size):
        for i in range(i1,i2):
            sigma_t_angle_tilt_disk[m,j]+=(np.nan_to_num(angle_tilt_disk[m,j,i]-t_angle_tilt_disk[m,j])**2/np.float(i2-i1))
            sigma_t_angle_tilt_corona[m,j]+=(np.nan_to_num(angle_tilt_corona[m,j,i]-t_angle_tilt_corona[m,j])**2/np.float(i2-i1))
            sigma_t_angle_tilt_jet1[m,j]+=(np.nan_to_num(angle_tilt_jet1[m,j,i]-t_angle_tilt_jet1[m,j])**2/np.float(i2-i1))
            sigma_t_angle_tilt_jet2[m,j]+=(np.nan_to_num(angle_tilt_jet2[m,j,i]-t_angle_tilt_jet2[m,j])**2/np.float(i2-i1))
            sigma_t_angle_prec_disk[m,j]+=(np.nan_to_num((angle_prec_disk[m,j,i]%360-t_angle_prec_disk[m,j]%360))**2/np.float(i2-i1))
            sigma_t_angle_prec_corona[m,j]+=(np.nan_to_num((angle_prec_corona[m,j,i]%360-t_angle_prec_corona[m,j]%360))**2/np.float(i2-i1))
            sigma_t_angle_prec_jet1[m,j]+=(np.nan_to_num((angle_prec_jet1[m,j,i]%360-t_angle_prec_jet1[m,j]%360))**2/np.float(i2-i1))
            sigma_t_angle_prec_jet2[m,j]+=(np.nan_to_num((angle_prec_jet2[m,j,i]%360-t_angle_prec_jet2[m,j]%360))**2/np.float(i2-i1))
    
def alloc_mem_time():
    global t, Mtot, t_Mdot,t_Edot,t_Edotj, t_Ldot, t_lum, t_prec_period, t_phibh, t_rad_avg,t_Rdot
    global t_angle_tilt_disk, t_angle_prec_disk, t_angle_tilt_corona, t_angle_prec_corona, t_angle_tilt_jet1,t_angle_prec_jet1, t_angle_tilt_jet2, t_angle_prec_jet2
    global sigma_t_angle_tilt_disk, sigma_t_angle_prec_disk, sigma_t_angle_tilt_corona, sigma_t_angle_prec_corona, sigma_t_angle_tilt_jet1,sigma_t_angle_prec_jet1, sigma_t_angle_tilt_jet2, sigma_t_angle_prec_jet2,pred_prec_angle
    global j_size, n_models
    max_size=22000

    t=np.zeros((n_models,max_size),dtype=mytype,order='F')
    
    for n in range(0,n_models):
        for i in range(0,max_size):
            t[n,i]=1000000
    t_Mdot=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_Edot=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_Edotj=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_Rdot=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_Ldot=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_lum=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_prec_period=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_phibh=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_tilt_disk=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_prec_disk=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_tilt_corona=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_prec_corona=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_tilt_jet1=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_prec_jet1=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_tilt_jet2=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_angle_prec_jet2=np.zeros((n_models,max_size),dtype=mytype,order='F')
    pred_prec_angle=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_tilt_disk=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_prec_disk=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_tilt_corona=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_prec_corona=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_tilt_jet1=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_prec_jet1=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_tilt_jet2=np.zeros((n_models,max_size),dtype=mytype,order='F')
    sigma_t_angle_prec_jet2=np.zeros((n_models,max_size),dtype=mytype,order='F')
    t_rad_avg=np.zeros((n_models,max_size),dtype=mytype,order='F')

	#Does bookkeeping, ie how many lines are in the file and what do those lines represent (nr_dumps and radial bins)
def set_aux_but(dir):
    global j1,j2,j_size_b, z_size_b,line_count_b
    f = open(dir+"/post_process_but.txt", 'r')
    line=f.readline()
    j_size_b=1
    z_size_b=1
    line=f.readline()
    line_list=line.split()
    t=myfloat(line_list[0])
    r=myfloat(line_list[1])
    line_count_b=1

    while(1):
        line=f.readline()
        if(line==''):
            break
        line_list=line.split()
        t1=myfloat(line_list[0])
        r1=myfloat(line_list[1])
        if(t1==t):
            line_count_b=line_count_b+1    
        if(r1==r):
            z_size_b=z_size_b+1   
        j_size_b=j_size_b+1    
    print(j_size_b)
    z_size_b=int(j_size_b/(z_size_b)) ##number of radial bins
    j_size_b=int(j_size_b/line_count_b) #number of temporal bins
    #z_size_b=int(z_size_b/j_size_b)

    line_count_b=int(line_count_b/z_size_b)
    f.close()

def calc_but(dir,m):
    global b_time, b_rad, b_theta, b_rho,b_pgas,b_pb, b_br,b_btheta,b_bphi,b_ur,b_utheta,b_uphi
    global j_size_b, z_size_b, line_count_b
    
    f = open(dir+"/post_process_but.txt", 'r')
    line=f.readline()    
    for j in range(0,j_size_b):
        for z in range(0,z_size_b):
            for i in range(0,line_count_b):
                line=f.readline()       
                line_list=line.split()
                b_time[m, j, z, i]=myfloat(line_list[0])
                b_rad[m, j, z, i]=myfloat(line_list[1]) 
                b_theta[m, j, z, i]=myfloat(line_list[2])  
                b_rho[m, j, z, i]=myfloat(line_list[3])   
                b_pgas[m, j, z, i]=myfloat(line_list[4])    
                b_pb[m, j, z, i]=myfloat(line_list[5])  
                b_br[m, j, z, i]=myfloat(line_list[6])  
                b_btheta[m, j, z, i]=myfloat(line_list[7]) 
                b_bphi[m, j, z, i]=myfloat(line_list[8])
                b_ur[m, j, z, i]=myfloat(line_list[9])
                b_utheta[m, j, z, i]=myfloat(line_list[10])
                b_uphi[m, j, z, i]=myfloat(line_list[11])
    '''
    sort_array=np.argsort(b_time[m,:,0,0])
    b_time[m,:,:,:]=b_time[m,sort_array,:,:]
    b_rad[m,:,:,:]=b_rad[m,sort_array,:,:]
    b_theta[m,:,:,:]=b_theta[m,sort_array,:,:]
    b_rho[m,:,:,:]=b_rho[m,sort_array,:,:]
    b_pgas[m,:,:,:]=b_pgas[m,sort_array,:,:]
    b_pb[m,:,:,:]=b_pb[m,sort_array,:,:]
    b_br[m,:,:,:]=b_br[m,sort_array,:,:]
    b_btheta[m,:,:,:]=b_btheta[m,sort_array,:,:]
    b_bphi[m,:,:,:]=b_bphi[m,sort_array,:,:]
    b_ur[m,:,:,:]=b_ur[m,sort_array,:,:]
    b_utheta[m,:,:,:]=b_utheta[m,sort_array,:,:]
    b_uphi[m,:,:,:]=b_uphi[m,sort_array,:,:]
    '''
    f.close()

def alloc_mem_but():
    global b_time, b_rad, b_theta, b_rho,b_pgas,b_pb, b_br,b_btheta,b_bphi,b_ur,b_utheta,b_uphi
    global n_models,color,label
    
    color=[None]*n_models
    label=[None]*n_models

    i_size=11600 #N_time
    j_size=10 #N_r
    z_size=800 #N_theta
    b_time=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    for n in range(0,n_models):
        for i in range(0,i_size):
            b_time[n,i,0,0]=100000+n+i
    b_rad=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_theta=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_rho=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_pgas=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_pb=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_br=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_btheta=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_bphi=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_ur=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_utheta=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    b_uphi=np.zeros((n_models,i_size, j_size,z_size),dtype=mytype,order='F')
    
def plc_but(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):  # plc
    global bsqorho
    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 1)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    z = kwargs.pop('z', 0)
    
    ax = plt.gca()
    
    if isfilled:
        res = ax.contourf(xcoord, ycoord,myvar, nc, extend='both', **kwargs)
    else:
        res = ax.contour(xcoord, ycoord, myvar,nc, extend='both', **kwargs)
    
    
    ax.contour(xcoord, ycoord, bsqorho, levels=np.arange(1,2,1),cb=0, colors='black', linewidths=4)
    plt.title(r"$b^{\hat{\phi}}$ at 40 $\mathrm{r_{g}}$" %b_rad[m,0,r,0], fontsize=30)
    plt.xticks(fontsize = "25")
    plt.yticks(fontsize = "25")
    plt.xlabel(r"t [$\mathrm{10^4r_{g}/c}$]", fontsize = '25')
    plt.ylabel(r"$\mathcal{\theta}$ [$\mathrm{rad}$]", fontsize = '25')
    
    ax.tick_params(axis='both', reset=False, which='both', length=8, width=2)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax, ticks=np.arange(-0.01,0.011,0.002)) 
    cb.ax.tick_params(labelsize=25)
    plt.tight_layout()
    plt.savefig("butterfly.png",dpi=300)
    return res

import time
from multiprocessing import Process
def post_process(dir, dump_start, dump_end, dump_stride):
    global axisym, lowres1,lowres2,lowres3, REF_1, REF_2, REF_3, set_cart, set_xc,tilt_angle, _dx1,_dx2, _dx3, Mdot, Edot,Ldot,phibh, rad_avg,H_over_R1, H_over_R2, interpolate_var, rad_avg,Rdot,lum, temp_tilt, temp_prec
    global alpha_r, alpha_b, alpha_eff, pitch_avg, aphi, export_visit, print_fieldlines, setmpi
    global sigma_Ju, gamma_Ju, E_Ju ,mass_Ju,temp_Ju
    global sigma_Jd, gamma_Jd, E_Jd, mass_Jd, temp_Ju
    global comm, numtasks, rank,notebook, RAD_M1, export_raytracing_GRTRANS, export_raytracing_RAZIEH
    global pgas_avg, rho_avg, pb_avg, Q_avg1_1,Q_avg1_2,Q_avg1_3, Q_avg2_1,Q_avg2_2,Q_avg2_3,flag_restore, DISK_THICKNESS
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, do_griddata, do_box, check_files

    do_unigrid=1 #To use gdump_griddata instead of loading grid block by block
    do_box=0 #Boundaries of region you want to load in, select -1 to load in everything, works only in combination with do_griddata, ignored otherwise
    r_min=0.0
    r_max=150.0
    theta_min=-1.2
    theta_max=5
    phi_min=-1.0
    phi_max=9
    lowres1 = 1
    lowres2 = 1
    lowres3 = 1
    axisym = 1
    set_mpi(1) #Enable if you want to use mpi
    notebook=10
    if(notebook==1):
        return(1)
    notebook=0
    os.chdir(dir)
    interpolate_var=0
    tilt_angle=0.0
    print_angles=0 #Generate post process files
    print_images=1 #Make images
    print_fieldlines=0
    print_but=0
    export_visit=0
    export_raytracing_BHOSS=0
    export_raytracing_GRTRANS = 0
    export_raytracing_RAZIEH = 0
    check_files=0 #Check files for integrity
    downscale_files=0
    kerr_schild=0 #Use only if coordinates are close to x1=log(r), x2= theta, x3= phi
    DISK_THICKNESS=0.03
    cutoff = 0.000001  # cutoff for density in averaging quantitites

    if (print_angles):
        f = open(dir + "/post_process%d.txt" % rank, "w")
        f_rad = open(dir + "/post_process_rad%d.txt" %rank, "w")
    if(print_but):
        f_but = open(dir + "/post_process_but%d.txt" %rank, "w")
    if (rank == 0):
        if (print_angles):
            f.write("t,phibh, Mdot, Edot, Edotj, Ldot, lambda, prec_period, tilt_disk, prec_disk, tilt_corona, prec_corona,tilt_jet1, prec_jet1, tilt_jet2, prec_jet2, rad_avg, Rdot\n")
            f_rad.write("t, r, phibh, Mdot, Edot, Edotj, Ldot, alpha_r, alpha_b,alpha_eff, H_o_R_real, H_o_R_thermal, rho_avg, pgas_avg, pb_avg, Q_avg1_1, Q_avg1_2, Q_avg1_3, Q_avg2_1, Q_avg2_2, Q_avg2_3, pitch_avg, tilt_disk, prec_disk, tilt_corona, prec_corona, tilt_jet1, prec_jet1, opening_jet1, tilt_jet2, prec_jet2, opening_jet2, Rdot, sigma_Ju, gamma_Ju, E_Ju, mass_Ju, temp_Ju,sigma_Jd, gamma_Jd, E_Jd, mass_Jd, temp_Jd\n")
        if(print_but):
            f_but.write("t, r, theta, rho, pgas, pb, b_r, b_theta, b_phi, u_r, u_theta, u_phi\n")
        if (os.path.isdir(dir + "/images") == 0):
            os.makedirs(dir + "/images")
        if (os.path.isdir(dir + "/visit") == 0):
            os.makedirs(dir + "/visit")
        if (os.path.isdir(dir + "/backup") == 0):
            os.makedirs(dir + "/backup")
        if (os.path.isdir(dir + "/RT") == 0):
            os.makedirs(dir + "/RT")
        if (os.path.isdir(dir + "/backup/gdumps") == 0):
            os.makedirs(dir + "/backup/gdumps")
        else:
            if(downscale_files == 1):
                os.system("rm " + dir +"/backup/gdumps/*")
    dir_images = dir + "/images"
    if (setmpi == 1):
        comm.barrier()
    set_metric=0
    count=0
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            fin = open("dumps%d/parameters" % i2, "rb")
            t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
            n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            fin.close()
            if(1):
                count+=1
    dumps_per_node=int(count/numtasks)
    if(count%numtasks!=0):
        dumps_per_node+=1
    count=0
    import pp_c

    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            fin = open("dumps%d/parameters" % i2, "rb")
            t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
            n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            fin.close()
            if(1):
                count+=1
                if(rank==(count-1)//dumps_per_node):
                    rblock_new(i2)
                    rpar_new(i2)
                    if(flag_restore):
                        restore_dump(dir, i2)
                    if(downscale_files==1):
                        rgdump_new(dir)
                        rdump_new(dir, i2)
                        downscale(dir, i2)
                        #griddataall()
                    else:
                        if(do_unigrid==1):
                            rgdump_griddata(dir)
                            rdump_griddata(dir, i2)
                        else:
                            gdump_new(dir)
                            rdump_new(dir, i2)
                        if(kerr_schild):
                            set_uniform_grid()
                    if(1):
                        misc_calc(calc_bu=1, calc_bsq=1, calc_esq=RESISTIVE)
                        if (export_raytracing_BHOSS==1):
                            dump_RT_BHOSS(dir, i2)
                        if (export_raytracing_BHOSS == 1 or export_raytracing_RAZIEH == 1 or print_angles or (print_images==1 and bs3new>10)):
                            angle_tilt_disk, angle_prec_disk, angle_tilt_corona, angle_prec_corona, angle_tilt_disk_avg, angle_prec_disk_avg, angle_tilt_corona_avg, angle_prec_corona_avg = pp_c.calc_precesion_accurate_disk_c(r, h, ph, rho, ug, uu, B, dxdxp, gcov, gcon, gdet, 1, tilt_angle, nb, bs1new,                                                                                                                                                                                                    bs2new, bs3new, gam, axisym)
                            temp_tilt = np.nan_to_num(angle_tilt_disk[0])
                            temp_prec = np.nan_to_num(angle_prec_disk[0])
                        else:
                            temp_tilt = 0.0
                            temp_prec = 0.0

                        if (export_raytracing_RAZIEH == 1):
                            dump_RT_RAZIEH(dir, i2, temp_tilt, temp_prec, advanced=1)

                        #set_pole()
                        if (export_visit == 1):
                            #createRGrids(dir, i2, 100)
                            dump_visit(dir,i2, 80)

                        #Set initial values
                        cell = 0
                        while (r[0, cell, 0, 0] < (1. + np.sqrt(1. - a * a))):
                            cell += 1

                        if(print_angles):
                            t1 = threading.Thread(target=calc_Mdot, args=())
                            t2 = threading.Thread(target=calc_Edot, args=())
                            t3 = threading.Thread(target=calc_phibh, args=())
                            t4 = threading.Thread(target=calc_rad_avg, args=())
                            t5 = threading.Thread(target=calc_Ldot, args=())
                            t1.start(), t2.start(), t3.start(), t4.start(),t5.start()
                            t1.join(),t2.join(),t3.join(),t4.join(),t5.join()

                            #Calculate luminosity function as in EHT code comparison
                            lum=0
                            #calc_lum()

                            z=0
                            t1 = threading.Thread(target=cool_disk, args=(DISK_THICKNESS,150))
                            t2 = threading.Thread(target=calc_jet_tot, args=())
                            t3 = threading.Thread(target=calc_jet, args=())
                            t4 = threading.Thread(target=set_tilted_arrays, args=(temp_tilt, temp_prec))
                            t1.start(), t2.start(), t3.start(), t4.start()
                            t1.join(), t2.join(), t3.join(), t4.join()

                            t1 = threading.Thread(target=calc_PrecPeriod,args=(temp_tilt,))
                            t2 = threading.Thread(target=calc_scaleheight,args=(temp_tilt, temp_prec, cutoff))
                            t3 = threading.Thread(target=calc_alpha, args=(cutoff,))
                            t4 = threading.Thread(target=psicalc, args=(temp_tilt, temp_prec))
                            t5 = threading.Thread(target=calc_profiles,args=(cutoff,)) #make sure Q is ready
                            t1.start(), t2.start(), t3.start(), t4.start(), t5.start()
                            t1.join(), t2.join(), t3.join(), t4.join(), t5.join()

                            f.write("%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n" % (t, phibh[0, cell], Mdot[0, cell], Edot[0, cell],Edotj[0, cell], Ldot[0, cell], 0.0, precperiod[0], angle_tilt_disk_avg[0],angle_prec_disk_avg[0], angle_tilt_corona_avg[0], angle_prec_corona_avg[0], tilt_angle_jet[0], prec_angle_jet[0],tilt_angle_jet[1], prec_angle_jet[1], rad_avg[0], Rdot.min()))
                            for g in range(0,bs1new):
                               f_rad.write("%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g  %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n"
                                           % (t, r[0,g,:,:].max(), aphi[0,g].max(), Mdot[0,g],Edot[0, g],Edotj[0, g], Ldot[0, g], alpha_r[0,g],alpha_b[0,g],alpha_eff[0,g], H_over_R1[0,g],H_over_R2[0,g],rho_avg[0,g],pgas_avg[0,g],
                                              pb_avg[0,g],Q_avg1_1[0,g],Q_avg1_2[0,g],Q_avg1_3[0,g],Q_avg2_1[0,g],Q_avg2_2[0,g],Q_avg2_3[0,g], pitch_avg[0,g], angle_tilt_disk[0,g], angle_prec_disk[0,g],angle_tilt_corona[0,g], angle_prec_corona[0,g],
                                              angle_jetEuu_up[0,g], angle_jetEuu_up[1,g],angle_jetEuu_up[2,g], angle_jetEuu_down[0,g],angle_jetEuu_down[1,g],angle_jetEuu_down[2,g], Rdot[g], sigma_Ju[0,g], gamma_Ju[0,g], E_Ju[0,g], mass_Ju[0,g], temp_Ju[0,g], sigma_Jd[0,g], gamma_Jd[0,g], E_Jd[0,g], mass_Jd[0,g], temp_Jd[0,g]))

                        if(print_but):
                            t1 = threading.Thread(target=print_butterfly, args=(f_but, 5, z))
                            t2 = threading.Thread(target=print_butterfly, args=(f_but, 10, z))
                            t3 = threading.Thread(target=print_butterfly, args=(f_but, 20, z))
                            t4 = threading.Thread(target=print_butterfly, args=(f_but, 40, z))
                            t1.start(), t2.start(), t3.start(), t4.start()
                            t1.join(), t2.join(), t3.join(), t4.join()

                        if (print_images):
                            z = 0
                            t1=threading.Thread(target=preset_transform_scalar, args=(temp_tilt, temp_prec))
                            t1.start()
                            plc_cart(rho, -8.0, 1.0,  10, z, dir_images + "/rho%d.png" % i2, r"log$(\rho)$ at %d $R_g/c$" % t)
                            if(RAD_M1):
                                plc_cart(E_rad / rho, -8, 2, 10, 0, dir_images + "/erad_o_rho%d.png" % i2, r"log$(E_{rad}/\rho)$ at %d $r_g/c$" % t)
                                plc_cart(ug / E_rad, -9, 3, 10, 0, dir_images + "/ug_o_erad%d.png" % i2, r"log$(u_{g}/E_{rad})$ at %d $r_g/c$" % t)
                                plc_cart(np.abs(((gam - 1.0) * ug + (1.0 / 3.0) * (E_rad)) / (bsq * 0.5)), -2, 4, 10, 0, dir_images + "/beta%d.png" % i2, r"log$(\beta)$ at %d $r_g/c$" % t)
                            #plc_cart(bsq / rho, -8, 2, 20, z, dir_images + "/bsq%d.png" % i2, r"log$(b^{2}/\rho)$ at %d $R_g/c$" % t)
                            #plc_cart(ug / rho, -8, 2.2, 20, z, dir_images + "/ug%d.png" % i2, r"log$(u_{g}/\rho)$ at %d $R_g/c$" % t)
                            #plc_cart((gam - 1) * 2 * ug / bsq, -2, 4.0, 20, z, dir_images + "/beta%d.png" % i2, r"log$(\beta)$ at %d $R_g/c$" % t)
                            t1.join()

                            #Very crude method of projecting onto midplane: Just shifts index
                            if(bs3new>10):
                                var2=transform_scalar(rho)
                                preset_project_vertical(var2)
                                plc_cart_xy1(rho, -8.0, 2.2, 20, z,1, dir_images + "/rhoxy%d.png" % i2, r"log$(\rho)$ at %d $R_g/c$" % t)
                                plc_cart_xy1(bsq / rho, -8, 2, 20, z,1, dir_images + "/bsqxy%d.png" % i2, r"log$(b^{2}/\rho)$ at %d $R_g/c$" % t)
                                plc_cart_xy1(ug / rho, -8, 2.2, 20, z,1, dir_images + "/ugxy%d.png" % i2, r"log$(u_{g}/\rho)$ at %d $R_g/c$" % t)
                                plc_cart_xy1((gam - 1) * 2 * ug / bsq, -2, 4.0, 20, z,1, dir_images + "/betaxy%d.png" % i2, r"log$(\beta)$ at %d $R_g/c$" % t)

                    if (rank == 0):
                        print("Post processed %d \n" % i2)
    if (print_angles):
        f.close()
        f_rad.close()
    if (print_but):
        f_but.close()
    if (setmpi == 1):
        comm.barrier()
    if (rank == 0):
        if (print_angles):
            print("Merging post processed files and cleaning up")
            f_tot = open(dir + "/post_process.txt", "wb")
            f_tot_rad = open(dir + "/post_process_rad.txt", "wb")
            for i in range(0,numtasks):
                shutil.copyfileobj(open(dir +"/post_process%d.txt" %i,'rb'), f_tot)
                os.remove(dir + "/post_process%d.txt" %i)
                shutil.copyfileobj(open(dir +"/post_process_rad%d.txt" %i,'rb'), f_tot_rad)
                os.remove(dir + "/post_process_rad%d.txt" %i)
            f_tot.close()
            f_tot_rad.close()
        if (print_but):
            f_tot_but = open(dir + "/post_process_but.txt", "wb")
            for i in range(0, numtasks):
                shutil.copyfileobj(open(dir + "/post_process_but%d.txt" % i, 'rb'), f_tot_but)
                os.remove(dir + "/post_process_but%d.txt" % i)
            f_tot_but.close()

if __name__ == "__main__":
    pass
    #dirr = "G:\\G\\HAMR\\RHAMR_CUDA3\\RHAMR\\RHAMR_CPU"
    #dirr = "/home/nkaaz"
    #dirr = "/gpfs/alpine/phy129/proj-shared/THIN_SHELLEY_TRANSFER/reduced"
    #post_process(dirr, 0,156,1)
