# coding: utf-8
# python pp.py build_ext --inplace
# In[21]:
# from __future__ import division__future__ import division
# from IPython.display import display

import os, sys, gc
import shutil

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

# only attempt to switch backend if outside of the jupyter notebook
if mpl.get_backend() != "module://ipykernel.pylab.backend_inline":
    mpl.use('Agg')
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
    '''
    Description: 
        For a given dump file, this loads the parameters file.
        The parameters file contains global information at this dump,
        i.e., current time, current timestep, cell widths, etc. 
    Parameters:
        dump         :   int     :   num. of dump file to read
    '''
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, fractheta,REF_1,REF_2,REF_3, RAD_M1
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, NODE, TIMELEVEL,flag_restore,r1,r2,r3

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
    _dx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r1
    _dx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r2
    _dx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r3

    tf = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    fractheta = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    for n in range(0,13):
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    if(trash==10):
        RAD_M1=1
    else:
        RAD_M1=0
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

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
    fin.close()

def rpar_old(dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, fractheta,REF_1,REF_2,REF_3
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, RAD_M1
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
    fractheta = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]

    for i in range(0, 15):
        temp_array[i] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    nb = n_active_total
    rhor = 1 + (1 - a ** 2) ** 0.5

    gd1_temp = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    gd2_temp = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    for n in range(0, nmax):
        block[n, AMR_REFINED] = gd1_temp[n]
        block[n, AMR_ACTIVE] = gd2_temp[n]

    i = 0;
    for n in range(0, nmax):
        if block[n, AMR_ACTIVE] == 1:
            n_ord[i] = n
            i += 1
    fin.close()

    RAD_M1=0
    if((nb2==6) or nb2==12 or nb2==24 or nb==48 or nb2==96):
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
    global tf, a, gam, cour, Rin, Rout, R0, fractheta, RAD_M1, NODE, TIMELEVEL
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
    fractheta.tofile(fin)
    for n in range(0, 13):
        np.int32(trash).tofile(fin)
    if (RAD_M1 == 1):
        trash=10
        np.int32(trash).tofile(fin)
    else:
        trash=0
        np.int32(trash).tofile(fin)
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
    '''
    Description: 
        I don't know, Matthew doing some weird shit. 
    Parameters:
        dump         :   int     :   num. of dump file to read
        dir          :   str     :   Location of dump files
    '''
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
        print(NODE[n],n_active_total_node[NODE[n]])
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
    '''
    Description: 
        For a given dump file, this loads the relevant grid data. I am unclear on the structure of gdumps. 
    Parameters:
        dump         :   int     :   num. of dump file to read
    '''
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6
    global AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12
    global block, nmax, n_ord

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
    for n in range(0, nmax):
        if block[n, AMR_ACTIVE] == 1:
            n_ord[i] = n
            i += 1

    fin.close()

def rgdump_new(dir):
    '''
    Description: 
        For a given dump file, this 
    Parameters:
        dir          :   str     :   Location of dump files
    '''
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
    global nb1, nb2, nb3
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

    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8):
        flag=1
    else:
        flag=0

    pp_c.rgdump_griddata(flag, interpolate_var, dir, axisym, n_ord,lowres1, lowres2, lowres3 ,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet,block, nb1, nb2, nb3, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

def rgdump_write(dir):
    '''
    Description: 
        For a given dump file, this 
    Parameters:
        dir          :   str     :   Location of dump files
    '''
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3
    import pp_c
    f1 = int(lowres1)
    f2 = int(lowres2)
    f3 = int(lowres3)
    pp_c.rgdump_write(0, dir +"/backup", axisym, n_ord,f1,f2,f3,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet)

def rdump_new(dir, dump):
    '''
    Description: 
        For a given dump file, this 
    Parameters:
        dump         :   int     :   num. of dump file to read
        dir          :   str     :   Location of dump files
    '''
    global rho, ug, uu,uu_rad, E_rad, RAD_M1, B, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3
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
    if(os.path.isfile("dumps%d/new_dump" %dump)):
        flag=1
    else:
        flag=0
    pp_c.rdump_new(flag, RAD_M1, dir, dump, n_active_total, lowres1, lowres2, lowres3,nb,bs1,bs2,bs3, rho,ug, uu, B, E_rad, uu_rad,gcov,gcon,axisym)

    _dx1 = _dx1 * lowres1
    _dx2 = _dx2 * lowres2
    _dx3 = _dx3 * lowres3

def rdump_griddata(dir, dump):
    global rho, ug, uu,uu_rad, E_rad, RAD_M1, B, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3, nb, nb1, nb2, nb3, n_ord, interpolate_var
    import pp_c

    ACTIVE1 = np.max(block[n_ord, AMR_LEVEL1])
    ACTIVE2 = np.max(block[n_ord, AMR_LEVEL2])
    ACTIVE3 = np.max(block[n_ord, AMR_LEVEL3])
    gridsizex1 = int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1/lowres1)
    gridsizex2 = int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2/lowres2)
    gridsizex3 = int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3/lowres3)

    if ((int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1) % lowres1) != 0 or (int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2) % lowres2) != 0 or (int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3) % lowres3) != 0):
        print("Incompatible lowres settings in rdump_griddata")

    # Allocate memory
    rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    if(RAD_M1):
        E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
        uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order='C')
    else:
        E_rad=np.copy(ug)
        uu_rad=np.copy(uu)
    if (os.path.isfile("dumps%d/new_dump" % dump)):
        flag = 1
    else:
        flag = 0
    pp_c.rdump_griddata(flag, interpolate_var, np.int32(RAD_M1), dir, dump, n_active_total, lowres1, lowres2, lowres3, nb,bs1,bs2,bs3, rho,ug, uu, B, E_rad, uu_rad, gcov,gcon,axisym,  n_ord,block, nb1,nb2,nb3, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3
    _dx1 = _dx1 * lowres1 * (1.0 / (1.0 + REF_1) ** ACTIVE1)
    _dx2 = _dx2 * lowres2 * (1.0 / (1.0 + REF_2) ** ACTIVE2)
    _dx3 = _dx3 * lowres3 * (1.0 / (1.0 + REF_3) ** ACTIVE3)
    nb2d = nb
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1

def rdump_write(dir, dump):
    global rho, ug, uu, B,uu_rad, E_rad,gcov, gcov,axisym, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, export_visit
    import pp_c
    if (os.path.isdir(dir + "/backup/dumps%d" %dump) == 0):
        os.makedirs(dir + "/backup/dumps%d" %dump)
    pp_c.rdump_write(0, RAD_M1, dir+"/backup", dump, n_active_total, lowres1, lowres2, lowres3,nb,bs1,bs2,bs3, rho,ug, uu, B, E_rad, uu_rad, gcov,gcon,axisym)

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
    global rho, ug, uu, B, gcov, gcov,axisym, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, export_visit,block, n_ord, nb1, nb2, nb3
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    import pp_c

    pp_c.griddata3D(nb, bs1new, bs2new, bs3new, nb1,nb2,nb3, n_ord, block, input, output, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

def griddata_2D(input, output, inter=1):
    global rho, ug, uu, B, gcov, gcov, axisym, nb2d, bs1, bs2, bs3, bs1new, bs2new, bs3new, lowres1, lowres2, lowres3, export_visit, block, n_ord, nb1, nb2, nb3
    global AMR_ACTIVE, AMR_LEVEL, AMR_LEVEL1, AMR_LEVEL2, AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    import pp_c

    pp_c.griddata2D(nb, bs1new, bs2new, bs3new, nb1,nb2,nb3, n_ord, block, input, output, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]))

def griddataall():
    global block, n_ord, rho, uu, uu_rad, E_rad, RAD_M1, ud, bu, bd, B, ug, dxdxp, bsq, r, h, ph, nb, nb1, nb2, nb3, bs1, bs2, bs3, alpha, gcon, gcov, gdet, REF_1, REF_2, REF_3,interpolate_var
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
    global bsq, rho, ug, uu, uu_rad, E_rad, RAD_M1

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
    global rho, gdet, bs1new, h, ph, H_over_R1, H_over_R2,h_new
    X = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    tilt_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.float32)
    prec_tmp = np.zeros((nb, bs1new, 1, 1), dtype=np.int32)
    H_over_R1 = np.zeros((nb, bs1new))
    h_avg = np.zeros((nb, bs1new, 1, 1))
    ph_new = np.copy(ph)
    h_new = np.copy(h)
    uu_proj = project_vector(uu)
    tilt_tmp[0, :, 0, 0] = tilt / 180.0 * np.pi
    prec_tmp[0, :, 0, 0] = np.int32(prec / 360.0 * bs3new)

    for i in range(0, bs1new):
        ph_new[0, i] = np.roll(ph[0, i], prec_tmp[0, i,0,0], axis=1)

    sph_to_cart2(X, h_new, ph_new)
    rotate_coord(X, -tilt_tmp)
    h_new, ph_new = cart_to_sph(X)

    norm = (rho * (rho > cutoff) * gdet).sum(-1).sum(-1)
    h_avg[:, :, 0, 0] = (rho * (rho > cutoff) * gdet * h_new).sum(-1).sum(-1) / norm
    H_over_R1 = (rho * (rho > cutoff) * np.abs(h_new - np.pi / 2) * gdet).sum(-1).sum(-1) / norm
    cs_avg=(gdet*(rho>cutoff)*rho**2.0*np.sqrt(2.0/np.pi*(gam-1)*ug/(rho+gam*ug))).sum(-1).sum(-1)
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

def project_vector(vector):
    '''
    NK: Appears to project vectors into the tilted frame.  
    '''
    global phi_to_theta, phi_to_phi, theta_to_theta, theta_to_phi
    vector_proj = np.copy(vector)
    vector_proj[1] = vector[1] * np.sqrt(gcov[1, 1])
    vector_proj[2] = vector[2] * np.sqrt(gcov[2, 2]) * theta_to_theta + vector[3] * np.sqrt(gcov[3, 3]) * phi_to_theta
    vector_proj[3] = vector[2] * np.sqrt(gcov[2, 2]) * theta_to_phi + vector[3] * np.sqrt(gcov[3, 3]) * phi_to_phi

    return vector_proj

def project_vertical(input_var):
    global bs1new, bs2new, x2, bs3new, offset_x2
    output_var = np.copy(input_var)

    for i in range(0, bs1new):
        for z in range(0, bs3new):
            output_var[0, i, :, z] = np.roll(input_var[0, i, :, z], np.int32(offset_x2[0, i, 0, z]), axis=0)
    return output_var

def preset_project_vertical():
    global gdet, bs1new, bs2new, x2, bs3new, offset_x2
    x2_avg = np.zeros((nb, bs1new, 1, bs3new), dtype=np.float32)

    norm = (rho * gdet).sum(2)
    x2_avg[:, :, 0, :] = (rho * gdet * x2).sum(2) / norm
    offset_x2 = -x2_avg / 2.0 * bs2new

def misc_calc(calc_bu=1, calc_bsq=1):
    global bu, bsq, bs1new,bs2new,bs3new,nb,uu,B,gcov, axisym, lum, Ldot, rad_avg
    import pp_c
    if(calc_bu==1):
        bu=np.copy(uu)
    else:
        bu=np.zeros((1, 1, 1, 1, 1), dtype=rho.dtype)
    if (calc_bsq == 1):
        bsq=np.copy(rho)
    else:
        bsq=np.zeros((1, 1, 1, 1), dtype=rho.dtype)
    pp_c.misc_calc(bs1new, bs2new, bs3new, nb,axisym,uu, B, bu, gcov, bsq, calc_bu, calc_bsq)

def Tcalcud_new(kapa,nu):
    global gam
    bd_nu = (gcov[nu,:]*bu).sum(0)
    ud_nu = (gcov[nu,:]*uu).sum(0)
    Tud= bsq * uu[kapa] * ud_nu + 0.5 * bsq * (kapa==nu) - bu[kapa] * bd_nu +(rho + ug + (gam - 1) * ug) * uu[kapa] * ud_nu + (gam - 1) * ug * (kapa==nu)
    return Tud

# Matrix inversion
def invert_matrix():
    global dxdxp_inv, dxdr_inv, axisym
    if(axisym):
        dxdxp_inv = np.zeros((4, 4, nb, bs1new, bs2new), dtype=np.float32, order='C')
        dxdr_inv = np.zeros((4, 4, nb, bs1new, bs2new), dtype=np.float32, order='C')
        for i in range(0, bs1new):
            for j in range(0, bs2new):
                dxdxp_inv[:, :, 0, i, j] = np.linalg.inv(dxdxp[:, :, 0, i, j])
                dxdr_inv[:, :, 0, i, j] = np.linalg.inv(dxdr[:, :, 0, i, j])
    else:
        dxdxp_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
        dxdr_inv = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
        for i in range(0, bs1new):
            for j in range(0, bs2new):
                for k in range(0, bs3new):
                    dxdxp_inv[:, :, 0, i, j, k] = np.linalg.inv(dxdxp[:, :, 0, i, j, k])
                    dxdr_inv[:, :, 0, i, j, k] = np.linalg.inv(dxdr[:, :, 0, i, j, k])

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
    angle_jet_var_down[0] = np.arccos(np.abs(var_flux_cart_down[0] * J_BH[1] + var_flux_cart_down[1] * J_BH[2] + var_flux_cart_down[2] * J_BH[3]) / rlength) * 180 / 3.14
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
    cs = np.sqrt(gam * (gam - 1) * ug / (rho + ug + (gam - 1) * ug))

    v_r = uu_proj[1]
    v_or = uu_proj[3]
    alpha_eff = (fact*v_r * v_or/uu[0]/uu[0]).sum(-1).sum(-1) / (fact*(cs ** 2)).sum(-1).sum(-1)

    pitch_avg = (fact * np.sqrt(bu_proj[1] * bu_proj[1] + bu_proj[2] * bu_proj[2])).sum(-1).sum(-1) / (fact * np.sqrt(bu_proj[3] * bu_proj[3])).sum(-1).sum(-1)

# NK: Calculate viscous stresses G_l, G_m, G_n in radially dependent warped shearing coordinates l,m,n. 
def calc_viscous_stresses(cutoff,tilt,prec):
    '''
    Description: 
        Here, we calculate the components of the stress tensor in the local frame of the warped annuli. We weight them by density, and use a cutoff to only select the minimum density. 

        Consider a curvilinear cell; there is a phi, theta, and radial face. I will assume that (i) there is no angular momentum flux through the phi cell, (ii), magnetic winds extract angular momentum
        from the theta face, and (iii), viscous stresses communicate angular momentum across the radial faces. 

        Here, we want to calculate the stresses across the radial faces. Then, we can relate these to the Q1, Q2, Q3 via similar methods to Ogilvie+Latter 2013. This is done as follows: We consider
        the orthogonal vectors vec{l}, d vec{l} / dr, and vec{l} x d vec{l} / dr, which we will call "l, m, and n". Here, l is the angular momentum unit vector of a tilted annulus, and varies as a                function of radius for a warped disk. It also depends on phase, the l,m,n basis is *not* curvilinear. So, what we want is to calculate the angular momentum flux in the l,m,n basis and transported         in the radial direction.

        Lets imagine rotating to the frame of a tilted annulus. Then we have Cartesian coordinates x', y', z'; in this frame, z' = 0. In this frame, \hat{l} ~ \hat{z'}, and we can choose \hat{m} ~ \hat{x'}         such that \hat{n} = \hat{l} x \hat{m} = \hat{y'}. 

        So, we can derive the angular momentum flux transported radially and convert them to them from that r,theta,phi basis to the x,y,z basis.


        Calculating the stresses: 

        From HARM paper, the total MHD stress tensor in relativity is:
        T_{ij} ~ (rho + internal energy + pressure + b^2)*u^i u^j + (p + 0.5*b^2) * g^(ij) - b^i *b^j
        For simplicity, we can take the non-relativistic 3x3 components of this to start. If we consider a curvilinear cell, we want just the angular momentum flux for now. First, we get the momentum flux through the cell,

        F = T_{ir} * \hat{r} = ( T_{rr} \hat{r} + T_{hr} \hat{h} + T_{pr} \hat{p} )

        Then, to get the angular momentum flux we simply need to take the cross product,

        N = r x p = r * \hat{r} x ( T_{rr} \hat{r} + T_{hr} \hat{h} + T_{pr} \hat{p} )

        or, 
        
        N = r * (T_{hr} \hat{p} - T_{pr} \hat{h} ) 

    Parameters:
        cutoff  :   float   :   density cutoff for averages
    '''
    global G_l, G_m, G_n

    # empty position 4-vector. We will turn this into Cartesian 4-vector in tilted frame. 
    X = np.zeros((4, nb, bs1new, bs2new, bs3new), dtype=np.float32)
    sph_to_cart2(X, h, ph)
    X_proj = project_vector(X)
    x_proj = X_proj[1]
    y_proj = X_proj[2]
    z_proj = X_proj[3]
    r_proj = (x_proj**2. + y_proj**2. + z_proj**2.)**0.5
    s_proj = (x_proj**2. + y_proj**2.)**0.5

    # get sine / cosine of phi/theta in radially dependent tilted frame. 
    sinph  = y_proj/s_proj
    cosph  = x_proj/s_proj
    sinth  = s_proj/r_proj
    costh  = z_proj/r_proj

    # To average density, need a normalization factor
    norm=(gdet*rho*(rho>cutoff)).sum(-1).sum(-1)
    # Here is the density weight
    fact=(gdet*rho*(rho>cutoff)) 
    
    # Project vector takes our B field and velocity 4-vectors and puts them in spherical coordinates with respect to midframe of tilted annulus.
    bu_proj = project_vector(bu)
    uu_proj = project_vector(uu)
    # NK: ug is energy density, not specific internal energy? If so pressure = (gam -1)*ug. 

    # G_p ~ r * T_hr
    G_p = fact*r*( ( rho + ug + (gam - 1)*ug + bsq)*uu_proj[2]*uu_proj[1] - bu_proj[2]*bu_proj[1] )
    # G_h ~ r * T_pr
    G_h = -1. *fact*r*( ( rho + ug + (gam - 1)*ug + bsq)*uu_proj[3]*uu_proj[1] - bu_proj[3]*bu_proj[1] )
  
    ## Calculate G_l,m,n. After calculating, we can average them. 
    # G_l = - G_h * sin(theta)
    G_l = (-G_h*sinth).sum(-1).sum(-1)/norm
    # G_m = G_h*cos(phi)*cos(theta) -  G_p*sin(phi)
    G_m = (G_h*cosph*costh - G_p*sinph).sum(-1).sum(-1)/norm
    # G_n = G_h*sin(phi)*cos(theta) + G_p*cos(phi) 
    G_n = (G_h*sinph*costh + G_p*cosph).sum(-1).sum(-1)/norm

# Print total mass of disk in code units
def calc_Mtot():
    global Mtot
    Mtot = np.sum((rho * uu[0]) * _dx1 * _dx2 * _dx3 * gdet)

# Calculate precession period
def calc_PrecPeriod(angle_tilt):
    global gam,a,precperiod
    #v_phi = 1 / (2 * np.pi) * 1 / (r ** 1.5 + a * r / r)
    #v_nod = v_phi * (r / r - np.sqrt(r / r - 4 * a / r ** 1.5 + 3 * a ** 2 / r ** 2))
    #T_phi = 1 / v_phi
    #T_nod = 1 / v_nod
    #T_nod = np.nan_to_num(T_nod)
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
    norm3 = (gdet * np.sqrt(rho * bsq) * (rho > cutoff)).sum(-1).sum(-1)
    fact1 = gdet * rho * (rho > cutoff)
    fact2 = gdet * np.sqrt(rho * bsq) * (rho > cutoff)
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
    global gcov_kerr, gcon_kerr, x1, x2, x3, r, h, ph, bs1new, bs2new, bs3new, rank, startx1, startx2, startx3, _dx1, _dx2, _dx3

    if(rank==0):
        print("Are you sure about using this function. Do not use for warped grids deviating from Kerr-Schild!")

    for i in range(0, bs1new):
        x1[:, i, :, :] = startx1 + i * _dx1
    for j in range(0, bs2new):
        x2[:, :, j, :] = startx2 + j * _dx2
    for z in range(0, bs3new):
        x3[:, :, :, z] = startx3 + z * _dx3

    r = np.log(x1)
    h = (x2+1)/2.0*np.pi
    ph = x3

# Calculate uniform coordinates and Kerr-Schild metric for Ray-Tracing
def set_KS():
    global gcov_kerr, gcon_kerr, x1, x2, x3, r, h, ph, bs1new, bs2new, bs3new
    # Set covariant Kerr metric in double
    gcov_kerr = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float64)

    set_uniform_grid()

    cth = np.cos(h)
    sth = np.sin(h)
    s2 = sth * sth;
    rho2 = r * r + a * a * cth * cth;

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
    gcon_kerr = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float64)
    for i in range(0, bs1new):
        for j in range(0, bs2new):
            gcon_kerr[:, :, 0, i, j, 0] = np.linalg.inv(gcov_kerr[:, :, 0, i, j, 0])

# Make file for raytracing
def dump_RT(dir, dump):
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

def plc_cart(var, min, max, rmax, offset, name, label,dolog=True):
    global aphi, r, h, ph, print_fieldlines,notebook
    fig = plt.figure(figsize=(64, 32))

    X = r*np.sin(h)
    Y = r*np.cos(h)
    if(nb==1):
        X[:,:,0]=0.0*X[:,:,0]
        X[:,:,bs2new-1]=0.0*X[:,:,bs2new-1]

    plotmax = int(10*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2)*plotmax:
            ilim = i
            break

    plt.subplot(1, 2, 1)
    if dolog:
        plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        res = plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    else:
        plc_new(var[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        res = plc_new(var[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(-800, 800, 0.05), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax, ymax=rmax)
        plc_new(aphi[:, 0:ilim], levels=np.arange(-800, 800, 0.05), cb=0,colors="black", isfilled=0, xcoord=-1 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax, ymax=rmax)
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
    if dolog:
        plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 10, ymax=rmax * 10)
        res = plc_new(np.log10((var))[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * 10, ymax=rmax * 10)
    else:
        plc_new(var[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 10, ymax=rmax * 10)
        res = plc_new(var[:, 0:ilim], levels=np.arange(min, max, (max-min)/300.0), cb=0, isfilled=1, xcoord=-1 * X[:, 0:ilim],ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * 10, ymax=rmax * 10)
    if (print_fieldlines == 1):
        plc_new(aphi[:, 0:ilim], levels=np.arange(-800, 800, 0.05), cb=0,colors="black", isfilled=0, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 10, ymax=rmax * 10)
        plc_new(aphi[:, 0:ilim], levels=np.arange(-800, 800, 0.05), cb=0,colors="black", isfilled=0, xcoord=-1 * X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=180 + offset, xmax=rmax * 10, ymax=rmax * 10)

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
    plt.savefig(name, dpi=40) # was dpi=400
    if (notebook==0):
        plt.close()

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
    plt.savefig("grid.png", dpi=150)
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
            res = ax.contourf(xcoord[0, :, int(bs2new // 2 + 1), :], ycoord[0, :, int(bs2new // 2), :],myvar[0, :, int(bs2new // 2 + 1), :], nc, extend='both', **kwargs)
        else:
            res = ax.contour(xcoord[0, :, int(bs2new // 2 + 1), :], ycoord[0, :, int(bs2new // 2), :], myvar[0, :, int(bs2new // 2 + 1), :],nc, extend='both', **kwargs)
    if (cb == True):  # use color bar
        plt.colorbar(res, ax=ax)
    if (xy == 1):
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
    return res

def transform_scalar_tot(input, tilt, prec):
    preset_transform_scalar(tilt, prec)
    output=transform_scalar(input)
    return output

def preset_transform_scalar(tilt, prec):
    global ti,tj,tk
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
    tj = (((h_new[0] - h[0]) / (h.max() - h.min()) * (bs2new - 1)) + tj) % bs2new
    tk = (((ph_new[0] - ph[0]) / (ph.max() - ph.min()) * (bs3new - 1)) + tk) % bs3new

def transform_scalar(input):
    global ti,tj,tk
    output = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32)

    output[0] = ndimage.map_coordinates(input[0], [[ti], [tj], [tk]], order=0, mode='nearest')
    return output

def plc_cart_xy1(var, min, max, rmax, offset, transform, name, label):
    fig = plt.figure(figsize=(64, 32))

    X = np.multiply(r, np.sin(ph))
    Y = np.multiply(r, np.cos(ph))
    if(transform==1):
        var2=transform_scalar(var)
    else:
        var2=var
    plotmax = int(10*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2.0)*plotmax:
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

import time
from multiprocessing import Process
def post_process(dir, dump_start, dump_end, dump_stride,savedir=""):
    '''
    Parameters:
        dir         :   str     :   location of dumps
        dump_start  :   int     :   first dump to load in post process
        dump_end    :   int     :   last  dump to load in post process
        dump_stride :   int     :   frequency of dumps loaded (1 ~ every dump in range is loaded
        savedir     :   str     :   location to put post processed files
    '''

    global axisym, lowres1,lowres2,lowres3, REF_1, REF_2, REF_3, set_cart, set_xc,tilt_angle, _dx1,_dx2, _dx3, Mdot, Edot,Ldot,phibh, rad_avg,H_over_R1, H_over_R2, interpolate_var, rad_avg,Rdot,lum
    global alpha_r, alpha_b, alpha_eff, pitch_avg, aphi, export_visit, print_fieldlines, setmpi
    global sigma_Ju, gamma_Ju, E_Ju ,mass_Ju,temp_Ju
    global sigma_Jd, gamma_Jd, E_Jd, mass_Jd, temp_Jd # NK: Fixed typo, this was "temp_Ju" again.
    global comm, numtasks, rank,notebook, RAD_M1
    global pgas_avg, rho_avg, pb_avg, Q_avg1_1,Q_avg1_2,Q_avg1_3, Q_avg2_1,Q_avg2_2,Q_avg2_3,flag_restore, r1, r2, r3
    global G_l, G_m, G_n # NK: Viscous stresses in warped shearing coordinates: l, m, n, which vary as a function of radius. 


    '''
    Globals defined:
        axisym              :
        lowres1,2,3         :
        REF_1,2,3           :
        set_cart            :
        set_xc              :
        tilt_angle          :
        _dx1,2,3            :
        Mdot                :
        Edot                :
        Ldot                :
        phibh               :
        rad_avg             :
        H_over_R1           :
        H_over_R2           :
        interpolate_var     :
        rad_avg             :
        Rdot                :
        lum                 :
        alpha_r             :
        alpha_b             :
        alpha_eff           :
        pitch_avg           :
        export_visit        :
        print_fieldlines    :
        setmpi              :
        sigma_Ju            :
        gamma_Ju            :
        E_Ju                :
        mass_Ju             :
        temp_Ju             :
        sigma_Jd            :
        gamma_Jd            :
        E_Jd                :
        mass_Jd             :
        temp_Jd             :
        comm                :
        numtasks            :
        rank                :
        notebook            :
        RAD_M1              :
        pgas_avg            :
        rho_avg             :
        pb_avg              :
        Q_avg1_1,2,3        :
        Q_avg2_1,2,3        :
        flag_restore        :
        r1,2,3              :
        G_l,m,n             : Viscous stresses in warped shearing coordinates l,m,n
    '''

    r1=4
    r2=4
    r3=4
    lowres1 = 8
    lowres2 = 2
    lowres3 = 4
    axisym = 1
    notebook=0
    set_mpi(1)
    os.chdir(dir)
    interpolate_var=0
    tilt_angle=0.0 #Need to be on for print_angles and prin_images
    print_angles=1
    print_images=1
    print_fieldlines=0
    print_but=0
    export_visit=0
    export_raytracing=0
    downscale_files=0
    kerr_schild=0 #if coordinates are close to x1=log(r), x2= theta, x3= phi
    DISK_THICKNESS=0.02
    cutoff = 0.00001  # cutoff for density in averaging quantitites

    # NK: Default savedir is just dir
    if (savedir == ""):
        savedir = np.copy(dir)

    # print angles should generally be on
    if (print_angles):
        f = open(savedir + "/post_process%d.txt" % rank, "w")
        f_rad = open(savedir + "/post_process_rad%d.txt" %rank, "w")
    # but = butterfly diagram? 
    if(print_but):
        f_but = open(savedir + "/post_process_but%d.txt" %rank, "w")
    # First MPI rank writes header for all output files
    if (rank == 0):
        if (print_angles):
            # Parameters as a function of time
            f.write("t,phibh, Mdot, Edot, Edotj, Ldot, lambda, prec_period, tilt_disk, prec_disk, tilt_corona, prec_corona,tilt_jet1, prec_jet1, tilt_jet2, prec_jet2, rad_avg, Rdot\n")
            # Parameters as a function of radius
            f_rad.write("t, r, phibh, Mdot, Edot, Edotj, Ldot, alpha_r, alpha_b,alpha_eff, H_o_R_real, H_o_R_thermal, rho_avg, pgas_avg, pb_avg, Q_avg1_1, Q_avg1_2, Q_avg1_3, Q_avg2_1, Q_avg2_2, Q_avg2_3, pitch_avg, tilt_disk, prec_disk, tilt_corona, prec_corona, tilt_jet1, prec_jet1, opening_jet1, tilt_jet2, prec_jet2, opening_jet2, Rdot, sigma_Ju, gamma_Ju, E_Ju, mass_Ju, temp_Ju,sigma_Jd, gamma_Jd, E_Jd, mass_Jd, temp_Jd, G_l, G_m, G_n\n")
        if(print_but):
            f_but.write("t, r, theta, rho, pgas, pb, b_r, b_theta, b_phi, u_r, u_theta, u_phi\n")
        if (os.path.isdir(savedir + "/images") == 0):
            os.makedirs(savedir + "/images")
        if (os.path.isdir(savedir + "/visit") == 0):
            os.makedirs(savedir + "/visit")
        if (os.path.isdir(savedir + "/backup") == 0):
            os.makedirs(savedir + "/backup")
        if (os.path.isdir(savedir + "/RT") == 0):
            os.makedirs(savedir + "/RT")
        if (os.path.isdir(savedir + "/backup/gdumps") == 0):
            os.makedirs(savedir + "/backup/gdumps")
        else:
            if(downscale_files == 1):
                os.system("rm " + savedir +"/backup/gdumps/*")
    dir_images = savedir + "/images"
    # setmpi == 1 means we are using mpi
    if (setmpi == 1):
        comm.barrier()
    # set metric?
    set_metric=0
    # count gets iterated through in the following loop
    count=0
    # iterate through desired dump files
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
            # get current time and block information
            t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
            n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            fin.close()
            # why does count get iterated at 240 * 32 interval?
            if(nstep % (240 * 32) == 0):
                count+=1
    # Split up dumps to read between mpi tasks
    dumps_per_node=int(count/numtasks)
    if(count%numtasks!=0):
        dumps_per_node+=1
    # Reset count to 0 for next loop
    count=0
    import pp_c

    # iterate through desired dump files, but this time split between mpi nodes
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
        if (os.path.isdir(savedir + "/backup/gdumps") == 0):
            os.makedirs(savedir + "/backup/gdumps")
        else:
            if(downscale_files == 1):
                os.system("rm " + savedir +"/backup/gdumps/*")
    dir_images = savedir + "/images"
    # setmpi == 1 means we are using mpi
    if (setmpi == 1):
        comm.barrier()
    # set metric?
    set_metric=0
    # count gets iterated through in the following loop
    count=0
    # iterate through desired dump files
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
            # get current time and block information
            t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
            n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            fin.close()
            # why does count get iterated at 240 * 32 interval?
            if(nstep % (240 * 32) == 0):
                count+=1
    # Split up dumps to read between mpi tasks
    dumps_per_node=int(count/numtasks)
    if(count%numtasks!=0):
        dumps_per_node+=1
    # Reset count to 0 for next loop
    count=0
    import pp_c

    # iterate through desired dump files, but this time split between mpi nodes
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
        if (os.path.isdir(savedir + "/backup/gdumps") == 0):
            os.makedirs(savedir + "/backup/gdumps")
        else:
            if(downscale_files == 1):
                os.system("rm " + savedir +"/backup/gdumps/*")
    dir_images = savedir + "/images"
    # setmpi == 1 means we are using mpi
    if (setmpi == 1):
        comm.barrier()
    # set metric?
    set_metric=0
    # count gets iterated through in the following loop
    count=0
    # iterate through desired dump files
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
            # get current time and block information
            t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
            n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            fin.close()
            # why does count get iterated at 240 * 32 interval?
            if(nstep % (240 * 32) == 0):
                count+=1
    # Split up dumps to read between mpi tasks
    dumps_per_node=int(count/numtasks)
    if(count%numtasks!=0):
        dumps_per_node+=1
    # Reset count to 0 for next loop
    count=0
    import pp_c

    # iterate through desired dump files, but this time split between mpi nodes
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
        if (os.path.isdir(savedir + "/backup/gdumps") == 0):
            os.makedirs(savedir + "/backup/gdumps")
        else:
            if(downscale_files == 1):
                os.system("rm " + savedir +"/backup/gdumps/*")
    dir_images = savedir + "/images"
    # setmpi == 1 means we are using mpi
    if (setmpi == 1):
        comm.barrier()
    # set metric?
    set_metric=0
    # count gets iterated through in the following loop
    count=0
    # iterate through desired dump files
    for i in range(0, (dump_end - dump_start) // dump_stride, 1):
        # index to get correct dump file no
        i2 = dump_start + i * dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" % i2)):
            # load dump parameter file
            fin = open("dumps%d/parameters" % i2, "rb")
                            plc_cart(rho*((rho*(r*(r>25)+r*((25/r)**3.0)*(r<=25)))>0.2), -8, 2.2, 10, z, dir_images + "/ug%d.png" % count, r"log$(u_{g}/\rho)$ at %d $R_g/c$" % t)
                            plc_cart((gam - 1) * 2 * ug / bsq, -2, 4.2, 10, z, dir_images + "/beta%d.png" % count, r"log$(\beta)$ at %d $R_g/c$" % t)
                            plc_cart(uu[0] * np.sqrt(-1 / gcon[0, 0]), 0, 1, 10, z, dir_images + r"/gam%d.png" % count, "log$(\gamma)$ at %d $R_g/c$" % t)
                            t1.join()

                            plc_cart_xy1(rho, -8.0, 2.2, 10, z,1, dir_images + "/rhoxy%d.png" % count, r"log$(\rho)$ at %d $R_g/c$" % t)
                            plc_cart_xy1(bsq/rho, -5, 2, 10, z,1, dir_images + "/bsqxy%d.png" % count, r"log$(b^{2}/\rho)$ at %d $R_g/c$" % t)
                            plc_cart_xy1(ug/rho,  -5, 2.2, 10, z,1, dir_images + "/ugxy%d.png" % count, r"log$(u_{g}/\rho)$ at %d $R_g/c$" % t)
                            plc_cart_xy1((gam - 1) * 2 * ug / bsq,  -5, 3.2, 10, z,1, dir_images + "/betaxy%d.png" % count, r"log$(\beta)$ at %d $R_g/c$" % t)

                            #Very crude method of projecting onto midplane: Just shifts index
                            #preset_project_vertical()
                            #quant = project_vertical(rho)
                            #plc_cart_xy1(quant, -12.0, -5.0, 40, z,0, dir_images + "/rhoxy%d.png" % count, r"log$(\rho)$ at %d $R_g/c$" % t)
                            #quant = project_vertical(bsq / rho)
                            #plc_cart_xy1(quant, -5, 2, 40, z, dir_images + "/bsqxy%d.png" % count, r"log$(b^{2}/\rho)$ at %d $R_g/c$" % t)
                            #quant = project_vertical(ug / rho)
                            #plc_cart_xy1(quant, -5, 2.2, 40, z, dir_images + "/ugxy%d.png" % count, r"log$(u_{g}/\rho)$ at %d $R_g/c$" % t)
                            #quant = project_vertical((gam - 1) * 2 * ug / bsq)
                            #plc_cart_xy1(quant, -5, 3.2, 40, z, dir_images + "/betaxy%d.png" % count, r"log$(\beta)$ at %d $R_g/c$" % t)

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
            f_tot = open(savedir + "/post_process.txt", "wb")
            f_tot_rad = open(savedir + "/post_process_rad.txt", "wb")
            for i in range(0,numtasks):
                shutil.copyfileobj(open(savedir +"/post_process%d.txt" %i,'rb'), f_tot)
                os.remove(savedir + "/post_process%d.txt" %i)
                shutil.copyfileobj(open(savedir +"/post_process_rad%d.txt" %i,'rb'), f_tot_rad)
                os.remove(savedir + "/post_process_rad%d.txt" %i)
            f_tot.close()
            f_tot_rad.close()
        if (print_but):
            f_tot_but = open(savedir + "/post_process_but.txt", "wb")
            for i in range(0, numtasks):
                shutil.copyfileobj(open(savedir + "/post_process_but%d.txt" % i, 'rb'), f_tot_but)
                os.remove(savedir + "/post_process_but%d.txt" % i)
            f_tot_but.close()

if __name__ == "__main__":
    dirr = "/home/nkaaz/research/software/H-AMR/harm2d/reduced"
    #dirr = "/gpfs/alpine/phy129/proj-shared/T65TOR/reduced"
    #dirr = "E:\G\HAMR\HAMR_CUDA\HAMR\HAMR"
    #post_process(dirr, 1237,1238,1)
