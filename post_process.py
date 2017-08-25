
# coding: utf-8

# In[21]:
#from __future__ import division__future__ import division
from IPython.display import display

import os, sys, gc
import shutil

import sympy as sym
#from sympy import *
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb
import operator

from matplotlib.gridspec import GridSpec
from distutils.dir_util import copy_tree
#add amsmath to the preamble
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

from matplotlib import rc
rc('text', usetex=True)
font = { 'size'   : 28}
rc('font', **font)
rc('xtick', labelsize=38) 
rc('ytick', labelsize=38) 
#rc('xlabel', **int(f)ont) 
#rc('ylabel', **int(f)ont) 



mpl.rcParams['font.family'] = 'serif'



mpl.rcParams['font.serif'] = 'cmr10'

mpl.rcParams['font.sans-serif'] = 'cmr10'

plt.switch_backend('agg') 
mpl.rcParams['axes.unicode_minus']=False
legend = {'fontsize': 20}
rc('legend',**legend)

axes = {'labelsize': 38}
rc('axes', **axes)

fontsize = 38
mytype=np.float32
#get_ipython().magic(u'matplotlib inline')


from sympy.interactive import printing
#printing.init_printing(use_latex=True)

#For ODE integration
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def avg(v):
    return( 0.5*(v[1:]+v[:-1]) )

def der(v):
    return( (v[1:]-v[:-1]) )

def shrink(matrix, f):
    return matrix.reshape(f, matrix.shape[0]/f, f, matrix.shape[1]/f, f, matrix.shape[2]/f).sum(axis=0).sum(axis=1).sum(axis=2)

global comm, numtasks, rank

cluster=1
if(cluster==1):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    rank = comm.Get_rank()
else:
    numtasks=1
    rank=0

def myfloat(f,acc="float32"):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1 or acc=="float32":
        return( np.float32(f) )
    else:
        return( np.float64(f) )
        
def rpar_new(dump):
    global t, n_active,n_active_total, nstep,Dtd,Dtl,Dtr,dump_cnt,rdump_cnt,dt,failed
    global bs1, bs2,bs3, nb1,nb2,nb3,startx1,startx2,startx3, _dx1,_dx2,_dx3
    global tf,a,gam,cour,Rin,Rout,R0,fractheta
    global nx,ny,nz,nb, rhor   
    
    fin = open( "dumps%d/parameters" %dump, "rb" )
    
    t =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    n_active =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    n_active_total =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nstep =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    Dtd =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtl =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtr =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    dump_cnt =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    rdump_cnt =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    dt =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    failed =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    
    bs1=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs2=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs3=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nmax=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb1=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb2=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb3=  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    startx1 =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx2=  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx3=  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx1=  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx2=  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx3=  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    
    tf =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    fractheta =  np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    
    for i in range(0,15):
        trash =  np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb=n_active_total
    nx = bs1*nb1
    ny = bs2*nb2
    nz = bs3*nb3
    rhor = 1+(1-a**2)**0.5
    
    gd1 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    gd2 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
    #print gd1.shape
    for n in range(0,nmax):
        block[n,AMR_REFINED]=gd1[n]
        block[n,AMR_ACTIVE]=gd2[n]
    
    i=0;
    for n in range(0,nmax):
        if block[n,AMR_ACTIVE]==1:
            n_ord[i]=n
            i+=1
    fin.close()
    #print bs1,bs2,bs3


def rgdump_new(lowres):
    #if (rank==0):
    #    print "rgdump"
    global ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,dxdxp, alpha,axisym  
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new
    global nb1,nb2,nb3
    bs1new = int(bs1/lowres)
    bs2new = int(bs2/lowres)
    bs3new = int(bs3/lowres)

    nx = bs1new*nb1
    ny = bs2new*nb2
    nz = bs3new*nb3

    f = int(lowres)

    #Allocate memory
    ti = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    tj = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    tk = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    x1 = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    x2 = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    x3= np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    r = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    h = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    ph = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)

    if axisym:
        gcov= np.zeros((4,4,nb,bs1new,bs2new,1),dtype=mytype)
        gcon= np.zeros((4,4,nb,bs1new,bs2new,1),dtype=mytype)
        gdet= np.zeros((nb,bs1new,bs2new,1),dtype=mytype)
        dxdxp= np.zeros((4,4,nb,bs1new,bs2new,1),dtype=mytype)
        #alpha = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)

        gcovtemp= np.zeros((16,bs1new,bs2new),dtype=mytype)
        gcontemp= np.zeros((16,bs1new,bs2new),dtype=mytype)
        dxdxptemp = np.zeros((16,bs1new,bs2new),dtype=mytype)
    else:
        gcov= np.zeros((4,4,nb,bs1new,bs2new,bs3new),dtype=mytype)
        gcon= np.zeros((4,4,nb,bs1new,bs2new,bs3new),dtype=mytype)
        gdet= np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
        dxdxp= np.zeros((4,4,nb,bs1new,bs2new,bs3new),dtype=mytype)
        #alpha = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)

        gcovtemp= np.zeros((16,bs1new,bs2new,bs3new),dtype=mytype)
        gcontemp= np.zeros((16,bs1new,bs2new,bs3new),dtype=mytype)
        dxdxptemp = np.zeros((16,bs1new,bs2new,bs3new),dtype=mytype)
    
    #Read in data for every block
    if 0:
        for n in range(0,n_active_total):
            fin = open("gdumps/gdump%d" %n_ord[n], "rb" )   
            if(n%100==0 and rank==0):
                print "Reading in rgdump block %d " %n
            gd=np.fromfile(fin, dtype=np.float64, count=58*bs3*bs2*bs1, sep='')
            gd=gd.reshape((-1,bs3*bs2*bs1), order='F')
            gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))

            ti[n,0:bs1new,0:bs2new,0:bs3new]=0
            tj[n,0:bs1new,0:bs2new,0:bs3new]=0
            tk[n,0:bs1new,0:bs2new,0:bs3new]=0
            x1[n,0:bs1new,0:bs2new,0:bs3new]=0
            x2[n,0:bs1new,0:bs2new,0:bs3new]=0
            x3[n,0:bs1new,0:bs2new,0:bs3new]=0
            r[n,0:bs1new,0:bs2new,0:bs3new]=0
            h[n,0:bs1new,0:bs2new,0:bs3new]=0
            ph[n,0:bs1new,0:bs2new,0:bs3new]=0
            gdet[n,0:bs1new,0:bs2new,0]=0
            for II in range(16):
                gcovtemp[II,0:bs1new,0:bs2new]=0
                gcontemp[II,0:bs1new,0:bs2new]=0
                dxdxptemp[II,0:bs1new,0:bs2new]=0

            for f1 in range(1):
                for f2 in range(1):
                    for f3 in range(1):
                        ti[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[0,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        tj[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[1,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        tk[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[2,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 
                        x1[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[3,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        x2[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[4,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        x3[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[5,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 
                        r[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[6,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        h[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[7,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        ph[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[8,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 

                    if axisym:
                        gdet[n,0:bs1new,0:bs2new,0] += (1/f**0)*gd[41,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                        for II in range(16):
                            gcovtemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[9+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                            gcontemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[25+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                            dxdxptemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[42+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]

            gcov[:,:,n,:,:,0] = gcovtemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)
            gcon[:,:,n,:,:,0] = gcontemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)
            dxdxp[:,:,n,:,:,0] = dxdxptemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)

            fin.close()
    if 1:   
        #Read in data for every block
        for n in range(0,n_active_total):
            fin = open("gdumps/gdump%d" %n_ord[n], "rb" )   
            if(n%100==0 and rank==0):
                print "Reading in rgdump block %d " %n
            gd=np.fromfile(fin, dtype=np.float64, count=9*bs3*bs2*bs1, sep='')
            gd=gd.reshape((-1,bs3*bs2*bs1), order='F')
            gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))

            ti[n,0:bs1new,0:bs2new,0:bs3new]=0
            tj[n,0:bs1new,0:bs2new,0:bs3new]=0
            tk[n,0:bs1new,0:bs2new,0:bs3new]=0
            x1[n,0:bs1new,0:bs2new,0:bs3new]=0
            x2[n,0:bs1new,0:bs2new,0:bs3new]=0
            x3[n,0:bs1new,0:bs2new,0:bs3new]=0
            r[n,0:bs1new,0:bs2new,0:bs3new]=0
            h[n,0:bs1new,0:bs2new,0:bs3new]=0
            ph[n,0:bs1new,0:bs2new,0:bs3new]=0


            for f1 in range(1):
                for f2 in range(1):
                    for f3 in range(1):
                        ti[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[0,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        tj[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[1,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        tk[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[2,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 
                        x1[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[3,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        x2[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[4,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        x3[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[5,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 
                        r[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[6,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                        h[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[7,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]
                        ph[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[8,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 

            gd=np.fromfile(fin, dtype=np.float64, count=49*bs2*bs1, sep='')
            gd=gd.reshape((-1,bs2*bs1), order='F')
            gd=gd.reshape((-1,1,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))

            gdet[n,0:bs1new,0:bs2new,0]=0
            for II in range(16):
                gcovtemp[II,0:bs1new,0:bs2new]=0
                gcontemp[II,0:bs1new,0:bs2new]=0
                dxdxptemp[II,0:bs1new,0:bs2new]=0

            for f1 in range(1):
                for f2 in range(1):
                    if axisym:
                        gdet[n,0:bs1new,0:bs2new,0] += (1/f**0)*gd[41-9,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                        for II in range(16):
                            gcovtemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[9+II-9,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                            gcontemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[25+II-9,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]
                            dxdxptemp[II,0:bs1new,0:bs2new] += (1/f**0)*gd[42+II-9,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,0]

            gcov[:,:,n,:,:,0] = gcovtemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)
            gcon[:,:,n,:,:,0] = gcontemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)
            dxdxp[:,:,n,:,:,0] = dxdxptemp.view().reshape((4,4,bs1new,bs2new),order='F').transpose(1,0,2,3)

            fin.close() 
            
def rblock_new(dump):
    global AMR_ACTIVE, AMR_LEVEL, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4,AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1,AMR_CORN2,AMR_CORN3,AMR_CORN4,AMR_CORN5,AMR_CORN6
    global AMR_CORN7,AMR_CORN8,AMR_CORN9,AMR_CORN10,AMR_CORN11,AMR_CORN12
    global block, nmax, n_ord
    
    AMR_ACTIVE =0
    AMR_LEVEL =1
    AMR_REFINED =2
    AMR_COORD1= 3
    AMR_COORD2=4
    AMR_COORD3= 5
    AMR_PARENT =6
    AMR_CHILD1= 7
    AMR_CHILD2 =8
    AMR_CHILD3 =9
    AMR_CHILD4= 10
    AMR_CHILD5 =11
    AMR_CHILD6 =12
    AMR_CHILD7= 13
    AMR_CHILD8 =14
    AMR_NBR1 =15
    AMR_NBR2 =16
    AMR_NBR3= 17
    AMR_NBR4 =18
    AMR_NBR5= 19
    AMR_NBR6= 20
    AMR_NODE= 21
    AMR_POLE= 22
    AMR_GROUP =23
    AMR_CORN1 =24
    AMR_CORN2= 25
    AMR_CORN3 =26
    AMR_CORN4 =27
    AMR_CORN5 =28
    AMR_CORN6 =29
    AMR_CORN7= 30
    AMR_CORN8 =31
    AMR_CORN9= 32
    AMR_CORN10= 33
    AMR_CORN11 =34
    AMR_CORN12 =35
      
    #Read in data for every block
    if(0):
        fin = open("gdumps/grid", "rb")  
    if(1):
        fin = open("dumps%d/grid" %dump, "rb")  
    nmax=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    #print nmax
    #Allocate memory
    block = np.zeros((nmax,36),dtype=np.int32)
    n_ord = np.zeros((nmax),dtype=np.int32)
    
    gd = np.fromfile(fin, dtype=np.int32, count=36*nmax, sep='')
    gd = gd.reshape((36,nmax), order='F').T
    block=gd
    fin.close()
    
def rdump_new(dump,lowres):
    #print "rdump"
    global rho,ug,uu,ud,B,bu,bd,gcov,gdet,bsq,bsq, nb2d
    f = int(lowres)
    bs1new = int(bs1/lowres)
    bs2new = int(bs2/lowres)
    bs3new = int(bs3/lowres)
    #print "reduce resolution", f, "times"
    #Allocate memory
    rho = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    ug = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    bsq = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
    uu= np.zeros((4,nb,bs1new,bs2new,bs3new),dtype=mytype)
    ud= np.zeros((4,nb,bs1new,bs2new,bs3new),dtype=mytype)
    bu= np.zeros((4,nb,bs1new,bs2new,bs3new),dtype=mytype)
    bd= np.zeros((4,nb,bs1new,bs2new,bs3new),dtype=mytype)
    B= np.zeros((4,nb,bs1new,bs2new,bs3new),dtype=mytype)

    nb2d=0
    if 0:
        fin = open( "dumps%d/new_dump" %(dump), "rb" )
        for n in range(0,n_active_total):
            #read image
            gd=np.fromfile(fin, dtype=np.float32, count=9*bs1*bs2*bs3, sep='')
            gd=gd.reshape((-1,bs1*bs2*bs3), order='F')
            gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))
            #if(n%100==0):
            #    print "loaded block", n
    
            if 1:
                rho[n,0:bs1new,0:bs2new,0:bs3new]=0
                ug[n,0:bs1new,0:bs2new,0:bs3new]=0
                for II in range(4):
                    uu[II,n,0:bs1new,0:bs2new,0:bs3new]=0
                    B[II,n,0:bs1new,0:bs2new,0:bs3new]=0
                for f1 in range(1):
                    for f2 in range(1):
                        for f3 in range(1):
                            rho[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[0,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                            ug[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[1,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]      
                            for II in range(4):
                                uu[II,n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[2+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]             
                            for II in range(3):
                                B[1+II,n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[6+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]          
    
            B[0,n]=0.0
            if gcov.shape[2]==bu.shape[1]:
                ud[:,n]=mdot(gcov[:,:,n],uu[:,n])
                bu[0,n]=B[1,n]*mdot(uu[:,n],gcov[1,:,n])+B[2,n]*mdot(uu[:,n],gcov[2,:,n])+B[3,n]*mdot(uu[:,n],gcov[3,:,n])     
                bu[1:4,n]=(B[1:4,n]+bu[0,n]*uu[1:4,n])/(uu[0,n])   
                bd[:,n]=mdot(gcov[:,:,n],bu[:,n]) 
                bsq[n] = bu[0,n]*bd[0,n]+bu[1,n]*bd[1,n]+bu[2,n]*bd[2,n]+bu[3,n]*bd[3,n] 
        fin.close()
        
    if 1:
        u_stride=200
        umax=int((n_active_total-n_active_total%u_stride)/u_stride)
        if(n_active_total%u_stride>0):
            umax=int(umax+1)
        
        for u in range(0,umax):
            fin = open( "dumps%d" %(dump) + "/new_dump%d" %(u), "rb" )
            for n in range(u*u_stride,min((u+1)*u_stride,n_active_total)):
                #read image
                gd=np.fromfile(fin, dtype=np.float32, count=9*bs1*bs2*bs3, sep='')
                gd=gd.reshape((-1,bs1*bs2*bs3), order='F')
                gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
                gd=myfloat(gd.transpose(0,3,2,1))

                if 1:
                    rho[n,0:bs1new,0:bs2new,0:bs3new]=0
                    ug[n,0:bs1new,0:bs2new,0:bs3new]=0
                    for II in range(4):
                        uu[II,n,0:bs1new,0:bs2new,0:bs3new]=0
                        B[II,n,0:bs1new,0:bs2new,0:bs3new]=0
                    for f1 in range(1):
                        for f2 in range(1):
                            for f3 in range(1):
                                rho[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[0,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]        
                                ug[n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[1,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]      
                                for II in range(4):
                                    uu[II,n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[2+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f]             
                                for II in range(3):
                                    B[1+II,n,0:bs1new,0:bs2new,0:bs3new] += (1/f**0)*gd[6+II,f1:int(bs1new*f):f,int(f2+lowres/2):int(bs2new*f):f,f3:int(bs3new*f):f] 
                B[0,n]=0.0
                if 0:
                    if gcov.shape[2]==bu.shape[1]:
                        ud[:,n]=mdot(gcov[:,:,n],uu[:,n])
                        bu[0,n]=B[1,n]*mdot(uu[:,n],gcov[1,:,n])+B[2,n]*mdot(uu[:,n],gcov[2,:,n])+B[3,n]*mdot(uu[:,n],gcov[3,:,n])     
                        bu[1:4,n]=(B[1:4,n]+bu[0,n]*uu[1:4,n])/(uu[0,n])   
                        bd[:,n]=mdot(gcov[:,:,n],bu[:,n]) 
                        bsq[n] = bu[0,n]*bd[0,n]+bu[1,n]*bd[1,n]+bu[2,n]*bd[2,n]+bu[3,n]*bd[3,n] 
            fin.close()        
    nb2d=nb
        
        
def rdiag_new(dump):
    global divb,fail1,fail2, lowres
    
    if (not lowres):
        #Allocate memory
        divb = np.zeros((nb,bs1,bs2,bs3),dtype=mytype)
        fail1 = np.zeros((nb,bs1,bs2,bs3),dtype=mytype)
        fail2 = np.zeros((nb,bs1,bs2,bs3),dtype=mytype)

        for n in range(0,n_active_total):
            #read image
            fin = open( "dumps%d/dumpdiag%d" %(dump,n_ord[n]), "rb" )
            gd=np.fromfile(fin, dtype=np.float32, count=3*bs1*bs2*bs3, sep='')
            gd=gd.reshape((-1,bs1*bs2*bs3), order='F')
            gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))
            divb[n] = gd[0]
            fail1[n] = gd[1]
            fail2[n] = gd[2]
            fin.close()
    else:    
        f = lowres
        #Allocate memory
        divb = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
        fail1 = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)
        fail2 = np.zeros((nb,bs1new,bs2new,bs3new),dtype=mytype)

        for n in range(0,n_active_total):
            #read image
            fin = open( "dumps%d/dumpdiag%d" %(dump,n_ord[n]), "rb" )
            gd=np.fromfile(fin, dtype=mytype, count=3*bs1*bs2*bs3, sep='')
            gd=gd.reshape((-1,bs1*bs2*bs3), order='F')
            gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
            gd=myfloat(gd.transpose(0,3,2,1))

            for i in range(bs1new):
                for j in range(bs2new):
                    for k in range(bs3new):
                        divb[n,i,j,k] = np.average(gd[0,i*f:(i+1)*f,j*f:(j+1)*f,k*f:(k+1)*f])   
                        fail1[n,i,j,k] = np.average(gd[1,i*f:(i+1)*f,j*f:(j+1)*f,k*f:(k+1)*f])  
                        fail2[n,i,j,k] = np.average(gd[2,i*f:(i+1)*f,j*f:(j+1)*f,k*f:(k+1)*f])  
            fin.close()
        
def mdot(a,b):
    global lowres
    
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
    """
    global c
   
    BS1 = bs1new
    BS2 = bs2new
    BS3 = bs3new
  
    if (a.ndim == 3 and b.ndim == 3) or (a.ndim == 4 and b.ndim == 4):
        c = (a*b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
          #c = np.empty(np.amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)   
        c = np.empty((4,BS1,BS2,BS3),dtype=mytype)   
        for i in range(a.shape[0]):
            c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
        #c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype) 
        c = np.empty((4, BS1,BS2,BS3),dtype=mytype)
        #print c.shape
        for i in range(b.shape[1]):
            #print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    elif a.ndim == 5 and b.ndim == 5:
        #c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype) 
        c = np.empty((4, BS1,BS2,BS3),dtype=mytype)
        #print c.shape
        for i in range(b.shape[1]):
            #print ((a*b[:,i,:,:,:]).sum(0)).shape
            c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    return c

def psicalc():
    global aphi
    """
    Computes the field vector potential integrating from both poles to maintain accuracy.
    """
    daphi = (gdet*B[1])*_dx2
    aphi=-daphi[:,:,::-1,:].cumsum(axis=2)[:,:,::-1,:]
    aphi+=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi2=daphi[:,:,:,:].cumsum(axis=2)[:,:,:,:]
    aphi[:,:,:ny/2,:] = aphi2[:,:,:ny/2,:]
    return(aphi)
    
def plc_new(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    global r,h,ph
    l = [None] * nb2d
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k',0)
    mirrory = kwargs.pop('mirrory',0)
    #cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled',False)
    xy = kwargs.pop('xy',0)
    xmax = kwargs.pop('xmax',10)
    ymax = kwargs.pop('ymax',5)
    z=kwargs.pop('z',0)
    if ax is None:
        ax = plt.gca()
    if xy==-1:
        xcoord = np.log10(r* np.sin(h))
        ind=np.zeros_like(h)
        if(np.cos(h[i,j])>0.0):
            ind[0]=1.0
        else:
            ind[0]=-1.0
        ycoord = np.log10(r* np.abs(np.cos(h)))
        if(ycoord[0]<0.0):
            ycoord[0]=0.0
        ycoord*=ind
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[0,:,:,z],ycoord[0,:,:,z],myvar[0,:,:,z],nc,**kwargs)
        else:
            res = ax.contour(xcoord[0,:,:,z],ycoord[0,:,:,z],myvar[0,:,:,z],nc,**kwargs)
    else:
        if isfilled:
            for i in range(0,nb):
                if(tk[i,0,0,0]<10):
                    res = ax.contourf(xcoord[i,:,:,z],ycoord[i,:,:,z],myvar[i,:,:,z],nc,**kwargs)
        else:
            for i in range(0,nb):
                if(tk[i,0,0,0]<10):
                    res = ax.contour(xcoord[i,:,:,z],ycoord[i,:,:,z],myvar[i,:,:,z],nc,**kwargs)
    if( cb == True): #use color bar
         plt.colorbar(res,ax=ax)
    if xy:
        if (xy>0):
            plt.xlim(-xmax,xmax)
        else:
            plt.xlim(0.0,xmax)
        plt.ylim(-ymax,ymax)
    return res


def plc_cart0(rmax, offset, name):
    fig=plt.figure(figsize=(64,32))

    X = np.multiply(r, np.sin(h))
    Y = np.multiply(r, np.cos(h)) 
    #rmax = np.amax(r)
    X[:,:,0]=0.0*X[:,:,0]
    X[:,:,bs2-1]=0.0*X[:,:,bs2-1]
    plotmax = int(rmax*np.sqrt(2))
    #print np.amax(r)

    ilim = len(r[0,:,0,0])-1
    #print ilim
    for i in range(len(r[0,:,0,0])):
        if r[0,i,0,0] > plotmax:
            print i
            ilim = i
            break

    rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]=0.5*(rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]+rho[:,:,bs2-1,offset])
    rho[:,:,bs2-1,offset]=rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]
    
    rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]=0.5*(rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]+rho[:,:,0,offset])
    rho[:,:,0,offset]=rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]
    

    gammamax = np.amax(uu[0,0,0:ilim])
    
    plt.subplot(1,2,1)
    plc_new(np.log10((rho)[0:ilim]),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    res=plc_new(np.log10((rho)[0:ilim]),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$",fontsize=60)
    plt.ylabel(r"$y / R_g$",fontsize=60)
    plt.title(r"Rest mass density $\log(\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
   
    plt.subplot(1,2,2)
    plc_new(np.log10((rho)[0:ilim]),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax*10, ymax=rmax*10)
    res=plc_new(np.log10((rho)[0:ilim]),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax*10, ymax=rmax*10)
    plt.xlabel(r"$x / R_g$",fontsize=60)
    plt.ylabel(r"$y / R_g$",fontsize=60)
    plt.title(r"Rest mass density $\log(\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
    plt.savefig(name,dpi=30)
    plt.close()
    
def plc_cart(rmax, offset, name):
    fig=plt.figure(figsize=(64,32))

    X = np.multiply(r, np.sin(h))
    Y = np.multiply(r, np.cos(h)) 
    #rmax = np.amax(r)

    X[:,:,0]=0.0*X[:,:,0]
    X[:,:,bs2-1]=0.0*X[:,:,bs2-1]

    plotmax = int(rmax*np.sqrt(2))
    #print np.amax(r)

    ilim = len(r[0,:,0,0])-1
    #print ilim
    for i in range(len(r[0,:,0,0])):
        if r[0,i,0,0] > plotmax:
            ilim = i
            break

    rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]=0.5*(rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]+rho[:,:,bs2-1,offset])
    rho[:,:,bs2-1,offset]=rho[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]
    
    rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]=0.5*(rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]+rho[:,:,0,offset])
    rho[:,:,0,offset]=rho[:,:,0,int(len(r[0,0,0,:])*.5)+offset]


    bsq[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]=0.5*(bsq[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]+bsq[:,:,bs2-1,offset])
    bsq[:,:,bs2-1,offset]=bsq[:,:,bs2-1,int(len(r[0,0,0,:])*.5)+offset]
    
    bsq[:,:,0,int(len(r[0,0,0,:])*.5)+offset]=0.5*(bsq[:,:,0,int(len(r[0,0,0,:])*.5)+offset]+bsq[:,:,0,offset])
    bsq[:,:,0,offset]=bsq[:,:,0,int(len(r[0,0,0,:])*.5)+offset]


    gammamax = np.amax(uu[0,0,0:ilim])
    #print "gamma max:", gammamax
    
    plt.subplot(1,2,1)
    plc_new(np.log10((bsq/rho)[0:ilim]),levels=np.arange(-7,2,0.05),cb=0,isfilled=1,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    res=plc_new(np.log10((bsq/rho)[0:ilim]),levels=np.arange(-7,2,0.05),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$",fontsize=60)
    plt.ylabel(r"$y / R_g$",fontsize=60)
    plt.title(r"log$(b^2/\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
   
    plt.subplot(1,2,2)
    plc_new(np.log10((bsq/rho)[0:ilim]),levels=np.arange(-7,2,0.05),cb=0,isfilled=1,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax*10, ymax=rmax*10)
    res=plc_new(np.log10((bsq/rho)[0:ilim]),levels=np.arange(-7,2,0.05),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax*10, ymax=rmax*10)
    plt.xlabel(r"$x / R_g$",fontsize=60)
    plt.ylabel(r"$y / R_g$",fontsize=60)
    plt.title(r"log$(b^2/\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
    plt.savefig(name,dpi=30)
    plt.close()

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plc_cart2(rmax, offset, name):
    fig=plt.figure(figsize=(64,36))
    psicalc()
    X = np.multiply(r, np.sin(h))
    Y = np.multiply(r, np.cos(h)) 
    #rmax = np.amax(r)

    plotmax = int(rmax*np.sqrt(2))
    #print np.amax(r)

    ilim = len(r[0,:,0,0])-1
    #print ilim

    gammamax = 10
    #print "gamma max:", gammamax
    
    plt.subplot(1,2,1)
    res=plc_new(uu[0,0:ilim],levels=np.arange(1,gammamax,0.05),cb=0,isfilled=1,xcoord=np.log10(r),ycoord=x2, xy=0,z=offset)
    plc_new(aphi[0:ilim],levels=np.arange(0.01,aphi.max(),aphi.max()/10),colors="k",linewidths=1,z=offset, xcoord=np.log10(r),ycoord=x2, xy=0)
    #plc_new(np.log10(uu[0,0:ilim]),levels=np.arange(0,np.log10(gammamax),0.01),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=0,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$log(r / R_g)$",fontsize=38)
    plt.ylabel(r"$X_2$",fontsize=38)
    plt.title(r"Lorentz factor $\gamma$",fontsize=42)
    plt.gca().set_aspect(2)

   
    plt.subplot(1,2,2)
    plc_new((uu)[0,0:ilim],levels=np.arange(1,gammamax,0.05),cb=0,isfilled=1,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    plc_new(aphi[0:ilim],levels=np.arange(0.01,aphi.max(),aphi.max()/10),colors="k",linewidths=1,cb=0,xcoord=X[0:ilim],ycoord=Y[0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    res=plc_new((uu)[0,0:ilim],levels=np.arange(1,gammamax,0.05),cb=0,isfilled=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax, ymax=rmax)
    plc_new(aphi[0:ilim],levels=np.arange(0.01,aphi.max(),aphi.max()/10),colors="k",linewidths=1,xcoord=-1*X[0:ilim],ycoord=Y[0:ilim], xy=1,z=int(len(r[0,0,0,:])*.5)+offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$",fontsize=38)
    plt.ylabel(r"$y / R_g$",fontsize=38)
    plt.title(r"Lorentz factor $\gamma$",fontsize=42)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
    plt.savefig(name,dpi=100)
    plt.close()
    
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
    fdd = np.zeros((4,4,nb, bs1,bs2,bs3),dtype=rho.dtype)
    #fdd[0,0]=0*gdet
    #fdd[1,1]=0*gdet
    #fdd[2,2]=0*gdet
    #fdd[3,3]=0*gdet
    fdd[0,1]=gdet*(uu[2]*bu[3]-uu[3]*bu[2]) # f_tr
    fdd[1,0]=-fdd[0,1]
    fdd[0,2]=gdet*(uu[3]*bu[1]-uu[1]*bu[3]) # f_th
    fdd[2,0]=-fdd[0,2]
    fdd[0,3]=gdet*(uu[1]*bu[2]-uu[2]*bu[1]) # f_tp
    fdd[3,0]=-fdd[0,3]
    fdd[1,3]=gdet*(uu[2]*bu[0]-uu[0]*bu[2]) # f_rp = gdet*B2
    fdd[3,1]=-fdd[1,3]
    fdd[2,3]=gdet*(uu[0]*bu[1]-uu[1]*bu[0]) # f_hp = gdet*B1
    fdd[3,2]=-fdd[2,3]
    fdd[1,2]=gdet*(uu[0]*bu[3]-uu[3]*bu[0]) # f_rh = gdet*B3
    fdd[2,1]=-fdd[1,2]
    #
    fuu = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=rho.dtype)
    #fuu[0,0]=0*gdet
    #fuu[1,1]=0*gdet
    #fuu[2,2]=0*gdet
    #fuu[3,3]=0*gdet
    fuu[0,1]=-1/gdet*(ud[2]*bd[3]-ud[3]*bd[2]) # f^tr
    fuu[1,0]=-fuu[0,1]
    fuu[0,2]=-1/gdet*(ud[3]*bd[1]-ud[1]*bd[3]) # f^th
    fuu[2,0]=-fuu[0,2]
    fuu[0,3]=-1/gdet*(ud[1]*bd[2]-ud[2]*bd[1]) # f^tp
    fuu[3,0]=-fuu[0,3]
    fuu[1,3]=-1/gdet*(ud[2]*bd[0]-ud[0]*bd[2]) # f^rp
    fuu[3,1]=-fuu[1,3]
    fuu[2,3]=-1/gdet*(ud[0]*bd[1]-ud[1]*bd[0]) # f^hp
    fuu[3,2]=-fuu[2,3]
    fuu[1,2]=-1/gdet*(ud[0]*bd[3]-ud[3]*bd[0]) # f^rh
    fuu[2,1]=-fuu[1,2]
    #
    # these 2 are equal in degen electrodynamics when d/dt=d/dphi->0
    omegaf1=fdd[0,1]/fdd[1,3] # = ftr/frp
    omegaf2=fdd[0,2]/fdd[2,3] # = fth/fhp
    #
    # from jon branch, 04/10/2012
    #
    #if 0:
    B1hat=B[1]*np.sqrt(gcov[1,1])
    B2hat=B[2]*np.sqrt(gcov[2,2])
    B3nonhat=B[3]
    v1hat=uu[1]*np.sqrt(gcov[1,1])/uu[0]
    v2hat=uu[2]*np.sqrt(gcov[2,2])/uu[0]
    v3nonhat=uu[3]/uu[0]
        #
    aB1hat=np.fabs(B1hat)
    aB2hat=np.fabs(B2hat)
    av1hat=np.fabs(v1hat)
    av2hat=np.fabs(v2hat)
        #
    vpol=np.sqrt(av1hat**2 + av2hat**2)
    Bpol=np.sqrt(aB1hat**2 + aB2hat**2)
        #
        #omegaf1b=(omegaf1*aB1hat+omegaf2*aB2hat)/(aB1hat+aB2hat)
        #E1hat=fdd[0,1]*np.sqrt(gn3[1,1])
        #E2hat=fdd[0,2]*np.sqrt(gn3[2,2])
        #Epabs=np.sqrt(E1hat**2+E2hat**2)
        #Bpabs=np.sqrt(aB1hat**2+aB2hat**2)+1E-15
        #omegaf2b=Epabs/Bpabs
        #
        # assume field swept back so omegaf is always larger than vphi (only true for outflow, so put in sign switch for inflow as relevant for disk near BH or even jet near BH)
        # GODMARK: These assume rotation about z-axis
    omegaf2b=np.fabs(v3nonhat) + np.sign(uu[1])*(vpol/Bpol)*np.fabs(B3nonhat)
        #
    omegaf1b=v3nonhat - B3nonhat*(v1hat*B1hat+v2hat*B2hat)/(B1hat**2+B2hat**2)
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

def Tcalcud_new():
    global Tud, TudEM, TudMA
    global mu, sigma
    global enth
    global unb, isunbound
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    Tud = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    TudMA = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    TudEM = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            if(kapa==nu): delta = 1
            else: delta = 0
            TudEM[kapa,nu] = bsq*uu[kapa]*ud[nu] + 0.5*bsq*delta - bu[kapa]*bd[nu]
            TudMA[kapa,nu] = w*uu[kapa]*ud[nu]+pg*delta
            #Tud[kapa,nu] = eta*uu[kapa]*ud[nu]+(pg+0.5*bsq)*delta-bu[kapa]*bd[nu]
            Tud[kapa,nu] = TudEM[kapa,nu] + TudMA[kapa,nu]
    mu = -Tud[1,0]/(rho*uu[1])
    sigma = TudEM[1,0]/TudMA[1,0]
    enth=1+ug*gam/rho
    unb=enth*ud[0]
    isunbound=(-unb>1.0)
    
def Tcalcuu_new():
    global Tuu
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    Tuu = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            Tuu[kapa,nu] = bsq*uu[kapa]*uu[nu] + 0.5*bsq*gcon[kapa,nu] - bu[kapa]*bu[nu] + w*uu[kapa]*uu[nu]+pg*gcon[kapa,nu]

def animate_movie(dumps, stride):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\Matthew\\Desktop\\ffmpeg-20160508-git-38eeb85-win32-static\\bin\\ffmpeg.exe'

    res=1
    fig=plt.figure(figsize=(12,8))

    def init():
        return res,

    def animate(i):
        rpar_new(i)
        rdump_new(i,lowres)
        griddataall()
        res=plc_cart2(rmax=100, offset=0)
        print i
        return res

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1, interval=25, blit=1)
    FFwriter = animation.FFMpegWriter(fps=24)
    anim.save('movie_0.mp4', writer = FFwriter, dpi=300)
    
def griddataall():
    global block, n_ord, rho,uu,ud,bu,bd,B,ug,dxdxp,bsq,r,h,ph,nb,nb1,nb2,nb3,bs1,bs2,bs3,alpha,gcon,gcov,gdet, REF_1,REF_2,REF_3
    global x1,x2,x3,ti,tj,tk,_dx1,_dx2,_dx3
    global gridsizex1,gridsizex2,gridsizex3
    global bs1new,bs2new,bs3new, lowres
    ACTIVE = np.max(block[n_ord,AMR_LEVEL])
    threeD = 1

    bs1new = int(bs1/lowres)
    bs2new = int(bs2/lowres)
    bs3new = int(bs3/lowres)
    
    BS1 = bs1new
    BS2 = bs2new
    BS3 = bs3new
   
    gridsizex1 = nb1*(1+REF_1)**ACTIVE*BS1
    gridsizex2 = nb2*(1+REF_2)**ACTIVE*BS2
    gridsizex3 = nb3*(1+REF_3)**ACTIVE*BS3

    rhogrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    x1grid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    x2grid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    x3grid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    tigrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    tjgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    tkgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    
    
    uugrid = np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    udgrid = np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    bugrid = np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    bdgrid = np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)

    Bgrid = np.zeros((4,1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    rgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    hgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    phgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    axisym = 1
    uggrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    bsqgrid = np.zeros((1,gridsizex1,gridsizex2,gridsizex3),dtype=mytype)
    
    if axisym:
        dxdxpgrid = np.zeros((4,4,1,gridsizex1,gridsizex2,1),dtype=mytype)
        gcongrid = np.zeros((4,4,1,gridsizex1,gridsizex2,1),dtype=mytype)
        gcovgrid = np.zeros((4,4,1,gridsizex1,gridsizex2,1),dtype=mytype)
        gdetgrid = np.zeros((1,gridsizex1,gridsizex2,1),dtype=mytype)
        #alphagrid = np.zeros((gridsizex1,gridsizex2))
      
    for n in range(nb):
        #print "start block", n
        P = ACTIVE - block[n_ord[n],AMR_LEVEL]
        #print P, ACTIVE, block[n_ord[n],AMR_LEVEL]
        coord1 = block[n_ord[n],AMR_COORD1]
        coord2 = block[n_ord[n],AMR_COORD2]
        coord3 = block[n_ord[n],AMR_COORD3]
        for Li in range((1+REF_1)**P):
            for Lj in range((1+REF_2)**P):
                for Lk in range((1+REF_3)**P):
                    rhogrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                            coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = rho[n,0:BS1,0:BS2,0:BS3]
                    uggrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                            coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = ug[n,0:BS1,0:BS2,0:BS3]
                    uugrid[:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                            coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = uu[:,n,0:BS1,0:BS2,0:BS3]
                    bugrid[:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                            coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = bu[:,n,0:BS1,0:BS2,0:BS3]
                    Bgrid[:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                            coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = B[:,n,0:BS1,0:BS2,0:BS3]
                    if(x1.shape[0]!=x1grid.shape[0]):
                        x1grid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = x1[n,0:BS1,0:BS2,0:BS3]
                        x2grid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = x2[n,0:BS1,0:BS2,0:BS3]
                        x3grid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = x3[n,0:BS1,0:BS2,0:BS3]
                        tigrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = ti[n,0:BS1,0:BS2,0:BS3]
                        tjgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = tj[n,0:BS1,0:BS2,0:BS3]
                        tkgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = tk[n,0:BS1,0:BS2,0:BS3]
                        rgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = r[n,0:BS1,0:BS2,0:BS3]
                        hgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = h[n,0:BS1,0:BS2,0:BS3]
                        phgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                                coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,
                                                                coord3*((1+REF_3)**P*BS3)+Lk: coord3*((1+REF_3)**P*BS3)+BS3*(1+REF_3)**P:(1+REF_3)**P] = ph[n,0:BS1,0:BS2,0:BS3]
        if(x1.shape[0]!=x1grid.shape[0]):
            if axisym:
                for Li in range((1+REF_1)**P):
                    for Lj in range((1+REF_2)**P):
                        dxdxpgrid[:,:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,0] = dxdxp[:,:,n,0:BS1,0:BS2,0]
                        gcongrid[:,:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,0] = gcon[:,:,n,0:BS1,0:BS2,0]
                        gcovgrid[:,:,0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,0] = gcov[:,:,n,0:BS1,0:BS2,0]
                        gdetgrid[0,coord1*((1+REF_1)**P*BS1)+Li:coord1*((1+REF_1)**P*BS1)+BS1*(1+REF_1)**P:(1+REF_1)**P,
                                                            coord2*((1+REF_2)**P*BS2)+Lj: coord2*((1+REF_2)**P*BS2)+BS2*(1+REF_2)**P:(1+REF_2)**P,0] = gdet[n,0:BS1,0:BS2,0]
   
    #return rhogrid,uugrid,udgrid,Bgrid,uggrid,bsqgrid,rgrid,hgrid,phgrid,alphagrid,dxdxpgrid,gcongrid,gcovgrid
    bs1 = gridsizex1
    bs2 = gridsizex2
    bs3 = gridsizex3
    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3
    _dx1=_dx1*lowres
    _dx2=_dx2*lowres
    _dx3=_dx3*lowres
    if(x1.shape[0]!=x1grid.shape[0]):
        x1 = x1grid
        x2 = x2grid
        x3 = x3grid
        ti = tigrid
        tj = tjgrid
        tk = tkgrid
        r = rgrid
        h = hgrid
        ph = phgrid
        dxdxp = dxdxpgrid
        gcon = gcongrid
        gcov = gcovgrid
        gdet = gdetgrid
    
    for n in range(1):
        udgrid[:,n]=mdot(gcov[:,:,n],uugrid[:,n])
        bugrid[0,n]=Bgrid[1,n]*mdot(uugrid[:,n],gcov[1,:,n])+Bgrid[2,n]*mdot(uugrid[:,n],gcov[2,:,n])+Bgrid[3,n]*mdot(uugrid[:,n],gcov[3,:,n])     
        bugrid[1:4,n]=(Bgrid[1:4,n]+bugrid[0,n]*uugrid[1:4,n])/uugrid[0,n]         
        bdgrid[:,n]=mdot(gcov[:,:,n],bugrid[:,n]) 
        bsqgrid[n] = bugrid[0,n]*bdgrid[0,n]+bugrid[1,n]*bdgrid[1,n]+bugrid[2,n]*bdgrid[2,n]+bugrid[3,n]*bdgrid[3,n]   
    rho = rhogrid
    uu = uugrid
    ud = udgrid
    ug = uggrid
    B = Bgrid
    bu = bugrid
    bd = bdgrid
    bsq = bsqgrid
  

    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1
    
#Export a VTK file for visualization in eg VisIt
def export_vtk(number):
    from numpy import mgrid, empty, sin, pi
    from tvtk.api import tvtk, write_data
    from mayavi import mlab

    # The actual points.
    pts = empty(rho[0].shape + (3,), dtype=float)
    pts[..., 0] = np.multiply(np.multiply(r,np.cos(ph[0])),np.sin(h[0]))
    pts[..., 1] = np.multiply(np.multiply(r,np.sin(ph[0])),np.sin(h[0]))
    pts[..., 2] = np.multiply(r,np.cos(h[0]))

    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size / 3, 3

    # Create the dataset.
    sg = tvtk.StructuredGrid(dimensions=rho[0].shape, points=pts)
    scalars = uu[0,0]
    scalars = scalars.T.copy()
    sg.point_data.scalars = scalars.ravel()
    sg.point_data.scalars.name = 'gamma'
    write_data(sg,'gamma%d.vtk'%number)
     
    sg = tvtk.StructuredGrid(dimensions=rho[0].shape, points=pts)
    scalars = np.log10(bsq/rho*r)[0]
    scalars = scalars.T.copy()
    sg.point_data.scalars = scalars.ravel()
    sg.point_data.scalars.name = 'bsqorho'
    write_data(sg,'bsqorho%d.vtk'%number)

    sg = tvtk.StructuredGrid(dimensions=rho[0].shape, points=pts)
    scalars = np.log10(rho[0])
    scalars = scalars.T.copy()
    sg.point_data.scalars = scalars.ravel()
    sg.point_data.scalars.name = 'rho'
    write_data(sg,'rho%d.vtk'%number)
    
    print number

#Calculate Jacobian transformation to Cartesian coordinates from spherical coordinates
def calc_cart():
    global dxdr
    dxdr=np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    dxdr[0,0]=1*x1/x1
    dxdr[0,1]=0*x1
    dxdr[0,2]=0*x1
    dxdr[0,3]=0*x1
    dxdr[1,0]=0*x1
    dxdr[1,1]=np.sin(h)*np.cos(ph)
    dxdr[1,2]=r*np.cos(h)*np.cos(ph)
    dxdr[1,3]=-r*np.sin(h)*np.sin(ph)
    dxdr[2,0]=0*x1
    dxdr[2,1]=np.sin(h)*np.sin(ph)
    dxdr[2,2]=r*np.cos(h)*np.sin(ph)
    dxdr[2,3]=r*np.sin(h)*np.cos(ph)
    dxdr[3,0]=0*x1
    dxdr[3,1]=np.cos(h)
    dxdr[3,2]=-r*np.sin(h)
    dxdr[3,3]=0*x1

#Matrix inversion
def invert_matrix():
    global dxdxp_inv, dxdr_inv
    dxdxp_inv=np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    dxdr_inv=np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    for i in range(0,bs1):
        for j in range(0,bs2):
            for k in range(0,bs3):
                dxdxp_inv[:,:,0,i,j,k]=np.linalg.inv(dxdxp[:,:,0,i,j,k])
                dxdr_inv[:,:,0,i,j,k]=np.linalg.inv(dxdr[:,:,0,i,j,k])

#Calculate tilt and precession angles fast without need for calc_cart
def calc_precesion_fast(dump,avg):
    global low_res
    rpar_new(dump)
    rdump_new(dump,lowres)
    griddataall()
    
    global angle_tilt,angle_prec
    S=np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    J=np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    J_car=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    J_BH=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    L=np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    S_r=np.zeros((nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    xc=np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')

    xc[0]=0*x1
    xc[1]=x1
    xc[2]=0*x2
    xc[3]=0*x3
    Tcalcuu_new()
    tilt = 0/180*3.141592
    for mu in range(0,4): 
        S[mu]=(Tuu[mu,0])*np.sqrt(np.abs(gcov[mu][mu]))*gdet*_dx1*_dx2*_dx3
    for mu in range(0,4):
        for nu in range(0,4):
            L[mu,nu]= ((xc[mu]*Tuu[nu,0]*np.sqrt(np.abs(gcov[mu][mu]*gcov[nu][nu]))-xc[nu]*Tuu[mu,0]*np.sqrt(np.abs(gcov[nu][nu]*gcov[mu][mu]))))*gdet*_dx1*_dx2*_dx3

    S_r=-S[0]*S[0]+S[1]*S[1]+S[2]*S[2]+S[3]*S[3]
    
    J[3]=(L[1,2]*S[0]-L[2,1]*S[0])/(2*np.sqrt(np.abs(-S_r)))
    J[2]=-(L[1,3]*S[0]-L[3,1]*S[0])/(2*np.sqrt(np.abs(-S_r)))
    J[1]=(L[2,3]*S[0]-L[3,2]*S[0])/(2*np.sqrt(np.abs(-S_r)))
    
    if(avg==0):
        J_car[1]=((J[1]*np.sin(h)*np.cos(ph)+J[2]*np.cos(h)*np.cos(ph)-J[3]*np.sin(ph))*_dx1*_dx2*_dx3*gdet).sum(3).sum(2)
        J_car[2]=((J[1]*np.sin(h)*np.sin(ph)+J[2]*np.cos(h)*np.sin(ph)+J[3]*np.cos(ph))*_dx1*_dx2*_dx3*gdet).sum(3).sum(2)
        J_car[3]=((J[1]*np.cos(h)-J[2]*np.sin(h))*_dx1*_dx2*_dx3*gdet).sum(3).sum(2)
    else:
        J_car[1]=((J[1]*np.sin(h)*np.cos(ph)+J[2]*np.cos(h)*np.cos(ph)-J[3]*np.sin(ph))*_dx1*_dx2*_dx3*gdet*(r<150)).sum(3).sum(2).sum(1)
        J_car[2]=((J[1]*np.sin(h)*np.sin(ph)+J[2]*np.cos(h)*np.sin(ph)+J[3]*np.cos(ph))*_dx1*_dx2*_dx3*gdet*(r<150)).sum(3).sum(2).sum(1)
        J_car[3]=((J[1]*np.cos(h)-J[2]*np.sin(h))*_dx1*_dx2*_dx3*gdet*(r<150)).sum(3).sum(2).sum(1)
    J_length=np.sqrt(J_car[1]*J_car[1]+J_car[2]*J_car[2]+J_car[3]*J_car[3])
    
    J_BH[1]=-np.sin(tilt)*J_car[1]/J_car[1]
    J_BH[2]=0*J_car[1]/J_car[1]
    J_BH[3]=np.cos(tilt)*J_car[1]/J_car[1]
    
    J_BH_length=np.sqrt(J_BH[1]*J_BH[1]+J_BH[2]*J_BH[2]+J_BH[3]*J_BH[3])
    
    angle_tilt=np.arccos((J_car[1]*J_BH[1]+J_car[2]*J_BH[2]+J_car[3]*J_BH[3])/(J_BH_length*J_length))*180/3.14
    angle_prec=np.arctan2(J_car[2],J_car[1])*180/3.14

global set_cart
global set_xc
set_cart=0
set_xc=0

#Calculate tilt and precession angles accurate with need for calc_cart
def calc_precesion_accurate_disk(dump, avg):
    global lowres, uu,bu
    global set_cart, set_xc
    if(set_cart!=1):
        calc_cart()
    global angle_tilt_disk,angle_prec_disk, xc
    Su=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    J_car=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    L=np.zeros((4,4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    S_r=np.zeros((nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    JBH_cross_D=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    J_BH=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    
    if(set_xc!=1):
        xc=np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
        xc[0]=0*x1
        xc[1]=r*np.sin(h)*np.cos(ph)
        xc[2]=r*np.sin(h)*np.sin(ph)
        xc[3]=r*np.cos(h)
        set_xc=1
    tilt = 0/180*3.141592
    
    uu_new=np.copy(uu)
    bu_new=np.copy(bu)
    uu[:,0]=mdot(dxdxp[:,:,0], uu[:,0])
    uu[:,0]=mdot(dxdr[:,:,0], uu[:,0])
    bu[:,0]=mdot(dxdxp[:,:,0], bu[:,0])
    bu[:,0]=mdot(dxdr[:,:,0], bu[:,0])
    Tcalcuu_new()
    
    if(avg==0):
        for mu in range(0,4): 
            Su[mu]=((Tuu[mu,0])*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
        for mu in range(0,4):
            for nu in range(0,4):
                L[mu,nu]= ((xc[mu]*Tuu[nu,0]-xc[nu]*Tuu[mu,0])*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    else:
        for mu in range(0,4): 
            Su[mu]=((Tuu[mu,0])*gdet*_dx1*_dx2*_dx3*(rho>0.0025)*(r<750)).sum(3).sum(2).sum(1)
        for mu in range(0,4):
            for nu in range(0,4):
                L[mu,nu]= ((xc[mu]*Tuu[nu,0]-xc[nu]*Tuu[mu,0])*gdet*_dx1*_dx2*_dx3*(rho>0.0025)*(r<750)).sum(3).sum(2).sum(1)
    
    S_r[0]=Su[0,0]*Su[0,0]+Su[1,0]*Su[1,0]+Su[2,0]*Su[2,0]+Su[3,0]*Su[3,0]

    J_car[3]=(L[1,2]*Su[0]-L[2,1]*Su[0]+L[2,0]*Su[1]-L[0,2]*Su[1]-L[1,0]*Su[2]+L[0,1]*Su[2])/(2*np.sqrt(np.abs(-S_r)))
    J_car[2]=-(L[1,3]*Su[0]-L[3,1]*Su[0]-L[1,0]*Su[3]+L[3,0]*Su[1]-L[0,3]*Su[1]+L[0,1]*Su[3])/(2*np.sqrt(np.abs(-S_r)))
    J_car[1]=(L[2,3]*Su[0]-L[3,2]*Su[0]-L[2,0]*Su[3]+L[0,2]*Su[3]+L[3,0]*Su[2]-L[0,3]*Su[2])/(2*np.sqrt(np.abs(-S_r)))
    J_length=np.sqrt(J_car[1]*J_car[1]+J_car[2]*J_car[2]+J_car[3]*J_car[3])

    J_BH[1]=-np.sin(tilt)*J_car[1]/J_car[1]
    J_BH[2]=0*J_car[1]/J_car[1]
    J_BH[3]=np.cos(tilt)*J_car[1]/J_car[1]
    J_BH_length=np.sqrt(J_BH[1]*J_BH[1]+J_BH[2]*J_BH[2]+J_BH[3]*J_BH[3])
    
    JBH_cross_D[1]=J_BH[2]*J_car[3]-J_BH[3]*J_car[2]
    JBH_cross_D[2]=J_BH[3]*J_car[1]-J_BH[1]*J_car[3]
    JBH_cross_D[3]=J_BH[1]*J_car[2]-J_BH[2]*J_car[1]
    JBH_cross_D_length=np.sqrt( JBH_cross_D[1]* JBH_cross_D[1]+JBH_cross_D[2]* JBH_cross_D[2]+JBH_cross_D[3]* JBH_cross_D[3])
    
    angle_tilt_disk=np.arccos((J_car[1]*J_BH[1]+J_car[2]*J_BH[2]+J_car[3]*J_BH[3])/(J_BH_length*J_length))*180/3.14
    angle_prec_disk=-np.arctan2(JBH_cross_D[1],JBH_cross_D[2])*180/3.14
    uu=np.copy(uu_new)
    bu=np.copy(bu_new)

#Calculate tilt and precession angles accurate with need for calc_cart
def calc_precesion_accurate_corona(dump, avg):
    global lowres, uu,bu
    global set_cart, set_xc
    if(set_cart!=1):
        calc_cart()
    global angle_tilt_corona,angle_prec_corona, xc
    Su=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    J_car=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    L=np.zeros((4,4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    S_r=np.zeros((nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    JBH_cross_D=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    J_BH=np.zeros((4,nb,bs1*(avg==0)+(avg==1)),dtype=np.float32,order='F')
    
    if(set_xc!=1):
        xc=np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
        xc[0]=0*x1
        xc[1]=r*np.sin(h)*np.cos(ph)
        xc[2]=r*np.sin(h)*np.sin(ph)
        xc[3]=r*np.cos(h)
        set_xc=1
    tilt = 0/180*3.141592
    
    uu_new=np.copy(uu)
    bu_new=np.copy(bu)
    uu[:,0]=mdot(dxdxp[:,:,0], uu[:,0])
    uu[:,0]=mdot(dxdr[:,:,0], uu[:,0])
    bu[:,0]=mdot(dxdxp[:,:,0], bu[:,0])
    bu[:,0]=mdot(dxdr[:,:,0], bu[:,0])
    Tcalcuu_new()
    
    if(avg==0):
        for mu in range(0,4): 
            Su[mu]=((Tuu[mu,0])*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
        for mu in range(0,4):
            for nu in range(0,4):
                L[mu,nu]= ((xc[mu]*Tuu[nu,0]-xc[nu]*Tuu[mu,0])*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)
    else:
        for mu in range(0,4): 
            Su[mu]=((Tuu[mu,0])*gdet*_dx1*_dx2*_dx3*(r<1500)).sum(3).sum(2).sum(1)
        for mu in range(0,4):
            for nu in range(0,4):
                L[mu,nu]= ((xc[mu]*Tuu[nu,0]-xc[nu]*Tuu[mu,0])*gdet*_dx1*_dx2*_dx3*(r<1500)).sum(3).sum(2).sum(1)
    
    S_r[0]=Su[0,0]*Su[0,0]+Su[1,0]*Su[1,0]+Su[2,0]*Su[2,0]+Su[3,0]*Su[3,0]

    J_car[3]=(L[1,2]*Su[0]-L[2,1]*Su[0]+L[2,0]*Su[1]-L[0,2]*Su[1]-L[1,0]*Su[2]+L[0,1]*Su[2])/(2*np.sqrt(np.abs(-S_r)))
    J_car[2]=-(L[1,3]*Su[0]-L[3,1]*Su[0]-L[1,0]*Su[3]+L[3,0]*Su[1]-L[0,3]*Su[1]+L[0,1]*Su[3])/(2*np.sqrt(np.abs(-S_r)))
    J_car[1]=(L[2,3]*Su[0]-L[3,2]*Su[0]-L[2,0]*Su[3]+L[0,2]*Su[3]+L[3,0]*Su[2]-L[0,3]*Su[2])/(2*np.sqrt(np.abs(-S_r)))
    J_length=np.sqrt(J_car[1]*J_car[1]+J_car[2]*J_car[2]+J_car[3]*J_car[3])

    J_BH[1]=-np.sin(tilt)*J_car[1]/J_car[1]
    J_BH[2]=0*J_car[1]/J_car[1]
    J_BH[3]=np.cos(tilt)*J_car[1]/J_car[1]
    J_BH_length=np.sqrt(J_BH[1]*J_BH[1]+J_BH[2]*J_BH[2]+J_BH[3]*J_BH[3])
    
    JBH_cross_D[1]=J_BH[2]*J_car[3]-J_BH[3]*J_car[2]
    JBH_cross_D[2]=J_BH[3]*J_car[1]-J_BH[1]*J_car[3]
    JBH_cross_D[3]=J_BH[1]*J_car[2]-J_BH[2]*J_car[1]
    JBH_cross_D_length=np.sqrt( JBH_cross_D[1]* JBH_cross_D[1]+JBH_cross_D[2]* JBH_cross_D[2]+JBH_cross_D[3]* JBH_cross_D[3])
    
    angle_tilt_corona=np.arccos((J_car[1]*J_BH[1]+J_car[2]*J_BH[2]+J_car[3]*J_BH[3])/(J_BH_length*J_length))*180/3.14
    angle_prec_corona=-np.arctan2(JBH_cross_D[1],JBH_cross_D[2])*180/3.14
    uu=np.copy(uu_new)
    bu=np.copy(bu_new)

def sub_calc_jet_tot(var):
    global lrho,bs2
    global XX, YY, ZZ, gdet,h
    rin=10
    rout=150
    var[lrho<0.5] = 0.0
    var = np.swapaxes(var, 3,0)
    var = var[:,(r[0,:,0,0]>rin) & (r[0,:,0,0]<rout),:,:]
    XX = XX[:,(r[0,:,0,0]>rin) & (r[0,:,0,0]<rout),:,:]
    YY = YY[:,(r[0,:,0,0]>rin) & (r[0,:,0,0]<rout),:,:]
    ZZ = ZZ[:,(r[0,:,0,0]>rin) & (r[0,:,0,0]<rout),:,:]
    gdetc = gdet[:,(r[0,:,0,0]>rin) & (r[0,:,0,0]<rout),:,:]

    var_up_tot = np.sum((gdetc[0,:,0:int(bs2/2),0]*var[:,:,0:int(bs2/2),0]))    
    var_flux_up_tot = np.zeros(3)   
    var_flux_up_tot[0] = np.sum((XX[:,:,0:int(bs2/2),0]*var[:,:,0:int(bs2/2),0]*gdetc[0,:,0:int(bs2/2),0]))/var_up_tot
    var_flux_up_tot[1] = np.sum((YY[:,:,0:int(bs2/2),0]*var[:,:,0:int(bs2/2),0]*gdetc[0,:,0:int(bs2/2),0]))/var_up_tot
    var_flux_up_tot[2] = np.sum((ZZ[:,:,0:int(bs2/2),0]*var[:,:,0:int(bs2/2),0]*gdetc[0,:,0:int(bs2/2),0]))/var_up_tot      

    r_up = np.linalg.norm(var_flux_up_tot)   

    angle_jet = np.zeros(4)
    angle_jet[1] = 90-np.arctan2(var_flux_up_tot[0], var_flux_up_tot[1])*180/np.pi
    angle_jet[0] = np.arccos(var_flux_up_tot[2]/r_up)*180/np.pi   
    
    var_down_tot = np.sum((gdetc[0,:,int(bs2/2):int(bs2),0]*var[:,:,int(bs2/2):int(bs2),0]))    
    var_flux_down_tot = np.zeros(3)   
    var_flux_down_tot[0] = np.sum((XX[:,:,int(bs2/2):int(bs2),0]*var[:,:,int(bs2/2):int(bs2),0]*gdetc[0,:,int(bs2/2):int(bs2),0]))/var_down_tot
    var_flux_down_tot[1] = np.sum((YY[:,:,int(bs2/2):int(bs2),0]*var[:,:,int(bs2/2):int(bs2),0]*gdetc[0,:,int(bs2/2):int(bs2),0]))/var_down_tot
    var_flux_down_tot[2] = np.sum((ZZ[:,:,int(bs2/2):int(bs2),0]*var[:,:,int(bs2/2):int(bs2),0]*gdetc[0,:,int(bs2/2):int(bs2),0]))/var_down_tot      

    r_down = np.linalg.norm(var_flux_down_tot)   

    angle_jet[3] = 90-np.arctan2(var_flux_down_tot[0], var_flux_down_tot[1])*180/np.pi
    angle_jet[2] = np.arctan2(var_flux_down_tot[2],r_down)*180/np.pi    

    return angle_jet

def calc_jet_tot():
    global Tuu, gdet, uu, bu,dxdxp,rho
    global XX, YY, ZZ   
    global lrho,r,h,ph,bs1,bs2,bs3
    lrho = np.log10(bsq*(rho**-1)*r)
    global angle_jet   
   
    uu_new=np.copy(uu)
    bu_new=np.copy(bu)
    uu[:,0]=mdot(dxdxp[:,:,0], uu[:,0])
    bu[:,0]=mdot(dxdxp[:,:,0], bu[:,0])
    Tcalcuu_new()
    
    cart = np.zeros((bs1,bs2,3,bs3))
    XX=np.sin(h)*np.cos(ph)
    YY=np.sin(h)*np.sin(ph)
    ZZ=np.cos(h)
   
    cart[:,:,0,:]=XX[0]
    cart[:,:,1,:]=YY[0]
    cart[:,:,2,:]=ZZ[0]   

    XX = np.swapaxes(XX,3,0)
    YY = np.swapaxes(YY,3,0)
    ZZ = np.swapaxes(ZZ,3,0)
    Euucut = np.copy(Tuu[0,1])
    angle_jet = sub_calc_jet_tot(Euucut)
    uu=np.copy(uu_new)
    bu=np.copy(bu_new)

#Calculate alpha viscosity parameter assuming no tilt
def calc_alpha():
    global alpha, alpha_eff
    alpha=np.log10((np.abs(bd[1]*bd[3]))*np.sqrt(np.abs(gcon[3,3]*gcon[1,1]))/(bsq/2)+(5/3-1)*ug)
    cs=np.sqrt(5/3*(5/3-1)*ug/(rho+ug+(5/3-1)*ug))
    v_r=uu[1]*np.sqrt(gcov[1,1])
    v_or=uu[3]*np.sqrt(gcov[3,3])
    alpha_eff=np.log10(np.abs(v_r*v_or/(cs**2)))

#Print average disk thickness
def calc_HoR():
    H_over_R=np.sqrt((rho**2*(h-3.14/2)**2*gdet).sum(3).sum(2).sum(1)/((rho**2*gdet).sum(3).sum(2).sum(1)))
    print H_over_R

#Print total mass of disk in code units
def calc_Mtot():
    global Mtot
    Mtot=((rho*uu[0])*_dx1*_dx2*_dx3*gdet ).sum(-1).sum(-1).sum(-1)
    #print Mtot

#Calculate precession period
def calc_PrecPeriod():
    v_phi=1/(2*3.141592)*1/(r**1.5+a*r/r)
    v_nod=v_phi*(r/r-np.sqrt(r/r-4*a/r**1.5+3*a**2/r**2))
    T_phi=1/v_phi
    T_nod=1/v_nod
    T_nod=np.nan_to_num(T_nod)
    L=rho*r*np.sqrt((np.sqrt(gcov[2,2])*uu[2])**2+(np.sqrt(gcov[3,3])*uu[3])**2)
    L_tot=(L*gdet*_dx1*_dx2*_dx3*(r>1)*(r<100)).sum(-1).sum(-1).sum(-1)
    tau_tot=(L*(2*0.9375/r**3)*gdet*_dx1*_dx2*_dx3*(r>1)*(r<100)).sum(-1).sum(-1).sum(-1)
    precperiod=2*np.pi*L_tot/tau_tot
    print precperiod

#Calculate mass accretion rate as function of radius
def calc_Mdot():
    global Mdot,Edot
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    TudEM = bsq*uu[1]*ud[0]  - bu[1]*bd[0]
    TudMA = w*uu[1]*ud[0]
    Tud=TudEM+TudMA
    Mdot=(-gdet*rho*uu[1]*_dx2*_dx3).sum(-1).sum(-1)
    Edot=(Tud*gdet*_dx2*_dx3).sum(-1).sum(-1)
    #plt.plot(r[0,:,bs2/2,0],Mdot[0])
    
#Calculate magnetic flux phibh as function of radius
def calc_phibh():
    global phibh
    #phibh = 0.7*(4*np.pi)**0.5*(np.abs(gdet*B[1])*_dx2*_dx3).sum(-1).sum(-1)/((-gdet*rho*uu[1]*_dx2*_dx3).sum(-1).sum(-1))**0.5
    phibh = 0.7*(4*np.pi)**0.5*(np.abs(gdet*B[1])*_dx2*_dx3).sum(-1).sum(-1)
    #plt.plot(r[0,:,bs2/2,0],phibh[0])
    #plt.xlim(0,100)
    #plt.ylim(0,15)

#Calculate the Q resolution paramters and their average Q_avg in the disk
def calc_Q():
    global Q, Q_avg
    Q=np.zeros((4,1,bs1,bs2,bs3),dtype=np.float32,order='F')
    Q_avg=np.zeros((4),dtype=np.float32,order='F')
    dx=np.zeros((4),dtype=np.float32,order='F')
    dx[1]=_dx1
    dx[2]=_dx2
    dx[3]=_dx3
    for dir in range(1,4):
        alf_speed=np.sqrt(np.abs(bu[dir]*bd[dir]/(rho+ug+bsq+(5/3-1)*ug)))
        vrot=np.sqrt((uu[3]*ud[3]+uu[2]*ud[2]+uu[1]*ud[1]))/uu[0]
        omegarot=vrot/r
        wavelength=2*3.14*alf_speed/omegarot
        delta_x=dx[dir]*np.sqrt(gcov[dir,dir])
        Q[dir]=wavelength/delta_x
        Q[dir]=np.nan_to_num(Q[dir])
        Q_avg[dir]=((gdet*rho*Q[dir]*(r>10.0)*(rho>0.01)).sum(-1).sum(-1).sum(-1))/((gdet*rho*(r>10.0)*(rho>0.01)).sum(-1).sum(-1).sum(-1))    
    
#Print (and plot) precession angle as function of time
def print_tilt_evolution(first_dump, last_dump, step, plot):
    tilt_t=np.zeros(((last_dump-first_dump)/step),dtype=np.float32,order='F')
    precession_t=np.zeros(((last_dump-first_dump)/step),dtype=np.float32,order='F')
    time=np.zeros(((last_dump-first_dump)/step),dtype=np.float32,order='F')
    for i in range (first_dump,last_dump,step):
        calc_precesion_fast(i,1)
        precession_t[(i-first_dump)/step]=angle_prec[0,50]
        tilt_t[(i-first_dump)/step]=angle_tilt[0,50]
        time[(i-first_dump)/step]=t
        print i
    if(plot==1):
        fig= plt.figure(figsize=(12,12))
        plt.plot(time,precession_t,color="red",lw=2)
        plt.plot(time,tilt_t,color="blue",lw=2)
        plt.xlim(time.min(),time.max())
        plt.ylim(0,80)
        plt.xlabel(r"$r(R_{G})$",size=30)
        plt.ylabel(r"Precession angle $\gamma$",size=30)
        plt.legend(loc="upper right",frameon=False,ncol=4)

#Plot aspect ratio of jcell from the polar axis
def plot_aspect(jcell=50):
    aspect = _dx1*dxdxp[1,1,:,:,:,0]/(r[:,:,:,0]*(_dx2*dxdxp[2,2,:,:,:,0]))
    for i in range(0,nb):
        plt.plot( r[i,:,jcell], aspect[i,:,jcell] )
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.xlabel(r"$\log_{10}r/R_{g}$")
    plt.ylabel(r"$dz/dR$")
    plt.savefig("aspect_ratio.png",dpi=300)
    
#Print precession angle as function of radius
def plot_precangle():
    fig= plt.figure(figsize=(6,6))
    for i in range(0,1):
        plt.plot(r[i,:,0,0],angle_prec[i],color="blue",label=r"S25A93",lw=2)
    plt.xlim(0,150)
    plt.ylim(0,60)
    plt.xlabel(r"$r(R_{G})$",size=30)
    plt.ylabel(r"${\rm Precession\ angle\ } \gamma$",size=30)
    plt.savefig("GammavsR0.png",dpi=300)

#Calculate and plot surface density
def plot_SurfaceDensity():
    SD=(rho*np.sqrt(gcov[2,2])*_dx2).sum(3).sum(2)
    plt.plot(np.log10(r[0,:,bs2/2,0]),np.log10(SD[0]))
    plt.xlim(0,4)
    plt.ylim(0,5)
    plt.xlabel(r"$\rm r(R_{G})$",size=30)
    plt.ylabel(r"\rm Surface density",size=30)
    plt.savefig("SD.png",dpi=300)
    
#Print tilt angle as function of radius
def plot_tiltangle():
    fig= plt.figure(figsize=(6,6))
    for i in range(0,nb):
        plt.plot(r[i,:,0,0],angle_tilt[i],color="blue",label=r"S25A93", lw=2)
    plt.xlim(0,150)
    plt.ylim(0,45)
    plt.xlabel(r"$\rm r(R_{G})$",size=30)
    plt.ylabel(r"\rm Tilt $\alpha$",size=30)
    plt.savefig("TiltvsR0.png",dpi=300)
    
def get_longest_path_vertices(cs, index):
    maxlen = 0
    maxind = -1
    paths = cs.collections[0].get_paths()
    for i,p in enumerate(paths):
        lenp = len(p.vertices)
        if lenp > maxlen:
            maxlen = lenp
            maxind = index
    if maxind < 0:
        print( "No paths found, using default one (0)" )
        maxind = 0
    print maxind
    return cs.collections[0].get_paths()[maxind].vertices

#Precalculates the parameters along the jet's field lines
def precalc_jetparam():
    faraday_new()
    Tcalcud_new()
    global ci, cj ,cr, cfitr,cbckeck,cbunching,cresult,cuu0,ch ,ceps,comega,cmu,csigma,csigma1,csigma2,cbsq, cbsqorho, cbsqoug,crhooug,chm87,cBpol,cuupar
    import scipy.ndimage as ndimage
    nb2d=1
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
    #cs=plc_new(aphi,levels=(0.55*0.65*aphi.max(),),xcoord=ti, ycoord=tj,xy=0,colors="red")

    nr=0 #number of radial lines
    cd = []
    cdi = []
    cdj = []
    vd = []
    Bpold = []
    cuu0d = []
    cugd = []
    chd = []
    for ri in range(0,nr):
        cd.append([])
        cdi.append([])
        cdj.append([])
        vd.append([])
        Bpold.append([])
        cuu0d.append([])
        cugd.append([])
        chd.append([])
        for i in range(0,nb2d):
            cd[ri].append(i+ri)
            cdi[ri].append(i+ri)
            cdj[ri].append(i+ri)
            vd[ri].append(i+ri)
            Bpold[ri].append(i+ri)
            cuu0d[ri].append(i+ri)
            cugd[ri].append(i+ri)
            chd[ri].append(i+ri)
    index=[0,0,2,2,0,0,0,0,0,0,0,0,0,0,0]
    for ri in range (0,nr):
        cd[ri]=plc_new(np.log10(r),levels=(ri+1.0,),colors="red",xcoord=ti, ycoord=tj,xy=0)

    for i in range (0,nb2d):
        if(tk[i,0,0,0]==0):


            for ri in range (0,nr):
                vd[ri][i]=get_longest_path_vertices(cd[ri][i], index[i])
                cdi[ri][i] = vd[ri][i][:,0]
                cdj[ri][i] = vd[ri][i][:,1]

                #Bpold[ri][i]=ndimage.map_coordinates(Bpol[i,:,:,0],np.array([cdi[ri],cdj[ri]]),order=1,mode="nearest")
                cuu0d[ri][i]=ndimage.map_coordinates((uu[0])[i,:,:,0],np.array([cdi[ri],cdj[ri]]),order=1,mode="nearest")
                chd[ri][i] = ndimage.map_coordinates(h[i,:,:,0],np.array([cdi[ri],cdj[ri]]),order=1,mode="nearest")
                cugd[ri][i]=ndimage.map_coordinates((ug)[i,:,:,0],np.array([cdi[ri],cdj[ri]]),order=1,mode="nearest")

            k=plc_new(aphi,levels=(0.25*0.65*aphi.max(),),xcoord=ti,i2=i, ycoord=tj,xy=0,colors="red")
            v[i] = get_longest_path_vertices(k, index[i])

            ci[i] = v[i][:,0]
            cj[i] = v[i][:,1]
            nu=1.2
            #cfitr[i]=ndimage.map_coordinates(((Bpol/Bpol+(omegaf2*r*np.sin(h))**2)**0.5)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #csigma1[i]=ndimage.map_coordinates((np.abs(mu/(omegaf2*r*np.sin(h))))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #csigma2[i]=ndimage.map_coordinates((mu*h/3.5)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")

            #cbcheck[i]=ndimage.map_coordinates((Bpol**2/(bsq-Bpol**2))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #cbcheck[i]=ndimage.map_coordinates(((bsq-Bpol**2)/rho)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cbunching[i] = ndimage.map_coordinates((3.14*(r*np.sin(h))**2*Bpol/(3.14*(r*np.sin(h))**2*Bpol)[i,500,0,0])[i,:,:,0],np.array([ci[i],cj[i]]),order=1,mode="nearest")
            #ccurrent[i]= ndimage.map_coordinates((np.sqrt(np.abs(B[3]*B[3]*gcov[3,3]+2*B[3]*B[1]*gcov[3,1]+2*B[3]*B[2]*gcov[3,2]+
             #                       2*B[3]*B[0]*gcov[3,0]))*r*np.sin(h))[i,:,:,0],np.array([[ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cresult[i]= ndimage.map_coordinates((sigma**-0.5*uu[0])[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cr[i] = ndimage.map_coordinates(r[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            ceps[i] = ndimage.map_coordinates((rho*uu[1]/B[1])[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            ch[i] = ndimage.map_coordinates(h[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cuu0[i] = ndimage.map_coordinates((uu[0])[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            comega[i]=ndimage.map_coordinates((omegaf2)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cmu[i] = ndimage.map_coordinates(mu[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            crho[i] = ndimage.map_coordinates(rho[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            crhooug[i] = ndimage.map_coordinates((rho/ug)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cug[i] = ndimage.map_coordinates(ug[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            csigma[i] = ndimage.map_coordinates(sigma[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #cbsq[i]=ndimage.map_coordinates((bsq)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #cbsqorho[i] = ndimage.map_coordinates((bsq/rho)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            cbsqoug[i] = ndimage.map_coordinates((bsq/ug)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            crhooug[i] = ndimage.map_coordinates((rho/ug)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")

            chm87[i] = ndimage.map_coordinates((r**(-0.42)/3.8)[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #cBpol=ndimage.map_coordinates(Bpol[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            #cuupar[i]=ndimage.map_coordinates((uu[1]*np.sqrt(gcov[1,1]))[i,:,:,0],np.array([ci[i]-ti[i,0,0,0],cj[i]-tj[i,0,0,0]]),order=1,mode="nearest")
            plt.plot(ci[i],cj[i],label=r"$\gamma$",color="blue",lw=1) 
            
def plt_jetparam():
    global which
    nb2d=1
    clen = [None] * nb2d
    ind = [None] * nb2d
    ind1 = [None] * nb2d
    ind2 = [None] * nb2d
    inds = [None] * nb2d
    indmax = [None] * nb2d
    indmax1 = [None] * nb2d
    indmax2 = [None] * nb2d

    whichpoles = [0,1]
    lws = [3,2]

    fig=plt.figure(figsize=(12,8))
    plt.tick_params('both', length=5, width=2, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    firsttime = 1

    maxi=0
    
    for i in range (0,nb2d):
        maxi=np.max(ci[i],maxi)
    for i in range (0,nb2d):
        if(tk[i,0,0,0]==0):

            clen[i] = len(cr[i])
            ind[i] = np.arange(clen[i])
            #indmax[i] = (np.where(ci[i] < maxi))[0][0]
            indmax1[i] = (np.where(ci[i][:clen[i]/2] == np.max(ci[i][:clen[i]/2])))[0][0]
            indmax2[i] = (np.where(ci[i][clen[i]/2:] == np.max(ci[i][clen[i]/2:])))[0][0]+clen[i]/2
            ind1[i] = ind[i] < indmax1[i]
            ind2[i] = ind[i] > indmax2[i]
            inds[i] = [ind1[i], ind2[i]]

            for whichpole,lw in zip(whichpoles,lws):
                which = inds[i][whichpole]

                #plt.plot(cr[i][which],cbunching[i][which]/10000000,label=r"$a_{fp}$",color="cyan",lw=lw)
                #plt.plot(cr[i][which],current[i][which],label=r"$I$",color="purple",lw=lw)
                #plt.plot(cr[i][which],cresult[i][which],label=r"$\delta$",color="orange",lw=lw)
                plt.plot(cr[i][which],cuu0[i][which],label=r"$\gamma$",color="red",lw=lw)
                plt.plot(cr[i][which],csigma[i][which],label=r"$\sigma$",color="green",lw=lw)
                #plt.plot(cr[i][which],csigma1[i][which],label=r"$\sigma_{1}$",color="grey",lw=lw)
                #plt.plot(cr[i][which],cbsqorho[i][which],label=r"$b^2/\rho$",color="cyan",lw=lw)
                #plt.plot(cr[i][which],crhooug[i][which],label=r"$\rho/u_{g}$",color="purple",lw=lw)
                plt.plot(cr[i][which],cmu[i][which],label=r"$\mu$",color="blue",lw=lw)
                plt.plot(cr[i][which],1000*ceps[i][which],label=r"$\mu$",color="cyan",lw=lw)

                #plt.plot(cr[i][which],comega[i][which],label=r"$\omega$",color="cyan",lw=lw)
                #plt.plot(cr[i][which],10000*cug[i][which],label=r"ug",color="pink",lw=lw)
                #plt.plot(cr[i],cbsqorho[i],label=r"$10^4\rho$",color="magenta",lw=lw)
                #plt.plot(cr[i][which],cbcheck[i][which],label="Bpol",color="magenta",lw=lw)
                #plt.plot(cr[i][which],0.5*cr[which]**(-2.5*5/3),label=r"$90r^{-3/2}$",color="orange",lw=lw)
                plt.plot(cr[i][which],2*chm87[i][which],label=r"$\theta_{M87}$",color="purple",lw=lw)
                #plt.plot(cr[i][which],cBpol[i][which],label=r"$\theta_{M87}$",color="yellow",lw=lw)
                if whichpole == 0:
                    plt.plot(cr[i][which],ch[i][which]*cresult[i][which],label=r"$\gamma*\theta/\sigma^{0.5}$",color="orange",lw=lw)
                    plt.plot(cr[i][which],ch[i][which],label=r"$\theta_{Matthew}$",color="black",lw=lw)
                    #plt.plot(cr[i][which],cmu[i][which]*ch[i][which]/3.84,label=r"$\sigma_{2}$",color="cyan",lw=lw)
                else:
                    plt.plot(cr[i][which],(np.pi-ch[i][which])*cresult[i][which],label=r"$\gamma*\theta/\sigma^{0.5}$",color="orange",lw=lw)
                    plt.plot(cr[i][which],np.pi-ch[i][which],label=r"$\theta$",color="black",lw=lw)
                    #plt.plot(cr[i][which],cmu[i][which]*(np.pi-ch[i][which])/3.84,label=r"$\sigma_{2}$",color="cyan",lw=lw)
                if firsttime==1:
                    plt.legend(loc="upper right",frameon=False,ncol=4)
                    #plt.xlim(rhor,t+100)
                    plt.ylim(1e-3,1e3)
                    axis_font = {'fontname':'Arial', 'size':'24'}
                    plt.tick_params(axis='both', which='major', labelsize=24)
                    plt.tick_params(axis='both', which='minor', labelsize=24)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel(r"$\log_{10}(r/R_{g})$", fontsize=30)
                    plt.grid(b=1)
                firsttime = 0
    plt.savefig("evolution.png",dpi=300)

def plt_jetparam_trans():
    R=[None]*nr
    powexp=2 #set to 10 for real plots to lower for debuggin
    for i in range (0,nb2d):
        if(tk[i,0,0,0]==0):
            for ri in range (3,4):
                j=0
                while cr[i][j]<20000:
                    j+=1 
                R[ri]=plt.scatter(np.sin(chd[ri][i])/np.sin(ch[i][j]),np.log10(Bpold[ri][i]),color="red",lw=1)
    '''plt.legend((R[0], R[1], R[2], R[3], R[4]),
               (r"$r=$10^1$ R_{g}$",r"$r=$10^2$ R_{g}$",r"$r=$10^3$ R_{g}$", r"$r=$10^4$ R_{g}$",r"$r=$10^5$ R_{g}$"),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=16)'''
    plt.xlim(0,0.99)
    plt.ylim(-5,-3)
    plt.xlabel(r"$\log_{10}R/R_{edge}$")
    plt.ylabel(r"$\log_{10}B_{p}$")
    plt.savefig("core.png",dpi=300)
    
def export_visit(dump):
    export_vtk(dump)
    import sys
    sys.path.insert(0,"C:\\Users\\Matthew\\AppData\\Local\\Programs\\LLNL\\VisIt 2.10.0\\lib\\site-packages")
    import visit
    visit.Launch()
    visit.OpenDatabase("rho%d.vtk" %dump, 0) 
    visit.DeleteAllPlots()
    visit.AddPlot("Pseudocolor", "rho")
    iso_atts = visit.IsosurfaceAttributes()
    iso_atts.contourMethod = iso_atts.Value
    iso_atts.variable = "rho"
    visit.AddOperator("Isosurface")
    for i in range(1):
        iso_atts.contourValue = (-2)
        visit.SetOperatorOptions(iso_atts)
        # set basic save options
        swatts = visit.SaveWindowAttributes()
        #
        # The 'family' option controls if visit automatically adds a frame number to 
        # the rendered files. For this example we will explicitly manage the output name.
        #
        swatts.family = 0
        #
        # select PNG as the output file format
        #
        swatts.format = swatts.PNG 
        #
        # set the width of the output image
        #
        swatts.width = 1024 
        #
        # set the height of the output image
        #
        swatts.height = 1024
        visit.DrawPlots()
        swatts.fileName = "rho_%d.png" %dump
        visit.SetSaveWindowAttributes(swatts)
        # For moviemaking, you'll need to save off the image
        visit.SaveWindow()

def make_movie():
    input_pattern = "rho_%d.png"
    output_movie = "rho.wmv"
    encoding.encode(input_pattern,output_movie,fdup=4)
            
def merge_dump(dir): 
    global n_ord, n_active_total  
    os.chdir(dir) #hamr
    destination = open('new_dump', 'wb')
    for i in glob.glob("dumpdiag*"):
        os.remove(i)
    length=len(os.listdir(dir))
    print "Length", length, "n_total", n_active_total, "n_ord[5]", n_ord[5], "dir", dir

    for i in range(0,n_active_total):
        shutil.copyfileobj(open('dump%d' %n_ord[i],'rb'), destination)
    destination.close()
    
def merge_dumps(dir): 
    dumps=0
    os.chdir(dir) #hamr
    rblock_new()
    while (os.path.isfile(dir + "/dumps%d/parameters" %dumps)):
        dumps=dumps+1 
    if(rank==0):
        print "nr_files", dumps
    
    for i in range(0,dumps):
        if (i%numtasks==rank):
            os.chdir(dir) #hamr
            rpar_new(i)
            merge_dump(dir + "/dumps%d" %i)

def backup_dump(dir1, dir2, dir3): 
    global n_ord, n_active_total  
    
    os.makedirs(dir2)
    os.chdir(dir2) #hamr
    destination2 = open('parameters', 'wb')
    
    os.makedirs(dir3)
    os.chdir(dir3) #hamr
    destination1 = open('new_dump', 'wb')
    destination3 = open('parameters', 'wb')
    
    os.chdir(dir1) #hamr
    length=len(os.listdir(dir1))  
    
    for i in range(0,n_active_total):
        os.chdir(dir2) #hamr
        destination = open('dump%d' %n_ord[i],'wb')
        os.chdir(dir1) #hamr
        shutil.copyfileobj(open('dump%d' %n_ord[i],'rb'), destination)
        destination.close()
    shutil.copyfileobj(open('new_dump','rb'), destination1)
    shutil.copyfileobj(open('parameters','rb'), destination2)
    shutil.copyfileobj(open('parameters','rb'), destination3)
    destination1.close()
    destination2.close()
    destination3.close()
    print "Length", length, "n_total", n_active_total, "n_ord[5]", n_ord[5], "dir", dir1

def backup_dumps(dir1, dir2, dir3): 
    dumps=0
    os.chdir(dir1) #hamr
    rblock_new()
    while (os.path.isfile(dir1 + "/dumps%d/parameters" %dumps)):
        dumps=dumps+1 
    if(rank==0):
        print "nr_files", dumps
        
    for i in range(0,dumps,10):
        if ((i/10)%numtasks==rank):
            os.chdir(dir1) #hamr
            rpar_new(i)
            backup_dump(dir1 + "/dumps%d" %i,dir2 + "/dumps%d" %i,dir3 + "/dumps%d" %i )
import glob
def delete_dump(dir,start,end,stride):
    dumps=0
    os.chdir(dir) #hamr
    rblock_new()
    while (os.path.isfile(dir + "/dumps%d/parameters" %dumps)):
        dumps=dumps+1 
    if(rank==0):
        print "nr_files", dumps
    
    for i in range(start,end,stride):
        if (i%numtasks==rank):
            os.chdir(dir) #hamr
            rpar_new(i)
            dir2=dir + "/dumps%d" %i
            os.chdir(dir2)
            for j in glob.glob("dump*"):
                os.remove(j)
            os.chdir(dir)

def plot_dumps(dir1, start, end, stride): 
    global lowres,axisym
    lowres=2
    axisym=1
    os.chdir(dir1) #hamr
    rblock_new()
    rpar_new(start)
    rgdump_new(lowres)
    
    for i in range(start,end,stride):
        if ((i)%numtasks==rank):
            print i
            if (os.path.isfile(dir1 + "/dumps%d/parameters" %i)):
                print "good%d"%i
                #os.chdir(dir1) #hamr
                rpar_new(i)
                rdump_new(i,lowres)
                griddataall()
                
                plc_cart0(25,0,'rho%d.png' %i)
                plc_cart(25,0,'bsq%d.png' %i)
            else:
                print "bad%d"%i

def plc_new_xy(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    global r,h,ph
    l = [None] * nb2d
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k',0)
    mirrory = kwargs.pop('mirrory',0)
    #cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled',False)
    xy = kwargs.pop('xy',0)
    xmax = kwargs.pop('xmax',10)
    ymax = kwargs.pop('ymax',5)
    z=kwargs.pop('z',0)
    if ax is None:
        ax = plt.gca()
    if isfilled:
        for i in range(0,nb):
            res = ax.contourf(xcoord[i,:,int(bs2/2),:],ycoord[i,:,int(bs2/2),:],myvar[i,:,int(bs2/2),:],nc,**kwargs)
    else:
        for i in range(0,nb):
            res = ax.contour(xcoord[i,:,int(bs2/2),:],ycoord[i,:,int(bs2/2),:],myvar[i,:,int(bs2/2),:],nc,**kwargs)  
    if( cb == True): #use color bar
         plt.colorbar(res,ax=ax)
    if (xy==1):
        if (xy==1):
            plt.xlim(-xmax,xmax)
        else:
            plt.xlim(0.0,xmax)
        plt.ylim(-ymax,ymax)
    return res

def plc_cart_xy1(rmax, offset, name):
    fig=plt.figure(figsize=(64,32))
    
    ph[:,:,:,0]=0.0*ph[:,:,:,0]
    ph[:,:,:,bs3-1]=0.0*ph[:,:,:,0]
    
    X = np.multiply(r, np.sin(ph))
    Y = np.multiply(r, np.cos(ph)) 
    #rmax = np.amax(r)
    
    avg=0.5*(rho[:,:,:,0]+rho[:,:,:,bs3-1])
    rho[:,:,:,bs3-1]=avg
    rho[:,:,:,0]=avg
    
    avg=0.5*(bsq[:,:,:,0]+bsq[:,:,:,bs3-1])
    bsq[:,:,:,bs3-1]=avg
    bsq[:,:,:,0]=avg
    
    plotmax = int(rmax*np.sqrt(2))
    #print np.amax(r)

    ilim = len(r[0,:,0,0])-1
    #print ilim
    for i in range(len(r[0,:,0,0])):
        if r[0,i,0,0] > plotmax:
            ilim = i
            break
    gammamax = np.amax(uu[0,0,0:ilim])
    #print "gamma max:", gammamax
    
    plt.subplot(1,2,1)
    res=plc_new_xy(np.log10(rho),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=X,ycoord=Y, xy=1,z=offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$",fontsize=48)
    plt.ylabel(r"$y / R_g$",fontsize=48)
    plt.title(r"log$(\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
   
    plt.subplot(1,2,2)
    res=plc_new_xy(np.log10(rho),levels=np.arange(-8,3,0.05),cb=0,isfilled=1,xcoord=X,ycoord=Y, xy=1,z=offset, xmax=rmax*5, ymax=rmax*5)
    plt.xlabel(r"$x / R_g$",fontsize=48)
    plt.ylabel(r"$y / R_g$",fontsize=48)
    plt.title(r"log$(\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
    plt.savefig(name,dpi=30)    
    plt.close('all')

def plc_cart_xy2(rmax, offset, name):
    fig=plt.figure(figsize=(64,32))
    
    ph[:,:,:,0]=0.0*ph[:,:,:,0]
    ph[:,:,:,bs3-1]=0.0*ph[:,:,:,0]
    
    X = np.multiply(r, np.sin(ph))
    Y = np.multiply(r, np.cos(ph)) 
    #rmax = np.amax(r)
    
    avg=0.5*(rho[:,:,:,0]+rho[:,:,:,bs3-1])
    rho[:,:,:,bs3-1]=avg
    rho[:,:,:,0]=avg
    
    avg=0.5*(bsq[:,:,:,0]+bsq[:,:,:,bs3-1])
    bsq[:,:,:,bs3-1]=avg
    bsq[:,:,:,0]=avg
    
    plotmax = int(rmax*np.sqrt(2))
    #print np.amax(r)

    ilim = len(r[0,:,0,0])-1
    #print ilim
    for i in range(len(r[0,:,0,0])):
        if r[0,i,0,0] > plotmax:
            ilim = i
            break
    gammamax = np.amax(uu[0,0,0:ilim])
    #print "gamma max:", gammamax
    
    plt.subplot(1,2,1)
    res=plc_new_xy(np.log10(bsq/rho),levels=np.arange(-8,2,0.05),cb=0,isfilled=1,xcoord=X,ycoord=Y, xy=1,z=offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$",fontsize=48)
    plt.ylabel(r"$y / R_g$",fontsize=48)
    plt.title(r"log$(b^2/\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
   
    plt.subplot(1,2,2)
    res=plc_new_xy(np.log10(bsq/rho),levels=np.arange(-8,2,0.05),cb=0,isfilled=1,xcoord=X,ycoord=Y, xy=1,z=offset, xmax=rmax*10, ymax=rmax*10)
    plt.xlabel(r"$x / R_g$",fontsize=48)
    plt.ylabel(r"$y / R_g$",fontsize=48)
    plt.title(r"log$(b^2/\rho)$ at %d $R_g/c$" %t,fontsize=60)
    ax=plt.gca()
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)
    plt.savefig(name,dpi=30)    
    plt.close('all')
    
def post_process(dir, dump_start, dump_end, dump_stride):
    global axisym, lowres,REF_1, REF_2,REF_3
    os.chdir(dir)
    lowres = 6
    axisym=1
    REF_1=1
    REF_2=1
    REF_3=1
    #rblock_new(0)
    #rpar_new(0)
    #rgdump_new(lowres)
    f = open(dir + "/post_process%d.txt" %rank, "w")
    if(rank==0):
        f.write("t, Mtot, Mdot, Edot, phibh, angle_tilt_disk, angle_prec_disk,angle_tilt_corona,angle_prec_corona, tilt_JU, prec_JU, tilt_JD, prec_JD \n")
        if (os.path.isdir(dir +"/images")==0):
            os.makedirs(dir +"/images")
    dir_images=dir +"/images"
    if( cluster==1):
        comm.barrier()
    for i in range(0, (dump_end-dump_start)/dump_stride, 1):
        i2=dump_start+i*dump_stride
        if (os.path.isfile(dir + "/dumps%d/parameters" %i2)):
            if((i)%numtasks==rank):               
                rblock_new(i2)
                rpar_new(i2)
                rgdump_new(lowres)
                rdump_new(i2,lowres)
                griddataall()
                
                calc_Mdot()
                calc_Mtot()
                calc_phibh()   
                Tcalcuu_new()
                calc_jet_tot()
                calc_precesion_accurate_disk(i2,1)
                set_cart=1
                calc_precesion_accurate_corona(i2,1)
               
                f.write("%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g \n" % (t, Mtot[0], Mdot[0,5], Edot[0,5], phibh[0,5],angle_tilt_disk[0], angle_prec_disk[0],angle_tilt_corona[0], angle_prec_corona[0], angle_jet[0], angle_jet[1], angle_jet[2], angle_jet[3]))
                
                plc_cart_xy1(8,0,dir_images + "/bsqxy%d.png" %i2) 
                plc_cart_xy2(8,0,dir_images + "/rhoxy%d.png" %i2)
                plc_cart(8,0,dir_images + "/bsq%d.png" %i2) 
                plc_cart0(8,0,dir_images + "/rho%d.png" %i2)
                if(rank==0):
                    print "Post processed %d \n" %i2
    f.close()
    if(cluster==1):        
        comm.barrier()
    if (rank==0):
        print "Merging post processed files and cleaning up" 
        f_tot = open(dir + "/post_process.txt", "w")
        for i in range(0,numtasks):
            shutil.copyfileobj(open(dir +"/post_process%d.txt" %i,'rb'), f_tot)
            os.remove(dir + "/post_process%d.txt" %i)
        f_tot.close()

dirr="/u/sciteam/liska/scratch/NEW/TDISKS25A93T45S"
post_process(dirr,0,9500,1)

