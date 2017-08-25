from __future__ import division
from IPython.display import display

import os, sys, gc
import sympy as sym
#from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pdb
import operator
import matplotlib
from matplotlib.gridspec import GridSpec
#add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

from matplotlib import rc
rc('text', usetex=True)
font = { 'size'   : 20}
rc('font', **font)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20) 
#rc('xlabel', **font) 
#rc('ylabel', **font) 

legend = {'fontsize': 20}
rc('legend',**legend)

axes = {'labelsize': 20}
rc('axes', **axes)

fontsize = 20

#%matplotlib inline


from sympy.interactive import printing
printing.init_printing(use_latex=True)

#For ODE integration
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def avg(v):
    return( 0.5*(v[1:]+v[:-1]) )

def der(v):
    return( (v[1:]-v[:-1]) )

def myfloat(f,acc="float64"):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1 or acc=="float32":
        return( np.float32(f) )
    else:
        return( np.float64(f) )
              
#read in a grid file
def rg3D(dump):
    global t,nb,nx,ny,nz,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp
    #read image
    fin = open( "dumps/" + dump, "rb" )
    t =  np.fromfile(fin, dtype=np.float64, count=1, sep='')
    nx = np.fromfile(fin, dtype=np.int, count=1, sep='')
    ny = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nz = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb=4
    n_rows = np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_columns =np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_stacks = np.fromfile(fin, dtype=np.int, count=1, sep='')
    fin.seek(3*8,1)
    _dx1=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx2=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx3=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(1*8+1*4,1)
    a=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gam=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(5*8+6*4,1)
    Rin=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    Rout=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    hslope=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    R0=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gd=np.fromfile(fin, dtype=np.float64, count=58*nb*nx*ny*nz, sep='')
    gd=gd.reshape((-1,nz*ny*nx*nb), order='F')
    gd=myfloat(gd.transpose(1,0))
    gd=gd[np.lexsort((gd[:,1], gd[:,0]))]
    gd=myfloat(gd.transpose(1,0))
    gd=gd.reshape((-1,nz,ny,nx*nb,1), order='F')
    nx=nx*nb
    nb=1
    gd=myfloat(gd.transpose(0,4,3,2,1))
    gd=gd[:,:,:,:,None]

    #make into 3D array: from (quantity,r,theta) to (quantity,r,theta,phi) with 1 cell in phi

    #clean up memory
    ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:].view()
    gv3 = gd[9:25].view().reshape((4,4,nb,nx,ny,nz),order='F').transpose(1,0,2,3,4,5)
    gn3 = gd[25:41].view().reshape((4,4,nb,nx,ny,nz),order='F').transpose(1,0,2,3,4,5)
    gcov = gv3
    gcon = gn3
    guu = gn3
    gdd = gv3
    gdet = gd[41]
    drdx = gd[42:58].view().reshape((4,4,nb,nx,ny,nz),order='F').transpose(1,0,2,3,4,5)
    dxdxp = drdx
         
def rdfmss(dump):
    global connected
    #read image
    
    fin = open( "dumps/fmss", "rb" )
  
    gd=np.fromfile(fin, dtype=np.float64, count=4*nx*ny*nz, sep='')
    gd=gd.reshape((-1,nz*ny*nx), order='F')
    gd=myfloat(gd.transpose(1,0))
    gd=gd[np.lexsort((gd[:,1], gd[:,0]))]
    gd=myfloat(gd.transpose(1,0))
    gd=gd.reshape((-1,nz,ny,nx), order='F')
    gd=myfloat(gd.transpose(0,3,2,1))
    #gd=gd[:,:,:,:,None]
    connected = gd[3]

def rd3D(dump):
    global gd, t,nb, nx,ny,nz,n_rows, n_columns, n_stacks, _dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor
    
    global nb2d
    
    
    #read image
    if dump==0:
        fin = open( "dumps/dump000", "rb" )
    elif dump<10:
        fin = open( "dumps/dump00%d" %dump, "rb" )
    elif dump<100:
        fin = open( "dumps/dump0%d" %dump, "rb" )
    else:
        fin = open( "dumps/dump%d" %dump, "rb" )
    t =  np.fromfile(fin, dtype=np.float64, count=1, sep='')

    nx = np.fromfile(fin, dtype=np.int, count=1, sep='')
    ny = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nz = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb=4
    nb2d=nb
    n_rows = np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_columns =np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_stacks = np.fromfile(fin, dtype=np.int, count=1, sep='')
    fin.seek(3*8,1)
    _dx1=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx2=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx3=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(1*8+1*4,1)
    a=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gam=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(5*8+6*4,1)
    Rin=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    Rout=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    hslope=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    R0=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gd=np.fromfile(fin, dtype=np.float64, count=39*nb*nx*ny*nz, sep='')
    gd=gd.reshape((-1,nz*ny*nx*nb), order='F')
    #gd=myfloat(gd.transpose(1,0))
    #gd=gd[np.lexsort((gd[:,1], gd[:,0]))]
    #gd=myfloat(gd.transpose(1,0))
    gd=gd.reshape((-1,nz,ny,nx, nb), order='F')
    gd=myfloat(gd.transpose(0,4,3,2,1))
    #gd=gd[:,:,:,:,None]
    ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:,:].view() 
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[11:14]
    B[1:4] = gd[14:17]
    divb = gd[17]
    uu = gd[18:22]
    ud = gd[22:26]
    bu = gd[26:30]
    bd = gd[30:34]
    bsq=divb
    for i in range(0, nb):
        bsq[i] = mdot(bu[:,i],bd[:,i])
    v1m,v1p,v2m,v2p=gd[34:38]
    gdet=gd[38]
    rhor = 1+(1-a**2)**0.5
    if "guu" in globals():
        #lapse
        alpha = (-guu[0,0])**(-0.5)
        
def rd3D2(dump):
    global gd, t,nb, nx,ny,nz,n_rows, n_columns, n_stacks, _dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor
    
    global nb2d
    
    
    #read image
    if dump==0:
        fin = open( "dumps/dump000", "rb" )
    elif dump<10:
        fin = open( "dumps/dump00%d" %dump, "rb" )
    elif dump<100:
        fin = open( "dumps/dump0%d" %dump, "rb" )
    else:
        fin = open( "dumps/dump%d" %dump, "rb" )
    t =  np.fromfile(fin, dtype=np.float64, count=1, sep='')

    nx = np.fromfile(fin, dtype=np.int, count=1, sep='')
    ny = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nz = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb = np.fromfile(fin, dtype=np.int, count=1, sep='')
    nb=4
    nb2d=nb
    n_rows = np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_columns =np.fromfile(fin, dtype=np.int, count=1, sep='')
    n_stacks = np.fromfile(fin, dtype=np.int, count=1, sep='')
    fin.seek(3*8,1)
    _dx1=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx2=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    _dx3=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(1*8+1*4,1)
    a=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gam=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    fin.seek(5*8+6*4,1)
    Rin=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    Rout=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    hslope=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    R0=np.fromfile(fin, dtype=np.float64, count=1, sep='')
    gd=np.fromfile(fin, dtype=np.float64, count=39*nb*nx*ny*nz, sep='')
    gd=gd.reshape((-1,nz*ny*nx*nb), order='F')
    gd=myfloat(gd.transpose(1,0))
    gd=gd[np.lexsort((gd[:,1], gd[:,0]))]
    gd=myfloat(gd.transpose(1,0))
    gd=gd.reshape((-1,nz,ny,nx*nb, 1), order='F')
    gd=myfloat(gd.transpose(0,4,3,2,1))
    nb=1
    nb2d=1
    nx=nx*4
    #gd=gd[:,:,:,:,None]
    ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:,:].view() 
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[11:14]
    B[1:4] = gd[14:17]
    divb = gd[17]
    uu = gd[18:22]
    ud = gd[22:26]
    bu = gd[26:30]
    bd = gd[30:34]
    bsq=divb
    for i in range(0, nb):
        bsq[i] = mdot(bu[:,i],bd[:,i])
    v1m,v1p,v2m,v2p=gd[34:38]
    gdet=gd[38]
    rhor = 1+(1-a**2)**0.5
    if "guu" in globals():
        #lapse
        alpha = (-guu[0,0])**(-0.5)
        
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
    print gd1.shape
    for n in range(0,nmax):
        block[n][AMR_REFINED]=gd1[n]
        block[n][AMR_ACTIVE]=gd2[n]
    
    i=0;
    for n in range(0,nmax):
        if block[n][AMR_ACTIVE]==1:
            n_ord[i]=n
            i+=1
    
def rgdump_new():
    global ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,dxdxp, alpha
    
    #Allocate memory
    ti = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    tj = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    tk = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    x1 = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    x2 = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    x3= np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    r = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    h = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    ph = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    gcov= np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float64)
    gcon= np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float64)
    gdet= np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    dxdxp= np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float64)
    alpha = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    
    #Read in data for every block
    for n in range(0,n_active_total):
        fin = open("gdumps/gdump%d" %n_ord[n], "rb" )   
        gd=np.fromfile(fin, dtype=np.float64, count=58*bs3*bs2*bs1, sep='')
        gd=gd.reshape((-1,bs3*bs2*bs1), order='F')
        gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
        gd=myfloat(gd.transpose(0,3,2,1))
        #gd=gd[:,:,:,None]
        ti[n] =gd[0]
        tj[n] =gd[1]
        tk[n]=gd[2]
        x1[n]=gd[3]
        x2[n] =gd[4]
        x3[n]=gd[5]
        r[n] =gd[6]
        h[n] =gd[7]
        ph[n] =gd[8]
        gcov[:,:,n] = gd[9:25].view().reshape((4,4,bs1,bs2,bs3),order='F').transpose(1,0,2,3,4)
        gcon[:,:,n] = gd[25:41].view().reshape((4,4,bs1,bs2,bs3),order='F').transpose(1,0,2,3,4)
        gdet[n] = gd[41]
        dxdxp[:,:,n] = gd[42:58].view().reshape((4,4,bs1,bs2,bs3),order='F').transpose(1,0,2,3,4)
        alpha[n] = (-gcon[0,0,n])**(-0.5)
        
def rblock_new():
    global AMR_ACTIVE, AMR_LEVEL, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4,AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_ACTIVATED,AMR_HOT
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
    AMR_ACTIVATED= 22
    AMR_HOT =23
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
    fin = open("gdumps/grid", "rb")  
    nmax=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    #Allocate memory
    block = np.zeros((nmax,36),dtype=np.int32)
    n_ord = np.zeros((nmax),dtype=np.int32)
    
    gd = np.fromfile(fin, dtype=np.int32, count=36*nmax, sep='')
    gd = gd.reshape((nmax,36), order='F')
    block=gd
    
def rdump_new(dump):
    global rho,ug,uu,ud,B,bu,bd,gdet,bsq,bsq, nb2d
    
    #Allocate memory
    rho = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    ug = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    bsq = np.zeros((nb,bs1,bs2,bs3),dtype=np.float64)
    uu= np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float64)
    ud= np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float64)
    bu= np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float64)
    bd= np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float64)
    B= np.zeros((4,nb,bs1,bs2,bs3),dtype=np.float64)
    
    nb2d=0
            
    for n in range(0,n_active_total):
        #read image
        fin = open( "dumps%d/dump%d" %(dump,n_ord[n]), "rb" )
        gd=np.fromfile(fin, dtype=np.float32, count=9*bs1*bs2*bs3, sep='')
        gd=gd.reshape((-1,bs1*bs2*bs3), order='F')
        gd=gd.reshape((-1,bs3,bs2,bs1), order='F')
        gd=myfloat(gd.transpose(0,3,2,1))
        rho[n] = gd[0]
        ug[n] = gd[1]
        uu[:,n] = gd[2:6]
        ud[:,n]=mdot(gcov[:,:,n],uu[:,n])
        B[0,n]=0.0
        B[1:4,n] = gd[6:9]
        bu[0,n]=B[1,n]*mdot(uu[:,n],gcov[1,:,n])+B[2,n]*mdot(uu[:,n],gcov[2,:,n])+B[3,n]*mdot(uu[:,n],gcov[3,:,n])
        bu[1:4,n]=(B[1:4,n]+bu[0,n]*uu[1:4,n])/uu[0,n]
        bd[:,n]=mdot(gcov[:,:,n],bu[:,n])
        bsq[n] = mdot(bu[:,n],bd[:,n])
        if(tk[n,0,0,0]==0):
            nb2d+=1
    nb2d=nb[0]
            
        
def rdiag_new(dump):
    global divb,fail1,fail2
    
    #Allocate memory
    divb = np.zeros((nb,bs1,bs2,bs3),dtype=np.float32)
    fail1 = np.zeros((nb,bs1,bs2,bs3),dtype=np.float32)
    fail2 = np.zeros((nb,bs1,bs2,bs3),dtype=np.float32)
    
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
        
def mdot(a,b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
    """
    if (a.ndim == 3 and b.ndim == 3) or (a.ndim == 4 and b.ndim == 4):
          c = (a*b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
          #c = np.empty(np.amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)   
          c = np.empty((4,bs1,bs2,bs3),dtype=np.float64)   
          for i in range(a.shape[0]):
                c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
          #c = np.empty(np.amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype) 
          c = np.empty((4,bs1,bs2,bs3),dtype=np.float64)  
          for i in range(b.shape[1]):
                c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    return c

def psicalc():
    """
    Computes the field vector potential
    """
    daphi = -(gdet*B[1]).mean(-1)*_dx2
    aphi=daphi[:,:,::-1].cumsum(axis=2)[:,:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    return(aphi)

def psicalc2():
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

def psicalc_new():
    """
    Computes the field vector potential integrating from both poles to maintain accuracy.
    """
    global aphi, aphi_temp, n_ord
    daphi = (gdet*B[1])*_dx2
    aphi_temp=-daphi[:,:,::-1,:].cumsum(axis=2)[:,:,::-1,:]
    aphi_temp+=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi2_temp=daphi[:,:,:,:].cumsum(axis=2)[:,:,:,:]
    aphi_temp[:,:,:bs2[0]/2,:] = aphi2_temp[:,:,:bs2[0]/2,:]
    
    aphi = np.zeros((nb, bs1,bs2,bs3),dtype=np.float64)
    np.copyto(aphi, aphi_temp)
    for i in range(0,bs2):
        aphi_temp[:,:,i,:]=aphi_temp[:,:,bs2[0]-1,:]
    
    
    n_ord = np.zeros(nmax,dtype=np.int32)
    index=0
    for n in range(0,nmax):
        if block[n][AMR_ACTIVE]==0:
            n_ord[index]=n
            index+=1
    
    for n in range(0,n_active_total):
        pres=n_ord[n]
        while 1:
            nbr=-1
            if block[pres,AMR_NBR1]>=0:
                if block[block[pres,AMR_NBR1],AMR_ACTIVE]==1:
                    nbr=block[block[pres,AMR_NBR1],AMR_ACTIVE]
                    type=0
            if nbr==-1:
                break
            else:
                if type==0:
                    aphi[:,n_ord[n],:,:,:]+=aphi_temp[:,nbr,:,:,:]
                pres=nbr
       
    #return(aphi)
def psicalc_3D():
    daphi = (gdet*B[1])*_dx2/np.sqrt(gcov[3,3])*gdet.max()/gcov[3,3].max()
    aphi=-daphi[:,:,::-1,:].cumsum(axis=2)[:,:,::-1,:]
    aphi+=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi2=daphi[:,:,:,:].cumsum(axis=2)[:,:,:,:]
    aphi[:,:,:bs2[0]/2,:] = aphi2[:,:,:bs2[0]/2,:]
    
def plco(myvar,xcoord=None,ycoord=None,ax=None,**kwargs):
    global r,h,ph
    plt.clf()
    return plc(myvar,xcoord,ycoord,ax,**kwargs)

def plc(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    global r,h,ph
  
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
    if ax is None:
        ax = plt.gca()
    if xy==1:
        xcoord = np.log10(r* np.sin(h))
        ind=np.zeros_like(h)
        for i in range(0,nx):
            for j in range (0,ny):
                if(np.cos(h[i,j])>0.0):
                    ind[i,j]=1.0
                else:
                    ind[i,j]=-1.0
        ycoord = np.log10(r* np.abs(np.cos(h)))
        for i in range(0,nx):
            for j in range (0,ny):
                if(ycoord[i,j]<0.0):
                    ycoord[i,j]=0.0
        ycoord*=ind
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,0],myvar[:,:,:,0],nc,**kwargs)
    elif xy==2:
        xcoord = r* np.sin(ph)
        ycoord = r* np.cos(ph)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
    elif xy==3:
        xcoord = r* np.sin(h)
        ycoord = r* np.cos(h)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
    else:
        if isfilled:
            for i in range(0,nb):
                res = ax.contourf(xcoord[i,:,:,0],ycoord[i,:,:,0],myvar[i,:,:,0],nc,**kwargs)
        else:
            for i in range(0,nb):
                res = ax.contour(xcoord[i,:,:,0],ycoord[i,:,:,0],myvar[i,:,:,0],nc,**kwargs)
    if( cb == True): #use color bar
        plt.colorbar(res,ax=ax)
    if xy:
        if (xy>0):
            plt.xlim(-xmax,xmax)
        else:
            plt.xlim(0.0,xmax)
        plt.ylim(-ymax,ymax)
    return res

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
    if xy==1:
        xcoord = np.log10(r* np.sin(h))
        ind=np.zeros_like(h)
        for i in range(0,nx):
            for j in range (0,ny):
                if(np.cos(h[i,j])>0.0):
                    ind[i,j]=1.0
                else:
                    ind[i,j]=-1.0
        ycoord = np.log10(r* np.abs(np.cos(h)))
        for i in range(0,nx):
            for j in range (0,ny):
                if(ycoord[i,j]<0.0):
                    ycoord[i,j]=0.0
        ycoord*=ind
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,0],myvar[:,:,:,0],nc,**kwargs)
    elif xy==2:
        xcoord = r* np.sin(ph)
        ycoord = r* np.cos(ph)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
    elif xy==3:
        xcoord = r* np.sin(h)
        ycoord = r* np.cos(h)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
    else:
        if isfilled:
            for i in range(0,nb):
                l[i] = ax.contourf(xcoord[i,:,:,z],ycoord[i,:,:,z],myvar[i,:,:,z],nc,**kwargs)
        else:
            for i in range(0,nb):
                l[i] = ax.contour(xcoord[i,:,:,z],ycoord[i,:,:,z],myvar[i,:,:,z],nc,**kwargs)
    if( cb == True): #use color bar
        for i in range(0,nb2d):
            plt.colorbar(l[i],ax=ax)
    if xy:
        if (xy>0):
            plt.xlim(-xmax,xmax)
        else:
            plt.xlim(0.0,xmax)
        plt.ylim(-ymax,ymax)
    return l

def plc_new2(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    global r,h,ph
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k',0)
    i2= kwargs.pop('i2',0)
    mirrory = kwargs.pop('mirrory',0)
    #cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled',False)
    xy = kwargs.pop('xy',0)
    xmax = kwargs.pop('xmax',10)
    ymax = kwargs.pop('ymax',5)
    if ax is None:
        ax = plt.gca()
    if xy==1:
        xcoord = np.log10(r* np.sin(h))
        ind=np.zeros_like(h)
        for i in range(0,nx):
            for j in range (0,ny):
                if(np.cos(h[i,j])>0.0):
                    ind[i,j]=1.0
                else:
                    ind[i,j]=-1.0
        ycoord = np.log10(r* np.abs(np.cos(h)))
        for i in range(0,nx):
            for j in range (0,ny):
                if(ycoord[i,j]<0.0):
                    ycoord[i,j]=0.0
        ycoord*=ind
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,0],myvar[:,:,:,0],nc,**kwargs)
    elif xy==2:
        xcoord = r* np.sin(ph)
        ycoord = r* np.cos(ph)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,ny[0]/2,:],ycoord[:,:,ny[0]/2,:],myvar[:,:,ny[0]/2,:],nc,**kwargs)
    elif xy==3:
        xcoord = r* np.sin(h)
        ycoord = r* np.cos(h)
        if mirrory: ycoord *= -1
        if isfilled:
            res = ax.contourf(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,:,0],ycoord[:,:,:,0],myvar[:,:,:,0],nc,**kwargs)
    else:
        if isfilled:
            res = ax.contourf(xcoord[i2,:,:,0],ycoord[i2,:,:,0],myvar[i2,:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[i2,:,:,0],ycoord[i2,:,:,0],myvar[i2,:,:,0],nc,**kwargs)
    if( cb == True): #use color bar
        plt.colorbar(res,ax=ax)
    if xy:
        if (xy>0):
            plt.xlim(-xmax,xmax)
        else:
            plt.xlim(0.0,xmax)
        plt.ylim(-ymax,ymax)
    return res

def faraday():
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
    fdd = np.zeros((4,4,nb, nx,ny,nz),dtype=rho.dtype)
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
    fuu = np.zeros((4,4,nb,nx,ny,nz),dtype=rho.dtype)
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
    B1hat=B[1]*np.sqrt(gv3[1,1])
    B2hat=B[2]*np.sqrt(gv3[2,2])
    B3nonhat=B[3]
    v1hat=uu[1]*np.sqrt(gv3[1,1])/uu[0]
    v2hat=uu[2]*np.sqrt(gv3[2,2])/uu[0]
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
    if 0:
        rhoc = np.zeros_like(rho)
        if nx>=2:
            rhoc[1:-1] += ((gdet*fuu[0,1])[2:]-(gdet*fuu[0,1])[:-2])/(2*_dx1)
        if ny>2:
            rhoc[:,1:-1] += ((gdet*fuu[0,2])[:,2:]-(gdet*fuu[0,2])[:,:-2])/(2*_dx2)
        if ny>=2 and nz > 1: #not sure if properly works for 2D XXX
            rhoc[:,0,:nz/2] += ((gdet*fuu[0,2])[:,1,:nz/2]+(gdet*fuu[0,2])[:,0,nz/2:])/(2*_dx2)
            rhoc[:,0,nz/2:] += ((gdet*fuu[0,2])[:,1,nz/2:]+(gdet*fuu[0,2])[:,0,:nz/2])/(2*_dx2)
        if nz>2:
            rhoc[:,:,1:-1] += ((gdet*fuu[0,3])[:,:,2:]-(gdet*fuu[0,3])[:,:,:-2])/(2*_dx3)
        if nz>=2:
            rhoc[:,:,0] += ((gdet*fuu[0,3])[:,:,1]-(gdet*fuu[0,3])[:,:,-1])/(2*_dx3)
            rhoc[:,:,-1] += ((gdet*fuu[0,3])[:,:,0]-(gdet*fuu[0,3])[:,:,-2])/(2*_dx3)
        rhoc /= gdet


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
            rhoc[1:-1] += ((gdet*fuu[0,1])[2:]-(gdet*fuu[0,1])[:-2])/(2*_dx1)
        if ny>2:
            rhoc[:,1:-1] += ((gdet*fuu[0,2])[:,2:]-(gdet*fuu[0,2])[:,:-2])/(2*_dx2)
        if ny>=2 and nz > 1: #not sure if properly works for 2D XXX
            rhoc[:,0,:nz/2] += ((gdet*fuu[0,2])[:,1,:nz/2]+(gdet*fuu[0,2])[:,0,nz/2:])/(2*_dx2)
            rhoc[:,0,nz/2:] += ((gdet*fuu[0,2])[:,1,nz/2:]+(gdet*fuu[0,2])[:,0,:nz/2])/(2*_dx2)
        if nz>2:
            rhoc[:,:,1:-1] += ((gdet*fuu[0,3])[:,:,2:]-(gdet*fuu[0,3])[:,:,:-2])/(2*_dx3)
        if nz>=2:
            rhoc[:,:,0] += ((gdet*fuu[0,3])[:,:,1]-(gdet*fuu[0,3])[:,:,-1])/(2*_dx3)
            rhoc[:,:,-1] += ((gdet*fuu[0,3])[:,:,0]-(gdet*fuu[0,3])[:,:,-2])/(2*_dx3)
        rhoc /= gdet
    '''

        
def writehdf5(fname = "jet.hdf5",data_dic=None):
    import h5py
    f = h5py.File(fname, "w")
    for name in data_dic.keys():
        f.create_dataset(name,data=data_dic[name])
    f.close()

def fieldcalctoth():
    """
    Computes the field vector potential
    """
    daphi = -(gdetB[1]).sum(-1)*_dx2/nz
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    return(aphi)

def Tcalcud():
    global Tud, TudEM, TudMA
    global mu, sigma
    global enth
    global unb, isunbound
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    if 'Tud' in globals():
        del Tud
    if 'TudMA' in globals():
        del TudMA
    if 'TudEM' in globals():
        del TudEM
    if 'mu' in globals():
        del mu
    if 'sigma' in globals():
        del sigma
    if 'unb' in globals():
        del unb
    if 'isunbound' in globals():
        del isunbound
    Tud = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
    TudMA = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
    TudEM = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
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
    global Tuu, TuuEM, TuuMA

    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    Tuu = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    TuuMA = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    TuuEM = np.zeros((4,4,nb,bs1,bs2,bs3),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            TuuEM[kapa,nu] = bsq*uu[kapa]*uu[nu] + 0.5*bsq*gcon[kapa][nu] - bu[kapa]*bu[nu]
            TuuMA[kapa,nu] = w*uu[kapa]*uu[nu]+pg*gcon[kapa][nu]
            #Tud[kapa,nu] = eta*uu[kapa]*ud[nu]+(pg+0.5*bsq)*delta-bu[kapa]*bd[nu]
            Tuu[kapa,nu] = TuuEM[kapa,nu] + TuuMA[kapa,nu]

    
def Tcalcuu():
    global Tuu, TuuEM, TuuMA
    global mu, sigma
    global enth
    global unb, isunbound
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    if 'Tuu' in globals():
        del Tuu
    if 'TuuMA' in globals():
        del TuuMA
    if 'TuuEM' in globals():
        del TuuEM
    if 'mu' in globals():
        del mu
    if 'sigma' in globals():
        del sigma
    if 'unb' in globals():
        del unb
    if 'isunbound' in globals():
        del isunbound
    Tuu = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
    TuuMA = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
    TuuEM = np.zeros((4,4,nb,nx,ny,nz),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            TuuEM[kapa,nu] = bsq*uu[kapa]*uu[nu] + 0.5*bsq*guu[kapa][nu] - bu[kapa]*bu[nu]
            TuuMA[kapa,nu] = w*uu[kapa]*uu[nu]+pg*guu[kapa][nu]
            #Tud[kapa,nu] = eta*uu[kapa]*ud[nu]+(pg+0.5*bsq)*delta-bu[kapa]*bd[nu]
            Tuu[kapa,nu] = TuuEM[kapa,nu] + TuuMA[kapa,nu]


def aux():
    faraday()
    Tcalcud()
    
def iofr(rval):
    rval = np.array(rval)
    if np.max(rval) < r[0,0,0]:
        return 0
    res = interp1d(r[:,0,0], ti[:,0,0], kind='linear', bounds_error = False, fill_value = 0)(rval)
    if len(res.shape)>0 and len(res)>0:
        res[rval<r[0,0,0]]*=0
        res[rval>r[nx-1,0,0]]=res[rval>r[nx-1,0,0]]*0+nx-1
    else:
        res = np.float64(res)
    return(np.floor(res+0.5).astype(int))

def prime2cart(V):
    global dxdxp
    Vr = dxdxp[1,1]*V[1]+dxdxp[1,2]*V[2]
    Vh = dxdxp[2,1]*V[1]+dxdxp[2,2]*V[2]
    Vp = V[3]*dxdxp[3,3]
    #
    Vrnorm=Vr
    Vhnorm=Vh*np.abs(r)
    Vpnorm=Vp*np.abs(r*np.sin(h))
    #
    Vznorm=Vrnorm*np.cos(h)-Vhnorm*np.sin(h)
    VRnorm=Vrnorm*np.sin(h)+Vhnorm*np.cos(h)
    Vxnorm=VRnorm*np.cos(ph)-Vpnorm*np.sin(ph)
    Vynorm=VRnorm*np.sin(ph)+Vpnorm*np.cos(ph)
    return(np.array([V[0],Vxnorm,Vynorm,Vznorm]))

def prime2cyl(V):
    global dxdxp
    Vr = dxdxp[1,1]*V[1]+dxdxp[1,2]*V[2]
    Vh = dxdxp[2,1]*V[1]+dxdxp[2,2]*V[2]
    Vp = V[3]*dxdxp[3,3]
    #
    Vrnorm=Vr
    Vhnorm=Vh*np.abs(r)
    Vpnorm=Vp*np.abs(r*np.sin(h))
    #
    Vznorm=Vrnorm*np.cos(h)-Vhnorm*np.sin(h)
    VRnorm=Vrnorm*np.sin(h)+Vhnorm*np.cos(h)
    return(np.array([V[0],VRnorm,Vznorm,Vpnorm]))

def getxyz(r,h,ph):
    x = r*np.sin(h)*cos(ph)
    y = r*np.sin(h)*sin(ph)
    z = r*np.cos(h)
    return x,y,z

def convert():
    global rho, ug,uu,ud,B,bu,bd,bsq,ti,tj,tk,x1,x2,x3,r,h,ph,alpha,gcov,gcon,gdet,dxdxp,nb,nb1,nb2,nb3,bs1,bs2,bs3
    rho=rho.reshape((1,nb*bs1*bs2*bs3), order='F')
    ug=ug.reshape((1,nb*bs1*bs2*bs3), order='F')
    uu=uu.reshape((4,1,nb*bs1*bs2*bs3), order='F')
    ud=ud.reshape((4,1,nb*bs1*bs2*bs3), order='F')
    B=B.reshape((4,1,nb*bs1*bs2*bs3), order='F')
    bu=bu.reshape((4,1,nb*bs1*bs2*bs3), order='F')
    bd=bd.reshape((4,1,nb*bs1*bs2*bs3), order='F')
    bsq=bsq.reshape((1,nb*bs1*bs2*bs3), order='F')
    ti=ti.reshape((1,nb*bs1*bs2*bs3), order='F')
    tj=tj.reshape((1,nb*bs1*bs2*bs3), order='F')
    tk=tk.reshape((1,nb*bs1*bs2*bs3), order='F')
    x1=x1.reshape((1,nb*bs1*bs2*bs3), order='F')
    x2=x2.reshape((1,nb*bs1*bs2*bs3), order='F')
    x3=x3.reshape((1,nb*bs1*bs2*bs3), order='F')
    r=r.reshape((1,nb*bs1*bs2*bs3), order='F')
    h=h.reshape((1,nb*bs1*bs2*bs3), order='F')
    ph=ph.reshape((1,nb*bs1*bs2*bs3), order='F')
    alpha=alpha.reshape((1,nb*bs1*bs2*bs3), order='F')
    gcov=gcov.reshape((4,4,1,nb*bs1*bs2*bs3), order='F')
    gcon=gcon.reshape((4,4,1,nb*bs1*bs2*bs3), order='F')
    gdet=gdet.reshape((1,nb*bs1*bs2*bs3), order='F')
    dxdxp=dxdxp.reshape((4,4,1,nb*bs1*bs2*bs3), order='F')

    sort_array=np.lexsort((tj[0], ti[0]))
    rho[0]=rho[0,sort_array]
    ug[0]=ug[0,sort_array]
    uu[:,0]=uu[:,0,sort_array]
    ud[:,0]=ud[:,0,sort_array]
    B[:,0]=B[:,0,sort_array]
    bu[:,0]=bu[:,0,sort_array]
    bd[:,0]=bd[:,0,sort_array]
    bsq[0]=bsq[0,sort_array]
    ti[0]=ti[0,sort_array]
    tj[0]=tj[0,sort_array]
    tk[0]=tj[0,sort_array]
    x1[0]=x1[0,sort_array]
    x2[0]=x2[0,sort_array]
    x3[0]=x2[0,sort_array]
    r[0]=r[0,sort_array]
    h[0]=h[0,sort_array]
    ph[0]=ph[0,sort_array]
    alpha[0]=alpha[0,sort_array]
    gcov[:,:,0]=gcov[:,:,0,sort_array]
    gcon[:,:,0]=gcon[:,:,0,sort_array]
    gdet[0]=gdet[0,sort_array]
    dxdxp[:,:,0]=dxdxp[:,:,0,sort_array]

    rho=rho.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    ug=ug.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    uu=uu.reshape((4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    ud=ud.reshape((4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    B=B.reshape((4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    bu=bu.reshape((4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    bd=bd.reshape((4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    bsq=bsq.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    ti=ti.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    tj=tj.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    tk=tk.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    x1=x1.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    x2=x2.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    x3=x3.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    r=r.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    h=h.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    ph=ph.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    alpha=alpha.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    gcov=gcov.reshape((4,4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    gcon=gcon.reshape((4,4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    gdet=gdet.reshape((1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')
    dxdxp=dxdxp.reshape((4,4,1,bs3*nb3,bs2*nb2,bs1*nb1), order='F')


    rho=myfloat(rho.transpose(0,3,2,1))
    ug=myfloat(ug.transpose(0,3,2,1))
    uu=myfloat(uu.transpose(0,1,4,3,2))
    ud=myfloat(ud.transpose(0,1,4,3,2))
    B=myfloat(B.transpose(0,1,4,3,2))
    bu=myfloat(bu.transpose(0,1,4,3,2))
    bd=myfloat(bd.transpose(0,1,4,3,2))
    bsq=myfloat(bsq.transpose(0,3,2,1))
    ti=myfloat(ti.transpose(0,3,2,1))
    tj=myfloat(tj.transpose(0,3,2,1))
    tk=myfloat(tk.transpose(0,3,2,1))
    x1=myfloat(x1.transpose(0,3,2,1))
    x2=myfloat(x2.transpose(0,3,2,1))
    x3=myfloat(x3.transpose(0,3,2,1))
    r=myfloat(r.transpose(0,3,2,1))
    h=myfloat(h.transpose(0,3,2,1))
    ph=myfloat(ph.transpose(0,3,2,1))
    alpha=myfloat(alpha.transpose(0,3,2,1))
    gcov=myfloat(gcov.transpose(0,1,2,5,4,3))
    gcon=myfloat(gcon.transpose(0,1,2,5,4,3))
    gdet=myfloat(gdet.transpose(0,3,2,1))
    dxdxp=myfloat(dxdxp.transpose(0,1,2,5,4,3))
    
    bs1=nb1*bs1
    bs2=nb2*bs2
    bs3=nb3*bs3
    nb=1
    nb1=1
    nb2=1
    nb3=1
    faraday_new()
    Tcalcud_new()
    Tcalcuu_new()

if __name__ == "__main__":
    print "Running in batch mode..."
    rblock_new()
    rpar_new(0)

    rgdump_new()
    rdump_new(0)
    #rdiag_new(0)

    faraday_new()
    Tcalcud_new()
