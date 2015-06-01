/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/
#include <malloc.h>
/*************************************************************************
    GLOBAL ARRAYS SECTION 
*************************************************************************/

/* for debug */
double(* psave)[NPR];
double(*  fsave)[NFAIL];
int(* failimage)[NFAIL];
double (** fimage);

/* grid functions */
double tbound[N_POINTS];
double (*  pbound)[NPR][N_POINTS];
double(*   p)[NPR];
double(*   ph)[NPR];
double(*  dq)[NPR];
double(*  F1)[NPR];
double(*  F2)[NPR];
double(*  F3)[NPR];
int(*  pflag);
double (* conn)[NDIM][NDIM][NDIM] ;
double (* gcon)[NPG][NDIM][NDIM] ;
double (* gcov)[NPG][NDIM][NDIM] ;
double (* gdet)[NPG] ;

/*GPU variables*/
int nr_workgroups;
FTYPE2 *F1_1;
FTYPE2 *F2_1;
FTYPE2 *F3_1;
FTYPE2 *dq_1;
FTYPE2 *p_1;
FTYPE2 *ph_1;
FTYPE2 *pbound_1;
FTYPE2 *gcov_GPU;
FTYPE2 *gcon_GPU;
FTYPE2 *conn_GPU;
FTYPE2 *gdet_GPU;
FTYPE2 *dtij_GPU;
FTYPE2 *Katm_GPU;
cl_int *pflag_GPU;
cl_int *failimage_GPU;
FTYPE2 diagflux[3 * MY_MAX(N1 + 4, N2 + 4)];

/*MPI variables*/
int nthreads;
int n_rows;
int n_columns;
int n_stacks;
int n1_MPI;
int n2_MPI;
int n3_MPI;
int N1_MPI;
int N2_MPI;
int N3_MPI;
int N1_MPI_offset;
int N2_MPI_offset;
int N3_MPI_offset;
int numtasks, rank, rc;
int max1D_MPI;
MPI_Request reqs[1000];
MPI_Status Stat[1];
double *send, *receive;
FTYPE2  *send1, *send2, *send3, *send4,*send5, *send6, *send7, *send8;
FTYPE2 *receive1, *receive2, *receive3, *receive4, *receive5, *receive6, *receive7, *receive8;
FTYPE2  *cornsend1, *cornsend2, *cornsend3, *cornsend4, *cornsend5, *cornsend7, *cornsend7, *cornsend8;
FTYPE2 *cornreceive1, *cornreceive2, *cornreceive3, *cornreceive4, *cornreceive5, *cornreceive6, *cornreceive7, *cornreceive8;
int *aN1_MPI_offset;
int *aN2_MPI_offset;
int *aN3_MPI_offset;
int *aN1_MPI;
int *aN2_MPI;
int *aN3_MPI;

#if(DO_FONT_FIX)
double Katm[N1];
#endif

/*************************************************************************
    GLOBAL VARIABLES SECTION 
*************************************************************************/
/* physics parameters */
double a ;
double gam ;

/* numerical parameters */
double Rin,Rout,hslope,R0,fractheta ;
double cour ;
double dV,dx[NPR],startx[NPR] ;
double dt ;
double t,tf ;
double rcurr,hcurr ;
int istart,istop,jstart,jstop, zstart, zstop ;
int icurr,jcurr,pcurr,ihere,jhere,phere ;
double dminarg1,dminarg2 ;
int nstep ;
double fval1,fval2;

/* output parameters */
double DTd ;
double DTl ;
double DTi ;
int    DTr ;
int    dump_cnt ;
int    image_cnt ;
int    rdump_cnt ;
int    nstroke ;

/* global flags */
int    failed ;
int    lim ;
double defcon ;

/* diagnostics */
double mdot = 0. ;
double edot = 0. ;
double ldot = 0. ;

/*OpenCL variables*/
cl_uint numPlatforms;	
cl_platform_id platform;	
cl_int status;
cl_uint numDevices;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueueGPU;
FILE* programHandle;
size_t programSize;
char *programBuffer;
cl_program GPU_program;
cl_kernel kernel_flux_ct1;
cl_kernel kernel_flux_ct2;
cl_kernel kernel_fluxcalcprep;
cl_kernel kernel_fluxcalc2D1;
cl_kernel kernel_fluxcalc2D2;
cl_kernel kernel_fix_flux;
cl_kernel kernel_diag_flux;
cl_kernel kernel_Utoprim;
cl_kernel kernel_Utoprim1;
cl_kernel kernel_Utoprim2;
cl_kernel kernel_Utoprimbound;
cl_kernel kernel_fixup;
cl_kernel kernel_fixuputoprim;
cl_kernel kernel_boundprim1;
cl_kernel kernel_boundprim2;
cl_kernel kernel_boundprim3;
cl_kernel kernel_boundsend;
cl_kernel kernel_boundrec;
cl_event event_utoprimbound[1];
int fix_mem;
int fix_mem2;
size_t global_work_size[1];
size_t global_work_size_bound[1];
size_t global_work_offset[1];
size_t global_work_size_special[1];
size_t global_work_size_special1[1];
size_t global_work_size_special2[1];
size_t global_work_size_special3[1];
size_t local_work_size[1];
double* BufferF1_1;
double* BufferF2_1;
double* BufferF3_1;
double* Bufferdq_1;
double* Bufferstorage1;
double* Bufferstorage2;
double* Bufferstorage3;
double* Bufferstorage4;
double* Bufferp_1;
double* Bufferph_1;
double* Bufferpbound_1;
double* Buffergcov;
double* Buffergcon;
double* Bufferconn;
double* Buffergdet;
double* Bufferdtij;
double* Bufferdiagflux;
int* Bufferpflag;
int* Bufferfailimage;
double* BufferKatm;
int* BufferaN1_MPI;
int* BufferaN1_MPI_offset;
int* BufferaN2_MPI;
int* BufferaN2_MPI_offset;
int* BufferaN3_MPI;
int* BufferaN3_MPI_offset;
int* Buffern1_MPI;
int* Buffern2_MPI;
int* Buffern3_MPI;
int* Bufferrank;
int N1_tot;
int N2_tot;
int N3_tot;
double* Bufferboundrec1_MPI;
double* Bufferboundrec2_MPI;
double* Bufferboundrec3_MPI;
double* Bufferboundrec4_MPI;
double* Bufferboundrec5_MPI;
double* Bufferboundrec6_MPI;
double* Bufferboundrec7_MPI;
double* Bufferboundrec8_MPI;
double* Bufferboundsend1_MPI;
double* Bufferboundsend2_MPI;
double* Bufferboundsend3_MPI;
double* Bufferboundsend4_MPI;
double* Bufferboundsend5_MPI;
double* Bufferboundsend6_MPI;
double* Bufferboundsend7_MPI;
double* Bufferboundsend8_MPI;
double* Buffercornrec1_MPI;
double* Buffercornrec2_MPI;
double* Buffercornrec3_MPI;
double* Buffercornrec4_MPI;
double* Buffercornrec5_MPI;
double* Buffercornrec6_MPI;
double* Buffercornrec7_MPI;
double* Buffercornrec8_MPI;
double* Buffercornsend1_MPI;
double* Buffercornsend2_MPI;
double* Buffercornsend3_MPI;
double* Buffercornsend4_MPI;
double* Buffercornsend5_MPI;
double* Buffercornsend6_MPI;
double* Buffercornsend7_MPI;
double* Buffercornsend8_MPI;

/*Timers*/
clock_t begin1, end1, begin2, end2;
double time_spent3;