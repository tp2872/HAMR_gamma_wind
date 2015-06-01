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

*********************************************************************************/
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS 
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include "cuda.h"
#include <mpi.h>
#include <omp.h>
/*************************************************************************
      COMPILE-TIME PARAMETERS : 
*************************************************************************/

/*Set execution mode. Note that GPU needs double precision support. You may require to modify LOCAL_WORK_SIZE*/
#define GPU_ENABLED 1
#define GPU_FAST 0
#define GPU_BENCHMARK 0
#define GPU_DEBUG 0
#define CPU_OPENMP 0
#define Intel_CL 0
#define TIMER 1

/*Set number of rows and columns for MPI processes*/
#define MPI_rows (2)
#define MPI_columns (2) 
#define MPI_stacks (1) 

/*Pin or don't pin memory for GPU transfers*/
#define PINNED 1

/*Wheter to set floors in lab frame*/
#define ZAMO_FLOOR (1)

/*Whether or not to allow inflow for fluxes (see fix_flux())*/
#define INFLOW 0

/*Wheter or not to use the full dispersion relation*/
#define FULL_DISP (0)

/*Set grid constants*/
#define BRAVO (1.0)
#define QUEBEC (2.0)
#define CHARLIE (0.55)
#define DELTA (3.0)

/* whether or not to use Font's  adiabatic/isothermal prim. var. inversion method: */
#define DO_FONT_FIX (1)

/* whether or not to rescale primitive variables before interpolating them for flux/BC's */
#define RESCALE     (0)

//which problem
#define MONOPOLE_PROBLEM_1D 1
#define MONOPOLE_PROBLEM_2D 2
#define BZ_MONOPOLE_2D 3
#define TORUS_PROBLEM 4
#define BONDI_PROBLEM_1D 5
#define BONDI_PROBLEM_2D 6

#define WHICHPROBLEM TORUS_PROBLEM

/** here are the few things that we change frequently **/
#if WHICHPROBLEM == MONOPOLE_PROBLEM_1D
#define N1       (2*386)      /* number of physical zones in X1-direction */
#define N2       (1)          /* number of physical zones in X2-direction */
#elif WHICHPROBLEM == MONOPOLE_PROBLEM_2D
#define N1       (2*386)      /* number of physical zones in X1-direction */
#define N2       (256)        /* number of physical zones in X2-direction */
#elif WHICHPROBLEM == BZ_MONOPOLE_2D
#define N1       (128)      /* number of physical zones in X1-direction */
#define N2       (128)        /* number of physical zones in X2-direction */
#elif WHICHPROBLEM == TORUS_PROBLEM
#define temp_N1  (100)        /* number of physical zones in X1-direction */
#define temp_N2  (100)        /* number of physical zones in X2-direction */
#define temp_N3  (100)        /* number of physical zones in X2-direction */
#elif WHICHPROBLEM == BONDI_PROBLEM_1D
#define N1       (256)        /* number of physical zones in X1-direction */
#define N2       (1)          /* number of physical zones in X2-direction */
#elif WHICHPROBLEM == BONDI_PROBLEM_2D
#define N1       (256)        /* number of physical zones in X1-direction */
#define N2       (256)        /* number of physical zones in X2-direction */
#endif

/*Set freezing point*/
#define ibound 0
#define jbound 0
#define zbound 0

#define boundfreeze1 0 //Movie playback(not yet fully implemented)
#define boundfreeze2 0 //Frozen boundary at i=ibound-1
#if boundfreeze2==1
#define N_POINTS 2
#elif boundfreeze1==1
#define N_POINTS 10
#else
#define N_POINTS 1
#endif

#define N1 (temp_N1)
#define N2 (temp_N2)
#define N3 (temp_N3)

//only allocate memory for ghost cells for non-trivial dimensions
#define NG (2)
#define N1M ((N1>1)?(N1+2*NG):(1))
#define N2M ((N2>1)?(N2+2*NG):(1))
#define N3M ((N3>1)?(N3+2*NG):(1))

#define N1G ((N1>1)?(NG):(0))
#define N2G ((N2>1)?(NG):(0))
#define N3G ((N3>1)?(NG):(0))

#define D1 (N1>1)
#define D2 (N2>1)
#define D3 (N3>1)

#define NPR        (8)        /* number of primitive variables */
#define NDIM       (4)        /* number of total dimensions.  Never changes */
#define NPG        (5)        /* number of positions on grid for grid functions */
#define COMPDIM    (2)        /* number of non-trivial spatial dimensions used in computation */
#define NIMG       (4)        /* Number of types of images to make, kind of */
#define NFAIL	   (5)        /* Number of types of failure images to make*/

/* how many cells near the poles to stabilize, choose 0 for no stabilization */
#define POLEFIX 1*(1-D3)

/** FIXUP PARAMETERS, magnitudes of rho and u, respectively, in the floor : **/
#define RHOMIN	(0.5*1.e-4)
#define UUMIN	(0.5*1.e-6)
#define RHOMINLIMIT (1.e-20)
#define UUMINLIMIT  (1.e-20)
#define POWRHO (2.5)

#define FLOORFACTOR (1.)
#define BSQORHOMAX (9.*FLOORFACTOR)
#define BSQOUMAX (250.*FLOORFACTOR)
#define UORHOMAX (50.*FLOORFACTOR)

/* A numerical convenience to represent a small non-zero quantity compared to unity:*/
#define SMALL	(1.e-20)

/* Max. value of gamma, the lorentz factor */
#define GAMMAMAX (50.)

/* maximum fractional increase in timestep per timestep */
#define SAFE	(1.3)

#define COORDSINGFIX 1
// whether to move polar axis to a bit larger theta
// theta value where singularity is displaced to
#define SINGSMALL (1.E-20)

/* I/O format strings used herein : */
#define FMT_DBL_OUT "%28.18e"
#define FMT_INT_OUT "%10d"

/*Define local work size for GPU. Needed to optimize GPU performance*/
#if(GPU_ENABLED == 1 || GPU_DEBUG == 1 || GPU_BENCHMARK==1) 
#define LOCAL_WORK_SIZE 64
#else
#define LOCAL_WORK_SIZE 1
#endif

/*************************************************************************
    MNEMONICS SECTION 
*************************************************************************/
/* boundary condition mnemonics */
#define OUTFLOW	(0)
#define SYMM	(1)
#define ASYMM	(2)
#define FIXED	(3)

/* mnemonics for primitive vars; conserved vars */
#define RHO	(0)	
#define UU	(1)
#define U1	(2)
#define U2	(3)
#define U3	(4)
#define B1	(5)
#define B2	(6)
#define B3	(7)

/* mnemonics for dimensional indices */
#define TT	(0)     
#define RR	(1)
#define TH	(2)
#define PH	(3)

/* mnemonics for centering of grid functions */
#define FACE1	(0)	
#define FACE2	(1)
#define CORN	(2)
#define CENT	(3)
#define FACE3	(4)

/* mnemonics for slope limiter */
#define MC	(0)
#define VANL	(1)
#define MINM	(2)

/* mnemonics for diagnostic calls */
#define INIT_OUT	(0)
#define DUMP_OUT	(1)
#define IMAGE_OUT	(2)
#define LOG_OUT		(3)
#define FINAL_OUT	(4)

/* Directional Mnemonics */
// -------------> r
// |         3    
// |        1-0   
// |         2    
// v            
// theta      
#define X1UP    (0)
#define X1DN    (1)
#define X2UP    (2)
#define X2DN    (3)


/* failure modes */
#define FAIL_UTOPRIM        (1)
#define FAIL_VCHAR_DISCR    (2)
#define FAIL_COEFF_NEG	    (3)
#define FAIL_COEFF_SUP	    (4)
#define FAIL_GAMMA          (5)
#define FAIL_METRIC         (6)

/* For rescale() operations: */
#define FORWARD 1
#define REVERSE 2

/*For Windows users*/
#ifndef M_PI 
#define M_PI 3.14159265358979323846264338327950288 
#endif 

/*************************************************************************
    GLOBAL ARRAY SECTION 
*************************************************************************/
/* for debug */
extern double (*  psave)[NPR] ;
extern double(*  fsave)[NFAIL];
extern int(* failimage)[NFAIL];
extern double (**  fimage);

/* grid functions */
extern double tbound[N_POINTS];
extern double (* pbound)[NPR][N_POINTS];
extern double(*   p)[NPR];
extern double(*   ph)[NPR];
extern double(*  dq)[NPR];
extern double(*  F1)[NPR];
extern double(*  F2)[NPR];
extern double(*  F3)[NPR];
extern int(*  pflag);
extern double (* conn)[NDIM][NDIM][NDIM] ;
extern double (* gcon)[NPG][NDIM][NDIM] ;
extern double (* gcov)[NPG][NDIM][NDIM] ;
extern double (* gdet)[NPG] ;

/*GPU variables*/
#define FTYPE2 double
extern int nr_workgroups;
extern FTYPE2 *F1_1;
extern FTYPE2 *F2_1;
extern FTYPE2 *F3_1;
extern FTYPE2 *dq_1;
extern FTYPE2 *p_1;
extern FTYPE2 *ph_1;
extern FTYPE2 *pbound_1;
extern FTYPE2 *gcov_GPU;
extern FTYPE2 *gcon_GPU;
extern FTYPE2 *conn_GPU;
extern FTYPE2 *gdet_GPU;
extern FTYPE2 *dtij_GPU;
extern FTYPE2 *Katm_GPU;
extern int *pflag_GPU;
extern int *failimage_GPU;

/*MPI variables*/
extern int nthreads;
extern int n_rows;
extern int n_columns;
extern int n_stacks;
extern int n1_MPI;
extern int n2_MPI;
extern int n3_MPI;
extern int N1_MPI;
extern int N2_MPI;
extern int N3_MPI;
extern int N1_MPI_offset;
extern int N2_MPI_offset;
extern int N3_MPI_offset;
extern int N1_tot;
extern int N2_tot;
extern int N3_tot;
extern int numtasks, rank, rc;
extern int max1D_MPI;
extern MPI_Request reqs[1000];
extern MPI_Status Stat[1];
extern double *send, *receive;
extern FTYPE2  *send1, *send2, *send3, *send4, *send5, *send6, *send7, *send8;
extern FTYPE2 *receive1, *receive2, *receive3, *receive4, *receive5, *receive6, *receive7, *receive8;
extern FTYPE2  *cornsend1, *cornsend2, *cornsend3, *cornsend4, *cornsend5, *cornsend7, *cornsend7, *cornsend8;
extern FTYPE2 *cornreceive1, *cornreceive2, *cornreceive3, *cornreceive4, *cornreceive5, *cornreceive6, *cornreceive7, *cornreceive8;
extern int *aN1_MPI_offset;
extern int *aN2_MPI_offset;
extern int *aN3_MPI_offset;
extern int *aN1_MPI;
extern int *aN2_MPI;
extern int *aN3_MPI;

#if(DO_FONT_FIX)
extern double Katm[N1];
#endif

/*************************************************************************
    GLOBAL VARIABLES SECTION 
*************************************************************************/
/* physics parameters */
extern double a ;
extern double gam ;

/* numerical parameters */
extern double Rin,Rout,hslope,R0,fractheta ;
extern double cour ;
extern double dV,dx[NPR],startx[NPR] ;
extern double dt ;
extern double t,tf ;
extern double x1curr,x2curr ;
extern int nstep ;

/* output parameters */
extern double DTd ;
extern double DTl ;
extern double DTi ;
extern int    DTr ;
extern int    dump_cnt ;
extern int    image_cnt ;
extern int    rdump_cnt ;
extern int    nstroke ;

/* global flags */
extern int failed ;
extern int lim ;
extern double defcon ;

/* diagnostics */
extern double mdot ;
extern double edot ;
extern double ldot ;

/* set global variables that indicate current local metric, etc. */
extern int icurr,jcurr,pcurr ;

struct of_geom {
	double gcon[NDIM][NDIM] ;
	double gcov[NDIM][NDIM] ;
	double g ;
} ;

struct of_state {
	double ucon[NDIM] ;
	double ucov[NDIM] ;
	double bcon[NDIM] ;
	double bcov[NDIM] ;
} ;

/*OpenCL variables decleration*/
extern cl_uint numPlatforms;
extern cl_platform_id platform;
extern int status;
extern cl_uint numDevices;
extern cl_device_id *devices;
extern cl_context context;
extern FILE* programHandle;
extern size_t programSize;
extern char *programBuffer;
extern cl_program GPU_program;
extern cl_kernel kernel_flux_ct1;
extern cl_kernel kernel_flux_ct2;
extern cl_kernel kernel_fluxcalcprep;
extern cl_kernel kernel_fluxcalc2D1;
extern cl_kernel kernel_fluxcalc2D2;
extern cl_kernel kernel_fix_flux;
extern cl_kernel kernel_diag_flux;
extern cl_kernel kernel_Utoprim;
extern cl_kernel kernel_Utoprim1;
extern cl_kernel kernel_Utoprim2;
extern cl_kernel kernel_Utoprimbound;
extern cl_kernel kernel_fixup;
extern cl_kernel kernel_fixuputoprim;
extern cl_kernel kernel_boundprim1;
extern cl_kernel kernel_boundprim2;
extern cl_kernel kernel_boundprim3;
extern cl_kernel kernel_boundsend;
extern cl_kernel kernel_boundrec;
extern cl_event event_utoprimbound[1];
extern cl_command_queue commandQueueGPU;
extern int fix_mem;
extern int fix_mem2;
extern size_t global_work_size[1];
extern size_t global_work_size_bound[1];
extern size_t global_work_offset[1];
extern size_t global_work_size_special[1];
extern size_t global_work_size_special1[1];
extern size_t global_work_size_special2[1];
extern size_t global_work_size_special3[1];
extern size_t local_work_size[1];
extern double* BufferF1_1;
extern double* BufferF2_1;
extern double* BufferF3_1;
extern double* Bufferdq_1;
extern double* Bufferstorage1;
extern double* Bufferstorage2;
extern double* Bufferstorage3;
extern double* Bufferstorage4;
extern double* Bufferp_1;
extern double* Bufferph_1;
extern double* Bufferpbound_1;
extern double* Buffergcov;
extern double* Buffergcon;
extern double* Bufferconn;
extern double* Buffergdet;
extern double* Bufferdtij;
extern double* Bufferdiagflux;
extern int* Bufferpflag;
extern int* Bufferfailimage;
extern double* BufferKatm;
extern int* BufferaN1_MPI;
extern int* BufferaN1_MPI_offset;
extern int* BufferaN2_MPI;
extern int* BufferaN2_MPI_offset;
extern int* BufferaN3_MPI;
extern int* BufferaN3_MPI_offset;
extern int* Buffern1_MPI;
extern int* Buffern2_MPI;
extern int* Buffern3_MPI;
extern int* Bufferrank;
extern double* Bufferboundrec1_MPI;
extern double* Bufferboundrec2_MPI;
extern double* Bufferboundrec3_MPI;
extern double* Bufferboundrec4_MPI;
extern double* Bufferboundrec5_MPI;
extern double* Bufferboundrec6_MPI;
extern double* Bufferboundrec7_MPI;
extern double* Bufferboundrec8_MPI;
extern double* Bufferboundsend1_MPI;
extern double* Bufferboundsend2_MPI;
extern double* Bufferboundsend3_MPI;
extern double* Bufferboundsend4_MPI;
extern double* Bufferboundsend5_MPI;
extern double* Bufferboundsend6_MPI;
extern double* Bufferboundsend7_MPI;
extern double* Bufferboundsend8_MPI;
extern double* Buffercornrec1_MPI;
extern double* Buffercornrec2_MPI;
extern double* Buffercornrec3_MPI;
extern double* Buffercornrec4_MPI;
extern double* Buffercornrec5_MPI;
extern double* Buffercornrec6_MPI;
extern double* Buffercornrec7_MPI;
extern double* Buffercornrec8_MPI;
extern double* Buffercornsend1_MPI;
extern double* Buffercornsend2_MPI;
extern double* Buffercornsend3_MPI;
extern double* Buffercornsend4_MPI;
extern double* Buffercornsend5_MPI;
extern double* Buffercornsend6_MPI;
extern double* Buffercornsend7_MPI;
extern double* Buffercornsend8_MPI;

/*Timing/benchmarking decleration*/
extern clock_t begin1, end1, begin2, end2;
extern double time_spent3;

/*************************************************************************
    MACROS
*************************************************************************/
/* loop over all active zones */
#define ZLOOP for(i=0;i<N1;i++)for(j=0;j<N2;j++)
#define ZLOOP_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)
#if (N3>1)
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)for(z=N3_MPI_offset;z<N3_MPI_offset + N3_MPI ;z++)
#else
#define ZLOOP3D for(i=0;i<N1;i++)for(j=0;j<N2;j++)for(z=0;z<N3;z++)
#define ZLOOP3D_MPI for(i=N1_MPI_offset;i<N1_MPI_offset + N1_MPI;i++)for(j=N2_MPI_offset;j<N2_MPI_offset + N2_MPI ;j++)for(z=N3_MPI_offset;z<N3_MPI_offset + N3_MPI ;z++)
#endif

/* loop over all active zones */
#define IMAGELOOP for(j=0;j<N2;j++)for(i=0;i<N1;i++)

/* specialty loop */
extern int istart,istop,jstart,jstop, zstart, zstop ;
#define ZSLOOP(istart,istop,jstart,jstop) for(i=istart;i<=istop;i++) for(j=jstart;j<=jstop;j++)
#if (N3>1)
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#else
#define ZSLOOP3D(istart, istop, jstart, jstop, zstart, zstop) for (i = istart; i <= istop; i++) for (j = jstart; j <= jstop; j++) for(z=zstart;z<=zstop;z++)
#endif

/* loop over Primitive variables */
#define PLOOP  for(k=0;k<NPR;k++)
/* loop over all Dimensions; second rank loop */
#define DLOOP  for(j=0;j<NDIM;j++) for(k=0;k<NDIM;k++)
/* loop over all Dimensions; first rank loop */
#define DLOOPA for(j=0;j<NDIM;j++)
/* loop over all Space dimensions; second rank loop */
#define SLOOP  for(j=1;j<NDIM;j++) for(k=1;k<NDIM;k++)
/* loop over all Space dimensions; first rank loop */
#define SLOOPA for(j=1;j<NDIM;j++)

extern double fval1,fval2;
#define MY_MIN(fval1,fval2) ( ((fval1) < (fval2)) ? (fval1) : (fval2))
#define MY_MAX(fval1,fval2) ( ((fval1) > (fval2)) ? (fval1) : (fval2))
extern FTYPE2 diagflux[3*MY_MAX(N1 + 4, N2 + 4)];

#define delta(i,j) ( (i == j) ? 1. : 0.)
#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) 

/*************************************************************************
    FUNCTION DECLARATIONS 
*************************************************************************/
double bl_gdet_func(double r, double th) ;
double bsq_calc(double *pr, struct of_geom *geom) ;
int    gamma_calc(double *pr, struct of_geom *geom, double *gamma) ;
double gdet_func(double lgcov[][NDIM]) ;
double mink(int j, int k) ; 
double ranc(int seed) ;
double slope_lim(double y1, double y2, double y3) ;
int restart_init(void) ;
void area_map(int i, int j, double (*prim)[NPR]);
void bcon_calc(double *pr, double *ucon, double *ucov, double *bcon) ;
void blgset(int i, int j, struct of_geom *geom);
void bl_coord(double *X, double *r, double *th, double *phi);
void bl_gcon_func(double r, double th, double gcov[][NDIM]) ;
void bl_gcov_func(double r, double th, double gcov[][NDIM]) ;
void bound_prim(double (*pr)[NPR], int MPI) ;
void bound_send(double(*prim)[NPR]);
void bound_rec(double(*prim)[NPR]);
void conn_func(double *X, struct of_geom *geom, double lconn[][NDIM][NDIM]);
void coord(int i, int j, int z, int loc, double *X) ;
void diag(int call_code) ;
void diag_flux(double (*F1)[NPR]);
void dump(FILE *fp) ;
void gdump(FILE *fp) ;
void fail(int fail_type) ;
void fixup(double((* pv)[NPR]));
void fixup1zone(int i, int j, int z, double prim[NPR]);
void fixup_utoprim( double (*pv)[NPR] )  ;
void set_Katm(void);
int  get_G_ATM( double *g_tmp );
void fix_flux(double(*F1)[NPR], double(*F2)[NPR], double(*F3)[NPR]);
void flux_ct(double(*F1)[NPR], double(*F2)[NPR], double(*F3)[NPR]);
void gaussj(double **tmp, int n, double **b, int m) ;
void gcon_func(double lgcov[][NDIM], double lgcon[][NDIM]) ;
void gcov_func(double *X, double lgcov[][NDIM]) ;
void get_geometry(int i, int j, int loc, struct of_geom *geom) ;
void get_geometry_direct(int ii, int jj, int ff, struct of_geom *geom);
void get_state(double *pr, struct of_geom *geom, struct of_state *q) ;
void image_all( int image_count ) ;
int index(int i, int j, int z);
int index2(int i, int j);
int index3(int i, int j);
void init(void) ;
void inflow_check(double *pr, int ii, int jj, int type);
void lower(double *a, struct of_geom *geom, double *b) ;
void ludcmp(double **a, int n, int *indx, double *d) ;
void mhd_calc(double *pr, int dir, struct of_state *q, double *mhd)  ;
void misc_source(double *ph, int ii, int jj, struct of_geom *geom, struct of_state *q, double *dU, double Dt) ;
void MPI_initialize(int argc, char *argv[]);
void primtoflux(double *pa, struct of_state *q, int dir, struct of_geom *geom,double *fl) ;
void primtoU(double *p, struct of_state *q, struct of_geom *geom, double *U);
void raise(double *v1, struct of_geom *geom, double *v2) ;
void rescale(double *pr, int which, int dir, int ii, int jj, int zz, int face,  struct of_geom *geom) ;
void restart_write(void) ;
void restart_read(FILE *fp) ;
void set_arrays(void) ;
void set_grid(void) ;
void set_points(void) ;
void step_ch(void) ;
void source(double *pa, struct of_geom *geom, int ii, int jj, double *Ua,double Dt) ;
void timestep(void) ;
void u_to_v(double *pr, int i, int j) ;
void ucon_calc(double *pr, struct of_geom *geom, double *ucon) ;
void usrfun(double *pr,int n,double *beta,double **alpha) ;
void Utoprim(double *Ua, struct of_geom *geom, double *pa) ;
int Utoprim_2d(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR]);
int Utoprim_1dvsq2fix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K );
int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K );
void vchar(double *pr, struct of_state *q, struct of_geom *geom,int dir, double *cmax, double *cmin) ;
int invert_matrix( double A[][NDIM], double Ainv[][NDIM] );
int LU_decompose( double A[][NDIM], int permute[] );
void LU_substitution( double A[][NDIM], double B[], int permute[] );
void GPU_benchmark(void);
void GPU_init(void);
void GPU_write(void);
void GPU_finish(void);
double fluxcalc_GPU(int dir, int flag);
void GPU_fixup(int flag);
void GPU_fixuputoprim(int flag);
void GPU_Utoprim(int flag, double Dt);
void GPU_fluxcalc2D(int dir, int flag);
void GPU_fluxcalcprep(int dir, int flag);
void GPU_flux_ct1(void);
void GPU_flux_ct2(void);
void GPU_fix_flux(void);
void GPU_diag_flux(void);
void GPU_boundprim(int flag, int MPI);
void GPU_boundprim1(int flag);
void GPU_boundprim2(int flag);
void GPU_boundprim3(int flag);
void GPU_boundsend(int flag);
void GPU_boundrec(int flag);
void GPU_step_ch();
void step_ch_debug();
void GPU_read(void);
