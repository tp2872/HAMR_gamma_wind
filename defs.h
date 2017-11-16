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
int tag_HLLC;
int tag_normal;
/* for debug */
double(*restrict psave)[NPR];
double(*restrict fsave)[NFAIL];
int(*restrict failimage[NB])[NFAIL];
double(** fimage);
double tilt_temp;
int max_levels;
double *connected;

/* grid functions */
double tbound[N_POINTS];
double(*restrict  pbound[NB])[NPR][N_POINTS];
double(*restrict  p[NB])[NPR];
double(*restrict  ph[NB])[NPR];
double(*restrict V[NB])[6];
double E_avg1[NB_1*NB_3][BS_1 + 2 * N1G];
double E_avg2[NB_1*NB_3][BS_1 + 2 * N1G];
double E_avg1_new[NB_1*NB_3][BS_1 + 2 * N1G];
double E_avg2_new[NB_1*NB_3][BS_1 + 2 * N1G];
double(*restrict E_avg_x[NB][2]);
double(*restrict E_avg_new_x[NB][2]);
double(*restrict E_avg_y[NB][2]);
double(*restrict E_avg_new_y[NB][2]);
double(*restrict E_corn[NB])[NDIM];
double(*restrict dE[NB])[2][NDIM][NDIM];
double(*restrict ps[NB])[NDIM];
double(*restrict psh[NB])[NDIM];
double(*restrict dq[NB])[NPR];
double(*restrict F1[NB])[NPR];
double(*restrict F2[NB])[NPR];
double(*restrict F3[NB])[NPR];
double(*restrict stor1[NB])[NPR];
double(*restrict stor2[NB])[NPR];
int(*restrict pflag[NB]);
double(*restrict conn[NB])[NDIM][NDIM][NDIM];
double(*restrict gcon[NB])[NPG][NDIM][NDIM];
double(*restrict gcov[NB])[NPG][NDIM][NDIM];
double(*restrict gdet[NB])[NPG];
double(*restrict dU_s[NB])[NPR];

/*GPU variables*/
//#define FTYPE2 double
FTYPE2 *F1_1[NB];
FTYPE2 *F2_1[NB];
FTYPE2 *F3_1[NB];
FTYPE2 *dq_1[NB];
FTYPE2 *p_1[NB];
FTYPE2 *ph_1[NB];
FTYPE2 *ps_1[NB];
FTYPE2 *psh_1[NB];
FTYPE2 *pbound_1[NB];
FTYPE2 *gcov_GPU[NB];
FTYPE2 *gcon_GPU[NB];
FTYPE2 *conn_GPU[NB];
FTYPE2 *gdet_GPU[NB];
FTYPE2 *dtij_GPU[NB];
FTYPE2 *dU_GPU[NB];
FTYPE2 *Katm_GPU[NB];
int *pflag_GPU[NB];
int *failimage_GPU[NB];

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
int count_node[20000];
int count_gpu[32];
int n_rows_GPU;
int n_columns_GPU;
int n_stacks_GPU;
int n1_MPI_GPU[NB];
int n2_MPI_GPU[NB];
int n3_MPI_GPU[NB];
int N1_GPU[NB];
int N2_GPU[NB];
int N3_GPU[NB];
int N1_GPU_offset[NB];
int N2_GPU_offset[NB];
int N3_GPU_offset[NB];
int numtasks, rank, local_rank, rc;
int max1D_MPI;
#if (MPI_enable)
MPI_Request req[100], boundreqs[NB][600], cornreqs[NB][16];
MPI_Status Statbound[NB][600], Statcorn[NB][16], Statrec[2];
MPI_Comm  mpi_cartcomm, mpi_self;
MPI_Comm row_comm[8];
#endif
int mpi_nbrs[4][2];
int mpi_corns[3][5][2];
double *send[NB], *receive[NB];
double  *send1[NB], *send2[NB], *send3[NB], *send4[NB], *send5[NB], *send6[NB];
double  *send1_fine[NB], *send2_fine[NB], *send3_fine[NB], *send4_fine[NB], *send5_fine[NB], *send6_fine[NB];
double  *send1_flux[NB], *send2_flux[NB], *send3_flux[NB], *send4_flux[NB], *send5_flux[NB], *send6_flux[NB], *send7_flux[NB], *send8_flux[NB];
double  *send1_E[NB], *send2_E[NB], *send3_E[NB], *send4_E[NB], *send5_E[NB], *send6_E[NB], *send7_E[NB], *send8_E[NB];
double *receive1_fine[NB], *receive2_fine[NB], *receive3_fine[NB], *receive4_fine[NB], *receive5_fine[NB], *receive6_fine[NB];
double  *receive1_3fine[NB], *receive1_4fine[NB], *receive1_7fine[NB], *receive1_8fine[NB];
double  *receive2_1fine[NB], *receive2_2fine[NB], *receive2_3fine[NB], *receive2_4fine[NB];
double  *receive3_1fine[NB], *receive3_2fine[NB], *receive3_5fine[NB], *receive3_6fine[NB];
double  *receive4_5fine[NB], *receive4_6fine[NB], *receive4_7fine[NB], *receive4_8fine[NB];
double  *receive5_1fine[NB], *receive5_3fine[NB], *receive5_5fine[NB], *receive5_7fine[NB];
double  *receive6_2fine[NB], *receive6_4fine[NB], *receive6_6fine[NB], *receive6_8fine[NB];

double *receive1_flux[NB], *receive2_flux[NB], *receive3_flux[NB], *receive4_flux[NB], *receive5_flux[NB], *receive6_flux[NB], *receive7_flux[NB], *receive8_flux[NB];
double *receive1_flux1[NB], *receive2_flux1[NB], *receive3_flux1[NB], *receive4_flux1[NB], *receive5_flux1[NB], *receive6_flux1[NB], *receive7_flux1[NB], *receive8_flux1[NB];

double  *receive1_3flux[NB], *receive1_4flux[NB], *receive1_7flux[NB], *receive1_8flux[NB];
double  *receive2_1flux[NB], *receive2_2flux[NB], *receive2_3flux[NB], *receive2_4flux[NB];
double  *receive3_1flux[NB], *receive3_2flux[NB], *receive3_5flux[NB], *receive3_6flux[NB];
double  *receive4_5flux[NB], *receive4_6flux[NB], *receive4_7flux[NB], *receive4_8flux[NB];
double  *receive5_1flux[NB], *receive5_3flux[NB], *receive5_5flux[NB], *receive5_7flux[NB];
double  *receive6_2flux[NB], *receive6_4flux[NB], *receive6_6flux[NB], *receive6_8flux[NB];

double  *receive1_3flux1[NB], *receive1_4flux1[NB], *receive1_7flux1[NB], *receive1_8flux1[NB];
double  *receive2_1flux1[NB], *receive2_2flux1[NB], *receive2_3flux1[NB], *receive2_4flux1[NB];
double  *receive3_1flux1[NB], *receive3_2flux1[NB], *receive3_5flux1[NB], *receive3_6flux1[NB];
double  *receive4_5flux1[NB], *receive4_6flux1[NB], *receive4_7flux1[NB], *receive4_8flux1[NB];
double  *receive5_1flux1[NB], *receive5_3flux1[NB], *receive5_5flux1[NB], *receive5_7flux1[NB];
double  *receive6_2flux1[NB], *receive6_4flux1[NB], *receive6_6flux1[NB], *receive6_8flux1[NB];
double  *receive1_3flux2[NB], *receive1_4flux2[NB], *receive1_7flux2[NB], *receive1_8flux2[NB];
double  *receive2_1flux2[NB], *receive2_2flux2[NB], *receive2_3flux2[NB], *receive2_4flux2[NB];
double  *receive3_1flux2[NB], *receive3_2flux2[NB], *receive3_5flux2[NB], *receive3_6flux2[NB];
double  *receive4_5flux2[NB], *receive4_6flux2[NB], *receive4_7flux2[NB], *receive4_8flux2[NB];
double  *receive5_1flux2[NB], *receive5_3flux2[NB], *receive5_5flux2[NB], *receive5_7flux2[NB];
double  *receive6_2flux2[NB], *receive6_4flux2[NB], *receive6_6flux2[NB], *receive6_8flux2[NB];
double *receive1_E[NB], *receive2_E[NB], *receive3_E[NB], *receive4_E[NB], *receive5_E[NB], *receive6_E[NB], *receive7_E[NB], *receive8_E[NB];
double *receive1_E1[NB], *receive2_E1[NB], *receive3_E1[NB], *receive4_E1[NB], *receive5_E1[NB], *receive6_E1[NB], *receive7_E1[NB], *receive8_E1[NB];

double  *receive1_3flux[NB], *receive1_4flux[NB], *receive1_7flux[NB], *receive1_8flux[NB];
double  *receive1_3E[NB], *receive1_4E[NB], *receive1_7E[NB], *receive1_8E[NB];
double  *receive2_1E[NB], *receive2_2E[NB], *receive2_3E[NB], *receive2_4E[NB];
double  *receive3_1E[NB], *receive3_2E[NB], *receive3_5E[NB], *receive3_6E[NB];
double  *receive4_5E[NB], *receive4_6E[NB], *receive4_7E[NB], *receive4_8E[NB];
double  *receive5_1E[NB], *receive5_3E[NB], *receive5_5E[NB], *receive5_7E[NB];
double  *receive6_2E[NB], *receive6_4E[NB], *receive6_6E[NB], *receive6_8E[NB];
double  *send1_3[NB], *send1_4[NB], *send1_7[NB], *send1_8[NB];
double  *receive1_3[NB], *receive1_4[NB], *receive1_7[NB], *receive1_8[NB];
double  *tempreceive1_3[NB], *tempreceive1_4[NB], *tempreceive1_7[NB], *tempreceive1_8[NB];

double  *send2_1[NB], *send2_2[NB], *send2_3[NB], *send2_4[NB];
double  *receive2_1[NB], *receive2_2[NB], *receive2_3[NB], *receive2_4[NB];
double  *tempreceive2_1[NB], *tempreceive2_2[NB], *tempreceive2_3[NB], *tempreceive2_4[NB];

double  *send3_1[NB], *send3_2[NB], *send3_5[NB], *send3_6[NB];
double  *receive3_1[NB], *receive3_2[NB], *receive3_5[NB], *receive3_6[NB];
double  *tempreceive3_1[NB], *tempreceive3_2[NB], *tempreceive3_5[NB], *tempreceive3_6[NB];

double  *send4_5[NB], *send4_6[NB], *send4_7[NB], *send4_8[NB];
double  *receive4_5[NB], *receive4_6[NB], *receive4_7[NB], *receive4_8[NB];
double  *tempreceive4_5[NB], *tempreceive4_6[NB], *tempreceive4_7[NB], *tempreceive4_8[NB];

double  *send5_1[NB], *send5_3[NB], *send5_5[NB], *send5_7[NB];
double  *receive5_1[NB], *receive5_3[NB], *receive5_5[NB], *receive5_7[NB];
double  *tempreceive5_1[NB], *tempreceive5_3[NB], *tempreceive5_5[NB], *tempreceive5_7[NB];

double  *send6_2[NB], *send6_4[NB], *send6_6[NB], *send6_8[NB];
double  *receive6_2[NB], *receive6_4[NB], *receive6_6[NB], *receive6_8[NB];
double  *tempreceive6_2[NB], *tempreceive6_4[NB], *tempreceive6_6[NB], *tempreceive6_8[NB];

double *receive1[NB], *receive2[NB], *receive3[NB], *receive4[NB], *receive5[NB], *receive6[NB];
double *tempreceive1[NB], *tempreceive2[NB], *tempreceive3[NB], *tempreceive4[NB], *tempreceive5[NB], *tempreceive6[NB];

double  *receive1_3E2[NB], *receive1_4E2[NB], *receive1_7E2[NB], *receive1_8E2[NB];
double  *receive2_1E2[NB], *receive2_2E2[NB], *receive2_3E2[NB], *receive2_4E2[NB];
double  *receive3_1E2[NB], *receive3_2E2[NB], *receive3_5E2[NB], *receive3_6E2[NB];
double  *receive4_5E2[NB], *receive4_6E2[NB], *receive4_7E2[NB], *receive4_8E2[NB];
double  *receive5_1E2[NB], *receive5_3E2[NB], *receive5_5E2[NB], *receive5_7E2[NB];
double  *receive6_2E2[NB], *receive6_4E2[NB], *receive6_6E2[NB], *receive6_8E2[NB];

double  *receive1_3E1[NB], *receive1_4E1[NB], *receive1_7E1[NB], *receive1_8E1[NB];
double  *receive2_1E1[NB], *receive2_2E1[NB], *receive2_3E1[NB], *receive2_4E1[NB];
double  *receive3_1E1[NB], *receive3_2E1[NB], *receive3_5E1[NB], *receive3_6E1[NB];
double  *receive4_5E1[NB], *receive4_6E1[NB], *receive4_7E1[NB], *receive4_8E1[NB];
double  *receive5_1E1[NB], *receive5_3E1[NB], *receive5_5E1[NB], *receive5_7E1[NB];
double  *receive6_2E1[NB], *receive6_4E1[NB], *receive6_6E1[NB], *receive6_8E1[NB];

double *send_E3_corn1[NB], *send_E3_corn2[NB], *send_E3_corn3[NB], *send_E3_corn4[NB], *send_E2_corn5[NB], *send_E2_corn6[NB],
*send_E2_corn7[NB], *send_E2_corn8[NB], *send_E1_corn9[NB], *send_E1_corn10[NB], *send_E1_corn11[NB], *send_E1_corn12[NB];
double *receive_E3_corn1_1[NB], *receive_E3_corn2_1[NB], *receive_E3_corn3_1[NB], *receive_E3_corn4_1[NB], *receive_E2_corn5_1[NB], *receive_E2_corn6_1[NB],
*receive_E2_corn7_1[NB], *receive_E2_corn8_1[NB], *receive_E1_corn9_1[NB], *receive_E1_corn10_1[NB], *receive_E1_corn11_1[NB], *receive_E1_corn12_1[NB];
double *receive_E3_corn1_2[NB], *receive_E3_corn2_2[NB], *receive_E3_corn3_2[NB], *receive_E3_corn4_2[NB], *receive_E2_corn5_2[NB], *receive_E2_corn6_2[NB],
*receive_E2_corn7_2[NB], *receive_E2_corn8_2[NB], *receive_E1_corn9_2[NB], *receive_E1_corn10_2[NB], *receive_E1_corn11_2[NB], *receive_E1_corn12_2[NB];
double *tempreceive_E3_corn1_1[NB], *tempreceive_E3_corn2_1[NB], *tempreceive_E3_corn3_1[NB], *tempreceive_E3_corn4_1[NB], *tempreceive_E2_corn5_1[NB], *tempreceive_E2_corn6_1[NB],
*tempreceive_E2_corn7_1[NB], *tempreceive_E2_corn8_1[NB], *tempreceive_E1_corn9_1[NB], *tempreceive_E1_corn10_1[NB], *tempreceive_E1_corn11_1[NB], *tempreceive_E1_corn12_1[NB];
double *tempreceive_E3_corn1_2[NB], *tempreceive_E3_corn2_2[NB], *tempreceive_E3_corn3_2[NB], *tempreceive_E3_corn4_2[NB], *tempreceive_E2_corn5_2[NB], *tempreceive_E2_corn6_2[NB],
*tempreceive_E2_corn7_2[NB], *tempreceive_E2_corn8_2[NB], *tempreceive_E1_corn9_2[NB], *tempreceive_E1_corn10_2[NB], *tempreceive_E1_corn11_2[NB], *tempreceive_E1_corn12_2[NB];
double *receive_E3_corn1_12[NB], *receive_E3_corn2_12[NB], *receive_E3_corn3_12[NB], *receive_E3_corn4_12[NB], *receive_E2_corn5_12[NB], *receive_E2_corn6_12[NB],
*receive_E2_corn7_12[NB], *receive_E2_corn8_12[NB], *receive_E1_corn9_12[NB], *receive_E1_corn10_12[NB], *receive_E1_corn11_12[NB], *receive_E1_corn12_12[NB];
double *receive_E3_corn1_22[NB], *receive_E3_corn2_22[NB], *receive_E3_corn3_22[NB], *receive_E3_corn4_22[NB], *receive_E2_corn5_22[NB], *receive_E2_corn6_22[NB],
*receive_E2_corn7_22[NB], *receive_E2_corn8_22[NB], *receive_E1_corn9_22[NB], *receive_E1_corn10_22[NB], *receive_E1_corn11_22[NB], *receive_E1_corn12_22[NB];

double *receive_E3_corn1[NB], *receive_E3_corn2[NB], *receive_E3_corn3[NB], *receive_E3_corn4[NB];
double *receive_E2_corn5[NB], *receive_E2_corn6[NB], *receive_E2_corn7[NB], *receive_E2_corn8[NB];
double *receive_E1_corn9[NB], *receive_E1_corn10[NB], *receive_E1_corn11[NB], *receive_E1_corn12[NB];
double *tempreceive_E3_corn1[NB], *tempreceive_E3_corn2[NB], *tempreceive_E3_corn3[NB], *tempreceive_E3_corn4[NB];
double *tempreceive_E2_corn5[NB], *tempreceive_E2_corn6[NB], *tempreceive_E2_corn7[NB], *tempreceive_E2_corn8[NB];
double *tempreceive_E1_corn9[NB], *tempreceive_E1_corn10[NB], *tempreceive_E1_corn11[NB], *tempreceive_E1_corn12[NB];


double  *cornsend1[NB], *cornsend2[NB], *cornsend3[NB], *cornsend4[NB], *cornsend5[NB], *cornsend7[NB], *cornsend7[NB], *cornsend8[NB];
double *cornreceive1[NB], *cornreceive2[NB], *cornreceive3[NB], *cornreceive4[NB], *cornreceive5[NB], *cornreceive6[NB], *cornreceive7[NB], *cornreceive8[NB];
int *aN1_MPI_offset;
int *aN2_MPI_offset;
int *aN3_MPI_offset;
int *aN1_MPI;
int *aN2_MPI;
int *aN3_MPI;
#if(DO_FONT_FIX)
double *Katm[NB];
#endif

/*************************************************************************
GLOBAL VARIABLES SECTION
*************************************************************************/
/* physics parameters */
double a;
double gam;

/* numerical parameters */
double Rin, Rout, hslope, R0, fractheta;
double cour;
double dV, dx[NB][NPR], startx[NPR];
double dt, bdt[NB][4];
double t, tf;
double rcurr, hcurr;
int istart, istop, jstart, jstop, zstart, zstop;
int icurr, jcurr, pcurr, ihere, jhere, phere;
double dminarg1, dminarg2;
int nstep, first_run;
double fval1, fval2;
double sourceflag, period_max;
double rmax;
int reduce_timestep;
int prestep_half[NB], prestep_full[NB];

/* output parameters */
double DTd;
double DTl;
double DTi;
int    DTr;
double tref;
int    dump_cnt;
int    image_cnt;
int    rdump_cnt;
int    nstroke;

/* global flags */
int    failed;
int    lim;
double defcon;
int flux_flag[NB];

/* diagnostics */
double mdot = 0.;
double edot = 0.;
double ldot = 0.;

/*CUDA variables decleration*/
double *NULL_POINTER[NB];
int gpu;
cudaError_t status;
int fix_mem[NB];
int fix_mem2[NB];
cudaStream_t commandQueue[NQ*N_GPU];
cudaStream_t commandQueueGPU[NB];
cudaEvent_t boundevent[NB][600];
cudaEvent_t boundevent1[NB][100];
cudaEvent_t boundevent2[NB][100];
int nr_workgroups[NB];
int nr_workgroups1[NB];
int nr_workgroups2[NB];
int nr_workgroups2_1[NB];
int nr_workgroups2_2[NB];
int nr_workgroups2_3[NB];
int nr_workgroups3[NB];
int nr_workgroups_special[NB];
int nr_workgroups_special1[NB];
int nr_workgroups_special2[NB];
int nr_workgroups_special3[NB];
int global_work_size[NB][1];
int global_work_size1[NB][1];
int global_work_size2[NB][1];
int global_work_size2_1[NB][1];
int global_work_size2_2[NB][1];
int global_work_size2_3[NB][1];
int global_work_size3[NB][1];
int global_work_size_bound[NB][1];
int global_work_offset[NB][1];
int global_work_size_special[NB][1];
int global_work_size_special1[NB][1];
int global_work_size_special2[NB][1];
int global_work_size_special3[NB][1];
int global_work_intransfer1[NB][1];
int global_work_intransfer2[NB][1];
int global_work_intransfer3[NB][1];
int local_work_size[1];
double * Bufferconn[NB];
double * Buffergcov[NB];
double * Buffergcon[NB];
double * Buffergdet[NB];
double * BufferF1_1[NB];
double * BufferF2_1[NB];
double * BufferF3_1[NB];
double * Bufferdq_1[NB];
double * BufferE_1[NB];
double * BufferV[NB];
double * BufferdU[NB];
double * Bufferradius[NB];
double * Bufferstorage1[NB];
double * Bufferstorage2[NB];
double * Bufferstorage3[NB];
double * Bufferstorage4[NB];
double * Buffereta_avg[NB];
double * Bufferp_1[NB];
double * Bufferph_1[NB];
double * Bufferps_1[NB];
double * Bufferpsh_1[NB];
double * Bufferpbound_1[NB];
double * Bufferdtij[NB];
double * Bufferdiagflux[NB];
int * Bufferpflag[NB];
int * Bufferfailimage[NB];
double * BufferKatm[NB];
double * Buffersend1[NB];
double * Buffersend1_3[NB];
double * Buffersend1_4[NB];
double * Buffersend1_7[NB];
double * Buffersend1_8[NB];
double * Buffersend2[NB];
double * Buffersend2_1[NB];
double * Buffersend2_2[NB];
double * Buffersend2_3[NB];
double * Buffersend2_4[NB];
double * Buffersend3[NB];
double * Buffersend3_1[NB];
double * Buffersend3_2[NB];
double * Buffersend3_5[NB];
double * Buffersend3_6[NB];
double * Buffersend4[NB];
double * Buffersend4_5[NB];
double * Buffersend4_6[NB];
double * Buffersend4_7[NB];
double * Buffersend4_8[NB];
double * Buffersend5[NB];
double * Buffersend5_1[NB];
double * Buffersend5_3[NB];
double * Buffersend5_5[NB];
double * Buffersend5_7[NB];
double * Buffersend6[NB];
double * Buffersend6_2[NB];
double * Buffersend6_4[NB];
double * Buffersend6_6[NB];
double * Buffersend6_8[NB];
double * Bufferrec1[NB];
double * Bufferrec1_3[NB];
double * Bufferrec1_4[NB];
double * Bufferrec1_7[NB];
double * Bufferrec1_8[NB];
double * Bufferrec2[NB];
double * Bufferrec2_1[NB];
double * Bufferrec2_2[NB];
double * Bufferrec2_3[NB];
double * Bufferrec2_4[NB];
double * Bufferrec3[NB];
double * Bufferrec3_1[NB];
double * Bufferrec3_2[NB];
double * Bufferrec3_5[NB];
double * Bufferrec3_6[NB];
double * Bufferrec4[NB];
double * Bufferrec4_5[NB];
double * Bufferrec4_6[NB];
double * Bufferrec4_7[NB];
double * Bufferrec4_8[NB];
double * Bufferrec5[NB];
double * Bufferrec5_1[NB];
double * Bufferrec5_3[NB];
double * Bufferrec5_5[NB];
double * Bufferrec5_7[NB];
double * Bufferrec6[NB];
double * Bufferrec6_2[NB];
double * Bufferrec6_4[NB];
double * Bufferrec6_6[NB];
double * Bufferrec6_8[NB];

double * tempBufferrec1[NB];
double * tempBufferrec1_3[NB];
double * tempBufferrec1_4[NB];
double * tempBufferrec1_7[NB];
double * tempBufferrec1_8[NB];
double * tempBufferrec2[NB];
double * tempBufferrec2_1[NB];
double * tempBufferrec2_2[NB];
double * tempBufferrec2_3[NB];
double * tempBufferrec2_4[NB];
double * tempBufferrec3[NB];
double * tempBufferrec3_1[NB];
double * tempBufferrec3_2[NB];
double * tempBufferrec3_5[NB];
double * tempBufferrec3_6[NB];
double * tempBufferrec4[NB];
double * tempBufferrec4_5[NB];
double * tempBufferrec4_6[NB];
double * tempBufferrec4_7[NB];
double * tempBufferrec4_8[NB];
double * tempBufferrec5[NB];
double * tempBufferrec5_1[NB];
double * tempBufferrec5_3[NB];
double * tempBufferrec5_5[NB];
double * tempBufferrec5_7[NB];
double * tempBufferrec6[NB];
double * tempBufferrec6_2[NB];
double * tempBufferrec6_4[NB];
double * tempBufferrec6_6[NB];
double * tempBufferrec6_8[NB];

double * Buffersend1flux[NB];
double * Buffersend2flux[NB];
double * Buffersend3flux[NB];
double * Buffersend4flux[NB];
double * Buffersend5flux[NB];
double * Buffersend6flux[NB];
double * Bufferrec1flux[NB];
double * Bufferrec1_3flux[NB];
double * Bufferrec1_4flux[NB];
double * Bufferrec1_7flux[NB];
double * Bufferrec1_8flux[NB];
double * Bufferrec2flux[NB];
double * Bufferrec2_1flux[NB];
double * Bufferrec2_2flux[NB];
double * Bufferrec2_3flux[NB];
double * Bufferrec2_4flux[NB];
double * Bufferrec3flux[NB];
double * Bufferrec3_1flux[NB];
double * Bufferrec3_2flux[NB];
double * Bufferrec3_5flux[NB];
double * Bufferrec3_6flux[NB];
double * Bufferrec4flux[NB];
double * Bufferrec4_5flux[NB];
double * Bufferrec4_6flux[NB];
double * Bufferrec4_7flux[NB];
double * Bufferrec4_8flux[NB];
double * Bufferrec5flux[NB];
double * Bufferrec5_1flux[NB];
double * Bufferrec5_3flux[NB];
double * Bufferrec5_5flux[NB];
double * Bufferrec5_7flux[NB];
double * Bufferrec6flux[NB];
double * Bufferrec6_2flux[NB];
double * Bufferrec6_4flux[NB];
double * Bufferrec6_6flux[NB];
double * Bufferrec6_8flux[NB];

double * Bufferrec1flux1[NB];
double * Bufferrec2flux1[NB];
double * Bufferrec3flux1[NB];
double * Bufferrec4flux1[NB];
double * Bufferrec5flux1[NB];
double * Bufferrec6flux1[NB];


double * Bufferrec1_3flux1[NB];
double * Bufferrec1_4flux1[NB];
double * Bufferrec1_7flux1[NB];
double * Bufferrec1_8flux1[NB];
double * Bufferrec2_1flux1[NB];
double * Bufferrec2_2flux1[NB];
double * Bufferrec2_3flux1[NB];
double * Bufferrec2_4flux1[NB];
double * Bufferrec3_1flux1[NB];
double * Bufferrec3_2flux1[NB];
double * Bufferrec3_5flux1[NB];
double * Bufferrec3_6flux1[NB];
double * Bufferrec4_5flux1[NB];
double * Bufferrec4_6flux1[NB];
double * Bufferrec4_7flux1[NB];
double * Bufferrec4_8flux1[NB];
double * Bufferrec5_1flux1[NB];
double * Bufferrec5_3flux1[NB];
double * Bufferrec5_5flux1[NB];
double * Bufferrec5_7flux1[NB];
double * Bufferrec6_2flux1[NB];
double * Bufferrec6_4flux1[NB];
double * Bufferrec6_6flux1[NB];
double * Bufferrec6_8flux1[NB];

double * Bufferrec1_3flux2[NB];
double * Bufferrec1_4flux2[NB];
double * Bufferrec1_7flux2[NB];
double * Bufferrec1_8flux2[NB];
double * Bufferrec2_1flux2[NB];
double * Bufferrec2_2flux2[NB];
double * Bufferrec2_3flux2[NB];
double * Bufferrec2_4flux2[NB];
double * Bufferrec3_1flux2[NB];
double * Bufferrec3_2flux2[NB];
double * Bufferrec3_5flux2[NB];
double * Bufferrec3_6flux2[NB];
double * Bufferrec4_5flux2[NB];
double * Bufferrec4_6flux2[NB];
double * Bufferrec4_7flux2[NB];
double * Bufferrec4_8flux2[NB];
double * Bufferrec5_1flux2[NB];
double * Bufferrec5_3flux2[NB];
double * Bufferrec5_5flux2[NB];
double * Bufferrec5_7flux2[NB];
double * Bufferrec6_2flux2[NB];
double * Bufferrec6_4flux2[NB];
double * Bufferrec6_6flux2[NB];
double * Bufferrec6_8flux2[NB];

double * Buffersend1fine[NB];
double * Buffersend2fine[NB];
double * Buffersend3fine[NB];
double * Buffersend4fine[NB];
double * Buffersend5fine[NB];
double * Buffersend6fine[NB];
double * Bufferrec1fine[NB];
double * Bufferrec2fine[NB];
double * Bufferrec3fine[NB];
double * Bufferrec4fine[NB];
double * Bufferrec5fine[NB];
double * Bufferrec6fine[NB];
double * Bufferrec1_3fine[NB];
double * Bufferrec1_4fine[NB];
double * Bufferrec1_7fine[NB];
double * Bufferrec1_8fine[NB];
double * Bufferrec2_1fine[NB];
double * Bufferrec2_2fine[NB];
double * Bufferrec2_3fine[NB];
double * Bufferrec2_4fine[NB];
double * Bufferrec3_1fine[NB];
double * Bufferrec3_2fine[NB];
double * Bufferrec3_5fine[NB];
double * Bufferrec3_6fine[NB];
double * Bufferrec4_5fine[NB];
double * Bufferrec4_6fine[NB];
double * Bufferrec4_7fine[NB];
double * Bufferrec4_8fine[NB];
double * Bufferrec5_1fine[NB];
double * Bufferrec5_3fine[NB];
double * Bufferrec5_5fine[NB];
double * Bufferrec5_7fine[NB];
double * Bufferrec6_2fine[NB];
double * Bufferrec6_4fine[NB];
double * Bufferrec6_6fine[NB];
double * Bufferrec6_8fine[NB];

double * Buffersend1E[NB];
double * Buffersend2E[NB];
double * Buffersend3E[NB];
double * Buffersend4E[NB];
double * Buffersend5E[NB];
double * Buffersend6E[NB];
double * Bufferrec1E[NB];
double * Bufferrec2E[NB];
double * Bufferrec3E[NB];
double * Bufferrec4E[NB];
double * Bufferrec5E[NB];
double * Bufferrec6E[NB];
double * Bufferrec1E1[NB];
double * Bufferrec2E1[NB];
double * Bufferrec3E1[NB];
double * Bufferrec4E1[NB];
double * Bufferrec5E1[NB];
double * Bufferrec6E1[NB];
double * Bufferrec1_3E[NB];
double * Bufferrec1_4E[NB];
double * Bufferrec1_7E[NB];
double * Bufferrec1_8E[NB];
double * Bufferrec2_1E[NB];
double * Bufferrec2_2E[NB];
double * Bufferrec2_3E[NB];
double * Bufferrec2_4E[NB];
double * Bufferrec3_1E[NB];
double * Bufferrec3_2E[NB];
double * Bufferrec3_5E[NB];
double * Bufferrec3_6E[NB];
double * Bufferrec4_5E[NB];
double * Bufferrec4_6E[NB];
double * Bufferrec4_7E[NB];
double * Bufferrec4_8E[NB];
double * Bufferrec5_1E[NB];
double * Bufferrec5_3E[NB];
double * Bufferrec5_5E[NB];
double * Bufferrec5_7E[NB];
double * Bufferrec6_2E[NB];
double * Bufferrec6_4E[NB];
double * Bufferrec6_6E[NB];
double * Bufferrec6_8E[NB];

double * Bufferrec1_3E1[NB];
double * Bufferrec1_4E1[NB];
double * Bufferrec1_7E1[NB];
double * Bufferrec1_8E1[NB];
double * Bufferrec2_1E1[NB];
double * Bufferrec2_2E1[NB];
double * Bufferrec2_3E1[NB];
double * Bufferrec2_4E1[NB];
double * Bufferrec3_1E1[NB];
double * Bufferrec3_2E1[NB];
double * Bufferrec3_5E1[NB];
double * Bufferrec3_6E1[NB];
double * Bufferrec4_5E1[NB];
double * Bufferrec4_6E1[NB];
double * Bufferrec4_7E1[NB];
double * Bufferrec4_8E1[NB];
double * Bufferrec5_1E1[NB];
double * Bufferrec5_3E1[NB];
double * Bufferrec5_5E1[NB];
double * Bufferrec5_7E1[NB];
double * Bufferrec6_2E1[NB];
double * Bufferrec6_4E1[NB];
double * Bufferrec6_6E1[NB];
double * Bufferrec6_8E1[NB];

double * Bufferrec1_3E2[NB];
double * Bufferrec1_4E2[NB];
double * Bufferrec1_7E2[NB];
double * Bufferrec1_8E2[NB];
double * Bufferrec2_1E2[NB];
double * Bufferrec2_2E2[NB];
double * Bufferrec2_3E2[NB];
double * Bufferrec2_4E2[NB];
double * Bufferrec3_1E2[NB];
double * Bufferrec3_2E2[NB];
double * Bufferrec3_5E2[NB];
double * Bufferrec3_6E2[NB];
double * Bufferrec4_5E2[NB];
double * Bufferrec4_6E2[NB];
double * Bufferrec4_7E2[NB];
double * Bufferrec4_8E2[NB];
double * Bufferrec5_1E2[NB];
double * Bufferrec5_3E2[NB];
double * Bufferrec5_5E2[NB];
double * Bufferrec5_7E2[NB];
double * Bufferrec6_2E2[NB];
double * Bufferrec6_4E2[NB];
double * Bufferrec6_6E2[NB];
double * Bufferrec6_8E2[NB];

double * BuffersendE1corn9[NB];
double * BuffersendE1corn10[NB];
double * BuffersendE1corn11[NB];
double * BuffersendE1corn12[NB];
double * BuffersendE2corn5[NB];
double * BuffersendE2corn6[NB];
double * BuffersendE2corn7[NB];
double * BuffersendE2corn8[NB];
double * BuffersendE3corn1[NB];
double * BuffersendE3corn2[NB];
double * BuffersendE3corn3[NB];
double * BuffersendE3corn4[NB];
double * BufferrecE1corn9[NB];
double * BufferrecE1corn10[NB];
double * BufferrecE1corn11[NB];
double * BufferrecE1corn12[NB];
double * BufferrecE2corn5[NB];
double * BufferrecE2corn6[NB];
double * BufferrecE2corn7[NB];
double * BufferrecE2corn8[NB];
double * BufferrecE3corn1[NB];
double * BufferrecE3corn2[NB];
double * BufferrecE3corn3[NB];
double * BufferrecE3corn4[NB];

double * tempBufferrecE1corn9[NB];
double * tempBufferrecE1corn10[NB];
double * tempBufferrecE1corn11[NB];
double * tempBufferrecE1corn12[NB];
double * tempBufferrecE2corn5[NB];
double * tempBufferrecE2corn6[NB];
double * tempBufferrecE2corn7[NB];
double * tempBufferrecE2corn8[NB];
double * tempBufferrecE3corn1[NB];
double * tempBufferrecE3corn2[NB];
double * tempBufferrecE3corn3[NB];
double * tempBufferrecE3corn4[NB];

double * BufferrecE1corn9_3[NB];
double * BufferrecE1corn9_7[NB];
double * BufferrecE1corn10_1[NB];
double * BufferrecE1corn10_5[NB];
double * BufferrecE1corn11_2[NB];
double * BufferrecE1corn11_6[NB];
double * BufferrecE1corn12_4[NB];
double * BufferrecE1corn12_8[NB];
double * BufferrecE2corn5_2[NB];
double * BufferrecE2corn5_4[NB];
double * BufferrecE2corn6_1[NB];
double * BufferrecE2corn6_3[NB];
double * BufferrecE2corn7_5[NB];
double * BufferrecE2corn7_7[NB];
double * BufferrecE2corn8_6[NB];
double * BufferrecE2corn8_8[NB];
double * BufferrecE3corn1_3[NB];
double * BufferrecE3corn1_4[NB];
double * BufferrecE3corn2_1[NB];
double * BufferrecE3corn2_2[NB];
double * BufferrecE3corn3_5[NB];
double * BufferrecE3corn3_6[NB];
double * BufferrecE3corn4_7[NB];
double * BufferrecE3corn4_8[NB];

double * tempBufferrecE1corn9_3[NB];
double * tempBufferrecE1corn9_7[NB];
double * tempBufferrecE1corn10_1[NB];
double * tempBufferrecE1corn10_5[NB];
double * tempBufferrecE1corn11_2[NB];
double * tempBufferrecE1corn11_6[NB];
double * tempBufferrecE1corn12_4[NB];
double * tempBufferrecE1corn12_8[NB];
double * tempBufferrecE2corn5_2[NB];
double * tempBufferrecE2corn5_4[NB];
double * tempBufferrecE2corn6_1[NB];
double * tempBufferrecE2corn6_3[NB];
double * tempBufferrecE2corn7_5[NB];
double * tempBufferrecE2corn7_7[NB];
double * tempBufferrecE2corn8_6[NB];
double * tempBufferrecE2corn8_8[NB];
double * tempBufferrecE3corn1_3[NB];
double * tempBufferrecE3corn1_4[NB];
double * tempBufferrecE3corn2_1[NB];
double * tempBufferrecE3corn2_2[NB];
double * tempBufferrecE3corn3_5[NB];
double * tempBufferrecE3corn3_6[NB];
double * tempBufferrecE3corn4_7[NB];
double * tempBufferrecE3corn4_8[NB];

double * BufferrecE1corn9_32[NB];
double * BufferrecE1corn9_72[NB];
double * BufferrecE1corn10_12[NB];
double * BufferrecE1corn10_52[NB];
double * BufferrecE1corn11_22[NB];
double * BufferrecE1corn11_62[NB];
double * BufferrecE1corn12_42[NB];
double * BufferrecE1corn12_82[NB];
double * BufferrecE2corn5_22[NB];
double * BufferrecE2corn5_42[NB];
double * BufferrecE2corn6_12[NB];
double * BufferrecE2corn6_32[NB];
double * BufferrecE2corn7_52[NB];
double * BufferrecE2corn7_72[NB];
double * BufferrecE2corn8_62[NB];
double * BufferrecE2corn8_82[NB];
double * BufferrecE3corn1_32[NB];
double * BufferrecE3corn1_42[NB];
double * BufferrecE3corn2_12[NB];
double * BufferrecE3corn2_22[NB];
double * BufferrecE3corn3_52[NB];
double * BufferrecE3corn3_62[NB];
double * BufferrecE3corn4_72[NB];
double * BufferrecE3corn4_82[NB];


double * Bufferboundrec1_MPI[NB];
double * Bufferboundrec2_MPI[NB];
double * Bufferboundrec3_MPI[NB];
double * Bufferboundrec4_MPI[NB];
double * Bufferboundrec5_MPI[NB];
double * Bufferboundrec6_MPI[NB];
double * Bufferboundrec7_MPI[NB];
double * Bufferboundrec8_MPI[NB];
double * Bufferboundsend1_MPI[NB];
double * Bufferboundsend2_MPI[NB];
double * Bufferboundsend3_MPI[NB];
double * Bufferboundsend4_MPI[NB];
double * Bufferboundsend5_MPI[NB];
double * Bufferboundsend6_MPI[NB];
double * Bufferboundsend7_MPI[NB];
double * Bufferboundsend8_MPI[NB];
int receive_tag;

/*Timers*/
clock_t begin1, end1, begin2, end2;
double time_spent3;

/*Parallel write*/
double *dump_buffer;
double(*restrict dxdxp_z[NB])[NDIM][NDIM];
double(*restrict dxpdx_z[NB])[NDIM][NDIM];
double ndt, ndt1, ndt2, ndt3;

//AMR stuff
int(*block)[NV];
int n_ord[NB], n_ord_gpu[N_GPU][NB], n_ord_total[NB], n_ord_RM[NB], n_ord_total_RM[NB];
int n_active, n_active_gpu[N_GPU], n_active_total, n_max;
MPI_Request request_timelevel[NB];

//I/O stuff
MPI_File fdump[100], fdumpdiag[100], rdump[NB];
MPI_Request req_block[NB][1];
MPI_Request req_block_rdump[NB][1];
MPI_Request req_blockdiag[NB][1];
float *array[NB], *array_diag[NB];
double *array_rdump[NB];
int first_dump, first_rdump;