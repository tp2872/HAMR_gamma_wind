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
#define restrict
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "config.h"

/*************************************************************************
GLOBAL ARRAY SECTION
*************************************************************************/
/* for debug */
extern int(*restrict failimage[NB])[NFAIL];
#if(DO_FONT_FIX)
extern double *Katm[NB];
#endif

/*CPU arrays*/
extern double(*restrict V[NB])[6];
extern double(*restrict p[NB])[NPR];
extern double E_avg1[NB_1*NB_3][BS_1 + 2 * N1G];
extern double E_avg2[NB_1*NB_3][BS_1 + 2 * N1G];
extern double E_avg1_new[NB_1*NB_3][BS_1 + 2 * N1G];
extern double E_avg2_new[NB_1*NB_3][BS_1 + 2 * N1G];
extern double(*restrict  ph[NB])[NPR];
extern double(*restrict E_corn[NB])[NDIM];
extern double(*restrict dE[NB])[2][NDIM][NDIM];
extern double(*restrict ps[NB])[NDIM];
extern double(*restrict psh[NB])[NDIM];
extern double(*restrict dq[NB])[NPR];
extern double(*restrict F1[NB])[NPR];
extern double(*restrict F2[NB])[NPR];
extern double(*restrict F3[NB])[NPR];
extern double(*restrict stor1[NB])[NPR];
extern double(*restrict stor2[NB])[NPR];
extern int(*restrict pflag[NB]);
extern double(*restrict conn[NB])[NDIM][NDIM][NDIM];
extern double(*restrict gcon[NB])[NPG][NDIM][NDIM];
extern double(*restrict gcov[NB])[NPG][NDIM][NDIM];
extern double(*restrict gdet[NB])[NPG];
extern double(*restrict dU_s[NB])[NPR];

/*GPU transfer arrays*/
extern double *F1_1[NB];
extern double *F2_1[NB];
extern double *F3_1[NB];
extern double *dq_1[NB];
extern double *dU_GPU[NB];
extern double *p_1[NB];
extern double *ph_1[NB];
extern double *ps_1[NB];
extern double *psh_1[NB];
extern double *gcov_GPU[NB];
extern double *gcon_GPU[NB];
extern double *conn_GPU[NB];
extern double *gdet_GPU[NB];
extern double *dtij_GPU[NB];
extern double *Katm_GPU[NB];
extern int *pflag_GPU[NB];
extern int *failimage_GPU[NB];

/*MPI arrays*/
extern double  *send1[NB], *send2[NB], *send3[NB], *send4[NB], *send5[NB], *send6[NB];
extern double  *send1_fine[NB], *send2_fine[NB], *send3_fine[NB], *send4_fine[NB], *send5_fine[NB], *send6_fine[NB];

extern double  *send1_flux[NB], *send2_flux[NB], *send3_flux[NB], *send4_flux[NB], *send5_flux[NB], *send6_flux[NB], *send7_flux[NB], *send8_flux[NB];
extern double  *send1_E[NB], *send2_E[NB], *send3_E[NB], *send4_E[NB], *send5_E[NB], *send6_E[NB], *send7_E[NB], *send8_E[NB];

extern double  *send1_3[NB], *send1_4[NB], *send1_7[NB], *send1_8[NB];
extern double  *receive1_3[NB], *receive1_4[NB], *receive1_7[NB], *receive1_8[NB];
extern double  *tempreceive1_3[NB], *tempreceive1_4[NB], *tempreceive1_7[NB], *tempreceive1_8[NB];

extern double  *send2_1[NB], *send2_2[NB], *send2_3[NB], *send2_4[NB];
extern double  *receive2_1[NB], *receive2_2[NB], *receive2_3[NB], *receive2_4[NB];
extern double  *tempreceive2_1[NB], *tempreceive2_2[NB], *tempreceive2_3[NB], *tempreceive2_4[NB];

extern double  *send3_1[NB], *send3_2[NB], *send3_5[NB], *send3_6[NB];
extern double  *receive3_1[NB], *receive3_2[NB], *receive3_5[NB], *receive3_6[NB];
extern double  *tempreceive3_1[NB], *tempreceive3_2[NB], *tempreceive3_5[NB], *tempreceive3_6[NB];

extern double  *send4_5[NB], *send4_6[NB], *send4_7[NB], *send4_8[NB];
extern double  *receive4_5[NB], *receive4_6[NB], *receive4_7[NB], *receive4_8[NB];
extern double  *tempreceive4_5[NB], *tempreceive4_6[NB], *tempreceive4_7[NB], *tempreceive4_8[NB];

extern double  *send5_1[NB], *send5_3[NB], *send5_5[NB], *send5_7[NB];
extern double  *receive5_1[NB], *receive5_3[NB], *receive5_5[NB], *receive5_7[NB];
extern double  *tempreceive5_1[NB], *tempreceive5_3[NB], *tempreceive5_5[NB], *tempreceive5_7[NB];

extern double  *send6_2[NB], *send6_4[NB], *send6_6[NB], *send6_8[NB];
extern double  *receive6_2[NB], *receive6_4[NB], *receive6_6[NB], *receive6_8[NB];
extern double  *tempreceive6_2[NB], *tempreceive6_4[NB], *tempreceive6_6[NB], *tempreceive6_8[NB];

extern double *receive1[NB], *receive2[NB], *receive3[NB], *receive4[NB], *receive5[NB], *receive6[NB];
extern double *tempreceive1[NB], *tempreceive2[NB], *tempreceive3[NB], *tempreceive4[NB], *tempreceive5[NB], *tempreceive6[NB];

extern double *receive1_fine[NB], *receive2_fine[NB], *receive3_fine[NB], *receive4_fine[NB], *receive5_fine[NB], *receive6_fine[NB];
extern double  *receive1_3fine[NB], *receive1_4fine[NB], *receive1_7fine[NB], *receive1_8fine[NB];
extern double  *receive2_1fine[NB], *receive2_2fine[NB], *receive2_3fine[NB], *receive2_4fine[NB];
extern double  *receive3_1fine[NB], *receive3_2fine[NB], *receive3_5fine[NB], *receive3_6fine[NB];
extern double  *receive4_5fine[NB], *receive4_6fine[NB], *receive4_7fine[NB], *receive4_8fine[NB];
extern double  *receive5_1fine[NB], *receive5_3fine[NB], *receive5_5fine[NB], *receive5_7fine[NB];
extern double  *receive6_2fine[NB], *receive6_4fine[NB], *receive6_6fine[NB], *receive6_8fine[NB];
extern double *receive1_flux[NB], *receive2_flux[NB], *receive3_flux[NB], *receive4_flux[NB], *receive5_flux[NB], *receive6_flux[NB], *receive7_flux[NB], *receive8_flux[NB];
extern double *receive1_flux1[NB], *receive2_flux1[NB], *receive3_flux1[NB], *receive4_flux1[NB], *receive5_flux1[NB], *receive6_flux1[NB], *receive7_flux1[NB], *receive8_flux1[NB];

extern double  *receive1_3flux[NB], *receive1_4flux[NB], *receive1_7flux[NB], *receive1_8flux[NB];
extern double  *receive2_1flux[NB], *receive2_2flux[NB], *receive2_3flux[NB], *receive2_4flux[NB];
extern double  *receive3_1flux[NB], *receive3_2flux[NB], *receive3_5flux[NB], *receive3_6flux[NB];
extern double  *receive4_5flux[NB], *receive4_6flux[NB], *receive4_7flux[NB], *receive4_8flux[NB];
extern double  *receive5_1flux[NB], *receive5_3flux[NB], *receive5_5flux[NB], *receive5_7flux[NB];
extern double  *receive6_2flux[NB], *receive6_4flux[NB], *receive6_6flux[NB], *receive6_8flux[NB];

extern double  *receive1_3flux1[NB], *receive1_4flux1[NB], *receive1_7flux1[NB], *receive1_8flux1[NB];
extern double  *receive2_1flux1[NB], *receive2_2flux1[NB], *receive2_3flux1[NB], *receive2_4flux1[NB];
extern double  *receive3_1flux1[NB], *receive3_2flux1[NB], *receive3_5flux1[NB], *receive3_6flux1[NB];
extern double  *receive4_5flux1[NB], *receive4_6flux1[NB], *receive4_7flux1[NB], *receive4_8flux1[NB];
extern double  *receive5_1flux1[NB], *receive5_3flux1[NB], *receive5_5flux1[NB], *receive5_7flux1[NB];
extern double  *receive6_2flux1[NB], *receive6_4flux1[NB], *receive6_6flux1[NB], *receive6_8flux1[NB];

extern double  *receive1_3flux2[NB], *receive1_4flux2[NB], *receive1_7flux2[NB], *receive1_8flux2[NB];
extern double  *receive2_1flux2[NB], *receive2_2flux2[NB], *receive2_3flux2[NB], *receive2_4flux2[NB];
extern double  *receive3_1flux2[NB], *receive3_2flux2[NB], *receive3_5flux2[NB], *receive3_6flux2[NB];
extern double  *receive4_5flux2[NB], *receive4_6flux2[NB], *receive4_7flux2[NB], *receive4_8flux2[NB];
extern double  *receive5_1flux2[NB], *receive5_3flux2[NB], *receive5_5flux2[NB], *receive5_7flux2[NB];
extern double  *receive6_2flux2[NB], *receive6_4flux2[NB], *receive6_6flux2[NB], *receive6_8flux2[NB];

extern double *receive1_E[NB], *receive2_E[NB], *receive3_E[NB], *receive4_E[NB], *receive5_E[NB], *receive6_E[NB], *receive7_E[NB], *receive8_E[NB];
extern double *receive1_E1[NB], *receive2_E1[NB], *receive3_E1[NB], *receive4_E1[NB], *receive5_E1[NB], *receive6_E1[NB], *receive7_E1[NB], *receive8_E1[NB];

extern double  *receive1_3E[NB], *receive1_4E[NB], *receive1_7E[NB], *receive1_8E[NB];
extern double  *receive2_1E[NB], *receive2_2E[NB], *receive2_3E[NB], *receive2_4E[NB];
extern double  *receive3_1E[NB], *receive3_2E[NB], *receive3_5E[NB], *receive3_6E[NB];
extern double  *receive4_5E[NB], *receive4_6E[NB], *receive4_7E[NB], *receive4_8E[NB];
extern double  *receive5_1E[NB], *receive5_3E[NB], *receive5_5E[NB], *receive5_7E[NB];
extern double  *receive6_2E[NB], *receive6_4E[NB], *receive6_6E[NB], *receive6_8E[NB];

extern double  *receive1_3E2[NB], *receive1_4E2[NB], *receive1_7E2[NB], *receive1_8E2[NB];
extern double  *receive2_1E2[NB], *receive2_2E2[NB], *receive2_3E2[NB], *receive2_4E2[NB];
extern double  *receive3_1E2[NB], *receive3_2E2[NB], *receive3_5E2[NB], *receive3_6E2[NB];
extern double  *receive4_5E2[NB], *receive4_6E2[NB], *receive4_7E2[NB], *receive4_8E2[NB];
extern double  *receive5_1E2[NB], *receive5_3E2[NB], *receive5_5E2[NB], *receive5_7E2[NB];
extern double  *receive6_2E2[NB], *receive6_4E2[NB], *receive6_6E2[NB], *receive6_8E2[NB];

extern double  *receive1_3E1[NB], *receive1_4E1[NB], *receive1_7E1[NB], *receive1_8E1[NB];
extern double  *receive2_1E1[NB], *receive2_2E1[NB], *receive2_3E1[NB], *receive2_4E1[NB];
extern double  *receive3_1E1[NB], *receive3_2E1[NB], *receive3_5E1[NB], *receive3_6E1[NB];
extern double  *receive4_5E1[NB], *receive4_6E1[NB], *receive4_7E1[NB], *receive4_8E1[NB];
extern double  *receive5_1E1[NB], *receive5_3E1[NB], *receive5_5E1[NB], *receive5_7E1[NB];
extern double  *receive6_2E1[NB], *receive6_4E1[NB], *receive6_6E1[NB], *receive6_8E1[NB];

extern double  *cornsend1[NB], *cornsend2[NB], *cornsend3[NB], *cornsend4[NB], *cornsend5[NB], *cornsend7[NB], *cornsend7[NB], *cornsend8[NB];
extern double *cornreceive1[NB], *cornreceive2[NB], *cornreceive3[NB], *cornreceive4[NB], *cornreceive5[NB], *cornreceive6[NB], *cornreceive7[NB], *cornreceive8[NB];

extern double *send_E3_corn1[NB], *send_E3_corn2[NB], *send_E3_corn3[NB], *send_E3_corn4[NB], *send_E2_corn5[NB], *send_E2_corn6[NB],
*send_E2_corn7[NB], *send_E2_corn8[NB], *send_E1_corn9[NB], *send_E1_corn10[NB], *send_E1_corn11[NB], *send_E1_corn12[NB];
extern double *receive_E3_corn1_1[NB], *receive_E3_corn2_1[NB], *receive_E3_corn3_1[NB], *receive_E3_corn4_1[NB], *receive_E2_corn5_1[NB], *receive_E2_corn6_1[NB],
*receive_E2_corn7_1[NB], *receive_E2_corn8_1[NB], *receive_E1_corn9_1[NB], *receive_E1_corn10_1[NB], *receive_E1_corn11_1[NB], *receive_E1_corn12_1[NB];
extern double *receive_E3_corn1_2[NB], *receive_E3_corn2_2[NB], *receive_E3_corn3_2[NB], *receive_E3_corn4_2[NB], *receive_E2_corn5_2[NB], *receive_E2_corn6_2[NB],
*receive_E2_corn7_2[NB], *receive_E2_corn8_2[NB], *receive_E1_corn9_2[NB], *receive_E1_corn10_2[NB], *receive_E1_corn11_2[NB], *receive_E1_corn12_2[NB];

extern double *tempreceive_E3_corn1_1[NB], *tempreceive_E3_corn2_1[NB], *tempreceive_E3_corn3_1[NB], *tempreceive_E3_corn4_1[NB], *tempreceive_E2_corn5_1[NB], *tempreceive_E2_corn6_1[NB],
*tempreceive_E2_corn7_1[NB], *tempreceive_E2_corn8_1[NB], *tempreceive_E1_corn9_1[NB], *tempreceive_E1_corn10_1[NB], *tempreceive_E1_corn11_1[NB], *tempreceive_E1_corn12_1[NB];
extern double *tempreceive_E3_corn1_2[NB], *tempreceive_E3_corn2_2[NB], *tempreceive_E3_corn3_2[NB], *tempreceive_E3_corn4_2[NB], *tempreceive_E2_corn5_2[NB], *tempreceive_E2_corn6_2[NB],
*tempreceive_E2_corn7_2[NB], *tempreceive_E2_corn8_2[NB], *tempreceive_E1_corn9_2[NB], *tempreceive_E1_corn10_2[NB], *tempreceive_E1_corn11_2[NB], *tempreceive_E1_corn12_2[NB];

extern double *receive_E3_corn1_12[NB], *receive_E3_corn2_12[NB], *receive_E3_corn3_12[NB], *receive_E3_corn4_12[NB], *receive_E2_corn5_12[NB], *receive_E2_corn6_12[NB],
*receive_E2_corn7_12[NB], *receive_E2_corn8_12[NB], *receive_E1_corn9_12[NB], *receive_E1_corn10_12[NB], *receive_E1_corn11_12[NB], *receive_E1_corn12_12[NB];
extern double *receive_E3_corn1_22[NB], *receive_E3_corn2_22[NB], *receive_E3_corn3_22[NB], *receive_E3_corn4_22[NB], *receive_E2_corn5_22[NB], *receive_E2_corn6_22[NB],
*receive_E2_corn7_22[NB], *receive_E2_corn8_22[NB], *receive_E1_corn9_22[NB], *receive_E1_corn10_22[NB], *receive_E1_corn11_22[NB], *receive_E1_corn12_22[NB];

extern double *receive_E3_corn1[NB], *receive_E3_corn2[NB], *receive_E3_corn3[NB], *receive_E3_corn4[NB];
extern double *receive_E2_corn5[NB], *receive_E2_corn6[NB], *receive_E2_corn7[NB], *receive_E2_corn8[NB];
extern double *receive_E1_corn9[NB], *receive_E1_corn10[NB], *receive_E1_corn11[NB], *receive_E1_corn12[NB];

extern double *tempreceive_E3_corn1[NB], *tempreceive_E3_corn2[NB], *tempreceive_E3_corn3[NB], *tempreceive_E3_corn4[NB];
extern double *tempreceive_E2_corn5[NB], *tempreceive_E2_corn6[NB], *tempreceive_E2_corn7[NB], *tempreceive_E2_corn8[NB];
extern double *tempreceive_E1_corn9[NB], *tempreceive_E1_corn10[NB], *tempreceive_E1_corn11[NB], *tempreceive_E1_corn12[NB];

/*CUDA arrays decleration*/
extern double *NULL_POINTER[NB];
extern cudaStream_t commandQueue[NB];
extern cudaStream_t commandQueueGPU[NB];
extern cudaEvent_t boundevent[NB][600];
extern cudaEvent_t boundevent1[NB][100];
extern cudaEvent_t boundevent2[NB][100];
extern int fix_mem[NB];
extern int fix_mem2[NB];
extern int nr_workgroups[NB];
extern int nr_workgroups1[NB];
extern int nr_workgroups2[NB];
extern int nr_workgroups2_1[NB];
extern int nr_workgroups2_2[NB];
extern int nr_workgroups2_3[NB];
extern int nr_workgroups3[NB];
extern int nr_workgroups_special[NB];
extern int nr_workgroups_special1[NB];
extern int nr_workgroups_special2[NB];
extern int nr_workgroups_special3[NB];
extern int global_work_size[NB][1];
extern int global_work_size1[NB][1];
extern int global_work_size2[NB][1];
extern int global_work_size2_1[NB][1];
extern int global_work_size2_2[NB][1];
extern int global_work_size2_3[NB][1];
extern int global_work_size3[NB][1];
extern int global_work_size_bound[NB][1];
extern int global_work_offset[NB][1];
extern int global_work_size_special[NB][1];
extern int global_work_size_special1[NB][1];
extern int global_work_size_special2[NB][1];
extern int global_work_size_special3[NB][1];
extern int local_work_size[1];
extern double * Bufferconn[NB];
extern double * Buffergcov[NB];
extern double * Buffergcon[NB];
extern double * Buffergdet[NB];
extern double * BufferF1_1[NB];
extern double * BufferF2_1[NB];
extern double * BufferF3_1[NB];
extern double * Bufferdq_1[NB];
extern double * BufferE_1[NB];
extern double * BufferV[NB];
extern double * BufferdU[NB];
extern double * Bufferradius[NB];
extern double * Bufferstorage1[NB];
extern double * Bufferstorage2[NB];
extern double * Bufferstorage3[NB];
extern double * Bufferstorage4[NB];
extern double * Buffereta_avg[NB];
extern double * Bufferp_1[NB];
extern double * Bufferph_1[NB];
extern double * Bufferps_1[NB];
extern double * Bufferpsh_1[NB];
extern double * Bufferdtij[NB];
extern int * Bufferpflag[NB];
extern int * Bufferfailimage[NB];
extern double * BufferKatm[NB];
extern double * Buffersend1[NB];
extern double * Buffersend1_3[NB];
extern double * Buffersend1_4[NB];
extern double * Buffersend1_7[NB];
extern double * Buffersend1_8[NB];
extern double * Buffersend2[NB];
extern double * Buffersend2_1[NB];
extern double * Buffersend2_2[NB];
extern double * Buffersend2_3[NB];
extern double * Buffersend2_4[NB];
extern double * Buffersend3[NB];
extern double * Buffersend3_1[NB];
extern double * Buffersend3_2[NB];
extern double * Buffersend3_5[NB];
extern double * Buffersend3_6[NB];
extern double * Buffersend4[NB];
extern double * Buffersend4_5[NB];
extern double * Buffersend4_6[NB];
extern double * Buffersend4_7[NB];
extern double * Buffersend4_8[NB];
extern double * Buffersend5[NB];
extern double * Buffersend5_1[NB];
extern double * Buffersend5_3[NB];
extern double * Buffersend5_5[NB];
extern double * Buffersend5_7[NB];
extern double * Buffersend6[NB];
extern double * Buffersend6_2[NB];
extern double * Buffersend6_4[NB];
extern double * Buffersend6_6[NB];
extern double * Buffersend6_8[NB];
extern double * Bufferrec1[NB];
extern double * Bufferrec1_3[NB];
extern double * Bufferrec1_4[NB];
extern double * Bufferrec1_7[NB];
extern double * Bufferrec1_8[NB];
extern double * Bufferrec2[NB];
extern double * Bufferrec2_1[NB];
extern double * Bufferrec2_2[NB];
extern double * Bufferrec2_3[NB];
extern double * Bufferrec2_4[NB];
extern double * Bufferrec3[NB];
extern double * Bufferrec3_1[NB];
extern double * Bufferrec3_2[NB];
extern double * Bufferrec3_5[NB];
extern double * Bufferrec3_6[NB];
extern double * Bufferrec4[NB];
extern double * Bufferrec4_5[NB];
extern double * Bufferrec4_6[NB];
extern double * Bufferrec4_7[NB];
extern double * Bufferrec4_8[NB];
extern double * Bufferrec5[NB];
extern double * Bufferrec5_1[NB];
extern double * Bufferrec5_3[NB];
extern double * Bufferrec5_5[NB];
extern double * Bufferrec5_7[NB];
extern double * Bufferrec6[NB];
extern double * Bufferrec6_2[NB];
extern double * Bufferrec6_4[NB];
extern double * Bufferrec6_6[NB];
extern double * Bufferrec6_8[NB];

extern double * tempBufferrec1[NB];
extern double * tempBufferrec1_3[NB];
extern double * tempBufferrec1_4[NB];
extern double * tempBufferrec1_7[NB];
extern double * tempBufferrec1_8[NB];
extern double * tempBufferrec2[NB];
extern double * tempBufferrec2_1[NB];
extern double * tempBufferrec2_2[NB];
extern double * tempBufferrec2_3[NB];
extern double * tempBufferrec2_4[NB];
extern double * tempBufferrec3[NB];
extern double * tempBufferrec3_1[NB];
extern double * tempBufferrec3_2[NB];
extern double * tempBufferrec3_5[NB];
extern double * tempBufferrec3_6[NB];
extern double * tempBufferrec4[NB];
extern double * tempBufferrec4_5[NB];
extern double * tempBufferrec4_6[NB];
extern double * tempBufferrec4_7[NB];
extern double * tempBufferrec4_8[NB];
extern double * tempBufferrec5[NB];
extern double * tempBufferrec5_1[NB];
extern double * tempBufferrec5_3[NB];
extern double * tempBufferrec5_5[NB];
extern double * tempBufferrec5_7[NB];
extern double * tempBufferrec6[NB];
extern double * tempBufferrec6_2[NB];
extern double * tempBufferrec6_4[NB];
extern double * tempBufferrec6_6[NB];
extern double * tempBufferrec6_8[NB];

extern double * Buffersend1flux[NB];
extern double * Buffersend2flux[NB];
extern double * Buffersend3flux[NB];
extern double * Buffersend4flux[NB];
extern double * Buffersend5flux[NB];
extern double * Buffersend6flux[NB];
extern double * Bufferrec1flux[NB];
extern double * Bufferrec1_3flux[NB];
extern double * Bufferrec1_4flux[NB];
extern double * Bufferrec1_7flux[NB];
extern double * Bufferrec1_8flux[NB];
extern double * Bufferrec2flux[NB];
extern double * Bufferrec2_1flux[NB];
extern double * Bufferrec2_2flux[NB];
extern double * Bufferrec2_3flux[NB];
extern double * Bufferrec2_4flux[NB];
extern double * Bufferrec3flux[NB];
extern double * Bufferrec3_1flux[NB];
extern double * Bufferrec3_2flux[NB];
extern double * Bufferrec3_5flux[NB];
extern double * Bufferrec3_6flux[NB];
extern double * Bufferrec4flux[NB];
extern double * Bufferrec4_5flux[NB];
extern double * Bufferrec4_6flux[NB];
extern double * Bufferrec4_7flux[NB];
extern double * Bufferrec4_8flux[NB];
extern double * Bufferrec5flux[NB];
extern double * Bufferrec5_1flux[NB];
extern double * Bufferrec5_3flux[NB];
extern double * Bufferrec5_5flux[NB];
extern double * Bufferrec5_7flux[NB];
extern double * Bufferrec6flux[NB];
extern double * Bufferrec6_2flux[NB];
extern double * Bufferrec6_4flux[NB];
extern double * Bufferrec6_6flux[NB];
extern double * Bufferrec6_8flux[NB];

extern double * Bufferrec1flux1[NB];
extern double * Bufferrec2flux1[NB];
extern double * Bufferrec3flux1[NB];
extern double * Bufferrec4flux1[NB];
extern double * Bufferrec5flux1[NB];
extern double * Bufferrec6flux1[NB];


extern double * Bufferrec1_3flux1[NB];
extern double * Bufferrec1_4flux1[NB];
extern double * Bufferrec1_7flux1[NB];
extern double * Bufferrec1_8flux1[NB];
extern double * Bufferrec2_1flux1[NB];
extern double * Bufferrec2_2flux1[NB];
extern double * Bufferrec2_3flux1[NB];
extern double * Bufferrec2_4flux1[NB];
extern double * Bufferrec3_1flux1[NB];
extern double * Bufferrec3_2flux1[NB];
extern double * Bufferrec3_5flux1[NB];
extern double * Bufferrec3_6flux1[NB];
extern double * Bufferrec4_5flux1[NB];
extern double * Bufferrec4_6flux1[NB];
extern double * Bufferrec4_7flux1[NB];
extern double * Bufferrec4_8flux1[NB];
extern double * Bufferrec5_1flux1[NB];
extern double * Bufferrec5_3flux1[NB];
extern double * Bufferrec5_5flux1[NB];
extern double * Bufferrec5_7flux1[NB];
extern double * Bufferrec6_2flux1[NB];
extern double * Bufferrec6_4flux1[NB];
extern double * Bufferrec6_6flux1[NB];
extern double * Bufferrec6_8flux1[NB];

extern double * Bufferrec1_3flux2[NB];
extern double * Bufferrec1_4flux2[NB];
extern double * Bufferrec1_7flux2[NB];
extern double * Bufferrec1_8flux2[NB];
extern double * Bufferrec2_1flux2[NB];
extern double * Bufferrec2_2flux2[NB];
extern double * Bufferrec2_3flux2[NB];
extern double * Bufferrec2_4flux2[NB];
extern double * Bufferrec3_1flux2[NB];
extern double * Bufferrec3_2flux2[NB];
extern double * Bufferrec3_5flux2[NB];
extern double * Bufferrec3_6flux2[NB];
extern double * Bufferrec4_5flux2[NB];
extern double * Bufferrec4_6flux2[NB];
extern double * Bufferrec4_7flux2[NB];
extern double * Bufferrec4_8flux2[NB];
extern double * Bufferrec5_1flux2[NB];
extern double * Bufferrec5_3flux2[NB];
extern double * Bufferrec5_5flux2[NB];
extern double * Bufferrec5_7flux2[NB];
extern double * Bufferrec6_2flux2[NB];
extern double * Bufferrec6_4flux2[NB];
extern double * Bufferrec6_6flux2[NB];
extern double * Bufferrec6_8flux2[NB];

extern double * Buffersend1fine[NB];
extern double * Buffersend2fine[NB];
extern double * Buffersend3fine[NB];
extern double * Buffersend4fine[NB];
extern double * Buffersend5fine[NB];
extern double * Buffersend6fine[NB];
extern double * Bufferrec1fine[NB];
extern double * Bufferrec2fine[NB];
extern double * Bufferrec3fine[NB];
extern double * Bufferrec4fine[NB];
extern double * Bufferrec5fine[NB];
extern double * Bufferrec6fine[NB];
extern double * Bufferrec1_3fine[NB];
extern double * Bufferrec1_4fine[NB];
extern double * Bufferrec1_7fine[NB];
extern double * Bufferrec1_8fine[NB];
extern double * Bufferrec2_1fine[NB];
extern double * Bufferrec2_2fine[NB];
extern double * Bufferrec2_3fine[NB];
extern double * Bufferrec2_4fine[NB];
extern double * Bufferrec3_1fine[NB];
extern double * Bufferrec3_2fine[NB];
extern double * Bufferrec3_5fine[NB];
extern double * Bufferrec3_6fine[NB];
extern double * Bufferrec4_5fine[NB];
extern double * Bufferrec4_6fine[NB];
extern double * Bufferrec4_7fine[NB];
extern double * Bufferrec4_8fine[NB];
extern double * Bufferrec5_1fine[NB];
extern double * Bufferrec5_3fine[NB];
extern double * Bufferrec5_5fine[NB];
extern double * Bufferrec5_7fine[NB];
extern double * Bufferrec6_2fine[NB];
extern double * Bufferrec6_4fine[NB];
extern double * Bufferrec6_6fine[NB];
extern double * Bufferrec6_8fine[NB];

extern double * Buffersend1E[NB];
extern double * Buffersend2E[NB];
extern double * Buffersend3E[NB];
extern double * Buffersend4E[NB];
extern double * Buffersend5E[NB];
extern double * Buffersend6E[NB];
extern double * Bufferrec1E[NB];
extern double * Bufferrec2E[NB];
extern double * Bufferrec3E[NB];
extern double * Bufferrec4E[NB];
extern double * Bufferrec5E[NB];
extern double * Bufferrec6E[NB];
extern double * Bufferrec1E1[NB];
extern double * Bufferrec2E1[NB];
extern double * Bufferrec3E1[NB];
extern double * Bufferrec4E1[NB];
extern double * Bufferrec5E1[NB];
extern double * Bufferrec6E1[NB];
extern double * Bufferrec1_3E[NB];
extern double * Bufferrec1_4E[NB];
extern double * Bufferrec1_7E[NB];
extern double * Bufferrec1_8E[NB];
extern double * Bufferrec2_1E[NB];
extern double * Bufferrec2_2E[NB];
extern double * Bufferrec2_3E[NB];
extern double * Bufferrec2_4E[NB];
extern double * Bufferrec3_1E[NB];
extern double * Bufferrec3_2E[NB];
extern double * Bufferrec3_5E[NB];
extern double * Bufferrec3_6E[NB];
extern double * Bufferrec4_5E[NB];
extern double * Bufferrec4_6E[NB];
extern double * Bufferrec4_7E[NB];
extern double * Bufferrec4_8E[NB];
extern double * Bufferrec5_1E[NB];
extern double * Bufferrec5_3E[NB];
extern double * Bufferrec5_5E[NB];
extern double * Bufferrec5_7E[NB];
extern double * Bufferrec6_2E[NB];
extern double * Bufferrec6_4E[NB];
extern double * Bufferrec6_6E[NB];
extern double * Bufferrec6_8E[NB];

extern double * Bufferrec1_3E1[NB];
extern double * Bufferrec1_4E1[NB];
extern double * Bufferrec1_7E1[NB];
extern double * Bufferrec1_8E1[NB];
extern double * Bufferrec2_1E1[NB];
extern double * Bufferrec2_2E1[NB];
extern double * Bufferrec2_3E1[NB];
extern double * Bufferrec2_4E1[NB];
extern double * Bufferrec3_1E1[NB];
extern double * Bufferrec3_2E1[NB];
extern double * Bufferrec3_5E1[NB];
extern double * Bufferrec3_6E1[NB];
extern double * Bufferrec4_5E1[NB];
extern double * Bufferrec4_6E1[NB];
extern double * Bufferrec4_7E1[NB];
extern double * Bufferrec4_8E1[NB];
extern double * Bufferrec5_1E1[NB];
extern double * Bufferrec5_3E1[NB];
extern double * Bufferrec5_5E1[NB];
extern double * Bufferrec5_7E1[NB];
extern double * Bufferrec6_2E1[NB];
extern double * Bufferrec6_4E1[NB];
extern double * Bufferrec6_6E1[NB];
extern double * Bufferrec6_8E1[NB];

extern double * Bufferrec1_3E2[NB];
extern double * Bufferrec1_4E2[NB];
extern double * Bufferrec1_7E2[NB];
extern double * Bufferrec1_8E2[NB];
extern double * Bufferrec2_1E2[NB];
extern double * Bufferrec2_2E2[NB];
extern double * Bufferrec2_3E2[NB];
extern double * Bufferrec2_4E2[NB];
extern double * Bufferrec3_1E2[NB];
extern double * Bufferrec3_2E2[NB];
extern double * Bufferrec3_5E2[NB];
extern double * Bufferrec3_6E2[NB];
extern double * Bufferrec4_5E2[NB];
extern double * Bufferrec4_6E2[NB];
extern double * Bufferrec4_7E2[NB];
extern double * Bufferrec4_8E2[NB];
extern double * Bufferrec5_1E2[NB];
extern double * Bufferrec5_3E2[NB];
extern double * Bufferrec5_5E2[NB];
extern double * Bufferrec5_7E2[NB];
extern double * Bufferrec6_2E2[NB];
extern double * Bufferrec6_4E2[NB];
extern double * Bufferrec6_6E2[NB];
extern double * Bufferrec6_8E2[NB];

extern double * BuffersendE1corn9[NB];
extern double * BuffersendE1corn10[NB];
extern double * BuffersendE1corn11[NB];
extern double * BuffersendE1corn12[NB];
extern double * BuffersendE2corn5[NB];
extern double * BuffersendE2corn6[NB];
extern double * BuffersendE2corn7[NB];
extern double * BuffersendE2corn8[NB];
extern double * BuffersendE3corn1[NB];
extern double * BuffersendE3corn2[NB];
extern double * BuffersendE3corn3[NB];
extern double * BuffersendE3corn4[NB];
extern double * BufferrecE1corn9[NB];
extern double * BufferrecE1corn10[NB];
extern double * BufferrecE1corn11[NB];
extern double * BufferrecE1corn12[NB];
extern double * BufferrecE2corn5[NB];
extern double * BufferrecE2corn6[NB];
extern double * BufferrecE2corn7[NB];
extern double * BufferrecE2corn8[NB];
extern double * BufferrecE3corn1[NB];
extern double * BufferrecE3corn2[NB];
extern double * BufferrecE3corn3[NB];
extern double * BufferrecE3corn4[NB];

extern double * tempBufferrecE1corn9[NB];
extern double * tempBufferrecE1corn10[NB];
extern double * tempBufferrecE1corn11[NB];
extern double * tempBufferrecE1corn12[NB];
extern double * tempBufferrecE2corn5[NB];
extern double * tempBufferrecE2corn6[NB];
extern double * tempBufferrecE2corn7[NB];
extern double * tempBufferrecE2corn8[NB];
extern double * tempBufferrecE3corn1[NB];
extern double * tempBufferrecE3corn2[NB];
extern double * tempBufferrecE3corn3[NB];
extern double * tempBufferrecE3corn4[NB];

extern double * BufferrecE1corn9_3[NB];
extern double * BufferrecE1corn9_7[NB];
extern double * BufferrecE1corn10_1[NB];
extern double * BufferrecE1corn10_5[NB];
extern double * BufferrecE1corn11_2[NB];
extern double * BufferrecE1corn11_6[NB];
extern double * BufferrecE1corn12_4[NB];
extern double * BufferrecE1corn12_8[NB];
extern double * BufferrecE2corn5_2[NB];
extern double * BufferrecE2corn5_4[NB];
extern double * BufferrecE2corn6_1[NB];
extern double * BufferrecE2corn6_3[NB];
extern double * BufferrecE2corn7_5[NB];
extern double * BufferrecE2corn7_7[NB];
extern double * BufferrecE2corn8_6[NB];
extern double * BufferrecE2corn8_8[NB];
extern double * BufferrecE3corn1_3[NB];
extern double * BufferrecE3corn1_4[NB];
extern double * BufferrecE3corn2_1[NB];
extern double * BufferrecE3corn2_2[NB];
extern double * BufferrecE3corn3_5[NB];
extern double * BufferrecE3corn3_6[NB];
extern double * BufferrecE3corn4_7[NB];
extern double * BufferrecE3corn4_8[NB];

extern double * tempBufferrecE1corn9_3[NB];
extern double * tempBufferrecE1corn9_7[NB];
extern double * tempBufferrecE1corn10_1[NB];
extern double * tempBufferrecE1corn10_5[NB];
extern double * tempBufferrecE1corn11_2[NB];
extern double * tempBufferrecE1corn11_6[NB];
extern double * tempBufferrecE1corn12_4[NB];
extern double * tempBufferrecE1corn12_8[NB];
extern double * tempBufferrecE2corn5_2[NB];
extern double * tempBufferrecE2corn5_4[NB];
extern double * tempBufferrecE2corn6_1[NB];
extern double * tempBufferrecE2corn6_3[NB];
extern double * tempBufferrecE2corn7_5[NB];
extern double * tempBufferrecE2corn7_7[NB];
extern double * tempBufferrecE2corn8_6[NB];
extern double * tempBufferrecE2corn8_8[NB];
extern double * tempBufferrecE3corn1_3[NB];
extern double * tempBufferrecE3corn1_4[NB];
extern double * tempBufferrecE3corn2_1[NB];
extern double * tempBufferrecE3corn2_2[NB];
extern double * tempBufferrecE3corn3_5[NB];
extern double * tempBufferrecE3corn3_6[NB];
extern double * tempBufferrecE3corn4_7[NB];
extern double * tempBufferrecE3corn4_8[NB];

extern double * BufferrecE1corn9_32[NB];
extern double * BufferrecE1corn9_72[NB];
extern double * BufferrecE1corn10_12[NB];
extern double * BufferrecE1corn10_52[NB];
extern double * BufferrecE1corn11_22[NB];
extern double * BufferrecE1corn11_62[NB];
extern double * BufferrecE1corn12_42[NB];
extern double * BufferrecE1corn12_82[NB];
extern double * BufferrecE2corn5_22[NB];
extern double * BufferrecE2corn5_42[NB];
extern double * BufferrecE2corn6_12[NB];
extern double * BufferrecE2corn6_32[NB];
extern double * BufferrecE2corn7_52[NB];
extern double * BufferrecE2corn7_72[NB];
extern double * BufferrecE2corn8_62[NB];
extern double * BufferrecE2corn8_82[NB];
extern double * BufferrecE3corn1_32[NB];
extern double * BufferrecE3corn1_42[NB];
extern double * BufferrecE3corn2_12[NB];
extern double * BufferrecE3corn2_22[NB];
extern double * BufferrecE3corn3_52[NB];
extern double * BufferrecE3corn3_62[NB];
extern double * BufferrecE3corn4_72[NB];
extern double * BufferrecE3corn4_82[NB];

extern double * Bufferboundrec1_MPI[NB];
extern double * Bufferboundrec2_MPI[NB];
extern double * Bufferboundrec3_MPI[NB];
extern double * Bufferboundrec4_MPI[NB];
extern double * Bufferboundrec5_MPI[NB];
extern double * Bufferboundrec6_MPI[NB];
extern double * Bufferboundrec7_MPI[NB];
extern double * Bufferboundrec8_MPI[NB];
extern double * Bufferboundsend1_MPI[NB];
extern double * Bufferboundsend2_MPI[NB];
extern double * Bufferboundsend3_MPI[NB];
extern double * Bufferboundsend4_MPI[NB];
extern double * Bufferboundsend5_MPI[NB];
extern double * Bufferboundsend6_MPI[NB];
extern double * Bufferboundsend7_MPI[NB];
extern double * Bufferboundsend8_MPI[NB];

/*************************************************************************
GLOBAL VARIABLES SECTION
*************************************************************************/
/* physics parameters */
extern double a;
extern double gam;

/* numerical parameters */
extern double Rin, Rout, hslope, R0, fractheta;
extern double cour;
extern double dV, dx[NB][NPR], startx[NPR];
extern double dt, bdt[NB][4];
extern double t, tf;
extern int nstep;
extern double sourceflag, period_max;
extern double rmax;
extern double ndt, ndt1, ndt2, ndt3;
extern int numtasks, rank, local_rank, rc;
extern int prestep_half[NB], prestep_full[NB];
extern int max_levels;
extern int reduce_timestep;
extern int nthreads,numdevices;
extern int gpu;
extern int status;

/* output parameters */
extern double DTd;
extern double DTl;
extern double DTi;
extern int    DTr;
extern double tref;
extern int    dump_cnt;
extern int    image_cnt;
extern int    rdump_cnt;

/* global flags */
extern int failed;
extern int lim;
extern double defcon;

/* set global variables that indicate current local metric, etc. */
extern int icurr, jcurr, pcurr;

struct of_geom {
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

struct of_state {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
};

/*Timing/benchmarking decleration*/
extern clock_t begin1, end1, begin2, end2;
extern double time_spent3;

/*Parallel write*/
extern double *dump_buffer;
extern double(*restrict dxdxp_z[NB])[NDIM][NDIM];
extern double(*restrict dxpdx_z[NB])[NDIM][NDIM];
extern float *array[NB], *array_diag[NB];
extern double *array_rdump[NB], *array_gdump1[NB], *array_gdump2[NB];
extern int first_dump, first_rdump, first_gdump;

/*AMR parameters*/
extern int(*block)[NV];
extern int n_ord[NB],  n_ord_total[NB], n_ord_RM[NB], n_ord_total_RM[NB];
extern int n_active,  n_active_total, n_max;
extern int N1_GPU[NB];
extern int N2_GPU[NB];
extern int N3_GPU[NB];
extern int count_node[20000];
extern int count_gpu[32];
extern int N1_GPU_offset[NB];
extern int N2_GPU_offset[NB];
extern int N3_GPU_offset[NB];

/*************************************************************************
FUNCTION DECLARATIONS
*************************************************************************/
//Output related
void dump_new(void);
void gdump_new(void);
void dump_params(FILE *fp);
double divb_calc(int n, int i, int j, int z);
void dump_params(FILE *fp);
void param_read(FILE *fp);
void rdump_block_read(FILE *fp, int n);
int restart_read_param(void);
void restart_write(void);
int restart_read(void);
void dump_read(void);
void gdump_read(FILE *fp);

/** Evolution/physics functions **/
double advance(int flag);
double advance_GPU(void);
void bound_prim(double(*restrict pr[NB])[NPR], int MPI);
double fluxcalc(double(*restrict pr[NB])[NPR], double(*restrict F[NB])[NPR], int dir, int flag, int n);
void   flux_ct(double(*restrict F1[NB])[NPR], double(*restrict F2[NB])[NPR], double(*restrict F3[NB])[NPR], int n);
void const_transport1(double(*restrict p[NB])[NPR], int n);
void const_transport_bound(void);
void const_transport2(double(*restrict psi[NB])[NDIM], double(*restrict psf[NB])[NDIM], double Dt, int n);
void utoprim(double(*restrict pi[NB])[NPR], double(*restrict pb[NB])[NPR], double(*restrict pf[NB])[NPR], double(*restrict psf[NB])[NDIM], double Dt, int n);
void E_average(void);
double bsq_calc(double * restrict pr, struct of_geom * restrict geom);
int    gamma_calc(double * restrict pr, struct of_geom * restrict geom, double *restrict gamma);
void bcon_calc(double * restrict pr, double * restrict ucon, double * restrict ucov, double * restrict bcon);
void read_E_avg(double E_avg1[NB_1*NB_3][BS_1 + 2 * N1G], double E_avg2[NB_1*NB_3][BS_1 + 2 * N1G], int n);
void write_E_avg(double E_avg1[NB_1*NB_3][BS_1 + 2 * N1G], double E_avg2[NB_1*NB_3][BS_1 + 2 * N1G], int n);
void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
double Drel(int dir, double v, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq);
double NewtonRaphson(double start, int max_count, int dir, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq);
void step_ch(void);
void primtoflux(double * restrict pa, struct of_state * restrict q, int dir, struct of_geom * restrict geom, double * restrict fl);
void primtoU(double * restrict p, struct of_state * restrict q, struct of_geom * restrict geom, double * restrict U);
void inflow_check(double *pr, int n, int ii, int jj, int zz, int type);
void source(double * restrict pa, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict Ua, double Dt);
void u_to_v(double *pr, int i, int j);
void fixup(double((*restrict pv[NB])[NPR]), int n);
void fixup1zone(int i, int j, int z, int n, double prim[NPR]);
void fixup_utoprim(double(*restrict pv[NB])[NPR], int n);
void ucon_calc(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon);
void usrfun(double *pr, int n, double *beta, double **alpha);
void calc_source();
void mhd_calc(double * restrict pr, int dir, struct of_state * restrict q, double * restrict mhd);
void misc_source(double * restrict ph, int ii, int jj, struct of_geom * restrict geom, struct of_state * restrict q, double * restrict dU, double Dt);
void Utoprim(double *Ua, struct of_geom *geom, double *pa);
void get_state(double *pr, struct of_geom *geom, struct of_state *q);
void fix_flux(double(*restrict F1[NB])[NPR], double(*restrict F2[NB])[NPR], double(*restrict F3[NB])[NPR], int n);
int Utoprim_2d(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR]);
int Utoprim_1dvsq2fix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K);
int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K);
void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *cmax, double *cmin, int a, int b, int c);
void step_ch_debug();
void GPU_benchmark(void);
void GPU_init(void);
void set_arrays_GPU(int n, int device);
void GPU_write(int n);
void GPU_finish(int n);
void GPU_hcor(int n);
void GPU_fixup(int flag, int n, double Dt);
void GPU_fixup_post(int n, double Dt);
void GPU_cleanup_post(int n);
void GPU_fixuputoprim(int flag, int n);
void GPU_Utoprim(int flag, int n, double Dt);
void GPU_fluxcalc2D(int dir, int flag, int n);
void GPU_fluxcalcprep(int dir, int flag, int ppm_enable, int n);
void GPU_flux_ct1(int n);
void GPU_flux_ct2(int n);
void GPU_fix_flux(int n);
void GPU_boundprim(int bound_force);
void GPU_boundprim1(int flag, int n);
void GPU_boundprim2(int flag, int n);
void GPU_step_ch();
void GPU_read(int n);
void GPU_consttransport1(int flag, double Dt, int n);
void GPU_consttransport2(int flag, double Dt, int n);
void GPU_consttransport3(int flag, double Dt, int n);
void GPU_consttransport3_post(double Dt, int n);
void GPU_consttransport_bound(void);
void read_time_GPU(void);
double fluxcalc_GPU(int n, int dir);

//Metric/Misc related
double bl_gdet_func(double r, double th);
double gdet_func(double lgcov[][NDIM]);
double mink(int j, int k);
double ranc(int seed);
double slope_lim(double y1, double y2, double y3);
void area_map(int i, int j, int n, double(*restrict prim[NB])[NPR]);
void blgset(int n, int i, int j, struct of_geom *geom);
void bl_coord(double * restrict X, double * restrict r, double * restrict th, double * restrict phi);
void bl_gcon_func(double r, double th, double gcov[][NDIM]);
void kerr_gcov_func(double r, double th, double gcov[][NDIM]);
void bl_gcov_func(double r, double th, double gcov[][NDIM]);
void conn_func(double *X, struct of_geom *geom, double lconn[][NDIM][NDIM]);
void coord(int n, int i, int j, int z, int loc, double *X);
void diag(int call_code);
void diag_flux(double(*F1[NB])[NPR]);
void fail(int fail_type);
void set_Katm(void);
void set_mag(void);
int  get_G_ATM(double *g_tmp);
void gcon_func(double lgcov[][NDIM], double lgcon[][NDIM]);
void gcov_func(double *X, double lgcov[][NDIM]);
void get_geometry(int n, int i, int j, int z, int loc, struct of_geom *geom);
void get_geometry_direct(int ii, int jj, int zz, int ff, struct of_geom *geom);
int index_3D(int n, int i, int j, int z);
int index_2D(int n, int i, int j, int z);
void init(void);
void lower(double * restrict a, struct of_geom * restrict geom, double * restrict b);
void ludcmp(double **a, int n, int *indx, double * d);
void raise(double * restrict v1, struct of_geom * restrict geom, double * restrict v2);
void rescale(double *pr, int which, int dir, int n, int ii, int jj, int zz, int face, struct of_geom *geom);
int invert_matrix(double A[][NDIM], double Ainv[][NDIM]);
int LU_decompose(double A[][NDIM], int permute[]);
void LU_substitution(double A[][NDIM], double B[], int permute[]);

//AMR Related
void MPI_initialize(int argc, char *argv[]);
void activate_blocks(void);
void set_corners(void);
void pre_refine(void);
void refine(int n);
int derefine_pole(void);
void refine_field(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict pb[NB])[NDIM]);
void derefine(int n);
void post_refine(void);
int AMR_coord_linear(int level, int i, int j, int z);
void AMR_coord_cart(int n, int *level, int *i, int *j, int *z);
void test_AMR(void);
void set_AMR(void);
void derefine(int n);
void check_nesting(int n);
double calc_rhomax(int n);
void check_refcrit(void);
void free_arrays(int n);
void set_timelevel_jet(void);
void set_prestep(void);
void mpi_synch(void);
void set_timelevel(void);
void rm_order(void);
void balance_load(void);
void balance_load_gpu(void);
void set_arrays_image(void);
void set_arrays(int n);
void set_grid(int n);
void set_points(int n);
double calc_mem(int n_blocks);
double B1_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);
double B2_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);
double B3_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);

//Boundary transfer related
void bound_send1(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double * Bufferp[NB], double * Bufferps[NB], int n);
void bound_rec1(double(*restrict prim[NB])[NPR], double * Bufferp[NB], int bound_force, int n);
void bound_send2(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double * Bufferp[NB], double * Bufferps[NB], int n);
void bound_rec2(double(*restrict prim[NB])[NPR], double * Bufferp[NB], int bound_force, int n);
void bound_send3(double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double * Bufferp[NB], double * Bufferps[NB], int n);
void bound_rec3(double(*restrict prim[NB])[NPR], double * Bufferp[NB], int bound_force, int n);
void pack_send1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double(*restrict ps[NB])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void unpack_receive1(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive2(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int reverse);
void unpack_receive3(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB], double *tempreceive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1receive[NB], double *temp2receive[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);

void flux_send1(double(*restrict F1[NB])[NPR], double * Bufferp[NB], int n);
void flux_rec1(double(*restrict F1[NB])[NPR], double * Bufferp[NB], int n, int calc_corr);
void flux_send2(double(*restrict F2[NB])[NPR], double * Bufferp[NB], int n);
void flux_rec2(double(*restrict F2[NB])[NPR], double * Bufferp[NB], int n, int calc_corr);
void flux_send3(double(*restrict F3[NB])[NPR], double * Bufferp[NB], int n);
void flux_rec3(double(*restrict F3[NB])[NPR], double * Bufferp[NB], int n, int calc_corr);
void pack_send1_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send2_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send3_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB], double(*restrict prim[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_flux_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_flux_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_flux_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void unpack_receive1_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);
void unpack_receive2_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);
void unpack_receive3_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);

void E_send1(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E_send2(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E_send3(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E_rec1(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void E_rec2(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void E_rec3(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void E1_send_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E2_send_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E3_send_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n);
void E1_receive_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void E2_receive_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void E3_receive_corn(double(*restrict E[NB])[NDIM], double * Bufferp[NB], int n, int calc_corr);
void pack_send1_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send2_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send3_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void unpack_receive1_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void unpack_receive2_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void unpack_receive3_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double *temp1[NB], double *temp2[NB],
	double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void pack_send_E1_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E2_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E3_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E1_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E2_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E3_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB], double(*restrict E[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void unpack_receive_E1_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);
void unpack_receive_E2_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);
void unpack_receive_E3_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB], double *temp1[NB], double *temp2[NB], double(*restrict prim[NB])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);

void B_rec1(double(*restrict F1[NB])[NDIM], double * Bufferp[NB], int n);
void B_rec2(double(*restrict F2[NB])[NDIM], double * Bufferp[NB], int n);
void B_rec3(double(*restrict F3[NB])[NDIM], double * Bufferp[NB], int n);
void B_send1(double(*restrict F1[NB])[NDIM], double * Bufferp[NB], int n);
void B_send2(double(*restrict F2[NB])[NDIM], double * Bufferp[NB], int n);
void B_send3(double(*restrict F3[NB])[NDIM], double * Bufferp[NB], int n);
void Bp_send1(double(*restrict F1[NB])[NDIM], int n);
void Bp_send2(double(*restrict F1[NB])[NDIM], int n);
void Bp_send3(double(*restrict F1[NB])[NDIM], int n);
void Bp_rec1(int n);
void Bp_rec2(int n);
void Bp_rec3(int n);
void pack_send_B1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict prim[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B_average1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B_average2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B_average3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB], double(*restrict F1[NB])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void unpack_receive_B1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent);
void unpack_receive_B2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent, int neg);
void unpack_receive_B3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB], double(*restrict prim[NB])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent);

//Cylindrification related
double Ftr(double x);
double Ftrgenlin(double x, double xa, double xb, double ya, double yb);
double Ftrgen(double x, double xa, double xb, double ya, double yb);
double Fangle(double x);
double limlin(double x, double x0, double dx, double y0);
double minlin(double x, double x0, double dx, double y0);
double mins(double f1, double f2, double df);
double maxs(double f1, double f2, double df);
double minmaxs(double f1, double f2, double df, double dir);
static double sinth0(double *X0, double *X, void(*vofx)(double*, double*));
static double sinth1in(double *X0, double *X, void(*vofx)(double*, double*));
static double th2in(double *X0, double *X, void(*vofx)(double*, double*));
static void to1stquadrant(double *Xin, double *Xout, int *ismirrored);
static double func1(double *X0, double *X, void(*vofx)(double*, double*));
static double func2(double *X0, double *X, void(*vofx)(double*, double*));
void vofx_cylindrified(double *Xin, void(*vofx)(double*, double*), double *Vout);
void vofx_matthewcoords(double *X, double *V);
void dxdxp_func(double *X, double dxdxp[][NDIM]);

//Rotation/ellipticity related
void sph_to_cart(double X[NDIM], double *r, double *th, double *phi);
void rotate_coord(double X[NDIM], double tilt);
void cart_to_sph(double X[NDIM], double *r, double *th, double *phi);
void rotate_vector(double V[NDIM], double pos[NDIM], double *r, double *th, double *phi, double tilt);
void elliptical_coord(double X_cart[NDIM], double pos_new[NDIM], double *r, double eccentricity);
void elliptical_vector(double X_cart[NDIM], double V_old[NDIM], double V_new[NDIM], double pos_new[NDIM], double *r, double *th, double eccentricity);

//HLLC related
void set_Mud(int n);
double fluxcalc_hllc(double(*restrict pr[NB])[NPR], double(*restrict F[NB])[NPR], int dir, int flag, int n);
void ctop_to_utop(double ctop[NDIM], double cmax[NDIM]);
void primtoflux_FT(double * restrict pr, struct of_state * restrict q, int dir, struct of_geom * restrict geom, double restrict flux[NPR]);
void vchar_FT(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js, double  restrict *vmax, double restrict *vmin, int n, int a, int b, int c);





