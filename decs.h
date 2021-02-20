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
#ifdef __unix__
#include <sys/time.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __APPLE__
#include <omp.h>
#endif
#include "config.h"

/*************************************************************************
GLOBAL ARRAY SECTION
*************************************************************************/
/* for debug */
extern int(*restrict failimage[NB_LOCAL])[NFAIL];
#if(DO_FONT_FIX)
extern double *Katm[NB_LOCAL];
#endif

/*CPU arrays*/
extern double(*restrict V[NB_LOCAL])[6];
extern double(*restrict p[NB_LOCAL])[NPR];
extern double (*E_avg1[N_LEVELS_3D])[BS_1 + 2 * N1G];
extern double (*E_avg2[N_LEVELS_3D])[BS_1 + 2 * N1G];
extern double (*E_avg1_new[N_LEVELS_3D])[BS_1 + 2 * N1G];
extern double (*E_avg2_new[N_LEVELS_3D])[BS_1 + 2 * N1G];
extern double(*restrict  ph[NB_LOCAL])[NPR];
extern double(*restrict E_corn[NB_LOCAL])[NDIM];
extern double(*restrict dE[NB_LOCAL])[2][NDIM][NDIM];
extern double(*restrict ps[NB_LOCAL])[NDIM];
extern double(*restrict psh[NB_LOCAL])[NDIM];
extern double(*restrict U_n[NB_LOCAL])[NPR];
extern double(*restrict U[NB_LOCAL])[NPR]; 
extern double(*restrict dq[NB_LOCAL])[NPR];
extern double(*restrict dU_MHD1[NB_LOCAL])[NPR];
extern double(*restrict dU_RAD0[NB_LOCAL])[NPR];
extern double(*restrict dU_RAD1[NB_LOCAL])[NPR];
extern double(*restrict F1[NB_LOCAL])[NPR];
extern double(*restrict F2[NB_LOCAL])[NPR];
extern double(*restrict F3[NB_LOCAL])[NPR];
extern int(*restrict pflag[NB_LOCAL]);
extern int(*restrict pflag_rad[NB_LOCAL]);
extern double(*restrict conn[NB_LOCAL])[NDIM][NDIM][NDIM];
extern double(*restrict gcon[NB_LOCAL])[NPG][NDIM][NDIM];
extern double(*restrict gcov[NB_LOCAL])[NPG][NDIM][NDIM];
extern double(*restrict gdet[NB_LOCAL])[NPG];
extern double(*restrict Mud[NB])[NDIM][NDIM][NDIM];
extern double(*restrict Mud_inv[NB])[NDIM][NDIM][NDIM];
extern double(*restrict dU_s[NB_LOCAL])[NPR];

#if (DOHELM)
/*EOS CPU arrays*/
extern double eos_f[EOSIMAX*EOSJMAX];
extern double eos_fd[EOSIMAX*EOSJMAX];
extern double eos_ft[EOSIMAX*EOSJMAX];
extern double eos_fdd[EOSIMAX*EOSJMAX];
extern double eos_ftt[EOSIMAX*EOSJMAX];
extern double eos_fdt[EOSIMAX*EOSJMAX];
extern double eos_fddt[EOSIMAX*EOSJMAX];
extern double eos_fdtt[EOSIMAX*EOSJMAX];
extern double eos_fddtt[EOSIMAX*EOSJMAX];
extern double eos_dpdf[EOSIMAX*EOSJMAX];
extern double eos_dpdfd[EOSIMAX*EOSJMAX];
extern double eos_dpdft[EOSIMAX*EOSJMAX];
extern double eos_dpdfdt[EOSIMAX*EOSJMAX];
extern double eos_ef[EOSIMAX*EOSJMAX];
extern double eos_efd[EOSIMAX*EOSJMAX];
extern double eos_eft[EOSIMAX*EOSJMAX];
extern double eos_efdt[EOSIMAX*EOSJMAX];
extern double eos_xf[EOSIMAX*EOSJMAX];
extern double eos_xfd[EOSIMAX*EOSJMAX];
extern double eos_xft[EOSIMAX*EOSJMAX];
extern double eos_xfdt[EOSIMAX*EOSJMAX];
extern double eos_t[EOSJMAX];
extern double eos_d[EOSIMAX];
extern double eos_dd[EOSIMAX];
extern double eos_dt[EOSJMAX];
#endif

/*GPU transfer arrays*/
extern double *F1_1[NB_LOCAL];
extern double *F2_1[NB_LOCAL];
extern double *F3_1[NB_LOCAL];
extern double *dq_1[NB_LOCAL];
extern double *dU_GPU[NB_LOCAL];
extern double *p_1[NB_LOCAL];
extern double *ph_1[NB_LOCAL];
extern double *ps_1[NB_LOCAL];
extern double *psh_1[NB_LOCAL];
extern double *gcov_GPU[NB_LOCAL];
extern double *gcon_GPU[NB_LOCAL];
extern double *Mud_GPU[NB_LOCAL];
extern double *Mud_inv_GPU[NB_LOCAL];
extern double *conn_GPU[NB_LOCAL];
extern double *gdet_GPU[NB_LOCAL];
extern double *dtij1_GPU[NB_LOCAL];
extern double *dtij2_GPU[NB_LOCAL];
extern double *dtij3_GPU[NB_LOCAL];
extern double *Katm_GPU[NB_LOCAL];
extern int *pflag_GPU[NB_LOCAL];
extern int *failimage_GPU[NB_LOCAL];
extern int failimage_counter[NFAIL];

/*MPI arrays*/
extern double  *send1[NB_LOCAL], *send2[NB_LOCAL], *send3[NB_LOCAL], *send4[NB_LOCAL], *send5[NB_LOCAL], *send6[NB_LOCAL];
extern double  *send1_fine[NB_LOCAL], *send2_fine[NB_LOCAL], *send3_fine[NB_LOCAL], *send4_fine[NB_LOCAL], *send5_fine[NB_LOCAL], *send6_fine[NB_LOCAL];

extern double  *send1_flux[NB_LOCAL], *send2_flux[NB_LOCAL], *send3_flux[NB_LOCAL], *send4_flux[NB_LOCAL], *send5_flux[NB_LOCAL], *send6_flux[NB_LOCAL], *send7_flux[NB_LOCAL], *send8_flux[NB_LOCAL];
extern double  *send1_E[NB_LOCAL], *send2_E[NB_LOCAL], *send3_E[NB_LOCAL], *send4_E[NB_LOCAL], *send5_E[NB_LOCAL], *send6_E[NB_LOCAL], *send7_E[NB_LOCAL], *send8_E[NB_LOCAL];

extern double  *send1_3[NB_LOCAL], *send1_4[NB_LOCAL], *send1_7[NB_LOCAL], *send1_8[NB_LOCAL];
extern double  *receive1_3[NB_LOCAL], *receive1_4[NB_LOCAL], *receive1_7[NB_LOCAL], *receive1_8[NB_LOCAL];
extern double  *tempreceive1_3[NB_LOCAL], *tempreceive1_4[NB_LOCAL], *tempreceive1_7[NB_LOCAL], *tempreceive1_8[NB_LOCAL];

extern double  *send2_1[NB_LOCAL], *send2_2[NB_LOCAL], *send2_3[NB_LOCAL], *send2_4[NB_LOCAL];
extern double  *receive2_1[NB_LOCAL], *receive2_2[NB_LOCAL], *receive2_3[NB_LOCAL], *receive2_4[NB_LOCAL];
extern double  *tempreceive2_1[NB_LOCAL], *tempreceive2_2[NB_LOCAL], *tempreceive2_3[NB_LOCAL], *tempreceive2_4[NB_LOCAL];

extern double  *send3_1[NB_LOCAL], *send3_2[NB_LOCAL], *send3_5[NB_LOCAL], *send3_6[NB_LOCAL];
extern double  *receive3_1[NB_LOCAL], *receive3_2[NB_LOCAL], *receive3_5[NB_LOCAL], *receive3_6[NB_LOCAL];
extern double  *tempreceive3_1[NB_LOCAL], *tempreceive3_2[NB_LOCAL], *tempreceive3_5[NB_LOCAL], *tempreceive3_6[NB_LOCAL];

extern double  *send4_5[NB_LOCAL], *send4_6[NB_LOCAL], *send4_7[NB_LOCAL], *send4_8[NB_LOCAL];
extern double  *receive4_5[NB_LOCAL], *receive4_6[NB_LOCAL], *receive4_7[NB_LOCAL], *receive4_8[NB_LOCAL];
extern double  *tempreceive4_5[NB_LOCAL], *tempreceive4_6[NB_LOCAL], *tempreceive4_7[NB_LOCAL], *tempreceive4_8[NB_LOCAL];

extern double  *send5_1[NB_LOCAL], *send5_3[NB_LOCAL], *send5_5[NB_LOCAL], *send5_7[NB_LOCAL];
extern double  *receive5_1[NB_LOCAL], *receive5_3[NB_LOCAL], *receive5_5[NB_LOCAL], *receive5_7[NB_LOCAL];
extern double  *tempreceive5_1[NB_LOCAL], *tempreceive5_3[NB_LOCAL], *tempreceive5_5[NB_LOCAL], *tempreceive5_7[NB_LOCAL];

extern double  *send6_2[NB_LOCAL], *send6_4[NB_LOCAL], *send6_6[NB_LOCAL], *send6_8[NB_LOCAL];
extern double  *receive6_2[NB_LOCAL], *receive6_4[NB_LOCAL], *receive6_6[NB_LOCAL], *receive6_8[NB_LOCAL];
extern double  *tempreceive6_2[NB_LOCAL], *tempreceive6_4[NB_LOCAL], *tempreceive6_6[NB_LOCAL], *tempreceive6_8[NB_LOCAL];

extern double *receive1[NB_LOCAL], *receive2[NB_LOCAL], *receive3[NB_LOCAL], *receive4[NB_LOCAL], *receive5[NB_LOCAL], *receive6[NB_LOCAL];
extern double *tempreceive1[NB_LOCAL], *tempreceive2[NB_LOCAL], *tempreceive3[NB_LOCAL], *tempreceive4[NB_LOCAL], *tempreceive5[NB_LOCAL], *tempreceive6[NB_LOCAL];

extern double *receive1_fine[NB_LOCAL], *receive2_fine[NB_LOCAL], *receive3_fine[NB_LOCAL], *receive4_fine[NB_LOCAL], *receive5_fine[NB_LOCAL], *receive6_fine[NB_LOCAL];
extern double  *receive1_3fine[NB_LOCAL], *receive1_4fine[NB_LOCAL], *receive1_7fine[NB_LOCAL], *receive1_8fine[NB_LOCAL];
extern double  *receive2_1fine[NB_LOCAL], *receive2_2fine[NB_LOCAL], *receive2_3fine[NB_LOCAL], *receive2_4fine[NB_LOCAL];
extern double  *receive3_1fine[NB_LOCAL], *receive3_2fine[NB_LOCAL], *receive3_5fine[NB_LOCAL], *receive3_6fine[NB_LOCAL];
extern double  *receive4_5fine[NB_LOCAL], *receive4_6fine[NB_LOCAL], *receive4_7fine[NB_LOCAL], *receive4_8fine[NB_LOCAL];
extern double  *receive5_1fine[NB_LOCAL], *receive5_3fine[NB_LOCAL], *receive5_5fine[NB_LOCAL], *receive5_7fine[NB_LOCAL];
extern double  *receive6_2fine[NB_LOCAL], *receive6_4fine[NB_LOCAL], *receive6_6fine[NB_LOCAL], *receive6_8fine[NB_LOCAL];
extern double *receive1_flux[NB_LOCAL], *receive2_flux[NB_LOCAL], *receive3_flux[NB_LOCAL], *receive4_flux[NB_LOCAL], *receive5_flux[NB_LOCAL], *receive6_flux[NB_LOCAL], *receive7_flux[NB_LOCAL], *receive8_flux[NB_LOCAL];
extern double *receive1_flux1[NB_LOCAL], *receive2_flux1[NB_LOCAL], *receive3_flux1[NB_LOCAL], *receive4_flux1[NB_LOCAL], *receive5_flux1[NB_LOCAL], *receive6_flux1[NB_LOCAL], *receive7_flux1[NB_LOCAL], *receive8_flux1[NB_LOCAL];

extern double  *receive1_3flux[NB_LOCAL], *receive1_4flux[NB_LOCAL], *receive1_7flux[NB_LOCAL], *receive1_8flux[NB_LOCAL];
extern double  *receive2_1flux[NB_LOCAL], *receive2_2flux[NB_LOCAL], *receive2_3flux[NB_LOCAL], *receive2_4flux[NB_LOCAL];
extern double  *receive3_1flux[NB_LOCAL], *receive3_2flux[NB_LOCAL], *receive3_5flux[NB_LOCAL], *receive3_6flux[NB_LOCAL];
extern double  *receive4_5flux[NB_LOCAL], *receive4_6flux[NB_LOCAL], *receive4_7flux[NB_LOCAL], *receive4_8flux[NB_LOCAL];
extern double  *receive5_1flux[NB_LOCAL], *receive5_3flux[NB_LOCAL], *receive5_5flux[NB_LOCAL], *receive5_7flux[NB_LOCAL];
extern double  *receive6_2flux[NB_LOCAL], *receive6_4flux[NB_LOCAL], *receive6_6flux[NB_LOCAL], *receive6_8flux[NB_LOCAL];

extern double  *receive1_3flux1[NB_LOCAL], *receive1_4flux1[NB_LOCAL], *receive1_7flux1[NB_LOCAL], *receive1_8flux1[NB_LOCAL];
extern double  *receive2_1flux1[NB_LOCAL], *receive2_2flux1[NB_LOCAL], *receive2_3flux1[NB_LOCAL], *receive2_4flux1[NB_LOCAL];
extern double  *receive3_1flux1[NB_LOCAL], *receive3_2flux1[NB_LOCAL], *receive3_5flux1[NB_LOCAL], *receive3_6flux1[NB_LOCAL];
extern double  *receive4_5flux1[NB_LOCAL], *receive4_6flux1[NB_LOCAL], *receive4_7flux1[NB_LOCAL], *receive4_8flux1[NB_LOCAL];
extern double  *receive5_1flux1[NB_LOCAL], *receive5_3flux1[NB_LOCAL], *receive5_5flux1[NB_LOCAL], *receive5_7flux1[NB_LOCAL];
extern double  *receive6_2flux1[NB_LOCAL], *receive6_4flux1[NB_LOCAL], *receive6_6flux1[NB_LOCAL], *receive6_8flux1[NB_LOCAL];

extern double  *receive1_3flux2[NB_LOCAL], *receive1_4flux2[NB_LOCAL], *receive1_7flux2[NB_LOCAL], *receive1_8flux2[NB_LOCAL];
extern double  *receive2_1flux2[NB_LOCAL], *receive2_2flux2[NB_LOCAL], *receive2_3flux2[NB_LOCAL], *receive2_4flux2[NB_LOCAL];
extern double  *receive3_1flux2[NB_LOCAL], *receive3_2flux2[NB_LOCAL], *receive3_5flux2[NB_LOCAL], *receive3_6flux2[NB_LOCAL];
extern double  *receive4_5flux2[NB_LOCAL], *receive4_6flux2[NB_LOCAL], *receive4_7flux2[NB_LOCAL], *receive4_8flux2[NB_LOCAL];
extern double  *receive5_1flux2[NB_LOCAL], *receive5_3flux2[NB_LOCAL], *receive5_5flux2[NB_LOCAL], *receive5_7flux2[NB_LOCAL];
extern double  *receive6_2flux2[NB_LOCAL], *receive6_4flux2[NB_LOCAL], *receive6_6flux2[NB_LOCAL], *receive6_8flux2[NB_LOCAL];

extern double *receive1_E[NB_LOCAL], *receive2_E[NB_LOCAL], *receive3_E[NB_LOCAL], *receive4_E[NB_LOCAL], *receive5_E[NB_LOCAL], *receive6_E[NB_LOCAL], *receive7_E[NB_LOCAL], *receive8_E[NB_LOCAL];
extern double *receive1_E1[NB_LOCAL], *receive2_E1[NB_LOCAL], *receive3_E1[NB_LOCAL], *receive4_E1[NB_LOCAL], *receive5_E1[NB_LOCAL], *receive6_E1[NB_LOCAL], *receive7_E1[NB_LOCAL], *receive8_E1[NB_LOCAL];

extern double  *receive1_3E[NB_LOCAL], *receive1_4E[NB_LOCAL], *receive1_7E[NB_LOCAL], *receive1_8E[NB_LOCAL];
extern double  *receive2_1E[NB_LOCAL], *receive2_2E[NB_LOCAL], *receive2_3E[NB_LOCAL], *receive2_4E[NB_LOCAL];
extern double  *receive3_1E[NB_LOCAL], *receive3_2E[NB_LOCAL], *receive3_5E[NB_LOCAL], *receive3_6E[NB_LOCAL];
extern double  *receive4_5E[NB_LOCAL], *receive4_6E[NB_LOCAL], *receive4_7E[NB_LOCAL], *receive4_8E[NB_LOCAL];
extern double  *receive5_1E[NB_LOCAL], *receive5_3E[NB_LOCAL], *receive5_5E[NB_LOCAL], *receive5_7E[NB_LOCAL];
extern double  *receive6_2E[NB_LOCAL], *receive6_4E[NB_LOCAL], *receive6_6E[NB_LOCAL], *receive6_8E[NB_LOCAL];

extern double  *receive1_3E2[NB_LOCAL], *receive1_4E2[NB_LOCAL], *receive1_7E2[NB_LOCAL], *receive1_8E2[NB_LOCAL];
extern double  *receive2_1E2[NB_LOCAL], *receive2_2E2[NB_LOCAL], *receive2_3E2[NB_LOCAL], *receive2_4E2[NB_LOCAL];
extern double  *receive3_1E2[NB_LOCAL], *receive3_2E2[NB_LOCAL], *receive3_5E2[NB_LOCAL], *receive3_6E2[NB_LOCAL];
extern double  *receive4_5E2[NB_LOCAL], *receive4_6E2[NB_LOCAL], *receive4_7E2[NB_LOCAL], *receive4_8E2[NB_LOCAL];
extern double  *receive5_1E2[NB_LOCAL], *receive5_3E2[NB_LOCAL], *receive5_5E2[NB_LOCAL], *receive5_7E2[NB_LOCAL];
extern double  *receive6_2E2[NB_LOCAL], *receive6_4E2[NB_LOCAL], *receive6_6E2[NB_LOCAL], *receive6_8E2[NB_LOCAL];

extern double  *receive1_3E1[NB_LOCAL], *receive1_4E1[NB_LOCAL], *receive1_7E1[NB_LOCAL], *receive1_8E1[NB_LOCAL];
extern double  *receive2_1E1[NB_LOCAL], *receive2_2E1[NB_LOCAL], *receive2_3E1[NB_LOCAL], *receive2_4E1[NB_LOCAL];
extern double  *receive3_1E1[NB_LOCAL], *receive3_2E1[NB_LOCAL], *receive3_5E1[NB_LOCAL], *receive3_6E1[NB_LOCAL];
extern double  *receive4_5E1[NB_LOCAL], *receive4_6E1[NB_LOCAL], *receive4_7E1[NB_LOCAL], *receive4_8E1[NB_LOCAL];
extern double  *receive5_1E1[NB_LOCAL], *receive5_3E1[NB_LOCAL], *receive5_5E1[NB_LOCAL], *receive5_7E1[NB_LOCAL];
extern double  *receive6_2E1[NB_LOCAL], *receive6_4E1[NB_LOCAL], *receive6_6E1[NB_LOCAL], *receive6_8E1[NB_LOCAL];

extern double  *cornsend1[NB_LOCAL], *cornsend2[NB_LOCAL], *cornsend3[NB_LOCAL], *cornsend4[NB_LOCAL], *cornsend5[NB_LOCAL], *cornsend7[NB_LOCAL], *cornsend7[NB_LOCAL], *cornsend8[NB_LOCAL];
extern double *cornreceive1[NB_LOCAL], *cornreceive2[NB_LOCAL], *cornreceive3[NB_LOCAL], *cornreceive4[NB_LOCAL], *cornreceive5[NB_LOCAL], *cornreceive6[NB_LOCAL], *cornreceive7[NB_LOCAL], *cornreceive8[NB_LOCAL];

extern double *send_E3_corn1[NB_LOCAL], *send_E3_corn2[NB_LOCAL], *send_E3_corn3[NB_LOCAL], *send_E3_corn4[NB_LOCAL], *send_E2_corn5[NB_LOCAL], *send_E2_corn6[NB_LOCAL],
*send_E2_corn7[NB_LOCAL], *send_E2_corn8[NB_LOCAL], *send_E1_corn9[NB_LOCAL], *send_E1_corn10[NB_LOCAL], *send_E1_corn11[NB_LOCAL], *send_E1_corn12[NB_LOCAL];
extern double *receive_E3_corn1_1[NB_LOCAL], *receive_E3_corn2_1[NB_LOCAL], *receive_E3_corn3_1[NB_LOCAL], *receive_E3_corn4_1[NB_LOCAL], *receive_E2_corn5_1[NB_LOCAL], *receive_E2_corn6_1[NB_LOCAL],
*receive_E2_corn7_1[NB_LOCAL], *receive_E2_corn8_1[NB_LOCAL], *receive_E1_corn9_1[NB_LOCAL], *receive_E1_corn10_1[NB_LOCAL], *receive_E1_corn11_1[NB_LOCAL], *receive_E1_corn12_1[NB_LOCAL];
extern double *receive_E3_corn1_2[NB_LOCAL], *receive_E3_corn2_2[NB_LOCAL], *receive_E3_corn3_2[NB_LOCAL], *receive_E3_corn4_2[NB_LOCAL], *receive_E2_corn5_2[NB_LOCAL], *receive_E2_corn6_2[NB_LOCAL],
*receive_E2_corn7_2[NB_LOCAL], *receive_E2_corn8_2[NB_LOCAL], *receive_E1_corn9_2[NB_LOCAL], *receive_E1_corn10_2[NB_LOCAL], *receive_E1_corn11_2[NB_LOCAL], *receive_E1_corn12_2[NB_LOCAL];

extern double *tempreceive_E3_corn1_1[NB_LOCAL], *tempreceive_E3_corn2_1[NB_LOCAL], *tempreceive_E3_corn3_1[NB_LOCAL], *tempreceive_E3_corn4_1[NB_LOCAL], *tempreceive_E2_corn5_1[NB_LOCAL], *tempreceive_E2_corn6_1[NB_LOCAL],
*tempreceive_E2_corn7_1[NB_LOCAL], *tempreceive_E2_corn8_1[NB_LOCAL], *tempreceive_E1_corn9_1[NB_LOCAL], *tempreceive_E1_corn10_1[NB_LOCAL], *tempreceive_E1_corn11_1[NB_LOCAL], *tempreceive_E1_corn12_1[NB_LOCAL];
extern double *tempreceive_E3_corn1_2[NB_LOCAL], *tempreceive_E3_corn2_2[NB_LOCAL], *tempreceive_E3_corn3_2[NB_LOCAL], *tempreceive_E3_corn4_2[NB_LOCAL], *tempreceive_E2_corn5_2[NB_LOCAL], *tempreceive_E2_corn6_2[NB_LOCAL],
*tempreceive_E2_corn7_2[NB_LOCAL], *tempreceive_E2_corn8_2[NB_LOCAL], *tempreceive_E1_corn9_2[NB_LOCAL], *tempreceive_E1_corn10_2[NB_LOCAL], *tempreceive_E1_corn11_2[NB_LOCAL], *tempreceive_E1_corn12_2[NB_LOCAL];

extern double *receive_E3_corn1_12[NB_LOCAL], *receive_E3_corn2_12[NB_LOCAL], *receive_E3_corn3_12[NB_LOCAL], *receive_E3_corn4_12[NB_LOCAL], *receive_E2_corn5_12[NB_LOCAL], *receive_E2_corn6_12[NB_LOCAL],
*receive_E2_corn7_12[NB_LOCAL], *receive_E2_corn8_12[NB_LOCAL], *receive_E1_corn9_12[NB_LOCAL], *receive_E1_corn10_12[NB_LOCAL], *receive_E1_corn11_12[NB_LOCAL], *receive_E1_corn12_12[NB_LOCAL];
extern double *receive_E3_corn1_22[NB_LOCAL], *receive_E3_corn2_22[NB_LOCAL], *receive_E3_corn3_22[NB_LOCAL], *receive_E3_corn4_22[NB_LOCAL], *receive_E2_corn5_22[NB_LOCAL], *receive_E2_corn6_22[NB_LOCAL],
*receive_E2_corn7_22[NB_LOCAL], *receive_E2_corn8_22[NB_LOCAL], *receive_E1_corn9_22[NB_LOCAL], *receive_E1_corn10_22[NB_LOCAL], *receive_E1_corn11_22[NB_LOCAL], *receive_E1_corn12_22[NB_LOCAL];

extern double *receive_E3_corn1[NB_LOCAL], *receive_E3_corn2[NB_LOCAL], *receive_E3_corn3[NB_LOCAL], *receive_E3_corn4[NB_LOCAL];
extern double *receive_E2_corn5[NB_LOCAL], *receive_E2_corn6[NB_LOCAL], *receive_E2_corn7[NB_LOCAL], *receive_E2_corn8[NB_LOCAL];
extern double *receive_E1_corn9[NB_LOCAL], *receive_E1_corn10[NB_LOCAL], *receive_E1_corn11[NB_LOCAL], *receive_E1_corn12[NB_LOCAL];

extern double *tempreceive_E3_corn1[NB_LOCAL], *tempreceive_E3_corn2[NB_LOCAL], *tempreceive_E3_corn3[NB_LOCAL], *tempreceive_E3_corn4[NB_LOCAL];
extern double *tempreceive_E2_corn5[NB_LOCAL], *tempreceive_E2_corn6[NB_LOCAL], *tempreceive_E2_corn7[NB_LOCAL], *tempreceive_E2_corn8[NB_LOCAL];
extern double *tempreceive_E1_corn9[NB_LOCAL], *tempreceive_E1_corn10[NB_LOCAL], *tempreceive_E1_corn11[NB_LOCAL], *tempreceive_E1_corn12[NB_LOCAL];

/*CUDA arrays decleration*/
extern double *NULL_POINTER[NB_LOCAL];
extern cudaStream_t commandQueue[NB_LOCAL];
extern cudaStream_t commandQueueGPU[NB_LOCAL];
extern cudaEvent_t boundevent[NB_LOCAL][600];
extern cudaEvent_t boundevent1[NB_LOCAL][100];
extern cudaEvent_t boundevent2[NB_LOCAL][100];
extern int fix_mem[NB_LOCAL];
extern int fix_mem2[NB_LOCAL];
extern int nr_workgroups[NB_LOCAL];
extern int nr_workgroups1[NB_LOCAL];
extern int nr_workgroups2[NB_LOCAL];
extern int nr_workgroups2_1[NB_LOCAL];
extern int nr_workgroups2_2[NB_LOCAL];
extern int nr_workgroups2_3[NB_LOCAL];
extern int nr_workgroups3[NB_LOCAL];
extern int nr_workgroups_special[NB_LOCAL];
extern int nr_workgroups_special1[NB_LOCAL];
extern int nr_workgroups_special2[NB_LOCAL];
extern int nr_workgroups_special3[NB_LOCAL];
extern int global_work_size[NB_LOCAL][1];
extern int global_work_size1[NB_LOCAL][1];
extern int global_work_size2[NB_LOCAL][1];
extern int global_work_size2_1[NB_LOCAL][1];
extern int global_work_size2_2[NB_LOCAL][1];
extern int global_work_size2_3[NB_LOCAL][1];
extern int global_work_size3[NB_LOCAL][1];
extern int global_work_size_bound[NB_LOCAL][1];
extern int global_work_offset[NB_LOCAL][1];
extern int global_work_size_special[NB_LOCAL][1];
extern int global_work_size_special1[NB_LOCAL][1];
extern int global_work_size_special2[NB_LOCAL][1];
extern int global_work_size_special3[NB_LOCAL][1];
extern int local_work_size[1];
extern double * Bufferconn[NB_LOCAL];
extern double * Buffergcov[NB_LOCAL];
extern double * Buffergcon[NB_LOCAL];
extern double * BufferMud[NB_LOCAL];
extern double * BufferMud_inv[NB_LOCAL];
extern double * Buffergdet[NB_LOCAL];
extern double * BufferF1_1[NB_LOCAL];
extern double * BufferF2_1[NB_LOCAL];
extern double * BufferF3_1[NB_LOCAL];
extern double * Bufferdq_1[NB_LOCAL];
extern double * BufferE_1[NB_LOCAL];
extern double * BufferV[NB_LOCAL];
extern double * BufferdU[NB_LOCAL];
extern double* BufferdU_MHD1[NB_LOCAL];
extern double* BufferdU_RAD0[NB_LOCAL];
extern double* BufferdU_RAD1[NB_LOCAL];
extern double* BufferU_n[NB_LOCAL];
extern double * Bufferradius[NB_LOCAL];
extern double * Bufferstorage1[NB_LOCAL];
extern double * Bufferstorage2[NB_LOCAL];
extern double * Bufferstorage3[NB_LOCAL];
extern double * Bufferstorage4[NB_LOCAL];
extern double * Buffereta_avg[NB_LOCAL];
extern double * Bufferp_1[NB_LOCAL];
extern double * Bufferph_1[NB_LOCAL];
extern double * Bufferps_1[NB_LOCAL];
extern double * Bufferpsh_1[NB_LOCAL];
extern double * Bufferdtij1[NB_LOCAL];
extern double * Bufferdtij2[NB_LOCAL];
extern double * Bufferdtij3[NB_LOCAL];
extern int * Bufferpflag[NB_LOCAL];
extern int * Bufferpflag_RAD[NB_LOCAL];
extern int * Bufferfailimage[NB_LOCAL];
extern double * BufferKatm[NB_LOCAL];
extern double * Buffersend1[NB_LOCAL];
extern double * Buffersend1_3[NB_LOCAL];
extern double * Buffersend1_4[NB_LOCAL];
extern double * Buffersend1_7[NB_LOCAL];
extern double * Buffersend1_8[NB_LOCAL];
extern double * Buffersend2[NB_LOCAL];
extern double * Buffersend2_1[NB_LOCAL];
extern double * Buffersend2_2[NB_LOCAL];
extern double * Buffersend2_3[NB_LOCAL];
extern double * Buffersend2_4[NB_LOCAL];
extern double * Buffersend3[NB_LOCAL];
extern double * Buffersend3_1[NB_LOCAL];
extern double * Buffersend3_2[NB_LOCAL];
extern double * Buffersend3_5[NB_LOCAL];
extern double * Buffersend3_6[NB_LOCAL];
extern double * Buffersend4[NB_LOCAL];
extern double * Buffersend4_5[NB_LOCAL];
extern double * Buffersend4_6[NB_LOCAL];
extern double * Buffersend4_7[NB_LOCAL];
extern double * Buffersend4_8[NB_LOCAL];
extern double * Buffersend5[NB_LOCAL];
extern double * Buffersend5_1[NB_LOCAL];
extern double * Buffersend5_3[NB_LOCAL];
extern double * Buffersend5_5[NB_LOCAL];
extern double * Buffersend5_7[NB_LOCAL];
extern double * Buffersend6[NB_LOCAL];
extern double * Buffersend6_2[NB_LOCAL];
extern double * Buffersend6_4[NB_LOCAL];
extern double * Buffersend6_6[NB_LOCAL];
extern double * Buffersend6_8[NB_LOCAL];
extern double * Bufferrec1[NB_LOCAL];
extern double * Bufferrec1_3[NB_LOCAL];
extern double * Bufferrec1_4[NB_LOCAL];
extern double * Bufferrec1_7[NB_LOCAL];
extern double * Bufferrec1_8[NB_LOCAL];
extern double * Bufferrec2[NB_LOCAL];
extern double * Bufferrec2_1[NB_LOCAL];
extern double * Bufferrec2_2[NB_LOCAL];
extern double * Bufferrec2_3[NB_LOCAL];
extern double * Bufferrec2_4[NB_LOCAL];
extern double * Bufferrec3[NB_LOCAL];
extern double * Bufferrec3_1[NB_LOCAL];
extern double * Bufferrec3_2[NB_LOCAL];
extern double * Bufferrec3_5[NB_LOCAL];
extern double * Bufferrec3_6[NB_LOCAL];
extern double * Bufferrec4[NB_LOCAL];
extern double * Bufferrec4_5[NB_LOCAL];
extern double * Bufferrec4_6[NB_LOCAL];
extern double * Bufferrec4_7[NB_LOCAL];
extern double * Bufferrec4_8[NB_LOCAL];
extern double * Bufferrec5[NB_LOCAL];
extern double * Bufferrec5_1[NB_LOCAL];
extern double * Bufferrec5_3[NB_LOCAL];
extern double * Bufferrec5_5[NB_LOCAL];
extern double * Bufferrec5_7[NB_LOCAL];
extern double * Bufferrec6[NB_LOCAL];
extern double * Bufferrec6_2[NB_LOCAL];
extern double * Bufferrec6_4[NB_LOCAL];
extern double * Bufferrec6_6[NB_LOCAL];
extern double * Bufferrec6_8[NB_LOCAL];

extern double * tempBufferrec1[NB_LOCAL];
extern double * tempBufferrec1_3[NB_LOCAL];
extern double * tempBufferrec1_4[NB_LOCAL];
extern double * tempBufferrec1_7[NB_LOCAL];
extern double * tempBufferrec1_8[NB_LOCAL];
extern double * tempBufferrec2[NB_LOCAL];
extern double * tempBufferrec2_1[NB_LOCAL];
extern double * tempBufferrec2_2[NB_LOCAL];
extern double * tempBufferrec2_3[NB_LOCAL];
extern double * tempBufferrec2_4[NB_LOCAL];
extern double * tempBufferrec3[NB_LOCAL];
extern double * tempBufferrec3_1[NB_LOCAL];
extern double * tempBufferrec3_2[NB_LOCAL];
extern double * tempBufferrec3_5[NB_LOCAL];
extern double * tempBufferrec3_6[NB_LOCAL];
extern double * tempBufferrec4[NB_LOCAL];
extern double * tempBufferrec4_5[NB_LOCAL];
extern double * tempBufferrec4_6[NB_LOCAL];
extern double * tempBufferrec4_7[NB_LOCAL];
extern double * tempBufferrec4_8[NB_LOCAL];
extern double * tempBufferrec5[NB_LOCAL];
extern double * tempBufferrec5_1[NB_LOCAL];
extern double * tempBufferrec5_3[NB_LOCAL];
extern double * tempBufferrec5_5[NB_LOCAL];
extern double * tempBufferrec5_7[NB_LOCAL];
extern double * tempBufferrec6[NB_LOCAL];
extern double * tempBufferrec6_2[NB_LOCAL];
extern double * tempBufferrec6_4[NB_LOCAL];
extern double * tempBufferrec6_6[NB_LOCAL];
extern double * tempBufferrec6_8[NB_LOCAL];

extern double * Buffersend1flux[NB_LOCAL];
extern double * Buffersend2flux[NB_LOCAL];
extern double * Buffersend3flux[NB_LOCAL];
extern double * Buffersend4flux[NB_LOCAL];
extern double * Buffersend5flux[NB_LOCAL];
extern double * Buffersend6flux[NB_LOCAL];
extern double * Bufferrec1flux[NB_LOCAL];
extern double * Bufferrec1_3flux[NB_LOCAL];
extern double * Bufferrec1_4flux[NB_LOCAL];
extern double * Bufferrec1_7flux[NB_LOCAL];
extern double * Bufferrec1_8flux[NB_LOCAL];
extern double * Bufferrec2flux[NB_LOCAL];
extern double * Bufferrec2_1flux[NB_LOCAL];
extern double * Bufferrec2_2flux[NB_LOCAL];
extern double * Bufferrec2_3flux[NB_LOCAL];
extern double * Bufferrec2_4flux[NB_LOCAL];
extern double * Bufferrec3flux[NB_LOCAL];
extern double * Bufferrec3_1flux[NB_LOCAL];
extern double * Bufferrec3_2flux[NB_LOCAL];
extern double * Bufferrec3_5flux[NB_LOCAL];
extern double * Bufferrec3_6flux[NB_LOCAL];
extern double * Bufferrec4flux[NB_LOCAL];
extern double * Bufferrec4_5flux[NB_LOCAL];
extern double * Bufferrec4_6flux[NB_LOCAL];
extern double * Bufferrec4_7flux[NB_LOCAL];
extern double * Bufferrec4_8flux[NB_LOCAL];
extern double * Bufferrec5flux[NB_LOCAL];
extern double * Bufferrec5_1flux[NB_LOCAL];
extern double * Bufferrec5_3flux[NB_LOCAL];
extern double * Bufferrec5_5flux[NB_LOCAL];
extern double * Bufferrec5_7flux[NB_LOCAL];
extern double * Bufferrec6flux[NB_LOCAL];
extern double * Bufferrec6_2flux[NB_LOCAL];
extern double * Bufferrec6_4flux[NB_LOCAL];
extern double * Bufferrec6_6flux[NB_LOCAL];
extern double * Bufferrec6_8flux[NB_LOCAL];

extern double * Bufferrec1flux1[NB_LOCAL];
extern double * Bufferrec2flux1[NB_LOCAL];
extern double * Bufferrec3flux1[NB_LOCAL];
extern double * Bufferrec4flux1[NB_LOCAL];
extern double * Bufferrec5flux1[NB_LOCAL];
extern double * Bufferrec6flux1[NB_LOCAL];


extern double * Bufferrec1_3flux1[NB_LOCAL];
extern double * Bufferrec1_4flux1[NB_LOCAL];
extern double * Bufferrec1_7flux1[NB_LOCAL];
extern double * Bufferrec1_8flux1[NB_LOCAL];
extern double * Bufferrec2_1flux1[NB_LOCAL];
extern double * Bufferrec2_2flux1[NB_LOCAL];
extern double * Bufferrec2_3flux1[NB_LOCAL];
extern double * Bufferrec2_4flux1[NB_LOCAL];
extern double * Bufferrec3_1flux1[NB_LOCAL];
extern double * Bufferrec3_2flux1[NB_LOCAL];
extern double * Bufferrec3_5flux1[NB_LOCAL];
extern double * Bufferrec3_6flux1[NB_LOCAL];
extern double * Bufferrec4_5flux1[NB_LOCAL];
extern double * Bufferrec4_6flux1[NB_LOCAL];
extern double * Bufferrec4_7flux1[NB_LOCAL];
extern double * Bufferrec4_8flux1[NB_LOCAL];
extern double * Bufferrec5_1flux1[NB_LOCAL];
extern double * Bufferrec5_3flux1[NB_LOCAL];
extern double * Bufferrec5_5flux1[NB_LOCAL];
extern double * Bufferrec5_7flux1[NB_LOCAL];
extern double * Bufferrec6_2flux1[NB_LOCAL];
extern double * Bufferrec6_4flux1[NB_LOCAL];
extern double * Bufferrec6_6flux1[NB_LOCAL];
extern double * Bufferrec6_8flux1[NB_LOCAL];

extern double * Bufferrec1_3flux2[NB_LOCAL];
extern double * Bufferrec1_4flux2[NB_LOCAL];
extern double * Bufferrec1_7flux2[NB_LOCAL];
extern double * Bufferrec1_8flux2[NB_LOCAL];
extern double * Bufferrec2_1flux2[NB_LOCAL];
extern double * Bufferrec2_2flux2[NB_LOCAL];
extern double * Bufferrec2_3flux2[NB_LOCAL];
extern double * Bufferrec2_4flux2[NB_LOCAL];
extern double * Bufferrec3_1flux2[NB_LOCAL];
extern double * Bufferrec3_2flux2[NB_LOCAL];
extern double * Bufferrec3_5flux2[NB_LOCAL];
extern double * Bufferrec3_6flux2[NB_LOCAL];
extern double * Bufferrec4_5flux2[NB_LOCAL];
extern double * Bufferrec4_6flux2[NB_LOCAL];
extern double * Bufferrec4_7flux2[NB_LOCAL];
extern double * Bufferrec4_8flux2[NB_LOCAL];
extern double * Bufferrec5_1flux2[NB_LOCAL];
extern double * Bufferrec5_3flux2[NB_LOCAL];
extern double * Bufferrec5_5flux2[NB_LOCAL];
extern double * Bufferrec5_7flux2[NB_LOCAL];
extern double * Bufferrec6_2flux2[NB_LOCAL];
extern double * Bufferrec6_4flux2[NB_LOCAL];
extern double * Bufferrec6_6flux2[NB_LOCAL];
extern double * Bufferrec6_8flux2[NB_LOCAL];

extern double * Buffersend1fine[NB_LOCAL];
extern double * Buffersend2fine[NB_LOCAL];
extern double * Buffersend3fine[NB_LOCAL];
extern double * Buffersend4fine[NB_LOCAL];
extern double * Buffersend5fine[NB_LOCAL];
extern double * Buffersend6fine[NB_LOCAL];
extern double * Bufferrec1fine[NB_LOCAL];
extern double * Bufferrec2fine[NB_LOCAL];
extern double * Bufferrec3fine[NB_LOCAL];
extern double * Bufferrec4fine[NB_LOCAL];
extern double * Bufferrec5fine[NB_LOCAL];
extern double * Bufferrec6fine[NB_LOCAL];
extern double * Bufferrec1_3fine[NB_LOCAL];
extern double * Bufferrec1_4fine[NB_LOCAL];
extern double * Bufferrec1_7fine[NB_LOCAL];
extern double * Bufferrec1_8fine[NB_LOCAL];
extern double * Bufferrec2_1fine[NB_LOCAL];
extern double * Bufferrec2_2fine[NB_LOCAL];
extern double * Bufferrec2_3fine[NB_LOCAL];
extern double * Bufferrec2_4fine[NB_LOCAL];
extern double * Bufferrec3_1fine[NB_LOCAL];
extern double * Bufferrec3_2fine[NB_LOCAL];
extern double * Bufferrec3_5fine[NB_LOCAL];
extern double * Bufferrec3_6fine[NB_LOCAL];
extern double * Bufferrec4_5fine[NB_LOCAL];
extern double * Bufferrec4_6fine[NB_LOCAL];
extern double * Bufferrec4_7fine[NB_LOCAL];
extern double * Bufferrec4_8fine[NB_LOCAL];
extern double * Bufferrec5_1fine[NB_LOCAL];
extern double * Bufferrec5_3fine[NB_LOCAL];
extern double * Bufferrec5_5fine[NB_LOCAL];
extern double * Bufferrec5_7fine[NB_LOCAL];
extern double * Bufferrec6_2fine[NB_LOCAL];
extern double * Bufferrec6_4fine[NB_LOCAL];
extern double * Bufferrec6_6fine[NB_LOCAL];
extern double * Bufferrec6_8fine[NB_LOCAL];

extern double * Buffersend1E[NB_LOCAL];
extern double * Buffersend2E[NB_LOCAL];
extern double * Buffersend3E[NB_LOCAL];
extern double * Buffersend4E[NB_LOCAL];
extern double * Buffersend5E[NB_LOCAL];
extern double * Buffersend6E[NB_LOCAL];
extern double * Bufferrec1E[NB_LOCAL];
extern double * Bufferrec2E[NB_LOCAL];
extern double * Bufferrec3E[NB_LOCAL];
extern double * Bufferrec4E[NB_LOCAL];
extern double * Bufferrec5E[NB_LOCAL];
extern double * Bufferrec6E[NB_LOCAL];
extern double * Bufferrec1E1[NB_LOCAL];
extern double * Bufferrec2E1[NB_LOCAL];
extern double * Bufferrec3E1[NB_LOCAL];
extern double * Bufferrec4E1[NB_LOCAL];
extern double * Bufferrec5E1[NB_LOCAL];
extern double * Bufferrec6E1[NB_LOCAL];
extern double * Bufferrec1_3E[NB_LOCAL];
extern double * Bufferrec1_4E[NB_LOCAL];
extern double * Bufferrec1_7E[NB_LOCAL];
extern double * Bufferrec1_8E[NB_LOCAL];
extern double * Bufferrec2_1E[NB_LOCAL];
extern double * Bufferrec2_2E[NB_LOCAL];
extern double * Bufferrec2_3E[NB_LOCAL];
extern double * Bufferrec2_4E[NB_LOCAL];
extern double * Bufferrec3_1E[NB_LOCAL];
extern double * Bufferrec3_2E[NB_LOCAL];
extern double * Bufferrec3_5E[NB_LOCAL];
extern double * Bufferrec3_6E[NB_LOCAL];
extern double * Bufferrec4_5E[NB_LOCAL];
extern double * Bufferrec4_6E[NB_LOCAL];
extern double * Bufferrec4_7E[NB_LOCAL];
extern double * Bufferrec4_8E[NB_LOCAL];
extern double * Bufferrec5_1E[NB_LOCAL];
extern double * Bufferrec5_3E[NB_LOCAL];
extern double * Bufferrec5_5E[NB_LOCAL];
extern double * Bufferrec5_7E[NB_LOCAL];
extern double * Bufferrec6_2E[NB_LOCAL];
extern double * Bufferrec6_4E[NB_LOCAL];
extern double * Bufferrec6_6E[NB_LOCAL];
extern double * Bufferrec6_8E[NB_LOCAL];

extern double * Bufferrec1_3E1[NB_LOCAL];
extern double * Bufferrec1_4E1[NB_LOCAL];
extern double * Bufferrec1_7E1[NB_LOCAL];
extern double * Bufferrec1_8E1[NB_LOCAL];
extern double * Bufferrec2_1E1[NB_LOCAL];
extern double * Bufferrec2_2E1[NB_LOCAL];
extern double * Bufferrec2_3E1[NB_LOCAL];
extern double * Bufferrec2_4E1[NB_LOCAL];
extern double * Bufferrec3_1E1[NB_LOCAL];
extern double * Bufferrec3_2E1[NB_LOCAL];
extern double * Bufferrec3_5E1[NB_LOCAL];
extern double * Bufferrec3_6E1[NB_LOCAL];
extern double * Bufferrec4_5E1[NB_LOCAL];
extern double * Bufferrec4_6E1[NB_LOCAL];
extern double * Bufferrec4_7E1[NB_LOCAL];
extern double * Bufferrec4_8E1[NB_LOCAL];
extern double * Bufferrec5_1E1[NB_LOCAL];
extern double * Bufferrec5_3E1[NB_LOCAL];
extern double * Bufferrec5_5E1[NB_LOCAL];
extern double * Bufferrec5_7E1[NB_LOCAL];
extern double * Bufferrec6_2E1[NB_LOCAL];
extern double * Bufferrec6_4E1[NB_LOCAL];
extern double * Bufferrec6_6E1[NB_LOCAL];
extern double * Bufferrec6_8E1[NB_LOCAL];

extern double * Bufferrec1_3E2[NB_LOCAL];
extern double * Bufferrec1_4E2[NB_LOCAL];
extern double * Bufferrec1_7E2[NB_LOCAL];
extern double * Bufferrec1_8E2[NB_LOCAL];
extern double * Bufferrec2_1E2[NB_LOCAL];
extern double * Bufferrec2_2E2[NB_LOCAL];
extern double * Bufferrec2_3E2[NB_LOCAL];
extern double * Bufferrec2_4E2[NB_LOCAL];
extern double * Bufferrec3_1E2[NB_LOCAL];
extern double * Bufferrec3_2E2[NB_LOCAL];
extern double * Bufferrec3_5E2[NB_LOCAL];
extern double * Bufferrec3_6E2[NB_LOCAL];
extern double * Bufferrec4_5E2[NB_LOCAL];
extern double * Bufferrec4_6E2[NB_LOCAL];
extern double * Bufferrec4_7E2[NB_LOCAL];
extern double * Bufferrec4_8E2[NB_LOCAL];
extern double * Bufferrec5_1E2[NB_LOCAL];
extern double * Bufferrec5_3E2[NB_LOCAL];
extern double * Bufferrec5_5E2[NB_LOCAL];
extern double * Bufferrec5_7E2[NB_LOCAL];
extern double * Bufferrec6_2E2[NB_LOCAL];
extern double * Bufferrec6_4E2[NB_LOCAL];
extern double * Bufferrec6_6E2[NB_LOCAL];
extern double * Bufferrec6_8E2[NB_LOCAL];

extern double * BuffersendE1corn9[NB_LOCAL];
extern double * BuffersendE1corn10[NB_LOCAL];
extern double * BuffersendE1corn11[NB_LOCAL];
extern double * BuffersendE1corn12[NB_LOCAL];
extern double * BuffersendE2corn5[NB_LOCAL];
extern double * BuffersendE2corn6[NB_LOCAL];
extern double * BuffersendE2corn7[NB_LOCAL];
extern double * BuffersendE2corn8[NB_LOCAL];
extern double * BuffersendE3corn1[NB_LOCAL];
extern double * BuffersendE3corn2[NB_LOCAL];
extern double * BuffersendE3corn3[NB_LOCAL];
extern double * BuffersendE3corn4[NB_LOCAL];
extern double * BufferrecE1corn9[NB_LOCAL];
extern double * BufferrecE1corn10[NB_LOCAL];
extern double * BufferrecE1corn11[NB_LOCAL];
extern double * BufferrecE1corn12[NB_LOCAL];
extern double * BufferrecE2corn5[NB_LOCAL];
extern double * BufferrecE2corn6[NB_LOCAL];
extern double * BufferrecE2corn7[NB_LOCAL];
extern double * BufferrecE2corn8[NB_LOCAL];
extern double * BufferrecE3corn1[NB_LOCAL];
extern double * BufferrecE3corn2[NB_LOCAL];
extern double * BufferrecE3corn3[NB_LOCAL];
extern double * BufferrecE3corn4[NB_LOCAL];

extern double * tempBufferrecE1corn9[NB_LOCAL];
extern double * tempBufferrecE1corn10[NB_LOCAL];
extern double * tempBufferrecE1corn11[NB_LOCAL];
extern double * tempBufferrecE1corn12[NB_LOCAL];
extern double * tempBufferrecE2corn5[NB_LOCAL];
extern double * tempBufferrecE2corn6[NB_LOCAL];
extern double * tempBufferrecE2corn7[NB_LOCAL];
extern double * tempBufferrecE2corn8[NB_LOCAL];
extern double * tempBufferrecE3corn1[NB_LOCAL];
extern double * tempBufferrecE3corn2[NB_LOCAL];
extern double * tempBufferrecE3corn3[NB_LOCAL];
extern double * tempBufferrecE3corn4[NB_LOCAL];

extern double * BufferrecE1corn9_3[NB_LOCAL];
extern double * BufferrecE1corn9_7[NB_LOCAL];
extern double * BufferrecE1corn10_1[NB_LOCAL];
extern double * BufferrecE1corn10_5[NB_LOCAL];
extern double * BufferrecE1corn11_2[NB_LOCAL];
extern double * BufferrecE1corn11_6[NB_LOCAL];
extern double * BufferrecE1corn12_4[NB_LOCAL];
extern double * BufferrecE1corn12_8[NB_LOCAL];
extern double * BufferrecE2corn5_2[NB_LOCAL];
extern double * BufferrecE2corn5_4[NB_LOCAL];
extern double * BufferrecE2corn6_1[NB_LOCAL];
extern double * BufferrecE2corn6_3[NB_LOCAL];
extern double * BufferrecE2corn7_5[NB_LOCAL];
extern double * BufferrecE2corn7_7[NB_LOCAL];
extern double * BufferrecE2corn8_6[NB_LOCAL];
extern double * BufferrecE2corn8_8[NB_LOCAL];
extern double * BufferrecE3corn1_3[NB_LOCAL];
extern double * BufferrecE3corn1_4[NB_LOCAL];
extern double * BufferrecE3corn2_1[NB_LOCAL];
extern double * BufferrecE3corn2_2[NB_LOCAL];
extern double * BufferrecE3corn3_5[NB_LOCAL];
extern double * BufferrecE3corn3_6[NB_LOCAL];
extern double * BufferrecE3corn4_7[NB_LOCAL];
extern double * BufferrecE3corn4_8[NB_LOCAL];

extern double * tempBufferrecE1corn9_3[NB_LOCAL];
extern double * tempBufferrecE1corn9_7[NB_LOCAL];
extern double * tempBufferrecE1corn10_1[NB_LOCAL];
extern double * tempBufferrecE1corn10_5[NB_LOCAL];
extern double * tempBufferrecE1corn11_2[NB_LOCAL];
extern double * tempBufferrecE1corn11_6[NB_LOCAL];
extern double * tempBufferrecE1corn12_4[NB_LOCAL];
extern double * tempBufferrecE1corn12_8[NB_LOCAL];
extern double * tempBufferrecE2corn5_2[NB_LOCAL];
extern double * tempBufferrecE2corn5_4[NB_LOCAL];
extern double * tempBufferrecE2corn6_1[NB_LOCAL];
extern double * tempBufferrecE2corn6_3[NB_LOCAL];
extern double * tempBufferrecE2corn7_5[NB_LOCAL];
extern double * tempBufferrecE2corn7_7[NB_LOCAL];
extern double * tempBufferrecE2corn8_6[NB_LOCAL];
extern double * tempBufferrecE2corn8_8[NB_LOCAL];
extern double * tempBufferrecE3corn1_3[NB_LOCAL];
extern double * tempBufferrecE3corn1_4[NB_LOCAL];
extern double * tempBufferrecE3corn2_1[NB_LOCAL];
extern double * tempBufferrecE3corn2_2[NB_LOCAL];
extern double * tempBufferrecE3corn3_5[NB_LOCAL];
extern double * tempBufferrecE3corn3_6[NB_LOCAL];
extern double * tempBufferrecE3corn4_7[NB_LOCAL];
extern double * tempBufferrecE3corn4_8[NB_LOCAL];

extern double * BufferrecE1corn9_32[NB_LOCAL];
extern double * BufferrecE1corn9_72[NB_LOCAL];
extern double * BufferrecE1corn10_12[NB_LOCAL];
extern double * BufferrecE1corn10_52[NB_LOCAL];
extern double * BufferrecE1corn11_22[NB_LOCAL];
extern double * BufferrecE1corn11_62[NB_LOCAL];
extern double * BufferrecE1corn12_42[NB_LOCAL];
extern double * BufferrecE1corn12_82[NB_LOCAL];
extern double * BufferrecE2corn5_22[NB_LOCAL];
extern double * BufferrecE2corn5_42[NB_LOCAL];
extern double * BufferrecE2corn6_12[NB_LOCAL];
extern double * BufferrecE2corn6_32[NB_LOCAL];
extern double * BufferrecE2corn7_52[NB_LOCAL];
extern double * BufferrecE2corn7_72[NB_LOCAL];
extern double * BufferrecE2corn8_62[NB_LOCAL];
extern double * BufferrecE2corn8_82[NB_LOCAL];
extern double * BufferrecE3corn1_32[NB_LOCAL];
extern double * BufferrecE3corn1_42[NB_LOCAL];
extern double * BufferrecE3corn2_12[NB_LOCAL];
extern double * BufferrecE3corn2_22[NB_LOCAL];
extern double * BufferrecE3corn3_52[NB_LOCAL];
extern double * BufferrecE3corn3_62[NB_LOCAL];
extern double * BufferrecE3corn4_72[NB_LOCAL];
extern double * BufferrecE3corn4_82[NB_LOCAL];

#if (DOHELM)
extern double * eos_table[1];
extern double * GPU_eos_table[1];
#endif

/*************************************************************************
GLOBAL VARIABLES SECTION
*************************************************************************/
/* physics parameters */
extern double a;
extern double gam;

/* numerical parameters */
extern double Rin, Rout, R0, fractheta, x1br, rbr, npow2, cpow2, x1max;
extern double cour;
extern double dV, dx[NB_LOCAL][NPR], startx[NPR];
extern double dt, bdt[NB_LOCAL][4];
extern int NODE_global[NB];
extern double t, tf;
extern int nstep;
extern double sourceflag, period_max;
extern double rmax;
extern double ndt, ndt1, ndt2, ndt3;
extern int numtasks, rank, local_rank, rc;
extern int prestep_half[NB_LOCAL], prestep_full[NB_LOCAL];
extern int max_levels;
extern int reduce_timestep;
extern int nthreads,numdevices;
extern int gpu, gpu_offset;
extern int status;
extern double y_max;

/* output parameters */
extern double DTd;
extern double DTd_reduced;
extern double DTl;
extern double DTi;
extern int    DTr;
extern double tref;
extern int    dump_cnt, dump_cnt_reduced;
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

struct of_trans {
	double Mud[NDIM][NDIM];
	double Mud_inv[NDIM][NDIM];
};

struct of_state {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
};

struct of_state_res {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
	double econ[NDIM];
	double ecov[NDIM];
};

struct of_state_rad {
    double ucon[NDIM];
    double ucov[NDIM];
};

/*Timing/benchmarking decleration*/
extern clock_t begin1, end1, begin2, end2;
extern double time_spent3;

/*Parallel write*/
extern double *dump_buffer;
extern double(*restrict dxdxp_z[NB_LOCAL])[NDIM][NDIM];
extern double(*restrict dxpdx_z[NB_LOCAL])[NDIM][NDIM];
extern float *array[NB_LOCAL], *array_reduced[NB_LOCAL], *array_diag[NB_LOCAL];
extern int *array_gdumpgrid, *array_rdumpgrid;
extern double *array_rdump[NB_LOCAL], *array_gdump1[NB_LOCAL], *array_gdump2[NB_LOCAL], *array_gdump1_reduced[NB_LOCAL], *array_gdump2_reduced[NB_LOCAL];
extern int first_dump, first_dump_reduced, first_rdump, first_gdump, restart_number;
extern FILE *fparam_dump, *fparam_dump_reduced, *fparam_restart;

/*AMR parameters*/
extern int(*block)[NV];
extern int *lin_coord[N_LEVELS];
extern int *lin_coord_RM[N_LEVELS];
extern double ref_val[MY_MAX(NB, 40000)];
extern int n_ord[NB_LOCAL], nl[NB], n_ord_total[NB], n_ord_RM[NB_LOCAL], n_ord_total_RM[NB], (*n_ord_node)[NB_LOCAL];
extern int mem_spot[NB_LOCAL], mem_spot_gpu[NB_LOCAL], mem_spot_gpu_bound[NB_LOCAL];
extern int n_active, *n_active_node, n_active_total, n_max;
extern int count_node[1], count_gpu[N_GPU];
extern int N1_GPU_offset[NB];
extern int N2_GPU_offset[NB];
extern int N3_GPU_offset[NB];

/*************************************************************************
FUNCTION DECLARATIONS
*************************************************************************/
//Output related
void dump_new(void);
void dump_new_reduced(void);
void gdump_new(void);
void gdump_new_reduced(void);
void dump_params(FILE *fp, int dump_reduced); 
double divb_calc(int n, int i, int j, int z);
void param_read(FILE *fp);
void rdump_block_read(FILE *fp, int n);
int restart_read_param(void);
void restart_write(void);
int restart_read(void);
void dump_read(void);
void gdump_read(FILE *fp);
void close_dump();
void close_dump_reduced();
void close_rdump();
void close_gdump();
void close_gdump_reduced();
double get_wall_time();

/** Evolution/physics functions **/
double advance(int flag);
double advance_GPU(void);
void bound_prim(double(*restrict pr[NB_LOCAL])[NPR], int MPI);
double fluxcalc(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n);
void   flux_ct(double(*restrict F1[NB_LOCAL])[NPR], double(*restrict F2[NB_LOCAL])[NPR], double(*restrict F3[NB_LOCAL])[NPR], int n);
void const_transport1(double(*restrict p[NB_LOCAL])[NPR], int n);
void const_transport1_M1_2(double(*restrict pb[NB_LOCAL])[NPR], int n);
void const_transport_bound(void);
void const_transport2(double(*restrict psi[NB_LOCAL])[NDIM], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n);
void utoprim(double(*restrict pi[NB_LOCAL])[NPR], double(*restrict pb[NB_LOCAL])[NPR], double(*restrict pf[NB_LOCAL])[NPR], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n);
void utoprim_M1_0(double Dt, int n);
void utoprim_M1_1(double Dt, int n);
void utoprim_M1_2(double Dt, int n);
void E_average(void);
double bsq_calc(double * restrict pr, struct of_geom * restrict geom);
int gamma_calc(double * restrict pr, struct of_geom * restrict geom, double *restrict gamma);
int gamma_calc_rad(double * restrict pr, struct of_geom * restrict geom, double * restrict gamma_rad);
void bcon_calc(double * restrict pr, double * restrict ucon, double * restrict ucov, double * restrict bcon);
void read_E_avg(double(*E_avg1)[BS_1 + 2 * N1G], double(*E_avg2)[BS_1 + 2 * N1G], int n);
void write_E_avg(double(*E_avg1)[BS_1 + 2 * N1G], double(*E_avg2)[BS_1 + 2 * N1G], int n);
void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
void step_ch(void);
void primtoflux(double * restrict pa, struct of_state * restrict q, struct of_state_rad * restrict q_rad, int dir, struct of_geom * restrict geom, double * restrict fl);
void primtoU(double * restrict p, struct of_state * restrict q, struct of_geom * restrict geom, double * restrict U);
void inflow_check(double *pr, int n, int ii, int jj, int zz, int type);
void source(double * restrict pa, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict U, double Dt);
void source_rad(double * restrict ph, struct of_geom * restrict geom, double * restrict dU);
void calc_ymax(void);
void implicit_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double cell_size);
void implicit_rad_solve_init(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size);
int implicit_rad_solve_PMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged);
int implicit_rad_solve_PRAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged);
int implicit_rad_solve_UMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged);
int implicit_rad_solve_EMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged);
int implicit_rad_solve_URAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged);
int subcycle_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double cell_size);
void u_to_v(double *pr, int i, int j);
void fixup(double((*restrict pv[NB_LOCAL])[NPR]), int n);
void fixup1zone(int i, int j, int z, int n, double prim[NPR]);
void fixup_utoprim(double(*restrict pv[NB_LOCAL])[NPR], int n);
void fixup_utoprim_rad(double((*restrict pv[NB_LOCAL])[NPR]), int n);
void ucon_calc(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon);
void ucon_calc_rad(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon_rad);
void usrfun(double *pr, int n, double *beta, double **alpha);
void calc_source();
void mhd_calc(double * restrict pr, int dir, struct of_state * restrict q, double * restrict mhd);
void mhd_calc_rad(double * restrict pr, int dir, struct of_state_rad * restrict q_rad, double * restrict mhd_rad);
double calc_kappa_abs(double * restrict ph);
double calc_kappa_emmit(double * restrict ph);
double calc_kappa_es(double * restrict ph);
void misc_source(double * restrict ph, int ii, int jj, struct of_geom * restrict geom, struct of_state * restrict q, double * restrict dU, double r, double Dt);
void Utoprim(double *Ua, struct of_geom *geom, double *pa);
int Rtoprim(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], int lim);
void get_state(double *pr, struct of_geom *geom, struct of_state *q);
void get_state_rad(double * restrict pr, struct of_geom * restrict geom, struct of_state_rad * restrict q_rad);
void calc_Gcon(double * restrict ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM]);
void fix_flux(double(*restrict F1[NB_LOCAL])[NPR], double(*restrict F2[NB_LOCAL])[NPR], double(*restrict F3[NB_LOCAL])[NPR], int n);
int Utoprim_2d(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim);
int Utoprim_3d_res(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim);
int Utoprim_NM(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim);
int Utoprim_1dvsq2fix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim);
int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim);
void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *cmax, double *cmin, int a, int b, int c);
void vchar_rad(double * restrict pr, struct of_state* restrict q, struct of_state_rad * restrict q_rad, struct of_geom * restrict geom, int js, double * restrict vmax, double * restrict vmin, double dx);
void step_ch_debug();
void GPU_benchmark(void);
void GPU_init(void);
void set_arrays_GPU(int n, int device);
void GPU_write(int n);
void GPU_finish(int n, int force_delete);
void GPU_hcor(int n);
void GPU_fixup(int flag, int n, double Dt);
void GPU_fixup_post(int n, double Dt);
void GPU_cleanup_post(int n);
void GPU_fixuputoprim(int flag, int n);
void GPU_Utoprim_M1_0(int n, double Dt);
void GPU_Utoprim_M1_1(int n, double Dt);
void GPU_Utoprim_M1_2(int n, double Dt);
void GPU_fluxcalc2D(int dir, int flag, int n);
void GPU_reconstruct_internal(int flag, int n);
void GPU_fluxcalcprep(int dir, int flag, int ppm_enable, int n);
void GPU_flux_ct1(int n);
void GPU_flux_ct2(int n);
void GPU_fix_flux(int n);
void GPU_boundprim(int bound_force);
void GPU_boundprim1(int flag, int n);
void GPU_boundprim2(int flag, int n);
void GPU_boundprim_trans(int flag, int n);
void GPU_step_ch();
void GPU_read(int n);
void GPU_consttransport1(int flag, double Dt, int n);
void GPU_consttransport2(int flag, double Dt, int n);
void GPU_consttransport2_M1_2(int flag, double Dt, int n);
void GPU_consttransport3(int flag, double Dt, int n);
void GPU_consttransport3_post(double Dt, int n);
void GPU_consttransport_bound(void);
void read_time_GPU(void);
double fluxcalc_GPU(int n, int dir);

//Resistivity related
//In phys_res.c
void primtoflux_res(double* restrict pr, struct of_state_res* restrict q_res, int dir, struct of_geom* restrict geom, double* restrict flux);
void econ_calc_res(double* restrict pr, struct of_geom* restrict geom, double* restrict ucon, double* restrict bcon);
void bcon_calc_res(double* restrict pr, struct of_geom* restrict geom, double* restrict ucon, double* restrict bcon);
void mhd_calc_res(double* restrict pr, int dir, struct of_geom* restrict geom, struct of_state_res* restrict q_res, double* restrict mhd);
void source_res(double* restrict ph, struct of_geom* restrict geom, int n, int ii, int jj, int zz, double* restrict dU, double* q, double Dt);
void calc_J(double p[NPR], double J[NDIM], double q, struct of_geom* restrict geom);
void get_state_res(double* restrict pr, struct of_geom* restrict geom, struct of_state_res* restrict q_res);
void vchar_res(struct of_geom* restrict geom, int js, double* restrict vmax, double* restrict vmin);
double divE_calc(double(*restrict p[NB_LOCAL])[NPR], int n, int i, int j, int z);
void lower_3(double* restrict ucon, struct of_geom* restrict geom, double* restrict ucov);
double lvc4u(int i, int j, int k, int l);
double lvc3u(int i, int j, int k);
int invert_matrix_3D(double A[][3], double Ainv[][3]);
int LU_decompose_3D(double A[][3], int permute[]);
void LU_substitution_3D(double A[][3], double B[], int permute[]);

//In step_ch_res.c
void step_ch_res();
double advance_res(int flag);
double fluxcalc_res(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n);
void utoprim_M1_0_res(double Dt, int n);
void utoprim_M1_1_res(double Dt, int n);
void utoprim_M1_2_res(double Dt, int n);
double fluxcalc_res(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n);

//In const_trans_res.c
void const_transport1_res(double(*restrict pb[NB_LOCAL])[NPR], int n);
void const_transport1_M1_2_res(double(*restrict pb[NB_LOCAL])[NPR], int n);
void const_transport2_res(double(*restrict psi[NB_LOCAL])[NDIM], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n);

//Metric/Misc related
double bl_gdet_func(double r, double th);
double gdet_func(double lgcov[][NDIM]);
double mink(int j, int k);
double ranc(int seed);
double slope_lim(double y1, double y2, double y3);
void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double *F);
void area_map(int i, int j, int n, double(*restrict prim[NB_LOCAL])[NPR]);
void blgset(int n, int i, int j, struct of_geom *geom);
void ksgset(int n, int i, int j, struct of_geom *geom);
void bl_coord(double * restrict X, double * restrict r, double * restrict th, double * restrict phi);
void bl_gcon_func(double r, double th, double gcov[][NDIM]);
void kerr_gcov_func(double r, double th, double gcov[][NDIM]);
void bl_gcov_func(double r, double th, double gcov[][NDIM]);
void conn_func(double *X, struct of_geom *geom, double lconn[][NDIM][NDIM]);
void coord(int n, int i, int j, int z, int loc, double *X);
void diag(int call_code);
void diag_flux(double(*F1[NB_LOCAL])[NPR]);
void fail(int fail_type);
void set_mag(void);
void gcon_func(double lgcov[][NDIM], double lgcon[][NDIM]);
void gcov_func(double *X, double lgcov[][NDIM]);
void get_geometry(int n, int i, int j, int z, int loc, struct of_geom *geom);
void get_trans(int n, int ii, int jj, int zz, int ff, struct of_trans * restrict trans);
void get_geometry_direct(int ii, int jj, int zz, int ff, struct of_geom *geom);
int index_3D(int n, int i, int j, int z);
int index_2D(int n, int i, int j, int z);
void init(void);
void lower(double * restrict a, struct of_geom * restrict geom, double * restrict b);
void ludcmp(double **a, int n, int *indx, double * d);
void raise(double * restrict v1, struct of_geom * restrict geom, double * restrict v2);
int invert_matrix(double A[][NDIM], double Ainv[][NDIM]);
int LU_decompose(double A[][NDIM], int permute[]);
void LU_substitution(double A[][NDIM], double B[], int permute[]);

//AMR Related
void MPI_initialize(int argc, char *argv[]);
void activate_blocks(void);
void average_grid(void);
void prolong_grid(void);
void set_corners(int tag);
void set_communicator(void);
void pre_refine(void);
int refine(int n);
int derefine_pole(void);
void refine_field(int n, int n_child, int offset_1, int offset_2, int offset_3, double(*restrict pb[NB_LOCAL])[NDIM]);
void derefine(int n);
void post_refine(void);
int AMR_coord_linear2(int l, int b2, int i, int j, int z);
int AMR_coord_linear(int level, int i, int j, int z);
void AMR_coord_cart(int n, int *level, int *i, int *j, int *z);
void test_AMR(void);
void set_AMR(void);
int check_nesting(int n);
double calc_refcrit(int n);
void synch_refcrit(void);
void check_refcrit(void);
void alloc_bounds_CPU(int n);
void free_arrays(int n);
void free_bound_cpu(int n);
void set_prestep(void);
void prestep_bound(void);
void mpi_synch(int tag);
void set_timelevel(int tag);
void rm_order1(void);
void balance_load(void);
void balance_load_gpu(void);
void set_arrays_image(void);
void set_arrays(int n);
void set_grid(int n);
void alloc_bounds_GPU(int n);
void free_bound_gpu(int n);
void set_points(int n);
void set_gridparam(void);
void set_ref(int n, int n_rec, int *ref_1, int *ref_2, int * ref_3);
double calc_mem(int n_blocks);
double B1_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);
double B2_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);
double B3_prolong(int n, int i, int j, int z, double offset_1, double offset_2, double offset_3, double(*restrict pb[NB_LOCAL])[NDIM],
	double b1_1, double b1_2, double b1_3, double b1_4, double b1_5, double b1_6, double b1_7, double b1_8,
	double b2_1, double b2_2, double b2_3, double b2_4, double b2_5, double b2_6, double b2_7, double b2_8,
	double b3_1, double b3_2, double b3_3, double b3_4, double b3_5, double b3_6, double b3_7, double b3_8
	, int n_rec1, int n_rec2, int n_rec3, int n_rec4, int n_rec5, int n_rec6);

//Boundary transfer related
void set_iprobe(int mode, int * flag);
void bound_send1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep);
void bound_rec1(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n);
void bound_send2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep);
void bound_rec2(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n);
void bound_send3(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int n, int prestep);
void bound_rec3(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], double * Bufferps[NB_LOCAL], int bound_force, int n);
void pack_send1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferps, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void unpack_receive1(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive2(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int reverse);
void unpack_receive3(int n, int n_rec, int i_offset, int i1, int i2, int j_offset, int j1, int j2, int z_offset, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *tempreceive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **tempBufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);
void unpack_receive_coarse3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1receive[NB_LOCAL], double *temp2receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double(*restrict psim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferps, double **Bufferboundreceive, double **temp1Bufferboundreceive, double **temp2Bufferboundreceive, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2, int mpi);

void flux_send1(double(*restrict F1[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n);
void flux_rec1(double(*restrict F1[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void flux_send2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n);
void flux_rec2(double(*restrict F2[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void flux_send3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n);
void flux_rec3(double(*restrict F3[NB_LOCAL])[NPR], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void pack_send1_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send2_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send3_flux(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int isize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1, cudaEvent_t *boundevent2);
void pack_send_flux_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_flux_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_flux_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NPR], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void unpack_receive1_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);
void unpack_receive2_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);
void unpack_receive3_flux(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NPR],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent1, int calc_corr);

void E_send1(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E_send2(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E_send3(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E_rec1(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void E_rec2(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void E_rec3(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void E1_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E2_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E3_send_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void E1_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void E2_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void E3_receive_corn(double(*restrict E[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n, int calc_corr);
void pack_send1_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send2_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send3_E(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void pack_send_E_average3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent1);
void unpack_receive1_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void unpack_receive2_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void unpack_receive3_E(int n, int n_rec, int n_rec2, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL],
	double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr, int d1, int d2, int e1, int e2);
void pack_send_E1_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E2_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E3_corn(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E1_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E2_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_E3_corn_course(int n, int n_rec, int i1, int i2, int j, int z, double *send[NB_LOCAL], double(*restrict E[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void unpack_receive_E1_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);
void unpack_receive_E2_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);
void unpack_receive_E3_corn(int n, int n_rec, int n_rec2, int i1, int i2, int j, int z, double *receive[NB_LOCAL], double *temp1[NB_LOCAL], double *temp2[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM],
	double **Bufferp, double **Bufferboundreceive, double **Buffertemp1, double **Buffertemp2, cudaEvent_t *boundevent, int calc_corr);

void B_rec1(double(*restrict F1[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void B_rec2(double(*restrict F2[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void B_rec3(double(*restrict F3[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void B_send1(double(*restrict F1[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void B_send2(double(*restrict F2[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void B_send3(double(*restrict F3[NB_LOCAL])[NDIM], double * Bufferp[NB_LOCAL], int n);
void Bp_send1(double(*restrict F1[NB_LOCAL])[NDIM], int n);
void Bp_send2(double(*restrict F1[NB_LOCAL])[NDIM], int n);
void Bp_send3(double(*restrict F1[NB_LOCAL])[NDIM], int n);
void Bp_rec1(int n);
void Bp_rec2(int n);
void Bp_rec3(int n);
void pack_send_B1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent);
void pack_send_B_average1(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3);
void pack_send_B_average2(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3);
void pack_send_B_average3(int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *send[NB_LOCAL], double(*restrict F1[NB_LOCAL])[NDIM], double **Bufferp, double **Bufferboundsend, cudaEvent_t *boundevent, int ref_1, int ref_2, int ref_3);
void unpack_receive_B1(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent);
void unpack_receive_B2(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent, int neg);
void unpack_receive_B3(int n, int n_rec, int i1, int i2, int j1, int j2, int z1, int z2, int jsize, int zsize, double *receive[NB_LOCAL], double(*restrict prim[NB_LOCAL])[NDIM], int div, double **Bufferp, double **Bufferboundreceive, cudaEvent_t *boundevent);

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
void vofx_sjetcoords( double *X, double *V );
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
double fluxcalc_hllc(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n);
double fluxcalc_hlld(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n);

//EOS related
void eos_init(void);
void eos_init_GPU(void);
void interp_eostable(double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele);
void test_eos(void);
void eos_helm(int calc_derivatives, double btemp, double den, double abar, double zbar, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dpresdd, double* cs2);
void eos_mode_rhou_entr(double den, double u_goal, double* entr);
void eos_mode_rhou_pres (double den, double u_goal, double *pres);
void eos_mode_rhou_pres_cs2(double den, double u_goal, double *pres, double *cs2);
void eos_mode_rhow_pres_dpdrho_dpde_d (double den, double w_goal, double *pres, double *dpdrho, double *dpde_d);
void eos_mode_rhow_pres_u (double den, double w_goal, double *pres, double *u);
void eos_mode_rhotemp_pres_min (double den, double *pres);
void eos_mode_rhopres_u (double den, double p_goal, double *u);
void eos_mode_rhou_temp (double den, double u_goal, double* temp);

