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

/* restart functions; restart_init and restart_dump */
#include "decs_MPI.h"
void primtoflux_FT(double * restrict pr, struct of_state * restrict q, int dir, struct of_geom * restrict geom, double restrict flux[NPR]);
void vchar_FT(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js, double  restrict *vmax, double restrict *vmin, int n, int a, int b, int c);
void ctop_to_utop(double ctop[NDIM], double cmax[NDIM]);
void set_Mud(int n);

void set_Mud(int n){
	#if(HLLC)
	int i, j, z;
	double A, B, C, D, E, F, G, H;
	struct of_geom geom;
	#if(!NSY)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]) {
	#else
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
	#endif
		//dir 1
		get_geometry(n, i, j, z, FACE1, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[1][1] - geom.gcon[0][1] * geom.gcon[0][1]), -0.5);
		C = pow(geom.gcov[3][3], -0.5);
		D = pow((geom.gcov[3][3])*(geom.gcov[2][2] * geom.gcov[3][3] - geom.gcov[2][3] * geom.gcov[2][3]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][1] = B*(geom.gcon[0][1] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[1][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][1] = B*(geom.gcon[0][1] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[1][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][2] = D*geom.gcov[3][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][1] = B*(geom.gcon[0][1] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[1][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][2] = -D*geom.gcov[2][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][3] = C;

		E = geom.gcon[0][1] * geom.gcon[1][2] - geom.gcon[1][1] * geom.gcon[0][2];
		F = geom.gcon[0][1] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[1][2];
		G = geom.gcon[0][1] * geom.gcon[1][3] - geom.gcon[1][1] * geom.gcon[0][3];
		H = geom.gcon[0][1] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[1][3];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][0] = B*geom.gcon[0][1];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][1] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][1] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[3][3]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][2] = 1./(D*geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[2][3]/geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][1] = (B*B / C)*geom.gcon[0][0]*(H+F*geom.gcov[2][3]/geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][2] = (1./C)*geom.gcov[2][3]/geom.gcov[3][3];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][3] = 1./C;

		//dir 2
		get_geometry(n, i, j, z, FACE2, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[2][2] - geom.gcon[0][2] * geom.gcon[0][2]), -0.5);
		C = pow(geom.gcov[1][1], -0.5);
		D = pow((geom.gcov[1][1])*(geom.gcov[3][3] * geom.gcov[1][1] - geom.gcov[3][1] * geom.gcov[3][1]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][1] = B*(geom.gcon[0][2] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[2][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][1] = B*(geom.gcon[0][2] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[2][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][2] = D*geom.gcov[1][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][1] = B*(geom.gcon[0][2] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[2][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][2] = -D*geom.gcov[3][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][3] = C;

		E = geom.gcon[0][2] * geom.gcon[2][3] - geom.gcon[2][2] * geom.gcon[0][3];
		F = geom.gcon[0][2] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[2][3];
		G = geom.gcon[0][2] * geom.gcon[2][1] - geom.gcon[2][2] * geom.gcon[0][1];
		H = geom.gcon[0][2] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[2][1];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][0] = B*geom.gcon[0][2];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][2] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][2] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[1][1]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][3] = 1. / (D*geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[3][1] / geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][2] = (B*B / C)*geom.gcon[0][0] * (H + F*geom.gcov[3][1] / geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][3] = (1. / C)*geom.gcov[3][1] / geom.gcov[1][1];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][1] = 1. / C;

		//dir 3
		get_geometry(n, i, j, z, FACE3, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[3][3] - geom.gcon[0][3] * geom.gcon[0][3]), -0.5);
		C = pow(geom.gcov[2][2], -0.5);
		D = pow((geom.gcov[2][2])*(geom.gcov[1][1] * geom.gcov[2][2] - geom.gcov[1][2] * geom.gcov[1][2]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][1] = B*(geom.gcon[0][3] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[3][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][1] = B*(geom.gcon[0][3] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[3][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][2] = D*geom.gcov[2][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][1] = B*(geom.gcon[0][3] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[3][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][2] = -D*geom.gcov[1][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][3] = C;

		E = geom.gcon[0][3] * geom.gcon[3][1] - geom.gcon[3][3] * geom.gcon[0][1];
		F = geom.gcon[0][3] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[3][1];
		G = geom.gcon[0][3] * geom.gcon[3][2] - geom.gcon[3][3] * geom.gcon[0][2];
		H = geom.gcon[0][3] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[3][2];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][0] = B*geom.gcon[0][3];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][3] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][3] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[2][2]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][1] = 1. / (D*geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[1][2] / geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][3] = (B*B / C)*geom.gcon[0][0] * (H + F*geom.gcov[1][2] / geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][1] = (1. / C)*geom.gcov[1][2] / geom.gcov[2][2];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][2] = 1. / C;
	}
	#endif
}

/***********************************************************************************************/
/***********************************************************************************************
fluxcalc():
---------
-- sets the numerical fluxes, avaluated at the cell boundaries using the slope limiter
slope_lim();

-- only has HLL and Lax-Friedrichs  approximate Riemann solvers implemented;

***********************************************************************************************/
double fluxcalc_hllc(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n)
{
	int i, j, z, k, idel, jdel, zdel, face, i1, j1, i2, j2;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR], F_HLL[NPR], U_HLL[NPR], vcon[NDIM], U_i[NPR], F1[NDIM][NPR], F_FT[NDIM][NPR], ptot;
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ndt, ndt_thread, dtij;
	double ctop;
	struct of_geom geom;
	struct of_state state_l, state_r, state_l_FT, state_r_FT, state_roe, qi;
	struct of_trans trans;
	double bsq;
	int max_i, max_j, max_z;
	double val;
	ndt = 1.e9;
	int ind0, ind1;
	int fail_HLLC = 0;
	int counter0 = 0;
	int counter1 = 0;
	double test;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }
	else { exit(10); }

		#pragma omp parallel private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l, state_r, state_roe, F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ctop, dtij, ind0, ind1, U_HLL, F_HLL, qi, vcon, U_i, bsq, fail_HLLC, test, ptot, F_FT,i1,i2,j1,j2,state_l_FT, state_r_FT, trans, F1)
		{
			ndt_thread = 1.e9;

			/* then evaluate slopes */
			#pragma omp for collapse(2) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)/nthreads)
			ZSLOOP3D(N1_GPU_offset[n] - D1, N1_GPU_offset[n] + BS_1 - 1 + D1, N2_GPU_offset[n] - D2, N2_GPU_offset[n] + BS_2 - 1 + D2, N3_GPU_offset[n] - D3, N3_GPU_offset[n] + BS_3 - 1 + D3){
				// #pragma ivdep
				PLOOP{
					dq[nl[n]][index_3D(n, i, j, z)][k] = slope_lim(pr[nl[n]][index_3D(n, i - idel, j - jdel, z - zdel)][k], pr[nl[n]][index_3D(n, i, j, z)][k], pr[nl[n]][index_3D(n, i + idel, j + jdel, z + zdel)][k]);
				}
			}

			#pragma omp for collapse(2) schedule(static,(BS_1+jdel+zdel+1)*(BS_2+idel+zdel+1)/nthreads)
			ZSLOOP((N1_GPU_offset[n] - jdel - zdel)*D1, (N1_GPU_offset[n] + BS_1)*D1, (N2_GPU_offset[n] - idel - zdel)*D2, (N2_GPU_offset[n] + BS_2)*D2) 	{
				for (z = (N3_GPU_offset[n] - idel - jdel)*D3; z <= (N3_GPU_offset[n] + BS_3)*D3; z++){
					get_geometry(n, i, j, z, face, &geom);
					get_trans(n, i, j, z, dir, &trans);

					ind0 = index_3D(n, i, j, z);
					ind1 = index_3D(n, i - idel, j - jdel, z - zdel);

					#pragma ivdep
					PLOOP{
						p_l[k] = pr[nl[n]][ind1][k] + 0.5*dq[nl[n]][ind1][k];
						p_r[k] = pr[nl[n]][ind0][k] - 0.5*dq[nl[n]][ind0][k];
						#if(STAGGERED)
						if ((dir == 1 && k == B1)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 2 && k == B2)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						if ((dir == 3 && k == B3)){
							if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
							else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
							p_r[k] = p_l[k];
						}
						#endif
					}

					#if(STAGGERED)
					if ((dir == 2) && ((j == 0 && (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3)) || (j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2])) && (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3)))){
						p_r[B1] = 0.;
						p_l[B1] = 0.;
					}
					#endif

					//First calculate HLL fluxes for F[B1], F[B2] and F[B3]
					get_state(p_l, &geom, &state_l);
					get_state(p_r, &geom, &state_r);

					for (k = B1; k <= B3; k++){
						#pragma ivdep
						F_l[k] = geom.g*(state_l.bcon[k - 4] * state_l.ucon[dir] - state_l.bcon[dir] * state_l.ucon[k - 4]);
						F_r[k] = geom.g*(state_r.bcon[k - 4] * state_r.ucon[dir] - state_r.bcon[dir] * state_r.ucon[k - 4]);
						U_l[k] = geom.g*(state_l.bcon[k - 4] * state_l.ucon[0] - state_l.bcon[0] * state_l.ucon[k - 4]);
						U_r[k] = geom.g*(state_r.bcon[k - 4] * state_r.ucon[0] - state_r.bcon[0] * state_r.ucon[k - 4]);
					}

					#if(DOKTOT )
					F_l[KTOT] = geom.g*(p_l[RHO] * state_l.ucon[dir] * p_l[KTOT]);
					F_r[KTOT] = geom.g*(p_r[RHO] * state_r.ucon[dir] * p_r[KTOT]);
					U_l[KTOT] = (F_l[KTOT]);
					U_r[KTOT] = (F_r[KTOT]);
					#endif

					vchar(p_l, &state_l, &geom, dir, &(cmax_l), &(cmin_l), i, j, z);
					vchar(p_r, &state_r, &geom, dir, &(cmax_r), &(cmin_r), i, j, z);
					cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
					cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
					ctop = MY_MAX(cmax, cmin);
					// evaluate restriction on timestep 
					dtij = fabs(cour*dx[nl[n]][dir] / ctop);
					if (dtij < ndt_thread) {
						ndt_thread = dtij;

					}
					
						//Calculate HLL flux for F[B1]-F[B3]
						#pragma ivdep
						for (k = B1; k < NPR; k++){
							F1[dir][k] = HLLF*((cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
								+ LAXF*(0.5*(F_l[k] + F_l[k] - ctop * (U_r[k] - U_l[k])));
						}



						//Transform 4 velocities and 4 magnetic fields to orthonormal frame
						for (i1 = 0; i1 < NDIM; i1++){
							state_l_FT.ucon[i1] = 0.0;
							state_l_FT.ucov[i1] = 0.0;
							state_l_FT.bcon[i1] = 0.0;
							state_l_FT.bcov[i1] = 0.0;
							state_r_FT.ucon[i1] = 0.0;
							state_r_FT.ucov[i1] = 0.0;
							state_r_FT.bcon[i1] = 0.0;
							state_r_FT.bcov[i1] = 0.0;
							for (j1 = 0; j1 < NDIM; j1++){
								state_l_FT.ucon[i1] += state_l.ucon[j1] * trans.Mud_inv[i1][j1];
								state_l_FT.ucov[i1] += state_l.ucov[j1] * trans.Mud[j1][i1];
								state_l_FT.bcon[i1] += state_l.bcon[j1] * trans.Mud_inv[i1][j1];
								state_l_FT.bcov[i1] += state_l.bcov[j1] * trans.Mud[j1][i1];
								state_r_FT.ucon[i1] += state_r.ucon[j1] * trans.Mud_inv[i1][j1];
								state_r_FT.ucov[i1] += state_r.ucov[j1] * trans.Mud[j1][i1];
								state_r_FT.bcon[i1] += state_r.bcon[j1] * trans.Mud_inv[i1][j1];
								state_r_FT.bcov[i1] += state_r.bcov[j1] * trans.Mud[j1][i1];
							}
						}
						if ((dot(state_l_FT.bcon, state_l_FT.bcov) - dot(state_l.bcon, state_l.bcov)) / dot(state_l.bcon, state_l.bcov)>0.00001) printf("test: %f %f \n", log(dot(state_l_FT.bcon, state_l_FT.bcov)), log(dot(state_l.bcon, state_l.bcov)));
						if ((dot(state_l_FT.ucon, state_l_FT.ucov) - dot(state_l.ucon, state_l.ucov)) / dot(state_l.ucon, state_l.ucov)>0.00001) printf("test2: %f %f \n", log(dot(state_l_FT.ucon, state_l_FT.ucov)), log(dot(state_l.ucon, state_l.ucov)));

						primtoflux_FT(p_l, &state_l_FT, dir, &geom, F_l);
						primtoflux_FT(p_r, &state_r_FT, dir, &geom, F_r);

						primtoflux_FT(p_l, &state_l_FT, 0, &geom, U_l);
						primtoflux_FT(p_r, &state_r_FT, 0, &geom, U_r);

						vchar_FT(p_l, &state_l_FT, &geom, dir, &(cmax_l), &(cmin_l), n, i, j, z);
						vchar_FT(p_r, &state_r_FT, &geom, dir, &(cmax_r), &(cmin_r), n, i, j, z);

						#if(0)
						for (i1 = 1; i1 < NDIM; i1++){
							if (dot(state_r_FT.bcon, state_r_FT.bcov) / p_r[RHO] > 0.0001 && dot(state_l_FT.bcon, state_l_FT.bcov) / p_r[RHO] > 0.0001){
								//Get wavespeed defined as maximum of left and right state
								cmax_roe = MY_MAX(cmax_r[i1], cmax_l[i1]);
								cmin_roe = MY_MIN(cmin_r[i1], cmin_l[i1]);
								fail_HLLC=0;

								//Set U_HLL and F_HLL
								for (k = 0; k < B1; k++){
									U_HLL[k] = (F_l[i1][k] - F_r[i1][k] + cmax_roe*F_r[0][k] - cmin_roe*F_l[0][k]) / (cmax_roe - cmin_roe+ SMALL);
									F_HLL[k] = (cmax_roe*F_l[i1][k] - cmin_roe*F_r[i1][k] + cmax_roe*cmin_roe*(F_r[0][k] - F_l[0][k])) / (cmax_roe - cmin_roe + SMALL);
								}

								/*Set strength of magnetic field (free parameter in solution, will just take the average of the left and right
								extrapolated state for the moment untill we know better)*/
								if(j1==1) U_HLL[B1] = p_l[B1];
								else if (j1 == 2) U_HLL[B2] = p_l[B2];
								else if (j1 == 3) U_HLL[B3] = p_l[B3];

								//If |B1|<0.001*|B2| || |B1|<0.001*|B3| revert to HLL flux
								if (fabs(U_HLL[B1]) < 0.001*fabs(U_HLL[B2]) || fabs(U_HLL[B1]) < 0.001*fabs(U_HLL[B3]) && i1 == 1) fail_HLLC = 1;
								if (fabs(U_HLL[B2]) < 0.001*fabs(U_HLL[B1]) || fabs(U_HLL[B2]) < 0.001*fabs(U_HLL[B3]) && i1 == 2) fail_HLLC = 1;
								if (fabs(U_HLL[B3]) < 0.001*fabs(U_HLL[B1]) || fabs(U_HLL[B3]) < 0.001*fabs(U_HLL[B1]) && i1 == 3) fail_HLLC = 1;

								//Solve for velocities of intermediate state and calculate magnetic field b.
								solve_HLLC(&qi, &geom, vcon, U_HLL, F_HLL, &fail_HLLC, i1);
								bsq = dot(qi.bcon, qi.bcov);

								//Calculate total pressure ptot=pgas+0.5*bsq
								ptot = F_HLL[UU+i1] + qi.bcon[i1] * qi.bcov[i1] - qi.ucov[i1] / qi.ucov[0] * (F_HLL[UU] + qi.bcon[i1] * qi.bcov[0]);
								if(ptot<0.) fail_HLLC=1;

								if (cmax_roe>0. && qi.ucon[i1] <= 0. && fail_HLLC == 0){
									//Set Rankine-Hugoniot jump conditions
									if (i1 == 1){
										U_i[RHO] = (cmax_roe - state_r.ucon[i1] / state_r.ucon[0]) / (cmax_roe - vcon[i1] + SMALL)*U_r[RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[i1] * qi.bcov[0] - F_r[i1][UU] + cmax_roe*U_r[UU]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U1] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[i1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[i1] - qi.bcon[i1] * qi.bcov[2] - F_r[i1][U2] + cmax_roe*U_r[U2]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[i1] - qi.bcon[i1] * qi.bcov[3] - F_r[i1][U3] + cmax_roe*U_r[U3]) / (cmax_roe - vcon[i1] + SMALL);
									}
									else if (i1 == 2){
										U_i[RHO] = (cmax_roe - state_r.ucon[i1] / state_r.ucon[0]) / (cmax_roe - vcon[i1] + SMALL)*U_r[RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[i1] * qi.bcov[0] - F_r[i1][UU] + cmax_roe*U_r[UU]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U2] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[i1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[i1] - qi.bcon[i1] * qi.bcov[3] - F_r[i1][U3] + cmax_roe*U_r[U3]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U1] = (qi.bcon[0] * qi.bcov[1] * vcon[i1] - qi.bcon[i1] * qi.bcov[1] - F_r[i1][U1] + cmax_roe*U_r[U1]) / (cmax_roe - vcon[i1] + SMALL);
									}
									else if (i1 == 3){
										U_i[RHO] = (cmax_roe - state_r.ucon[i1] / state_r.ucon[0]) / (cmax_roe - vcon[i1] + SMALL)*U_r[RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[i1] * qi.bcov[0] - F_r[i1][UU] + cmax_roe*U_r[UU]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U3] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[i1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U1] = (qi.bcon[0] * qi.bcov[1] * vcon[i1] - qi.bcon[i1] * qi.bcov[1] - F_r[i1][U1] + cmax_roe*U_r[U1]) / (cmax_roe - vcon[i1] + SMALL);
										U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[i1] - qi.bcon[i1] * qi.bcov[2] - F_r[i1][U2] + cmax_roe*U_r[U2]) / (cmax_roe - vcon[i1] + SMALL);
									}

									//Calculate HLLC flux
									for (k = 0; k < B1; k++) F_FT[i1][k] = (F_r[i1][k] + cmax_roe*(U_i[k] - F_r[0][k]));
								}
								else if (cmin_roe<0. && qi.ucon[i1] >= 0. && fail_HLLC == 0){
									//Set Rankine-Hugoniot jump conditions
									if (i1 == 1){
										U_i[RHO] = (cmin_roe - state_l.ucon[i1] / state_l.ucon[0]) / (cmin_roe - vcon[i1] + SMALL)*F_l[0][RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[1] * qi.bcov[0] - F_l[i1][UU] + cmin_roe*F_l[0][UU]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U1] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[i1] - qi.bcon[i1] * qi.bcov[2] - F_l[i1][U2] + cmin_roe*F_l[0][U2]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[i1] - qi.bcon[i1] * qi.bcov[3] - F_l[i1][U3] + cmin_roe*F_r[0][U3]) / (cmin_roe - vcon[i1] + SMALL);
									}
									else if (i1 == 2){
										U_i[RHO] = (cmin_roe - state_l.ucon[i1] / state_l.ucon[0]) / (cmin_roe - vcon[i1] + SMALL)*F_l[0][RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[i1] * qi.bcov[0] - F_l[i1][UU] + cmin_roe*F_l[0][UU]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U2] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[i1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U3] = (qi.bcon[0] * qi.bcov[3] * vcon[i1] - qi.bcon[i1] * qi.bcov[3] - F_l[i1][U3] + cmin_roe*F_l[0][U3]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U1] = (qi.bcon[0] * qi.bcov[1] * vcon[i1] - qi.bcon[i1] * qi.bcov[1] - F_l[i1][U1] + cmin_roe*F_r[0][U1]) / (cmin_roe - vcon[i1] + SMALL);
									}
									else if (i1 == 3){
										U_i[RHO] = (cmin_roe - state_l.ucon[i1] / state_l.ucon[0]) / (cmin_roe - vcon[i1] + SMALL)*F_l[0][RHO];
										U_i[UU] = ((qi.bcon[0] * qi.bcov[0] - ptot)* vcon[i1] - qi.bcon[i1] * qi.bcov[0] - F_l[i1][UU] + cmin_roe*F_l[0][UU]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U3] = (U_i[UU] - ptot + qi.bcon[0] * qi.bcov[0])*qi.ucov[i1] / qi.ucov[0] - qi.bcon[0] * qi.bcov[i1];
										U_i[U1] = (qi.bcon[0] * qi.bcov[1] * vcon[i1] - qi.bcon[i1] * qi.bcov[1] - F_l[i1][U1] + cmin_roe*F_l[0][U1]) / (cmin_roe - vcon[i1] + SMALL);
										U_i[U2] = (qi.bcon[0] * qi.bcov[2] * vcon[i1] - qi.bcon[i1] * qi.bcov[2] - F_l[i1][U2] + cmin_roe*F_r[0][U2]) / (cmin_roe - vcon[i1] + SMALL);
									}

									//Calculate HLLC flux
									for (k = 0; k < B1; k++) F_FT[i1][k] = (F_l[i1][k] + cmin_roe*(U_i[k] - F_l[0][k]));
								}
								else if (cmin_roe >= 0. && fail_HLLC == 0){
									for (k = 0; k < B1; k++) F_FT[i1][k] = F_l[i1][k];
								}
								else if (cmax_roe <= 0. && fail_HLLC == 0){
									for (k = 0; k < B1; k++) F_FT[i1][k] = F_r[i1][k];
								}
								else{ //revert to HLL flux
									for (k = 0; k < B1; k++){
										F_FT[i1][k] = HLLF*((cmax[i1] * F_l[i1][k] + cmin[i1] * F_r[i1][k] - cmax[i1] * cmin[i1] * (F_r[0][k] - F_l[0][k])) / (cmax[i1] + cmin[i1] + SMALL))
											+ LAXF*(0.5*(F_l[i1][k] + F_r[i1][k] - ctop[i1] * (F_r[0][k] - F_l[0][k])));
									}
									fail_HLLC = 0;
								}
							}
							else{//Magnetic field too weak for HLLC solver
#pragma ivdep
								for (k = 0; k < B1; k++){
									F_FT[i1][k] = HLLF*((cmax[i1] * F_l[i1][k] + cmin[i1] * F_r[i1][k] - cmax[i1] * cmin[i1] * (F_r[0][k] - F_l[0][k])) / (cmax[i1] + cmin[i1] + SMALL))
										+ LAXF*(0.5*(F_l[i1][k] + F_r[i1][k] - ctop[i1] * (F_r[0][k] - F_l[0][k])));
								}
				}
			}
						#else
						cmax_roe = MY_MAX(cmax_r, cmax_l);
						cmin_roe = MY_MIN(cmin_r, cmin_l);
						double int_velocity = geom.gcon[0][dir] / (sqrt(geom.gcon[0][dir] * geom.gcon[0][dir] - geom.gcon[0][0] * geom.gcon[dir][dir]));
						for (i1 = 0; i1 < NDIM; i1++)for (k = 0; k < B1; k++) F_FT[i1][k] = 0.;

						if (cmax_roe > int_velocity && cmin_roe < int_velocity) for (k = 0; k < B1; k++) F_FT[0][k] = (F_l[k] - F_r[k] + cmax_roe*U_r[k] - cmin_roe*U_l[k]) / (cmax_roe - cmin_roe + SMALL);
						else if (cmax_roe < int_velocity) for (k = 0; k < B1; k++) F_FT[0][k] = U_r[k];
						else for (k = 0; k < B1; k++) F_FT[0][k] = U_l[k];

						//for (k = 0; k < B1; k++){
						//	F_FT[dir][k] = HLLF*((cmax * F_l[k] + cmin * F_r[k] - cmax * cmin * (U_r[k] - U_l[k])) / (cmax + cmin + SMALL))
						//		+ LAXF*(0.5*(F_l[k] + F_r[k] - ctop * (U_r[k] - U_l[k])));
						//}
						if (cmax_roe > int_velocity && cmin_roe < int_velocity) for (k = 0; k < B1; k++) F_FT[dir][k] = HLLF*((cmax_roe * F_l[k] - cmin_roe * F_r[k] + cmax_roe * cmin_roe * (U_r[k] - U_l[k])) / (cmax_roe - cmin_roe + SMALL));
						else if (cmax_roe < int_velocity) for (k = 0; k < B1; k++) F_FT[dir][k] = F_r[k];
						else for (k = 0; k < B1; k++) F_FT[dir][k] = F_l[k];
						#endif
						//Transform from orthonormal frame to coordinate basis
						for (j1 = 0; j1<NDIM; j1++){
							F1[dir][j1 + UU] = 0.;
							for (i2 = 0; i2<NDIM; i2++) {
								for (j2 = 0; j2<NDIM; j2++){
									F1[dir][j1 + UU] += F_FT[i2][j2 + UU] * trans.Mud[dir][i2] * trans.Mud_inv[j2][j1];
								}
							}
						}

						//Transform wave from orthonormal basis to coordinate basis
						F1[dir][RHO] = 0.;
						for (j1 = 0; j1 < NDIM; j1++){
							F1[dir][RHO] += F_FT[j1][RHO] * trans.Mud[dir][j1];
						}

						F1[dir][UU] += F1[dir][RHO];
						PLOOP F[nl[n]][ind0][k] = F1[dir][k];
						#if(!TRANS_BOUND)
						if (dir == 2 && (j == 0 || j == N2 * pow(1 + REF_2, block[n][AMR_LEVEL]))) {
							//#pragma ivdep
							PLOOP F[nl[n]][ind0][k] = 0.;
						}
						#endif
				}
		}
		#pragma omp critical
		{
			if (ndt_thread < ndt){
				ndt = ndt_thread;
			}
		}
	}
	return(ndt);
}

void ctop_to_utop(double ctop[NDIM], double cmax[NDIM]){
	#if(HLLC)
	double a = -1 + ctop[1] * ctop[1] + ctop[2] * ctop[2] + ctop[3] * ctop[3];

	cmax[0] = 1. / sqrt(fabs(a));
	cmax[1] = ctop[1] * cmax[0];
	cmax[2] = ctop[2] * cmax[0];
	cmax[3] = ctop[3] * cmax[0];
	#endif
}

void primtoflux_FT(double * restrict pr, struct of_state * restrict q, int dir,struct of_geom * restrict geom, double restrict flux[NPR])
{
	#if(HLLC)
	int j, k;
	double mhd[NDIM];
	double r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];
	P = (gam - 1.)*u;
	w = P + r + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	ptot = P + 0.5*bsq;

	/* particle number flux */
	flux[RHO] = pr[RHO] * q->ucon[dir];

	/* single row of mhd stress tensor,
	* first index up, second index down */
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];

	/* MHD stress-energy tensor w/ first index up,
	* second index down. */
	#pragma ivdep
	for (k = 0; k < 4; k++){
		flux[k + 1] = mhd[k];
	}

	PLOOP flux[k] *= geom->g;
	#endif
}

void vchar_FT(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int dir, double  restrict *vmax, double restrict *vmin, int n, int a, int b, int c)
{
	#if(HLLC)
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2, rho, u;
	double Acon_0, Acon_js, Bcon_0, Bcon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	if (dir == 1){
		Acon_0 = 0;
		Acon_js = 1.;
	}
	else if (dir == 2){
		Acon_0 =0.;
		Acon_js = 1.;
	}
	else if (dir == 3){
		Acon_0 = 0.;
		Acon_js = 1.;
	}

	/* find fast magnetosonic speed */
	bsq = dot(q->bcon, q->bcov);
	rho = pr[RHO];
	u = pr[UU];
	#if AMD
	EF = fma(gam, u, rho);
	#else
	EF = rho + gam*u;
	#endif
	EE = bsq + EF;
	cs2 = gam*(gam - 1.)*u / EF;
	va2 = bsq / EE;



	/* find fast magnetosonic speed */
	cs2 = gam*(gam - 1.)*pr[UU] / EF;
	va2 = bsq / EE;
	cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

	/* check on it! */
	if (cms2 < 0.) {
		//fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL;
	}
	if (cms2 > 1.) {
		//fail(FAIL_COEFF_SUP) ;
		cms2 = 1.;
	}

	/* now require that speed of wave measured by observer
	q->ucon is cms2 */
	Asq = Acon_js;
	Bsq = geom->gcon[0][0];// dot(Bcon, Bcov);
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	AB = Acon_0;
	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;

	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2.* fma(-(AB + AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4.*A*C);
	#else
	A = Bu2 - (Bsq + Bu2)*cms2;
	B = 2.*(AuBu - (AB + AuBu)*cms2);
	C = Au2 - (Asq + Au2)*cms2;
	discr = B*B - 4.*A*C;
	#endif
	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		/*fprintf(stderr,"\n\t %g %g %g %g %g\n",A,B,C,discr,cms2) ;
		fprintf(stderr,"\n\t q->ucon: %g %g %g %g\n",q->ucon[0],q->ucon[1],
		q->ucon[2],q->ucon[3]) ;
		fprintf(stderr,"\n\t q->bcon: %g %g %g %g\n",q->bcon[0],q->bcon[1],
		q->bcon[2],q->bcon[3]) ;
		fprintf(stderr,"\n\t Acon: %g %g %g %g\n",Acon[0],Acon[1],
		Acon[2],Acon[3]) ;
		fprintf(stderr,"\n\t Bcon: %g %g %g %g\n",Bcon[0],Bcon[1],
		Bcon[2],Bcon[3]) ;
		fail(FAIL_VCHAR_DISCR) ;*/
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	#if( FULL_DISP ) 
	double vp2, vm2;
	vp2 = NewtonRaphson(vp, 1, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vm2 = NewtonRaphson(vm, 1, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vp = vp2;
	vm = vm2;
	#endif

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
}
	else {
		*vmax = vm;
		*vmin = vp;
	}

	return;
	#endif
}
