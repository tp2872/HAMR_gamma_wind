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

#include "decs.h"

/***********************************************************************************************/
/***********************************************************************************************
  primtoflux():
  ---------
   --  calculate fluxes in direction dir, 
        
***********************************************************************************************/

void primtoflux_res(double * restrict pr, struct of_state_res * restrict q_res, int dir, struct of_geom * restrict geom, double * restrict flux)
{
	int k;
	double alpha, beta[NDIM], Ecov[3], Bcov[3];

	/* particle number flux */
	flux[RHO] = pr[RHO]*q_res->ucon[dir] ;

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc_res(pr, dir, geom, q_res, &flux[UU]) ;
	flux[UU] += flux[RHO];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//Beta in 3+1
	beta[0] = 0;
	beta[1] = geom->gcon[0][1] * alpha * alpha;
	beta[2] = geom->gcon[0][2] * alpha * alpha;
	beta[3] = geom->gcon[0][3] * alpha * alpha;

	/*Maxwell tensor */
	lower_3(&pr[E1], geom, Ecov);

	#pragma ivdep
	if (dir == 0) {
		flux[E1] = pr[E1];
		flux[E2] = pr[E2];
		flux[E3] = pr[E3];
	}
	else {
		flux[E1] = 0.;
		flux[E2] = 0.;
		flux[E3] = 0.;
		for (k = 1; k <= 3; k++) {
			flux[E1] += beta[1] * pr[E1 + (dir - 1)] - beta[dir] * beta[1] - (alpha * alpha / geom->g) * lvc3u(dir, 1, k) * Ecov[k - 1];
			flux[E2] += beta[2] * pr[E1 + (dir - 1)] - beta[dir] * beta[2] - (alpha * alpha / geom->g) * lvc3u(dir, 2, k) * Ecov[k - 1];
			flux[E3] += beta[3] * pr[E1 + (dir - 1)] - beta[dir] * beta[3] - (alpha * alpha / geom->g) * lvc3u(dir, 3, k) * Ecov[k - 1];

		}
	}

	/* dual of Maxwell tensor */
	lower_3(&pr[B1], geom, Bcov);

	#pragma ivdep
	if (dir == 0) {
		flux[B1] = pr[B1];
		flux[B2] = pr[B2];
		flux[B3] = pr[B3];
	}
	else {
		flux[E1] = 0.;
		flux[E2] = 0.;
		flux[E3] = 0.;
		for (k = 1; k <= 3; k++) {
			flux[B1] += beta[1] * pr[B1 + (dir - 1)] - beta[dir] * beta[1] + (alpha * alpha / geom->g) * lvc3u(dir, 1, k) * Bcov[k - 1];
			flux[B2] += beta[2] * pr[B1 + (dir - 1)] - beta[dir] * beta[2] + (alpha * alpha / geom->g) * lvc3u(dir, 2, k) * Bcov[k - 1];
			flux[B3] += beta[3] * pr[B1 + (dir - 1)] - beta[dir] * beta[3] + (alpha * alpha / geom->g) * lvc3u(dir, 3, k) * Bcov[k - 1];
		}
	}

	//Entropy advection
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * (GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA);
	#endif
    
	#pragma ivdep
	PLOOP flux[k] *= geom->g ;
}

/* calculate magnetic field four-vector */
void econ_calc_res(double* restrict pr, struct of_geom* restrict geom, double* restrict ucon, double* restrict bcon)
{
	double alpha, gamma, ncon[NDIM], E_dot_v, utcov[3], Bcov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[0][1] * alpha;
	ncon[2] = -geom->gcon[0][2] * alpha;
	ncon[3] = -geom->gcon[0][3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector
	lower_3(&(pr[U1]), geom, utcov);
	lower_3(&(pr[B1]), geom, Bcov);
	E_dot_v = pr[E1] * utcov[0] + pr[E2] * utcov[1] + pr[E3] * utcov[2];

	//Final calculation of rest frame magnetic field
	bcon[0] = alpha * (E_dot_v)*ncon[0];
	bcon[1] = alpha * (E_dot_v)*ncon[1] + gamma * (alpha * pr[E1]) - (alpha / geom->g) * (utcov[1] * Bcov[2] - utcov[2] * Bcov[1]);
	bcon[2] = alpha * (E_dot_v)*ncon[2] + gamma * (alpha * pr[B2]) - (alpha / geom->g) * (utcov[2] * Bcov[0] - utcov[0] * Bcov[2]);
	bcon[3] = alpha * (E_dot_v)*ncon[3] + gamma * (alpha * pr[B3]) - (alpha / geom->g) * (utcov[0] * Bcov[1] - utcov[1] * Bcov[0]);

	return;
}

/* calculate magnetic field four-vector */
void bcon_calc_res(double * restrict pr, struct of_geom* restrict geom, double* restrict ucon, double * restrict bcon)
{
	double alpha, gamma, ncon[NDIM], B_dot_v, utcov[3], Ecov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[0][1] * alpha;
	ncon[2] = -geom->gcon[0][2] * alpha;
	ncon[3] = -geom->gcon[0][3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector
	lower_3(&(pr[U1]), geom, utcov);
	lower_3(&(pr[E1]), geom, Ecov);
	B_dot_v = pr[B1] * utcov[0] + pr[B2] * utcov[1] + pr[B3] * utcov[2];

	//Final calculation of rest frame magnetic field
	bcon[0] = alpha * (B_dot_v) * ncon[0];
	bcon[1] = alpha * (B_dot_v) * ncon[1] + gamma * (alpha * pr[B1]) - (alpha / geom->g) * (utcov[1] * Ecov[2] - utcov[2] * Ecov[1]);
	bcon[2] = alpha * (B_dot_v) * ncon[2] + gamma * (alpha * pr[B2]) - (alpha / geom->g) * (utcov[2] * Ecov[0] - utcov[0] * Ecov[2]);
	bcon[3] = alpha * (B_dot_v) * ncon[3] + gamma * (alpha * pr[B3]) - (alpha / geom->g) * (utcov[0] * Ecov[1] - utcov[1] * Ecov[0]);

	return ;
}

/* MHD stress tensor, with first index up, second index down */
void mhd_calc_res(double * restrict pr, int dir, struct of_geom* restrict geom, struct of_state_res * restrict q_res, double * restrict mhd)
{
	int j, lambda, beta, kappa;
	double P,w,bsq,esq, eta,ptot, mhd_u[NDIM], mhd_d[NDIM], alpha;

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//Calculate contraction term
	DLOOPA{
		for (lambda = 0; lambda < 4; lambda++)for (beta = 0; beta < 4; beta++)for (kappa = 0; kappa < 4; kappa++) {
			mhd_u[j] = -q_res->ucov[lambda] * q_res->ecov[beta] * q_res->bcov[kappa] * (q_res->ucon[dir] * (alpha / geom->g) * lvc4u(j, lambda, beta, kappa) + q_res->ucon[dir] * (alpha / geom->g) * lvc4u(j, lambda, beta, kappa));
		}
	}
	lower(mhd_u, geom, mhd_d);

    #if DOHELM
    // Helmholtz EOS
    eos_mode_rhou_pres (pr[RHO], pr[UU], &P);
    #else
    // Ideal gas EOS
	P = (GAMMA - 1.) * pr[UU];
    #endif
    
    w = P + pr[RHO] + pr[UU];
	bsq = dot(q_res->bcon, q_res->bcov) ;
	esq = dot(q_res->econ, q_res->ecov);
	eta = w + bsq + esq;
	ptot = P + 0.5*(bsq + esq);

	//single row of mhd stress tensor, first index up, second index down
	#pragma ivdep
	DLOOPA mhd[j] = eta * q_res->ucon[dir] * q_res->ucov[j] + ptot * delta(dir, j) - q_res->bcon[dir] * q_res->bcov[j] - q_res->econ[dir] * q_res->ecov[j]+mhd_d[j];
}

/* add in (explicit) geometricc source terms to equations of motion */
void source_res(double * restrict ph,  struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU, double Dt)
{
    double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg, J[NDIM], beta[NDIM], alpha, gamma, utcov[3], Bcov[3], vdotE, q;
	int j,k ;
	struct of_state_res q_res;

	get_state_res(ph, geom, &q_res) ;
	mhd_calc_res(ph, 0, geom, &q_res, mhd[0]) ;
	mhd_calc_res(ph, 1, geom, &q_res, mhd[1]) ;
	mhd_calc_res(ph, 2, geom, &q_res, mhd[2]) ;
	mhd_calc_res(ph, 3, geom, &q_res, mhd[3]) ;

	#pragma ivdep
	PLOOP dU[k] = 0.;
	
	//contract mhd stress tensor with connection
	DLOOP {
		dU[UU] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][0][j];
		dU[U1] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][1][j];
		dU[U2] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][2][j];
		dU[U3] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][3][j];
	}

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//Beta in 3+1
	beta[1] = geom->gcon[0][1] * alpha * alpha;
	beta[2] = geom->gcon[0][2] * alpha * alpha;
	beta[3] = geom->gcon[0][3] * alpha * alpha;

	//Calculate charge density from divergence of electric field
	q = (alpha / geom->g) * divE_calc(ph, n, ii, jj, zz); //Calculate charge density first

	//Calculate electric current J
	gamma = q_res.ucon[0] * alpha;
	lower_3(&(ph[U1]), geom, utcov);
	lower_3(&(ph[B1]), geom, Bcov);
	vdotE = alpha / gamma * (ph[E1] * utcov[0]+ ph[E2] * utcov[1] + ph[E3] * utcov[2]);
	J[1] = q * ph[U1] / gamma + gamma / ETA * (ph[E1] * alpha + alpha / geom->g * (utcov[1] * Bcov[2] - utcov[2] * Bcov[1]) / gamma - vdotE * ph[U1] / gamma);
	J[2] = q * ph[U2] / gamma + gamma / ETA * (ph[E2] * alpha + alpha / geom->g * (utcov[2] * Bcov[0] - utcov[0] * Bcov[2]) / gamma - vdotE * ph[U2] / gamma);
	J[3] = q * ph[U3] / gamma + gamma / ETA * (ph[E3] * alpha + alpha / geom->g * (utcov[0] * Bcov[1] - utcov[1] * Bcov[0]) / gamma - vdotE * ph[U3] / gamma);

	//Calculate source term for electric field
	dU[E1] = -J[1] + beta[1] * q / alpha;
	dU[E2] = -J[2] + beta[2] * q / alpha;
	dU[E3] = -J[3] + beta[3] * q / alpha;

	//Add disk cooling term
	#if(COOL_DISK)
	double X[NDIM], r, th, phi;
	coord(n, ii, jj, zz, CENT, X);
	bl_coord(X, &r, &th, &phi);
	misc_source(ph, ii, jj, geom, &q, dU, r, Dt);
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g ;
}

void calc_J(double p[NPR], double J[NDIM], double q, struct of_geom* restrict geom) {
	struct of_state_res q_res;
	get_state_res(ph, geom, &q_res);

}

/* find ucon, ucov, bcon, bcov from primitive variables */
void get_state_res(double * restrict pr, struct of_geom * restrict geom, struct of_state_res * restrict q_res)
{
	//get ucon
	ucon_calc(pr, geom, q_res->ucon) ;
	lower(q_res->ucon, geom, q_res->ucov) ;

	//get bcon
	bcon_calc_res(pr, q_res->ucon, q_res->ucov, q_res->bcon) ;
	lower(q_res->bcon, geom, q_res->bcov) ;

	//get econ
	econ_calc_res(pr, q_res->ucon, q_res->ucov, q_res->econ);
	lower(q_res->econ, geom, q_res->ecov);

	return ;
}

void vchar_res( struct of_geom * restrict geom, int js,double * restrict vmax, double * restrict vmin){
	*vmax = sqrt(geom->gcon[js][js]);
	*vmin = -sqrt(geom->gcon[js][js]);

	return ;
}

double divE_calc(double(*restrict p[NB_LOCAL])[NPR],  int n, int i, int j, int z) {
	int di = (N1 > 1);
	int dj = (N2 > 1);
	int dz = (N3 > 1);
	double dive = 0.0;
	int zsize = 1, zoffset = 0, zlevel = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0 && GPU_ENABLED==1)
	if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && j < N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n]) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if ((block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) && j >= N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = round(pow(2.0, (double)zlevel));
	zoffset = (z - N3_GPU_offset[n]) % zsize;
	#endif

	/* Constrained transport defn */
	#if(STAGGERED_E)
	#if(N1>1)
	for (u = 0; u < zsize; u++) {
		dive += 0.25 * (pse[nl[n]][index_3D(n, i + di, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i + di, j, z - zoffset + u)][FACE1] - pse[nl[n]][index_3D(n, i, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE1]) / ((double)(zsize)*dx[nl[n]][1]);
	}
	#endif
	#if(N2>1)
	for (u = 0; u < zsize; u++) {
		dive += 0.25 * (pse[nl[n]][index_3D(n, i, j + dj, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j + dj, z - zoffset + u)][FACE2] - pse[nl[n]][index_3D(n, i, j, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE2]) / ((double)(zsize)*dx[nl[n]][2]);
	}
	#endif
	#if(N3>1)
	dive += 0.25 * (pse[nl[n]][index_3D(n, i, j, z - zoffset + dz * zsize)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + dz * zsize)][FACE3] - pse[nl[n]][index_3D(n, i, j, z - zoffset)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset)][FACE3]) / ((double)(zsize)*dx[nl[n]][3]);
	#endif
	dive = fabs(divb);
	#else

	/* Flux-ct defn */
	dive = fabs(
		#if(N1>1)
		0.25 * (
			+ p[nl[n]][index_3D(n, i, j, z)][E1] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
			+ p[nl[n]][index_3D(n, i, j, z - dz)][E1] * gdet[nl[n]][index_2D(n, i, j, z - dz)][CENT]
			+ p[nl[n]][index_3D(n, i, j - dj, z)][E1] * gdet[nl[n]][index_2D(n, i, j - dj, z)][CENT]
			+ p[nl[n]][index_3D(n, i, j - dj, z - dz)][E1] * gdet[nl[n]][index_2D(n, i, j - dj, z - dz)][CENT]
			- p[nl[n]][index_3D(n, i - 1, j, z)][E1] * gdet[nl[n]][index_2D(n, i - 1, j, z)][CENT]
			- p[nl[n]][index_3D(n, i - 1, j, z - dz)][E1] * gdet[nl[n]][index_2D(n, i - 1, j, z - dz)][CENT]
			- p[nl[n]][index_3D(n, i - 1, j - dj, z)][E1] * gdet[nl[n]][index_2D(n, i - 1, j - dj, z)][CENT]
			- p[nl[n]][index_3D(n, i - 1, j - dj, z - dz)][E1] * gdet[nl[n]][index_2D(n, i - 1, j - dj, z - dz)][CENT]
			) / ((double)(zsize)*dx[nl[n]][1])
		#endif
		#if(N2>1)
		+ 0.25 * (
			+ p[nl[n]][index_3D(n, i, j, z)][E2] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
			+ p[nl[n]][index_3D(n, i, j, z - dz)][E2] * gdet[nl[n]][index_2D(n, i, j, z - dz)][CENT]
			+ p[nl[n]][index_3D(n, i - di, j, z)][E2] * gdet[nl[n]][index_2D(n, i - di, j, z)][CENT]
			+ p[nl[n]][index_3D(n, i - di, j, z - dz)][E2] * gdet[nl[n]][index_2D(n, i - di, j, z - dz)][CENT]
			- p[nl[n]][index_3D(n, i, j - 1, z)][E2] * gdet[nl[n]][index_2D(n, i, j - 1, z)][CENT]
			- p[nl[n]][index_3D(n, i, j - 1, z - dz)][E2] * gdet[nl[n]][index_2D(n, i, j - 1, z - dz)][CENT]
			- p[nl[n]][index_3D(n, i - di, j - 1, z)][E2] * gdet[nl[n]][index_2D(n, i - di, j - 1, z)][CENT]
			- p[nl[n]][index_3D(n, i - di, j - 1, z - dz)][E2] * gdet[nl[n]][index_2D(n, i - di, j - 1, z - dz)][CENT]
			) / ((double)(zsize)*dx[nl[n]][2])
		#endif
		#if(N3>1)
		+ 0.25 * (
			+ p[nl[n]][index_3D(n, i, j, z)][E3] * gdet[nl[n]][index_2D(n, i, j, z)][CENT]
			+ p[nl[n]][index_3D(n, i - di, j, z)][E3] * gdet[nl[n]][index_2D(n, i - di, j, z)][CENT]
			+ p[nl[n]][index_3D(n, i, j - dj, z)][E3] * gdet[nl[n]][index_2D(n, i, j - dj, z)][CENT]
			+ p[nl[n]][index_3D(n, i - di, j - dj, z)][E3] * gdet[nl[n]][index_2D(n, i - di, j - dj, z)][CENT]
			- p[nl[n]][index_3D(n, i, j, z - 1)][E3] * gdet[nl[n]][index_2D(n, i, j, z - 1)][CENT]
			- p[nl[n]][index_3D(n, i - di, j, z - 1)][E3] * gdet[nl[n]][index_2D(n, i - di, j, z - 1)][CENT]
			- p[nl[n]][index_3D(n, i, j - dj, z - 1)][E3] * gdet[nl[n]][index_2D(n, i, j - dj, z - 1)][CENT]
			- p[nl[n]][index_3D(n, i - di, j - dj, z - 1)][E3] * gdet[nl[n]][index_2D(n, i - di, j - dj, z - 1)][CENT]
			) / ((double)(zsize)*dx[nl[n]][3])
		#endif
	);
	#endif
	return dive;
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
void lower_3(double* restrict ucon, struct of_geom* restrict geom, double* restrict ucov)
{
	int i, j;
	for (i = 0; i < 3; i++) {
		ucov[i] = 0.0;
	}
	for (j = 0; j < 3; j++) {
		for (i = 0; i < 3; i++) {
			ucov[i] += geom->gcov[i + 1][j + 1] * ucon[j];
		}
	}
	return;
}

double lvc4u(int i, int j, int k, int l) {
	double lvc4u;

	if ((i + j + k + l) != 6) {
		lvc4u = 0.0;
	}
	else if ((i + j == 1) || (i + j == 5)) {
		if ((j + k) % 4 == 3) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if ((i + j == 2) || (i + j == 4)) {
		if ((j + k) % 4 == 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if (i + j == 3) {
		if ((j + k) % 4 != 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else lvc4u = 0.0;

	return (-lvc4u);
}

double lvc3u(int i, int j, int k) {
	double lvc3u;
			
	if (i > 0 && j > 0) {
		if ((i == j) || (j == k) || (k == i)) lvc3u = 0.;
		else if ((i + 1 == j) || (i - 2 == j)) lvc3u = 1.;
		else lvc3u = -1;
	}

	return (lvc3u);
}