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
	#if(RESISTIVE)
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
	lower_3(&(pr[B1]), geom, Bcov);
	if (dir == 0) {
		flux[E1] = pr[E1];
		flux[E2] = pr[E2];
		flux[E3] = pr[E3];
	}
	else {
		flux[E1] = beta[1] * pr[E1 + (dir - 1)] - beta[dir] * pr[E1];
		flux[E2] = beta[2] * pr[E1 + (dir - 1)] - beta[dir] * pr[E2];
		flux[E3] = beta[3] * pr[E1 + (dir - 1)] - beta[dir] * pr[E3];
		for (k = 0; k < 3; k++) {
			flux[E1] -= lvc3u(0, dir - 1, k) * (alpha * alpha / (geom->g)) * (Bcov[k]);
			flux[E2] -= lvc3u(1, dir - 1, k) * (alpha * alpha / (geom->g)) * (Bcov[k]);
			flux[E3] -= lvc3u(2, dir - 1, k) * (alpha * alpha / (geom->g)) * (Bcov[k]);
		}
	}
	
	/* dual of Maxwell tensor */
	lower_3(&(pr[E1]), geom, Ecov);
	if (dir == 0) {
		flux[B1] = pr[B1];
		flux[B2] = pr[B2];
		flux[B3] = pr[B3];
	}
	else {
		flux[B1] = beta[1] * pr[B1 + (dir - 1)] - beta[dir] * pr[B1];
		flux[B2] = beta[2] * pr[B1 + (dir - 1)] - beta[dir] * pr[B2];
		flux[B3] = beta[3] * pr[B1 + (dir - 1)] - beta[dir] * pr[B3];
		for (k = 0; k < 3; k++) {
			flux[B1] += lvc3u(0, dir - 1, k) * (alpha * alpha / (geom->g)) * (Ecov[k]);
			flux[B2] += lvc3u(1, dir - 1, k) * (alpha * alpha / (geom->g)) * (Ecov[k]);
			flux[B3] += lvc3u(2, dir - 1, k) * (alpha * alpha / (geom->g)) * (Ecov[k]);
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
	#endif
}

/* calculate magnetic field four-vector */
void econ_calc_res(double* restrict pr, struct of_geom* restrict geom, double* restrict ucon, double* restrict ucov, double* restrict econ)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], E_dot_v,  Bcov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[0][1] * alpha;
	ncon[2] = -geom->gcon[0][2] * alpha;
	ncon[3] = -geom->gcon[0][3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA!
	lower_3(&(pr[B1]), geom, Bcov);
	E_dot_v = pr[E1] * ucov[1] + pr[E2] * ucov[2] + pr[E3] * ucov[3];

	//Final calculation of rest frame magnetic field
	econ[0] = alpha * (E_dot_v)*ncon[0];
	econ[1] = alpha * (E_dot_v)*ncon[1] + gamma * (alpha * pr[E1]) + (alpha * alpha / geom->g) * (ucov[2] * Bcov[2] - ucov[3] * Bcov[1]);
	econ[2] = alpha * (E_dot_v)*ncon[2] + gamma * (alpha * pr[E2]) + (alpha * alpha / geom->g) * (ucov[3] * Bcov[0] - ucov[1] * Bcov[2]);
	econ[3] = alpha * (E_dot_v)*ncon[3] + gamma * (alpha * pr[E3]) + (alpha * alpha / geom->g) * (ucov[1] * Bcov[1] - ucov[2] * Bcov[0]);

	return;
	#endif
}

/* calculate magnetic field four-vector */
void bcon_calc_res(double * restrict pr, struct of_geom* restrict geom, double* restrict ucon, double* restrict ucov, double * restrict bcon)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], B_dot_v, Ecov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[0][1] * alpha;
	ncon[2] = -geom->gcon[0][2] * alpha;
	ncon[3] = -geom->gcon[0][3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA
	lower_3(&(pr[E1]), geom, Ecov);
	B_dot_v = pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3];

	//Final calculation of rest frame magnetic field
	bcon[0] = alpha * (B_dot_v) * ncon[0];
	bcon[1] = alpha * (B_dot_v)*ncon[1] + gamma * (alpha * pr[B1]) - (alpha * alpha / geom->g) * (ucov[2] * Ecov[2] - ucov[3] * Ecov[1]);
	bcon[2] = alpha * (B_dot_v)*ncon[2] + gamma * (alpha * pr[B2]) - (alpha * alpha / geom->g) * (ucov[3] * Ecov[0] - ucov[1] * Ecov[2]);
	bcon[3] = alpha * (B_dot_v)*ncon[3] + gamma * (alpha * pr[B3]) - (alpha * alpha / geom->g) * (ucov[1] * Ecov[1] - ucov[2] * Ecov[0]);

	return;
	#endif
}

/* MHD stress tensor, with first index up, second index down */
void mhd_calc_res(double * restrict pr, int dir, struct of_geom* restrict geom, struct of_state_res * restrict q_res, double * restrict mhd)
{
	#if(RESISTIVE)
	int j, lambda, beta, kappa;
	double P,w,bsq,esq, eta,ptot, mhd_u[NDIM], mhd_d[NDIM], alpha;


	//Lapse in 3+1
	alpha = 1.0;

	//Calculate contraction term
	DLOOPA{
		mhd_u[j] = 0.;
		for (lambda = 0; lambda < NDIM; lambda++)for (beta = 0; beta < NDIM; beta++)for (kappa = 0; kappa < NDIM; kappa++) {
			mhd_u[j] += q_res->ucov[lambda] * q_res->ecov[beta] * q_res->bcov[kappa] * (q_res->ucon[dir] * (1.0 / geom->g) * lvc4u(j, lambda, beta, kappa) + q_res->ucon[j] * (1.0 / geom->g) * lvc4u(dir, lambda, beta, kappa));
		}
	}
	lower(mhd_u, geom, mhd_d);

    #if DOHELM   
    eos_mode_rhou_pres (pr[RHO], pr[UU], &P); // Helmholtz EOS
    #else
	P = (GAMMA - 1.) * pr[UU]; // Ideal gas EOS
    #endif
    
    w = P + pr[RHO] + pr[UU];
	bsq = dot(q_res->bcon, q_res->bcov) ;
	esq = dot(q_res->econ, q_res->ecov);
	eta = w + (bsq + esq);
	ptot = P + 0.5*(bsq + esq);

	//single row of mhd stress tensor, first index up, second index down
	#pragma ivdep
	DLOOPA mhd[j] = eta * q_res->ucon[dir] * q_res->ucov[j] + ptot * delta(dir, j) - q_res->bcon[dir] * q_res->bcov[j] - q_res->econ[dir] * q_res->ecov[j] + mhd_d[j];
	#endif
}

/* add in (explicit) geometricc source terms to equations of motion */
void source_res(double * restrict ph,  struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU, double *q, double Dt)
{
	#if(RESISTIVE)
	double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg, J[NDIM], beta[NDIM], alpha, gamma, vdotE, q_local;
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
	
	//Implicit source term; do not use in this function
	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0][0]);

	//Beta in 3+1
	beta[1] = geom->gcon[0][1] * alpha * alpha;
	beta[2] = geom->gcon[0][2] * alpha * alpha;
	beta[3] = geom->gcon[0][3] * alpha * alpha;

	//Calculate charge density from divergence of electric field
	if(ETA<0.000001)q_local = 0.;
	else q_local = (alpha / geom->g) * q[0];

	//Calculate explicit part of electric current J
	gamma = q_res.ucon[0] * alpha;
	vdotE = alpha / gamma * (ph[E1] * q_res.ucov[1]+ ph[E2] * q_res.ucov[2] + ph[E3] * q_res.ucov[3]);
	J[1] = q_local * ph[U1] / gamma;
	J[2] = q_local * ph[U2] / gamma;
	J[3] = q_local * ph[U3] / gamma;

	//Calculate source term for electric field
	dU[E1] = -alpha * J[1] + beta[1] * q_local / alpha;
	dU[E2] = -alpha * J[2] + beta[2] * q_local / alpha;
	dU[E3] = -alpha * J[3] + beta[3] * q_local / alpha;

	//Add disk cooling term
	double X[NDIM], r, th, phi;
	#if(COOL_DISK)
	coord(n, ii, jj, zz, CENT, X);
	bl_coord(X, &r, &th, &phi);
	misc_source(ph, ii, jj, geom, &q, dU, r, Dt);
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g ;
	#endif
}

/* find ucon, ucov, bcon, bcov from primitive variables */
void get_state_res(double * restrict pr, struct of_geom * restrict geom, struct of_state_res * restrict q_res)
{
	#if(RESISTIVE)
	//get ucon
	ucon_calc(pr, geom, q_res->ucon) ;
	lower(q_res->ucon, geom, q_res->ucov) ;

	//get bcon
	bcon_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->bcon) ;
	lower(q_res->bcon, geom, q_res->bcov) ;

	//get econ
	econ_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->econ);
	lower(q_res->econ, geom, q_res->ecov);
	#endif
}

void vchar_res( struct of_geom * restrict geom, int js,double * restrict vmax, double * restrict vmin){
	//#if(RESISTIVE)
	double sqrtgamma, ncon_js,alpha,beta, vm, vp;
	alpha = 1. / sqrt(-geom->gcon[0][0]);
	beta= geom->gcon[0][js] * alpha * alpha;
	ncon_js = -alpha * geom->gcon[0][js];

	sqrtgamma = sqrt(geom->gcon[js][js] + ncon_js * ncon_js);

	vp = alpha * sqrtgamma - beta;
	vm = -alpha * sqrtgamma - beta;

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
}
	else {
		*vmax = vm;
		*vmin = vp;
	}

	//#endif
}

double divE_calc(double(*restrict p[NB_LOCAL])[NPR],  int n, int i, int j, int z) {
	#if(RESISTIVE)
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
	#else
	return(0.0);
	#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
void lower_3(double* restrict ucon, struct of_geom* restrict geom, double* restrict ucov)
{
	#if(RESISTIVE)
	int i, j;
	for (i = 0; i < 3; i++) {
		ucov[i] = 0.0;
		for (j = 0; j < 3; j++) {
			ucov[i] += geom->gcov[i + 1][j + 1] * ucon[j];
		}
	}
	return;
	#endif
}

//4D Levi-cevita symbol (not tensor)
double lvc4u(int i, int j, int k, int l) {
	double lvc4u;

	if ((i==j) || (i==k) || (i==l) || (j==k) || (j==l) || (k==l)) {
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

	return (lvc4u);
}

//3D Levi-cevita symbol (not tensor)
double lvc3u(int i, int j, int k) {
	double lvc3u;
			
	if ((i == j) || (j == k) || (k == i)) lvc3u = 0.;
	else if ((i + 1 == j) || (i - 2 == j)) lvc3u = 1.;
	else lvc3u = -1.;

	return (lvc3u);
}