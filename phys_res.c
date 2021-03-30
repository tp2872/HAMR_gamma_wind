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
	double sqrtgamma_inv;

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

	//Inverse of 3-metric
	sqrtgamma_inv = alpha / (geom->g);

	/*Maxwell tensor */
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	if (dir == 0) {
		flux[E1] = pr[E1];
		flux[E2] = pr[E2];
		flux[E3] = pr[E3];
	}
	else {
		flux[E1] = beta[1] * pr[E1 + (dir - 1)] - beta[dir] * pr[E1];
		flux[E2] = beta[2] * pr[E1 + (dir - 1)] - beta[dir] * pr[E2];
		flux[E3] = beta[3] * pr[E1 + (dir - 1)] - beta[dir] * pr[E3];
		flux[E1] -= lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 0 - (dir - 1))]);
		flux[E2] -= lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 1 - (dir - 1))]);
		flux[E3] -= lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 2 - (dir - 1))]);
	}

	/* dual of Maxwell tensor */
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	if (dir == 0) {
		flux[B1] = pr[B1];
		flux[B2] = pr[B2];
		flux[B3] = pr[B3];
	}
	else {
		flux[B1] = beta[1] * pr[B1 + (dir - 1)] - beta[dir] * pr[B1];
		flux[B2] = beta[2] * pr[B1 + (dir - 1)] - beta[dir] * pr[B2];
		flux[B3] = beta[3] * pr[B1 + (dir - 1)] - beta[dir] * pr[B3];
		flux[B1] += lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 0 - (dir - 1))]);
		flux[B2] += lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 1 - (dir - 1))]);
		flux[B3] += lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 2 - (dir - 1))]);
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
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	E_dot_v = alpha*(pr[E1] * ucov[1] + pr[E2] * ucov[2] + pr[E3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	econ[0] = (E_dot_v)*ncon[0];
	econ[1] = (E_dot_v)*ncon[1] + gamma * (alpha * pr[E1]) + (alpha * alpha / geom->g) * (ucov[2] * Bcov[2] - ucov[3] * Bcov[1]);
	econ[2] = (E_dot_v)*ncon[2] + gamma * (alpha * pr[E2]) + (alpha * alpha / geom->g) * (ucov[3] * Bcov[0] - ucov[1] * Bcov[2]);
	econ[3] = (E_dot_v)*ncon[3] + gamma * (alpha * pr[E3]) + (alpha * alpha / geom->g) * (ucov[1] * Bcov[1] - ucov[2] * Bcov[0]);

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
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	B_dot_v = alpha * (pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	bcon[0] = (B_dot_v) * ncon[0];
	bcon[1] = (B_dot_v) * ncon[1] + gamma * (alpha * pr[B1]) - (alpha * alpha / geom->g) * (ucov[2] * Ecov[2] - ucov[3] * Ecov[1]);
	bcon[2] = (B_dot_v) * ncon[2] + gamma * (alpha * pr[B2]) - (alpha * alpha / geom->g) * (ucov[3] * Ecov[0] - ucov[1] * Ecov[2]);
	bcon[3] = (B_dot_v) * ncon[3] + gamma * (alpha * pr[B3]) - (alpha * alpha / geom->g) * (ucov[1] * Ecov[1] - ucov[2] * Ecov[0]);

	return;
	#endif
}

/* MHD stress tensor, with first index up, second index down */
void mhd_calc_res(double * restrict pr, int dir, struct of_geom* restrict geom, struct of_state_res * restrict q_res, double * restrict mhd)
{
	#if(RESISTIVE)
	int j, lambda, beta, kappa;
	double P,w,bsq,esq, eta,ptot, mhd_u[NDIM], mhd_d[NDIM];

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
	double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg, J[NDIM], beta[NDIM], alpha, gamma;
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

	//Calculate relative Lorentz factor
	gamma = q_res.ucon[0] * alpha;

	//Calculate explicit part of electric current J sourceterm
	dU[E1] = -alpha * q[0] * ph[U1] / gamma + beta[1] * q[0];
	dU[E2] = -alpha * q[0] * ph[U2] / gamma + beta[2] * q[0];
	dU[E3] = -alpha * q[0] * ph[U3] / gamma + beta[3] * q[0];

	//Add disk cooling term
	#if(COOL_DISK)
	double X[NDIM], r, th, phi;
	coord(n, ii, jj, zz, CENT, X);
	bl_coord(X, &r, &th, &phi);
	misc_source(ph, ii, jj, geom, &q, dU, r, Dt);
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g ;
	#endif
}

/* returns b^2 (i.e., twice magnetic pressure) */
double bsq_calc_res(double* restrict pr, struct of_geom* restrict geom)
{
	double ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM];
	ucon_calc(pr, geom, ucon);
	lower(ucon, geom, ucov);
	bcon_calc_res(pr, geom, ucon, ucov, bcon);
	lower(bcon, geom, bcov);

	return(dot(bcon, bcov));
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
}

void vchar_res2(double* restrict pr, struct of_state_res* restrict q, struct of_geom* restrict geom, int js, double* restrict vmax, double* restrict vmin)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2;
	double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	DLOOPA Acov[j] = 0.;
	Acov[js] = 1.;
	raise(Acov, geom, Acon);

	DLOOPA Bcov[j] = 0.;
	Bcov[0] = 1.;
	raise(Bcov, geom, Bcon);

	/* find fast magnetosonic speed */
	bsq = dot(q->bcon, q->bcov);

	#if DOHELM
	// Helmholtz EOS
	double xpres;
	eos_mode_rhou_pres_cs2(pr[RHO], pr[UU], &xpres, &cs2);
	va2 = bsq / (bsq + pr[RHO] + pr[UU] + xpres);
	#else
	// Ideal gas EOS
	EF = pr[RHO] + GAMMA * pr[UU];
	EE = bsq + EF;
	va2 = bsq / EE;
	cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / EF;
	#endif

	cms2 = cs2 + va2 - cs2 * va2;	/* and there it is... */

	/* check on it! */
	if (cms2 < 0.) {
		fail(FAIL_COEFF_NEG);
		cms2 = SMALL;
	}
	if (cms2 > 1.) {
		fail(FAIL_COEFF_SUP);
		cms2 = 1.;
	}

	/* now require that speed of wave measured by observer q->ucon is cms2 */
	Asq = dot(Acon, Acov);
	Bsq = dot(Bcon, Bcov);
	Au = dot(Acov, q->ucon);
	Bu = dot(Bcov, q->ucon);
	AB = dot(Acon, Bcov);
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * cms2;
	B = 2. * (AuBu - (AB + AuBu) * cms2);
	C = Au2 - (Asq + Au2) * cms2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		fprintf(stderr, "\n\t %g %g %g %g %g\n", A, B, C, discr, cms2);
		fprintf(stderr, "\n\t q->ucon: %g %g %g %g\n", q->ucon[0], q->ucon[1],
			q->ucon[2], q->ucon[3]);
		fprintf(stderr, "\n\t q->bcon: %g %g %g %g\n", q->bcon[0], q->bcon[1],
			q->bcon[2], q->bcon[3]);
		fprintf(stderr, "\n\t Acon: %g %g %g %g\n", Acon[0], Acon[1],
			Acon[2], Acon[3]);
		fprintf(stderr, "\n\t Bcon: %g %g %g %g\n", Bcon[0], Bcon[1],
			Bcon[2], Bcon[3]);
		fail(FAIL_VCHAR_DISCR);
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
	}
	else {
		*vmax = vm;
		*vmin = vp;
	}

	return;
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
	dz = (N3 > 1) * zsize;
	#endif

	/* Constrained transport defn */
	#if(STAGGERED_E)
	#if(N1>1)
	for (u = 0; u < zsize; u++) {
		dive += 0.25 * (pse[nl[n]][index_3D(n, i + di, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i + di, j, z - zoffset + u)][FACE1] - pse[nl[n]][index_3D(n, i, j, z - zoffset + u)][1] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE1]) / ((double)*dx[nl[n]][1]);
	}
	#endif
	#if(N2>1)
	for (u = 0; u < zsize; u++) {
		dive += 0.25 * (pse[nl[n]][index_3D(n, i, j + dj, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j + dj, z - zoffset + u)][FACE2] - pse[nl[n]][index_3D(n, i, j, z - zoffset + u)][2] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + u)][FACE2]) / ((double)*dx[nl[n]][2]);
	}
	#endif
	#if(N3>1)
	dive += 0.25 * (pse[nl[n]][index_3D(n, i, j, z - zoffset + dz * zsize)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset + dz * zsize)][FACE3] - pse[nl[n]][index_3D(n, i, j, z - zoffset)][3] * gdet[nl[n]][index_2D(n, i, j, z - zoffset)][FACE3]) / ((double)(zsize)*dx[nl[n]][3]);
	#endif
	#else
	/* Flux-ct defn */
	dive = (
		#if(N1>1)
			(p[nl[n]][index_3D(n, i+D1, j, z)][E1] * gdet[nl[n]][index_2D(n, i+D1, j, z)][CENT] - p[nl[n]][index_3D(n, i - D1, j, z)][E1] * gdet[nl[n]][index_2D(n, i - D1, j, z)][CENT]) / (2.0*(double)dx[nl[n]][1])
		#endif
		#if(N2>1)
			+ (p[nl[n]][index_3D(n, i, j+D2, z)][E2] * gdet[nl[n]][index_2D(n, i, j+D2, z)][CENT] - p[nl[n]][index_3D(n, i, j - D2, z)][E2] * gdet[nl[n]][index_2D(n, i, j - D2, z)][CENT]) / (2.0*(double)dx[nl[n]][2])
		#endif
		#if(N3>1)
			+ (p[nl[n]][index_3D(n, i, j, z+D3*zsize)][E3] * gdet[nl[n]][index_2D(n, i, j, z+D3*zsize)][CENT] - p[nl[n]][index_3D(n, i, j, z - D3*zsize)][E3] * gdet[nl[n]][index_2D(n, i, j, z - D3*zsize)][CENT]) / (2.0*(double)(zsize)*dx[nl[n]][3])
		#endif
	);
	#endif
	return (dive / gdet[nl[n]][index_2D(n, i, j, z)][CENT]);
	#else
	return(0.0);
	#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
void lower_3(double* restrict ucon, double restrict gcov[NDIM][NDIM], double* restrict ucov)
{
	#if(RESISTIVE)
	int i, j;
	for (i = 0; i < 3; i++) {
		ucov[i] = 0.0;
		for (j = 0; j < 3; j++) {
			ucov[i] += gcov[i + 1][j + 1] * ucon[j];
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