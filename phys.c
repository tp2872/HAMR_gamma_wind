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

void primtoflux(double * restrict pr, struct of_state * restrict q, struct of_state_rad * restrict q_rad, int dir, struct of_geom * restrict geom, double * restrict flux)
{
	int j,k ;
	double n = 1. / (GAMMA - 1.);

	/* particle number flux */
	flux[RHO] = pr[RHO]*q->ucon[dir] ;

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc(pr, dir, q, &flux[UU]) ;
	flux[UU] += flux[RHO];

	//Radiation energy tensor
	#if(RAD_M1)
	mhd_calc_rad(pr, dir, q_rad, &flux[UU_RAD]);
	#endif

	/* dual of Maxwell tensor */
	#pragma ivdep
	for (k = B1; k <= B3; k++){
		flux[k] = q->bcon[k-4] * q->ucon[dir] - q->bcon[dir] * q->ucon[k-4];
	}

	//Entropy advection
	#if(DOKTOT )
	flux[KTOT] = flux[RHO] * log(pow((GAMMA-1.0)*pr[UU], n) / pow(pr[RHO], n + 1));
	//flux[KTOT] = flux[RHO] * pr[KTOT];
	#endif

	#pragma ivdep
	PLOOP flux[k] *= geom->g ;
}

/* calculate magnetic field four-vector */
void bcon_calc(double * restrict pr, double * restrict ucon, double * restrict ucov, double * restrict bcon)
{
	int j ;

	bcon[0] = pr[B1]*ucov[1] + pr[B2]*ucov[2] + pr[B3]*ucov[3] ;
	/*#pragma ivdep*/
	for(j=1;j<4;j++)
		bcon[j] = (pr[B1-1+j] + bcon[0]*ucon[j])/ucon[0] ;

	return ;
}

/* MHD stress tensor, with first index up, second index down */
void mhd_calc(double * restrict pr, int dir, struct of_state * restrict q, double * restrict mhd)
{
	int j ;
	double r,u,P,w,bsq,eta,ptot ;

    r = pr[RHO] ;
    u = pr[UU] ;
    P = (gam - 1.)*u ;
    w = P + r + u ;
	bsq = dot(q->bcon,q->bcov) ;
	eta = w + bsq ;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	#pragma ivdep
	DLOOPA mhd[j] = eta*q->ucon[dir]*q->ucov[j] + ptot*delta(dir,j) - q->bcon[dir]*q->bcov[j] ;
}

/* Radiation stress tensor, with first index up, second index down */
void mhd_calc_rad(double * restrict pr, int dir, struct of_state_rad * restrict q_rad, double * restrict mhd_rad)
{
	int j;
	/* single row of mhd stress tensor, first index up, second index down */
	#pragma ivdep
	DLOOPA mhd_rad[j] = 4./3.*pr[UU_RAD]*q_rad->ucon[dir] * q_rad->ucov[j] + 1./3.*pr[UU_RAD]*delta(dir, j);
}

/* add in (explicit) geometricc source terms to equations of motion */
void source(double * restrict ph, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU, double Dt)
{
	double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg;
	int j,k ;
	struct of_state q ;
	struct of_state_rad q_rad;

	get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, &q, mhd[0]) ;
	mhd_calc(ph, 1, &q, mhd[1]) ;
	mhd_calc(ph, 2, &q, mhd[2]) ;
	mhd_calc(ph, 3, &q, mhd[3]) ;

	#pragma ivdep
	PLOOP dU[k] = 0.;
	
	//contract mhd stress tensor with connection
	DLOOP {
		dU[UU] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][0][j];
		dU[U1] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][1][j];
		dU[U2] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][2][j];
		dU[U3] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][3][j];
	}

	//Add disk cooling term
	#if(COOL_DISK)
	double X[NDIM],r,th,phi;
	coord(n, ii,jj, zz, CENT,X) ;
	bl_coord(X,&r,&th, &phi) ;
	misc_source(ph, ii, jj, geom, &q, dU,r, Dt) ;
	#endif

	//Add M1 radiation source terms
	#if(RAD_M1)
	get_state_rad(ph, geom, &q_rad);
	mhd_calc_rad(ph, 0, &q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, &q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, &q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, &q_rad, mhd_rad[3]);

	//contract radiation stress tensor with connection
	DLOOP{
		dU[UU_RAD] += mhd_rad[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][0][j];
		dU[U1_RAD] += mhd_rad[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][1][j];
		dU[U2_RAD] += mhd_rad[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][2][j];
		dU[U3_RAD] += mhd_rad[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][3][j];
	}
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g ;
}

/* add in (implicit) radiation 4-force source term to equations of motion */
void source_rad(double * restrict ph, struct of_geom * restrict geom,  double * restrict dU)
{
	#if(RAD_M1)
	double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], ucon[NDIM], ucov[NDIM], Tg;
	int j, k;
	struct of_state_rad q_rad;
	
	PLOOP dU[k] = 0.;

	//Add M1 radiation terms
	get_state_rad(ph, geom, &q_rad);
	mhd_calc_rad(ph, 0, &q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, &q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, &q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, &q_rad, mhd_rad[3]);

	//Add radiation 4-force
	ucon_calc(ph, geom, ucon);
	lower(ucon, geom, ucov);
	calc_Gcon(ph, Gcon, ucon, ucov, mhd_rad);
	lower(Gcon, geom, Gcov);

	dU[UU] = Gcov[0];
	dU[U1] = Gcov[1];
	dU[U2] = Gcov[2];
	dU[U3] = Gcov[3];

	dU[UU_RAD] = -Gcov[0];
	dU[U1_RAD] = -Gcov[1];
	dU[U2_RAD] = -Gcov[2];
	dU[U3_RAD] = -Gcov[3];

	Tg = (GAMMA - 1.)*(ph[UU]) / (ph[RHO]);
	dU[KTOT] = -1. / Tg * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}

//Calculate radiation 4-force
void calc_Gcon(double * restrict ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM]) {
	int i;
	double lambda, Tg, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], arad;
	kappa_abs = calc_kappa_abs(ph);
	kappa_emmit =calc_kappa_emmit(ph);
	kappa_es = calc_kappa_es(ph);
	arad = ARAD / (MASS_DENSITY_SCALE * C_CGS * C_CGS / pow(MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.));

	Tg = (GAMMA - 1.)*ph[UU] / ph[RHO];
	lambda = kappa_abs*arad*pow(Tg,4.);
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs*R_dot_ucon[i] + lambda*ucon[i]) - kappa_es*(R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3])*ucon[i]);
	}
}

//Calculate total absorption opacity
double calc_kappa_abs(double * restrict ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	double Tg = fabs(MMW*MH_CGS*(GAMMA - 1.)*(ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS*ph[RHO] * MASS_DENSITY_SCALE));
	double Tr = fabs(pow(ph[UU_RAD] * ENERGY_DENSITY_SCALE * ARAD, 0.25));
	kappa_m = 0.1*Z_AB;
	kappa_h = 1.1*pow(10., -25.)*sqrt(Z_AB*ph[RHO])*pow(Tg, 7.7);
	kappa_chianti = 4.0*pow(10., 34.)*(Z_AB / 0.02)*Ye*pow(Tg, -1.7)*pow(Tr, -3.);
	kappa_bf = 3.0*pow(10., 25.)*Z_AB*(1. + X_AB + 0.75*Y_AB)*ph[RHO] * pow(Tg, -0.5)*pow(Tr, -3.0)*log(1. + 1.6*(Tr / Tg));
	kappa_ff = 4.0*pow(10., 22.)*(1. + X_AB)*(1. - Z_AB)*ph[RHO] * pow(Tg, -0.5)*pow(Tr, -3.0)*log(1. + 1.6*(Tr / Tg))*(1. + 4.4*pow(10., -10.)*Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	kappa_abs = 1.7*ph[RHO]*pow(10., -25.)*pow(Tg, -7. / 2.)*pow(MH_CGS,-2.);

	return kappa_abs*(ph[RHO]*MASS_DENSITY_SCALE)*R_G_CGS;
}

//Calculate total emmission opacity
double calc_kappa_emmit(double * restrict ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	double Tg = fabs(MMW*MH_CGS*(GAMMA - 1.)*(ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS*ph[RHO] * MASS_DENSITY_SCALE));
	//Tg = fabs((GAMMA - 1.)*(ph[UU]) / (ph[RHO]));

	kappa_m = 0.1*Z_AB;
	kappa_h = 1.1*pow(10., -25.)*sqrt(Z_AB*ph[RHO])*pow(Tg, 7.7);
	kappa_chianti = 4.0*pow(10., 34.)*(Z_AB / 0.02)*Ye*pow(Tg, -4.7);
	kappa_bf = 3.0*pow(10., 25.)*Z_AB*(1. + X_AB + 0.75*Y_AB)*ph[RHO] * pow(Tg, -3.5)*log(1. + 1.6);
	kappa_ff = 4.0*pow(10., 22.)*(1. + X_AB)*(1. - Z_AB)*ph[RHO] * pow(Tg, -3.5)*log(1. + 1.6)*(1. + 4.4*pow(10., -10.)*Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	kappa_abs = 1.7*ph[RHO] *pow(10., -25.)*pow(Tg, -7. / 2.)*pow(MH_CGS, -2.);

	return kappa_abs*(ph[RHO] * MASS_DENSITY_SCALE)*R_G_CGS;
}

//Calculate total (electron) scattering opacity
double calc_kappa_es(double * restrict ph) {
	double kappa_es;
	double Tg = MMW*MH_CGS*(GAMMA - 1.)*(ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS*ph[RHO] * MASS_DENSITY_SCALE);
	kappa_es = 0.2*(1 + X_AB) / (1. + pow(Tg / (4.5*pow(10., 8.)), 0.86));
	kappa_es = 0.2*(1 + X_AB);
	return kappa_es*(ph[RHO]*MASS_DENSITY_SCALE)*R_G_CGS;
}

/* returns b^2 (i.e., twice magnetic pressure) */
double bsq_calc(double * restrict pr, struct of_geom * restrict geom)
{
	struct of_state q ;

	get_state(pr,geom,&q) ;
	return( dot(q.bcon,q.bcov) ) ;
}

/* find ucon, ucov, bcon, bcov from primitive variables */
void get_state(double * restrict pr, struct of_geom * restrict geom, struct of_state * restrict q)
{

	/* get ucon */
	ucon_calc(pr, geom, q->ucon) ;
	lower(q->ucon, geom, q->ucov) ;
	bcon_calc(pr, q->ucon, q->ucov, q->bcon) ;
	lower(q->bcon, geom, q->bcov) ;

	return ;
}

/* find ucon, ucov, bcon, bcov from radiation primitive variables */
void get_state_rad(double * restrict pr, struct of_geom * restrict geom, struct of_state_rad * restrict q_rad)
{
	/* get radiation ucon */
	ucon_calc_rad(pr, geom, q_rad->ucon);
	lower(q_rad->ucon, geom, q_rad->ucov);

	return;
}

/* find contravariant four-velocity */
void ucon_calc(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon)
{
	double alpha,gamma ;
	double beta[NDIM] ;
	int j ;

	alpha = 1./sqrt(-geom->gcon[0][0]) ;
	 #pragma ivdep
	SLOOPA beta[j] = geom->gcon[0][j]*alpha*alpha ;

	if( gamma_calc(pr,geom,&gamma) ) { 
	  fflush(stderr);
	  fprintf(stderr,"\nucon_calc(): gamma failure \n");
	  fflush(stderr);
	  fail(FAIL_GAMMA);
	}

	ucon[0] = gamma/alpha ;
	 #pragma ivdep
	SLOOPA ucon[j] = pr[U1+j-1] - gamma*beta[j]/alpha ;

	return ;
}

/* find contravariant radiation four-velocity */
void ucon_calc_rad(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon_rad)
{
	double alpha, gamma_rad;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0][0]);
	#pragma ivdep
	SLOOPA beta[j] = geom->gcon[0][j] * alpha*alpha;

	if (gamma_calc_rad(pr, geom, &gamma_rad)) {
		fflush(stderr);
		fprintf(stderr, "\nucon_calc(): gamma_rad failure \n");
		fflush(stderr);
		fail(FAIL_GAMMA);
	}

	ucon_rad[0] = gamma_rad / alpha;
	#pragma ivdep
	SLOOPA ucon_rad[j] = pr[U1_RAD + j - 1] - gamma_rad*beta[j] / alpha;

	return;
}

/* find gamma-factor wrt normal observer */
int gamma_calc(double * restrict pr, struct of_geom * restrict geom, double * restrict gamma)
{
        double qsq ;
        qsq =  geom->gcov[1][1]*pr[U1]*pr[U1]  + geom->gcov[2][2]*pr[U2]*pr[U2] + geom->gcov[3][3]*pr[U3]*pr[U3] + 2.*(geom->gcov[1][2]*pr[U1]*pr[U2]+ geom->gcov[1][3]*pr[U1]*pr[U3] + geom->gcov[2][3]*pr[U2]*pr[U3]);
        if( qsq < 0. ){
          if( fabs(qsq) > 1.E-10 ){ // then assume not just machine precision
            fprintf(stderr,"gamma_calc():  failed: qsq = %28.18e \n", qsq);
            fprintf(stderr,"v[1-3] = %28.18e %28.18e %28.18e  \n",pr[U1],pr[U2],pr[U3]);
	    *gamma = 1.;
	    return (1);
	  }
          else qsq=1.E-10; // set floor
        }

        *gamma = sqrt(1. + qsq) ;

        return(0) ;
}

/* find gamma-factor wrt normal observer */
int gamma_calc_rad(double * restrict pr, struct of_geom * restrict geom, double * restrict gamma_rad)
{
	double qsq;
	qsq = geom->gcov[1][1] * pr[U1_RAD] * pr[U1_RAD] + geom->gcov[2][2] * pr[U2_RAD] * pr[U2_RAD] + geom->gcov[3][3] * pr[U3_RAD] * pr[U3_RAD] + 2.*(geom->gcov[1][2] * pr[U1_RAD] * pr[U2_RAD] + geom->gcov[1][3] * pr[U1_RAD] * pr[U3_RAD] + geom->gcov[2][3] * pr[U2_RAD] * pr[U3_RAD]);
	if (qsq < 0.) {
		if (fabs(qsq) > 1.E-10) { // then assume not just machine precision
			fprintf(stderr, "gamma_calc_rad():  failed: qsq = %28.18e \n", qsq);
			fprintf(stderr, "v[1-3] = %28.18e %28.18e %28.18e  \n", pr[U1_RAD], pr[U2_RAD], pr[U3_RAD]);
			*gamma_rad = 1.;
			return (1);
		}
		else qsq = 1.E-10; // set floor
	}

	*gamma_rad = sqrt(1. + qsq);

	return(0);
}

/*  
 * VCHAR():
 * 
 * calculate components of magnetosonic velocity 
 * corresponding to primitive variables p 
 *
 * cfg 7-10-01
 * 
 */

void vchar(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js,double * restrict vmax, double * restrict vmin, int a, int b, int c)
{
	double discr,vp,vm,bsq,EE,EF,va2,cs2,cms2,rho,u ;
	double Acov[NDIM],Bcov[NDIM],Acon[NDIM],Bcon[NDIM] ;
	double Asq,Bsq,Au,Bu,AB,Au2,Bu2,AuBu,A,B,C ;
	int j ;

	 #pragma ivdep
	DLOOPA Acov[j] = 0. ;
	Acov[js] = 1. ;
	raise(Acov,geom,Acon) ;
	
	 #pragma ivdep
	DLOOPA Bcov[j] = 0. ;
	Bcov[0] = 1. ;
	raise(Bcov,geom,Bcon) ;

	/* find fast magnetosonic speed */
	bsq = dot(q->bcon,q->bcov) ;
	rho = pr[RHO] ;
	u = pr[UU] ;
	EF = rho + gam*u ;
	EE = bsq + EF ;
	va2 = bsq/EE ;
	cs2 = gam*(gam - 1.)*u/EF ;
	cms2 = cs2 + va2 - cs2*va2 ;	/* and there it is... */

	/* check on it! */
	if(cms2 < 0.) {
		fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL ;
	}
	if(cms2 > 1.) {
		fail(FAIL_COEFF_SUP) ;
		cms2 = 1. ;
	}

	/* now require that speed of wave measured by observer q->ucon is cms2 */
	Asq = dot(Acon,Acov) ;
	Bsq = dot(Bcon,Bcov) ;
	Au =  dot(Acov,q->ucon) ;
	Bu =  dot(Bcov,q->ucon) ;
	AB =  dot(Acon,Bcov) ;
	Au2 = Au*Au ;
	Bu2 = Bu*Bu ;
	AuBu = Au*Bu ;

	A =      Bu2  - (Bsq + Bu2)*cms2 ;
	B = 2.*( AuBu - (AB + AuBu)*cms2 ) ;
	C =      Au2  - (Asq + Au2)*cms2 ;

	discr = B*B - 4.*A*C ;
	if((discr<0.0)&&(discr>-1.e-10)) discr=0.0;
	else if(discr < -1.e-10) {
		fprintf(stderr,"\n\t %g %g %g %g %g\n",A,B,C,discr,cms2) ;
		fprintf(stderr,"\n\t q->ucon: %g %g %g %g\n",q->ucon[0],q->ucon[1],
				q->ucon[2],q->ucon[3]) ;
		fprintf(stderr,"\n\t q->bcon: %g %g %g %g\n",q->bcon[0],q->bcon[1],
				q->bcon[2],q->bcon[3]) ;
		fprintf(stderr,"\n\t Acon: %g %g %g %g\n",Acon[0],Acon[1],
				Acon[2],Acon[3]) ;
		fprintf(stderr,"\n\t Bcon: %g %g %g %g\n",Bcon[0],Bcon[1],
				Bcon[2],Bcon[3]) ;
		fail(FAIL_VCHAR_DISCR) ;
		discr = 0. ;
	}

	discr = sqrt(discr) ;
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	if(vp > vm) {
		*vmax = vp ;
		*vmin = vm ;
	}
	else {
		*vmax = vm ;
		*vmin = vp ;
	}

	return ;
}

//Calculate radiative wave velocity
void vchar_rad(double * restrict pr, struct of_state_rad * restrict q_rad, struct of_geom * restrict geom, int js, double * restrict vmax, double * restrict vmin, double dx){
	double discr, vp, vm, tau, kappa_tot, crad2;
	double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	#pragma ivdep
	DLOOPA Acov[j] = 0.;
	Acov[js] = 1.;
	raise(Acov, geom, Acon);

	#pragma ivdep
	DLOOPA Bcov[j] = 0.;
	Bcov[0] = 1.;
	raise(Bcov, geom, Bcon);

	/* find radiation wave speed */
	kappa_tot = calc_kappa_abs(pr) + calc_kappa_es(pr);
	tau = kappa_tot*sqrt(geom->gcov[js][js])*dx;
	crad2 = MY_MIN(1.0 / 3.0, pow(4. / (3.*tau),2.));

	/* check on it! */
	if (crad2 < 0.) {
		fail(FAIL_COEFF_NEG);
		crad2 = SMALL;
	}
	if (crad2 > 1.) {
		fail(FAIL_COEFF_SUP);
		crad2 = 1.;
	}

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Asq = dot(Acon, Acov);
	Bsq = dot(Bcon, Bcov);
	Au = dot(Acov, q_rad->ucon);
	Bu = dot(Bcov, q_rad->ucon);
	AB = dot(Acon, Bcov);
	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;

	A = Bu2 - (Bsq + Bu2)*crad2;
	B = 2.*(AuBu - (AB + AuBu)*crad2);
	C = Au2 - (Asq + Au2)*crad2;

	discr = B*B - 4.*A*C;
	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		fprintf(stderr, "Failed in vchar_rad");
		exit(0);
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

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

/* Add any additional source terms (e.g. cooling functions) */
void misc_source(double *ph, int ii, int jj, struct of_geom *geom, struct of_state *q, double *dU, double r, double Dt) 
{
	double epsilon = ph[UU] / ph[RHO];
	double om_kepler = 1. / (pow(r, 3. / 2.) + a);
	double T_target = M_PI / 2.*pow(H_OVER_R*r*om_kepler, 2.);
	double Y = (gam - 1.)*epsilon / T_target;
	double lambda = om_kepler*ph[UU] * sqrt(Y - 1. + fabs(Y - 1.));
	double int_energy = q->ucov[0] * q->ucon[0] * ph[UU];
	double bsq = dot(q->bcon, q->bcov);

	if (bsq / ph[RHO]<1. || r<10.){
		if (fabs(q->ucov[0] * lambda)*Dt<0.1*fabs(int_energy)){
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
		}
		else{
			lambda *= (0.1*fabs(int_energy)) / (fabs(q->ucov[0] * lambda)*Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
		}
	}
}
