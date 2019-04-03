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

void primtoflux(double * restrict pr, struct of_state * restrict q, int dir,
struct of_geom * restrict geom, double * restrict flux)
{
	int j,k ;
	double mhd[NDIM];

	/* particle number flux */
	flux[RHO] = pr[RHO]*q->ucon[dir] ;
	mhd_calc(pr, dir, q, mhd) ;

	/* MHD stress-energy tensor w/ first index up, 
	 * second index down. */
	#pragma ivdep
	for (k = 0; k < 4; k++){
		flux[k+1] = mhd[k] ;
	}
	flux[UU] += flux[RHO];

	/* dual of Maxwell tensor */
	#pragma ivdep
	for (k = B1; k <= B3; k++){
		flux[k] = q->bcon[k-4] * q->ucon[dir] - q->bcon[dir] * q->ucon[k-4];
	}
	#if(DOKTOT )
	flux[KTOT] = flux[RHO] * pr[KTOT];
	#endif
	#pragma ivdep
	PLOOP flux[k] *= geom->g ;
}

/* calculate "conserved" quantities; provided strictly for
 * historical reasons */
void primtoU(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, double * restrict U)
{
	primtoflux(pr,q,0,geom, U) ;
	return ;
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

	/* single row of mhd stress tensor, 
	 * first index up, second index down */
	#pragma ivdep
	DLOOPA mhd[j] = eta*q->ucon[dir]*q->ucov[j]
		+ ptot*delta(dir,j) - q->bcon[dir]*q->bcov[j] ;

}

/* add in source terms to equations of motion */
void source(double * restrict ph, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU,
		double Dt)
{
	double mhd[NDIM][NDIM] ;
	int j,k ;
	struct of_state q ;

	get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, &q, mhd[0]) ;
	mhd_calc(ph, 1, &q, mhd[1]) ;
	mhd_calc(ph, 2, &q, mhd[2]) ;
	mhd_calc(ph, 3, &q, mhd[3]) ;

	/* contract mhd stress tensor with connection */
	 #pragma ivdep
	PLOOP dU[k] = 0. ;

	DLOOP {
		dU[UU] += mhd[j][k] * conn[nl[n]][index_2D(n,ii,jj,zz)][k][0][j] ;
		dU[U1] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][1][j];
		dU[U2] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][2][j];
		dU[U3] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][3][j];
		//fprintf(stderr, "(%d,%d,%f):%f\n", j, k, gcon[index_2D(ii, jj)][0][k][j] / gcon[index_2D(ii, jj)][0][j][k], log(fabs(gcon[index_2D(ii, jj)][0][k][j])));
	}
	#if(COOL_DISK)
	double X[NDIM],r,th,phi;
	coord(n, ii,jj, zz, CENT,X) ;
	bl_coord(X,&r,&th, &phi) ;
	misc_source(ph, ii, jj, geom, &q, dU,r, Dt) ;
	#endif
	#pragma ivdep
	PLOOP dU[k] *= geom->g ;

	/* done! */
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

/* find gamma-factor wrt normal observer */
int gamma_calc(double * restrict pr, struct of_geom * restrict geom, double * restrict gamma)
{
        double qsq ;
        qsq =     geom->gcov[1][1]*pr[U1]*pr[U1]
                + geom->gcov[2][2]*pr[U2]*pr[U2]
                + geom->gcov[3][3]*pr[U3]*pr[U3]
            + 2.*(geom->gcov[1][2]*pr[U1]*pr[U2]
                + geom->gcov[1][3]*pr[U1]*pr[U3]
                + geom->gcov[2][3]*pr[U2]*pr[U3]) ;
        if( qsq < 0. ){
          if( fabs(qsq) > 1.E-10 ){ // then assume not just machine precision
            fprintf(stderr,"gamma_calc():  failed: i,j,qsq = %d %d %28.18e \n", icurr,jcurr,qsq);
            fprintf(stderr,"v[1-3] = %28.18e %28.18e %28.18e  \n",pr[U1],pr[U2],pr[U3]);
	    *gamma = 1.;
	    return (1);
	  }
          else qsq=1.E-10; // set floor
        }

        *gamma = sqrt(1. + qsq) ;

        return(0) ;
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

void vchar(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js,
	double * restrict vmax, double * restrict vmin, int a, int b, int c)
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

//	if(cs2 < 0.) cs2 = SMALL ;
//	if(cs2 > 1.) cs2 = 1. ;
//	if(va2 < 0.) va2 = SMALL ;
//	if(va2 > 1.) va2 = 1. ;

	cms2 = cs2 + va2 - cs2*va2 ;	/* and there it is... */

	//cms2 *= 1.1 ;

	/* check on it! */
	if(cms2 < 0.) {
		fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL ;
	}
	if(cms2 > 1.) {
		fail(FAIL_COEFF_SUP) ;
		cms2 = 1. ;
	}

	/* now require that speed of wave measured by observer 
	   q->ucon is cms2 */
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
	
	#if( FULL_DISP ) 
	double vp2, vm2;
	vp2 = NewtonRaphson(vp, 5, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vm2 = NewtonRaphson(vm, 5, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vp = vp2;
	vm = vm2;
	#endif

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

double NewtonRaphson(double start, int max_count, int dir, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq)
{
	int count = 0;
	double dx = start/1000000.0;
	double x = start;
	double diff, derivative;
	do{
		diff = Drel(dir, x, ucon, ucov, bcon, geom, E, vasq, csq);
		derivative = (Drel(dir, x + dx, ucon, ucov, bcon, geom, E, vasq, csq) - diff) / dx;
		count++;
		x = x - diff / (derivative);
	} while (Drel(dir, x*0.99999, ucon, ucov, bcon, geom, E, vasq, csq)*Drel(dir, x*1.00001, ucon, ucov, bcon, geom, E, vasq, csq)>0.0 && (count < max_count));
	if (count >= 3){
		x = start;
	}
	return x;
}

/*Calculate soundspeed*/
double Drel(int dir, double v, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq){
	double kcov[NDIM], kcon[NDIM], Kcov[NDIM], Kcon[NDIM];
	double om, omsq, ksq, kvasq, cfsq, result;
	int i;
	kcov[0] = -v; kcov[1] = 0.0; kcov[2] = 0.0; kcov[3] = 0.0;
	if (dir == 1){
		kcov[1] = 1.0;
	}
	if (dir == 2){
		kcov[2] = 1.0;
	}
	if (dir == 3){
		kcov[3] = 1.0;
	}
	raise(kcov, geom, kcon);
	om = dot(ucon, kcov);
	omsq = pow(om, 2.0);
	#pragma ivdep
	for (i = 0; i < NDIM; i++){
		Kcov[i] = kcov[i] + ucov[i] * om;
		Kcon[i] = kcon[i] + ucon[i] * om;
	}
	ksq = dot(Kcov, Kcon);
	kvasq = pow(dot(kcov, bcon), 2.0) / E;
	cfsq = vasq + csq*(1.0 - vasq);

	result = 0.5*(cfsq*ksq + csq*kvasq + sqrt(pow(cfsq*ksq + csq*kvasq, 2.0) - 4.0*ksq*csq*kvasq)) - omsq;
	return result;
}