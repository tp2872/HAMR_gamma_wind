#include "include.h"
#include "decs.h"

/***********************************************************************************************/
/***********************************************************************************************
  primtoflux():
  ---------
   --  calculate fluxes in direction dir, 
        
***********************************************************************************************/

void primtoflux(double * restrict pr, struct of_state * restrict q, struct of_state_rad * restrict q_rad, int dir, struct of_geom * restrict geom, double * restrict flux
	#if(TWO_T)
	, double gamma_g
	#endif
)
{
	int j,k ;

	/* particle number flux */
	flux[RHO] = pr[RHO]*q->ucon[dir] ;

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc(pr, dir, q, &flux[UU] 
		#if(TWO_T)
		, gamma_g
		#endif
	);
	flux[UU] += flux[RHO];

	/* dual of Maxwell tensor */
	#pragma ivdep
	for (k = B1; k <= B3; k++){
		flux[k] = q->bcon[k-4] * q->ucon[dir] - q->bcon[dir] * q->ucon[k-4];
	}

	//Radiation energy tensor
	#if(RAD_M1)
	mhd_calc_rad(pr, dir, q_rad, &flux[UU_RAD]);
	#endif

	//Flux of electron and ion entropies
	#if(TWO_T)
	flux[ENTRE] = flux[RHO] * pr[ENTRE];
	flux[ENTRI] = flux[RHO] * pr[ENTRI];
	#endif

	//Flux of photon number
	#if(P_NUM)
	flux[PHOTON] = pr[PHOTON] * q_rad->ucon[dir];
	#endif

	//Entropy advection
	#if(DOKTOT)
	flux[KTOT] = flux[RHO] * calc_entropy(pr
		#if(TWO_T)
		, gamma_g
		#endif
	);
	#endif
    
	for (k = 0; k < NPR; k++) flux[k] *= geom->g;
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
void mhd_calc(double * restrict pr, int dir, struct of_state * restrict q, double * restrict mhd 
	#if(TWO_T)
	, double gamma_g
	#endif
){
	int j ;
	double r,u,P,w,bsq,eta,ptot ;

    r = pr[RHO] ;
    u = pr[UU] ;
    
    #if DOHELM
    // Helmholtz EOS
    eos_mode_rhou_pres (pr, &P);
	#elif(TWO_T)
    // Ideal gas EOS
	P = (gamma_g - 1.) * u;
	#else
	P = (GAMMA - 1.) * u;
    #endif
    
    w = P + r + u ;
	bsq = dot(q->bcon,q->bcov) ;
	eta = w + bsq ;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	#pragma ivdep
	DLOOPA mhd[j] = eta*q->ucon[dir]*q->ucov[j] + ptot*delta(dir,j) - q->bcon[dir]*q->bcov[j] ;
}

//Calculates gas entropy
double calc_entropy(double* pr
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	double entr;
	#if(DOHELM)
	eos_mode_rhou_entr(pr, &entr);
	//entr = exp(KTOT_FACTOR * entr);
	#elif(TWO_T)
		#if(0)
		double Theta;
		//For variable entropy
		Theta = (gamma_g - 1.0) * pr[UU] / pr[RHO] * MU_G;
			#if(FULL_ENTROPY)
			entr = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO]);
			#else
			entr = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO];
			#endif
		#else
		double P = (GAMMA - 1.0) * pr[UU];
			#if(FULL_ENTROPY)
			entr = 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
			#else
			entr = P * pow(pr[RHO], -GAMMA);
			#endif
		#endif
	#else 
	double P = (GAMMA - 1.0) * pr[UU];
		#if(FULL_ENTROPY)
		entr = 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
		#else
		entr = P * pow(pr[RHO], -GAMMA);
		#endif
	#endif

	return entr;
}

/* add in (explicit) geometricc source terms to equations of motion */
void source(double * restrict ph, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU, double Dt
	#if(TWO_T)
	, double gamma_g
	#endif
)
{
    double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg;
	int j,k ;
	struct of_state q ;
    struct of_state_rad q_rad;

	get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, &q, mhd[0]
		#if(TWO_T)
		, gamma_g
		#endif
	) ;
	mhd_calc(ph, 1, &q, mhd[1]
		#if(TWO_T)
		, gamma_g
		#endif
	);
	mhd_calc(ph, 2, &q, mhd[2]
		#if(TWO_T)
		, gamma_g
		#endif
	);
	mhd_calc(ph, 3, &q, mhd[3]
		#if(TWO_T)
		, gamma_g
		#endif
	);

	#pragma ivdep
	PLOOP dU[k] = 0.;
	
	//contract mhd stress tensor with connection
	DLOOP {
		dU[UU] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][0][j];
		dU[U1] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][1][j];
		dU[U2] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][2][j];
		dU[U3] += mhd[j][k] * conn[nl[n]][index_2D(n, ii, jj, zz)][k][3][j];
	}

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

/* returns b^2 (i.e., twice magnetic pressure) */
double bsq_calc(double * restrict pr, struct of_geom * restrict geom)
{
	double ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM];
	ucon_calc(pr, geom, ucon);
	lower(ucon, geom, ucov);
	bcon_calc(pr, ucon, ucov, bcon);
	lower(bcon, geom, bcov);

	return(dot(bcon, bcov));
}

/* find ucon, ucov, bcon, bcov from primitive variables */
void get_state(double * restrict pr, struct of_geom * restrict geom, struct of_state * restrict q)
{

	/* get ucon */
	ucon_calc(pr, geom, q->ucon) ;
	lower(q->ucon, geom, q->ucov) ;
	bcon_calc(pr, q->ucon, q->ucov, q->bcon) ;
	lower(q->bcon, geom, q->bcov) ;

	#if(CALC_MDOT)
	int k;
	for (k = 0; k < NDIM; k++) {
		q->bcon[k] *= magnetic_density_scale_cpu;
		q->bcov[k] *= magnetic_density_scale_cpu;
	}
	#endif

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
	SLOOPA ucon[j] = pr[U1 + j - 1] - gamma * beta[j] / alpha;

	return ;
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

void vchar(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js,double * restrict vmax, double * restrict vmin
	#if(TWO_T)
	, double gamma_g
	#endif
)
{
	double discr,vp,vm,bsq,EE,EF,va2,cs2,cms2;
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

    #if DOHELM
    // Helmholtz EOS
    double xpres;
    eos_mode_rhou_pres_cs2 (pr, &xpres, &cs2);
    va2 = bsq/(bsq + pr[RHO] + pr[UU] + xpres);
    #else
    // Ideal gas EOS
	#if(TWO_T)
    EF = pr[RHO] + gamma_g * pr[UU];
	#else
	EF = pr[RHO] + GAMMA * pr[UU];
	#endif
    EE = bsq + EF ;
    va2 = bsq/EE ;
	#if(TWO_T)
    cs2 = gamma_g *(gamma_g - 1.)* pr[UU] /EF ;
	#else
	cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / EF;
	#endif
    #endif

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
	//cms2 = 0.95;
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

