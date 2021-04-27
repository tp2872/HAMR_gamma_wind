#include "decs.h"

/***********************************************************************************************/
/***********************************************************************************************
  primtoflux():
  ---------
   --  calculate fluxes in direction dir, 
        
***********************************************************************************************/

void primtoflux(double * restrict pr, struct of_state * restrict q, struct of_state_rad * restrict q_rad, int dir, struct of_geom * restrict geom, double * restrict flux, double gamma_g)
{
	int j,k ;

	/* particle number flux */
	flux[RHO] = pr[RHO]*q->ucon[dir] ;

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc(pr, dir, q, &flux[UU], gamma_g) ;
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

	#if(TWO_T)
	/* Flux of Entropy */
	flux[ENTRE] = flux[RHO] * pr[ENTRE];
	flux[ENTRI] = flux[RHO] * pr[ENTRI];
	#endif

	//Entropy advection
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (gamma_g - 1.) * log((gamma_g - 1.) * pr[UU] * pow(pr[RHO], -gamma_g));
	#else
	flux[KTOT] = flux[RHO] * (gamma_g - 1.) * pr[UU] * pow(pr[RHO], -gamma_g);
	#endif
    
	for (k = 0; k < NPR_U; k++) flux[k] *= geom->g;
	#if(TWO_T)
	flux[ENTRE] *= geom->g;
	flux[ENTRI] *= geom->g;
	#endif
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
void mhd_calc(double * restrict pr, int dir, struct of_state * restrict q, double * restrict mhd, double gamma_g)
{
	int j ;
	double r,u,P,w,bsq,eta,ptot ;

    r = pr[RHO] ;
    u = pr[UU] ;
    
    #if DOHELM
    // Helmholtz EOS
    eos_mode_rhou_pres (r, u, &P);
	#else
    // Ideal gas EOS
	P = (gamma_g - 1.) * u;
    #endif
    
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
void source(double * restrict ph, struct of_geom * restrict geom, int n, int ii, int jj, int zz, double * restrict dU, double Dt, double gamma_g)
{
    double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], Tg;
	int j,k ;
	struct of_state q ;
    struct of_state_rad q_rad;

	get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, &q, mhd[0], gamma_g) ;
	mhd_calc(ph, 1, &q, mhd[1], gamma_g) ;
	mhd_calc(ph, 2, &q, mhd[2], gamma_g) ;
	mhd_calc(ph, 3, &q, mhd[3], gamma_g) ;

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

/* Add implicit radiation 4-force source term to equations of motion */
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

	#if(DOKTOT)
	Tg = (GAMMA - 1.)*(ph[UU]) / (ph[RHO]);
		#if(FULL_ENTROPY)
		dU[KTOT] = -1. / Tg * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
		#else
			#if(TWO_T)
				#if(FIXEDGAMMA)
				double dK_dS = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0); //Multiply the next line with this to get evolution for K=P/rho^gamma instead of S=1/(gamma-1)*log(P/rho^gamma)
				dU[KTOT] = -dK_dS * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
				#else
				fprintf(stderr, "Source rad is not fully implemented yet! \n");
				#endif
			#else
		double dK_dS = (GAMMA - 1.)/ pow(ph[RHO], GAMMA -1.0); //Multiply the next line with this to get evolution for K=P/rho^gamma instead of S=1/(gamma-1)*log(P/rho^gamma)
		dU[KTOT] = -dK_dS * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
		#endif
		#endif
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}
//gamg*(ue+ui)=gami*ui+game*ue
//gamg=(gami*ui+game*ue)/(ue+ui)

//Calculate radiation 4-force
void calc_Gcon(double * restrict ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM]) {
	int i;
	double lambda, Tg, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], arad;
	kappa_abs =  calc_kappa_abs(ph);
	kappa_emmit = calc_kappa_emmit(ph);
	kappa_es = calc_kappa_es(ph);
	arad = ARAD / (ENERGY_DENSITY_SCALE) * pow(MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.);

	Tg = (GAMMA - 1.)*ph[UU] / ph[RHO];
	lambda = kappa_emmit * arad* pow(Tg, 4.); //in units of erg/s/cm^3
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs*R_dot_ucon[i] + lambda*ucon[i]) - kappa_es*(R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3])*ucon[i]);
	}
}

//Calculate total absorption opacity
double calc_kappa_abs(double* ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	#if (DOHELM)
	double Tg;
	eos_mode_rhou_temp(ph[RHO], ph[UU], &Tg);
	#else
	double Tg = fabs(MU_G * MH_CGS * (GAMMA - 1.) * (ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * ph[RHO] * MASS_DENSITY_SCALE));
	#endif
	double Tr = fabs(pow(ph[UU_RAD] * ENERGY_DENSITY_SCALE / ARAD, 0.25));

	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO] * MASS_DENSITY_SCALE) * pow(Tg, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * ph[RHO] * MASS_DENSITY_SCALE * (Z_AB / 0.02) * Ye * pow(Tg, -1.7) * pow(Tr, -3.);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Tg, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Tg));
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Tg, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Tg)) * (1. + 4.4 * pow(10., -10.) * Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	//kappa_abs = kappa_bf;//1.7 * pow(10., -25.) * pow(Tg, -7. / 2.) * pow(MH_CGS, -2.);

	return(kappa_abs * ph[RHO] * MASS_DENSITY_SCALE * R_G_CGS);
}

//Calculate total emmission opacity
double calc_kappa_emmit(double* ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	#if (DOHELM)
	double Tg;
	eos_mode_rhou_temp(ph[RHO], ph[UU], &Tg);
	#else
	double Tg = fabs(MU_G * MH_CGS * (GAMMA - 1.) * (ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * ph[RHO] * MASS_DENSITY_SCALE));
	#endif
	double Tr = fabs(pow(ph[UU_RAD] * ENERGY_DENSITY_SCALE / ARAD, 0.25));

	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO] * MASS_DENSITY_SCALE) * pow(Tg, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * ph[RHO] * MASS_DENSITY_SCALE * (Z_AB / 0.02) * Ye * pow(Tg, -4.7);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Tg, -3.5)  * log(1. + 1.6 * (Tg / Tg));
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Tg, -3.5) * log(1. + 1.6 * (Tg / Tg)) * (1. + 4.4 * pow(10., -10.) * Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	//kappa_abs = kappa_bf;//1.7 * pow(10., -25.) * pow(Tg, -7. / 2.) * pow(MH_CGS, -2.);

	return(kappa_abs * ph[RHO] * MASS_DENSITY_SCALE * R_G_CGS);
}
//Calculate total (electron) scattering opacity
double calc_kappa_es(double * restrict ph) {
	double kappa_es;
	#if (DOHELM)
	double Tg;
	eos_mode_rhou_temp(ph[RHO], ph[UU], &Tg);
	#else
	double Tg = MU_G*MH_CGS*(GAMMA - 1.)*(ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS*ph[RHO] * MASS_DENSITY_SCALE);
	#endif
	kappa_es = 0.2*(1 + X_AB) / (1. + pow(Tg / (4.5*pow(10., 8.)), 0.86));
	kappa_es = 0.2*(1 + X_AB);
	return(kappa_es* (ph[RHO] * MASS_DENSITY_SCALE)* R_G_CGS);
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
	SLOOPA ucon[j] = pr[U1 + j - 1] - gamma * beta[j] / alpha;

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
		fprintf(stderr, "\nucon_calc_rad(): gamma_rad failure \n");
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

void vchar(double * restrict pr, struct of_state * restrict q, struct of_geom * restrict geom, int js,double * restrict vmax, double * restrict vmin, double gamma_g)
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
    eos_mode_rhou_pres_cs2 (pr[RHO], pr[UU], &xpres, &cs2);
    va2 = bsq/(bsq + pr[RHO] + pr[UU] + xpres);
    #else
    // Ideal gas EOS
    EF = pr[RHO] + gamma_g * pr[UU];
    EE = bsq + EF ;
    va2 = bsq/EE ;
    cs2 = gamma_g *(gamma_g - 1.)* pr[UU] /EF ;
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

//Calculate radiative wave velocity
void vchar_rad(double * restrict pr, struct of_state* restrict q, struct of_state_rad * restrict q_rad, struct of_geom * restrict geom, int js, double * restrict vmax, double * restrict vmin, double dx){
	double discr, vp, vm, tau, kappa_tot, crad2, cmin_rad, cmax_rad, cmin_mhd, cmax_mhd;
	double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	/*Do preliminary calculations*/
	#pragma ivdep
	DLOOPA Acov[j] = 0.;
	Acov[js] = 1.;
	raise(Acov, geom, Acon);

	#pragma ivdep
	DLOOPA Bcov[j] = 0.;
	Bcov[0] = 1.;
	raise(Bcov, geom, Bcon);

	Asq = dot(Acon, Acov);
	Bsq = dot(Bcon, Bcov);
	AB = dot(Acon, Bcov);

	/* find radiation wave speed */
	crad2 = 1.0 / 3.0;

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

	Au = dot(Acov, q_rad->ucon);
	Bu = dot(Bcov, q_rad->ucon);

	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;

	A = Bu2 - (Bsq + Bu2)*crad2;
	B = 2.*(AuBu - (AB + AuBu)*crad2);
	C = Au2 - (Asq + Au2)*crad2;

	discr = B*B - 4.*A*C;
	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		fprintf(stderr, "\n\t %g %g %g %g %g\n", A, B, C, discr, crad2);
		fprintf(stderr, "\n\t q->ucon_rad1: %g %g %g %g\n", q_rad->ucon[0], q_rad->ucon[1],
			q_rad->ucon[2], q_rad->ucon[3]);
		fprintf(stderr, "\n\t Acon: %g %g %g %g\n", Acon[0], Acon[1],
			Acon[2], Acon[3]);
		fprintf(stderr, "\n\t Bcon: %g %g %g %g\n", Bcon[0], Bcon[1],
			Bcon[2], Bcon[3]);
		fail(FAIL_VCHAR_DISCR);
		exit(0);
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	if (vp > vm) {
		cmax_rad = vp;
		cmin_rad = vm;
	}
	else {
		cmax_rad = vm;
		cmin_rad = vp;
	}

	/* find radiation wave speed */
	kappa_tot = calc_kappa_abs(pr) + calc_kappa_es(pr);
	tau = kappa_tot * sqrt(geom->gcov[js][js]) * dx;
	crad2 = MY_MIN(pow(4. / (3. * tau), 2.), 0.999999999);

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
	Au = dot(Acov, q->ucon);
	Bu = dot(Bcov, q->ucon);
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		fprintf(stderr, "\n\t %g %g %g %g %g\n", A, B, C, discr, crad2);
		fprintf(stderr, "\n\t q->ucon_rad2: %g %g %g %g\n", q_rad->ucon[0], q_rad->ucon[1],
			q_rad->ucon[2], q_rad->ucon[3]);
		fprintf(stderr, "\n\t Acon: %g %g %g %g\n", Acon[0], Acon[1],
			Acon[2], Acon[3]);
		fprintf(stderr, "\n\t Bcon: %g %g %g %g\n", Bcon[0], Bcon[1],
			Bcon[2], Bcon[3]);
		fail(FAIL_VCHAR_DISCR);
		exit(0);
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	if (vp > vm) {
		cmax_mhd = vp;
		cmin_mhd = vm;
	}
	else {
		cmax_mhd = vm;
		cmin_mhd = vp;
	}

	/*Set velocity as minimum of optically thin and optically thick limit*/
	*vmax = MY_MIN(cmax_mhd, cmax_rad);
	*vmin = MY_MAX(cmin_mhd, cmin_rad);

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

#if(TWO_T)
//Calculate fraction of heat that goes into electrons on ions based on temperature ratio at previous timestep: 
double calc_delta(double* restrict ph, double bsq) {
	double fel, c1, c2, c3, Te, Ti, beta, ratio, delta;

	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * ph[ENTRE]) * pow(ph[RHO], GAMMAE));
		Ti = fabs(exp((gami - 1.0) * ph[ENTRI]) * pow(ph[RHO], GAMMA));
		#else
		Te = fabs(ph[ENTRE] * pow(ph[RHO], GAMMAE));
		Ti = fabs(ph[ENTRI] * pow(ph[RHO], GAMMA));
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * exp(pr[ENTRE]), 2. / 3.)) - 1.0)) / (MU_E * MASS_RATIO);
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * exp(pr[ENTRI]), 2. / 3.)) - 1.0)) / MU_I;
		#else
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * pr[ENTRE], 2. / 3.)) - 1.0)) / (MU_E * MASS_RATIO);
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * pr[ENTRI], 2. / 3.)) - 1.0)) / MU_I;
		#endif
	#endif

	ratio = fabs((Te * MU_E) / (Ti * MU_I));
	c1 = 0.92;
	if ((Ti * MU_I) > (Te * MU_E)) {
		c2 = 1.6 * ratio;
		c3 = 18.0 - 5.0 * log10(ratio);
	}
	else {
		c2 = 1.2 * ratio;
		c3 = 18.0;
	}

	beta = (Te + Ti) / (0.5 * bsq);
	if (!isfinite(beta)) beta = 10000.0;
	fel = c1 * (c2 * c2 + pow(beta, 2.0 + 0.2 * log10(ratio))) / (c3 * c3 + pow(beta, 2.0 + 0.2 * log10(ratio))) * sqrt((MH_CGS / ME_CGS) * (MU_I * Ti) / (MU_E * Te)) * exp(-1.0 / beta);

	//Calculate delta
	delta = 1. / (1. + fel);

	return delta;
}

void heating(double* ph, struct of_state* q)
{
	double Theta_e, Theta_i, u_e, u_i, dis, ughat;
	double fel, game, gami, pgas;

	fprintf(stderr, "Function heating not implemented! \n");

	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs(exp((game - 1.0) * ph[ENTRE])* pow(ph[RHO], game) * (MU_E * MASS_RATIO));
		Theta_i = fabs(exp((gami - 1.0) * ph[ENTRI])* pow(ph[RHO], gami) * MU_I);
		#else
		Theta_e = fabs(ph[ENTRE] * pow(ph[RHO], game) * (MU_E * MASS_RATIO));
		Theta_i = fabs(ph[ENTRI] * pow(ph[RHO], gami) * MU_I);
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * exp(pr[ENTRE]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * exp(pr[ENTRI]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * pr[ENTRE], 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + pow(25.0 * pr[RHO] * pr[ENTRI], 2. / 3.)) - 1.0));
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif
	
	//Calculate total gas pressure and internal energies
	pgas = (Theta_e / (MU_E * MASS_RATIO) + Theta_i / MU_I) * ph[RHO];

	//Calculate internal energy
	u_e = (Theta_e / (MU_E * MASS_RATIO)) * ph[RHO] / (game - 1.0);
	u_i = (Theta_i / MU_I) * ph[RHO] / (gami - 1.0);

	//Calculate which fraction goes into electrons
	//fel = 1.0 / (1.0 + calc_delta(ph, q, Theta_e / MASS_RATIO, Theta_i, pgas));

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i); 

	//Calculate dissipation: ph[UU] is allways positive (guaranteed by utoprim)
	dis = max(ph[UU] - ughat, 0.);

	//Update internal energies
	u_e += fel * dis;
	u_i += (1. - fel) * dis;

	if (u_e < 0.01 * u_i) {
		ughat = u_e + u_i;
		u_e = 0.01 * ughat;
		u_i = 0.99 * ughat;
	}
	else if (u_i < 0.01 * u_e) {
		ughat = u_i + u_e;
		u_i = 0.01 * ughat;
		u_e = 0.99 * ughat;
	}
	if (isnan(u_i))fprintf(stderr, "nanerror: %f \n", u_i);
	
	// convert back to entropy 
	#if(FIXEDGAMMA)
		#if(FULL_ENTROPY)
		ph[ENTRE] = 1.0 / (game - 1.) * log((game - 1.0) * ue * pow(ph[RHO], -game));
		ph[ENTRI] = 1.0 / (gami - 1.) * log((gami - 1.0) * ui * pow(ph[RHO], -gami));
		#else
		ph[ENTRE] = (game - 1.0) * u_e * pow(ph[RHO], -game);
		ph[ENTRI] = (gami - 1.0) * u_i * pow(ph[RHO], -gami);
		#endif
	#else
		//Calculate Theta
		//u_o_rho=theta_e/MU/( (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e)-1)
		double u_o_rho = u_e / ph[RHO];
		Theta_e = 1.0 / 30.0 * (sqrt(25.0 * (MU_E * MASS_RATIO) * (MU_E * MASS_RATIO) * u_o_rho * u_o_rho + 180.0 * MU_E * u_o_rho + 36.0) + 5.0 * (MU_E * MASS_RATIO) * u_o_rho - 6.0);
		u_o_rho = u_i / ph[RHO];
		Theta_i = 1.0 / 30.0 * (sqrt(25.0 * MU_I * MU_I * u_o_rho * u_o_rho + 180.0 * MU_I * u_o_rho + 36.0) + 5.0 * MU_I * u_o_rho - 6.0);

		#if(FULL_ENTROPY)
		ph[ENTRE] = pow(Theta_e, 1.5) * pow(Theta_e + 0.4, 1.5) / rho;
		ph[ENTRI] = pow(Theta_i, 1.5) * pow(Theta_i + 0.4, 1.5) / rho;
		#else
		ph[ENTRE] = log(pow(Theta_e, 1.5) * pow(Theta_e + 0.4, 1.5) / rho);
		ph[ENTRI] = log(pow(Theta_I, 1.5) * pow(Theta_i + 0.4, 1.5) / rho);
		#endif
	#endif

	return;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy and gas density
double calc_gamma_gas_conserved(double*  S, double rho) {
	double gamg, game, gami, Theta_e, Theta_i;
	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)) * (MU_E * MASS_RATIO)); //Actually theta_e=MU_E*Te meant
		Theta_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(S[0] * pow(rho, game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(S[1] * pow(rho, gami - 1.0) * MU_I);
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
	fprintf(stderr, "Var gamma not implemented yet! \n")
		#if(FULL_ENTROPY)
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * rho * exp(S[0]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * rho * exp(S[1]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * rho * S[0], 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * rho * S[1], 2. / 3.)) - 1.0));
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	if (!isfinite(gamg) || gamg > 2.0 || gamg < 1.0) fprintf(stderr, "Gamma_error_conserved: %f %f %f %f \n", gamg, log10(S[0]), log10(S[1]), log10(rho));

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
double calc_gamma_gas_prim(double* pr) {
	double gamg, game, gami, Theta_e, Theta_i;
	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(pr[ENTRE] * pow(pr[RHO], game - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(pr[ENTRI] * pow(pr[RHO], gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(pr[ENTRE] * pow(pr[RHO], game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(pr[ENTRI] * pow(pr[RHO], gami - 1.0) * MU_I);
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
	fprintf(stderr, "Var gamma not implemented yet! \n")
		#if(FULL_ENTROPY)
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * pr[RHO] * exp(pr[ENTRE]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * pr[RHO] * exp(pr[ENTRI]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * pr[RHO] * pr[ENTRE], 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * pr[RHO] * pr[ENTRI], 2. / 3.)) - 1.0));
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	if (!isfinite(gamg) || gamg > 1.00001*GAMMA || gamg < 0.99999*GAMMAE) {
		fprintf(stderr, "Gamma_error_prim: %f %f %f %f %f \n", gamg, log10(pr[ENTRE]), log10(pr[ENTRI]), log10(pr[RHO]), log10(pr[UU]));
	}
	
	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy, gas density and w=W*(1-vsq)
double calc_gamma_gas_w(double* S, double rho, double w, double fel ) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante, S_new[2];

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(FIXEDGAMMA)   
		game = GAMMAE;
		gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
	fprintf(stderr, "Var gamma not implemented yet! \n")
		#if(FULL_ENTROPY)
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * exp(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * exp(S[1]), 2. / 3.)) - 1.0) / (MU_I));
		#else
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * S[0], 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * S[1], 2. / 3.)) - 1.0) / (MU_I));
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = max(quantg / gamg - ughat, 0.);

	//Update internal energy of electrons
	u_e += fel * dis;

	//quante = game / (game - 1.0) * pe; //quant=(gam)/(gam-1)*p
	//quante = game / (game - 1.0) * (Te*MU_E)/MU_E*rho;
	//quante*MU_E/rho = (10.0 + 20.0 * x) / (6.0 + 15.0 * x) / ((10.0 + 20.0 * x) / (6.0 + 15.0 * x) - 1.0) * (x); x=MU_E*Te

	quante = game * u_e; //quant=(gam)/(gam-1)*p
	if (quante > 0.99 * quantg) quante = 0.99 * quantg;
	if (quante < 0.01 * quantg) quante = 0.01 * quantg;

	quanti = quantg - quante;
	#if(FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
	Te = pe / rho;
	Ti = pi / rho;
	#else
	//Use analytical inversions
	double C = rho / (MU_E * MASS_RATIO);
	pe = -(0.25 * (C - 0.5 * quante)) + 0.0559017 * sqrt(20.0 * C * C + 44.0 * C * quante + 5.0 * quante * quante);
	C = rho / MU_I;
	pi = -(0.25 * (C - 0.5 * quanti)) + 0.0559017 * sqrt(20.0 * C * C + 44.0 * C * quanti + 5.0 * quanti * quanti);
	Te = pe / rho;
	Ti = pi / rho;
	game = (10.0 + 20.0 * Te * (MU_E * MASS_RATIO)) / (6.0 + 15.0 * Te * (MU_E * MASS_RATIO));
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	//if (!isfinite(gamg) || gamg > 1.00001 * GAMMA || gamg < 0.99999 * GAMMAE) fprintf(stderr, "Gamma_error_w: %f %f %f %f\n", gamg, log10(Te), log10(Ti), log10(fabs(rho)));
	
	return gamg;
}

//Update electron and ion entropy based on found w in Newton Raphson solver
double set_S_w(double* S, double rho, double w, double fel) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante, S_new[2];

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(FIXEDGAMMA)   
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
	fprintf(stderr, "Var gamma not implemented yet! \n")
		#if(FULL_ENTROPY)
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * exp(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * exp(S[1]), 2. / 3.)) - 1.0) / (MU_I));
		#else
		Te = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * S[0], 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + pow(25.0 * rho * S[1], 2. / 3.)) - 1.0) / (MU_I));
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = max(quantg / gamg - ughat, 0.);

	//Update internal energy of electrons
	u_e += fel * dis;

	//quante = game / (game - 1.0) * pe; //quant=(gam)/(gam-1)*p
	//quante = game / (game - 1.0) * (Te*MU_E)/MU_E*rho;
	//quante*MU_E/rho = (10.0 + 20.0 * x) / (6.0 + 15.0 * x) / ((10.0 + 20.0 * x) / (6.0 + 15.0 * x) - 1.0) * (x); x=MU_E*Te

	quante = game * u_e; //quant=(gam)/(gam-1)*p
	if (quante > 0.99 * quantg) quante = 0.99 * quantg;
	if (quante < 0.01 * quantg) quante = 0.01 * quantg;

	quanti = quantg - quante;

	#if(FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
		#if(FULL_ENTROPY)
		S[0] = 1.0 / (game - 1.0) * log(pe * pow(rho, -game));
		S[1] = 1.0 / (gami - 1.0) * log(pi * pow(rho, -gami));
		#else
		S[0] = pe * pow(rho, -game);
		S[1] = pi * pow(rho, -gami);
		#endif
	Te = pe / rho;
	Ti = pi / rho;
	#else
	//Use analytical inversions
	double C = rho / (MU_E * MASS_RATIO);
	pe = -(0.25 * (C - 0.5 * quante)) + 0.0559017 * sqrt(20.0 * C * C + 44.0 * C * quante + 5.0 * quante * quante);
	C = rho / MU_I;
	pi = -(0.25 * (C - 0.5 * quanti)) + 0.0559017 * sqrt(20.0 * C * C + 44.0 * C * quanti + 5.0 * quanti * quanti);
	Te = pe / rho;
	Ti = pi / rho;
	game = (10.0 + 20.0 * Te * (MU_E * MASS_RATIO)) / (6.0 + 15.0 * Te * (MU_E * MASS_RATIO));
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
		#if(FULL_ENTROPY)
		S[0] = pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
		S[1] = pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho;
		#else
		S[0] = log(pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho);
		#endif
	#endif

	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	if (!isfinite(gamg) || gamg > 1.00001 * GAMMA || gamg < 0.99999 * GAMMAE) fprintf(stderr, "Gamma_error_w2: %f \n", gamg);
	
	return gamg;
}

double set_S_u(double* S, double rho, double u) {
	
}

void Coulomb_exchange(double* restrict ph, double Dt)
{
	double u_e, u_i, m_e, m_i, n_e, n_i, th_e, th_i, entr_e, entr_i, q_coulomb, dU_coulomb;
	double game, gamp, frac_uu;
	double Dt_sub, Dt_elapsed;
	int iter_uu, cond;

	n_e = ph[RHO] / MH_CGS / MU_E;
	n_i = ph[RHO] / MH_CGS / MU_I;

	entr_e = ph[ENTRE];
	entr_i = ph[ENTRI];

	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gamp = GAMMA;
	#else     // variable gamma: Sadowski+17 & Chael+19
	m_e = ME_CGS;
	m_i = MH_CGS * MU_I;
	#endif

	// Initialize the subscycle times step
	Dt_sub = Dt;
	Dt_elapsed = 0.;
	iter_uu = 0;
	// Start of subcycle: in each subcycle the internal temperature is allowed to be at most halved
	do {
		#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		u_e = 1. / (game - 1.) * entr_e * pow(ph[RHO], game);
		u_i = 1. / (gamp - 1.) * entr_i * pow(ph[RHO], gamp);

		th_e = (game - 1.) * u_e / n_e / ME_CGS;
		th_i = (gamp - 1.) * u_i / n_i / MH_CGS;
		#else     // variable gamma: Sadowski+17 & Chael+19
		u_e = calc_ufromsrho(entr_e, ph[RHO], ELECTRONS);
		u_i = calc_ufromsrho(entr_i, ph[RHO], IONS);

		th_e = calc_thetafromsnm(entr_e, n_e, m_e);
		th_i = calc_thetafromsnm(entr_i, n_i, m_i);
		#endif

		q_coulomb = calc_CoulombCoupling(n_e, th_e, th_i);   // heating rate for Coulomb Coupling
		dU_coulomb = q_coulomb * Dt_sub;
		if (dU_coulomb < 0.5 * u_i) {

			u_e += dU_coulomb;
			u_i -= dU_coulomb;

			cond = 0;
		}
		else {
			Dt_sub *= 0.5 * u_i / dU_coulomb;   // adjusted subcycle time interval
			Dt_elapsed += Dt_sub;

			if (Dt_elapsed >= Dt) {
				frac_uu = 1. - (Dt_elapsed - Dt) / Dt_sub;  // fractional energy would be exchanged.
				cond = 0;
			}
			else {
				Dt_sub = Dt - Dt_elapsed;   // the time interval for the next subcycle
				frac_uu = 1.;
				iter_uu++;
				if (iter_uu > 10) {          // to avoid run-away process, the maximum iteration is set.
					cond = 0;
				}
				else {
					cond = 1;
				}
			}

			u_e += 0.5 * u_i * frac_uu;
			u_i -= 0.5 * u_i * frac_uu;
		}

		#if(FIXEDGAMMA)
		entr_e = (game - 1.) * u_e * pow(ph[RHO], -game);
		entr_i = (gamp - 1.) * u_i * pow(ph[RHO], -gamp);
		#else
		entr_e = calc_sfromrhou(ph[RHO], u_e, ELECTRONS);
		entr_i = calc_sfromrhou(ph[RHO], u_i, IONS);
		#endif
	} while (cond);

	ph[ENTRE] = entr_e;
	ph[ENTRI] = entr_i;
}

/* Entropy related functions for variable gamma (Sadowski+17 & Chael+19) */
double calc_sfromrhou(double rho, double uint, int type) {
	double mass, numd, res, theta; //Temperature;

	if (type == IONS) {
		mass = MH_CGS * MU_I;            // mass of particle in code unit
		numd = rho / (MH_CGS * MU_I);      // number desity in code unit
	}
	else if (type == ELECTRONS) {
		mass = ME_CGS*MU_E;
		numd = rho / (MH_CGS * MU_E);
	}
	else {
		fprintf(stderr, "error in the type of fluids \n");
		exit(1236);
	}
	//Temperature = calc_Tfromnmu(numd, mass, uint);
	//res = calc_sfromrhoT(rho, Temperature, type);
	theta = calc_thetafromnmu(numd, mass, uint);
	res = calc_sfromntheta(numd, theta);

	return res;
}

/* solves for gamma_int from the equation of state with the inconsistent gamma_int equation in Sadowski et al. 2017 (eq. A16)
 note: need to test with direct root-finding (e.g., Newton-Rapson) by Chael et al. 2018  */
double calc_thetafromnmu(double n, double m, double u) {
	double res, theta_max;

	if (u > m * n) {   // regular form of the quadratic solution
		res = (-6. * m * n + 5. * u + sqrt(36. * m * m * n * n + 180. * m * n * u + 25. * u * u)) / (30. * m * n);
	}
	else {   // this form of the quadratic solution is better for very small values of u
		res = 8 * u / (6. * m * n - 5. * u + sqrt(36. * m * m * n * n + 180. * m * n * u + 25. * u * u));
	}

	theta_max = TMAX * BOLTZ_CGS / m;
	if (isfinite(res))
	{
		res = min(res, theta_max);
	}
	else
	{
		res = theta_max;
	}

	return res;
}

/*
double calc_Tfromnmu(double n, double m, double u) {
	double k = BOLTZ;
	double T;

	if (u > m * n){   // regular form of the quadratic solution
		T = (-6.*k*m*n + 5.*k*u + k * sqrt(36.*pow(m, 2)*pow(n, 2) + 180.*m*n*u + 25.*pow(u, 2))) / (30.*pow(k, 2)*n);
	}
	else{   // this form of the quadratic solution is better for very small values of u
		T = 8 * m*u / (6.*k*m*n - 5.*k*u + k * sqrt(36.*pow(m, 2)*pow(n, 2) + 180.*m*n*u + 25.*pow(u, 2)));
	}

	return T;
}
*/

/* Entropy with smoothly transitions from theta=0 to theta=1
   Chael et al. 2018 (eq. A6) */
double calc_sfromntheta(double numd, double theta) {
	double res;

	res = numd * BOLTZ_CGS * log(sqrt(theta * theta * theta * (theta + 0.4) * (theta + 0.4) * (theta + 0.4)) / numd);  // entropy per unit volume
	return res;
}

/*
double calc_sfromrhoT(double rho, double Temperature, int type) {
	double theta, numd, res;

	if (type == IONS)
	{
		numd = rho / M_PROTON / MU_I;     // number density  in code unit
		theta = BOLTZ * Temperature / MU_I / M_PROTON;    // dimensionless temperature
	}
	else if (type == ELECTRONS)
	{
		numd = rho / M_PROTON / MU_E;
		theta = BOLTZ * Temperature / M_ELECTRON;
	}
	else {
		fprintf(stderr, "error in the type of fluids \n");
		exit(1236);
	}

	res = numd * BOLTZ*log(sqrt(theta*theta*theta*(theta + 0.4)*(theta + 0.4)*(theta + 0.4)) / numd);  // entropy per unit volume
	return res;
}
*/

double calc_thetafromsnm(double s, double numd, double mass)
{
	double theta, ex;
	double theta_max, rhs;

	theta_max = TMAX * BOLTZ_CGS / mass;

	ex = exp(s / numd / BOLTZ_CGS);   // s is the entropy per unit volume -> /numd : per particle

	if (isfinite(ex))
	{
		rhs = cbrt(numd * ex * numd * ex);
		theta = 0.2 * (-1. + sqrt(1. + 25. * rhs));
	}
	else
	{
		theta = theta_max;
	}

	if (!isfinite(theta))
		theta = theta_max;

	return theta;
}

double calc_ufromsrho(double s, double rho, int type)
{
	double numd, mass, theta, res;

	if (type == IONS)
	{
		mass = MH_CGS * MU_I;
		numd = rho / (MH_CGS * MU_I);      // number desity in real unit
	}
	else if (type == ELECTRONS)
	{
		mass = ME_CGS * MU_E;
		numd = rho / (MH_CGS * MU_I);
	}
	else {
		fprintf(stderr, "error in the type of fluids \n");
		exit(1236);
	}

	theta = calc_thetafromsnm(s, numd, mass);

	res = numd * mass * theta * (6. + 15. * theta) / (4. + 5. * theta);  // approximated internal energy in Sadowski+15 (A15).
														 // this should be replaced by the eq (A14) if we want to solve it directly by root-finding method. 

	return res;
}

double calc_Tfromtheta(double theta, int type)
{
	double mass, res;

	if (type == IONS) mass = MH_CGS * MU_I;
	else if (type == ELECTRONS) mass = ME_CGS * MU_E;;
 
	res = mass * theta / BOLTZ_CGS;

	return res;
}

double source_Coulomb(double *p){
	double th_mean, th_sum, Theta_e, Theta_i, coeff, ne_cgs, T_e, T_i;
	double K2e, K2i, K0, K1;
	double theta_min = 1.e-2;
	double coulog = 20.;   // Coulomb logarithm ( ln Lambda )
	double res;

	#if(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(p[ENTRI] * pow(p[RHO], GAMMA - 1.0)) * MU_I);
		#else
		Theta_e = fabs(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(p[ENTRI] * pow(p[RHO], GAMMA - 1.0) * MU_I);
		#endif
	#else     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * p[RHO] * exp(p[ENTRE]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * p[RHO] * exp(p[ENTRI]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 * pow(25.0 * p[RHO] * p[ENTRE], 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 * pow(25.0 * p[RHO] * p[ENTRI], 2. / 3.)) - 1.0));
		#endif
	#endif

	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	//note that average number density in Sadowski+17 (eq (20)) is assumed to be n_ave = ne_cgs.this can be updated 
	ne_cgs = p[RHO] * MASS_DENSITY_SCALE / (MU_E * MH_CGS);    // calculation in cgs unit

	T_e = Theta_e / BOLTZ_CGS * (ME_CGS * C_CGS * C_CGS);
	T_i = Theta_i / BOLTZ_CGS * (MH_CGS * C_CGS * C_CGS);

	coeff *= ne_cgs * ne_cgs * (T_i - T_e);

	th_sum = Theta_e + Theta_i;
	th_mean = Theta_e * Theta_i / (Theta_e + Theta_i);


	if (Theta_i < theta_min && Theta_e < theta_min) // approximated equations at small theta
	{
		res = coeff / sqrt(0.5 * M_PI * th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (Theta_i < theta_min)
	{
		//bessel function
		K2e = bessk(2.0, 1. / Theta_e);
		res = coeff / K2e / exp(1. / Theta_e) * sqrt(Theta_e) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (Theta_e < theta_min)
	{
		//bessel function
		K2i = bessk(2.0, 1. / Theta_i);

		res = coeff / K2i / exp(1. / Theta_i) * sqrt(Theta_i) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else // general form in Sadowski+17 (eq 20)
	{
		//bessel functions
		K2e = bessk(2.0, 1. / Theta_e);
		K2i = bessk(2.0, 1. / Theta_i);
		K0 = bessk0(1.0 / th_mean);
		K1 = bessk1(1.0 / th_mean);

		res = coeff / (K2e * K2i) * ((2. * th_sum * th_sum + 1.) / th_sum * K1 + 2. * K0);
	}

	if (!isfinite(res)) res = 0.;

	res = res / ENERGY_DENSITY_SCALE * R_GOC_CGS;     // unit conversion from cgs to grid unit
	return res;
}

double calc_CoulombCoupling(double n_e, double theta_e, double theta_i)
{
	double th_mean, th_sum, coeff, ne_cgs, T_e, T_i;
	double K2e, K2i, K0, K1;
	double theta_min = 1.e-2;
	double coulog = 20.;   // Coulomb logarithm ( ln Lambda )
	double res;

	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	/* note that average number density in Sadowski+17 (eq (20)) is assumed to be n_ave = ne_cgs.
	   this can be updated */
	ne_cgs = n_e / R_G_CGS / R_G_CGS / R_G_CGS;    // calculation in cgs unit

	T_e = theta_e * BOLTZ_CGS / ME_CGS / C_CGS / C_CGS;
	T_i = theta_i * BOLTZ_CGS / MH_CGS / C_CGS / C_CGS;

	coeff *= ne_cgs * ne_cgs * (T_i - T_e);

	th_sum = theta_e + theta_i;
	th_mean = theta_e * theta_i / (theta_e + theta_i);


	if (theta_i < theta_min && theta_e < theta_min) // approximated equations at small theta
	{
		res = coeff / sqrt(0.5 * M_PI * th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (theta_i < theta_min)
	{
		//bessel function
		#if(GSL_ENABLED)
		K2e = gsl_sf_bessel_Kn(2, 1. / theta_e);
		#else
		K2e = bessk(2, 1. / theta_e);
		#endif

		res = coeff / K2e / exp(1. / theta_e) * sqrt(theta_e) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (theta_e < theta_min)
	{
		//bessel function
		#if(GSL_ENABLED)
		K2i = gsl_sf_bessel_Kn(2, 1. / theta_i);
		#else
		K2i = bessk(2, 1. / theta_i);
		#endif

		res = coeff / K2i / exp(1. / theta_i) * sqrt(theta_i) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else // general form in Sadowski+17 (eq 20)
	{
		//bessel functions
		#if(GSL_ENABLED)
		K2e = gsl_sf_bessel_Kn(2, 1. / theta_e);
		K2i = gsl_sf_bessel_Kn(2, 1. / theta_i);
		K0 = gsl_sf_bessel_Kn(0, 1. / th_mean);
		K1 = gsl_sf_bessel_Kn(1, 1. / th_mean);
		#else
		K2e = bessk(2, 1. / theta_e);
		K2i = bessk(2, 1. / theta_i);
		K0 = bessk0(1. / th_mean);
		K1 = bessk1(1. / th_mean);
		#endif

		res = coeff / K2e / K2i * ((2. * th_sum * th_sum + 1.) / th_sum * K1 + 2. * K0);
	}

	if (!isfinite(res)) res = 0.;

	res = res / ENERGY_DENSITY_SCALE * R_GOC_CGS;     // unit conversion from cgs to grid unit
	return res;
}
#endif

#if(!GSL_ENABLED)
// Some bessel functions
double bessi0(double x) {
	double ax, ans, y;

	if ((ax = fabs(x)) < 3.75) {
		y = x / 3.75, y = y * y;
		ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
			+ y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
	}
	else {
		y = 3.75 / ax;
		ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
			+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
				+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
					+ y * 0.392377e-2))))))));
	}

	return ans;
}

double bessi1(double x) {
	double ax, ans, y;

	if ((ax = fabs(x)) < 3.75) {
		y = x / 3.75, y = y * y;
		ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
			+ y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
	}
	else {
		y = 3.75 / ax;
		ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
			- y * 0.420059e-2));
		ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
			+ y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
		ans *= (exp(ax) / sqrt(ax));
	}

	return (x < 0.0 ? -ans : ans);
}

double bessk0(double x) {
	double y, ans;

	if (x <= 2.0) {
		y = x * x / 4.0;
		ans = (-log(x / 2.0) * bessi0(x)) + (-0.57721566 + y * (0.42278420
			+ y * (0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2
				+ y * (0.10750e-3 + y * 0.74e-5))))));
	}
	else {
		y = 2.0 / x;
		ans = (exp(-x) / sqrt(x)) * (1.25331414 + y * (-0.7832358e-1
			+ y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2
				+ y * (-0.251540e-2 + y * 0.53208e-3))))));
	}

	return ans;
}

double bessk1(double x) {
	double y, ans;

	if (x <= 2.0) {
		y = x * x / 4.0;
		ans = (log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144
			+ y * (-0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1
				+ y * (-0.110404e-2 + y * (-0.4686e-4)))))));
	}
	else {
		y = 2.0 / x;
		ans = (exp(-x) / sqrt(x)) * (1.25331414 + y * (0.23498619
			+ y * (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2
				+ y * (0.325614e-2 + y * (-0.68245e-3)))))));
	}

	return ans;
}

double bessk(int n, double x) {
	int j;
	double bk, bkm, bkp, tox;

	tox = 2.0 / x;
	bkm = bessk0(x);
	bk = bessk1(x);
	for (j = 1; j < n; j++) {
		bkp = bkm + j * tox * bk;
		bkm = bk;
		bk = bkp;
	}

	return bk;
}
#endif
