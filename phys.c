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
		#if (DOHELM)
		, gpu_eos_table
		#endif
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
    eos_mode_rhou_pres (r, u, &P);
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
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	double entr;
	#if(DOHELM)
	eos_mode_rhou_entr(gpu_eos_table, pr[RHO], pr[UU], &entr);
	entr = xentr;
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

/* Radiation stress tensor, with first index up, second index down */
void mhd_calc_rad(double * restrict pr, int dir, struct of_state_rad * restrict q_rad, double * restrict mhd_rad)
{
    int j;
    /* single row of mhd stress tensor, first index up, second index down */
    #pragma ivdep
    DLOOPA mhd_rad[j] = (4./3.)*pr[UU_RAD]*q_rad->ucon[dir] * q_rad->ucov[j] + (1./3.)*pr[UU_RAD]*delta(dir, j);
}

double calc_Te(double* ph) {
	double Te;

	#if(TWO_T)
		#if(CONSTANTGAMMA)
			#if(FULL_ENTROPY)
			Te = exp((GAMMA - 1.0) * ph[ENTRE]) * pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Te = exp((GAMMAE - 1.0) * ph[ENTRE]) * pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMAE - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMAE - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY_VARGAMMA)
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * exp(ph[ENTRE]), 2. / 3.)) - 1.0)/ (MU_E*MASS_RATIO);
			#else
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * ph[ENTRE]) - 1.0) / (MU_E * MASS_RATIO);
			#endif
		#endif
	#else
	Te = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Te;
}

double calc_Ti(double* ph) {
	double Ti;

	#if(TWO_T)
		#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Ti = exp((GAMMA - 1.0) * ph[ENTRI]) * pow(p[nl[n]][index_3D(n, i, j, z)][RHO], GAMMA - 1.0);
			#else
			Ti = ph[ENTRI] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY_VARGAMMA)
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * exp(ph[ENTRI]), 2. / 3.)) - 1.0) / MU_I;
			#else
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * ph[ENTRI]) - 1.0) / MU_I;
			#endif
		#endif
	#else
	Ti = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Ti;
}

double calc_Tr(double* ph, double ucon[NDIM], double ucon_rad[NDIM], double ucov[NDIM]
	#if(P_NUM)
	, double *exp_xi
	#endif
) {
	double Tr, u_dot_urad, u_dot_u, Ehat;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	u_dot_urad = ucov[0] * ucon_rad[0] + ucov[1] * ucon_rad[1] + ucov[2] * ucon_rad[2] + ucov[3] * ucon_rad[3];
	u_dot_u = ucon[0] * ucov[0] + ucon[1] * ucov[1] + ucon[2] * ucov[2] + ucon[3] * ucov[3];
	Ehat = energy_density_scale * ((4. / 3.) * ph[UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * ph[UU_RAD] * u_dot_u);

	//Get radiation temperature either assuming blackbody or diluted blackbody
	#if(P_NUM)
	double  Nhat;
	Nhat = fabs(-ph[PHOTON] * mass_density_scale * u_dot_urad);
	//Tr = Ehat / (BOLTZ_CGS * Nhat * (3. - 2.449724 * Nhat * Nhat * Nhat * Nhat / (CK_CGS * Ehat * Ehat * Ehat)));
	//Tr = Ehat / (BOLTZ_CGS * Nhat * (0.33333 + 0.060725 / (0.646756 + 0.121982 * CK_CGS * Ehat * Ehat * Ehat / (Nhat * Nhat * Nhat * Nhat))));
	Tr = Ehat / (BOLTZ_CGS * Nhat * 2.701178);
	exp_xi[0] = MY_MIN(1.64676 / (0.646756 + 0.121982 * CK_CGS * Ehat * Ehat * Ehat / (Nhat * Nhat * Nhat * Nhat)), 0.99);
	#else
	Tr = pow(Ehat / ARAD, 0.25);
	#endif

	return Tr;
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

/* Add implicit radiation 4-force source term to equations of motion */
void source_rad(double * restrict ph, struct of_geom * restrict geom,  double * restrict dU
	#if(TWO_T)
	, double gamma_g
	#endif
)
{
	#if(RAD_M1)
	double mhd[NDIM][NDIM], mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM+P_NUM], ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM], Tg, bsq, dK_dS;
	int j, k;
	struct of_state_rad q_rad;
	#if(TWO_T)
	double src_coulomb;
	#endif
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
	bcon_calc(ph, ucon, ucov, bcon);
	lower(bcon, geom, bcov);
	bsq = bcon[0] * bcov[0] + bcon[1] * bcov[1] + bcon[2] * bcov[2] + bcon[3] * bcov[3];

	calc_Gcon(ph, Gcon, ucon, ucov, q_rad.ucon, q_rad.ucov, mhd_rad, bsq
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, &(dU[PHOTON])
		#endif
	);
	lower(Gcon, geom, Gcov);

	//Add source term
	dU[UU] = Gcov[0];
	dU[U1] = Gcov[1];
	dU[U2] = Gcov[2];
	dU[U3] = Gcov[3];

	dU[UU_RAD] = -Gcov[0];
	dU[U1_RAD] = -Gcov[1];
	dU[U2_RAD] = -Gcov[2];
	dU[U3_RAD] = -Gcov[3];

	//Entropy source term
	#if(DOKTOT)
		#if (DOHELM)
		eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &dK_dS);
		#elif(TWO_T)
			#if(VARGAMMA || FIXEDGAMMA)
			double Theta, entr, C;
			//For variable entropy
			C = ph[UU] / ph[RHO] * MU_G;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY)
				dK_dS = (1.0 / Theta) * (MU_G);
				#else
				dK_dS = (ph[KTOT] / Theta) * (MU_G);
				#endif
			#else
				#if(FULL_ENTROPY)
				dK_dS = ph[RHO] / (gamma_g - 1.) * ph[UU]);
				#else
				dK_dS = (gamma_g - 1.) / pow(ph[RHO], gamma_g - 1.0);
				#endif
			#endif
		#else
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / (GAMMA - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0);
			#endif
		#endif
		dU[KTOT] = -dK_dS * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
	#endif

	//Electron entropy source term for radiative cooling and coulomb coupling
	#if(TWO_T)
	#if(FIXEDGAMMA || CONSTANTGAMMA)
		double dK_dS_i;
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / ((GAMMAE - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMAE - 1.) / pow(ph[RHO], GAMMAE - 1.0);
			#endif
		#elif(VARGAMMA)
			double Theta_e, Theta_i, dK_dS_i;
			//For variable entropy
			#if(FULL_ENTROPY_VARGAMMA)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(ph[RHO] * exp(ph[ENTRE])), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(ph[RHO] * exp(ph[ENTRI])), 2. / 3.)) - 1.0);
			dK_dS_i = (1.0 / Theta_i) * (MU_I);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * fabs(ph[ENTRE])) - 1.0);
			dK_dS = (2. / 3.) * (ph[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * fabs(ph[ENTRI])) - 1.0);
			dK_dS_i = (2. / 3.) * (ph[ENTRI] / Theta_i) * (MU_I);
			#endif
		#endif
		if (!isfinite(dK_dS))dK_dS = 0.0;
		if (!isfinite(dK_dS_i))dK_dS_i = 0.0;
		dU[ENTRE] = -dK_dS * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
		src_coulomb = source_Coulomb(ph);
		dU[ENTRE] += dK_dS * src_coulomb;
		dU[ENTRI] -= dK_dS_i * src_coulomb;
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}

//Calculate radiation 4-force
void calc_Gcon(double * restrict ph, double Gcon[NDIM+P_NUM], double ucon[NDIM], double ucov[NDIM], double ucon_rad[NDIM], double ucov_rad[NDIM],  double mhd_rad[NDIM][NDIM], double bsq
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double* source_photon
	#endif
) {
#if(RAD_M1)
	int i;
	double lambda, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], Tr, Te;
	#if(P_NUM)
	double exp_xi;
	#endif
	#if(COMPTON)
	double Theta_e, Theta_r, G0;
	#endif
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	//Calculate radiation temperature in rest frame of fluid
	Tr = calc_Tr(ph, ucon, ucon_rad, ucov
		#if(P_NUM)
		, &exp_xi
		#endif
	);

	kappa_abs = calc_kappa_abs(ph, bsq, Tr
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, exp_xi
		#endif
	);
	kappa_emmit = calc_kappa_emmit(ph, bsq, Tr
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, exp_xi
		#endif
	);
	kappa_es = calc_kappa_es(ph
		#if(TWO_T)
		, gamma_g
		#endif
	);

	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	//Calculate emmission rate
	lambda = kappa_emmit * (ARAD / energy_density_scale) * Te * Te * Te * Te; //in units of erg/(Rg/c)/cm^3

	//Calculate non-Compton scattering source term
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs * R_dot_ucon[i] + lambda * ucon[i]) - kappa_es * (R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]) * ucon[i]);
	}

		//Evaluate comptonization term
		#if(P_NUM || COMPTON)
		double Ehat, Nhat,u_dot_urad, u_dot_u;

		//Misc variables
		u_dot_urad = ucon[0] * ucov_rad[0] + ucon[1] * ucov_rad[1] + ucon[2] * ucov_rad[2] + ucon[3] * ucov_rad[3];
		u_dot_u = ucon[0] * ucov[0] + ucon[1] * ucov[1] + ucon[2] * ucov[2] + ucon[3] * ucov[3];
		Ehat = ((4. / 3.) * ph[UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * ph[UU_RAD] * (u_dot_u));
		#endif

		#if(P_NUM)
		Nhat = -ph[PHOTON]  * u_dot_urad;

		//Source term for photons
		source_photon[0] = -kappa_abs * Nhat + (kappa_emmit / mass_density_scale_cpu * ARAD * Te * Te * Te * Te / (BOLTZ_CGS * Te * 2.701178));
		#endif

		//Compton scattering term is added
		#if(COMPTON)
		Theta_e = Te * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
		Theta_r = Tr * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
		G0 = -kappa_es * Ehat * 4.0 * (Theta_e - Theta_r) * (1.0 + 3.683 * Theta_e + 4.0 * Theta_e * Theta_e) / ((1.0 + Theta_e));
		for (i = 0; i < NDIM; i++) Gcon[i] += ucon[i] * G0;
		#endif
	#endif
}

//Calculate total absorption opacity
double calc_kappa_abs(double* ph, double bsq, double Tr
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
	) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff, kappa_sy, Te, ne, zeta;
	double Ye = (1. + X_AB) / 2.;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif
	ne = ph[RHO] * mass_density_scale / (MU_E * MH_CGS);
	zeta = 4. * M_PI * ME_CGS * ME_CGS * ME_CGS * pow(C_CGS, 5.0) * Tr / (3.0 * E_CGS * BOLTZ_CGS * PLANCK_CGS * sqrt(bsq) * Te * Te);

	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO] * mass_density_scale) * pow(Te, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * ph[RHO] * mass_density_scale * (Z_AB / 0.02) * Ye * pow(Te, -1.7) * pow(Tr, -3.);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * mass_density_scale_cpu * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te));
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * mass_density_scale_cpu * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te)) * (1. + 4.4 * pow(10., -10.) * Te);
	kappa_sy = 1.59 * pow(10., -30.) * ne * 4. * M_PI * bsq * energy_density_scale * pow(Te, -2.) * pow(zeta, -3.) * (1. + 5.444 * pow(zeta, -0.666666) + 7.218 * pow(zeta, -4.3333333));
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	//kappa_abs = kappa_bf; // 1.7 * pow(10., -25.) * pow(fabs(Te), -7. / 2.) * pow(MH_CGS, -2.);

	if (!isfinite(kappa_abs)) kappa_abs = 0.0;
	return(kappa_abs * (ph[RHO] * mass_density_scale) * R_G_CGS);
}

//Calculate total emmission opacity
double calc_kappa_emmit(double* ph, double bsq, double Tr
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff, kappa_sy, Te, ne;
	double Ye = (1. + X_AB) / 2.;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	ne = ph[RHO] * mass_density_scale / (MU_E * MH_CGS);

	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO] * mass_density_scale) * pow(Te, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * ph[RHO] * mass_density_scale * (Z_AB / 0.02) * Ye * pow(Te, -1.7) * pow(Te, -3.);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * mass_density_scale_cpu * pow(Te, -3.5) * log(1. + 1.6);
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * mass_density_scale_cpu * pow(Te, -3.5) * log(1. + 1.6) * (1. + 4.4 * pow(10., -10.) * Te);
	kappa_sy = 1.59 * pow(10., -30.) * ne * 4. * M_PI * bsq * energy_density_scale * pow(Te, -2.);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	//kappa_abs = kappa_bf; // 1.7 * pow(10., -25.) * pow(fabs(Te), -7. / 2.) * pow(MH_CGS, -2.);

	if (!isfinite(kappa_abs)) kappa_abs = 0.0;
	return(kappa_abs * (ph[RHO] * mass_density_scale) * R_G_CGS);
}
//Calculate total (electron) scattering opacity
double calc_kappa_es(double * restrict ph
	#if(TWO_T)
	, double gamma_g
	#endif
	) {
	double kappa_es, Te;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	#endif

	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	kappa_es = 0.2 * (1 + X_AB) / (1. + pow(Te / (4.5 * pow(10., 8.)), 0.86));
	kappa_es = 0.2 * (1 + X_AB);

	if (!isfinite(kappa_es)) kappa_es = 0.0;
	return(kappa_es * (ph[RHO] * mass_density_scale) * R_G_CGS);
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
    eos_mode_rhou_pres_cs2 (pr[RHO], pr[UU], &xpres, &cs2);
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

//Calculate radiative wave velocity
void vchar_rad(double * restrict pr, struct of_state* restrict q, struct of_state_rad * restrict q_rad, struct of_geom * restrict geom, int js, double * restrict vmax, double * restrict vmin, double dx
	#if(TWO_T)
	, double gamma_g
	#endif
){
	double discr, vp, vm, tau, kappa_tot, crad2, cmin_rad, cmax_rad, cmin_mhd, cmax_mhd, bsq, Tr;
	double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;
	#if(P_NUM)
	double exp_xi;
	#endif

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

	/* find radiation wave speed in fluid frame based on optical depth */
	//Calculate optical depth
	bsq = q->bcon[0] * q->bcov[0] + q->bcon[1] * q->bcov[1] + q->bcon[2] * q->bcov[2] + q->bcon[3] * q->bcov[3];
	Tr = calc_Tr(pr, q->ucon, q_rad->ucon, q->ucov
		#if(P_NUM)
		, &exp_xi
		#endif
	);
	kappa_tot = (calc_kappa_es(pr
		#if(TWO_T)
		, gamma_g
		#endif
	) + calc_kappa_abs(pr, bsq, Tr
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, exp_xi
		#endif
	));
	tau = kappa_tot * sqrt(geom->gcov[js][js]) * dx;
	crad2 = MY_MIN(pow(4. / (3. * tau), 2.), 1./3.);

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
	double delta;
	#if(HEAT_HOWES)
	double fel, c1, c2, c3, Te, Ti, beta_i, ratio;
	
	Te = calc_Te(ph);
	Ti = calc_Ti(ph);

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

	beta_i = ((Ti) * ph[RHO]) / (0.5 * bsq);
	if (!isfinite(beta_i) || beta_i >10000.0) beta_i = 10000.0;
	fel = c1 * (c2 * c2 + pow(beta_i, 2.0 + 0.2 * log10(ratio))) / (c3 * c3 + pow(beta_i, 2.0 + 0.2 * log10(ratio))) * sqrt((MH_CGS / ME_CGS) * (MU_I * Ti) / (MU_E * Te)) * exp(-1.0 / beta_i);

	//Calculate delta
	delta = 1. / (1. + fel);
	#elif(HEAT_ROWAN)
	double sigma_w, beta_i, beta_max, Ti;

	Ti = calc_Ti(ph);

	#if(CONSTANTGAMMA)
	sigma_w = bsq / (ph[RHO] + GAMMA * ph[UU]);
	#elif(FIXEDGAMMA)
	double Te = calc_Te(ph);
	sigma_w = bsq / (ph[RHO] + GAMMAE / (GAMMAE - 1.0) * Te * ph[RHO] + GAMMA / (GAMMA - 1.0) * Ti * ph[RHO]);
	#else
	double Te = calc_Te(ph);
	double game, gami;
	game = (10.0 + 20.0 * Te * MU_E) / (6.0 + 15.0 * Te * MU_E);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	sigma_w = bsq / (ph[RHO] + game / (game - 1.0) * Te * ph[RHO] + gami / (gami - 1.0) * Ti * ph[RHO]);
	#endif

	beta_max = 1.0 / (4.0 * sigma_w);
	beta_i = MY_MIN((Ti * ph[RHO]) / (0.5 * bsq), beta_max);

	//Calculate delta
	delta = 0.5 * exp((beta_i / beta_max - 1.0) / (0.8 + sqrt(sigma_w)));
	#else
	//Set delta to constant value
	delta=0.5;
	#endif

	if (!isfinite(delta) || delta > 1.0 || delta < 0.0) delta = 0.5;

	return delta;
}

void heating(double* ph, struct of_state* q)
{

	return;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy and gas density
double calc_gamma_gas_conserved(double*  S, double rho) {
	double gamg, game, gami, Theta_e, Theta_i;
	
	#if(CONSTANTGAMMA)
	gamg = GAMMA;
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)) * (MU_E * MASS_RATIO)); //Actually theta_e=MU_E*Te meant
		Theta_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(S[0] * pow(rho, game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(S[1] * pow(rho, gami - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[0]) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[1]) - 1.0));
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif

	#if(!CONSTANTGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	#endif

	if (!isfinite(gamg) || gamg > 2.0 || gamg < 1.0) fprintf(stderr, "Gamma_error_conserved: %f %f %f %f \n", gamg, log10(S[0]), log10(S[1]), log10(rho));

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
double calc_gamma_gas_prim(double* pr) {
	double gamg, game, gami, Theta_e, Theta_i;

	#if(CONSTANTGAMMA)
	gamg = GAMMA;
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(pr[ENTRE] * pow(pr[RHO], game - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(pr[ENTRI] * pow(pr[RHO], gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(pr[ENTRE] * pow(pr[RHO], game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(pr[ENTRI] * pow(pr[RHO], gami - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * exp(pr[ENTRE]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * exp(pr[ENTRI]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO], 2. / 3.) * pr[ENTRE]) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO], 2. / 3.) * pr[ENTRI]) - 1.0));
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif

	#if(!CONSTANTGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	#endif

	if (!isfinite(gamg) || gamg > 1.00001*GAMMA || gamg < 0.99999*GAMMAE) {
		fprintf(stderr, "Gamma_error_prim: %f %f %f %f %f \n", gamg, log10(pr[ENTRE]), log10(pr[ENTRI]), log10(pr[RHO]), log10(pr[UU]));
		exit(0);
	}
	
	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy, gas density and w=W*(1-vsq)
double calc_gamma_gas_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(CONSTANTGAMMA)
	game = GAMMA;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(FIXEDGAMMA)     
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
		Te = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0)) / (MU_E * MASS_RATIO);
		Ti = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0)) / MU_I;
		#else
		Te = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[0]) - 1.0)) / (MU_E * MASS_RATIO);
		Ti = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[1]) - 1.0)) / MU_I;
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	#else
	gamg = GAMMA;
	#endif

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = quantg / gamg - ughat;

	//Update internal energy of electrons
	if (dis == -100.0) {
		quante = game * u_e;
		quanti = gami * u_i;
		double factor = quantg / (quante + quanti);
		quante *= factor;
		quanti *= factor;
	}
	else {
		u_e += delta * dis;
		#if(VARGAMMA)
		double C, Theta;
		C = u_e / rho * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
		game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		#endif
		quante = game * u_e; //quant=(gam)/(gam-1)*p
	}

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//Use analytical inversions
	double C = MU_E * MASS_RATIO / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C * quante * quante + 44.0 * C * quante + 20.0) + 5.0 * C * quante - 10.0) / (MU_E * MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C * quanti * quanti + 44.0 * C * quanti + 20.0) + 5.0 * C * quanti - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	#else
	gamg = GAMMA;
	#endif

	return gamg;
}

//Update electron and ion entropy based on found w in Newton Raphson solver
double set_S_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(CONSTANTGAMMA)
	game = GAMMA;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(FIXEDGAMMA)   
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
		Te = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0) / (MU_I));
		#else
		Te = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[0]) - 1.0) / (MU_E * MASS_RATIO));
		Ti = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * S[1]) - 1.0) / (MU_I));
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	#else
	gamg = GAMMA;
	#endif

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = quantg / gamg - ughat;

	//Update internal energy of electrons
	if (dis == -100.0) {
		quante = game * u_e;
		quanti = gami * u_i;
		double factor = quantg / (quante + quanti);
		quante *= factor;
		quanti *= factor;
	}
	else {
		u_e += delta * dis;
		#if(VARGAMMA)
		double C, Theta;
		C = u_e / rho * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
		game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		#endif
		quante = game * u_e; //quant=(gam)/(gam-1)*p
	}

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//Use analytical inversions
	double C = MU_E * MASS_RATIO / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C * quante * quante + 44.0 * C * quante + 20.0) + 5.0 * C * quante - 10.0) / (MU_E * MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C * quanti * quanti + 44.0 * C * quanti + 20.0) + 5.0 * C * quanti - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
        #if(FULL_ENTROPY_VARGAMMA)
		S[0] = log(pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
		S[1] = pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho;
		#endif
	#endif

	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	#else
	gamg = GAMMA;
	#endif

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
		res = MY_MIN(res, theta_max);
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
	double th_mean, th_sum, Theta_e, Theta_i, coeff, n_cgs, ne_cgs, T_e, T_i;
	double K2e, K2i, K0, K1;
	double theta_min = 1.e-2;
	double coulog;
	double res;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double mass_density_scale = mass_density_scale_cpu;
	double energy_density_scale = mass_density_scale_cpu * C_CGS * C_CGS;
	#endif

	#if(CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(p[ENTRE] * pow(p[RHO], GAMMA - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(p[ENTRI] * pow(p[RHO], GAMMA - 1.0)) * MU_I);
		#else
		Theta_e = fabs(p[ENTRE] * pow(p[RHO], GAMMA - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(p[ENTRI] * pow(p[RHO], GAMMA - 1.0) * MU_I);
		#endif
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(p[ENTRI] * pow(p[RHO], GAMMA - 1.0)) * MU_I);
		#else
		Theta_e = fabs(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(p[ENTRI] * pow(p[RHO], GAMMA - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO] * exp(p[ENTRE]), 2. / 3.)) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO] * exp(p[ENTRI]), 2. / 3.)) - 1.0));
		#else
		Theta_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO], 2. / 3.) * p[ENTRE]) - 1.0));
		Theta_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO], 2. / 3.) * p[ENTRI]) - 1.0));
		#endif
	#endif

	//note that average number density in Sadowski+17 (eq (20)) is assumed to be n_ave = ne_cgs.this can be updated 
	ne_cgs = p[RHO] * mass_density_scale / (MU_E * MH_CGS);    // calculation in cgs unit
	n_cgs = p[RHO] * mass_density_scale / (MH_CGS);    // calculation in cgs unit

	T_e = Theta_e / BOLTZ_CGS * (ME_CGS * C_CGS * C_CGS);
	T_i = Theta_i / BOLTZ_CGS * (MH_CGS * C_CGS * C_CGS);

	coulog = 35.4 + log(T_e / (10.e7) * sqrt(10.e-3 / ne_cgs));// Coulomb logarithm ( ln Lambda )
	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	coeff *= ne_cgs * n_cgs * (T_i - T_e);

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

	res = res / energy_density_scale * R_GOC_CGS;     // unit conversion from cgs to grid unit
	return (res);
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
