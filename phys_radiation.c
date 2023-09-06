#include "include.h"
#include "decs.h"

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
		double dK_dS_i=0.;
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
	eos_mode_rhou_temp(ph, &Te);
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
	eos_mode_rhou_temp(ph, &Te);
	//Tg *= (MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS);	
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
	#if(WHICHPROBLEM == RAD_PULSE)
	kappa_abs = 1e-30;// 0.;
	#endif
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
	eos_mode_rhou_temp(ph, &Te);
	//Te *= (MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS);
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
	#if(WHICHPROBLEM == RAD_PULSE)
	kappa_abs = 1e-30;// 0.;
	#endif
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
	eos_mode_rhou_temp(ph, &Te);
	//Te *= (MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	kappa_es = 0.2 * (1 + X_AB) / (1. + pow(Te / (4.5 * pow(10., 8.)), 0.86));
	kappa_es = 0.2 * (1 + X_AB);

	#if(WHICHPROBLEM == RAD_PULSE)
	kappa_es = KAPPARADPULSE;
	return(kappa_es);
	#else 
	if (!isfinite(kappa_es)) kappa_es = 0.0;
	return(kappa_es * (ph[RHO] * mass_density_scale) * R_G_CGS);
	#endif
}

/* find ucon, ucov, bcon, bcov from radiation primitive variables */
void get_state_rad(double* restrict pr, struct of_geom* restrict geom, struct of_state_rad* restrict q_rad)
{
	/* get radiation ucon */
	ucon_calc_rad(pr, geom, q_rad->ucon);
	lower(q_rad->ucon, geom, q_rad->ucov);

	return;
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