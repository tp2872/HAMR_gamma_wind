
//Calculate radiation temperature in rest frame of fluid
__device__ double calc_Tr(double* ph, double ucon[NDIM], double ucon_rad[NDIM], double ucov[NDIM]
	#if(P_NUM)
	, double *exp_xi
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	double Tr, u_dot_urad, u_dot_u, Ehat;
	#if(!CALC_MDOT)
		#if(P_NUM)
		double mass_density_scale = MASS_DENSITY_SCALE;
		#endif
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
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
	//Tr = pow(Ehat / ARAD, 0.25);
	exp_xi[0] = MY_MIN(1.64676 / (0.646756 + 0.121982 * CK_CGS * Ehat * Ehat * Ehat / (Nhat * Nhat * Nhat * Nhat)), 1.0);
	#else
	Tr = pow(Ehat / ARAD, 0.25);
	#endif

	return Tr;
}

__device__ double calc_Te(double* ph) {
	double Te;

	#if(TWO_T)
		#if(CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Te = exp((GAMMA - 1.0) * ph[ENTRE]) * pow(ph[RHO], GAMMA - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Te = exp((GAMMAE - 1.0) * ph[ENTRE]) * pow(ph[RHO], GAMMAE - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMAE - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY_VARGAMMA)
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * fabs(exp(ph[ENTRE])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
			#else
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * fabs(ph[ENTRE])) - 1.0) / (MU_E * MASS_RATIO);
			#endif
		#endif
	#else
	Te = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Te;
}

__device__ double calc_Ti(double* ph) {
	double Ti;

	#if(TWO_T)
		#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Ti = exp((GAMMA - 1.0) * ph[ENTRI]) * pow(ph[RHO], GAMMA - 1.0);
			#else
			Ti = ph[ENTRI] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY_VARGAMMA)
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * fabs(exp(ph[ENTRI])), 2. / 3.)) - 1.0) / MU_I;
			#else
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO], 2. / 3.) * fabs(ph[ENTRI])) - 1.0) / MU_I;
			#endif
		#endif
	#else
	Ti = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Ti;
}

__device__ void primtoflux_rad(double* pr, struct of_state_rad* q_rad, int dir, struct of_geom* geom, double* flux){
	#if(RAD_M1)
	int k;

	//Radiation energy tensor
	mhd_calc_rad(pr, dir, q_rad, &flux[UU_RAD]);
	for (k = UU_RAD; k <= U3_RAD; k++) flux[k] *= geom->g;

		//Flux of photon number
		#if(P_NUM)
		flux[PHOTON] = pr[PHOTON] * q_rad->ucon[dir];
		flux[PHOTON] *= geom->g;
		#endif
	#endif
	return;
}

__device__ void mhd_calc_rad(double * pr, int dir, struct of_state_rad * q_rad, double * mhd_rad){
	int j;
	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd_rad[j] = 4. / 3. * pr[UU_RAD] * q_rad->ucon[dir] * q_rad->ucov[j] + 1. / 3. * pr[UU_RAD] * delta(dir, j);
}

__device__ void source_rad(double *  ph, struct of_geom *  geom, struct of_state* q, struct of_state_rad* q_rad, double * dU
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
)
{
	#if(RAD_M1)
	double mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM],dK_dS, bsq;
	int k;
	#if(TWO_T)
	double src_coulomb, dK_dS_i;
	#endif
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double magnetic_density_scale = MASS_DENSITY_SCALE;
	#endif

	PLOOP dU[k] = 0.;

	//Add M1 radiation terms
	mhd_calc_rad(ph, 0, q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, q_rad, mhd_rad[3]);

	//Add radiation 4-force
	bsq = q->bcon[0] * q->bcov[0] + q->bcon[1] * q->bcov[1] + q->bcon[2] * q->bcov[2] + q->bcon[3] * q->bcov[3];

	calc_Gcon(ph, Gcon, q->ucon, q->ucov, q_rad->ucon, q_rad->ucov, mhd_rad, bsq
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, &(dU[PHOTON])
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);
	lower(Gcon, geom->gcov, Gcov);

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
			#if(0)
			double Theta, C;
			//For variable entropy
			C = ph[UU] / ph[RHO] * MU_G;
			Theta= (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY)
				dK_dS = (1.0 / Theta) * (MU_G);
				#else
				dK_dS = (ph[KTOT] / Theta) * (MU_G);
				#endif
			#else
				#if(FULL_ENTROPY)
				dK_dS = ph[RHO] / (GAMMA - 1.) * ph[UU]);
				#else
				dK_dS = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0);
				#endif
			#endif
		#else
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / (GAMMA - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0); 
			#endif
		#endif
		dU[KTOT] = -dK_dS * (Gcov[0] * q->ucon[0] + Gcov[1] * q->ucon[1] + Gcov[2] * q->ucon[2] + Gcov[3] * q->ucon[3]);
	#endif

	//Electron entropy source term for radiative cooling and coulomb coupling
	#if(TWO_T)
		#if(FIXEDGAMMA || CONSTANTGAMMA)
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / ((GAMMAE - 1.) * ph[UU]);
			dK_dS_i = ph[RHO] / ((GAMMA - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMAE - 1.) / pow(ph[RHO], GAMMAE - 1.0);
			dK_dS_i = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(VARGAMMA)
			double Theta_e, Theta_i;
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
		dU[ENTRE] = -dK_dS * (Gcov[0] * q->ucon[0] + Gcov[1] * q->ucon[1] + Gcov[2] * q->ucon[2] + Gcov[3] * q->ucon[3]);
		src_coulomb = source_Coulomb(ph
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		dU[ENTRE] += dK_dS * src_coulomb;
		dU[ENTRI] -= dK_dS_i * src_coulomb;
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}

//Calculate radiation 4-force
__device__ void calc_Gcon(double * ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double ucon_rad[NDIM], double ucov_rad[NDIM], double mhd_rad[NDIM][NDIM], double bsq
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double *source_photon
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	#if(RAD_M1)
	int i;
	double lambda, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], Tr, Te;
	#if(P_NUM || COMPTON)
	double exp_xi, kappa_abs_ph, kappa_emmit_ph;
	double Ehat, Nhat, u_dot_urad, u_dot_u;
	#endif
	#if(COMPTON)
	double G0, Theta_e, Theta_r;
	#endif
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double magnetic_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE*C_CGS*C_CGS;
	#else
	double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
	#endif

	//Calculate radiation temperature in rest frame of fluid
	Tr = calc_Tr(ph, ucon, ucon_rad, ucov
		#if(P_NUM)
		, &exp_xi
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);
	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	//Calculate opacities
	calc_kappa_new(ph, bsq, Tr, Te, &kappa_abs, &kappa_emmit, &kappa_es
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(P_NUM)
		, &kappa_abs_ph
		, &kappa_emmit_ph
		, exp_xi
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);

	//Calculate emmission rate
	lambda = kappa_emmit * (ARAD / energy_density_scale) * Te * Te * Te * Te; //in units of erg/(Rg/c)/cm^3

	//Calculate non-Compton scattering source term
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs * R_dot_ucon[i] + lambda * ucon[i]) - kappa_es * (R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]) * ucon[i]);
	}

		//Evaluate comptonization term
		//Misc variables-->Merge with calc_Tr
		#if(P_NUM || COMPTON)
		u_dot_urad = ucov[0] * ucon_rad[0] + ucov[1] * ucon_rad[1] + ucov[2] * ucon_rad[2] + ucov[3] * ucon_rad[3];
		u_dot_u = ucon[0] * ucov[0] + ucon[1] * ucov[1] + ucon[2] * ucov[2] + ucon[3] * ucov[3];
		Ehat = ((4. / 3.) * ph[UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * ph[UU_RAD] * u_dot_u);
		#endif

		#if(P_NUM)
		Nhat = -ph[PHOTON] * u_dot_urad;
		
		//Source term for photons
		source_photon[0] = -kappa_abs_ph * Nhat + (kappa_emmit_ph / mass_density_scale * ARAD * Te * Te * Te * Te / (BOLTZ_CGS * Te * 2.701178));
		#endif

		//Compton scattering term is added
		#if(COMPTON)
		Theta_e = Te * 1.6863687454173171e-10;
		Theta_r = Tr * 1.6863687454173171e-10;
		G0 = -kappa_es * Ehat * 4.0 * (Theta_e - Theta_r) * (1.0 + 3.683 * Theta_e + 4.0 * Theta_e * Theta_e) / ((1.0 + Theta_e));
		for (i = 0; i < NDIM; i++) Gcon[i] += ucon[i] * G0;
		#endif
	#endif
}

//Calculate radiative wave velocity
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int dir, double* vmax, double* vmin, double dx
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	#if(RAD_M1)
	double discr, vp, vm, tau, kappa_abs, kappa_es, kappa_tot, crad2, cmin_rad, cmax_rad, cmin_mhd, cmax_mhd, bsq, Tr, Te;
	double Acon_0, Acon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	#if(P_NUM)
	double exp_xi, kappa_abs_ph;
	#endif
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double magnetic_density_scale = MASS_DENSITY_SCALE;
	#endif

	if (dir == 1) {
		Acon_0 = geom->gcon[1];
		Acon_js = geom->gcon[4];
	}
	else if (dir == 2) {
		Acon_0 = geom->gcon[2];
		Acon_js = geom->gcon[7];
	}
	else if (dir == 3) {
		Acon_0 = geom->gcon[3];
		Acon_js = geom->gcon[9];
	}

	/* find radiation wave speed at 1./3. speed of light (==isotrpic in radiation frame) */
	crad2 = 1.0 / 3.0;

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q_rad->ucon[dir];
	Bu = q_rad->ucon[0];
	AB = Acon_0;
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10)discr = 0.;

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

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
		,  &exp_xi
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);

	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, pr[RHO], pr[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(pr) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(pr) * MU_G * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	//Calculate opacities
	calc_kappa_new(pr, bsq, Tr, Te, &kappa_abs, NULL, &kappa_es
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, Tr //Fake value for r; We do not need to know kappa_emmit
		#endif
		#if(P_NUM)
		, &kappa_abs_ph
		, NULL
		, exp_xi
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);
	#if(P_NUM)
	kappa_tot = MY_MIN(kappa_abs, kappa_abs_ph) + kappa_es;
	#else
	kappa_tot = kappa_abs + kappa_es;
	#endif
	tau = kappa_tot * sqrt(geom->gcov[(dir == 1) * 4 + (dir == 2) * 7 + (dir == 3) * 9]) * dx;
	crad2 = 16. / (9. * tau * tau);
	//crad2 = 1. / 3.;
	/* check on it! */
	if (crad2 < 0.) crad2 = SMALL;
	if (crad2 > 1.) crad2 = 1.;

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Au = q_rad->ucon[dir];
	Bu = q_rad->ucon[0];
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.; 

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

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.;

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);
	double cmax_mhd2, cmin_mhd2;
	if (vp > vm) {
		cmax_mhd2 = vp;
		cmin_mhd2 = vm;
	}
	else {
		cmax_mhd2 = vm;
		cmin_mhd2 = vp;
	}

	/*Set velocity as minimum of optically thin and optically thick limit*/
	*vmax = MY_MIN(MY_MAX(cmax_mhd, cmax_mhd2), cmax_rad);
	*vmin = MY_MAX(MY_MIN(cmin_mhd, cmin_mhd2), cmin_rad);

	return;
	#endif
}

//Calculate total absorption opacity
__device__ void calc_kappa_new(double* ph, double bsq, double Tr, double Te, double *kappa_abs, double *kappa_emmit, double *kappa_es
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double *kappa_abs_ph 
	, double *kappa_emmit_ph
	, double exp_xi
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	double kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff_abs, kappa_ff_emmit, kappa_HOPAL, kappa_COPAL, kappa_fe, kappa_ff_unity, kappa_sy_abs, kappa_sy_emmit, kappa_dc,  ne, p_theta, scaling_factor;
	double Ree, Rei, Theta_e, Theta_gamma, zeta, nu_mu, phi;
	#if(P_NUM)
	double one_exp_xi, a, b, c, d, e;
	one_exp_xi = 1.0 - exp_xi;
	#endif
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE, magnetic_density_scale = MASS_DENSITY_SCALE;
	double magnetic_density_scale_2 = sqrt(MASS_DENSITY_SCALE) * C_CGS;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;	
	#else
	double magnetic_density_scale_2 = sqrt(mass_density_scale) * C_CGS;
	double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
	#endif

	ne = ph[RHO] * mass_density_scale / (MU_E * MH_CGS);
	Theta_e = Te * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
	nu_mu = 1.5 * E_CGS * sqrt(bsq * 4. * M_PI + 0.00000001 * ph[RHO]) * magnetic_density_scale_2 * Theta_e * Theta_e / (2.0 * M_PI * ME_CGS * C_CGS);

	#if(OP_EXTRA)
	Theta_gamma = Tr * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
	zeta = Tr / Te;

	//Calc free-free absorption opacity
	if (kappa_abs != NULL || kappa_emmit != NULL) {
		if (Theta_e <= 1.0) {
			Rei = 1. + 1.76 * pow(Theta_e, 1.34);
			Ree = 1.7 * Theta_e * (1.0 + 1.1 * Theta_e + Theta_e * Theta_e - 1.06 * pow(Theta_e, 2.5));
		}
		else {
			Rei = 1.4 * sqrt(Theta_e) * (log(1.12 * Theta_e + 0.48) + 1.5);
			Ree = 1.7 * sqrt(Theta_e) * (1.46 * (1.28 + log(1.12 * Theta_e)));
		}
		#if(P_NUM)
		a = 0.188 * pow(exp_xi, 13.9) - 0.2 * pow(one_exp_xi, 0.565) + 0.356;
		b = 0.0722 * pow(exp_xi, 1.36) + 0.255 * pow(one_exp_xi, 0.313) + 3.06;
		c = -1.41 * pow(exp_xi, 3.08) - 1.44 * pow(one_exp_xi, 0.128) + 5.99;
		kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * a * pow(zeta, -b) * log(1.0 + c * zeta);
		kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * log(1.0 + 4.52);
		#else
		kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1.0 - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * pow(zeta, -3.14) * log(1.0 + 4.52 * zeta);
		kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1.0 - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * log(1.0 + 4.52);
		#endif
		scaling_factor = kappa_ff_abs / kappa_ff_emmit;
	}

	//Calculate synchrotron opacities
	#if(P_NUM)
		#if(AGN)
		//AGN
		if (kappa_abs != NULL){
			a = -0.0295 * pow(exp_xi, 2.29) - 0.143 * pow(one_exp_xi, 0.251) + 0.236;
			b = 0.00977 * pow(exp_xi, 730.0) + 0.0291 * pow(one_exp_xi, 0.48) + 2.58;
			c = 1.29 * pow(exp_xi, 1.59) + 3.46 * pow(one_exp_xi, 0.234) + 2.15;
			d = -78.1 * pow(exp_xi, 66.0) - 40.3 * pow(one_exp_xi, 0.899) + 87.4;
			e = 0.415 * pow(exp_xi, 0.399) + 1.04 * pow(one_exp_xi, 0.252) + 2.68;

			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}

		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		#else
		//XRB
		if (kappa_abs != NULL){
			a = -2.31e-8 * pow(exp_xi, 34.) - 8.24e-9 * pow(one_exp_xi, 2.42) + 1.27;
			b = -0.0261 * pow(exp_xi, 738.0) - 0.00475 * pow(one_exp_xi, 1.55) + 1.06;
			c = 0.000179 * pow(exp_xi, 432.0) + 0.0000411 * pow(one_exp_xi, 0.372) + 0.000584;
			d = -17.7 * pow(exp_xi, 49.4) - 3.33 * pow(one_exp_xi, 2.76) + 18.3;
			e = 0.427 * pow(exp_xi, 0.654) + 1.23 * pow(one_exp_xi, 0.214) + 2.49;

			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}
		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		#endif
	#else
		#if(AGN)
		//AGN
		//a = 0.206;
		//b = 2.59;
		//c = 3.44;
		//d = 9.33;
		//e = 3.09;

		if (kappa_abs != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		#else
		//XRB
		//a = 1.27;
		//b = 1.03;
		//c = 0.000763;
		//d = 0.616;
		//e = 2.91;
		if (kappa_abs != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		#endif
	#endif
	
	//Calculate double compton opacity
	/*
	#if(1)
		#if(P_NUM)
		//Absorption opacity
		a = 6.7 * pow(exp_xi, 0.942) + 4.16 * pow(one_exp_xi, 1.69) + 3.1e-8;
		b = -0.0021 * pow(exp_xi, 0.0217) - 0.0334 * pow(one_exp_xi, 0.469) + 0.042;
		c = -0.18 * pow(exp_xi, 33.0) + 0.201 * pow(one_exp_xi, 0.258) + 3.8;
		d = 0.0169 * pow(exp_xi, 35.4) - 0.0626 * pow(one_exp_xi, 0.35) + 0.118;
		p_theta = pow(1.0 + Theta_e, -3.0);
		kappa_dc_abs = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * mass_density_scale);
		kappa_dc_abs *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
	
		//Emmission opacity
		a = 0.488 * pow(exp_xi, 1.75) - 0.0589 * pow(one_exp_xi, 10.7) + 6.34;
		b = 0.0282 * pow(exp_xi, 1.56) + 0.0142 * pow(one_exp_xi, 0.361) + 0.00875;
		c = -0.16 * pow(exp_xi, 15.4) + 0.184 * pow(one_exp_xi, 0.366) + 3.78;
		d = 0.015 * pow(exp_xi, 26.3) - 0.0256 * pow(one_exp_xi, 0.398) + 0.119;
		p_theta = pow(1.0 + Theta_gamma, -3.0);
		kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * mass_density_scale);
		kappa_dc_emmit *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		#else
		//Absorption opacity
		p_theta = pow(1.0 + Theta_e, -3.0);
		kappa_dc_abs = 7.36e-46 * ne * Tr * Tr * 1.0 * p_theta / (ph[RHO] * mass_density_scale);
		kappa_dc_abs *= 1.0 / ((1.0 / 6.83 + 1.0 / (0.0374 * pow(Theta_gamma, -3.63))) + 1.0 / (0.134 * pow(Theta_gamma, -3.63 / 3.0)));

		//Emmission opacity
		p_theta = pow(1.0 + Theta_gamma, -3.0);
		kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * 1.0 * p_theta / (ph[RHO] * mass_density_scale);
		kappa_dc_emmit *= 1.0 / ((1.0 / 6.83 + 1.0 / (0.0374 * pow(Theta_gamma, -3.63))) + 1.0 / (0.134 * pow(Theta_gamma, -3.63 / 3.0)));
		#endif
	#endif
	*/

	//Calculate molecular opacity
	kappa_m = 3.0 * Z_AB; //No scaling factor

	//Calculate H- opacity
	kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * mass_density_scale) * pow(Te, 7.7);

	//Calculate Chianti opacity
	kappa_chianti = 3.0e34 * ph[RHO] * mass_density_scale * (0.1 + Z_AB / 0.02) * X_AB * (1 + X_AB) * pow(Te, -4.7);

	//Calculate iron opacity
	kappa_fe = 0.3 * (Z_AB / 0.02) * exp(-6.0 * pow(-12.0 + log(Te), 2.0)); //No scaling factor

	//Calculate bound-free opacity
	kappa_bf = 1.2e24 * 750.0 * Z_AB * (1.0 + X_AB + 0.75 * Y_AB) * ph[RHO] * mass_density_scale * pow(Te, -3.5) * log(1. + 1.6 );

	//Calculate COPAL terms conform Mckinney+2017
	kappa_COPAL = 3.0e-13 * kappa_chianti * pow(Te, 1.6) * pow(ph[RHO] * mass_density_scale, -0.4);

	//Calculate HOPAL terms conform Mckinney+2017
	kappa_HOPAL = 1.0e4 * pow(Te, -1.2) * kappa_h;

	//Calculate total absorption opacity
	if (kappa_abs != NULL) {
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_HOPAL * scaling_factor) + 1.0 / (kappa_COPAL * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_h * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;

		if (!isfinite(kappa_abs[0])) kappa_abs[0] = 0.0;
		else kappa_abs[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;
	}
	if (kappa_emmit != NULL) {
		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_HOPAL) + 1.0 / kappa_COPAL + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;

		if (!isfinite(kappa_emmit[0])) kappa_emmit[0] = 0.0;
		else kappa_emmit[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;
	}

		//Calculate number absorption and emmission opacities
		#if(P_NUM)
		//Calc free-free opacity
		if (kappa_abs_ph != NULL || kappa_emmit_ph != NULL) {
			a = 21.0 * pow(exp_xi, 5.0) - 2.06 * one_exp_xi + 4.0;
			b = -0.412 * pow(exp_xi, 59.1) + 0.000894 * pow(one_exp_xi, 10.2) + 3.15;
			c = 5.27 * pow(exp_xi, 69.2) + 2.39 * pow(one_exp_xi, 0.552);
			kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * a * pow(zeta, -b) * log(1 + c * zeta);
			kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * mass_density_scale) * pow(Te, -3.5) * (Rei + Ree) * 25.0 * log(1.0 + 5.27); //Watch out with coefficients
			scaling_factor = kappa_ff_abs / kappa_ff_emmit;
		}

		//Calculate synchrotron opacities
		if (kappa_abs_ph != NULL) {
			#if(AGN)
			//AGN
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			a = 10.8 * pow(exp_xi, 172.0) - 20.4 * pow(one_exp_xi, 0.699) + 29.2;
			b = -0.18 * pow(exp_xi, 31.9) + 0.425 * pow(one_exp_xi, 0.179) + 2.76;
			c = 0.0207 * pow(exp_xi, 9.69) + 0.0506 * pow(one_exp_xi, 0.804) + 0.0314;
			d = 1.51e6 * pow(exp_xi, 2830.0) - 1.4e5 * pow(one_exp_xi, 3.06e-12) + 1.4e5;
			e = 0.1 * pow(exp_xi, 1.95) + 1.57 * pow(one_exp_xi, 0.124);
			#else
			//XRB
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			a = -0.000359 * pow(exp_xi, 1.31) - 0.000552 * pow(one_exp_xi, 0.135) + 0.00209;
			b = 0.035 * pow(exp_xi, 5.43) + 0.0433 * pow(one_exp_xi, 0.159) + 0.948;
			c = -0.122 * pow(exp_xi, 37.1) - 0.0685 * pow(one_exp_xi, 2.8) + 1.04;
			d = -8.59 * pow(exp_xi, 155.0) - 6.47 * pow(one_exp_xi, 0.436) + 8.71;
			e = -0.447 * pow(exp_xi, 394.0) + 0.506 * pow(one_exp_xi, 0.155) + 2.45;
			#endif
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}
		if (kappa_emmit_ph != NULL) {
			#if(AGN)
			//AGN
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (40.0 * pow(phi, -2.58) * log(1.0 + 0.0522 * phi)) + 1.0 / (1.65e6 * pow(phi, -0.1)));
			#else
			//XRB
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.00173 * pow(phi, -0.983) * log(1.0 + 0.921 * phi)) + 1.0 / (0.123 * pow(phi, -2.0)));
			#endif
		}

		//Calculate double compton number absorption opacity
		/*
		#if(1)
		if (kappa_abs_ph != NULL) {
			p_theta = pow(1.0 + Theta_e, -3.0);
			a = 29.4 * pow(exp_xi, 285.0) - 76.4 * pow(one_exp_xi, 0.136) + 87.5;
			b = 0.196 * pow(exp_xi, 18.1) - 1.12 * pow(one_exp_xi, 0.134) + 1.16;
			c = -0.8 * pow(exp_xi, 21.6) + 0.0427 * pow(one_exp_xi, 182.0) + 3.93;
			d = 1.87 * pow(exp_xi, 309.0) - 2.72 * pow(one_exp_xi, 0.106) + 2.86;
			kappa_dc = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * mass_density_scale);
			kappa_dc *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		}

		//Calculate double compton number emmission opacity
		if (kappa_emmit_ph != NULL) {
			p_theta = pow(1.0 + Theta_gamma, -3.0);
			a = -81.7 * pow(exp_xi, 1.01) - 94.8 * pow(one_exp_xi, 0.925) + 198.0;
			b = 1.31 * pow(exp_xi, 1.12) + 1.05 * pow(one_exp_xi, 0.249) + 4.8e-11;
			c = -0.418 * pow(exp_xi, 14.8) + 0.442 * pow(one_exp_xi, 0.361) + 3.44;
			d = 1.37 * pow(exp_xi, 31.3) - 1.38 * pow(one_exp_xi, 0.316) + 3.38;
			kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * mass_density_scale);
			kappa_dc_emmit *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		}
		#endif
		*/

		//Calculate total number absorption opacity
		if (kappa_abs_ph != NULL) {
			kappa_abs_ph[0] = 1. / (1. / (kappa_m + kappa_HOPAL * scaling_factor) + 1.0 / (kappa_COPAL * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
			kappa_abs_ph[0] = 1. / (1. / (kappa_m + kappa_h * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
			if (!isfinite(kappa_abs_ph[0])) kappa_abs_ph[0] = 0.0;
			else kappa_abs_ph[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;
		}
		if (kappa_emmit_ph != NULL) {
			kappa_emmit_ph[0] = 1. / (1. / (kappa_m + kappa_HOPAL) + 1.0 / kappa_COPAL + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
			kappa_emmit_ph[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;

			if (!isfinite(kappa_emmit_ph[0])) kappa_emmit_ph[0] = 0.0;
			else kappa_emmit_ph[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;
		}
		#endif
	#else
	kappa_m = 30.0 * 0.1 * Z_AB;
	if (kappa_abs != NULL) {
		//zeta = 4. * M_PI * ME_CGS * ME_CGS * ME_CGS * pow(C_CGS, 5.0) * Tr / (3.0 * E_CGS * BOLTZ_CGS * PLANCK_CGS * sqrt(bsq * 4. * M_PI + 0.00000001*ph[RHO]) * magnetic_density_scale_2 * Te * Te);	
		//zeta = MY_MIN(zeta, 1.e5);
		kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * mass_density_scale) * pow(Te, 7.7);
		kappa_chianti = 30.0e33 * ph[RHO] * mass_density_scale * (0.1 + Z_AB / 0.02) * X_AB * (1.0 + X_AB) * pow(Te, -1.7) * pow(Tr, -3.);
		kappa_bf = 30.0 * 3.0e25 * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * mass_density_scale * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te));
		kappa_ff_abs = 30.0 * 4.0e22 * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * mass_density_scale * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te)) * (1. + 4.4e-10 * Te);
		//kappa_sy_abs = 1.59e-30 * ne * 4. * M_PI * bsq * energy_density_scale * Te * pow(Tr, -3.) / (ph[RHO] * mass_density_scale) / (1. + 5.444 * pow(zeta, -0.666666) + 7.218 * pow(zeta, -1.3333333));;
		#if(AGN)
		phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
		kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
		kappa_sy_abs *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		#else
		phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
		kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
		kappa_sy_abs *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		#endif
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_abs)) + kappa_sy_abs;

		if (!isfinite(kappa_abs[0])) kappa_abs[0] = 0.0;
		else kappa_abs[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;

		#if(P_NUM)
		if (kappa_abs_ph != NULL) {
			kappa_abs_ph[0] = kappa_abs[0] - kappa_sy_abs * ((ph[RHO] * mass_density_scale) * R_G_CGS);
			//kappa_sy_abs = 1.59e-30 * ne * 4. * M_PI * bsq * energy_density_scale * Te * pow(Tr, -3.) / (ph[RHO] * mass_density_scale) * 0.868 * zeta/ (1.0 + 0.589 * pow(zeta, -1.0 / 3.0) + 0.087 * pow(zeta, -2.0 / 3.0));	
			#if(AGN)
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (40.0 * pow(phi, -2.58) * log(1.0 + 0.0522 * phi)) + 1.0 / (1.65e6 * pow(phi, -0.1)));
			#else
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * mass_density_scale);
			kappa_sy_abs *= 1.0 / (1.0 / (0.00173 * pow(phi, -0.983) * log(1.0 + 0.921 * phi)) + 1.0 / (0.123 * pow(phi, -2.0)));;
			#endif
					
			kappa_abs_ph[0] += (kappa_sy_abs * ((ph[RHO] * mass_density_scale) * R_G_CGS));
			if (!isfinite(kappa_abs_ph[0])) kappa_abs_ph[0] = 0.0;
		}
		#endif
	}
	if (kappa_emmit != NULL) {
		//zeta = 4. * M_PI * ME_CGS * ME_CGS * ME_CGS * pow(C_CGS, 5.0) * Te / (3.0 * E_CGS * BOLTZ_CGS * PLANCK_CGS * sqrt(bsq * 4. * M_PI + 0.00000001 * ph[RHO]) * magnetic_density_scale_2 * Te * Te);
		//zeta = MY_MIN(zeta, 1.e5);
		kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * mass_density_scale) * pow(Te, 7.7);
		kappa_chianti = 30.0e33 * ph[RHO] * mass_density_scale * (0.1 + Z_AB / 0.02) * X_AB * (1.0 + X_AB) * pow(Te, -4.7);
		kappa_bf = 30.0 * 3.0e25 * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * mass_density_scale * pow(Te, -3.5) * log(1. + 1.6);
		kappa_ff_emmit = 30.0 * 4.0e22 * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * mass_density_scale * pow(Te, -3.5) * log(1. + 1.6) * (1. + 4.4e-10 * Te);
		//kappa_sy_emmit = 1.59e-30 * ne * 4. * M_PI * bsq * energy_density_scale * pow(Te, -2.) / (ph[RHO] * mass_density_scale); /// (1. + 5.444 * pow(zeta, -0.666666) + 7.218 * pow(zeta, -1.3333333));
		#if(AGN)
		phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
		kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
		kappa_sy_emmit *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		#else
		phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
		kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
		kappa_sy_emmit *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		#endif

		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
		if (!isfinite(kappa_emmit[0])) kappa_emmit[0] = 0.0;
		else kappa_emmit[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;

		#if(P_NUM)
		if (kappa_emmit_ph != NULL) {
			kappa_emmit_ph[0] = kappa_emmit[0] - kappa_sy_emmit * ((ph[RHO] * mass_density_scale) * R_G_CGS);
			//kappa_sy_emmit = 1.59e-30 * ne * 4. * M_PI * bsq * energy_density_scale * pow(Te, -2.) / (ph[RHO] * mass_density_scale); //* 0.868 * zeta / (1.0 + 0.589 * pow(zeta, -1.0 / 3.0) + 0.087 * pow(zeta, -2.0 / 3.0));
			//double test = ARAD * Te * Te * Te / (BOLTZ_CGS * 2.701178);
			//kappa_sy_emmit = 1.44e5 * ne * sqrt(4. * M_PI * bsq * energy_density_scale) / test / (ph[RHO] * mass_density_scale)/C_CGS; //* 0.868 * zeta / (1.0 + 0.589 * pow(zeta, -1.0 / 3.0) + 0.087 * pow(zeta, -2.0 / 3.0));
			#if(AGN)
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (40.0 * pow(phi, -2.58) * log(1.0 + 0.0522 * phi)) + 1.0 / (1.65e6 * pow(phi, -0.1)));
			#else
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * mass_density_scale);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.00173 * pow(phi, -0.983) * log(1.0 + 0.921 * phi)) + 1.0 / (0.123 * pow(phi, -2.0)));
			#endif

			kappa_emmit_ph[0] += (kappa_sy_emmit * ((ph[RHO] * mass_density_scale) * R_G_CGS));
			if (!isfinite(kappa_emmit_ph[0])) kappa_emmit_ph[0] = 0.0;
		}
		#endif	
	}
	#endif

	#if(COOL_STOP)
		#if(TWO_T)
		double epsilon = ((gamma_g - 1.) * ph[UU] + 0.3333 * ph[UU_RAD]) / ph[RHO];
		#else
		double epsilon = ((GAMMA - 1.) * ph[UU] + 0.3333 * ph[UU_RAD]) / ph[RHO];
		#endif	
	double om_kepler = 1. / (pow(r, 3. / 2.) + BH_SPIN);
	double T_target = M_PI / 2. * pow(STOP_SCALEHEIGHT * r * om_kepler, 2.);
		#if(TWO_T)
		double Y = (gamma_g - 1.) * epsilon / T_target; 
		#else
		double Y = (GAMMA - 1.) * epsilon / T_target;
		#endif
	if (Y < 1.0) {
		if (kappa_emmit != NULL) kappa_emmit[0] *= pow(Y, 4.0);
		#if(P_NUM)
		if (kappa_emmit_ph != NULL) kappa_emmit_ph[0] *= pow(Y, 4.0);
		#endif
	}
	#endif

	#if(WHICHPROBLEM == RAD_PULSE)
	kappa_abs[0] = 0.0;
	kappa_emmit[0] = 0.0;
	#endif

	//Calculate electron scattering opacity
	if (kappa_es != NULL) {
		kappa_es[0] = 0.2 * (1 + X_AB) / (1. + pow(fabs(Te) / (4.5 * pow(10., 8.)), 0.86));
		//kappa_es[0] = 0.2 * (1 + X_AB);
		if (!isfinite(kappa_es[0])) kappa_es[0] = 0.0;
		else kappa_es[0] *= (ph[RHO] * mass_density_scale) * R_G_CGS;

		#if(WHICHPROBLEM == RAD_PULSE)
		kappa_es[0] = KAPPARADPULSE;
		#endif
	}
}

/* find ucon, ucov, bcon, bcov from radiation primitive variables */
__device__ void get_state_rad(double* pr, struct of_geom* geom, struct of_state_rad* q_rad)
{
	/* get radiation ucon */
	ucon_calc_rad(pr, geom, q_rad->ucon);
	lower(q_rad->ucon, geom->gcov, q_rad->ucov);

	return;
}

/* find contravariant radiation four-velocity */
__device__ void ucon_calc_rad(double * pr, struct of_geom * geom, double *ucon_rad)
{
	double alpha, gamma;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0]);
	#pragma unroll 4
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;

	gamma_calc_rad(pr, geom, &gamma);

	ucon_rad[0] = gamma / alpha;
	#if AMD
	#pragma unroll 4
	SLOOPA ucon_rad[j] = fma(-gamma, beta[j] / alpha, pr[U1_RAD + j - 1]);
	#else
	#pragma unroll 4
	SLOOPA ucon_rad[j] = pr[U1_RAD + j - 1] - gamma*beta[j] / alpha;
	#endif

	return;
}

__device__ int gamma_calc_rad(double *  pr, struct of_geom *  geom, double *  gamma_rad)
{
	double qsq_rad;
	#if AMD
	qsq_rad = fma(geom->gcov[4], pr[U1_RAD] * pr[U1_RAD], fma(geom->gcov[7], pr[U2_RAD] * pr[U2_RAD],geom->gcov[9] * pr[U3_RAD] * pr[U3_RAD]))
		+ 2.*fma(geom->gcov[5], pr[U1_RAD] * pr[U2_RAD], fma(geom->gcov[6], pr[U1_RAD] * pr[U3_RAD],geom->gcov[8] * pr[U2_RAD] * pr[U3_RAD]));
	#else
	qsq_rad = geom->gcov[4] * pr[U1_RAD] * pr[U1_RAD]+ geom->gcov[7] * pr[U2_RAD] * pr[U2_RAD]+ geom->gcov[9] * pr[U3_RAD] * pr[U3_RAD]
		+ 2.*(geom->gcov[5] * pr[U1_RAD] * pr[U2_RAD]+ geom->gcov[6] * pr[U1_RAD] * pr[U3_RAD]+ geom->gcov[8] * pr[U2_RAD] * pr[U3_RAD]);
	#endif

	if (qsq_rad < 0.) {
		if (fabs(qsq_rad) > 1.E-10) { // then assume not just machine precision
			*gamma_rad = 1.;
			return (1);
		}
		else qsq_rad = 1.E-10; // set floor
	}

	*gamma_rad = sqrt(1. + qsq_rad);

	return(0);
}
