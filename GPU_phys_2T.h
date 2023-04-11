
#if(TWO_T)
//Calculate fraction of heat that goes into electrons on ions based on temperature ratio at previous timestep: 
__device__ double calc_delta(double* ph, double bsq) {
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

__device__ void heating(double* ph, struct of_state* q)
{
	
}

__device__ double source_Coulomb(double* p
	#if(CALC_MDOT)
	, double mass_density_scale
	#endif
) {
	double th_mean, th_sum, Theta_e, Theta_i, coeff, n_cgs, ne_cgs, T_e, T_i;
	double K2e, K2i, K0, K1;
	double theta_min = 1.e-2;
	double coulog; 
	double res;
	#if(!CALC_MDOT)
	double mass_density_scale = MASS_DENSITY_SCALE;
	double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
	#else
	double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
	#endif

	#if(CONSTANTGAMMA)
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
		#if(FULL_ENTROPY_VARGAMMA)
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(p[RHO] * exp(p[ENTRE])), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(p[RHO] * exp(p[ENTRI])), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO], 2. / 3.) * fabs(p[ENTRE])) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO], 2. / 3.) * fabs(p[ENTRI])) - 1.0);
		#endif
	#endif

	//note that average number density in Sadowski+17 (eq (20)) is assumed to be n_ave = ne_cgs.this can be updated 
	ne_cgs = p[RHO] * mass_density_scale / (MU_E * MH_CGS);    // calculation in cgs unit
	n_cgs = p[RHO] * mass_density_scale / (MH_CGS);    // calculation in cgs unit

	T_e = Theta_e / BOLTZ_CGS * (ME_CGS * C_CGS * C_CGS);
	T_i = Theta_i / BOLTZ_CGS * (MH_CGS * C_CGS * C_CGS);

	coulog = 35.4 + log(T_e / (1.0e7) * sqrt(1.0e-3 / ne_cgs));// Coulomb logarithm ( ln Lambda )
	#if(HIGH_MDOT)
	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	#else
	coeff = 1.5 * ME_CGS / MH_CGS *(X_AB+Y_AB*0.25)* coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	#endif
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

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy and gas density
__device__ double calc_gamma_gas_conserved(double* S, double rho) {
	double gamg;
	#if(!CONSTANTGAMMA)
	double game, gami, Theta_e, Theta_i;
	#endif

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
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(exp(S[0])), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(exp(S[1])), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[0])) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[1])) - 1.0);
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif
	
	#if(VARGAMMA || FIXEDGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
	#endif

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
__device__ double calc_gamma_gas_prim(double* pr) {
	double gamg;
	#if(!CONSTANTGAMMA)
	double game, gami, Theta_e, Theta_i;
	#endif

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
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * fabs(exp(pr[ENTRE])), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * fabs(exp(pr[ENTRI])), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO], 2. / 3.) * fabs(pr[ENTRE])) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO], 2. / 3.) * fabs(pr[ENTRI])) - 1.0);
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif

	#if(VARGAMMA || FIXEDGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
	#endif

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy, gas density and w=W*(1-vsq)
__device__ double calc_gamma_gas_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;
	#if(VARGAMMA)
	double ratio, C, Theta;
	#endif

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
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[1])), 2. / 3.)) - 1.0) / MU_I;
		#else
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[0])) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[1])) - 1.0) / (MU_I);
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
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
	u_e += delta * dis;
	#if(VARGAMMA)
	C = u_e / rho * MU_E * MASS_RATIO;
	Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
	game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
	#endif
	quante = game * u_e; //quant=(gam)/(gam-1)*p

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	if(u_e <= 0.0) quante = FLOOR_ENTROPY * quantg;
	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//Use analytic expressions
	C = MU_E * MASS_RATIO * quante / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / (MU_E * MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I * quanti / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gas eos gammma
	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
	#else
	gamg = GAMMA;
	#endif

	return gamg;
}

//Update electron and ion entropy based on found w in Newton Raphson solver
__device__ double set_S_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;
	#if(VARGAMMA)
	double C, Theta;
	#endif

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
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(exp(S[1])), 2. / 3.)) - 1.0) / (MU_I);
		#else
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[0])) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[1])) - 1.0) / (MU_I);
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
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
	u_e += delta * dis;
	#if(VARGAMMA)
	C = u_e / rho * MU_E * MASS_RATIO;
	Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
	game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
	#endif
	quante = game * u_e; //quant=(gam)/(gam-1)*p

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	if(u_e <= 0.0) quante = FLOOR_ENTROPY * quantg;

	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;

	//Set entropy
		#if(FULL_ENTROPY)
		S[0] = 1.0 / (game - 1.0) * log(pe * pow(rho, -game));
		S[1] = 1.0 / (gami - 1.0) * log(pi * pow(rho, -gami));
		#else
		S[0] = pe * pow(rho, -game);
		S[1] = pi * pow(rho, -gami);
		#endif
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//quant*C=(10.0 + 20.0 * x * C) / (6.0 + 15.0 * x * C)/((10.0 + 20.0 * x * C) / (6.0 + 15.0 * x * C)-1)*x*C
	//B=(10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)/((10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)-1)*Theta --> Solved this analytically in Wolfram
	//C= (MU_E * MASS_RATIO) / RHO
	//B=quant*C
	//x=pe
	//Theta=x*C

	//Use analytic expressions
	C = MU_E * MASS_RATIO * quante / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / (MU_E * MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I * quanti / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);

	//Set entropy
		#if(FULL_ENTROPY_VARGAMMA)
		S[0] = log(pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = (Te * (MU_E * MASS_RATIO)) * (Te * (MU_E * MASS_RATIO) + 0.4) / pow(rho, 2. / 3.);
		S[1] = (Ti * MU_I) * (Ti * MU_I + 0.4) / pow(rho, 2. / 3.);
		#endif
	#endif

	#if(FIXEDGAMMA || VARGAMMA)
	gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	if (!isfinite(gamg))gamg = GAMMA;
	#else
	gamg = GAMMA;
	#endif

	return gamg;
}

// Some bessel functions
__device__ double bessi0(double x) {
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

__device__ double bessi1(double x) {
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

__device__ double bessk0(double x) {
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

__device__ double bessk1(double x) {
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

__device__ double bessk(int n, double x) {
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