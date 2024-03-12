#include "include.h"
#include "decs.h"


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