#include "include.h"
#include "decs_MPI.h"

#define epsem (2.22E-16)
void raise_g(double vcov[], double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); 
int Rtoprim_nu_calc(double U[NPR_NU], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_NU], int lim);
#if(NEUTRINOS_M1)

//Inversion from radiation conserved to primitive quantities
int Rtoprim_nu(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], int lim) {
	double U_tmp[NPR_NU], prim_tmp[NPR_NU];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);

	for (int sp = 0; sp < NU_SPECIES; sp++) {
		//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
		for (i = 0; i < NPR_NU; i++) U_tmp[i] = alpha * U[i + index_nu(UU_NU, sp)] / gdet;

		//Transform the PRIMITIVE variables into the new system
		for (i = 0; i < NPR_NU; i++) prim_tmp[i] = prim[i + index_nu(UU_NU, sp)];

		//Do inversion
		ret = Rtoprim_nu_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

		//Transform new primitive variables back if there was no problem
		for (i = 0; i < NPR_NU; i++) {
			prim[i + index_nu(UU_NU, sp)] = prim_tmp[i];
		}
	}

	return(ret);
}

// Limits radiation with either BASIC or TYPE2 approaches
int Rtoprim_nu_calc(double U[NPR_NU], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_NU], int lim)
{
	double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f;
	double Tnu;
	int i, returnval = 0;

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise_g(Qcov, gcon, Qcon);

	ncov = -sqrt(-1. / gcon[0][0]);
	ncon[0] = gcon[0][0] * ncov;
	ncon[1] = gcon[0][1] * ncov;
	ncon[2] = gcon[0][2] * ncov;
	ncon[3] = gcon[0][3] * ncov;

	Qdotn = Qcon[0] * ncov; //-Erad in McKinney2013
	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn * Qdotn; //Utilde^2 in McKinney2013

	if (Qtsq < 0.0) {
		Qtsq = 0.0;
		Qtcon[1] = 0.;
		Qtcon[2] = 0.;
		Qtcon[3] = 0.;
	}

	y = Qtsq / (Qdotn * Qdotn + 1.e-150); //Definition from McKinney2013. Should only range [0,1].
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	prim[4] = U[4] / sqrt(gammasq);

	if (y > y_max || y < 0. || isnan(Qdotn) || Qdotn > 0.0 || isnan(prim[1]) || isnan(prim[2]) || isnan(prim[3])) {
		Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++)prim[i] = GAMMAMAX_NU * Qtcon[i] / Uabs;

		qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3]
			+ 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
		if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_NU * GAMMAMAX_NU - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;

		if (lim == TYPE2) {
			if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
				Qdotn = -(1e-30 + sqrt(fabs(Qtsq) / y_max));
			}
			pressure = -Qdotn / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
			prim[0] = 1e-30 + pressure * 3.; // Erad = 3*p_rad	

			returnval = 1;
		}
		else if (lim == TYPE3) {
			// If energy density is negative, reset it to floor value
			if (Qdotn > 0.0) {
				prim[0] = 1.e-30;
				prim[1] = 0.;
				prim[2] = 0.;
				prim[3] = 0.;
				Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
				prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
			}
			// Causality violation: rescale!
			else if (y > y_max) {
				pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
				prim[0] = pressure * 3.; // Erad = 3*p_rad
				for (i = 1; i < 4; i++) prim[i] = Qtcon[i] / (4. * pressure * GAMMAMAX_NU);
				prim[4] = U[4] / GAMMAMAX_NU;
			}
		}
		else {
			if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
				//if (Qdotn > 0.0 || y < 0.0) {
				prim[0] = 1.e-30;
				prim[1] = 0.;
				prim[2] = 0.;
				prim[3] = 0.;
				Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
				prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
			}
			else {
				//else if (y > y_max) {
				pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
				prim[0] = pressure * 3.; // Erad = 3*p_rad
				//for (i = 1; i < 4; i++) prim[i] = Qtcon[i] / (4. * pressure * GAMMAMAX_NU);
				//prim[4] = U[4] / GAMMAMAX_NU;
			}
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
		}
		if (!isfinite(prim[0]))prim[0] = 1.e-30;
		if (!isfinite(prim[1]))prim[1] = 0.0;
		if (!isfinite(prim[2]))prim[2] = 0.0;
		if (!isfinite(prim[3]))prim[3] = 0.0;

		//Floor on photon number+
		Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		if (lim != TYPE3) prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
		//prim[4] = U[4];
		returnval = 1;
	}

	if (prim[4] < 0.0) {
		Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
		returnval = 1;
	}

	return(returnval);
}

void calc_ymax(void) {
	int keep_iterating, n_iter;
	double E_old, E_new, errx, dEdy, y_new, y_old;
	keep_iterating = 1;
	n_iter = 0;
	y_old = 0.9998; //Gives gamma=25

	//Calculate deviation from 0
	E_old = GAMMAMAX_NU * GAMMAMAX_NU - (2.0 - y_old + sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old);

	while (keep_iterating) {
		//Calculate gradient dEdy
		dEdy = (0.375 * y_old - 0.25 * sqrt(4.0 - 3.0 * y_old) - 0.625) / (sqrt(4.0 - 3.0 * y_old) * (1.0 - y_old) * (1.0 - y_old));

		/* Make the newton step: */
		y_new = MY_MIN(y_old - (E_old) / dEdy, 0.99999999999999);

		//Calculate deviation from 0
		E_new = GAMMAMAX_NU * GAMMAMAX_NU - (2.0 - y_new + sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new);

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		errx = fabs(E_new) / (GAMMAMAX_NU * GAMMAMAX_NU);

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER * 4))) {
			keep_iterating = 0;
		}
		//fprintf(stderr, "y_max set to %f and gamma_rad becomes %f \n", y_new, sqrt((2.0 - y_new + sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new)));

		y_old = y_new;
		E_old = E_new;

		n_iter++;
	}   // END of while(keep_iterating)
	y_max = y_new;
}

void init_nulib_table(void) {
	FILE* fp;
	#if (NU_SPECIES>1)
	char fname_nulib_table[] = "nulib_table_Nsp3.bdat";
	#else
	char fname_nulib_table[] = "nulib_table_Nsp1.bdat";
	#endif
	fp = fopen(fname_nulib_table, "rb");
	if (NULL == fp) {
		fprintf(stderr, "Couldn't open %s for reading, exiting\n", fname_nulib_table);
		exit(1234);
	}

	//..read the nulib table
	fread(&nu_kappa_emiss[0], sizeof(double), NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES, fp);
	fread(&nu_kappa_abs[0], sizeof(double), NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES, fp);
	fread(&nu_kappa_scatt[0], sizeof(double), NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES, fp);
	fread(&nu_kappa_emiss_N[0], sizeof(double), NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES, fp);

	fclose(fp);

	return;
}

// Nulib tables interpolation
void interp_nulib_table(double rho, double Tgas, double ye, int species, double* nu_avg, double* kappa_A, double* kappa_S, double* nu_N) {
	int iat = (int)((log10(rho) - nulib_dlo) * (double)(NULIB_RHO - 1) / (nulib_dhi - nulib_dlo)) + 1;
	int jat = (int)((log10(Tgas) - nulib_tlo) * (double)(NULIB_TEMP - 1) / (nulib_thi - nulib_tlo)) + 1;
	int kat = (int)((ye - nulib_ylo) * (double)(NULIB_YE - 1) / (nulib_yhi - nulib_ylo)) + 1;
	iat = MY_MAX(1, MY_MIN(iat, NULIB_RHO - 1)) - 1;
	jat = MY_MAX(1, MY_MIN(jat, NULIB_TEMP - 1)) - 1;
	kat = MY_MAX(1, MY_MIN(kat, NULIB_YE - 1)) - 1;

	double dstp = (nulib_dhi - nulib_dlo) / (double)(NULIB_RHO - 1);
	double tstp = (nulib_thi - nulib_tlo) / (double)(NULIB_TEMP - 1);
	double ystp = (nulib_yhi - nulib_ylo) / (double)(NULIB_YE - 1);
	double nulib_d_iat = pow(10.0, (nulib_dlo + iat * dstp));
	double nulib_t_jat = pow(10.0, (nulib_tlo + jat * tstp));
	double nulib_y_kat = (nulib_ylo + kat * ystp);
	double nulib_dd_iat = pow(10.0, (nulib_dlo + (iat + 1) * dstp)) - pow(10.0, (nulib_dlo + iat * dstp));
	double nulib_dt_jat = pow(10.0, (nulib_tlo + (jat + 1) * tstp)) - pow(10.0, (nulib_tlo + jat * tstp));
	double nulib_dy_kat = ystp;

	double xd = MY_MAX((rho - nulib_d_iat) / nulib_dd_iat, 0.0);
	double xt = MY_MAX((Tgas - nulib_t_jat) / nulib_dt_jat, 0.0);
	double xy = MY_MAX((ye - nulib_y_kat) / nulib_dy_kat, 0.0);
	double mxd = 1.0 - xd;
	double mxt = 1.0 - xt;
	double mxy = 1.0 - xy;

	*nu_avg = 
		nu_kappa_emiss[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt* mxd* mxy +
		nu_kappa_emiss[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * xd * mxy +
		nu_kappa_emiss[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * mxd * mxy +
		nu_kappa_emiss[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * xd * mxy +
		nu_kappa_emiss[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * mxd * xy +
		nu_kappa_emiss[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * xd * xy +
		nu_kappa_emiss[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * mxd * xy +
		nu_kappa_emiss[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * xd * xy;
	*kappa_A = 
		nu_kappa_abs[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt* mxd* mxy +
		nu_kappa_abs[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * xd * mxy +
		nu_kappa_abs[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * mxd * mxy +
		nu_kappa_abs[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * xd * mxy +
		nu_kappa_abs[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * mxd * xy +
		nu_kappa_abs[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * xd * xy +
		nu_kappa_abs[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * mxd * xy +
		nu_kappa_abs[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * xd * xy;
	*kappa_S = 
		nu_kappa_scatt[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt* mxd* mxy +
		nu_kappa_scatt[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * xd * mxy +
		nu_kappa_scatt[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * mxd * mxy +
		nu_kappa_scatt[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * xd * mxy +
		nu_kappa_scatt[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * mxd * xy +
		nu_kappa_scatt[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * xd * xy +
		nu_kappa_scatt[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * mxd * xy +
		nu_kappa_scatt[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * xd * xy;
	*nu_N = 
		nu_kappa_emiss_N[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt* mxd* mxy +
		nu_kappa_emiss_N[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * xd * mxy +
		nu_kappa_emiss_N[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * mxd * mxy +
		nu_kappa_emiss_N[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * xd * mxy +
		nu_kappa_emiss_N[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * mxd * xy +
		nu_kappa_emiss_N[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * xd * xy +
		nu_kappa_emiss_N[(iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * mxd * xy +
		nu_kappa_emiss_N[(iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * xd * xy;

	return;
}

// Neutrino temperature calculation: needs EOS
void calc_neutrino_temperature(double* ph, double ener_nu_avg, double* Tnu_over_Tgas, int species) {

	// Call EOS to get gas temperature and electron chemical potential
	double mu_ele;
	eos_mode_rhotemp_etaele(ph[RHO], ph[UU], ph[YE], &mu_ele
		#if (DONUCLEAR)
		, ph[XALPHA], ph[XATM]
		#endif
	);

	double mu_n, mu_p, mu_nu;
	calc_mu_np(ph[RHO], ph[UU], 1.0 - ph[YE], ph[YE], &mu_n, &mu_p);
	mu_nu = mu_p + mu_ele - mu_n + (MP_CGS + ME_CGS - MN_CGS) * C_CGS * C_CGS / (BOLTZ_CGS * ph[UU]);

	double F2, F3;

	#if (NU_SPECIES == 1) 
	species = 2;
	#endif

	// 0 == electron neutrino
	// 1 == electron antineutrino (mu_nua = - mu_nu)
	// 2 == heavy lepton neutrinos (mu_nux = 0)
	if (species == 0) {
		F2 = calc_fermiint2(mu_nu);
		F3 = calc_fermiint3(mu_nu);
	}
	else if (species == 1) {
		F2 = calc_fermiint2(-mu_nu);
		F3 = calc_fermiint3(-mu_nu);
	}
	else if (species == 2) {
		F2 = calc_fermiint2(0.0);
		F3 = calc_fermiint3(0.0);
	}

	*Tnu_over_Tgas = (ener_nu_avg * C_CGS * C_CGS) * F2 / (F3 + 1e-30) / (BOLTZ_CGS * ph[UU]);
	*Tnu_over_Tgas = MY_MIN(1.0, *Tnu_over_Tgas);
	// 222.
	//if (*Tnu_over_Tgas != *Tnu_over_Tgas) printf("\n\t [sp=%d] T_nu/T_g = %e, <e>=%e, T_g=%e (F2, F3 = %e %e)", species, *Tnu_over_Tgas, ener_nu_avg, ph[UU], F2, F3);
}

double calc_fermiint2(double x) {
	if (x > 0.001)
		return (x * x * x / 3.0 + 3.2899 * x) / (1.0 - exp(-1.8246 * x));
	else
		return 2.0 * exp(x) / (1.0 + 0.1092 * exp(0.8908 * x));
}

double calc_fermiint3(double x) {
	if (x > 0.001)
		return (x * x * x * x / 4.0 + 4.9348 * x * x + 11.3644) / (1.0 + exp(-1.9039 * x));
	else
		return 6.0 * exp(x) / (1.0 + 0.0559 * exp(0.9069 * x));
}

// Neutron-proton chemical potentials assuming ideal gas
// From: NuLib code
void calc_mu_np(double rho, double T_gas, double x_n, double x_p, double* mu_n, double* mu_p) {
	x_n = MY_MAX(x_n, 1e-20);
	x_p = MY_MAX(x_p, 1e-20);

	double n_n = x_n * rho * MASS_DENSITY_SCALE / MN_CGS;
	double n_p = x_p * rho * MASS_DENSITY_SCALE / MP_CGS;

	if (n_n > 0.0)
		*mu_n = log(0.5 * n_n * pow(PLANCK_CGS * PLANCK_CGS / (2.0 * M_PI * MN_CGS * BOLTZ_CGS * T_gas), 1.5));
	else
		*mu_n = 0.0;

	if (n_p > 0.0)
		*mu_p = log(0.5 * n_p * pow(PLANCK_CGS * PLANCK_CGS / (2.0 * M_PI * MP_CGS * BOLTZ_CGS * T_gas), 1.5));
	else
		*mu_p = 0.0;

	// Danat: didn't include Coulomb corrections for mu_p for now
}

void eos_mode_rhotemp_etaele(double dens, double temp, double ye, double* mu_ele
	#if (DONUCLEAR)
	, double x_alpha, double x_atm
	#endif
) {
	double free, df_d, df_t, df_dd, df_tt, df_dt, etaele, dpepdd;
	temp *= conv_T_CODE2CGS;
	dens *= conv_dens_CODE2CGS;
	interp_eostable(dens, temp, dens * ye, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, &etaele 
		#if (DONUCLEAR)
		, x_alpha, x_atm
		#endif
		);
	*mu_ele = etaele;
}
#endif
