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

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
	for (i = 0; i < NPR_NU; i++) U_tmp[i] = alpha * U[i + UU_NU] / gdet;

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i < NPR_NU; i++) prim_tmp[i] = prim[i + UU_NU];

	//Do inversion
	ret = Rtoprim_nu_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i < NPR_NU; i++) {
		prim[i + UU_NU] = prim_tmp[i];
	}

	return(ret);
}

// Limits radiation with either BASIC or TYPE2 approaches
int Rtoprim_nu_calc(double U[NPR_NU], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_NU], int lim)
{
	double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f;
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

	//Check for bad values of -Erad and U_tilde^2; If values are nan floor them
	if (!isfinite(Qdotn)) Qdotn = -(1.e-30);
	if (!isfinite(Qtsq)) Qtsq = 0.0;

	y = Qtsq / (Qdotn * Qdotn + 1.e-30); //Definition from McKinney2013. Should only range [0,1].
	if (y < 0. || isnan(y)) y = 0.;
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_nu as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_nu

	// utilde ^i _nu = gam_nu * Utilde^i / (4 * p * gam_nu^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	prim[4] = U[4] / sqrt(gammasq);

	if (Qdotn > 0.) { //Negative internal energy
		if (lim == TYPE2) {
			Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
			for (i = 1; i < 4; i++) {
				if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
				prim[i] = Qtcon[i] / Uabs;
			}
			qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3]
				+ 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
			if (qsq < 1.E-10) qsq = 1.E-10; // set floor
			gammasq = 1. + qsq;

			f = 0.;// sqrt((GAMMAMAX_NU * GAMMAMAX_NU - 1.) / (gammasq - 1.));
			if (f < 10000000.0) {
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			}
			Qdotn = -(1.e-150 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_nu

			returnval = 1;
		}
		else {
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
			gammasq = 1.0;
		}

		prim[4] = U[4] / sqrt(gammasq);
	}
	else if (y > y_max) {
		Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++) {
			if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
			prim[i] = Qtcon[i] / Uabs;
		}
		qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3]
			+ 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
		if (qsq < 1.E-10 || !isfinite(qsq)) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_NU * GAMMAMAX_NU - 1.) / (gammasq - 1.));
		//if (f < 10000000.0 && y<1.0-0.000000001){
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;
		//}
		//else {
		//	prim[1] = 0;
		//	prim[2] = 0;
		//	prim[3] = 0;
	//	}
		if (lim == TYPE2) {
			Qdotn = -(1.e-30 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_nu
			returnval = 1;
		}
		else if (!isfinite(prim[0])) {
			prim[0] = 1.e-30;
		}

		prim[4] = U[4] / sqrt(GAMMAMAX_NU * GAMMAMAX_NU);
	}
	return returnval;
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
	char fname_nulib_table[] = "nulib_table.bdat";

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
#endif
