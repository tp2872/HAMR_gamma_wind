#include "decs_MPI.h"
#define MACHINE_PREC (pow(10., -14.))

#define epsem (2.22E-16)
double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); int Rtoprim_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);

int implicit_rad_solve(double pb[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt) {
	double error_t;
	int flag, do_staged;

	//Check if fluid is in extreme radiation subdominant regime
	if ((U[UU_RAD] / U[UU]) < 10.0 * MACHINE_PREC || (pb[UU_RAD] / pb[UU]) < 10.0 * MACHINE_PREC) {
		//If error is below set margin, accept solution, otherwise try PRAD
		implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is below set margin, accept solution, otherwise try PRAD staged
		implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 1, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 1, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 1, 1);
	}
	else {
		//If error is below set margin, accept solution, otherwise try PRAD
		implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 1, 0);

		//If error is below set margin, accept solution, otherwise try PRAD staged
		implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is below set margin, accept solution, otherwise try PRAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 0, 1);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U, geom, dU, Dt, &error_t, 1, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U, geom, dU, Dt, &error_t, 1, 1);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U, geom, dU, Dt, &error_t, 1, 1);
	}

	//As final resort attempt subcycling
	if (error_t > pow(10., -7.)) subcycle_rad_solve(pb, U, geom, Dt);
}


int subcycle_rad_solve(double pb[NPR], double U[NPR], struct of_geom geom, double Dt) {
	double factor, remainder=1.0, dU[NPR], Uh[NPR], ph[NPR], fraction;
	int flag=0, keep_iterating = 1, nstep = 0, k;

	//Something like Courant number in implicity scheme; e.g. we do not let the source term in each step be bigger than this
	fraction = 0.25;

	//Set halfstep variables
	for (k = 0; k < NPR; k++) ph[k] = pb[k];

	while (keep_iterating && nstep < 1000) {
		
		//Set size of subcycling timestep
		if (fraction * MY_MIN(U[UU], U[UU_RAD]) >= fabs(dU[UU_RAD] * Dt)) keep_iterating = 0;
		factor = MY_MIN(fraction * MY_MIN(U[UU], U[UU_RAD]) / fabs(dU[UU_RAD] * Dt), remainder);
		remainder -= factor;

		//Half step in 2nd order scheme
		source_rad(ph, &geom, dU);

		Uh[UU_RAD] += 0.5 * factor * Dt * dU[UU_RAD];
		Uh[U1_RAD] += 0.5 * factor * Dt * dU[U1_RAD];
		Uh[U2_RAD] += 0.5 * factor * Dt * dU[U2_RAD];
		Uh[U3_RAD] += 0.5 * factor * Dt * dU[U3_RAD];
		Uh[UU] += 0.5 * factor * Dt * dU[UU];
		Uh[U1] += 0.5 * factor * Dt * dU[U1];
		Uh[U2] += 0.5 * factor * Dt * dU[U2];
		Uh[U3] += 0.5 * factor * Dt * dU[U3];

		flag = Utoprim_2d(Uh, geom.gcov, geom.gcon, geom.g, ph,NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(Uh, geom.gcov, geom.gcon, geom.g, ph, ph[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(Uh, geom.gcov, geom.gcon, geom.g, ph, ph[KTOT], NEWT_TOL);
				}
			}
		}
		#endif

		Rtoprim(Uh, geom.gcov, geom.gcon, geom.g, ph, BASIC);

		//Full step in 2nd order scheme
		source_rad(ph, &geom, dU);
		U[UU_RAD] += factor * Dt * dU[UU_RAD];
		U[U1_RAD] += factor * Dt * dU[U1_RAD];
		U[U2_RAD] += factor * Dt * dU[U2_RAD];
		U[U3_RAD] += factor * Dt * dU[U3_RAD];
		U[UU] += factor * Dt * dU[UU];
		U[U1] += factor * Dt * dU[U1];
		U[U2] += factor * Dt * dU[U2];
		U[U3] += factor * Dt * dU[U3];

		flag = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pb, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pb, pb[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pb, pb[KTOT], NEWT_TOL);
				}
			}
		}
		#endif

		Rtoprim(U, geom.gcov, geom.gcon, geom.g, pb, BASIC);

		nstep++;
	}
	if (nstep >= 1000) {
		return 1;
	}
	else {
		return 0;
	}
}

int implicit_rad_solve_PMHD(double pb[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt, double *error_t, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb[NPR], dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset= pow(10., -10.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating, fail, n_iter_jacob;
	
	//Set guess values for primitives after implicit step based on optical depth
	kappa_abs = calc_kappa_abs(pb);
	kappa_emmit = calc_kappa_emmit(pb);
	kappa_es = calc_kappa_es(pb);
	tau = (kappa_abs + kappa_emmit + kappa_es) * pb[RHO];
	if (tau > 0.66) {
		for (k = 0; k < NPR; k++) {
			pb_old[k] = pb[k];
		}
	}
	else {
		Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL);
		Rtoprim(U, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);
	}

	//Set norm for error calculation
	norm = 0.25 * (fabs(U[UU_RAD]) + fabs(U[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));

	//If error allready small skip the implicit solve
	if (do_entropy == 1) error_new[0] = 0.25 * T_GAS * (fabs(Dt * dU_new[KTOT]) / (norm));
	else error_new[0] = 0.25 * (fabs(Dt * dU_new[UU_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_new[U1_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_new[U2_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_new[U3_RAD]) / norm);
	if (do_entropy == 1) error_new[0] += 0.25 * (fabs(U[UU_RAD]) + fabs(U[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));
	else error_new[0] += 0.25 * (fabs(U_new[UU] - U[UU] - Dt * dU_new[UU]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U1] - U[U1] - Dt * dU_new[U1]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U2] - U[U2] - Dt * dU_new[U2]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U3] - U[U3] - Dt * dU_new[U3]) / norm);

	/* Start the Newton-Raphson iterations : */
	n_iter = 0;
	keep_iterating = 1;
	while (keep_iterating) {
		//Calculate jacobian dEdpb
		get_state(pb_old, &geom, &q);
		mhd_calc(pb_old, 0, &q, &U_old[UU]);
		for (k = UU; k <= U3; k++)U_old[k] *= geom->g;
		source_rad(pb_old, &geom, dU_old);
		for (k = UU; k <= U3; k++) E_old[k] = (U_old[k] - U[k] - Dt * dU_old[k]);		

		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;
			fail = 0;
			for (k = UU; k <= U3; k++) dpb[k] = 0.;
			do {
				if (i == UU) {
					dpb[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb[i] + dpb[i];
				}
				else {
					dpb[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU][i - UU]);
					pb_new[i] = pb[i] + dpb[i];
				}

				get_state(pb_new, &geom, &q);
				pb_new[RHO] = U[RHO] / q.ucon[0] / geom->g; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				mhd_calc(pb_new, 0, &q, &U_new[UU]); // Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;

				U_new[UU_RAD] = U[UU_RAD] - (U_new[UU] - U[UU]);
				U_new[U1_RAD] = U[U1_RAD] - (U_new[U1] - U[U1]);
				U_new[U2_RAD] = U[U2_RAD] - (U_new[U2] - U[U2]);
				U_new[U3_RAD] = U[U3_RAD] - (U_new[U3] - U[U3]);

				fail = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				source_rad(pb_new, &geom, dU_new);

				for (k = U1; k <= U3; k++) {
					E_new[k] = (U_new[k] - U[k] - Dt * dU_new[k]);
					dEdpb[k - UU][i - UU] = (E_new[k] - E_old[k]) / dpb[i];
				}
				if (do_entropy == 1)E_new[UU] = (U_new[KTOT] - U[KTOT] - Dt * dU_new[KTOT]);
				else E_new[k] = (U_new[UU] - U[UU] - Dt * dU_new[UU]);

				n_iter_jacob++;
			} while (fail && n_iter_jacob<6);
		}

		if (invert_matrix(dEdpb, dEdpb_inv) == 1) {
			fprintf(stderr, "Implicit rad solve error \n");
			fail = 1;
			break;;
		}

		//Tg = (GAMMA-1.0)*pb_new[UU] / pb_new[RHO];
		//error += fabs((U_new[KTOT] - U[KTOT])*Tg + Dt*dU_new[KTOT]);
		//norm += U[KTOT] * Tg;
		//error_norm = error / norm;

		/* Make the newton step: */
		if (do_staged == 0) {
			for (k = 0; k < 4; k++) {
				dpb[k + 1] = -(E_old[1] * dEdpb_inv[k][0] + E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
				pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
			}
		}
		else {
			if (n_iter == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb[k + 1] = -(E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]); 
					pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
				}
			}
			if (n_iter == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb[k + 1] = -(E_old[1] * dEdpb_inv[k][0]);
					pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
				}
			}
			else {
				if (n_iter == 0) {
					for (k = 0; k < 4; k++) { //full 4d step
						dpb[k + 1] = -(E_old[1] * dEdpb_inv[k][0] + E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
						pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
					}
				}
			}
		}
		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5*fabs(pb_new[UU]);

		//Obtain new conserved quantaties for MHD variables
		get_state(pb_new, &geom, &q);
		pb_new[RHO] = U[RHO] / q.ucon[0] / geom->g;
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		U_new[KTOT] = geom->g*pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);
		for (k = UU; k <= U3; k++)U_new[k] *= geom->g;

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U[UU_RAD] - (U_new[UU] - U[UU]);
		U_new[U1_RAD] = U[U1_RAD] - (U_new[U1] - U[U1]);
		U_new[U2_RAD] = U[U2_RAD] - (U_new[U2] - U[U2]);
		U_new[U3_RAD] = U[U3_RAD] - (U_new[U3] - U[U3]);

		//Get new radiation primitives
		pb_new[RHO] = U_new[RHO] / q.ucon[0] / geom->g; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
		Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);
		source_rad(pb_new, &geom, dU_new);

		//Calculate iterated error
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		if (do_entropy == 1) error_new[n_iter % 5] = 0.25 * T_GAS* (fabs(U_new[KTOT] - U[KTOT] - Dt * dU_new[KTOT]) / (norm));
		else error_new[n_iter % 5] = 0.25 * (fabs(U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
		else offset = pow(10., -8.);
		
		//If we've reached the tolerance level, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= pow(10,-12.)) || (n_iter >= 20)) {
			keep_iterating = 0;
		}
				
		//If the residual drops below machine precision, stop iterating
		if ((fabs(U_new[UU]/U[UU]) - 1.0) + (fabs(U_new[UU_RAD] / U[UU_RAD]) - 1.0) < 10.*epsem) {
			keep_iterating = 0;
		}

		if (keep_iterating) {
			for (k = 0; k < NPR; k++) pb_old[k] = pb_new[k];
		}
		n_iter++;

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5]+ error_new[(n_iter - 3) % 5]+ error_new[(n_iter - 2) % 5])<0.25*(error_new[(n_iter - 1) % 5]+ error_new[(n_iter - 0) % 5]))) {
			keep_iterating=0;
		}
	}

	//Recalculate radiation energy density with BASIC limiter if radiation energy density turns negatie
	if (pb_new[UU_RAD] < 0.0)Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);

	//Calculate total error
	n_iter = n_iter - 1;
	if (do_entropy == 1) error_new[n_iter % 5] += 0.25 * (fabs(U[UU_RAD]) + fabs(U[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));
	else error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U[UU] - Dt * dU_new[UU]) / norm);
	error_new[n_iter % 5] += 0.25 * (fabs(U_new[U1] - U[U1] - Dt * dU_new[U1]) / norm);
	error_new[n_iter % 5] += 0.25 * (fabs(U_new[U2] - U[U2] - Dt * dU_new[U2]) / norm);
	error_new[n_iter % 5] += 0.25 * (fabs(U_new[U3] - U[U3] - Dt * dU_new[U3]) / norm);

	//If error decreased, update variables
	if (fabs(error_new[n_iter % 5]) < error_t[0]) {
		error_t[0] = error_new[n_iter % 5];
		for (k = 0; k < NPR; k++) {
			pb[k] = pb_new[k];
		}
	}
	else return(0);
}

// This method iterates R^t_mu
// change to *geom pointer
// change dU to more intuitive name for jacobian calc
// change pow() to e
int implicit_rad_solve_URAD(double pb[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dEdU[4][4], dEdU_inv[4][4], bsq, error_new;
	double dU_offset[NPR]; // change all dU to this. then copy into dU at end
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating;

	// Initialize various parameters and variables:
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		dpb[k] = 0.;
		U_old[k] = U[k];
		U_new[k] = U[k];
		dU_offset[k] = 0.;

	}

	n_iter = 0;
	error_new = 1000000000000000.0;
	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;

	/*
	 ////// BEV: do I need anything like this (from entropy_UMHD):
	k = UU;
	pb_old[k] = 10 * pb[k];
	pb_new[k] = 10 * pb[k];
	U[UU] = U[UU] - U[RHO];
	//////
	 */

	while (keep_iterating) {
		//Calculate jacobian dEdU
		source_rad(pb_old, &geom, dU_old);
		for (k = UU_RAD; k <= U3_RAD; k++) E_old[k] = (U_old[k] - U[k] - Dt * dU_old[k]);

		for (i = UU_RAD; i <= U3_RAD; i++) {
			if (i == UU_RAD) {
				for (k = UU_RAD; k <= U3_RAD; k++) dU[k] = 0.;
				dU[i] = pow(10., -9.) * (U_old[UU_RAD]);
			}
			else {
				for (k = UU_RAD; k <= U3_RAD; k++) dU[k] = 0.;
				dU[i] = pow(10., -9.) / sqrt(fabs(geom.gcov[i - UU_RAD][i - UU_RAD])) * U_old[UU_RAD];
			}

			// Calculate new radiation U_(i+1) --> update: now for all prims. before it was just for UU to U3
			for (k = 0; k < NPR; k++) U_new[k] = U_old[k] + dU_offset[k];

			// Step 1: Set DeltaTumu = -DeltaRtmu
			U_new[UU] = U[UU] - (U_new[UU_RAD] - U[UU_RAD]);
			U_new[U1] = U[U1] - (U_new[U1_RAD] - U[U1_RAD]);
			U_new[U2] = U[U2] - (U_new[U2_RAD] - U[U2_RAD]);
			U_new[U3] = U[U3] - (U_new[U3_RAD] - U[U3_RAD]);

			// Step 3: Invert from conserved (latest soln for T^t_mu(i+1) and R^t_mu(i+1) to gas and rad prims (latest soln for P(i+1)
			Utoprim_2d(U_new, geom.gcov, geom.gcon, geom.g, pb_new, NEWT_TOL);
			Rtoprim(U_new, geom.gcov, geom.gcon, geom.g, pb_new, BASIC);

			// Step 4: Recompute all U from P_i+1 for consistency
			// (in case of inversion failure or modification)
			get_state_rad(pb_new, &geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			get_state(pb_new, &geom, &q);
			mhd_calc(pb_new, 0, &q, &U_new[UU]);
			source_rad(pb_new, &geom, dU_new);

			// This loop specific to the solver: what prims are iterated
			// BEV: is this right?
			for (k = UU_RAD; k <= U3_RAD; k++) {
				E_new[k] = (U_new[k] - U[k] - Dt * dU_new[k]);
				dEdU[k - UU_RAD][i - UU_RAD] = (E_new[k] - E_old[k]) / dU_offset[i];
			}
		}

		if (invert_matrix(dEdU, dEdU_inv) == 1) {
			fprintf(stderr, "Error urad \n");
			break;;
		}

		/* Make the newton step: */
		for (k = 0; k < 4; k++) {

			dU_offset[k + UU_RAD] = -(E_old[1] * dEdU_inv[k][0] + E_old[2] * dEdU_inv[k][1] + E_old[3] * dEdU_inv[k][2] + E_old[4] * dEdU_inv[k][3]);
			U_new[k + UU_RAD] = U_old[k + UU_RAD] + dU_offset[k + UU_RAD];
		}

		// Step 1: Set DeltaTumu = -DeltaRtmu
		U_new[UU] = U[UU] - (U_new[UU_RAD] - U[UU_RAD]);
		U_new[U1] = U[U1] - (U_new[U1_RAD] - U[U1_RAD]);
		U_new[U2] = U[U2] - (U_new[U2_RAD] - U[U2_RAD]);
		U_new[U3] = U[U3] - (U_new[U3_RAD] - U[U3_RAD]);

		// Step 3: Invert from conserved (latest soln for T^t_mu(i+1) and R^t_mu(i+1) to gas and rad prims (latest soln for P(i+1)
		Utoprim_2d(U_new, geom.gcov, geom.gcon, geom.g, pb_new, NEWT_TOL);
		Rtoprim(U_new, geom.gcov, geom.gcon, geom.g, pb_new, BASIC);

		// Step 4: Recompute all U from P_i+1 for consistency
		// (in case of inversion failure or modification)
		get_state_rad(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		get_state(pb_new, &geom, &q);
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		source_rad(pb_new, &geom, dU_new);


		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		error_new = 0.25 * (fabs(U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]) / fabs(U[UU_RAD]));
		error_new += sqrt(geom.gcov[1][1]) * 0.25 * (fabs(U_new[U1_RAD] - U[U1_RAD] - Dt * dU_new[U1_RAD]) / fabs(U[UU_RAD]));
		error_new += sqrt(geom.gcov[2][2]) * 0.25 * (fabs(U_new[U2_RAD] - U[U2_RAD] - Dt * dU_new[U2_RAD]) / fabs(U[UU_RAD]));
		error_new += sqrt(geom.gcov[3][3]) * 0.25 * (fabs(U_new[U3_RAD] - U[U3_RAD] - Dt * dU_new[U3_RAD]) / fabs(U[UU_RAD]));

		//if(n_iter>=0)printf("iter: %d test: %f \n", n_iter, log(error_new)/log(10.));

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(error_new) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		else {
			for (k = 0; k < NPR; k++) {
				pb_old[k] = pb_new[k];
				U_old[k] = U_new[k];
			}
		}

		n_iter++;
	}

	if (fabs(error_new) > MIN_NEWT_TOL * 1000.) {
		return(1);
	}
	if (fabs(error_new) <= NEWT_TOL * 1000.) {
		for (k = 0; k < NPR; k++) pb[k] = pb_new[k];
		return(0);
	}

	return(0);
}

// This method iterates E_RAD an U_rad
int implicit_rad_solve_PRAD(double pb[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb[NPR], dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating;

	// Initialize various parameters and variables:
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		dpb[k] = 0.;
	}

	n_iter = 0;
	error_new = 1000000000000000.0;
	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;
	while (keep_iterating) {
		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			if (i == UU_RAD) {
				for (k = UU_RAD; k <= U3_RAD; k++) dpb[k] = 0.;
				dpb[i] = pow(10., -9.)*(pb_old[UU_RAD]);
			}
			else {
				for (k = UU_RAD; k <= U3_RAD; k++) dpb[k] = 0.;
				dpb[i] = pow(10., -9.) / sqrt(fabs(geom.gcov[i - UU_RAD][i - UU_RAD]));
			}

			// Calculate new radiation P_(i+1) --> update: now for all prims. before it was just for UU to U3
			for (k = 0; k < NPR; k++) pb_new[k] = pb_old[k] + dpb[k];

			// Step 1: Compute R^t,_mu from radiation P_(i+1)
			get_state_rad(pb_new, &geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);

			// Step 2: Estimate G_s using old prims P_n
			source_rad(pb_old, &geom, dU_old);
			//BEV Q: do I have to do anything separately to get G_s?

			// Step 3: Set DeltaTumu = -DeltaRtmu
			U_new[UU] = U[UU] - (U_new[UU_RAD] - U[UU_RAD]);
			U_new[U1] = U[U1] - (U_new[U1_RAD] - U[U1_RAD]);
			U_new[U2] = U[U2] - (U_new[U2_RAD] - U[U2_RAD]);
			U_new[U3] = U[U3] - (U_new[U3_RAD] - U[U3_RAD]);

			// Step 4: Invert T^t_mu to gas prims (latest gas variables in P_i+1)
			Utoprim_2d(U_new, geom.gcov, geom.gcon, geom.g, pb_new, NEWT_TOL);

			// Step 5: Recompute R^t_mu for consistency
			get_state_rad(pb_new, &geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			source_rad(pb_new, &geom, dU_new);

			// This loop specific to the solver: what prims are iterated
			for (k = UU_RAD; k <= U3_RAD; k++) {
				E_old[k] = (U_old[k] - U[k] - Dt*dU_old[k]);
				E_new[k] = (U_new[k] - U[k] - Dt*dU_new[k]);
				dEdpb[k - UU_RAD][i - UU_RAD] = (E_new[k] - E_old[k]) / dpb[i];
			}
		}

		if (invert_matrix(dEdpb, dEdpb_inv) == 1) {
			fprintf(stderr, "Error prad \n");
			break;;
		}

		//Tg = (GAMMA - 1.)*pb_new[UU] / pb_new[RHO];
		//error += fabs((U_new[KTOT] - U[KTOT])*Tg + Dt*dU_new[KTOT]);
		//norm += U[KTOT] * Tg;
		//error_norm = error / norm;

		/* Make the newton step: */
		for (k = 0; k < 4; k++) {
			dpb[k + UU_RAD] = -(E_old[1] * dEdpb_inv[k][0] + E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
			pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb[k + UU_RAD];
		}

		// Step 1: Compute R^t,_mu from radiation P_(i+1)
		get_state(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);

		// Step 2: Estimate G_s using old prims P_n
		source_rad(pb_old, &geom, dU_old);

		//get_state_rad(pb_new, &geom, &q_rad);
		//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom.g;

		U_new[UU] = U[UU] - (U_new[UU_RAD] - U[UU_RAD]);
		U_new[U1] = U[U1] - (U_new[U1_RAD] - U[U1_RAD]);
		U_new[U2] = U[U2] - (U_new[U2_RAD] - U[U2_RAD]);
		U_new[U3] = U[U3] - (U_new[U3_RAD] - U[U3_RAD]);

		// Step 4: Invert T^t_mu to gas prims (latest gas variables in P_i+1_
		// BEV: which function does this? There are multiple utoprims
		Utoprim_2d(U_old, geom.gcov, geom.gcon, geom.g, pb_new,NEWT_TOL);

		// Step 5: Recompute T^t_mu for consistency
		get_state_rad(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		source_rad(pb_new, &geom, dU_new);

		//for (k = UU; k <= U3; k++) {
		//printf("test2: %f \n", log(fabs(Dt*dU_new[k] / U_new[k])) / log(10.));
		//}

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		error_new = 0.25 * (fabs(U_new[UU_RAD] - U[UU_RAD] - Dt*dU_new[UU_RAD]) / fabs(U[UU_RAD]));
		error_new += 0.25 * (fabs(U_new[U1_RAD] - U[U1_RAD] - Dt * dU_new[U1_RAD]) / fabs(U[UU_RAD]));
		error_new += 0.25 * (fabs(U_new[U2_RAD] - U[U2_RAD] - Dt * dU_new[U2_RAD]) / fabs(U[UU_RAD]));
		error_new += 0.25 * (fabs(U_new[U3_RAD] - U[U3_RAD] - Dt * dU_new[U3_RAD]) / fabs(U[UU_RAD]));

		//if(n_iter>=0)printf("iter: %d test: %f \n", n_iter, log(error_new)/log(10.));

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(error_new) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		else {
			for (k = 0; k < NPR; k++) pb_old[k] = pb_new[k];
		}

		n_iter++;
	}

	if (fabs(error_new) > MIN_NEWT_TOL*1000.) {
		return(1);
	}
	if (fabs(error_new) <= NEWT_TOL*1000.) {
		for (k = 0; k < NPR; k++) pb[k] = pb_new[k];
		return(0);
	}

	return(0);
}


// Entropy source terms for each implicit step
// Can lead to out of bounds values from inverting gas primitives
int implicit_rad_solve_entropy_UMHD(double pb[NPR], double U[NPR], struct of_geom geom, double dU[NPR], double Dt) {

}

//Inversion from radiation conserved to primitive quantities
int Rtoprim(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], int lim) {

	double U_tmp[NPR_R], prim_tmp[NPR_R];
	double prim_tmp_gas[4]; //BEV added
	int i, ret;
	double alpha;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	/* Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	for (i = 0; i <= U3_RAD - UU_RAD; i++) {
		U_tmp[i] = alpha * U[i + NPR_U] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	for (i = 0; i <= U3_RAD - UU_RAD; i++) {
		prim_tmp[i] = prim[i + NPR_U]; //radiation prims

	}

	//BEV added. need to send gas prims as well
	for (i = 0; i <= 3; i++) {
		prim_tmp_gas[i] = prim[i + 1];
	}

	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, prim_tmp_gas, lim);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		for (i = 0; i <= U3_RAD - UU_RAD; i++) {
			prim[i + NPR_U] = prim_tmp[i];
		}
	}

	return(ret);
}

// Limits radiation with either BASIC or TYPE2 approaches
int Rtoprim_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], double prim_gas[4], int lim)
{

	double Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs;
	double gammasq, y, pressure, f, ymax;
	int i;

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise_g(Qcov, gcon, Qcon);

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	Qdotn = Qcon[0] * ncov[0]; //-Erad in McKinney2013

	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	Qsq = 0.;

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn * Qdotn; //Utilde^2 in McKinney2013

	//if Erad <= 0
	//done for all approaches
	if (-Qdotn <= 0.) {
		prim[0] = pow(10, -300.);
		for (i = 1; i < 4; i++) prim_gas[i] = 0; //BEV: using gas velocities, my understanding of paper
		//return 0;
	}

	y = Qtsq / (Qdotn * Qdotn); //Definition from McKinney2013. Should only range [0,1].

	// if 0 > y > -eps_m
	//done for all approaches
	if (y<0.0 && y>(-epsm)) {
		y = 0; //BEV: guessing this should be 0 for gammasq calculation
		for (i = 1; i < 4; i++) prim[i] = 0.0; //BEV: rad velocities
	}

	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y); //This is NaN problem when y>1

	// best to get solution during evolution
	if (lim == BASIC) {
		ymax = 0.999922; //ymax such that gamma = the gamma_max set by Matthew --> BEV: make outside func. ymax is global like gammamax
		if (y<0.0 && y>(-epsm)) {
			y = 0; //BEV: guessing this should be 0
			for (i = 1; i < 4; i++) prim[i] = 0.0; //rad velocities
		}
		//if (y >= ymax) {
		//    y = ymax; //BEV: HOW TO RESCALE 4-velocities?
		//}
		if (gammasq > GAMMAMAX * GAMMAMAX) { //BEV: equivalent to asking if y>=ymax
			f = sqrt((GAMMAMAX * GAMMAMAX - 1.) / (gammasq - 1.));
			for (i = 1; i < 4; i++) prim[i] *= f;
		}
	}


	// Good for smooth Newton stepping
	if (lim == TYPE2) {
		ymax = 1. - 0.5 * (GAMMAMAX * GAMMAMAX);
		if ((y > ymax)) { //BEV changed from if (y > ymax) then changed back
			Qdotn *= sqrt(ymax / y); //-Erad = -Erad * sqrt(ymax/y) --> BEV: why rescaling rad energy density?

			Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(-Qdotn) + pow(10., -150.));
			for (i = 1; i < 4; i++) {
				prim_gas[i] = sqrt(gammasq) * Qtcon[i] / Uabs;
			}
			// then need to rescale to give desired gam_rad,max --> I think the following?
			f = sqrt((GAMMAMAX * GAMMAMAX - 1.) / (gammasq - 1.));
			for (i = 1; i < 4; i++) prim_gas[i] *= f;

			//y = ymax;
		}
		if (y > (1. - 100. * epsm)) {
			ymax = 0.999922; //ymax such that gamma = the gamma_max set by Matthew
			Qdotn = -(pow(10., -150.) + sqrt(Qtsq / ymax));
		}
	}

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	//printf("gammasq: %f pres: %f \n", ncon[0],log(U[0]));
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	//BEV adding after prim[0] is calculated
	if ((lim == BASIC) && (y<0 && y>(-epsm))) {
		return(0); //BEV: WHY return 0 before the following?
	}


	//BEV: WHY DOING THIS?
	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++)prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	//Matthew's OG version of BASIC
	/*
	if (lim == BASIC) {
		if (y <= 0.) {
			for (i = 1; i < 4; i++)prim[i] = 0.;
		}

		if (gammasq > GAMMAMAX*GAMMAMAX) {
		f = sqrt((GAMMAMAX*GAMMAMAX - 1.) / (gammasq - 1.));
		for (i = 1; i < 4; i++) prim[i] *= f;
		}

	}
	*/

	// BEV: in what case do I return(1)? What specifically fails?

	/* done! */
	return(0);
}