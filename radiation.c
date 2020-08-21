#include "decs_MPI.h"

#define epsem (2.22E-16)
double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); 
int Rtoprim_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);

int implicit_rad_solve(double pb[NPR], double U_i[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt, double cell_size) {
	double error_t=pow(10., 9.);
	int flag, k;
	double pb_i[NPR];
	double delta_U, delta_Ur;
	PLOOP{
		pb_i[k] = pb[k];
		dU[k] = 0.;
	}
	delta_U = (U_i[UU] - U[UU]) / (fabs(U_i[UU]) + fabs(U[UU]));
	delta_Ur = (U_i[UU_RAD] - U[UU_RAD]) / (fabs(U_i[UU_RAD]) + fabs(U[UU_RAD]));

	//Check if fluid is in extreme radiation subdominant regime
	if ((U[UU_RAD] / U[UU]) < 10.0 * epsem || (pb[UU_RAD] / pb[UU]) < 10.0 * epsem || fabs(delta_Ur/delta_U) < 10.0 * epsem) {
		//If error is below set margin, accept solution, otherwise try PRAD
		implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t,cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is below set margin, accept solution, otherwise try PMHDwith entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t,cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is below set margin, accept solution, otherwise try PRAD staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 1);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 1);
	}
	else {
		//If error is below set margin, accept solution, otherwise try PRAD
		implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PMHD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_URAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		if (error_t > pow(10, -9.)) implicit_rad_solve_PRAD(pb, U_i, U, geom, dU, Dt, &error_t, cell_size, 1, 0);

	}

	//As final resort attempt subcycling
	if (error_t > pow(10., -7.)) {
		subcycle_rad_solve(pb_i, U_i, U, geom, Dt);
		PLOOP pb[k] = pb_i[k];
	}
}


int subcycle_rad_solve(double pb[NPR], double U[NPR], struct of_geom *geom, double Dt) {
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

		flag = Utoprim_2d(Uh, geom->gcov, geom->gcon, geom->g, ph,NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(Uh, geom->gcov, geom->gcon, geom->g, ph, ph[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(Uh, geom->gcov, geom->gcon, geom->g, ph, ph[KTOT], NEWT_TOL);
				}
			}
		}
		#endif
		Rtoprim(Uh, geom->gcov, geom->gcon, geom->g, ph, BASIC);

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

		flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U, geom->gcov, geom->gcon, geom->g, pb, pb[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pb, pb[KTOT], NEWT_TOL);
				}
			}
		}
		#endif
		Rtoprim(U, geom->gcov, geom->gcon, geom->g, pb, BASIC);

		nstep++;
	}
	if (nstep >= 1000) {
		fprintf(stderr, "Error in subcycling: too many timesteps \n");
		exit(0);
	}

	return 1;
}

int implicit_rad_solve_PMHD(double pb[NPR], double U_i[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt, double *error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb[NPR], dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset= pow(10., -8.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, fail, n_iter_jacob, flag, count_increase = 0;
	
	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = 0.;

	//Set variables to old values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_i[k];
		dU_old[k] = 0.;
	}

	if (error_t[0] > pow(10., -4.)) {
		//Set guess values for primitives after implicit step based on optical depth
		flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
				}
			}
		}	
		#endif	 
		if (flag == 0)Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);

		kappa_abs = calc_kappa_abs(pb_old);
		kappa_emmit = calc_kappa_emmit(pb_old);
		kappa_es = calc_kappa_es(pb_old);
		tau = (kappa_abs + kappa_emmit + kappa_es) * cell_size;
		if (tau > 0.66) {
			for (k = 0; k < NPR; k++) {
				pb_old[k] = pb[k];
				U_old[k] = U_i[k];
			}
		}

		//Set norm for error calculation
		source_rad(pb_old, &geom, dU_old);

		//Calculate itereated error at start of iteration
		norm = (fabs(U_i[UU]) + fabs(U_i[UU]) + fabs(Dt * dU_old[UU]));
		error_new[0] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3]) / norm);
		if (do_entropy == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU]) / norm);

		//Calculate total error at start of iteration
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			error_new[0] += 0.25 * T_GAS * (fabs(Dt * dU_old[KTOT]) / (norm));
		}
		norm = (fabs(U_i[UU_RAD]) + fabs(U_i[UU_RAD]) + fabs(Dt * dU_old[UU_RAD]));
		if (do_entropy == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3_RAD]) / norm);

		//If we've reached the tolerance level, exit immediately
		if ((fabs(error_new[0]) <= pow(10, -12.))) {
			error_t[0] = error_new[0];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_old[k];
				U[k] = U_old[k] + Dt * dU[k];
				dU[k] = dU_old[k];
			}
			return 0;
		}
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;
			fail = 0;
			for (k = UU; k <= U3; k++) dpb[k] = 0.;

			if (n_iter == 0) {
				for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
					E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
				}
			}
			else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

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
				pb_new[RHO] = (U_i[RHO]/ geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				mhd_calc(pb_new, 0, &q, &U_new[UU]); // Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL);
				#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
						}
					}
				}
				#endif

				if (flag == 0) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

					//Calculate source function and jacobian
					source_rad(pb_new, &geom, dU_new);
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdpb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb[i];
					}
					if (do_entropy == 1) {
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdpb[0][i - UU] = (E_new[0] - E_old[0]) / dpb[i];
				}
				n_iter_jacob++;
			} while (flag && n_iter_jacob<10);
		}

		if (n_iter_jacob == 10) return 1;

		if (invert_matrix(dEdpb, dEdpb_inv) == 1) {
			fprintf(stderr, "Implicit rad solve error \n");
			fail = 1;
			break;
		}

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.;
			for (k = 0; k < 4; k++) {
				dpb[k + 1] = -D*(E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
				pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
			}
		}
		else {
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if( n_iter == 8) D = 0.25;
			else D = 1.;

			if (n_iter/4 == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb[k + 1] = -(E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]); 
					pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
				}
			}
			if (n_iter/4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb[k + 1] = -(E_old[0] * dEdpb_inv[k][0]);
					pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb[k + 1] = -(E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + 1] = pb_old[k + 1] + dpb[k + 1];
				}
			}
		}

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5*fabs(pb_new[UU]);

		//Obtain new conserved quantaties for MHD variables
		get_state(pb_new, &geom, &q);
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0] ;
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
		U_new[KTOT] = geom->g*(pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives
		Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);
		source_rad(pb_new, &geom, dU_new);

		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q, &U_new[UU_RAD]);
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Calculate iterated error
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));	
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
		else offset = pow(10., -8.);
		
		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= pow(10,-12.)) || (n_iter >= 20)) {
			keep_iterating = 0;
		}
				
		//If the residual drops below machine precision, stop iterating
		if (((fabs(pb_new[UU] / pb_old[UU]) - 1.0) + (fabs(pb_new[U1] / pb_old[U1]) - 1.0) + (fabs(pb_new[U2] / pb_old[U2]) - 1.0) + (fabs(pb_new[U3] / pb_old[U3]) - 1.0)) < 10.*epsem) {
			keep_iterating = 0;
		}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5]+ error_new[(n_iter - 3) % 5]+ error_new[(n_iter - 2) % 5])<0.25*(error_new[(n_iter - 1) % 5]+ error_new[(n_iter - 0) % 5]))) {
			keep_iterating=0;
		}

		//If error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		if (keep_iterating) {
			for (k = 0; k < NPR; k++) {
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}

			for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
			if (do_entropy == 1) {
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			}
			else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);
		}

		n_iter++;
	}
	n_iter--;

	//Recalculate radiation energy density with BASIC limiter if radiation energy density turns negative. Only for PMHD and UMHD methods!
	if (pb_new[UU_RAD] < 0.0) Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);

	//Calculate total error
	if (do_entropy == 1) {
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		error_new[n_iter % 5] += 0.25 * T_GAS*(fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / norm);
	}
	norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
	if (do_entropy == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
	error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
	error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
	error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

	//If error decreased compared to start value, update variables
	if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5])<pow(10.,-4.)) {
		error_t[0] = error_new[n_iter % 5];
		for (k = 0; k < NPR; k++) {
			pb[k] = pb_new[k];
			U[k] = U_new[k];
			dU[k] = dU_new[k];
		}
	}
	return(0);
}

// This method iterates R^t_mu
// change to *geom pointer
// change dU to more intuitive name for jacobian calc
// change pow() to e
int implicit_rad_solve_URAD(double pb[NPR], double U_i[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dEdU[4][4], dEdU_inv[4][4], bsq, error_new[5], offset = pow(10., -10.);;
	double dU_offset[NPR], kappa_abs, kappa_emmit, kappa_es, tau, norm, T_GAS; // change all dU to this. then copy into dU at end
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating, flag, fail, n_iter_jacob;

	//Set guess values for primitives after implicit step based on optical depth
	kappa_abs = calc_kappa_abs(pb);
	kappa_emmit = calc_kappa_emmit(pb);
	kappa_es = calc_kappa_es(pb);
	tau = (kappa_abs + kappa_emmit + kappa_es);
	if (tau > 0.66) {
		for (k = 0; k < NPR; k++) {
			pb_old[k] = pb[k];
			pb_new[k] = pb[k];
			dU[k] = 0.;
			U_old[k] = U_i[k];
			U_new[k] = U_i[k];
			dU_offset[k] = 0.;
		}
	}
	else {
		flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
				}
			}
		}
		#endif
		Rtoprim(U, geom->gcov, geom->gcon, geom->g, pb, BASIC);
	}

	//Set norm for error calculation
	source_rad(pb_old, &geom, dU_new);
	norm = 0.25 * (fabs(U_i[UU_RAD]) + fabs(U_i[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));

	//If error allready small skip the implicit solve
	if (do_entropy == 1) error_new[0] = 0.25 * T_GAS * (fabs(Dt * dU_new[KTOT]) / (norm));
	else error_new[0] = 0.25 * (fabs(Dt * dU_new[UU_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_new[U1_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_new[U2_RAD]) / norm);
	error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_new[U3_RAD]) / norm);
	if (do_entropy == 1) error_new[0] += 0.25 * (fabs(U_i[UU_RAD]) + fabs(U_i[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));
	else error_new[0] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
	error_new[0] += 0.25 * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

	//If we've reached the tolerance level, stop iterating
	if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
		return 0;
	}

	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;

	while (keep_iterating) {
		//Calculate jacobian dEdU
		source_rad(pb_old, &geom, dU_old);
		for (k = UU_RAD; k <= U3_RAD; k++) E_old[k] = (U_old[k] - U_i[k] - Dt * dU_old[k]);

		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
			fail = 0;
			for (k = UU_RAD; k <= U3_RAD; k++) dU_i[k] = 0.;
			do {
				if (i == UU_RAD) {
					//if (do_entropy == 1) {
					//	dU_offset[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[KTOT]);
					//	U_new[i] = U_i[i] + dU_offset[i];
					//}
					//else {
						dU_offset[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]);
						U_new[i] = U_i[i] + dU_offset[i];
					//}
				}
				else {
					dU_offset[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU][i - UU]);
					U_new[i] = U_i[i] + dU_offset[i];
				}

				U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
				U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
				U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
				U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
				
				#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
						}
					}
				}
				#endif
				fail = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				source_rad(pb_new, &geom, dU_new);

				for (k = U1_RAD; k <= U3_RAD; k++) {
					E_new[k] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdU_i[k - UU][i - UU] = (E_new[k] - E_old[k]) / dU_i[i];
				}
				if (do_entropy == 1)E_new[UU] = (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
				else E_new[k] = (U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]);

				n_iter_jacob++;
			} while (fail && n_iter_jacob < 6);
		}

		if (invert_matrix(dEdU, dEdU_inv) == 1) {
			fprintf(stderr, "Error urad \n");
			break;;
		}

		/* Make the newton step: */
		if (do_staged == 0) {
			for (k = 0; k < 4; k++) {
				dU[k + 1] = -(E_old[1] * dEdU_inv[k][0] + E_old[2] * dEdU_inv[k][1] + E_old[3] * dEdU_inv[k][2] + E_old[4] * dEdU_inv[k][3]);
				U_new[k + 1] = pb_old[k + 1] + dU[k + 1];
			}
		}
		else {
			if (n_iter == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dU[k + 1] = -(E_old[2] * dEdU_inv[k][1] + E_old[3] * dEdU_inv[k][2] + E_old[4] * dEdU_inv[k][3]);
					U_new[k + 1] = pb_old[k + 1] + dU[k + 1];
				}
			}
			if (n_iter == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dU[k + 1] = -(E_old[1] * dEdU_inv[k][0]);
					U_new[k + 1] = pb_old[k + 1] + dU[k + 1];
				}
			}
			else {
				if (n_iter == 0) {
					for (k = 0; k < 4; k++) { //full 4d step
						dU[k + 1] = -(E_old[1] * dEdU_inv[k][0] + E_old[2] * dEdU_inv[k][1] + E_old[3] * dEdU_inv[k][2] + E_old[4] * dEdU_inv[k][3]);
						U_new[k + 1] = pb_old[k + 1] + dU[k + 1];
					}
				}
			}
		}
		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

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
		flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
				}
			}
		}
		#endif

		// Step 4: Recompute all U from P_i+1 for consistency
		// (in case of inversion failure or modification)
		get_state_rad(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		get_state(pb_new, &geom, &q);
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		source_rad(pb_new, &geom, dU_new);

		//Calculate iterated error
		norm = 0.25 * (fabs(U[UU_RAD]) + fabs(U[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		if (do_entropy == 1) error_new[n_iter % 5] = 0.25 * T_GAS * (fabs(U_new[KTOT] - U[KTOT] - Dt * dU_new[KTOT]) / (norm));
		else error_new[n_iter % 5] = 0.25 * (fabs(U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
		else offset = pow(10., -8.);

		//If we've reached the tolerance level, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If the residual drops below machine precision, stop iterating
		if ((fabs(U_new[UU] / U[UU]) - 1.0) + (fabs(U_new[UU_RAD] / U[UU_RAD]) - 1.0) < 10. * epsem) {
			keep_iterating = 0;
		}

		if (keep_iterating) {
			for (k = 0; k < NPR; k++) pb_old[k] = pb_new[k];
		}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.25 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			keep_iterating = 0;
		}
		else n_iter++;
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

// This method iterates E_RAD an U_rad
int implicit_rad_solve_PRAD(double pb[NPR], double U_i[NPR], double U[NPR], struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb[NPR], dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset = pow(10., -10.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter, keep_iterating, fail, n_iter_jacob, flag;

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
		flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pb_old, pb_old[KTOT], NEWT_TOL);
				}
			}
		}
		#endif		
		Rtoprim(U, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);
	}

	//Set norm for error calculation
	source_rad(pb_old, &geom, dU_new);
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

	//If we've reached the tolerance level, stop iterating
	if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
		return 0;
	}

	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;
	while (keep_iterating) {
		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
			fail = 0;
			for (k = UU_RAD; k <= U3_RAD; k++) dpb[k] = 0.;
			do {
				if (i == UU_RAD) {
					dpb[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb[i] + pb[i];
				}
				else {
					dpb[i] = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU][i - UU]);
					pb_new[i] = pb[i] + dpb[i];
				}

				get_state(pb_old, &geom, &q);
				pb_new[RHO] = U[RHO] / q.ucon[0] / geom->g; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				mhd_calc(pb_new, 0, &q, &U_new[UU]); // Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);

				U_new[UU_RAD] = U[UU_RAD] - (U_new[UU] - U[UU]);
				U_new[U1_RAD] = U[U1_RAD] - (U_new[U1] - U[U1]);
				U_new[U2_RAD] = U[U2_RAD] - (U_new[U2] - U[U2]);
				U_new[U3_RAD] = U[U3_RAD] - (U_new[U3] - U[U3]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
						}
					}
				}
				#endif
				fail = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				source_rad(pb_new, &geom, dU_new);

				for (k = U1_RAD; k <= U3_RAD; k++) {
					E_new[k] = (U_new[k] - U[k] - Dt * dU_new[k]);
					dEdpb[k - UU][i - UU] = (E_new[k] - E_old[k]) / dU[i];
				}
				if (do_entropy == 1)E_new[UU] = (U_new[KTOT] - U[KTOT] - Dt * dU_new[KTOT]);
				else E_new[k] = (U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]);

				n_iter_jacob++;
			} while (fail && n_iter_jacob < 6);
		}

		if (invert_matrix(dEdpb, dEdpb_inv) == 1) {
			fprintf(stderr, "Error prad \n");
			break;;
		}

		/* Make the newton step: */
		if (do_staged == 0) {
			for (k = 0; k < 4; k++) {
				dpb[k + UU_RAD] = -(E_old[1] * dEdpb_inv[k][0] + E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
				pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb[k + UU_RAD];
			}
		}
		else {
			if (n_iter == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb[k + UU_RAD] = -(E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb[k + UU_RAD];
				}
			}
			if (n_iter == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb[k + UU_RAD] = -(E_old[1] * dEdpb_inv[k][0]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb[k + UU_RAD];
				}
			}
			else {
				if (n_iter == 0) {
					for (k = 0; k < 4; k++) { //full 4d step
						dpb[k + UU_RAD] = -(E_old[1] * dEdpb_inv[k][0] + E_old[2] * dEdpb_inv[k][1] + E_old[3] * dEdpb_inv[k][2] + E_old[4] * dEdpb_inv[k][3]);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb[k + UU_RAD];
					}
				}
			}
		}

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
		flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, pb_new[KTOT], NEWT_TOL);
				}
			}
		}	
		#endif

		// Step 5: Recompute T^t_mu for consistency
		get_state_rad(pb_new, &geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		source_rad(pb_new, &geom, dU_new);

		//Calculate iterated error
		norm = 0.25 * (fabs(U[UU_RAD]) + fabs(U[UU]) + fabs(U_new[UU_RAD]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU_RAD]) + fabs(Dt * dU_new[UU]));
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		if (do_entropy == 1) error_new[n_iter % 5] = 0.25 * T_GAS * (fabs(U_new[KTOT] - U[KTOT] - Dt * dU_new[KTOT]) / (norm));
		else error_new[n_iter % 5] = 0.25 * (fabs(U_new[UU_RAD] - U[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
		else offset = pow(10., -8.);

		//If we've reached the tolerance level, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If the residual drops below machine precision, stop iterating
		if ((fabs(U_new[UU] / U[UU]) - 1.0) + (fabs(U_new[UU_RAD] / U[UU_RAD]) - 1.0) < 10. * epsem) {
			keep_iterating = 0;
		}

		if (keep_iterating) {
			for (k = 0; k < NPR; k++) pb_old[k] = pb_new[k];
		}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.25 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			keep_iterating = 0;
		}
		else n_iter++;
	}

	//Recalculate radiation energy density with BASIC limiter if radiation energy density turns negatie
	if (pb_new[UU_RAD] < 0.0)Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);

	//Calculate total error
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

	double Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq=0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f, ymax;
	int i;

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise_g(Qcov, gcon, Qcon);
	
	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	
	Qdotn = Qcon[0] * ncov[0]; //-Erad in McKinney2013
	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn * Qdotn; //Utilde^2 in McKinney2013

	y = Qtsq / (Qdotn * Qdotn); //Definition from McKinney2013. Should only range [0,1].
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y); 

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++)prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);
	
	if (-Qdotn <= 0.) {
		prim[0] = pow(10., -300.);
		for (i = 1; i < 4; i++)prim[i] = 0.0;
	}
	if (y<0 && y>-epsem) {
		for (i = 1; i < 4; i++)prim[i] = 0.0;
	}

	if (y >= y_max) {
		if (lim == BASIC) {
			if (gammasq > GAMMAMAX * GAMMAMAX) {
				f = sqrt((GAMMAMAX * GAMMAMAX - 1.) / (gammasq - 1.));
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			}
		}
		else if (lim == TYPE2) {
			Uabs = 0.5 * (sqrt(Qtsq) + fabs(Qdotn) + pow(10., -150.));
			for (i = 1; i < 4; i++)prim[i] = Qtcon[i] / Uabs;
			
			qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3] + 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
			if (qsq < 0.) qsq = 1.E-10; // set floor
			gammasq = sqrt(1. + qsq);

			if (gammasq > GAMMAMAX * GAMMAMAX) {
				f = sqrt((GAMMAMAX * GAMMAMAX - 1.) / (gammasq - 1.));
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			}
			
			Qdotn = -(pow(10., -150.) + sqrt(Qtsq / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX * GAMMAMAX - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad
		}
	}

	return(0);
}

void calc_ymax(void) {
	
	int keep_iterating, n_iter;
	double E, errx, dEdy, y_new, y_old;
	keep_iterating = 1;
	n_iter = 0;
	y_old = 0.98; //Gives gamma=25

	while (keep_iterating) {
		//Calculate deviation from 0
		E = GAMMAMAX * GAMMAMAX - (2.0 - y_old+sqrt(4.0-3.0*y_old)) / (4.0-4.0*y_old);
		
		//Calculate gradient dEdy
		dEdy = (0.375*y_old-0.25*sqrt(4.0-3.0*y_old-0.625)/(sqrt(4.0-3.0*y_old)*(1.0-y_old)*(1.0-y_old)));

		/* Make the newton step: */
		y_new = y_old - (E) / dEdy;

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		errx = fabs(y_new - y_old) / (fabs(y_new)+fabs(y_old));

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER))) {
			keep_iterating = 0;
		}

		y_old = y_new;

		n_iter++;
	}   // END of while(keep_iterating)
	y_max = y_new;
}