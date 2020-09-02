#include "decs_MPI.h"

#define epsem (2.22E-16)
void raise_g(double vcov[], double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); 
int Rtoprim_calc(double U[NPR_R], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);

void implicit_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double cell_size) {
	double error_t=pow(10., -3.99999);
	int flag=0, k;
	double delta_U, delta_Ur, U_ft[NPR], pb_i[NPR];

	//Initialize variables
	PLOOP{
		dU[k] = 0.;
		U_ft[k] = U_i[k];
		pb_i[k] = pb[k];
	}
	delta_U = (U_n[UU] - U_i[UU] + pow(10.,-150.)) / (fabs(U_i[UU]) + fabs(U_n[UU]));
	delta_Ur = (U_n[UU_RAD] - U_i[UU_RAD] + pow(10., -150.)) / (fabs(U_i[UU_RAD]) + fabs(U_n[UU_RAD]));

	//Check if fluid is in extreme radiation subdominant regime
	if (((U_n[UU_RAD] / U_n[UU]) < 10.0 * epsem) || ((pb[UU_RAD] / pb[UU]) < 10.0 * epsem) || (fabs(delta_Ur) < 10.0 * fabs(delta_U) * epsem)) {
		//If error is below set margin, accept solution, otherwise try PRAD
		//flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft, pflag, pflag_rad,geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD
		if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t,cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is below set margin, accept solution, otherwise try PMHDwith entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t,cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is below set margin, accept solution, otherwise try PRAD staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 1);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 1);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 1);
	}
	else {
		//If error is below set margin, accept solution, otherwise try PRAD
		flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is still below set margin, accept solution, otherwise try URAD
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 0);

		//If error is below set margin, accept solution, otherwise try PRAD with entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PMHD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_URAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 0);

		//If error is still below set margin, accept solution, otherwise try URAD with entropy
		//if (error_t > pow(10, -9.) || flag) flag = implicit_rad_solve_PRAD(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 1, 0);
	}

	//As final resort attempt subcycling
	if (error_t > pow(10., -7.) || flag) {
		PLOOP{
			//U_ft[k] = U_i[k];
		  //  dU[k] = 0.;
		//	pb[k] = pb_i[k];
		}
		//if (subcycle_rad_solve(pb, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, cell_size) != 0) return 1;
		//else {
		//	PLOOP U_f[k] = U_ft[k];
		//}
	}
	//else {
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i[k]) / Dt;
		}
	//}
}

int implicit_rad_solve_PMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double *error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset= pow(10., -8.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, fail, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;
	
	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = dU[k];
	}

	if (error_t[0] > pow(10., -4.)) {
		//Set guess values for primitives after implicit step based on optical depth
		pflag[0] = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
		#if(DO_FONT_FIX)
		if (pflag[0]) {
			#if DOKTOT
			pflag[0] = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
			#endif
			if (pflag[0]) {
				pflag[0] = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
			}
		}	
		#endif	 
		pflag_rad[0] = Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);

		//Calculate source term for U_i
		source_rad(pb_old, geom, dU_old);

		//Calculate iterated error at start of iteration
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
		if (do_entropy == 0 && pflag_rad[0] == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3_RAD]) / norm);

		//If we've reached the tolerance level, exit immediately
		if (((fabs(error_new[0]) < pow(10, -4.)) && (fabs(error_new[0]) < error_t[0]))) {
			error_t[0] = error_new[0];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_old[k];
				U_f[k] = U_old[k] + Dt * dU_old[k];
				dU[k] = dU_old[k];
			}
			if ((fabs(error_new[0]) < pow(10, -12.))) {
				keep_iterating = 0;
				return 0;
			}
		}
		else { //Determine the estimate for the implicit solver
			kappa_abs = calc_kappa_abs(pb_old);
			//kappa_emmit = calc_kappa_emmit(pb_old);
			kappa_es = calc_kappa_es(pb_old);
			tau = (kappa_abs + kappa_es) * cell_size;
			if (tau > 0.66) {
				for (k = 0; k < NPR; k++) {
					//pb_old[k] = pb[k];
					//U_old[k] = U_i[k];
				}
			}
		}
	}

	double pb_min[NPR];
	PLOOP pb_min[k] = pb_old[k];
	double error_start;
	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		if (n_iter == 0) error_start= error_new[0];
		PLOOP U_new[k] = U_old[k];
		//if (n_iter == 0)fprintf(stderr, "error_start: %f, n_iter: %d \n", log10(error_new[0]), n_iter);

		//Calculate reference error at n_iter==0
		if (n_iter == 0) {
			for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
			if (do_entropy == 1) {
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			}
			else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);
		}

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;
			fail = 0;

			do {
				PLOOP pb_new[k] = pb_old[k];
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU] + pb_old[RHO]);
					pb_new[i] = pb_old[i] + dpb;
				}
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU][i - UU]);
					pb_new[i] = pb_old[i] + dpb;
				}

				get_state(pb_new, geom, &q);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				mhd_calc(pb_new, 0, &q, &U_new[UU]); // Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] = U_new[UU] + U_new[RHO];
				if (do_entropy == 1) {
					#if(FULL_ENTROPY)
					U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA)));
					#else
					U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
					#endif
				}

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				//Calculate source function and jacobian
				source_rad(pb_new, geom, dU_new);
				for (k = U1; k <= U3; k++) {
					E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdpb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb;
				}
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb[0][i - UU] = (E_new[0] - E_old[0]) / dpb;

				//Invert Jacobian
				flag = invert_matrix(dEdpb, dEdpb_inv);
				
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.3));

			if (flag) {
				//fprintf(stderr, "Implicit rad Jacobian error \n");
				return 1;
			}
		}

		//Set primitive variables before Newton step
		PLOOP pb_new[k] = pb_old[k];

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.0;
			for (k = 0; k < 4; k++) {
				dpb = -D*(E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
				if (k == 0) dpb = MY_MIN(0.5*pb_old[UU], fabs(dpb))*fabs(dpb)/dpb;
				else dpb = MY_MIN(0.5 * pb_old[k+UU], dpb);
				pb_new[k + UU] = pb_old[k + UU] + dpb;
			}
		}
		else {
			//Set damping factor for Newton Raphson method
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if(n_iter == 8) D = 0.25;
			else D = 1.;

			if (n_iter/4 == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			if (n_iter/4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
		}

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5*fabs(pb_new[UU]);

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0] ;
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
		U_new[UU] = U_new[UU] + U_new[RHO];

		#if(FULL_ENTROPY)
		U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
		#else
		U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
		#endif

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives using TYPE2 limiter
		flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

		//Get radiative source term
		source_rad(pb_new, geom, dU_new);

		//Calculate iterated error
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < pow(10., -9.)) offset = pow(10., -10.);
		//else offset = pow(10., -8.);
		
		//Calculate total error
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
			error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / norm);
		}
		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= pow(10,-12.)) || (n_iter >= 20)) {
			keep_iterating = 0;
			//if (n_iter >= 20)fprintf(stderr, "error: %f %f %f %f %f, n_iter: %d \n", log10(error_start), log10(error_new[0]), log10(error_new[1]), log10(error_new[2]), log10(error_new[3]), n_iter);
			//if (n_iter >= 20)fprintf(stderr, "pb_ratio: %f %f %f %f \n", log10(fabs(pb_min[UU])), log10(fabs( pb_new[UU])), log10(fabs(pb_min[U2] / pb_new[U2])), log10(fabs(pb_min[U2] / pb_new[U2])));
		}

		//If the residual drops below machine precision, stop iterating
		//if (((fabs(pb_new[UU] / pb_old[UU]) - 1.0) + (fabs(pb_new[U1] / pb_old[U1]) - 1.0) + (fabs(pb_new[U2] / pb_old[U2]) - 1.0) + (fabs(pb_new[U3] / pb_old[U3]) - 1.0)) < 10.*epsem) {
			//keep_iterating = 0;
		//}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5]+ error_new[(n_iter - 3) % 5]+ error_new[(n_iter - 2) % 5])<1.0*(error_new[(n_iter - 1) % 5]+ error_new[(n_iter - 0) % 5]))) {
			keep_iterating=0;
			//fprintf(stderr, "error: %f %f %f %f %f, n_iter: %d \n", log10(error_start), log10(error_new[0]), log10(error_new[1]), log10(error_new[2]), log10(error_new[3]), n_iter);
			//fprintf(stderr, "pb_ratio: %f %f %f %f \n", log10(fabs(pb_min[UU])), log10(fabs(pb_new[UU])), log10(fabs(pb_min[U2] / pb_new[U2])), log10(fabs(pb_min[U2] / pb_new[U2])));
		}

		//If error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
			//fprintf(stderr, "error: %f %f %f %f, n_iter: %d \n", log10(error_start), log10(error_new[0]), log10(error_new[2]), log10(error_new[3]), n_iter);
			//fprintf(stderr, "pb_ratio: %f %f %f %f \n", log10(fabs(pb_min[UU])), log10(fabs(pb_new[UU])), log10(fabs(pb_min[U2] / pb_new[U2])), log10(fabs(pb_min[U2] / pb_new[U2])));
		}

		if (keep_iterating) {
			//Recompute R_t^mu for consistency
			get_state_rad(pb_new, geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

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
		else {
			//Recalculate radiation energy density with BASIC limiter if radiation energy density turns negative. Only for PMHD and UMHD methods!
			if(flag_rad) flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);

			//Recompute R_t^mu for consistency
			get_state_rad(pb_new, geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;
		}

		//If error decreased compared to start value, update variables
		if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5]) < pow(10., -4.)) {
			error_t[0] = error_new[n_iter % 5];
			for (k = 0; k < NPR; k++) {
				pb_min[k] = pb_new[k];
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
		}

		n_iter++;
	}
	return(0);
}

// This method iterates R^t_mu
int implicit_rad_solve_URAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4][4], dEdUb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, fail, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = dU[k];
	}

	if (error_t[0] > pow(10., -4.)) {
		//Set guess values for primitives after implicit step based on optical depth
		flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
				}
			}
		}
		#endif	 
		flag_rad = Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);

		//Calculate source term for U_i
		source_rad(pb_old, geom, dU_old);

		//Calculate itereated error at start of iteration
		norm = (fabs(U_i[UU_RAD]) + fabs(U_i[UU_RAD]) + fabs(Dt * dU_old[UU_RAD]));
		error_new[0] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3_RAD]) / norm);
		if (do_entropy == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU_RAD]) / norm);

		//Calculate total error at start of iteration
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			error_new[0] += 0.25 * T_GAS * (fabs(Dt * dU_old[KTOT]) / (norm));
		}
		norm = (fabs(U_i[UU]) + fabs(U_i[UU]) + fabs(Dt * dU_old[UU]));
		if (do_entropy == 0 && flag_rad == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3]) / norm);

		//If we've reached the tolerance level, exit immediately
		if (((fabs(error_new[0]) < pow(10, -4.)) && (fabs(error_new[0]) < error_t[0]))) {
			error_t[0] = error_new[0];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_old[k];
				U_f[k] = U_old[k] + Dt * dU_old[k];
				dU[k] = dU_old[k];
			}
			if ((fabs(error_new[0]) < pow(10, -12.))) {
				keep_iterating = 0;
				return 0;
			}
		}
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//if (n_iter == 0)fprintf(stderr, "error_start: %f, n_iter: %d \n", log10(error_new[0]), n_iter);

		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
			fail = 0;

			if (n_iter == 0) {
				for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_old[UU_RAD] / pb_old[RHO];
					E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
				}
				else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);
			}

			do {
				PLOOP U_new[k] = U_old[k];
				if (i == UU_RAD) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]);
					U_new[i] = U_old[i] + dUb;
				}
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]) * sqrt(geom->gcov[i - UU_RAD][i - UU_RAD]);
					U_new[i] = U_old[i] + dUb;
				}

				U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
				U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
				U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
				U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
				U_new[KTOT] = U_old[KTOT];

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				/*#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
						}
					}
				}
				#endif*/	 

				if (flag == 0) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

					//Calculate source term using new variables
					source_rad(pb_new, geom, dU_new);

					//Calculate source function and jacobian
					for (k = U1_RAD; k <= U3_RAD; k++) {
						E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dUb;
					}
					if (do_entropy == 1) {
						get_state(pb_new, geom, &q);
						#if(FULL_ENTROPY)
						U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
						#else
						U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);
						#endif
						T_GAS = (GAMMA - 1.) * pb_new[UU_RAD] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
					dEdUb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dUb;

					flag = invert_matrix(dEdUb, dEdUb_inv);
				}
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.3));

			if (flag) {
				//fprintf(stderr, "Implicit rad Jacobian error \n");
				return 1;
			}
		}

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.;
			for (k = 0; k < 4; k++) {
				dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
				U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
			}
		}
		else {
			//Set damping factor for Newton Raphson method
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if (n_iter == 8) D = 0.25;
			else D = 1.;

			if ((n_iter / 4) == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
					U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
				}
			}
			if ((n_iter / 4) == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dUb = -D * (E_old[0] * dEdUb_inv[k][0]);
					U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
					U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
				}
			}
		}

		//Derive new conserved quantaties for MHD variables
		U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
		U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
		U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
		U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

		//Estimate conserved entropy
		source_rad(pb_old, geom, dU_old);
		U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

		//Reset variables in case inversion fails
		PLOOP{
			pb_new[k] = pb_old[k];
			dU_new[k] = dU_old[k];
		}

		flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
		/*#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				}
			}
		}
		#endif */

		if (flag == 0) {
			//Get new radiation primitives using TYPE2 limiter
			flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

			//Make sure that internal energy stays positive
			if (pb_new[UU_RAD] < 0.0) pb_new[UU_RAD] = 0.5 * fabs(pb_new[UU_RAD]);

			//Recompute T_t^mu for consistency
			U_new[RHO] = U_i[RHO];
			get_state(pb_new, geom, &q);
			mhd_calc(pb_new, 0, &q, &U_new[UU]);
			for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
			U_new[UU] += U_new[RHO];

			#if(FULL_ENTROPY)
			U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
			#else
			U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA);
			#endif

			//Recompute R_t^mu for consistency
			get_state_rad(pb_new, geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

			//Get radiative source term
			source_rad(pb_new, geom, dU_new);

			//Calculate iterated error
			norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
			error_new[n_iter % 5] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
			if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);

			//Set correct offset for Jacobian for next iteration
			if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
			else offset = pow(10., -8.);

			//Calculate total error
			if (do_entropy == 1) {
				T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
				error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / norm);
			}
			norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
			if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
	
			//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
			if ((fabs(error_new[n_iter % 5]) <= pow(10, -9.)) || (n_iter >= 20)) {
				keep_iterating = 0;
			}

			//If the residual drops below machine precision, stop iterating
			//if (((fabs(U_new[UU_RAD] / U_old[UU_RAD]) - 1.0) + (fabs(U_new[U1_RAD] / U_old[U1_RAD]) - 1.0) + (fabs(U_new[U2_RAD] / U_old[U2_RAD]) - 1.0) + (fabs(U_new[U3_RAD] / U_old[U3_RAD]) - 1.0)) < 10. * epsem) {
				//keep_iterating = 0;
			//}

			//fprintf(stderr, "error: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

			//If error increasing stop iterating
			if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.5 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
				keep_iterating = 0;
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

				for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_old[UU_RAD] / pb_old[RHO];
					E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
				}
				else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);
			}

			//If error decreased compared to start value, update variables
			if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5]) < pow(10., -4.)) {
				error_t[0] = error_new[n_iter % 5];
				for (k = 0; k < NPR; k++) {
					pb[k] = pb_new[k];
					U_f[k] = U_new[k];
					dU[k] = dU_new[k];
				}
			}
		}
		else {
			keep_iterating = 0;
		}
		n_iter++;
	}

	return(0);
}

// This method iterates E_RAD an U_rad
int implicit_rad_solve_PRAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, tau, kappa_abs, kappa_emmit, kappa_es, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, fail, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = (U_f[k]-U_i[k])/Dt;
	}

	if (error_t[0] > pow(10., -4.)) {
		//Set guess values for primitives after implicit step based on optical depth
		flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
		#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, BASIC);
				}
			}
		}
		#endif	 
		if (flag == 0) flag_rad = Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb_old, BASIC);

		//Calculate source term for U_i
		source_rad(pb_old, geom, dU_old);

		//Calculate itereated error at start of iteration
		norm = (fabs(U_i[UU_RAD]) + fabs(U_i[UU_RAD]) + fabs(Dt * dU_old[UU_RAD]));
		error_new[0] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2_RAD]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3_RAD]) / norm);
		if (do_entropy == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU_RAD]) / norm);

		//Calculate total error at start of iteration
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			error_new[0] += 0.25 * T_GAS * (fabs(Dt * dU_old[KTOT]) / (norm));
		}
		norm = (fabs(U_i[UU]) + fabs(U_i[UU]) + fabs(Dt * dU_old[UU]));
		if (do_entropy == 0 && flag_rad == 0) error_new[0] += 0.25 * (fabs(Dt * dU_old[UU]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(Dt * dU_old[U1]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(Dt * dU_old[U2]) / norm);
		error_new[0] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(Dt * dU_old[U3]) / norm);

		//If we've reached the tolerance level, exit immediately
		if (((fabs(error_new[0]) < pow(10, -4.)) && (fabs(error_new[0]) < error_t[0]))) {
			error_t[0] = error_new[0];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_old[k];
				U_f[k] = U_old[k] + Dt * dU_old[k];
				dU[k] = dU_old[k];
			}
			if ((fabs(error_new[0]) < pow(10, -12.))) {
				keep_iterating = 0;
				return 0;
			}
		}
		else { //Determine the estimate for the implicit solver
			kappa_abs = calc_kappa_abs(pb_old);
			kappa_emmit = calc_kappa_emmit(pb_old);
			kappa_es = calc_kappa_es(pb_old);
			tau = (kappa_abs + kappa_es) * cell_size;
			if (tau > 0.66) {
				for (k = 0; k < NPR; k++) {
					//pb_old[k] = pb[k];
					//U_old[k] = U_i[k];
				}
			}
		}
	}

	double error_temp = error_new[0];
	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Set error at start
		if (n_iter == 0) {
			for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
			if (do_entropy == 1) {
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			}
			else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);
		}

		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
			fail = 0;

			do {
				PLOOP{
					pb_new[k] = pb_old[k];
					U_new[k] = U_old[k];
				}
				if (i == UU_RAD) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU_RAD]);
					pb_new[i] = pb_old[i] + dpb;
				}
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU_RAD][i - UU_RAD]);
					pb_new[i] = pb_old[i] + dpb;
				}

				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++) U_new[k] *= geom->g;
				get_state(pb_new, geom, &q);
				
				U_new[RHO] = U_i[RHO];
				U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
				U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
				U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
				U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
				#if(FULL_ENTROPY)
				U_new[KTOT] = geom->g * pb_old[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_old[UU] * pow(pb_old[RHO], -GAMMA));
				#else
				U_new[KTOT] = geom->g * pb_old[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_old[UU] * pow(pb_old[RHO], -GAMMA);
				#endif

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				/*#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
						}
					}
				}
				#endif*/

				if (flag == 0) {
					//Calculate source function and jacobian
					source_rad(pb_new, geom, dU_new);
					for (k = U1_RAD; k <= U3_RAD; k++) {
						E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdpb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dpb;
					}
					if (do_entropy == 1) {
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
					dEdpb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dpb;
		
					//Try to find Jacobian
					flag = invert_matrix(dEdpb, dEdpb_inv);
				}
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.3));

			if (flag) {
				//fprintf(stderr, "Implicit rad Jacobian error \n");
				return 1;
			}
		}

		//Set primitive variables before Newton step
		PLOOP{ 
			pb_new[k] = pb_old[k];
			U_new[k] = U_old[k];
		}

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.;
			for (k = 0; k < 4; k++) {
				dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
				pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
			}
		}
		else {
			//Set damping factor for Newton Raphson method
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if (n_iter == 8) D = 0.25;
			else D = 1.;

			if (n_iter / 4 == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
				}
			}
			if (n_iter / 4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
				}
			}
		}

		//Make sure that radiation internal energy stays positive
		if (pb_new[UU_RAD] < 0.0) pb_new[UU_RAD] = 0.5 * fabs(pb_new[UU_RAD]);

		//Obtain new radiation conserved quantaties from radiation primitive variables
		get_state_rad(pb_new, geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Derive new MHD conserved quantaties for radiation variables
		U_new[RHO] = U_i[RHO];
		U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
		U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
		U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
		U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

		//Estimate conserved entropy
		source_rad(pb_old, geom, dU_old);
		U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

		//Get MHD primitives
		flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
		/*#if(DO_FONT_FIX)
		if (flag) {
			#if DOKTOT
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			#endif
			if (flag) {
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				}
			}
		}
		#endif	*/ 
		
		if (flag == 0) {
			//Recompute T_t^mu for consistency
			get_state(pb_new, geom, &q);
			mhd_calc(pb_new, 0, &q, &U_new[UU]);
			for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
			U_new[UU] += U_i[RHO];

			//Compute new entropy from MHD variables
			#if(FULL_ENTROPY)
			U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA)));
			#else
			U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
			#endif

			//Get radiative source term
			source_rad(pb_new, geom, dU_new);

			//Calculate iterated error
			norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
			error_new[n_iter % 5] = 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
			if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);

			//Set correct offset for Jacobian for next iteration
			if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
			else offset = pow(10., -8.);

			//Calculate total error
			if (do_entropy == 1) {
				T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
				error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / norm);
			}
			norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
			if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
			error_new[n_iter % 5] += 0.25 * sqrt(geom->gcov[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

			//If iterated error increasing stop iterating
			if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.25 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
				keep_iterating = 0;
				//fprintf(stderr, "n_iter: %d n_iter_jacob: %d, error0: %f, error1: %f, error2: %f, error3: %f, \n", n_iter, n_iter_jacob, log10(error_temp), log10(error_new[(n_iter - 2) % 5]), log10(error_new[(n_iter - 1) % 5]), log10(error_new[n_iter % 5]));
			}

			//If error increased more than 4 times stop iterating
			if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
				count_increase++;
				if (count_increase >= 5) keep_iterating = 0;
				//fprintf(stderr, "n_iter: %d n_iter_jacob: %d, error0: %f, error1: %f, error2: %f, error3: %f, \n", n_iter, n_iter_jacob, log10(error_temp), log10(error_new[(n_iter - 2) % 5]), log10(error_new[(n_iter - 1) % 5]), log10(error_new[n_iter % 5]));
			}

			//If we've reached the tolerance level in total error or we exceeded more than 20 iterations, stop iterating
			if ((fabs(error_new[n_iter % 5]) <= pow(10, -9.)) || (n_iter >= 20)) {
				keep_iterating = 0;
				//fprintf(stderr, "n_iter: %d n_iter_jacob: %d, error0: %f, error1: %f, error2: %f, error3: %f, \n", n_iter, n_iter_jacob, log10(error_temp), log10(error_new[(n_iter - 2) % 5]), log10(error_new[(n_iter - 1) % 5]), log10(error_new[n_iter % 5]));
			}

			//If the residual drops below machine precision, stop iterating
			//if (((fabs(pb_new[UU_RAD] / pb_old[UU_RAD]) - 1.0) + (fabs(pb_new[U1_RAD] / pb_old[U1_RAD]) - 1.0) + (fabs(pb_new[U2_RAD] / pb_old[U2_RAD]) - 1.0) + (fabs(pb_new[U3_RAD] / pb_old[U3_RAD]) - 1.0)) < 10. * epsem) {
				//keep_iterating = 0;

			//}
	
			if (keep_iterating) {
				for (k = 0; k < NPR; k++) {
					U_old[k] = U_new[k];
					pb_old[k] = pb_new[k];
					dU_old[k] = dU_new[k];
				}

				for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
					E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
				}
				else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);
			}

			//If error decreased compared to start value, update variables
			if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5]) < pow(10., -4.)) {
				error_t[0] = error_new[n_iter % 5];
				for (k = 0; k < NPR; k++) {
					pb[k] = pb_new[k];
					U_f[k] = U_new[k];
					dU[k] = dU_new[k];
				}
			}
		}
		else {
			//In case of inversion failure exit iteration
			keep_iterating = 0;
			//fprintf(stderr, "n_iter: %d n_iter_jacob: %d, error0: %f, error1: %f, error2: %f, error3: %f, \n", n_iter, n_iter_jacob, log10(error_temp), log10(error_new[(n_iter - 2) % 5]), log10(error_new[(n_iter - 1) % 5]), log10(error_new[n_iter % 5]));		
		}
		n_iter++;
	}

	return(0);
}

int subcycle_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double cell_size) {
	double factor, remainder = 1.0,  Uh[NPR], U_new[NPR], ph[NPR], pb_new[NPR], pb_old[NPR], fraction;
	double kappa_abs, kappa_emmit, kappa_es, tau;
	int flag = 0, keep_iterating = 1, nstep = 0, k;
	struct of_state q;
	struct of_state_rad q_rad;

	for (k = 0; k < NPR; k++) pb_old[k] = pb[k];

	//Get primitive variables belonging to U_i
	flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, lim);
	#if(DO_FONT_FIX)
	if (flag) {
		#if DOKTOT
		flag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, lim);
		#endif
		if (flag) {
			if (flag) {
				flag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, lim);
			}
		}
	}
	#endif

	if (!flag) {
		Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb, BASIC);

		//Find optical depth for one cell
		kappa_abs = calc_kappa_abs(pb);
		kappa_emmit = calc_kappa_emmit(pb);
		kappa_es = calc_kappa_es(pb);
		tau = (kappa_abs + kappa_emmit + kappa_es) * cell_size;

		//Something like Courant number in implicity scheme; e.g. we do not let the source term in each step be bigger than this
		fraction = 0.25 * MY_MIN(1.0 / tau, 1.0);

		//Set temporary variables to their initial values
		for (k = 0; k < NPR; k++) {
			ph[k] = pb_old[k];
			pb_new[k] = pb_old[k];
			Uh[k] = U_i[k];
			U_new[k] = U_i[k];
		}

		while (keep_iterating && nstep < 1000) {
			//Calculate source term for half-step
			source_rad(pb_new, geom, dU);

			//Set size of subcycling timestep
			if (fraction * MY_MIN(U_i[UU], U_i[UU_RAD]) >= fabs(dU[UU_RAD] * Dt)) keep_iterating = 0;
			factor = MY_MIN(fraction * MY_MIN(U_i[UU], U_i[UU_RAD]) / fabs(dU[UU_RAD] * Dt), remainder);
			remainder -= factor;

			Uh[UU_RAD] = U_new[UU_RAD] + 0.5 * factor * Dt * dU[UU_RAD];
			Uh[U1_RAD] = U_new[U1_RAD] + 0.5 * factor * Dt * dU[U1_RAD];
			Uh[U2_RAD] = U_new[U2_RAD] + 0.5 * factor * Dt * dU[U2_RAD];
			Uh[U3_RAD] = U_new[U3_RAD] + 0.5 * factor * Dt * dU[U3_RAD];
			Uh[UU] = U_new[UU] + 0.5 * factor * Dt * dU[UU];
			Uh[U1] = U_new[U1] + 0.5 * factor * Dt * dU[U1];
			Uh[U2] = U_new[U2] + 0.5 * factor * Dt * dU[U2];
			Uh[U3] = U_new[U3] + 0.5 * factor * Dt * dU[U3];

			flag = Utoprim_2d(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, lim);
			#if(DO_FONT_FIX)
			if (flag) {
				#if DOKTOT
				flag = Utoprim_1dvsq2fix1(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, lim);
				#endif
				if (flag) {
					if (flag) {
						flag = Utoprim_1dfix1(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, lim);
					}
				}
			}
			#endif
			Rtoprim(Uh, geom->gcov, geom->gcon, geom->g, ph, BASIC);

			if (!flag) {
				//Calculate source term for full step
				source_rad(ph, geom, dU);
				U_new[UU_RAD] += factor * Dt * dU[UU_RAD];
				U_new[U1_RAD] += factor * Dt * dU[U1_RAD];
				U_new[U2_RAD] += factor * Dt * dU[U2_RAD];
				U_new[U3_RAD] += factor * Dt * dU[U3_RAD];
				U_new[UU] += factor * Dt * dU[UU];
				U_new[U1] += factor * Dt * dU[U1];
				U_new[U2] += factor * Dt * dU[U2];
				U_new[U3] += factor * Dt * dU[U3];

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, lim);
				#if(DO_FONT_FIX)
				if (flag) {
					#if DOKTOT
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, lim);
					#endif
					if (flag) {
						if (flag) {
							flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, lim);
						}
					}
				}
				#endif
				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);
			}

			//In case of inversion failure reset and start over again with smaller step
			if (flag) {
				for (k = 0; k < NPR; k++) {
					ph[k] = pb_old[k];
					pb_new[k] = pb_old[k];
					Uh[k] = U_i[k];
					U_new[k] = U_i[k];
				}
				fraction /= 2.;
				remainder = 1.;
			}
			else {
				//Recalculate conserved quantities for consistency
				get_state(pb_new, geom, &q);
				get_state_rad(pb_new, geom, &q_rad);
				primtoflux(pb_new, &q, &q_rad, 0, geom, U_new);
			}

			nstep++;
		}
	}

	if (nstep >= 1000) {
		fprintf(stderr, "Error in subcycling: too many timesteps! \n");
		return 1;
	}
	else if (remainder >= pow(10., -9.) || flag) {
		fprintf(stderr, "Error in subcycling: Remainder not 0 or failed MHD inversion! \n");
		return 2;
	}
	else {
		for (k = 0; k < NPR; k++) {
			pb[k] = pb_new[k];
			U_f[k] = U_new[k];
			dU[k] = dU[k];
		}
	}

	return 0;
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

	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

	/* Transform new primitive variables back if there was no problem : */
	for (i = 0; i <= U3_RAD - UU_RAD; i++) {
		prim[i + NPR_U] = prim_tmp[i];
	}

	return(ret);
}

// Limits radiation with either BASIC or TYPE2 approaches
int Rtoprim_calc(double U[NPR_R], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim)
{
	double Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq=0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, gammasq2, y, pressure, f, ymax;
	int i, returnval=0;

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise_g(Qcov, gcon, Qcon);
	
	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	
	Qdotn = Qcon[0] * ncov[0]; //-Erad in McKinney2013
	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn * Qdotn; //Utilde^2 in McKinney2013

	y = Qtsq / (Qdotn*Qdotn+pow(10.,-50)); //Definition from McKinney2013. Should only range [0,1].
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y); 

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	if ((-Qdotn) <= 0. || isnan(-Qdotn)) {
		prim[0] = pow(10., -36.);
		//if (y > 1. || isnan(y)) returnval = 1;
		y = 0.;
	}
	if (y<=0.){
		for (i = 1; i < 4; i++) prim[i] = 0.0;
		y = 0.;
	}
	if (y>=1.0 || isnan(y)) {
		prim[0] = pow(10., -36.);
		prim[1] = 0.;
		prim[2] = 0.;
		prim[3] = 0.;
		gammasq = 1.0;

		// Get Ebar and p_rad as usual
		pressure = -Qdotn / (4. * gammasq - 1.);
		prim[0] = pressure * 3.; // Erad = 3*p_rad

		// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
		for (i = 1; i < 4; i++) {
			if (!isnan(Qtcon[i])) {
				prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);
			}
		}
		y = 0.;
		returnval = 0;
	}
	if (y > y_max ) {
		Uabs = 0.5 * (sqrt(Qtsq) + fabs(Qdotn) + pow(10., -150.));
		for (i = 1; i < 4; i++)prim[i] = Qtcon[i] / Uabs;
		qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3] + 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
		if (qsq < 0. && fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;

		if (lim == TYPE2) {
			Qdotn = -(pow(10., -150.) + sqrt(Qtsq / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad		
			returnval = 0;
		}		
	}
	return(returnval);
}

void calc_ymax(void) {
	int keep_iterating, n_iter;
	double E_old, E_new, errx, dEdy, y_new, y_old;
	keep_iterating = 0;
	n_iter = 0;
	y_old = 0.999998; //Gives gamma=25

	//Calculate deviation from 0
	E_old = GAMMAMAX_RAD * GAMMAMAX_RAD - (2.0 - y_old + sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old);

	while (keep_iterating) {	
		//Calculate gradient dEdy
		dEdy = (0.375 * y_old - 0.25 * sqrt(4.0 - 3.0 * y_old - 0.625) / (sqrt(4.0 - 3.0 * y_old) * (1.0 - y_old) * (1.0 - y_old)));

		/* Make the newton step: */
		y_new = MY_MIN(y_old - (E_old) / dEdy,0.99999999999999999999999);

		//Calculate deviation from 0
		E_new = GAMMAMAX_RAD * GAMMAMAX_RAD - (2.0 - y_new + sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new);

		/****************************************/
		/* Calculate the convergence criterion for iterated variables */
		/****************************************/
		errx = fabs(E_new) / (GAMMAMAX_RAD * GAMMAMAX_RAD);

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER))) {
			keep_iterating = 0;
		}

		y_old = y_new;
		E_old = E_new;

		n_iter++;
	}   // END of while(keep_iterating)
	y_max = 0.999998;

	fprintf(stderr, "y_max set to %f and gamma_rad becomes %f \n", y_max, sqrt((2.0 - y_max + sqrt(4.0 - 3.0 * y_max)) / (4.0 - 4.0 * y_max)));
}