#include "decs_MPI.h"

#define epsem (2.22E-16)
void raise_g(double vcov[], double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); 
int Rtoprim_calc(double U[NPR_R], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);
#if(RAD_M1)
void implicit_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double cell_size) {
	double error_t = 1.e-3;
	int k;
	double U_ft[NPR], pb_i[NPR], U_n_temp[NPR], U_i_temp[NPR];

	PLOOP{
		U_n_temp[k] = U_n[k];
		U_i_temp[k] = U_i[k];
	}

	//Initialize temporary variables
	PLOOP{
		dU[k] = 0.;
		U_ft[k] = U_i_temp[k];
		pb_i[k] = pb[k];
	}

	//Set initial values and error before attempting implicit solver
	implicit_rad_solve_init(pb_i, U_n_temp, U_i_temp, U_ft, geom, dU, Dt, &error_t, cell_size
		#if(DOHELM)
		, gpu_eos_table
		#endif
	);

	//If we've reached the tolerance level, exit immediately and update variables
	if (error_t < 1.e-12) {
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
	else {
		if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
		);

		//As final resort attempt subcycling
		if (error_t > 1.e-7) {
			//subcycle_rad_solve(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, cell_size);
		}

		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
}

//Calculate initial error for source term and set initial guess values
void implicit_rad_solve_init(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size) {
	double kappa_abs, kappa_es, tau = 0., norm;
	int k, pflag, pflag_rad;
	struct of_state q;
	struct of_state_rad q_rad;

	//Calculate optical depth
	kappa_abs = calc_kappa_abs(pb
		#if(DOHELM)
		, gpu_eos_table
		#endif
	);
	kappa_es = calc_kappa_es(pb
		#if(DOHELM)
		, gpu_eos_table
		#endif
	);
	tau = (kappa_abs + kappa_es) * cell_size;

	//if (tau < 0.66) {
		//Set guess values for primitives after implicit step based on optical depth
	pflag = Utoprim_2d(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
	);
	#if(!DO_FONT_FIX)
	if (pflag) {
		pflag = Utoprim_1dvsq2fix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
	}
	if (pflag) pflag = Utoprim_1dfix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY);
	#endif	 

	//Even if MHD inversion fails, use updated value of radiation variable as gues
	pflag_rad = Rtoprim(U_f, geom->gcov, geom->gcon, geom->g, pb, BASIC);
	//}

	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q);
	if (pflag) pb[RHO] = U_i[RHO] / geom->g / q.ucon[0];
	mhd_calc(pb, 0, &q, &U_f[UU]
		#if(DOHELM)
		, gpu_eos_table
		#endif
	);
	for (k = UU; k <= U3; k++)U_f[k] *= geom->g;
	U_f[UU] += U_f[RHO];

	//Recompute entropy for consistency
	#if(FULL_ENTROPY)
	U_f[KTOT] = geom->g * pb[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb[UU] * pow(pb[RHO], -GAMMA));
	#else
	U_f[KTOT] = geom->g * pb[RHO] * q.ucon[0] * (GAMMA - 1.) * pb[UU] * pow(pb[RHO], -GAMMA);
	#endif

	//Recompute R_t^mu for consistency
	get_state_rad(pb, geom, &q_rad);
	mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
	for (k = UU_RAD; k <= U3_RAD; k++)U_f[k] *= geom->g;

	//Calculate source term for U_i
	source_rad(pb, geom, dU
		#if(DOHELM)
		, gpu_eos_table
		#endif
	);

	//Calculate iterated error at start of iteration
	norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	error_t[0] = 0.25 * (fabs(U_f[UU] - U_i[UU] - Dt * dU[UU]) / norm);
	norm = (fabs(sqrt(geom->gcon[1][1]) * U_i[U1]) + fabs(U_f[U1]) + fabs(Dt * dU[U1]));
	norm += (fabs(sqrt(geom->gcon[2][2]) * U_i[U2]) + fabs(U_f[U2]) + fabs(Dt * dU[U2]));
	norm += (fabs(sqrt(geom->gcon[3][3]) * U_i[U3]) + fabs(U_f[U3]) + fabs(Dt * dU[U3]));
	error_t[0] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_f[U1] - U_i[U1] - Dt * dU[U1]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_f[U2] - U_i[U2] - Dt * dU[U2]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_f[U3] - U_i[U3] - Dt * dU[U3]) / norm);

	//Calculate total error at start of iteration
	norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
	error_t[0] += 0.25 * (fabs(U_f[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU]) / norm);
	norm = (fabs(sqrt(geom->gcon[1][1]) * U_i[U1_RAD]) + fabs(U_f[U1_RAD]) + fabs(Dt * dU[U1_RAD]));
	norm += (fabs(sqrt(geom->gcon[2][2]) * U_i[U2_RAD]) + fabs(U_f[U2_RAD]) + fabs(Dt * dU[U2_RAD]));
	norm += (fabs(sqrt(geom->gcon[3][3]) * U_i[U3_RAD]) + fabs(U_f[U3_RAD]) + fabs(Dt * dU[U3_RAD]));
	error_t[0] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_f[U1_RAD] - U_i[U1_RAD] - Dt * dU[U1_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_f[U2_RAD] - U_i[U2_RAD] - Dt * dU[U2_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_f[U3_RAD] - U_i[U3_RAD] - Dt * dU[U3_RAD]) / norm);
}

int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged
		#if(DOHELM)
		, double* gpu_eos_table
		#endif
	) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4][4], dEdpb_inv[4][4], error_new[5], offset = 1.e-8;
	double T_GAS, norm, norm_S, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;

	//Set error to previous value
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = dU[k];
		U_new[k] = U_old[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;

			do {
				PLOOP pb_new[k] = pb_old[k];
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb_old[i] + dpb;
				}
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i-UU][i-UU]);
					pb_new[i] = pb_old[i] + dpb;
				}

				// Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				get_state(pb_new, geom, &q);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
				);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] = U_new[UU] + U_new[RHO];

				if (do_entropy == 1) {
					#if(FULL_ENTROPY)
					U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
					#else
					U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
					#endif				
				}

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Calculate source function and jacobian
				source_rad(pb_new, geom, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
				);
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
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		//Set primitive variables before Newton step
		PLOOP pb_new[k] = pb_old[k];

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.0;
			for (k = 0; k < 4; k++) {
				dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
				pb_new[k + UU] = pb_old[k + UU] + dpb;
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
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			if (n_iter / 4 == 1) {
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
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		mhd_calc(pb_new, 0, &q, &U_new[UU]
			#if(DOHELM)
			, gpu_eos_table
			#endif
		);
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

		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Get radiative source term
		source_rad(pb_new, geom, dU_new
			#if(DOHELM)
			, gpu_eos_table
			#endif
		);

		//Calculate iterated error
		norm = sqrt(geom->gcon[1][1]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
		norm += sqrt(geom->gcon[2][2]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
		norm += sqrt(geom->gcon[3][3]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
		else {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
			error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
			#else
			double dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
			error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / dK_dS)) / (norm);
			#endif
		}

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10;
		else offset = 1.e-8;

		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		norm = sqrt(geom->gcon[1][1]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
		norm += sqrt(geom->gcon[2][2]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
		norm += sqrt(geom->gcon[3][3]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= 1.e-12) || (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.5 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			keep_iterating = 0;
		}

		//If error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		//Reset variables if Newton step succesfull
		if (keep_iterating) {
			for (k = 0; k < NPR; k++) {
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}
		}

		//If error decreased compared to start value, update variables
		if (fabs(error_new[n_iter % 5]) < error_t[0]) {
			error_t[0] = error_new[n_iter % 5];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
		}

		n_iter++;
		}
	return(0);
}


// This method iterates T^t_mu
int implicit_rad_solve_UMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4][4], dEdUb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase_gas = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_old[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;
			do {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]);
					U_new[i] = U_old[i] + dUb;
				}
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[i - UU][i - UU]);
					U_new[i] = U_old[i] + dUb;
				}

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
				#if(DO_FONT_FIX)
				if (flag && (n_iter_jacob > 1)) {
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
					if (flag) {
						flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
					}
				}
				#endif

				if (flag == 0) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

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

					//Calculate source term using new variables
					source_rad(pb_new, geom, dU_new);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					if (do_entropy == 1) {
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;

					flag = invert_matrix(dEdUb, dEdUb_inv);
				}
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		n_iter_fail = 0;
		while (n_iter_fail < 10) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
					U_new[k + UU] = U_old[k + UU] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
			}

			//Derive new conserved quantaties for MHD variables
			U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
			U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
			U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
			U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			#if(DO_FONT_FIX)
			if (flag && (n_iter_fail > 1)) {
				flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				}
			}
			#endif

			if (flag == 0) {
				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

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
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				//fprintf(stderr, "error_i: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Calculate total error	
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					#if(FULL_ENTROPY)
					norm_S = T_GAS * (fabs(U_i[KTOT]) + fabs(U_new[KTOT]) + fabs(Dt * dU_new[KTOT]));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (pb_new[UU]) / pow(pb_new[RHO], GAMMA);
					norm_S = T_GAS * (fabs(U_i[RHO] / (GAMMA - 1.) * log(U_i[KTOT] / U_i[RHO])) + fabs(U_new[RHO] / (GAMMA - 1.) * log(U_new[KTOT] / U_new[RHO])) + fabs(Dt * dU_new[KTOT] / dK_dS));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / dK_dS) / (norm);
					#endif
				}
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//fprintf(stderr, "error_t: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If the residual drops below machine precision, stop iterating
				//if (((fabs(U_new[UU] / U_old[UU]) - 1.0) + (fabs(U_new[U1] / U_old[U1]) - 1.0) + (fabs(U_new[U2] / U_old[U2]) - 1.0) + (fabs(U_new[U3] / U_old[U3]) - 1.0)) < 10. * epsem) {
					//keep_iterating = 0;
				//}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.5 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
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
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 10) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates Su^t and T^t_i
int implicit_rad_solve_EMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4][4], dEdUb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase_gas = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_old[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;
			do {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[KTOT]);
					U_new[KTOT] = U_old[KTOT] + dUb;
				}
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[i - UU][i - UU]);
					U_new[i] = U_old[i] + dUb;
				}

				//Invert conserved MHD quantities using entropy based methods
				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
				if (flag) flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);

				if (flag == 0) {
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

					//Set radiation conserved quantities
					U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
					U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
					U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
					U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Calculate source term using new variables
					source_rad(pb_new, geom, dU_new);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					if (do_entropy == 1) {
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;

					flag = invert_matrix(dEdUb, dEdUb_inv);
				}
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		n_iter_fail = 0;
		while (n_iter_fail < 10) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
					if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
					else U_new[k + UU] = U_old[k + UU] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
			}

			//Invert conserved MHD quantities using entropy based methods
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			if (flag) flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);

			if (flag == 0) {
				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

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

				//Derive new conserved quantaties for MHD variables
				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);
		
				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Get radiative source term
				source_rad(pb_new, geom, dU_new);

				//Calculate iterated error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				//fprintf(stderr, "error_i: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Calculate total error	
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					#if(FULL_ENTROPY)
					norm_S = T_GAS * (fabs(U_i[KTOT]) + fabs(U_new[KTOT]) + fabs(Dt * dU_new[KTOT]));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (pb_new[UU]) / pow(pb_new[RHO], GAMMA);
					norm_S = T_GAS * (fabs(U_i[RHO] / (GAMMA - 1.) * log(U_i[KTOT] / U_i[RHO])) + fabs(U_new[RHO] / (GAMMA - 1.) * log(U_new[KTOT] / U_new[RHO])) + fabs(Dt * dU_new[KTOT] / dK_dS));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / dK_dS) / (norm);
					#endif
				}
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//fprintf(stderr, "error_t: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If the residual drops below machine precision, stop iterating
				//if (((fabs(U_new[UU] / U_old[UU]) - 1.0) + (fabs(U_new[U1] / U_old[U1]) - 1.0) + (fabs(U_new[U2] / U_old[U2]) - 1.0) + (fabs(U_new[U3] / U_old[U3]) - 1.0)) < 10. * epsem) {
					//keep_iterating = 0;
				//}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.5 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
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
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 10) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates R^t_mu
int implicit_rad_solve_URAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4][4], dEdUb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1,  n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase_gas = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_old[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
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
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
				#if(DO_FONT_FIX)
				if (flag && (n_iter_jacob > 1)) {
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
					if (flag) {
						flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
					}
				}
				#endif

				if (flag == 0) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, TYPE2);

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

					//Calculate source term using new variables
					source_rad(pb_new, geom, dU_new);

					//Calculate source function and jacobian
					for (k = U1_RAD; k <= U3_RAD; k++) {
						E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dUb;
					}
					if (do_entropy == 1) {
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
					dEdUb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dUb;

					flag = invert_matrix(dEdUb, dEdUb_inv);
				}
				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		n_iter_fail = 0;
		while(n_iter_fail<10){
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]);
					U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

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

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			#if(DO_FONT_FIX)
			if (flag && (n_iter_fail > 1)) {
				flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				}
			}
			#endif

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
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				//fprintf(stderr, "error_i: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				//else offset = pow(10., -8.);

				//Calculate total error	
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					#if(FULL_ENTROPY)
					norm_S = T_GAS * (fabs(U_i[KTOT]) + fabs(U_new[KTOT]) + fabs(Dt * dU_new[KTOT]));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (pb_new[UU]) / pow(pb_new[RHO], GAMMA);
					norm_S = T_GAS * (fabs(U_i[RHO] / (GAMMA - 1.) * log(U_i[KTOT] / U_i[RHO])) + fabs(U_new[RHO] / (GAMMA - 1.) * log(U_new[KTOT] / U_new[RHO])) + fabs(Dt * dU_new[KTOT] / dK_dS));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))/dK_dS) / (norm);
					#endif
				}
				if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

				//fprintf(stderr, "error_t: %f, n_iter: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If the residual drops below machine precision, stop iterating
				//if (((fabs(U_new[UU_RAD] / U_old[UU_RAD]) - 1.0) + (fabs(U_new[U1_RAD] / U_old[U1_RAD]) - 1.0) + (fabs(U_new[U2_RAD] / U_old[U2_RAD]) - 1.0) + (fabs(U_new[U3_RAD] / U_old[U3_RAD]) - 1.0)) < 10. * epsem) {
					//keep_iterating = 0;
				//}


				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 0.5 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//If gas negative more than 2 times stop iterating
				if (pb_new[UU]<0.) {
					count_increase_gas++;
					if (count_increase > 2) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
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
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 10) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates E_RAD an U_rad
int implicit_rad_solve_PRAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4][4], dEdpb_inv[4][4], bsq, error_new[5], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail=0, keep_iterating = 1, flag, n_iter_jacob, count_increase = 0, count_increase_gas = 0;

	//Set error to 0
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_old[k];
		dU_old[k] = dU[k];
	}

	double error_temp = error_new[0];
	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Set reference error at start
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		for (i = UU_RAD; i <= U3_RAD; i++) {
			n_iter_jacob = 0;
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
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
				#if(DO_FONT_FIX)
				if (flag && (n_iter_jacob>1)) {
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);	
					if (flag) {
						flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2);
					}
				}
				#endif

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
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		n_iter_fail = 0;
		while (n_iter_fail < 10) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1. / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

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
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Get MHD primitives
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
			#if(DO_FONT_FIX)
			if (flag && (n_iter_fail > 1)) {
				flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				if (flag) {
					flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2);
				}
			}
			#endif

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
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				//else offset = pow(10., -8.);
				//fprintf(stderr, "error_i: %f, n_iter: %d, n_iter_fail: %d pb_uu: %f pb_uurad: %f \n", log10(error_new[n_iter % 5]), n_iter, n_iter_fail, log10(pb_new[UU]), log10(pb_new[UU_RAD]));

				//Calculate total error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					#if(FULL_ENTROPY)
					norm_S = T_GAS * (fabs(U_i[KTOT]) + fabs(U_new[KTOT]) + fabs(Dt * dU_new[KTOT]));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (pb_new[UU]) / pow(pb_new[RHO], GAMMA);
					norm_S = T_GAS * (fabs(U_i[RHO] / (GAMMA - 1.) * log(U_i[KTOT] / U_i[RHO])) + fabs(U_new[RHO] / (GAMMA - 1.) * log(U_new[KTOT] / U_new[RHO])) + fabs(Dt * dU_new[KTOT] / dK_dS));
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / dK_dS) / (norm);
					#endif
				}
				if (do_entropy == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

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

				//If gas negative more than 2 times stop iterating
				if (pb_new[UU] < 0.) {
					count_increase_gas++;
					//if (count_increase > 2) keep_iterating = 0;
				}

				//If we've reached the tolerance level in total error or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5]) <= pow(10, -12.)) || (n_iter >= 20)) {
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
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 10) return(1);
			}
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
	flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, TYPE2);
	#if(DO_FONT_FIX)
	if (flag) {
		flag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, TYPE2);
		if (flag) {
			flag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, TYPE2);
		}
	}
	#endif

	if (!flag) {
		Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb_old, TYPE2);

		//Something like Courant number in implicity scheme; e.g. we do not let the source term in each step be bigger than this
		fraction = 0.25;// MY_MIN(1.0 / (tau * Dt), 1.0);

		//Set temporary variables to their initial values
		for (k = 0; k < NPR; k++) {
			ph[k] = pb_old[k];
			pb_new[k] = pb_old[k];
			Uh[k] = U_i[k];
			U_new[k] = U_i[k];
		}

		while (keep_iterating && nstep<100) {
			//Calculate source term for half-step
			source_rad(pb_new, geom, dU);

			//Find optical depth for one cell
			//kappa_abs = calc_kappa_abs(pb_new);
			//kappa_es = calc_kappa_es(pb_new);
			//tau = (kappa_abs + kappa_es) * cell_size;

			//Set size of subcycling timestep
			//if (fraction * MY_MIN(U_new[UU], U_new[UU_RAD]) >= fabs(dU[UU_RAD] * Dt)) keep_iterating = 0;
			//factor = MY_MIN(fraction * MY_MIN(fabs(U_new[UU]), fabs(U_new[UU_RAD])) / fabs(dU[UU_RAD] * Dt), remainder);
			factor = MY_MIN(1./100., remainder);
			remainder -= factor;

			Uh[UU_RAD] = U_new[UU_RAD] + 0.5 * factor * Dt * dU[UU_RAD];
			Uh[U1_RAD] = U_new[U1_RAD] + 0.5 * factor * Dt * dU[U1_RAD];
			Uh[U2_RAD] = U_new[U2_RAD] + 0.5 * factor * Dt * dU[U2_RAD];
			Uh[U3_RAD] = U_new[U3_RAD] + 0.5 * factor * Dt * dU[U3_RAD];
			Uh[UU] = U_new[UU] + 0.5 * factor * Dt * dU[UU];
			Uh[U1] = U_new[U1] + 0.5 * factor * Dt * dU[U1];
			Uh[U2] = U_new[U2] + 0.5 * factor * Dt * dU[U2];
			Uh[U3] = U_new[U3] + 0.5 * factor * Dt * dU[U3];
			Uh[KTOT] = U_new[KTOT] + 0.5 * factor * Dt * dU[KTOT];

			flag = Utoprim_2d(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC);
			#if(DO_FONT_FIX)
			if (flag) {
				flag = Utoprim_1dvsq2fix1(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC);
				if (flag) {
					flag = Utoprim_1dfix1(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC);
				}
			}
			#endif

			if (!flag) {
				Rtoprim(Uh, geom->gcov, geom->gcon, geom->g, ph, TYPE2);

				//Recompute T_t^mu for consistency
				Uh[RHO] = U_i[RHO];
				get_state(ph, geom, &q);
				mhd_calc(ph, 0, &q, &Uh[UU]);
				for (k = UU; k <= U3; k++)Uh[k] *= geom->g;
				Uh[UU] += Uh[RHO];

				//Recompute entropy for consistency
				#if(FULL_ENTROPY)
				Uh[KTOT] = geom->g * ph[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * ph[UU] * pow(ph[RHO], -GAMMA));
				#else
				Uh[KTOT] = geom->g * ph[RHO] * q.ucon[0] * (GAMMA - 1.) * ph[UU] * pow(pb[RHO], -GAMMA);
				#endif

				//Recompute R_t^mu for consistency
				get_state_rad(ph, geom, &q_rad);
				mhd_calc_rad(ph, 0, &q_rad, &Uh[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)Uh[k] *= geom->g;

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
				U_new[KTOT] += factor * Dt * dU[KTOT];

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC);
				#if(DO_FONT_FIX)
				if (flag) {
					flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC);
					if (flag) {
						flag = Utoprim_1dfix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC);
					}
				}
				#endif

				if (!flag) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, BASIC);
					
					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q);
					mhd_calc(pb_new, 0, &q, &U_new[UU]);
					for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
					U_new[UU] += U_new[RHO];

					//Recompute entropy for consistency
					#if(FULL_ENTROPY)
					U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * ph[UU] * pow(pb_new[RHO], -GAMMA));
					#else
					U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * ph[UU] * pow(pb_new[RHO], -GAMMA);
					#endif

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;
				}
			}

			//In case of inversion failure reset and start over again with smaller step
			if (flag) {
				/*for (k = 0; k < NPR; k++) {
					ph[k] = pb_old[k];
					pb_new[k] = pb_old[k];
					Uh[k] = U_i[k];
					U_new[k] = U_i[k];
				}
				fraction /= 2.;
				remainder = 1.;*/
				keep_iterating = 0;
			}
			if (remainder < pow(10., -5.)) keep_iterating = 0;
			nstep++;
		}
	}

	if (nstep >= 100) {
		//fprintf(stderr, "Error in subcycling: too many timesteps! \n");
		return 1;
	}
	else if (flag) {
		//fprintf(stderr, "Failed MHD inversion! \n");
		return 2;
	}
	else if (remainder >= pow(10., -9.)) {
		//fprintf(stderr, "Error in subcycling: Remainder not 0 ! \n");
		return 3;
	}
	else {
		for (k = 0; k < NPR; k++) {
			pb[k] = pb_new[k];
			U_f[k] = U_new[k];
			dU[k] = (U_f[k]-U_i[k])/Dt;
		}
	}

	return 0;
}

//Inversion from radiation conserved to primitive quantities
int Rtoprim(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], int lim) {
	double U_tmp[NPR_R], prim_tmp[NPR_R];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
	for (i = 0; i <= U3_RAD - UU_RAD; i++) U_tmp[i] = alpha * U[i + NPR_U] / gdet;

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i <= U3_RAD - UU_RAD; i++) prim_tmp[i] = prim[i + NPR_U]; //radiation prims

	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i <= U3_RAD - UU_RAD; i++) {
		prim[i + NPR_U] = prim_tmp[i];
	}

	return(ret);
}

// Limits radiation with either BASIC or TYPE2 approaches
int Rtoprim_calc(double U[NPR_R], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim)
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

	y = Qtsq / (Qdotn * Qdotn + 1.e-150); //Definition from McKinney2013. Should only range [0,1].
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	if (isnan(Qdotn) || prim[0] < 0. || isnan(y) || y < 0.) {
		prim[0] = 1.e-150;
		prim[1] = 0.;
		prim[2] = 0.;
		prim[3] = 0.;

		// Get Ebar and p_rad as usual
		if (!isnan(Qdotn) && Qdotn < 0.0) {
			pressure = -Qdotn / (4. - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad
		}

		return 0;
	}
	if (y > y_max) {
		Uabs = 0.5 * (sqrt(Qtsq) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++)prim[i] = Qtcon[i] / Uabs;

		qsq = gcov[1][1] * prim[1] * prim[1] + gcov[2][2] * prim[2] * prim[2] + gcov[3][3] * prim[3] * prim[3]
			+ 2. * (gcov[1][2] * prim[1] * prim[2] + gcov[1][3] * prim[1] * prim[3] + gcov[2][3] * prim[2] * prim[3]);
		if (qsq < 0. && fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;

		if (y < 1. - 100. * NUMEPSILON) {
			if (lim == TYPE2) Qdotn = -(1.e-150 + sqrt(Qtsq / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			returnval = (prim[0] < 0.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad		
		}
		else {
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
			pressure = -Qdotn / (4. * 1. - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad		
		}
		return 0;
		//else if (y>1.-100.*NUMEPSILON){
		//	prim[1] = 0.;
		//	prim[2] = 0.;
		//	prim[3] = 0.;
		//	pressure = -Qdotn / (4. - 1.);
		//	prim[0] = pressure * 3.; // Erad = 3*p_rad
		//}
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
	E_old = GAMMAMAX_RAD * GAMMAMAX_RAD - (2.0 - y_old + sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old);

	while (keep_iterating) {	
		//Calculate gradient dEdy
		dEdy = (0.375 * y_old - 0.25 * sqrt(4.0 - 3.0 * y_old) - 0.625) / (sqrt(4.0 - 3.0 * y_old) * (1.0 - y_old) * (1.0 - y_old));

		/* Make the newton step: */
		y_new = MY_MIN(y_old - (E_old) / dEdy,0.99999999999999);

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
		if (((fabs(errx) <= NEWT_TOL)) || (n_iter >= (MAX_NEWT_ITER*4))) {
			keep_iterating = 0;
		}
		//fprintf(stderr, "y_max set to %f and gamma_rad becomes %f \n", y_new, sqrt((2.0 - y_new + sqrt(4.0 - 3.0 * y_new)) / (4.0 - 4.0 * y_new)));

		y_old = y_new;
		E_old = E_new;

		n_iter++;
	}   // END of while(keep_iterating)
	y_max = y_new;
	fprintf(stderr, "Gamma_RAD set to : %f\n", sqrt((2.0 - y_old + sqrt(4.0 - 3.0 * y_old)) / (4.0 - 4.0 * y_old)));
}
#endif
