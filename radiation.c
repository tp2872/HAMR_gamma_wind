#include "decs_MPI.h"

#define epsem (2.22E-16)
void raise_g(double vcov[], double gcon[][NDIM], double vcon[]);
void lower_g(double vcon[], double gcov[][NDIM], double vcov[]);
void ncov_calc(double gcon[][NDIM], double ncov[]); 
int Rtoprim_calc(double U[NPR_R], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_R], int lim);
#if(RAD_M1)
void implicit_rad_solve(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int *pflag, int *pflag_rad, struct of_geom *geom, double dU[NPR], double Dt, double cell_size
	#if(TWO_T)
	, double fel
	#endif
) {
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
		#if(TWO_T)
		, fel
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
void implicit_rad_solve_init(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size
	#if(TWO_T)
	, double fel
	#endif
) {
	double kappa_abs, kappa_es, tau, norm, bsq, Tr;
	int k, pflag = 0, pflag_rad = 0;
	struct of_state q;
	struct of_state_rad q_rad;

	#if(TWO_T)
	double gamma_g = calc_gamma_gas_prim(pb);
	#endif

	//Calculate optical depth
	get_state(pb, geom, &q);
	bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
	get_state_rad(pb, geom, &q_rad);
	Tr = calc_Tr(pb, q.ucon, q_rad.ucon, q_rad.ucov);
	kappa_abs = calc_kappa_abs(pb, bsq, Tr
		#if(TWO_T)
		, gamma_g
		#endif
	);
	kappa_es = calc_kappa_es(pb
		#if(TWO_T)
		, gamma_g
		#endif
	);
	tau = (kappa_abs + kappa_es) * cell_size;
	tau = 0.0;
	//Set guess values for primitives after implicit step based on optical depth
	if (tau < 0.66) {
		pflag = Utoprim_2d(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
			#if(TWO_T)
			, fel
			#endif
		);
		#if(DO_FONT_FIX)
		if (pflag) {
			pflag = Utoprim_1dvsq2fix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC);
			if (pflag) pflag = Utoprim_1dfix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC);
		}
		#endif	 

		//Even if MHD inversion fails, use updated value of radiation variable as gues
		if (!pflag)pflag_rad = Rtoprim(U_f, geom->gcov, geom->gcon, geom->g, pb, TYPE2);
	}

	//Set electron entropy variables
	#if(TWO_T)
	if (pflag == 0) U_i[ENTRE] = pb[ENTRE] * U_i[RHO]; //Apply heating only if primary (energy based) inversion succeeds; Otherwise assume adiabatic evolution of electrons
	U_f[ENTRE] = U_i[ENTRE];
	U_f[ENTRI] = U_i[ENTRI];
	#endif

	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q);
	if (pflag) pb[RHO] = U_i[RHO] / geom->g / q.ucon[0];
	#if(TWO_T)
	gamma_g = calc_gamma_gas_prim(pb);
	#endif
	mhd_calc(pb, 0, &q, &U_f[UU]
	#if(TWO_T)
		, gamma_g
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
	#if(P_NUM)
	U_f[PHOTON] = geom->g * pb[PHOTON] * q_rad.ucon[0];
	#endif
	for (k = UU_RAD; k <= U3_RAD; k++) U_f[k] *= geom->g;

	//Calculate source term for U_i
	source_rad(pb, geom, dU
		#if(TWO_T)
		, gamma_g
		#endif
	);

	//In low optical depth limit reset U_i to U_f
	if (tau < 0.66) {
		//for (k = 0; k < NPR; k++) U_i[k] = U_f[k];
	}

	//Calculate iterated error at start of iteration
	norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	error_t[0] = 0.25 * (fabs(U_f[UU] - U_i[UU] - Dt * dU[UU]) / norm);
	#if(TWO_T)
		#if(FIXEDGAMMA)
		double dK_dS = (GAMMAE - 1.) / pow(pb[RHO], GAMMAE - 1.0);
		#else
		fprintf(stderr, "Not implemented yet! \n");
		#endif
		error_t[0] += 0.25 * (fabs(U_f[ENTRE] - U_i[ENTRE] - Dt * dU[ENTRE]) / dK_dS / norm);
	#endif
	#if(P_NUM)
	norm = (fabs(U_i[PHOTON]) + fabs(U_f[PHOTON]) + fabs(Dt * dU[PHOTON]));
	error_t[0] += 0.25 * (fabs(U_f[PHOTON] - U_i[PHOTON] - Dt * dU[PHOTON]) / norm);
	#endif
	norm = (fabs(sqrt(geom->gcon[1][1]) * U_i[U1]) + fabs(U_f[U1]) + fabs(Dt * dU[U1]));
	norm += (fabs(sqrt(geom->gcon[2][2]) * U_i[U2]) + fabs(U_f[U2]) + fabs(Dt * dU[U2]));
	norm += (fabs(sqrt(geom->gcon[3][3]) * U_i[U3]) + fabs(U_f[U3]) + fabs(Dt * dU[U3]));
	error_t[0] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_f[U1] - U_i[U1] - Dt * dU[U1]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_f[U2] - U_i[U2] - Dt * dU[U2]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_f[U3] - U_i[U3] - Dt * dU[U3]) / norm);


	//Calculate total error at start of iteration
	norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
	if (pflag_rad == 0)error_t[0] += 0.25 * (fabs(U_f[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU]) / norm);
	norm = (fabs(sqrt(geom->gcon[1][1]) * U_i[U1_RAD]) + fabs(U_f[U1_RAD]) + fabs(Dt * dU[U1_RAD]));
	norm += (fabs(sqrt(geom->gcon[2][2]) * U_i[U2_RAD]) + fabs(U_f[U2_RAD]) + fabs(Dt * dU[U2_RAD]));
	norm += (fabs(sqrt(geom->gcon[3][3]) * U_i[U3_RAD]) + fabs(U_f[U3_RAD]) + fabs(Dt * dU[U3_RAD]));
	error_t[0] += 0.25 * sqrt(geom->gcon[1][1]) * (fabs(U_f[U1_RAD] - U_i[U1_RAD] - Dt * dU[U1_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[2][2]) * (fabs(U_f[U2_RAD] - U_i[U2_RAD] - Dt * dU[U2_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[3][3]) * (fabs(U_f[U3_RAD] - U_i[U3_RAD] - Dt * dU[U3_RAD]) / norm);
}

int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged) {
		double U_new[NPR], U_old[NPR], U_prev[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdpb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[5], offset = 1.e-8;
	double T_GAS, dK_dS, norm, norm_S, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;
	#if(TWO_T)
	double gamma_g, ue, ui;
	#endif

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
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4+TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
			n_iter_jacob = 0;

			do {
				PLOOP pb_new[k] = pb_old[k];
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb_old[i] + dpb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[ENTRE]);
					pb_new[ENTRE] = pb_old[ENTRE] + dpb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[PHOTON]);
					pb_new[PHOTON] = pb_old[PHOTON] + dpb;
				}
				#endif
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[i - UU][i - UU]);
					pb_new[i] = pb_old[i] + dpb;
				}

				// Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				get_state(pb_new, geom, &q);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				#if(TWO_T)
					//Set for 2T fluid entropy of ions based on electron entropy
					#if(FIXEDGAMMA)
					ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
					if (ue > 0.99 * pb_new[UU]) ue = 0.99 * pb_new[UU];
					if (ue < 0.01 * pb_new[UU]) ue = 0.01 * pb_new[UU];
					pb_new[ENTRE]= (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
					ui = pb_new[UU] - ue;
					pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
					#else
					fprintf(stderr, "Not implemented yet! \n");
					#endif
				U_new[ENTRE] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRE];
				U_new[ENTRI] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRI];
				gamma_g = calc_gamma_gas_prim(pb_new);
				#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
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
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Calculate radiative (including coulomb) source term
				source_rad(pb_new, geom, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);

				//Calculate Jacobian
				for (k = U1; k <= U3; k++) {
					E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdpb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb;
				}
				#if(TWO_T)
				E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
				dEdpb[4][i - UU] = (E_new[4] - E_old[4]) / dpb;
				#endif
				#if(P_NUM)
				E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
				dEdpb[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dpb;
				#endif
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb[0][i - UU] = (E_new[0] - E_old[0]) / dpb;

				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdpb, dEdpb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdpb, dEdpb_inv);
				#else
				flag = invert_matrix(dEdpb, dEdpb_inv);
				#endif

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
				dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
					#if(TWO_T)
					 + E_old[4] * dEdpb_inv[k][4]
					#endif
					#if(P_NUM)
					+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
					#endif
					);
				pb_new[k + UU] = pb_old[k + UU] + dpb;
			}
			#if(TWO_T)
			dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + TWO_T] * dEdpb_inv[4][4 + TWO_T]
				#endif	
				);
			pb_new[ENTRE] = pb_old[ENTRE] + dpb;
			#endif
			#if(P_NUM)
			dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			pb_new[PHOTON] = pb_old[PHOTON] + dpb;
			#endif
		}
		else {
			//Set damping factor for Newton Raphson method
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if (n_iter == 8) D = 0.25;
			else D = 1.;

			if (n_iter / 4 == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			if (n_iter / 4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0]
						#if(TWO_T)
						+E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			#if(TWO_T)
			dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
				#if(P_NUM)
				+E_old[4 + P_NUM] * dEdpb_inv[4][4 + P_NUM]
				#endif	
				);
			pb_new[ENTRE] = pb_old[ENTRE] + dpb;
			#endif
			#if(P_NUM)
			dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
				);
			pb_new[PHOTON] = pb_old[PHOTON] + dpb;
			#endif
		}

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);
		#if(TWO_T)
		if (pb_new[ENTRE] < 0.0) pb_new[ENTRE] = 0.5 * fabs(pb_new[ENTRE]);
		#endif
		#if(P_NUM)
		if (pb_new[PHOTON] < 0.0) pb_new[PHOTON] = 0.5 * fabs(pb_new[PHOTON]);
		#endif

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		#if(TWO_T)
			#if(FIXEDGAMMA)
			ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
			if (ue > 0.99 * pb_new[UU]) ue = 0.99 * pb_new[UU];
			if (ue < 0.01 * pb_new[UU]) ue = 0.01 * pb_new[UU];
			pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
			ui = pb_new[UU] - ue;
			pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
			#else
			fprintf(stderr, "Not implemented yet! \n");
			#endif
		U_new[ENTRE] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRE];
		U_new[ENTRI] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRI];
		gamma_g = calc_gamma_gas_prim(pb_new);
		#endif
		mhd_calc(pb_new, 0, &q, &U_new[UU]
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
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
		if (flag_rad) for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_new[k];

		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		#if(P_NUM)
		U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
		#endif
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Get radiative source term
		source_rad(pb_new, geom, dU_new
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
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
			dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
			error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / dK_dS)) / (norm);
			#endif
		}
		#if(TWO_T)
			#if(FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
			#else
			fprintf(stderr, "Not implemented yet! \n");
			#endif
			//norm =  (fabs(U_i[ENTRE]) + fabs(U_new[ENTRE]) + fabs(Dt * dU_new[ENTRE]));
			error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
		#endif
		#if(P_NUM)
		norm =  (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
		error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		#endif

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
			if (flag_rad) {
				Rtoprim(U_prev, geom->gcov, geom->gcon, geom->g, pb, BASIC);

				//Recompute R_t^mu for consistency
				get_state_rad(pb, geom, &q_rad);
				mhd_calc_rad(pb, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;
			}

		}

		n_iter++;
	}

	return(0);
}


// This method iterates T^t_mu
int implicit_rad_solve_UMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged){
	return(0);
}

// This method iterates Su^t and T^t_i
int implicit_rad_solve_EMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged){
	return(0);
}

// This method iterates R^t_mu
int implicit_rad_solve_URAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged){
	return(0);
}

// This method iterates E_RAD an U_rad
int implicit_rad_solve_PRAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, int do_entropy, int do_staged){
	return(0);
}

int subcycle_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double cell_size){
	double factor, remainder = 1.0,  Uh[NPR], U_new[NPR], ph[NPR], pb_new[NPR], pb_old[NPR], fraction;
	double kappa_abs, kappa_emmit, kappa_es, tau;
	int flag = 0, keep_iterating = 1, nstep = 0, k;
	struct of_state q;
	struct of_state_rad q_rad;
	#if(TWO_T)
	double gamma_g;
	#endif

	for (k = 0; k < NPR; k++) pb_old[k] = pb[k];

	//Get primitive variables belonging to U_i
	flag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb_old, NEWT_TOL, TYPE2
		#if(TWO_T)
		, 0.0
		#endif
	);
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
			source_rad(pb_new, geom, dU
				#if(TWO_T)
				, gamma_g
				#endif
			);

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

			flag = Utoprim_2d(Uh, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC
				#if(TWO_T)
				, 0.0
				#endif
			);
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
				mhd_calc(ph, 0, &q, &Uh[UU]
					#if(TWO_T)
					, gamma_g
					#endif
				);
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
				source_rad(ph, geom, dU
					#if(TWO_T)
					, gamma_g
					#endif
				);
				U_new[UU_RAD] += factor * Dt * dU[UU_RAD];
				U_new[U1_RAD] += factor * Dt * dU[U1_RAD];
				U_new[U2_RAD] += factor * Dt * dU[U2_RAD];
				U_new[U3_RAD] += factor * Dt * dU[U3_RAD];
				U_new[UU] += factor * Dt * dU[UU];
				U_new[U1] += factor * Dt * dU[U1];
				U_new[U2] += factor * Dt * dU[U2];
				U_new[U3] += factor * Dt * dU[U3];
				U_new[KTOT] += factor * Dt * dU[KTOT];

				flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
					#if(TWO_T)
					, 0.0
					#endif
				);
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
					mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(TWO_T)
						, gamma_g
					#endif
					);
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
	double U_tmp[NPR_R + P_NUM], prim_tmp[NPR_R + P_NUM];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
	for (i = 0; i < NPR_R; i++) U_tmp[i] = alpha * U[i + UU_RAD] / gdet;
	#if(P_NUM)
	U_tmp[4] = alpha * U[PHOTON] / gdet;
	#endif

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i < NPR_R; i++) prim_tmp[i] = prim[i + UU_RAD]; //radiation prims
	#if(P_NUM)
	prim_tmp[4] = prim[PHOTON];
	#endif

	//Do inversion
	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, lim);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i < NPR_R; i++) {
		prim[i + UU_RAD] = prim_tmp[i];
	}
	#if(P_NUM)
	prim[PHOTON] = prim_tmp[4];
	#endif

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

	//Check for bad values of -Erad and U_tilde^2; If values are nan floor them
	if (!isfinite(Qdotn)) Qdotn = -(1.e-30);
	if (!isfinite(Qtsq)) Qtsq = 0.0;

	y = Qtsq / (Qdotn * Qdotn + 1.e-30); //Definition from McKinney2013. Should only range [0,1].
	if (y < 0. || isnan(y)) y = 0.;
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	#if(P_NUM)
	prim[4] = U[4] / sqrt(gammasq);
	#endif

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

			f = 0.;// sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
			if (f < 10000000.0) {
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			}
			Qdotn = -(1.e-150 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad

			returnval = 1;
		}
		else {
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
			gammasq = 1.0;
		}

		#if(P_NUM)
		prim[4] = U[4] / sqrt(gammasq);
		#endif
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

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
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
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad
			returnval = 1;
		}
		else if (!isfinite(prim[0])) {
			prim[0] = 1.e-30;
		}

		#if(P_NUM)
		prim[4] = U[4] / sqrt(GAMMAMAX_RAD * GAMMAMAX_RAD);
		#endif
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
}
#endif
