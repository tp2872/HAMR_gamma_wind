
#if(RAD_M1)
__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	double error_t[2]; 
	int k;
	double  U_ft[NPR], pb_i[NPR], U_n_temp[NPR], U_i_temp[NPR], U_prev[NPR];
	#if(P_NUM)	
		#if(!CALC_MDOT)
		double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
		#else
		double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
		#endif
	#endif

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
	implicit_rad_solve_init(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, geom, dU, Dt, error_t, cell_size, y_max, pflag_rad
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);

	//If we've reached the tolerance level, exit immediately and update variables
	if (error_t[1] < 1.e-12 && pflag_rad[0]==0) {
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
	else {
		#if (HIGH_MDOT)
		if (error_t[1] > 1.e-9 || pflag_rad[0])implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		
		/*if (pflag_rad[0])implicit_rad_solve_URAD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);*/
		#else
		if (error_t[1] > 1.e-9 || pflag_rad[0])implicit_rad_solve_URAD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);

		if (pflag_rad[0])implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		#endif
		/*if (pflag_rad[0])implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		if (pflag_rad[0])implicit_rad_solve_EMHD(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);*/
	
		//If used TYPE2 limiter for inversion radiative quantities, redo with BASIC limiter
		if(pflag_rad[0]){
			struct of_state_rad q_rad;

			//Invert using basic limiter
			Rtoprim(U_prev, geom->gcov, geom->gcon, geom->g, pb_i, y_max, BASIC
				#if(CALC_MDOT)
				, mass_density_scale, magnetic_density_scale
				#endif
			);

			//Recompute R_t^mu for consistency
			get_state_rad(pb_i, geom, &q_rad);
			mhd_calc_rad(pb_i, 0, &q_rad, &U_ft[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_ft[k] *= geom->g;

			//Recompute photon number
			#if(P_NUM)
			double Tr;
			Tr = pow(pb_i[UU_RAD] * energy_density_scale / ARAD, 0.25);
			pb_i[PHOTON] = pb_i[UU_RAD] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
			U_ft[PHOTON] = geom->g * pb_i[PHOTON] * q_rad.ucon[0];
			#endif
		}

		#if(TWO_T)
		U_ft[ENTRE] = pb_i[ENTRE] * U_ft[RHO];
		U_ft[ENTRI] = pb_i[ENTRI] * U_ft[RHO];
		#endif

		//Set final quantitities
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
}
//Calculate initial error for source term and set initial guess values
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int *pflag_rad
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	double norm, dK_dS;
	int k, pflag, do_entropy=0;
	struct of_state q;
	struct of_state_rad q_rad;
	#if(TWO_T)
	double gamma_g;// = calc_gamma_gas_prim(pb);
	#endif
	#if(P_NUM)
	double exp_xi;
	#endif

	//Set guess values for primitives after implicit step based on optical depth
	#if(NEWMAN)
	pflag = Utoprim_NM(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
	);
	#else
	pflag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
	);
	#endif
	#if(DO_FONT_FIX)
	if (pflag) {
		pflag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
		);
		#if(!TWO_T)
		if (pflag){
			pflag = Utoprim_1dfix1(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if(TWO_T)
			, fel
			#endif
			);
		}
		#endif
		if (!pflag) do_entropy = 1;
	}
	#endif	 

	//Even if MHD inversion fails, use updated value of radiation variable as gues
	pflag_rad[0] = Rtoprim(U_i, geom->gcov, geom->gcon, geom->g, pb, y_max, TYPE2
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);	
	if (pflag_rad[0]) {
		for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_i[k];
		#if(P_NUM)
		U_prev[PHOTON] = U_i[PHOTON];
		#endif
	}

	//Set electron entropy variables after inversion; Apply heating only if primary (energy based) inversion succeeds; Otherwise assume adiabatic evolution of electrons
	#if(TWO_T)
	//if (pflag == 0) {
		U_i[ENTRE] = pb[ENTRE] * U_i[RHO];
		U_i[ENTRI] = pb[ENTRI] * U_i[RHO];

		//Set for 2T fluid entropy of ions based on electron entropy
		/*#if(VARGAMMA)
		double ue, ui;
		double Theta, gam, C;

		//Calculate ue
			#if(FULL_ENTROPY_VARGAMMA)
			Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb[RHO] * exp(pb[ENTRE])), 2. / 3.)) - 1.0));
			#else
			Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb[RHO] * pb[ENTRE]), 2. / 3.)) - 1.0));
			#endif
		gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		ue = Theta / (MU_E * MASS_RATIO) * pb[RHO] / (gam - 1.0);

		//Check limits
		if (ue > (1.0 - FLOOR_ENTROPY) * pb[UU]) ue = (1.0 - FLOOR_ENTROPY) * pb[UU];
		if (ue < FLOOR_ENTROPY * pb[UU]) ue = FLOOR_ENTROPY * pb[UU];
		ui = pb[UU] - ue;

		//Set electron entropy
		C = ue / pb[RHO] * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pb[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO]);
			#else
			pb[ENTRE] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO];
			#endif

		//Set ion entropy
		C = ui / pb[RHO] * MU_I;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pb[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO]);
			#else
			pb[ENTRI] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO];
			#endif
		get_state(pb, geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		U_i[ENTRE] = geom->g * pb[RHO] * q.ucon[0] * pb[ENTRE];
		U_i[ENTRI] = geom->g * pb[RHO] * q.ucon[0] * pb[ENTRI];
		#endif	*/
	//}
	U_f[ENTRE] = U_i[ENTRE];
	U_f[ENTRI] = U_i[ENTRI];
	#endif

	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q
	#if(CALC_MDOT)
	,  magnetic_density_scale
	#endif
	);
	#if(TWO_T)
	gamma_g = calc_gamma_gas_prim(pb);
	#endif
	mhd_calc(pb, 0, &q, &U_f[UU]
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	for (k = UU; k <= U3; k++)U_f[k] *= geom->g;
	U_f[UU] += U_f[RHO];

	//Recompute entropy for consistency
	#if(DOKTOT)
	U_f[KTOT] = U_f[RHO] * calc_entropy(pb
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	U_i[KTOT] = U_f[KTOT];
	#endif

	//Reset inverted variables (both in case of success and failure)
	U_i[RHO] = U_f[RHO];
	U_i[UU] = U_f[UU];
	U_i[U1] = U_f[U1];
	U_i[U2] = U_f[U2];
	U_i[U3] = U_f[U3];

	//if (pflag_rad[0]) {
	//	pb[UU_RAD] = pb_old[UU_RAD];
	//	pb[U1_RAD] = pb_old[U1_RAD];
	//	pb[U2_RAD] = pb_old[U2_RAD];
	//	pb[U3_RAD] = pb_old[U3_RAD];
	//	pflag_rad[0] = 0;
	//}

	//Recompute R_t^mu for consistency
	get_state_rad(pb, geom, &q_rad);
	//mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
	//for (k = UU_RAD; k <= U3_RAD; k++) U_f[k] *= geom->g;
	U_f[UU_RAD] = U_i[UU_RAD];
	U_f[U1_RAD] = U_i[U1_RAD];
	U_f[U2_RAD] = U_i[U2_RAD];
	U_f[U3_RAD] = U_i[U3_RAD];

	//Recommpute photon number after inversion
	#if(P_NUM)
	//U_f[PHOTON] = geom->g * pb[PHOTON] * q_rad.ucon[0];
	U_f[PHOTON] = U_i[PHOTON];
	#endif

	//Calculate source term for U_i
	source_rad(pb, geom, &q, &q_rad, dU
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);

	//Calculate iterated error at start of iteration
	norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	if (do_entropy == 0) error_t[0] = 0.25 * (fabs(U_f[UU] - U_i[UU] - Dt * dU[UU]) / norm);
	else {
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMA - 1.) / pow(pb[RHO], GAMMA - 1.0);
			#elif(VARGAMMA)
			double Theta_i;
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO] * exp(pb[ENTRI]), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_i) * (MU_I);
				#else
				Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO], 2. / 3.) * fabs(pb[ENTRI])) - 1.0);
				dK_dS = 2. / 3. * (pb[ENTRI] / Theta_i) * (MU_I);
				#endif
			#endif
			error_t[0] = 0.25 * (fabs((U_f[ENTRI] - U_i[ENTRI] - Dt * dU[ENTRI]))) / (norm * dK_dS);
		#else
			#if(FULL_ENTROPY)
			dK_dS = pb[RHO] / ((GAMMA - 1.) * pb[UU]);
			#else
			dK_dS = (GAMMA - 1.) / pow(pb[RHO], GAMMA - 1.0);
			#endif
			error_t[0] = 0.25 * (fabs((U_f[KTOT] - U_i[KTOT] - Dt * dU[KTOT]))) / (norm * dK_dS);
		#endif
	}
	#if(TWO_T)
		#if(CONSTANTGAMMA || FIXEDGAMMA)
		dK_dS = (GAMMAE - 1.) / pow(pb[RHO], GAMMAE - 1.0);
		#elif(VARGAMMA)
		//Notes
		//Q = P * uu * dS;
		//Q = P * uu * 1 / kappa * dkappa;
		//Q = P / rho  * 1 / kappa * d(rho * uu * kappa);

		//For old entropy
		//Q = P * uu * rho ^ gamma / P * dkappa / (gamma - 1);
		//Q = rho ^ (gamma - 1.0) * d(kappa * rho * uu) / (gamma - 1);
		double Theta_e;
			//For variable entropy
			#if(FULL_ENTROPY_VARGAMMA)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO] * exp(pb[ENTRE]), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO], 2. / 3.) * fabs(pb[ENTRE])) - 1.0);
			dK_dS = 2. / 3. * (pb[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			#endif
		#endif
	error_t[0] += 0.25 * (fabs(U_f[ENTRE] - U_i[ENTRE] - Dt * dU[ENTRE]) / (dK_dS * norm));
	#endif
	#if(P_NUM)
	if (!pflag_rad[0]) {
		norm = (fabs(U_i[PHOTON]) + fabs(U_f[PHOTON]) + fabs(Dt * dU[PHOTON]));
		error_t[0] += 0.25 * (fabs(U_f[PHOTON] - U_i[PHOTON] - Dt * dU[PHOTON]) / norm);
	}
	else {
		norm = (fabs(U_i[PHOTON]) + fabs(U_prev[PHOTON]) + fabs(Dt * dU[PHOTON]));
		error_t[0] += 0.25 * (fabs(U_prev[PHOTON] - U_i[PHOTON] - Dt * dU[PHOTON]) / norm);
	}
	#endif
	norm = (fabs(sqrt(geom->gcon[4]) * U_i[U1]) + fabs(U_f[U1]) + fabs(Dt * dU[U1]));
	norm += (fabs(sqrt(geom->gcon[7]) * U_i[U2]) + fabs(U_f[U2]) + fabs(Dt * dU[U2]));
	norm += (fabs(sqrt(geom->gcon[9]) * U_i[U3]) + fabs(U_f[U3]) + fabs(Dt * dU[U3]));
	if(norm == 0.0) norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	error_t[0] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1] - U_i[U1] - Dt * dU[U1]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2] - U_i[U2] - Dt * dU[U2]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3] - U_i[U3] - Dt * dU[U3]) / norm);
	
	//Set total error to iterated error
	error_t[1] = error_t[0];

	//Calculate total error at start of iteration
	if (do_entropy == 0) {
		if (!pflag_rad[0]) {
			norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
			error_t[1] += 0.25 * (fabs(U_f[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU]) / norm);
		}
		else {
			norm = (fabs(U_i[UU_RAD]) + fabs(U_prev[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
			error_t[1] += 0.25 * (fabs(U_prev[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU]) / norm);
		}
	}
	norm = (fabs(sqrt(geom->gcon[4]) * U_i[U1_RAD]) + fabs(U_f[U1_RAD]) + fabs(Dt * dU[U1_RAD]));
	norm += (fabs(sqrt(geom->gcon[7]) * U_i[U2_RAD]) + fabs(U_f[U2_RAD]) + fabs(Dt * dU[U2_RAD]));
	norm += (fabs(sqrt(geom->gcon[9]) * U_i[U3_RAD]) + fabs(U_f[U3_RAD]) + fabs(Dt * dU[U3_RAD]));
	if(norm == 0.0) norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
	error_t[1] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1_RAD] - U_i[U1_RAD] - Dt * dU[U1_RAD]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2_RAD] - U_i[U2_RAD] - Dt * dU[U2_RAD]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3_RAD] - U_i[U3_RAD] - Dt * dU[U3_RAD]) / norm);
}

__device__ int implicit_rad_solve_PMHD_fast(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR], U_old_prev[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[1+TWO_T+P_NUM], dpb,  dEdpb_inv[1 + TWO_T + P_NUM][1 + TWO_T + P_NUM], error_new[5*2], offset = 1.e-9;
	double dK_dS, norm, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad=0, count_increase = 0, count_increase2 = 0;	
	#if(TWO_T)
	int flag_floor_kappa;
	double gamma_g, ue, ui;
		#if(!CONSTANTGAMMA)
		double Theta_e, Theta_i;
		#endif
	#else
	double T_GAS;
	#endif

	//Set error to previous value
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = dU[k];
		U_new[k] = U_old[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		E_old[1] = (1.0 / dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[1 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRI])), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			for (i = 0; i < 1 + TWO_T + P_NUM; i++) {
				PLOOP pb_new[k] = pb_old[k];
				#if(TWO_T)
				U_new[ENTRE] = U_old[ENTRE];
				#endif
				#if(P_NUM)
				U_new[PHOTON] = U_old[PHOTON];
				#endif
				if (i == 0) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[UU] = pb_old[UU] + dpb;
				}
				#if(TWO_T)
				else if (i == 1) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dpb;
				}
				#endif
				#if(P_NUM)
				else if (i == 1 + TWO_T) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dpb;
				}
				#endif

				// Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				#if(TWO_T)
					pb_new[ENTRE] = U_new[ENTRE] / U_new[RHO];
					//Set for 2T fluid entropy of ions based on electron entropy
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
					ui = pb_new[UU] - ue;
					pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
					#elif(VARGAMMA)
					double Theta, gam, C;
					
					//Calculate ue
						#if(FULL_ENTROPY_VARGAMMA)
						Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRE])), 2. / 3.)) - 1.0));
						#else
						Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0));
						#endif
					gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
					ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

					//Check limits
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					ui = pb_new[UU] - ue;

					//Set electron entropy
					C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
						#if(FULL_ENTROPY_VARGAMMA)
						pb_new[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
						#else
						pb_new[ENTRE] = (Theta) * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
						#endif

					//Set ion entropy
					C = ui / pb_new[RHO] * MU_I;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
						#if(FULL_ENTROPY_VARGAMMA)
						pb_new[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
						#else
						pb_new[ENTRI] = (Theta) * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
						#endif
					#endif
				//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
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

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				//Invert radiation variables
				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number for consistency
				//#if(P_NUM)
				//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Calculate radiative (including coulomb) source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Calculate Jacobian
				#if(TWO_T)
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
						//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				E_new[1] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
				dEdpb_inv[1][i] = (E_new[1] - E_old[1]) / dpb;
				#endif
				#if(P_NUM)
				E_new[1 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
				dEdpb_inv[1 + TWO_T][i] = (E_new[1 + TWO_T] - E_old[1 + TWO_T]) / dpb;
				#endif
				if (do_entropy == 1) {
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRI])), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
					#else
						#if(FULL_ENTROPY)
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						#else
						T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
						#endif
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					#endif
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb_inv[0][i] = (E_new[0] - E_old[0]) / dpb;
			}

			//Invert Jacobian
			#if(P_NUM && TWO_T)
			flag = invert_matrix_3D(dEdpb_inv, dEdpb_inv);
			#elif(P_NUM || TWO_T)
			flag = invert_matrix_2D(dEdpb_inv, dEdpb_inv);
			#else
			flag = invert_matrix_1D(dEdpb_inv, dEdpb_inv);
			#endif

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;
		
		//Set primitive variables before Newton step
		PLOOP pb_new[k] = pb_old[k];

		/* Make the newton step: */
		D = 1.0;
		dpb = -D * (E_old[0] * dEdpb_inv[0][0]
			#if(TWO_T)
				+ E_old[1] * dEdpb_inv[0][1]
			#endif
			#if(P_NUM)
			+ E_old[1 + TWO_T] * dEdpb_inv[0][1 + TWO_T]
			#endif
			);
		pb_new[UU] = pb_old[UU] + dpb;
		#if(TWO_T)
		dpb = -D * (E_old[0] * dEdpb_inv[1][0] + E_old[1] * dEdpb_inv[1][1]
			#if(P_NUM)
			+ E_old[2] * dEdpb_inv[1][2]
			#endif	
			);
		U_new[ENTRE] = U_old[ENTRE] + dpb;
		#endif
		#if(P_NUM)
		dpb = -D * (E_old[0] * dEdpb_inv[1 + TWO_T][0] + E_old[1] * dEdpb_inv[1 + TWO_T][1]
			#if(TWO_T)
			+ E_old[1 + TWO_T] * dEdpb_inv[1 + TWO_T][1 + TWO_T]
			#endif			
			);
		U_new[PHOTON] = U_old[PHOTON] + dpb;
		#endif

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);
		
		//Make sure that electron entropy stays positive
		#if(TWO_T)
		//if (U_new[ENTRE] < 0.0) U_new[ENTRE] = 0.5 * fabs(U_new[ENTRE]);
		#endif
		
		//Make sure that photon number stays positive
		#if(P_NUM)
		//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
		#endif

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		#if(TWO_T)
			pb_new[ENTRE] = U_new[ENTRE] / U_new[RHO];
			flag_floor_kappa = 0;
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
			ui = pb_new[UU] - ue;
			pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
			#elif(VARGAMMA)
			double Theta, gam, C;

			//Calculate ue
				#if(FULL_ENTROPY_VARGAMMA)
				Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRE])), 2. / 3.)) - 1.0));
				#else
				Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0));
				#endif
			gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
			ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			ui = pb_new[UU] - ue;
		
			//Set electron entropy
			C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				pb_new[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
				#else
				pb_new[ENTRE] = Theta * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
				#endif

			//Set ion entropy
			C = ui / pb_new[RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				pb_new[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
				#else
				pb_new[ENTRI] = Theta * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
				#endif
			#endif
		//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
		U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
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

		//Recalculate gas entropy for consistency
		#if(DOKTOT)
		U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives using TYPE2 limiter
		flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		if (flag_rad) PLOOP U_old_prev[k] = U_new[k];		
		
		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Recompute photon number
		//#if(P_NUM)
		//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
		//#endif

		//Get radiative source term
		source_rad(pb_new, geom, &q, &q_rad, dU_new
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);

		//Calculate iterated error
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		if (do_entropy == 0)error_new[n_iter % 5] = 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
		else {
			#if(TWO_T)
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRI])), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * pb_new[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				error_new[n_iter % 5] = 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
			#else
				#if(FULL_ENTROPY)
				dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
				error_new[n_iter % 5] = 0.25 * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm * dK_dS);
				#else
				dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
				error_new[n_iter % 5] = 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
				#endif
			#endif
		}
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
			double Theta_e;
			//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		if(flag_floor_kappa==0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (norm * dK_dS));
		#endif
		#if(P_NUM)
		if (flag_rad == 0) {
			norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
			error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		}
		else {
			norm = (fabs(U_i[PHOTON]) + fabs(U_old_prev[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
			error_new[n_iter % 5] += 0.25 * (fabs(U_old_prev[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		}
		#endif

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10 ;
		else offset = 1.e-9;

		//Set total error to iterated error
		error_new[n_iter % 5 + 5] = error_new[n_iter % 5];
		if (do_entropy == 0) {
			if (flag_rad == 0) {
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
			}
			else {
				norm = (fabs(U_i[UU_RAD]) + fabs(U_old_prev[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_old_prev[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
			}
		}

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12)|| (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If residual drops below bound exit
		//double residual = 0.;
		//for (k = 0; k < NPR; k++)residual += fabs(pb_new[k]/pb_old[k]-1.0);
		//if (residual<1.0e-15) {
			//keep_iterating = 0;
		//}

		//If total error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
			keep_iterating = 0;
		}

		//If iterated error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			//keep_iterating = 0;
		}

		//If total error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		//If iterated error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase2++;
			//if (count_increase2 >= 5) keep_iterating = 0;
		}

		//If error decreased compared to start value, update variables
		if (((fabs(error_new[n_iter % 5 + 5]) < error_t[1])) && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
			error_t[0] = error_new[n_iter % 5];
			error_t[1] = error_new[n_iter % 5 + 5];

			//Reset radiation inversion error
			if (flag_rad) {
				for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_old_prev[k];
				#if(P_NUM)
				U_prev[PHOTON] = U_old_prev[PHOTON];
				#endif
			}
			pflag_rad[0] = flag_rad;

			//Update variables
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
		}

		//Reset variables if Newton step succesfull
		if (keep_iterating) {
			for (k = 0; k < NPR; k++) {
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}
		}
		n_iter++;
	}

	return(0);
}

__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR], U_old_prev[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb,  dEdpb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[5*2], offset = 1.e-9;
	double dK_dS, norm, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad=0, count_increase = 0, count_increase2 = 0;
	#if(TWO_T)
	int flag_floor_kappa;
	double gamma_g, ue, ui;
		#if(VARGAMMA)
		double Theta_i, Theta_e;
		#endif
	#else
	double T_GAS;
	#endif

	//Set error to previous value
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

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
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		E_old[4] = (1.0 / dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRI])), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP pb_new[k] = pb_old[k];
				#if(TWO_T)
				U_new[ENTRE] = U_old[ENTRE];
				#endif
				#if(P_NUM)
				U_new[PHOTON] = U_old[PHOTON];
				#endif
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb_old[i] + dpb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dpb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dpb;
				}
				#endif
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					pb_new[i] = pb_old[i] + dpb;
				}

				// Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				#if(TWO_T)
					pb_new[ENTRE] = U_new[ENTRE] / U_new[RHO];
					//Set for 2T fluid entropy of ions based on electron entropy
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
					ui = pb_new[UU] - ue;
					pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
					#elif(VARGAMMA)
					double Theta, gam, C;
					
					//Calculate ue
						#if(FULL_ENTROPY_VARGAMMA)
						Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRE])), 2. / 3.)) - 1.0));
						#else
						Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0));
						#endif
					gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
					ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

					//Check limits
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					ui = pb_new[UU] - ue;

					//Set electron entropy
					C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
						#if(FULL_ENTROPY_VARGAMMA)
						pb_new[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
						#else
						pb_new[ENTRE] = (Theta) * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
						#endif

					//Set ion entropy
					C = ui / pb_new[RHO] * MU_I;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
						#if(FULL_ENTROPY_VARGAMMA)
						pb_new[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
						#else
						pb_new[ENTRI] = (Theta) * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
						#endif
					#endif
				//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
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

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				//Invert radiation variables
				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number for consistency
				//#if(P_NUM)
				//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Calculate radiative (including coulomb) source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Calculate Jacobian
				for (k = U1; k <= U3; k++) {
					E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdpb_inv[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb;
				}
				#if(TWO_T)
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
						//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				E_new[4] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
				dEdpb_inv[4][i - UU] = (E_new[4] - E_old[4]) / dpb;
				#endif
				#if(P_NUM)
				E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
				dEdpb_inv[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dpb;
				#endif
				if (do_entropy == 1) {
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRI])), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
					#else
						#if(FULL_ENTROPY)
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						#else
						T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
						#endif
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					#endif
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb_inv[0][i - UU] = (E_new[0] - E_old[0]) / dpb;
			}

			//Invert Jacobian
			#if(P_NUM && TWO_T)
			flag = invert_matrix_6D(dEdpb_inv, dEdpb_inv);
			#elif(P_NUM || TWO_T)
			flag = invert_matrix_5D(dEdpb_inv, dEdpb_inv);
			#else
			flag = invert_matrix_4D(dEdpb_inv, dEdpb_inv);
			#endif

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;
		
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
						+ E_old[4] * dEdpb_inv[k][4]
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
						+ E_old[4] * dEdpb_inv[k][4]
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
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
		}
		#if(TWO_T)
		dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
			#if(P_NUM)
			+ E_old[4 + P_NUM] * dEdpb_inv[4][4 + P_NUM]
			#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dpb;
		#endif
		#if(P_NUM)
		dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4] * dEdpb_inv[4 + TWO_T][4]
			#if(TWO_T)
			+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
			#endif			
			);
		U_new[PHOTON] = U_old[PHOTON] + dpb;
		#endif

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);
		
		//Make sure that electron entropy stays positive
		#if(TWO_T)
		//if (U_new[ENTRE] < 0.0) U_new[ENTRE] = 0.5 * fabs(U_new[ENTRE]);
		#endif
		
		//Make sure that photon number stays positive
		#if(P_NUM)
		//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
		#endif

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		#if(TWO_T)
			pb_new[ENTRE] = U_new[ENTRE] / U_new[RHO];
			flag_floor_kappa = 0;
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
			ui = pb_new[UU] - ue;
			pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
			#elif(VARGAMMA)
			double Theta, gam, C;

			//Calculate ue
				#if(FULL_ENTROPY_VARGAMMA)
				Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRE])), 2. / 3.)) - 1.0));
				#else
				Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0));
				#endif
			gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
			ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				//flag_floor_kappa = 1;
			}
			ui = pb_new[UU] - ue;
		
			//Set electron entropy
			C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				pb_new[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
				#else
				pb_new[ENTRE] = Theta * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
				#endif

			//Set ion entropy
			C = ui / pb_new[RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY_VARGAMMA)
				pb_new[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO]);
				#else
				pb_new[ENTRI] = Theta * (Theta + 0.4) / pow(pb_new[RHO], 2. / 3.);
				#endif
			#endif
		//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
		U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
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

		//Recalculate gas entropy for consistency
		#if(DOKTOT)
		U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives using TYPE2 limiter
		flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);
		if (flag_rad) PLOOP U_old_prev[k] = U_new[k];		
		
		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Recompute photon number
		//#if(P_NUM)
		//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
		//#endif

		//Get radiative source term
		source_rad(pb_new, geom, &q, &q_rad, dU_new
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
			#if(COOL_STOP)
			, r
			#endif
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);

		//Calculate iterated error
		norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
		norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
		norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
		if(norm == 0.0)	norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
		else {
			#if(TWO_T)
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * exp(pb_new[ENTRI])), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_new[RHO] * pb_new[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				error_new[n_iter % 5] += 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
			#else
				#if(FULL_ENTROPY)
				dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm * dK_dS);
				#else
				dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
				error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
				#endif
			#endif
		}
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
			double Theta_e;
			//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		if(flag_floor_kappa==0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (norm * dK_dS));
		#endif
		#if(P_NUM)
		if (flag_rad == 0) {
			norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
			error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		}
		else {
			norm = (fabs(U_i[PHOTON]) + fabs(U_old_prev[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
			error_new[n_iter % 5] += 0.25 * (fabs(U_old_prev[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		}
		#endif

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10 ;
		else offset = 1.e-9;

		//Set total error to iterated error
		error_new[n_iter % 5 + 5] = error_new[n_iter % 5];
		if (do_entropy == 0) {
			if (flag_rad == 0) {
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
			}
			else {
				norm = (fabs(U_i[UU_RAD]) + fabs(U_old_prev[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_old_prev[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
			}
		}
		norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
		norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
		norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
		if (norm == 0.0) norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
		//double bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12)|| (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If residual drops below bound exit
		//double residual = 0.;
		//for (k = 0; k < NPR; k++)residual += fabs(pb_new[k]/pb_old[k]-1.0);
		//if (residual<1.0e-15) {
			//keep_iterating = 0;
		//}

		//If total error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
			keep_iterating = 0;
		}

		//If iterated error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			//keep_iterating = 0;
		}

		//If total error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		//If iterated error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase2++;
			//if (count_increase2 >= 5) keep_iterating = 0;
		}

		//If error decreased compared to start value, update variables
		if (((fabs(error_new[n_iter % 5 + 5]) < error_t[1])) && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
			error_t[0] = error_new[n_iter % 5];
			error_t[1] = error_new[n_iter % 5 + 5];

			//Reset radiation inversion error
			if (flag_rad) {
				for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_old_prev[k];
				#if(P_NUM)
				U_prev[PHOTON] = U_old_prev[PHOTON];
				#endif
			}
			pflag_rad[0] = flag_rad;

			//Update variables
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
		}

		//Reset variables if Newton step succesfull
		if (keep_iterating) {
			for (k = 0; k < NPR; k++) {
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates T^t_mu
__device__ int implicit_rad_solve_UMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double norm, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad=0, count_increase = 0;
	#if(TWO_T)
	double gamma_g;
		#if(VARGAMMA)
		 double Theta_i, Theta_e;
		#endif
	#else
	double T_GAS;
	#endif

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		E_old[4] = (1.0/ dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO] * exp(pb_old[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]);
					U_new[i] = U_old[i] + dUb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					U_new[i] = U_old[i] + dUb;
				}

				//Set entropy based on values in previous iteration
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				//Invert gas conserved to gas primitives
				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);
				#if(DO_FONT_FIX)
				if (flag && (n_iter_jacob > 1)) {
					//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2, FULL_ENTROPY
					//#if (DOHELM)
					//, gpu_eos_table
					//#endif
					//#if(TWO_T)
					//, 0.0
					//#endif
					//);
				}
				#endif

				if (flag == 0) {
					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q
					#if(CALC_MDOT)
					, magnetic_density_scale
					#endif
					);
					#if(TWO_T)
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
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif

					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Set radiation conserved quantities
					U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
					U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
					U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
					U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

					//Invert radiation conserved variables to primitives
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					//#if(P_NUM)
					//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					//#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
							#else
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
							#endif
						#endif
					E_new[4] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(TWO_T)
							#if(CONSTANTGAMMA || FIXEDGAMMA)
							dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
							#elif(VARGAMMA)
								//For variable entropy
								#if(FULL_ENTROPY_VARGAMMA)
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
								dK_dS = (1.0 / Theta_i) * (MU_I);
								#else
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
								dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
								#endif
							#endif
						E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
						#else
							#if(FULL_ENTROPY)
							T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
							#else
							T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
							#endif
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
						#endif
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;
				}
			}

			//Invert Jacobian
			if(flag==0){
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif	
			}
			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
						);
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
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Invert gas conserved quantities to primitives
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);
			#if(DO_FONT_FIX)
			if (flag && (n_iter_fail > 1)) {
				//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2, FULL_ENTROPY
				//#if (DOHELM)
				//, gpu_eos_table
				//#endif
				//#if(TWO_T)
				//, 0.0
				//#endif
				//);
			}
			#endif

			if (flag == 0) {
				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

				//Make sure that the photon number stays positive
				#if(P_NUM)
				//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				#if(TWO_T)
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
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Derive new conserved quantaties for MHD variables
				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number
				//#if(P_NUM)
				//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				if (norm == 0.0) norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
					#else
						#if(FULL_ENTROPY)
						dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
						error_new[n_iter % 5] += 0.25 * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm * dK_dS);
						#else
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
						#endif
					#endif
				}
				#if(TWO_T)
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10;
				else offset = 1.e-8;

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
					error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				if (norm == 0.0) norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
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
				if (((fabs(error_new[n_iter % 5 + 5]) < error_t[1]) || (pflag_rad[0]==1 && flag_rad==0)) && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];

					if (flag_rad) {
						U_prev[UU_RAD] = U_new[UU_RAD];
						U_prev[U1_RAD] = U_new[U1_RAD];
						U_prev[U2_RAD] = U_new[U2_RAD];
						U_prev[U3_RAD] = U_new[U3_RAD];
						#if(P_NUM)
						U_prev[PHOTON] = U_new[PHOTON];
						#endif
					}
					pflag_rad[0] = flag_rad;

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
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}


// This method iterates Su^t and T^t_i
__device__ int implicit_rad_solve_EMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR],  pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM],  error_new[10], offset = pow(10., -8.);
	double  norm, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, count_increase = 0,  flag_rad =0;
	#if(TWO_T)
	double gamma_g;
		#if(VARGAMMA)
		 double Theta_i, Theta_e;
		#endif
	#else
	double T_GAS;
	#endif

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

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
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
				dK_dS = 2. / 3. * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		E_old[4] = (1.0 / dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);		
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO] * exp(pb_old[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					#if(TWO_T)
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRI]);
					U_new[ENTRI] = U_old[ENTRI] + dUb;
					#else
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[KTOT]);
					U_new[KTOT] = U_old[KTOT] + dUb;
					#endif
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					U_new[i] = U_old[i] + dUb;
				}

				//Invert conserved MHD quantities using entropy based methods
				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC, FULL_ENTROPY		
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);

				if (flag == 0) {
					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q
					#if(CALC_MDOT)
					, magnetic_density_scale
					#endif
					);
					#if(TWO_T)
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
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif

					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Set radiation conserved quantities
					U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
					U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
					U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
					U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					//#if(P_NUM)
					//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					//#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
							#else
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
							#endif
						#endif
					E_new[4] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(TWO_T)
							#if(CONSTANTGAMMA || FIXEDGAMMA)
							dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
							#elif(VARGAMMA)
								//For variable entropy
								#if(FULL_ENTROPY_VARGAMMA)
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
								dK_dS = (1.0 / Theta_i) * (MU_I);
								#else
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
								dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
								#endif
							#endif
						E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
						#else
							#if(FULL_ENTROPY)
							T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
							#else
							T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
							#endif
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
						#endif
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
					);
					#if(TWO_T)
					if (k == 0) U_new[ENTRI] = U_old[ENTRI] + dUb;
					else U_new[k + UU] = U_old[k + UU] + dUb;
					#else
					if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
					else U_new[k + UU] = U_old[k + UU] + dUb;
					#endif
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						#if(TWO_T)
						if (k == 0) U_new[ENTRI] = U_old[ENTRI] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#else
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#endif
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						#if(TWO_T)
						if (k == 0) U_new[ENTRI] = U_old[ENTRI] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#else
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#endif
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						#if(TWO_T)
						if (k == 0) U_new[ENTRI] = U_old[ENTRI] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#else
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
						#endif
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Invert conserved MHD quantities using entropy based methods
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC, FULL_ENTROPY
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);

			if (flag == 0) {
				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

				//Make sure that the photon number stays positive
				#if(P_NUM)
				//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				#if(TWO_T)
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
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Derive new conserved quantaties for MHD variables
				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Compute photon number
				//#if(P_NUM)
				///U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				if (norm == 0.0) norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
					#else
						#if(FULL_ENTROPY)
						dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
						error_new[n_iter % 5] += 0.25 * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm * dK_dS);
						#else
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
						#endif
					#endif
				}
				#if(TWO_T)
					#if(CONSTANTGAMMA || FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error	
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
					error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				if (norm == 0.0) norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
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
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];

					if (flag_rad) {
						for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_new[k];
						#if(P_NUM)
						U_prev[PHOTON] = U_new[PHOTON];
						#endif
					}
					pflag_rad[0] = flag_rad;

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
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates R^t_mu
__device__ int implicit_rad_solve_URAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], U_prev_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double norm, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;
	#if(TWO_T)
	double gamma_g;
		#if(!CONSTANTGAMMA)
		double Theta_e, Theta_i;
		#endif
	#else
	double T_GAS;
	#endif

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
		dU_new[k] = dU[k];
	}

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
					#else
					Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
					#endif
				#endif
		E_old[4] = (1.0 / dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO] * exp(pb_old[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = 2. / 3. * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU_RAD; i <= U3_RAD + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU_RAD) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]);
					U_new[i] = U_old[i] + dUb;
				}
				#if(TWO_T)
				else if (i == U3_RAD + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3_RAD + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]) * sqrt(geom->gcov[4 * (i == U1_RAD) + 7 * (i == U2_RAD) + 9 * (i == U3_RAD)]);
					U_new[i] = U_old[i] + dUb;
				}

				U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
				U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
				U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
				U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC			
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);

				if (flag == 0) {
					//Invert radiation conserved quantities
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					//#if(P_NUM)
					//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					//#endif

					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q
					#if(CALC_MDOT)
					, magnetic_density_scale
					#endif
					);
					#if(TWO_T)
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
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif
	
					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
						#if(CALC_MDOT)
						, mass_density_scale, magnetic_density_scale
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1_RAD; k <= U3_RAD; k++) {
						E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dUb;
					}
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
							#else
							Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
							#endif
						#endif
					E_new[4] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU_RAD] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU_RAD] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(TWO_T)
							#if(CONSTANTGAMMA || FIXEDGAMMA)
							dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
							#elif(VARGAMMA)
								//For variable entropy
								#if(FULL_ENTROPY_VARGAMMA)
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
								dK_dS = (1.0 / Theta_i) * (MU_I);
								#else
								Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
								dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
								#endif
							#endif
						E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
						#else
							#if(FULL_ENTROPY)
							T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
							#else
							T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
							#endif
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
						#endif
					}
					else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
					dEdUb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dUb;
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
						);
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
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Derive new conserved quantaties for MHD variables
			U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
			U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
			U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
			U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Invert updated conserved quantities
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);

			if (flag == 0) {
				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);
				PLOOP U_prev_old[k] = U_new[k];

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				//mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				//for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number
				//#if(P_NUM)
				//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Make sure that the photon number stays positive
				#if(P_NUM)
				//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				#if(TWO_T)
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
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);
			
				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				if(norm == 0.0)	norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
					error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				}
				else {
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = 2. / 3. * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
					#else
						#if(FULL_ENTROPY)
						dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
						#else
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
					#endif
				}
				#if(TWO_T)
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					#if(CONSTANTGAMMA)
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					#elif(FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = (2. / 3.) * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				if(norm == 0.0) norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//If gas negative more than 2 times stop iterating
				//if (pb_new[UU] < 0.) {
				//	count_increase_gas++;
					//if (count_increase > 2) keep_iterating = 0;
				//}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];

					//Reset radiation inversion error
					if (flag_rad) {
						for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_prev_old[k];
						#if(P_NUM)
						U_prev[PHOTON] = U_prev_old[PHOTON];
						#endif
					}
					pflag_rad[0] = flag_rad;

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
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}
	return(0);
}

// This method iterates E_RAD an U_rad
__device__ int implicit_rad_solve_PRAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
#if(CALC_MDOT)
, double mass_density_scale, double magnetic_density_scale
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdpb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double norm, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, flag, n_iter_jacob, count_increase = 0;
	#if(TWO_T)
	double gamma_g;
		#if(!CONSTANTGAMMA)
		double Theta_e, Theta_i;
		#endif
	#else
	double T_GAS;
	#endif

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
		dU_new[k] = dU[k];
	}

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Set reference error at start
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
			#if(CONSTANTGAMMA || FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_old[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
				//For variable entropy
				#if(FULL_ENTROPY_VARGAMMA)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pb_old[RHO] * exp(pb_old[ENTRE])), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRE])) - 1.0);
				dK_dS = (2. / 3.) * (pb_old[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		E_old[4] = (1.0 / dK_dS) * (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(TWO_T)	
				#if(CONSTANTGAMMA || FIXEDGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
				#elif(VARGAMMA)
					//For variable entropy
					#if(FULL_ENTROPY_VARGAMMA)
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO] * exp(pb_old[ENTRI]), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_i) * (MU_I);
					#else
					Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_old[RHO], 2. / 3.) * fabs(pb_old[ENTRI])) - 1.0);
					dK_dS = (2. / 3.) * (pb_old[ENTRI] / Theta_i) * (MU_I);
					#endif
				#endif
				E_old[0] = (1.0 / dK_dS) * (U_old[ENTRI] - U_i[ENTRI] - Dt * dU_old[ENTRI]);
			#else
				#if(FULL_ENTROPY)
				T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
				#else
				T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
				#endif			
				E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
			#endif
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU_RAD; i <= U3_RAD + TWO_T + P_NUM; i++) {
				if(flag == 0){
					PLOOP{
						pb_new[k] = pb_old[k];
						U_new[k] = U_old[k];
					}
					if (i == UU_RAD) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU_RAD]);
						pb_new[i] = pb_old[i] + dpb;
					}
					#if(TWO_T)
					else if (i == U3_RAD + TWO_T) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
						U_new[ENTRE] = U_old[ENTRE] + dpb;
					}
					#endif
					#if(P_NUM)
					else if (i == U3_RAD + TWO_T + P_NUM) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
						U_new[PHOTON] = U_old[PHOTON] + dpb;
					}
					#endif
					else {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[4 * (i == U1_RAD) + 7 * (i == U2_RAD) + 9 * (i == U3_RAD)]);
						pb_new[i] = pb_old[i] + dpb;
					}

					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++) U_new[k] *= geom->g;
					#if(P_NUM)
					pb_new[PHOTON] = (1.0 / geom->g) * U_new[PHOTON] / q_rad.ucon[0];
					#endif

					U_new[RHO] = U_i[RHO];
					U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
					U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
					U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
					U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
					U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

					tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
					flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, 0.0
						#endif
					);

					if (flag == 0) {
						//Recompute T_t^mu for consistency
						U_new[RHO] = U_i[RHO];
						get_state(pb_new, geom, &q
						#if(CALC_MDOT)
						, magnetic_density_scale
						#endif
						);
						#if(TWO_T)
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
						U_new[UU] += U_new[RHO];

						//Electron and ion entropies
						#if(TWO_T)
						//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
						U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
						#endif
	
						//Recalculate gas entropy for consistency
						#if(DOKTOT)
						U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
							#if (DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, gamma_g
							#endif
						);
						#endif

						//Recompute photon number
						//#if(P_NUM)
						//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
						//#endif

						//Calculate source function and jacobian
						source_rad(pb_new, geom, &q, &q_rad, dU_new
							#if(DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, gamma_g
							#endif
							#if(COOL_STOP)
							, r
							#endif
							#if(CALC_MDOT)
							, mass_density_scale, magnetic_density_scale
							#endif
						);

						//Calculate Jacobian
						for (k = U1_RAD; k <= U3_RAD; k++) {
							E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
							dEdpb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dpb;
						}
						#if(TWO_T)
							#if(CONSTANTGAMMA || FIXEDGAMMA)
							dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
							#elif(VARGAMMA)
								//For variable entropy
								#if(FULL_ENTROPY_VARGAMMA)
								Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
								dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
								#else
								Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
								dK_dS = (2. / 3.) * (pb_new[ENTRE] / Theta_e) * (MU_E*MASS_RATIO);
								#endif
							#endif
						E_new[4] = (1.0 / dK_dS) * (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
						dEdpb[4][i - UU_RAD] = (E_new[4] - E_old[4]) / dpb;
						#endif
						#if(P_NUM)
						E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
						dEdpb[4 + TWO_T][i - UU_RAD] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dpb;
						#endif
						if (do_entropy == 1) {
							#if(TWO_T)
								#if(CONSTANTGAMMA || FIXEDGAMMA)
								dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
								#elif(VARGAMMA)
									//For variable entropy
									#if(FULL_ENTROPY_VARGAMMA)
									Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
									dK_dS = (1.0 / Theta_i) * (MU_I);
									#else
									Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
									dK_dS = (2. / 3.) * (pb_new[ENTRI] / Theta_i) * (MU_I);
									#endif
								#endif
							E_new[0] = (1.0 / dK_dS) * (U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]);					
							#else
								#if(FULL_ENTROPY)
								T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
								#else
								T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
								#endif
							E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
							#endif
						}
						else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
						dEdpb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dpb;
					}
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdpb, dEdpb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdpb, dEdpb_inv);
				#else
				flag = invert_matrix_4D(dEdpb, dEdpb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1. / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
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
						dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
				if (n_iter / 4 == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dpb = -D * (E_old[0] * dEdpb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif	
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
			}

			#if(TWO_T)
			dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdpb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dpb;
			#endif
			#if(P_NUM)
			dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4] * dEdpb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dpb;
			#endif

			//Make sure that radiation internal energy stays positive
			if (pb_new[UU_RAD] < 0.0) pb_new[UU_RAD] = 0.5 * fabs(pb_new[UU_RAD]);

			//Make sure that electron entropy stays positive
			//#if(TWO_T)
			//if (U_new[ENTRE] < 0.0) U_new[ENTRE] = 0.5 * fabs(U_new[ENTRE]);
			//#endif

			//Make sure that photon number stays positive
			//if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);

			//Obtain new radiation conserved quantaties from radiation primitive variables
			get_state_rad(pb_new, geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

			//Obtain photon number primitive quantity
			#if(P_NUM)
			pb_new[PHOTON] = (1.0 / geom->g) * U_new[PHOTON] / q_rad.ucon[0];
			#endif

			//Derive new MHD conserved quantaties for radiation variables
			U_new[RHO] = U_i[RHO];
			U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
			U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
			U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
			U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

			//Estimate conserved entropy
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Get MHD primitives
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);

			if (flag == 0) {
				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				#if(TWO_T)
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
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				//U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif
	
				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Recompute photon number
				//#if(P_NUM)
				//U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				//#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
					#if(CALC_MDOT)
					, mass_density_scale, magnetic_density_scale
					#endif
				);

				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				if(norm == 0.0)	norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
					error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				}
				else {
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					#if(TWO_T)
						#if(CONSTANTGAMMA || FIXEDGAMMA)
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#elif(VARGAMMA)
							//For variable entropy
							#if(FULL_ENTROPY_VARGAMMA)
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRI]), 2. / 3.)) - 1.0);
							dK_dS = (1.0 / Theta_i) * (MU_I);
							#else
							Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRI])) - 1.0);
							dK_dS = (2. / 3.) * (pb_new[ENTRI] / Theta_i) * (MU_I);
							#endif
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[ENTRI] - U_i[ENTRI] - Dt * dU_new[ENTRI]))) / (norm * dK_dS);
					#else
						#if(FULL_ENTROPY)
						dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
						#else
						dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
						#endif
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
					#endif
				}
				#if(TWO_T)
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					#if(CONSTANTGAMMA)
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					#elif(FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY_VARGAMMA)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO], 2. / 3.) * fabs(pb_new[ENTRE])) - 1.0);
						dK_dS = (2. / 3.) * (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				if(norm == 0.0) norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//If gas negative more than 2 times stop iterating
				//if (pb_new[UU] < 0.) {
				//	count_increase_gas++;
				//	if (count_increase > 2) keep_iterating = 0;
				//}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];
					pflag_rad[0] = 0;
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
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}
	return(0);
}

//3D
//Inversion from radiation conserved to primitive quantities
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
){
	double U_tmp[NPR_R + P_NUM], prim_tmp[NPR_R+P_NUM];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

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
	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, y_max, lim
		#if(CALC_MDOT)
		, mass_density_scale, magnetic_density_scale
		#endif
	);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i < NPR_R; i++) {
		prim[i + UU_RAD] = prim_tmp[i];
	}
	#if(P_NUM)
	prim[PHOTON] = prim_tmp[4];
	#endif

	return(ret);
}

__device__ int Rtoprim_calc(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
) {
	double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f;
	int i, returnval = 0;
	#if(P_NUM)
	double Tr;
		#if(!CALC_MDOT)
		double energy_density_scale = MASS_DENSITY_SCALE * C_CGS * C_CGS;
		#else
		double energy_density_scale = mass_density_scale * C_CGS * C_CGS;
		#endif
	#endif

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise(Qcov, gcon, Qcon);

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

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

	#if(P_NUM)
	prim[4] = U[4] / sqrt(gammasq);
	#endif

	/*if (isnan(Qdotn) || Qdotn > 0.0) {
		prim[0] = 1.e-30;
		prim[1] = 0.;
		prim[2] = 0.;
		prim[3] = 0.;

		//Floor on photon number+
		#if(P_NUM)
		Tr = pow(prim[0] * energy_density_scale / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif

		returnval = 1;
	}*/
	if (y > y_max || y < 0. || isnan(Qdotn) || Qdotn > 0.0 || isnan(prim[1]) || isnan(prim[2]) || isnan(prim[3])) {
		Uabs = 0.5 * (fabs(Qdotn) + sqrt(fabs(Qtsq)) + 1.e-150);
		for (i = 1; i < 4; i++)prim[i] = GAMMAMAX_RAD * Qtcon[i] / Uabs;

		qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
			+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
		if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;

		//if (y < 1. - 100. * NUMEPSILON) {
		if (lim==TYPE2) {
			// Get Ebar and p_rad as usual
			//if (y > 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
			Qdotn = -(1.e-30 + sqrt(fabs(Qtsq) / y_max));
				
			//Get gammasq
			gammasq = (2. - y_max + sqrt(4. - 3. * y_max)) / (4. - 4. * y_max);

			// Get Ebar and p_rad as usual
			pressure = -Qdotn / (4. * gammasq - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad

			// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
			for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

			returnval = 1;
		}
		else {
			if (1) {
				prim[0] = 1.e-30;
				prim[1] = 0.;
				prim[2] = 0.;
				prim[3] = 0.;
			}
			else {
				pressure = fabs(Qdotn) / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
				prim[0] = pressure * 3.; // Erad = 3*p_rad
			}
		}
		if (!isfinite(prim[0]))prim[0] = 1.e-30;
		if (!isfinite(prim[1]))prim[1] = 0.0;
		if (!isfinite(prim[2]))prim[2] = 0.0;
		if (!isfinite(prim[3]))prim[3] = 0.0;

		//Floor on photon number+
		#if(P_NUM)
		Tr = pow(prim[0] * energy_density_scale / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif
	}
	if (prim[0] < 1.e-30) {
		//prim[0] = 1.e-30;
		//prim[1] = 0.0;
		//prim[2] = 0.0;
		//prim[3] = 0.0;
		//returnval = 1;
	}

	#if(P_NUM)
	if (prim[4] < 0.0) {
		Tr = pow(prim[0] * energy_density_scale / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		returnval = 1;
	}
	#endif

	return(returnval);
}
#endif