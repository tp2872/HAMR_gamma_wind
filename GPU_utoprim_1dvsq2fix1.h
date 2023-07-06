

__device__ int Utoprim_1dvsq2fix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim, int full_entropy
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha, K_atm;
	#if(TWO_T)
	double S[2];
	#endif
	#if(DO_YE)
    double ye_new = U[YE] / U[RHO];
    validate_ye(&ye_new);
    #endif
	#if (DONUCLEAR)
    double x_alpha_new = U[XALPHA] / U[RHO];
    double x_atm_new = U[XATM] / U[RHO];
    validate_abund(&x_alpha_new);
    validate_abund(&x_atm_new);
    #endif

	if (U[0] <= 0.) {
		return(-100);
	}

	//First update the primitive B-fields
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into the new system
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma unroll 3
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	#pragma unroll 5
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	//Set entropy to kappa from log(kappa) if necessary
	#if(DOHELM)
	K_atm = U[KTOT] / U[RHO];
	#else
	if (full_entropy) K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	else K_atm = U[KTOT] / U[RHO];
	#endif

	//Set electron and ion entropies
	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_new_body2(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim
        #if(DOHELM)
        , gpu_eos_table
        #endif
        #if (DO_YE)
        , ye_new
        #endif
        #if(DONUCLEAR)
        , &x_alpha_new, &x_atm_new
        #endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
			
			#if (DO_YE)
			prim[YE] = U[YE] / U[RHO];
			validate_ye(&prim[YE]);
			#endif
        
			#if (DONUCLEAR)
			prim[XALPHA] = U[XALPHA] / U[RHO];
			prim[XATM] = U[XATM] / U[RHO];
			validate_abund(&prim[XALPHA]);
			validate_abund(&prim[XATM]);
			#endif
		}


		//Set entropy variables
		#if(TWO_T)
		prim[ENTRE] = S[0];
		prim[ENTRI] = S[1];
		#endif
	}

	return(ret);
}

__device__ int Utoprim_new_body2(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim
    #if(DOHELM)
    , const double* __restrict__ gpu_eos_table
    #endif
    #if(DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, gammasq, gamma, gtmp, W, utsq, vsq;
	int    i, retval=0;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];
	lower(Bcon, gcov, Bcov);

	#pragma unroll 4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	Qsq = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	//Calculate W from last timestep and use  for guess */
	utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
	utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
	utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
	utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
	utsq += 2 * gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
	utsq += gcov[9] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 3 - 1]; //1,2

	if ((utsq < 0.) && (fabs(utsq) < 1.0e-13)) {
		utsq = fabs(utsq);
	}
	if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
		retval = 2;
		return(retval);
	}

	gammasq = 1. + utsq;
	gamma = sqrt(gammasq);

	// Always calculate rho from D and gamma so that using D in EOS remains consistent
	//   i.e. you don't get positive values for dP/d(vsq) .
	rho0 = D / gamma;
   #if(DOHELM)
    // 1. Helmholtz EOS
    double dpdrho, dudrho;
		#if (DOHELM_TEMPERATURE)
		double xTgas = prim[UU];
		eos_mode_rhotemp_s_pres_u(gpu_eos_table, rho0, &xTgas,
			#if (DO_YE)
			ye, 
			#else
			1.0,
			#endif
			K_atm, &p, &u, &dpdrho, &dudrho
			#if (DONUCLEAR)
			, x_alpha, x_atm
			#endif
		);
		#else
		eos_mode_rhos_upres(gpu_eos_table, rho0, K_atm,
			#if (DO_YE)
			ye, 
			#else
			1.0,
			#endif
			&p, &u, &dpdrho, &dudrho);
		#endif
	#elif(TWO_T)
	double gamma_g = calc_gamma_gas_conserved(S, prim[RHO]);
	u = prim[UU];
	p = (gamma_g - 1.) * u;
	#else
	// 2. Gamma EOS
	u = prim[UU];
	p = (GAMMA - 1.)*u;
	#endif

	//Initialize independent variables for Newton-Raphson:
	x_1d[0] = 1. - 1. / gammasq;

	//Find vsq via Newton-Raphson:
	retval = general_newton_raphson2(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, tolerance
        #if(DOHELM)
        , gpu_eos_table
        #endif
        #if(DOHELM_TEMPERATURE)
        //, prim[UU]
        , &xTgas
        #endif
        #if (DO_YE)
        , ye
        #endif
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

	/* Problem with solver, so return denoting error before doing anything further */
	if (retval != 0) {
		retval = retval * 100 + 1;
		return(retval);
	}

	// Calculate v^2 :
	vsq = x_1d[0];
	if ((vsq >= 1.) || (vsq < 0.)) {
		retval = 4;
		return(retval);
	}

	//Find W from this vsq:
	W = W_of_vsq2(vsq, &p, &rho0, &u, D, K_atm
        #if(DOHELM)
        , gpu_eos_table
        #endif
        #if(DOHELM_TEMPERATURE)
        , &xTgas
        #endif
        #if(DO_YE)
        , ye
        #endif
        #if(DONUCLEAR)
        , x_alpha, x_atm
        #endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	
	//Recover the primitive variables from the scalars and conserved variables:
	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;

	// User may want to handle this case differently, e.g. do NOT return upon
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.)) {
		retval = 5;
		return(retval);
	}

	if ((u <= 0.) && (lim==BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
    #if (DOHELM_TEMPERATURE)
    prim[UU] = xTgas;
	#else
    prim[UU] = u;
    #endif

	#if(TWO_T)
	set_S_kappa(rho0, K_atm, S, fel);
	#endif

	#pragma unroll 3
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}

	/* done! */
	return(retval);
}

__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(DOHELM_TEMPERATURE)
    //, double temp_guess
    , double *temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int    n_iter,  i_extra, doing_extra;
	double W, W_old, rho, p, u;
	int   keep_iterating;

	//Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;

	x_old[0] = x[0];

	W = W_old = 0.;

	n_iter = 0;

	//Start the Newton-Raphson iterations
	keep_iterating = 1;
	#if (DOHELM_TEMPERATURE)
    //double xTgas_ini = *temp_prev;
    #endif
	while (keep_iterating) {
		#if (DOHELM_TEMPERATURE)
        //*temp_prev = xTgas_ini;
        #endif
		func_1d_gnr2(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm
            #if(DOHELM)
            , gpu_eos_table
            #endif
            #if(DOHELM_TEMPERATURE)
            //, temp_guess
            , temp_prev
            #endif
            #if (DO_YE)
            , ye
            #endif
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
			#if(TWO_T)
			,  S
			, fel
			#endif
		);/* returns with new dx, f, df */

		//Set old values
		errx = 0.;
		x_old[0] = x[0];

		//Make Newton step
		x[0] += dx[0];

		validate_x2(x, x_old);

		//Calculate W=w*gamma
		W_old = W;
		W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm
            #if(DOHELM)
            , gpu_eos_table
            #endif
            #if(DOHELM_TEMPERATURE)
            , temp_prev
            #endif
            #if(DO_YE)
            , ye
            #endif
            #if(DONUCLEAR)
            , x_alpha, x_atm
            #endif
			#if(TWO_T)
			, S
			, fel
			#endif
		);
		errx = (W == 0.) ? fabs(W - W_old) : fabs((W - W_old) / W);
		errx += (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 1;

		if (doing_extra == 1) i_extra++;

		//See if we've done the extra iterations, or have done too many iterations:
		if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		n_iter++;
	} 

	//Check for bad untrapped divergences
	if ((isfinite(f) == 0) || (isfinite(df) == 0))return(2);

	// Return in different ways depending on whether a solution was found:
	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance) return(0);

	return(0);
}

__device__ void validate_x2(double x[1], double x0[1]){
	double small = 1.e-10;
	x[0] = (x[0] >= 1.0) ? (0.5*(x0[0] + 1.)) : x[0];
	x[0] = (x[0] <  -small) ? (0.5*x0[0]) : x[0];
	x[0] = fabs(x[0]);
	return;
}

__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm
    #if(DOHELM)
    , const double* __restrict__ gpu_eos_table
    #endif
    #if(DOHELM_TEMPERATURE)
    //, double temp_guess
    , double *temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double W, Wsq, dWdvsq, fact_tmp, rho, p, u;
	//vsq = x[0];

	// Calculate best value for W given current guess for vsq: 
	#if(DOHELM)
		#if(DOHELM_TEMPERATURE)
		//double xtemp = temp_guess;
		#endif
    // Helmholtz EOS
    dWdvsq_calc2_helmholtz(gpu_eos_table, x[0], D, K_atm, &W, &dWdvsq
        #if(DOHELM_TEMPERATURE)
        , temp_prev
        //, &xtemp
        #endif
        #if (DO_YE)
        , ye
        #endif
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    Wsq = W * W;
    #else
	//Gamma EOS
	W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(DOHELM_TEMPERATURE)
		, &xtemp
		#endif
		#if(DO_YE)
		, ye
		#endif
		#if(DONUCLEAR)
		, *x_alpha, *x_atm
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	Wsq = W * W;

		// Doing this assuming  P = (G-1) u :
		#if(TWO_T)
		double p_new, u_new, rho_new, vsq_new, W_new, dvsq;
		dvsq = MY_MIN(1.e-8, 1.0 - (x[0] + fabs(1.e-8)));
		vsq_new = x[0] + dvsq;
		W_new=W_of_vsq2(vsq_new, &p_new, &rho_new, &u_new, D, K_atm
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(DOHELM_TEMPERATURE)
			, &xtemp
			#endif
			#if(DO_YE)
			, ye
			#endif
			#if(DONUCLEAR)
			, *x_alpha, *x_atm
			#endif
			#if(TWO_T)
			, S
			, fel
			#endif
		);
		dWdvsq = (W_new - W) / dvsq;
		#else
		dWdvsq = dWdvsq_calc2(x[0], rho, p);
		#endif
	#endif

	fact_tmp = (Bsq + W);
	resid[0] = Qtsq - x[0] * fact_tmp * fact_tmp + QdotBsq * (Bsq + 2.*W) / Wsq;
	jac[0][0] = -fact_tmp * (fact_tmp + 2. * dWdvsq * (x[0] + QdotBsq / (W*Wsq)));
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
}

__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm
    #if(DOHELM)
    , const double* __restrict__ gpu_eos_table
    #endif
    #if(DOHELM_TEMPERATURE)
    , double *temp_prev
    #endif
    #if(DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double gtmp;
	gtmp = (1. - vsq);
	rho[0] = D * sqrt(gtmp);
	#if(DOHELM)
    // 1. Helmholtz EOS
    double dpdrho, dudrho;
		#if (DOHELM_TEMPERATURE)
		eos_mode_rhotemp_s_pres_u(gpu_eos_table, *rho, temp_prev,
			#if (DO_YE)
			ye, 
			#else 
			1.0,
			#endif
			K_atm, p, u, &dpdrho, &dudrho
			#if (DONUCLEAR)
			, x_alpha, x_atm
			#endif
		);
		#else
		eos_mode_rhos_upres(gpu_eos_table, *rho, K_atm, 
			#if (DO_YE)
			ye, 
			#else
			1.0,
			#endif
			p, u, &dpdrho, &dudrho);
		#endif
	#elif(TWO_T)
		//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
		double game, gami, pe, pi, T_e, T_i;
		#if(!CONSTANTGAMMA)
		double gamg;
		#endif

		#if(CONSTANTGAMMA)
		game = GAMMA;
		gami = GAMMA;
			#if(FULL_ENTROPY)
			T_e = fabs((game - 1.0) * exp(S[0] * pow(rho[0], game - 1.0)));
			T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho[0], gami - 1.0)));
			#else
			T_e = fabs(S[0] * pow(rho[0], game - 1.0));
			T_i = fabs(S[1] * pow(rho[0], gami - 1.0));
			#endif
		#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		game = GAMMAE;
		gami = GAMMA;
			#if(FULL_ENTROPY)
			T_e = fabs((game - 1.0) * exp(S[0] * pow(rho[0], game - 1.0)));
			T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho[0], gami - 1.0)));
			#else
			T_e = fabs(S[0] * pow(rho[0], game - 1.0));
			T_i = fabs(S[1] * pow(rho[0], gami - 1.0));
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY_VARGAMMA)
			T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
			T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * exp(S[1])), 2. / 3.)) - 1.0) / MU_I);
			#else
			T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho[0], 2. / 3.) * fabs(S[0])) - 1.0) / (MU_E * MASS_RATIO));
			T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho[0], 2. / 3.) * fabs(S[1])) - 1.0) / MU_I);
			#endif
		#endif

		//Calculate gas pressures
		pe = T_e * rho[0];
		pi = T_i * rho[0];

		//Update internal energy of electrons
		p[0] = (pe + pi);

		//Limit temperature ratios
		if (pe > (1.0 - 0.5 * FLOOR_ENTROPY) * p[0]) pe = (1.0 - FLOOR_ENTROPY) * p[0];
		if (pe < 0.5 * FLOOR_ENTROPY * p[0]) pe = FLOOR_ENTROPY * p[0];
		pi = p[0] - pe;

		//Set temperature
		T_e = pe / rho[0];
		T_i = pi / rho[0];

		//Calculate the internal energy
		#if(CONSTANTGAMMA)
		u[0] = p[0] / (GAMMA - 1.0);
		#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		game = GAMMAE;
		gami = GAMMA;
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + T_i / T_e)) / ((T_i / T_e) * (game - 1.0) + 1.0 * (gami - 1.0));
		u[0] = p[0] / (gamg - 1.0);
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		game = (10.0 + 20.0 * T_e * MU_E * MASS_RATIO) / (6.0 + 15.0 * T_e * MU_E * MASS_RATIO);
		gami = (10.0 + 20.0 * T_i * MU_I) / (6.0 + 15.0 * T_i * MU_I);
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + T_i / T_e)) / ((T_i / T_e) * (game - 1.0) + 1.0 * (gami - 1.0));
		u[0] = p[0] / (gamg - 1.0);
		#endif
	#else
	p[0] = K_atm * pow(rho[0], GAMMA);
	u[0] = p[0] / (GAMMA - 1.);
	#endif
	return((rho[0] + u[0] + p[0]) / gtmp);
}

__device__ void set_S_kappa(double rho, double K_atm, double* S, double fel) {
	//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
	double game, gami, p, pe, pi, T_e, T_i;

	#if(CONSTANTGAMMA)
	game = GAMMA;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		T_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)));
		T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)));
		#else
		T_e = fabs(S[0] * pow(rho, game - 1.0));
		T_i = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		T_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)));
		T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)));
		#else
		T_e = fabs(S[0] * pow(rho, game - 1.0));
		T_i = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY_VARGAMMA)
		T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[1])), 2. / 3.)) - 1.0) / MU_I;
		#else
		T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[0])) - 1.0) / (MU_E * MASS_RATIO);
		T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho, 2. / 3.) * fabs(S[1])) - 1.0) / MU_I;
		#endif
	#endif

	//Calculate gas pressures
	pe = T_e * rho;
	pi = T_i * rho;

	//Update internal energy of electrons
	p = (pe + pi);

	//Limit temperature ratios
	if (pe > (1.0 - 0.5 * FLOOR_ENTROPY) * p) pe = (1.0 - FLOOR_ENTROPY) * p;
	if (pe < 0.5 * FLOOR_ENTROPY * p) pe = FLOOR_ENTROPY * p;
	pi = p - pe;

	//Set temperature
	T_e = pe / rho;
	T_i = pi / rho;

	//Calculate the internal energy
	#if(CONSTANTGAMMA || FIXEDGAMMA)
		#if(FULL_ENTROPY)
		S[0] = 1.0 / (game - 1.0) * log(pe * pow(rho, -game));
		S[1] = 1.0 / (gami - 1.0) * log(pi * pow(rho, -gami));
		#else
		S[0] = pe * pow(rho, -game);
		S[1] = pi * pow(rho, -gami);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY_VARGAMMA)
		S[0] = log(pow(T_e * (MU_E * MASS_RATIO), 1.5) * pow(T_e * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(T_i * MU_I, 1.5) * pow(T_i * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = (T_e * (MU_E * MASS_RATIO)) * (T_e * (MU_E * MASS_RATIO) + 0.4) / pow(rho, 2. / 3.);
		S[1] = (T_i * MU_I) * (T_i * MU_I + 0.4) / pow(rho, 2. / 3.);
		#endif
	#endif
}

//W=((rho+u+p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+u+p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+gam/(gam-1)*p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+gam/(gam-1)*kappa*rho^gamma)/(1-vsq))
//dWdvsq=(0.5*rho+u+p)/(1-vsq)^2 + d(u+p)/dvsq/(1-vsq)
//dWdvsq=(0.5*rho+u+p)/(1-vsq)^2 +
__device__ double dWdvsq_calc2(double vsq, double rho, double p){
	return((GAMMA*(2. - GAMMA)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
}

// Entropy inversion, Helmholtz EOS
#if(DOHELM)
__device__ void dWdvsq_calc2_helmholtz(const double* __restrict__ gpu_eos_table, double vsq, double D, double K_atm, double* W, double* dWdvsq
    #if(DOHELM_TEMPERATURE)
    , double *temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
)
{
    double gtmp;
    double p, u, dpdrho, dudrho;
    gtmp = (1. - vsq);

    #if (DOHELM_TEMPERATURE)
    eos_mode_rhotemp_s_pres_u(gpu_eos_table, D * sqrt(gtmp), temp_prev,
        #if(DO_YE)
        ye, 
        #else
        1.0,
        #endif
        K_atm, &p, &u, &dpdrho, &dudrho
        #if(DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    #else
    eos_mode_rhos_upres(gpu_eos_table, D * sqrt(gtmp), K_atm,
		#if (DO_YE)
		ye, 
		#else
		1.0,
		#endif
		&p, &u, &dpdrho, &dudrho);
    #endif

    *W = (D * sqrt(gtmp) + u + p) / gtmp;
    *dWdvsq = ((0.5 * D * sqrt(gtmp) * (1.0 - dpdrho - dudrho) + p + u) / (gtmp * gtmp));
}
#endif