
__device__ int Utoprim_2d(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if(TWO_T)
    , double fel
    #endif
) {
    double U_tmp[NPR_U], prim_tmp[NPR_HD];
    int i, ret;
    double alpha;
    #if(TWO_T)
    double S[NPR_2T];
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

    /* First update the primitive B-fields */
    #pragma unroll 3
    for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

    /* Set the geometry variables: */
    alpha = 1.0 / sqrt(-gcon[0]);

    /* Transform the CONSERVED variables into the new system */
    U_tmp[RHO] = alpha * U[RHO] / gdet;
    U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
    #pragma unroll 3
    for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
    #pragma unroll 3
    for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

    /* Transform the PRIMITIVE variables into the new system */
    #pragma unroll 5
    for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

    //Calculate entropy variable for 2T fluids to recover EOS gamma
    #if(TWO_T)
    S[0] = U[ENTRE] / U[RHO];
    S[1] = U[ENTRI] / U[RHO];
    #endif

    ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
        #if (DOHELM)
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

    /* Transform new primitive variables back if there was no problem : */
    if (ret == 0) {
        #pragma unroll 5
        for (i = 0; i < BCON1; i++) {
            prim[i] = prim_tmp[i];
        }

        #if(TWO_T)
        prim[ENTRE] = S[0];
        prim[ENTRI] = S[1];
        #endif
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

#if(DOFLR)
    prim[FLR] = U[FLR] / U[RHO];
#endif
#if (NEUTRON_STAR)
    prim[FLRFRAC] = U[FLRFRAC] / U[RHO];
#endif

    return(ret);
}

__device__ int Utoprim_new_body(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
    , double *S
    , double fel
    #endif
){
    double x_2d[NEWT_DIM_2];
    double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
    double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
    int i, retval=0, i_increase;
    double Bsq, QdotBsq, Qtsq, Qdotn, D;
    #if(TWO_T)
    double gamma_g;
    #endif

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
    for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

    #if AMD
    Qtsq = fma(Qdotn, Qdotn, Qsq);
    #else
    Qtsq = Qsq + Qdotn*Qdotn;
    #endif
    D = U[RHO];

    /* calculate W from last timestep and use for guess */
    utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
    utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
    utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
    utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
    utsq += 2 * gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
    utsq += gcov[9] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 3 - 1]; //3,3

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
    #if (DOHELM_TEMPERATURE)
    double xTgas = prim[UU];
    #else
    u = prim[UU];
    #endif

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    #if (DOHELM_TEMPERATURE)
    eos_mode_rhotemp_pres_u(gpu_eos_table, rho0, xTgas,
        #if (DO_YE)
        ye, 
        #else 
        1.0,
        #endif
        &p, &u
        #if(DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    #else
    eos_mode_rhou_pres (gpu_eos_table, rho0, u
        #if (DO_YE)
		, ye
		#else
		, 1.0
		#endif
        , &p);
    #endif
    #elif(TWO_T)
    gamma_g = calc_gamma_gas_conserved(S, prim[RHO]);
    p = (gamma_g - 1.) * u;
    #else
    // 2. Ideal gas EOS
    p = (GAMMA - 1.)*u;
    #endif

    w = rho0 + u + p;

    W_last = w*gammasq;

    // Make sure that W is large enough so that v^2 < 1 :
    i_increase = 0;
    while (((W_last*W_last*W_last * (W_last + 2.*Bsq) - QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq))&& (i_increase < 10)) {
        W_last *= 10.;
        i_increase++;
    }
    /*while (((W_last * W_last * Qtsq + QdotBsq * (2. * W_last + Bsq)) >= W_last * W_last * (Bsq * Bsq + W_last) * (Bsq * Bsq + W_last)) && (i_increase < 10)) {
        W_last *= 10.;
        i_increase++;
    }*/

    // Calculate W and vsq:
    x_2d[0] = fabs(W_last);
    x_2d[1] = x1_of_x0(W_last, Bsq, Qtsq, QdotBsq);
    retval = general_newton_raphson(x_2d, Bsq, Qtsq, QdotBsq, Qdotn, D, tolerance
        #if (DOHELM)
        , gpu_eos_table
        #endif
        #if(DOHELM_TEMPERATURE)
        //, prim[UU]
        , &xTgas
        #endif
        #if (DO_YE)
        , ye
        #endif
        #if(TWO_T)
        , S
        , fel
        #endif
        #if(DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );

    W = x_2d[0];
    vsq = x_2d[1];

    //Problem with solver, so return denoting error before doing anything further */
    if ((retval != 0) || (W == FAIL_VAL)) {
        retval = retval * 100 + 1;
        return(retval);
    }
    else{
        if (W <= 0. || W > W_TOO_BIG) {
            retval = 3;
            return(retval);
        }
    }

    //Calculate v^2:
    if (vsq >= 1.) {
        retval = 4;
        return(retval);
    }

    //Recover the primitive variables from the scalars and conserved variables:
    gtmp = sqrt(1. - vsq);
    gamma = 1. / gtmp;
    rho0 = D * gtmp;

    w = W * (1. - vsq);

    // EOS-specific calls:
    #if (DOHELM)
		// 1. Helmholtz EOS
		#if (DOHELM_TEMPERATURE)
		eos_mode_rhotemp_w_pres_u (gpu_eos_table, rho0, &xTgas,
			#if (DO_YE)
			ye,
			#else 
			1.0,
			#endif
			w-rho0, &p, &u
			#if(DONUCLEAR)
			, x_alpha, x_atm
			#endif
		);
		#else
		eos_mode_rhow_pres_u (gpu_eos_table, rho0, w-rho0, 
            #if (DO_YE)
			ye,
			#else 
			1.0,
			#endif
            &p, &u);
		#endif
    #elif(TWO_T)
    gamma_g = set_S_w(S, rho0, w
        #if(TWO_T)
        , fel
        #endif
    );
    u = (w - rho0) / gamma_g;
    p = (gamma_g - 1.0) * u;
    #else
    // 2. Ideal gas EOS
    p = (GAMMA - 1.)*(w - rho0) / GAMMA;
    u = w - (rho0 + p);
    #endif

    // User may want to handle this case differently, e.g. do NOT return upon
    // a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
    if ((rho0 <= 0.)) {
        retval = 5;
        return(retval);
    }
    if ((u <= 0.) && (lim==BASIC)) {
        //if (u == 0.) {
            //prim[UU] = eos_temp_low;
        //}
        //else {
            retval = 6;
            //if (rho0 > 1e-5) printf("\t\t\n (u2prim 6) w:%e rho0:%e vsq:%e gamma:%e\n", w, rho0, vsq, gamma);
            return(retval);
        //}
    }

    prim[RHO] = rho0;
    #if (DOHELM_TEMPERATURE)
    prim[UU] = xTgas;
    #else
    prim[UU] = u;
    #endif

    #if AMD
    #pragma unroll 3
    for (i = 1; i < 4; i++) {
        Qtcon[i] = fma(ncon[i], Qdotn, Qcon[i]);
        prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (fma(QdotB, Bcon[i] / W, Qtcon[i]));
    }
    #else
    #pragma unroll 3
    for (i = 1; i < 4; i++) {
        Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
        prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
    }
    #endif
    /* done! */
    return(retval);
}

__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq){
    double Wsq, Xsq;
    Wsq = W*W;
    Xsq = (Bsq + W) * (Bsq + W);
    #if AMD
    return((fma(Wsq, Qtsq, QdotBsq * (Bsq + 2.*W))) / (Wsq*Xsq));
    #else
    return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
    #endif
}

__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq){
    double vsq;
    vsq = fabs(vsq_calc(x0, Bsq, Qtsq, QdotBsq)); // guaranteed to be positive 
    return((vsq > 1.) ? (1.0 - 1.e-15) : vsq);
}

__device__ void validate_x(double x[2], double x0[2], double D){

    /* Always take the absolute value of x[0] and check to see if it's too big:  */
    x[0] = fabs(x[0]);
    x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];

    x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
    x[1] = (x[1] > 1.) ? (1. - 1.e-15) : x[1];  /* if it's too big   */

    //x[0] = MY_MAX(x[0], D / sqrt(1. - x[1]));
    return;
}

__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DOHELM_TEMPERATURE)
    //, double temp_guess
    , double *temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
    , double *S
    , double fel
    #endif
){
    double f, df, dx[NEWT_DIM_2], x_old[NEWT_DIM_2];
    double resid[NEWT_DIM_2], jac[NEWT_DIM_2][NEWT_DIM_2];
    double errx;
    int n_iter, id, i_extra, doing_extra, keep_iterating;

    // Initialize various parameters and variables:
    errx = 1.;
    df = f = 1.;
    i_extra = doing_extra = 0;
    for (id = 0; id < NEWT_DIM_2; id++)  x_old[id] = x[id];

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
        func_vsq(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D
            #if (DOHELM)
            , gpu_eos_table
            #endif
            #if (DOHELM_TEMPERATURE)
            //, temp_guess
            , temp_prev
            #endif
            #if (DO_YE)
            , ye
            #endif
            #if(DONUCLEAR)
            , x_alpha, x_atm
            #endif
            #if(TWO_T)
            , S
            , fel
            #endif
        );  /* returns with new dx, f, df */

        //Save old values before calculating the new
        errx = 0.;
        for (id = 0; id < NEWT_DIM_2; id++) x_old[id] = x[id];

        //Make the newton step
        for (id = 0; id < NEWT_DIM_2; id++) x[id] += dx[id];

        /****************************************/
        /* Make sure that the new x[] is physical : */
        /****************************************/
        validate_x(x, x_old, D);

        /****************************************/
        /* Calculate the convergence criterion */
        /****************************************/
        errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

        /*****************************************************************************/
        /* If we've reached the tolerance level, then just do a few extra iterations */
        /*  before stopping                                                          */
        /*****************************************************************************/

        if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0))doing_extra = 1;
        if (doing_extra == 1) i_extra++;
        if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
            keep_iterating = 0;
        }

        n_iter++;
    } 

    //Check for bad untrapped divergences
    if ((isfinite(f) == 0) || (isfinite(df) == 0)) return(2);

    //Depending on error return OK or error
    if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
    if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
    if (fabs(errx) <= tolerance)return(0);
    
    return(0);
}

__device__ void func_vsq(double x[], double dx[], double resid[], double jac[][NEWT_DIM_2], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DOHELM_TEMPERATURE)
    //, double temp_guess
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if(DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
	#if(TWO_T)
    , double *S
    , double fel
    #endif
){
	double Wsq, p_tmp, dPdvsq, dPdW, gtmp;
	double t11, t16, t18, t2, t21, t23,t24, t25, t3, t35, t36, t4, t40, t9;

	//W = x[0];
	//vsq = x[1];

	Wsq = x[0] * x[0];
	gtmp = 1. - x[1];

	// EOS-specific calls:
	#if (DOHELM)
    // 1. Helmholtz EOS
    double rho = D * sqrt(gtmp);
    double dpdrho, dpde_d;
		#if (DOHELM_TEMPERATURE)
		//double xtemp = temp_guess;
		eos_mode_rhotemp_w_pres_dpdrho_dpde_d(gpu_eos_table, rho 
            , temp_prev,
            //, &xtemp,
			#if (DO_YE)
			ye,
			#else
			1.0, 
			#endif
			(x[0] * gtmp) - rho, &p_tmp, &dpdrho, &dpde_d
			#if(DONUCLEAR)
			, x_alpha, x_atm
			#endif
		);
		#else
		eos_mode_rhow_pres_dpdrho_dpde_d(gpu_eos_table, rho, (x[0] * gtmp) - rho, 
            #if (DO_YE)
			ye,
			#else
			1.0, 
			#endif
            &p_tmp, &dpdrho, &dpde_d);
		#endif
		#if (inversion_w_edits)
		// Danat: edit (DIMARK)
		double dudp = rho / dpde_d;
		dPdW = 1.0 / (1.0 + dudp) * gtmp;
		dPdvsq = (-x[0] + 0.5 * D / sqrt(gtmp) * (1. - dpdrho * dudp)) / (1. + dudp);
		#else 
		double dpdeps_o_rho = dpde_d / rho;
		double dpdvsq_1 = -0.5 * D / sqrt(gtmp) * dpdrho;
		double dpdvsq_2 = -0.5 * (x[0] + p_tmp / sqrt(gtmp)) / rho;
		dPdW = (dpdeps_o_rho / (1.0 + dpdeps_o_rho)) * gtmp;
		dPdvsq = (dpdvsq_1 + dpde_d * dpdvsq_2) / (1.0 + dpdeps_o_rho);
		#endif
    #elif(TWO_T)
	double gamma_eos1, gamma_eos2, w, rho, dgamma, factor, dvsq, dW;

	//Temporary variables
	gtmp = 1. - x[1];
	w = x[0] * gtmp;
	rho = D * sqrt(gtmp);
	gamma_eos1 = calc_gamma_gas_w(S, rho, w, fel);
	factor = (gamma_eos1 - 1.) / gamma_eos1;
	p_tmp = factor * (x[0] * gtmp - D * sqrt(gtmp));

	//Offset sizes
	dW = 1.e-8 * rho;
	dvsq = MY_MIN(1.e-8, fabs(1.0 - (x[1] + 1.e-8)));

	//Calculate dPdW
	gamma_eos2 = calc_gamma_gas_w(S, rho, (x[0] + dW) * gtmp, fel);
	dgamma =  (gamma_eos2 - gamma_eos1) / dW;
	dPdW = factor * gtmp + (x[0] * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;

	//Calculate dPdvsq
	gamma_eos2 = calc_gamma_gas_w(S, D * sqrt(fabs(1.0 - (x[1] + dvsq))), x[0] * (1.0 - (x[1] + dvsq)), fel);
	dgamma = (gamma_eos2 - gamma_eos1) / dvsq;
	dPdvsq = factor * (0.5 * D / sqrt(gtmp) - x[0]) + (x[0] * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;
	#else
	// 2. Ideal gas EOS
	#if(AMD)
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#else
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#endif
	#endif

	// These expressions were calculated using Mathematica, but fmae into efficient  code using Maple.  Since we know the analytic form of the equations, we can explicitly calculate the Newton-Raphson step: 
	#if(AMD)
	t2 = fma(-0.5, Bsq, dPdvsq);
	t3 = Bsq + x[0];
	t4 = t3*t3;
	t9 = 1. / Wsq;
	t11 = fma(QdotBsq, (Bsq + 2.0* x[0])*t9, fma(-x[1], t4, Qtsq));
	t16 = QdotBsq*t9;
	t18 = -fma(0.5, Bsq*(1.0 + x[1]), Qdotn) + fma(0.5, t16, -x[0] + p_tmp);
	t21 = 1. / t3;
	t23 = 1. / x[0];
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = fma(t25, t3, (fma(-2.0, dPdvsq, Bsq))*(fma(x[1], Wsq* x[0], QdotBsq))*t9*t23);
	t36 = 1. / t35;
	dx[0] = -(fma(t2, t11, t4*t18))*t21*t36;
	t40 = (x[1] + t24)*t3;
	dx[1] = -(-fma(t25, t11, 2.0*t40*t18))*t21*t36;
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;
	*df = fma(-resid[0], resid[0], -resid[1] * resid[1]);
	#else
	t2 = -0.5*Bsq + dPdvsq;
	t3 = Bsq + x[0];
	t4 = t3*t3;
	t9 = 1. / Wsq;
	t11 = Qtsq - x[1] *t4 + QdotBsq*(Bsq + 2.0* x[0])*t9;
	t16 = QdotBsq*t9;
	t18 = -Qdotn - 0.5*Bsq*(1.0 + x[1]) + 0.5*t16 - x[0] + p_tmp;
	t21 = 1. / t3;
	t23 = 1. / x[0];
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = t25*t3 + (Bsq - 2.0*dPdvsq)*(QdotBsq + x[1] *Wsq* x[0])*t9*t23;
	t36 = 1. / t35;
	dx[0] = -(t2*t11 + t4*t18)*t21*t36;
	t40 = (x[1] + t24)*t3;
	dx[1] = -(-t25*t11 - 2.0*t40*t18)*t21*t36;
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;
	*df = -resid[0] * resid[0] - resid[1] * resid[1];
	#endif
	*f = -0.5 * (*df);
}
