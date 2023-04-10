
__device__ int Utoprim_1dfix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim, int full_entropy
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

	if (U[0] <= 0.) return(-100);

	//First update the primitive B-fields
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	#if(DOKTOT)
	if(full_entropy)K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	else K_atm = U[KTOT] / U[RHO];
	#endif

	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_new_body3(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	#if (DO_YE)
    prim[YE] = U[YE] / U[RHO];
    validate_ye(&prim[YE]);
    #endif

    #if (DONUCLEAR)
    prim[XALPHA] = U[XALPHA] / U[RHO];
    prim[XATM] = U[XATM] / U[RHO];
    #endif

	return(ret);
}

__device__ int Utoprim_new_body3(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, retval=0, i_increase;
	double W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

	Bcon[0] = 0.;
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];
	lower(Bcon, gcov, Bcov);

	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);

	Bsq = 0.;
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	
	QdotBsq = QdotB*QdotB;

	ncov=-sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
	utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
	utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
	utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
	utsq += 2*gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
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

	rho0 = D / gamma;
	p = K_atm * pow(rho0, GAMMA);
	u = p / (GAMMA - 1.);
	w = rho0 + u + p;

	W_last = w*gammasq;

	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq) - QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq)) && (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	W_for_gnr2 = W_for_gnr2_old = W_last;
	rho_for_gnr2 = rho_for_gnr2_old = rho0;

	x_1d[0] = W_last;
	retval = general_newton_raphson3(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old, tolerance
		#if(TWO_T)
		, S
		, fel
		#endif
	);

	W = x_1d[0];

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

	vsq = vsq_calc(W, Bsq, Qtsq, QdotBsq);
	if (vsq >= 1.) {
		retval = 4;
		return(retval);
	}

	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;
	rho0 = D * gtmp;

	w = W * (1. - vsq);

	p = K_atm * pow(rho0, G_ATM);
	u = p / (GAMMA - 1.);

	if ((rho0 <= 0.)) {
		retval = 5;
		return(retval);
	}

	if ((u <= 0.) && (lim==BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}

	return(retval);
}

__device__ int general_newton_raphson3(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old, double tolerance
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double f, df, x_old[NEWT_DIM_1], dx[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int n_iter=0,  i_extra=0, doing_extra=0;
	int keep_iterating, i_increase;

	errx = 1.;
	df = f = 1.;
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		#if(USE_ISENTROPIC)   
		func_1d_orig1(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old
			#if(TWO_T)
			, S
			, fel
			#endif
		);  /* returns with new dx, f, df */
		#endif

		//Save old values before calculating the new
		x_old[0] = x[0];

		//Make Newton step
		x[0] += dx[0];

		i_increase = 0;
		while (((x[0] * x[0] * x[0] * (x[0] + 2.*Bsq) - QdotBsq*(2.*x[0] + Bsq)) <= x[0] * x[0] * (Qtsq - Bsq*Bsq)) && (i_increase < 10)) {
			x[0] -= (1.*i_increase) * dx[0] / 10.;
			i_increase++;
		}

		//Make sure value remains physical
		x[0] = fabs(x[0]);

		//Calculate the convergence criterion
		errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 1;
		if (doing_extra == 1) i_extra++;
		if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		n_iter++;
	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
		return(2);
	}

	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance) return(0);

	return(0);
}

__device__ int gnr2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double f, df, x_old[NEWT_DIM_1], dx[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int n_iter, i_extra, doing_extra;
	int keep_iterating;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		func_gnr2_rho(x, dx, resid, jac, &f, &df, D, K_atm, W_for_gnr2
			#if(TWO_T)
			,  S
			, fel
			#endif
		);  /* returns with new dx, f, df */
		
		//Save old values before calculating the new
		x_old[0] = x[0];

		//Make the newton step
		x[0] += dx[0];

		//Make sure x[0] is physical
		x[0] = fabs(x[0]);

		//Calculate the convergence criterion
		errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 0;
		if (doing_extra == 1) i_extra++;
		if (((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
		return(2);
	}

	if (fabs(errx) > MIN_NEWT_TOL2)return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL2) && (fabs(errx) > NEWT_TOL2))return(0);
	if (fabs(errx) <= NEWT_TOL2)return(0);

	return(0);
}

//isentropic version:   eq.  (27)
__device__ void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	int ntries;
	double  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	double  t32, t33, t34, t38, t5, t51, t67, t8,  x_rho[1], rho, rho_g;

	//W = x[0];
	W_for_gnr2 = x[0];

	// get rho from NR:
	rho_g = x_rho[0] = rho_for_gnr2;

	ntries = 0;
	while ((gnr2(x_rho, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2
		#if(TWO_T)
		,  S
		, fel
		#endif
		)) && (ntries++ < 10)) {
		rho_g *= 10.;
		x_rho[0] = rho_g;
	}

	rho = rho_for_gnr2 = x_rho[0];

	Dc = D;
	t1 = Dc*Dc;
	t2 = QdotBsq*t1;
	t3 = t2*Bsq;
	t5 = Bsq*Bsq;
	t8 = t1*Bsq;
	t10 = t1* x[0];
	t21 = x[0] * x[0];
	t23 = rho*rho;
	t26 = 1. / t1;
	resid[0] = (t3 + (2.0*t2 + ((Qtsq - t5)*t1 + (-2.0*t8 - t10)* x[0])* x[0])* x[0] + (t5 + (2.0*Bsq + x[0])* x[0])*t21*t23)*t26 / t21;
	t29 = t1*t1;
	t30 = QdotBsq*t29;
	t32 = GAMMA*K_atm;
	t33 = pow(rho, 1.0*GAMMA);
	t34 = t32*t33;
	t38 = t23 * t33;
	t51 = GAMMA*t1*K_atm*t33;
	t67 = t21* x[0];

	jac[0][0] = -2.0*(t30*Bsq*t34 + (t30*t34+ ((-t38*Bsq*t32 + Bsq*GAMMA*t1*K_atm*t33)*t1+ (-t38*GAMMA*K_atm + t51)*t1* x[0])*t21)* x[0] + ((-t3 + (-t2 + (-t8 - t10)*t21)* x[0])* x[0] + (-t5 - Bsq* x[0])*t67*t23)*t23)*t26 / (t51 - x[0] *t23) / t67;
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);

	return;
}

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double A, B, C, rho, W, B0;

	A = D*D;
	B0 = A * GAMMA * K_atm;
	B = B0 / (GAMMA - 1.);
	rho = x[0];
	W = W_for_gnr2;
	C = pow(rho, GAMMA - 1.);
	resid[0] = rho*W - A - B*C;
	jac[0][0] = W - B0 * C / rho;
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
	return;
}