
#if(USE_3D_INV)
// 3D inversion for tabulated EOS
__device__ int Utoprim_3D_T(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
#endif
)
{
    double U_tmp[NPR_U], prim_tmp[NPR_HD];
    int i, ret;
    double alpha;
#if(DO_YE)
    double ye_new = U[YE] / U[RHO];
    validate_ye(&ye_new);
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

    ret = Utoprim_new_3D_T(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
        #if (DOHELM)
        , gpu_eos_table
        #endif
        #if (DO_YE)
        , ye_new
        #endif
    );

    /* Transform new primitive variables back if there was no problem : */
    if (ret == 0) {
        #pragma unroll 5
        for (i = 0; i < BCON1; i++) {
            prim[i] = prim_tmp[i];
        }
        #if (DO_YE)
        prim[YE] = U[YE] / U[RHO];
        validate_ye(&prim[YE]);
        #endif
    }

    return(ret);
}

__device__ int Utoprim_new_3D_T(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
#endif
#if (DO_YE)
    , double ye
#endif
) {
    /*
        Variables are:
            W = 1. / sqrt(1. - vsq)
            Z = rho * wenth * gammasq
            T = Tgas
    */

    double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
    double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
    int i, retval = 0, i_increase;
    double Bsq, QdotBsq, Qtsq, Qdotn, D;
    double xTgas = prim[UU];

    // Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
    Bcon[0] = 0.;
#pragma unroll 3
    for (i = 1; i < 4; i++) Bcon[i] = U[BCON1 + i - 1];

    lower(Bcon, gcov, Bcov);
#pragma unroll 4
    for (i = 0; i < 4; i++) Qcov[i] = U[QCOV0 + i];
    raise(Qcov, gcon, Qcon);

    Bsq = 0.;
#pragma unroll 3
    for (i = 1; i < 4; i++) Bsq += Bcon[i] * Bcov[i];

    QdotB = 0.;
#pragma unroll 4
    for (i = 0; i < 4; i++) QdotB += Qcov[i] * Bcon[i];
    QdotBsq = QdotB * QdotB;

    ncov = -sqrt(-1. / gcon[0]);
    ncon[0] = gcon[0] * ncov;
    ncon[1] = gcon[1] * ncov;
    ncon[2] = gcon[2] * ncov;
    ncon[3] = gcon[3] * ncov;

    Qdotn = Qcon[0] * ncov;

    Qsq = 0.;
    for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];

#if AMD
    Qtsq = fma(Qdotn, Qdotn, Qsq);
#else
    Qtsq = Qsq + Qdotn * Qdotn;
#endif
    D = U[RHO];

    /* calculate W from last timestep and use for guess */
    utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
    utsq += 2. * gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
    utsq += 2. * gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
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

    // EOS-specific calls:
#if (DOHELM)
// 1. Helmholtz EOS
    prim[RHO] = rho0;
    eos_mode_rhotemp_pres_u(gpu_eos_table, rho0, xTgas,
#if (DO_YE)
        ye,
#else 
        1.0,
#endif
        &p, &u
    );
#else
    // 2. Ideal gas EOS
    p = (GAMMA - 1.) * u;
#endif

    w = rho0 + u + p;

    W_last = w * gammasq;

    double x_3d[3];

    /*
    int safe_guess = get_safe_guess_NR_3D_T(x_3d, D, Bsq, Qdotn 
		#if(DO_YE)
		, ye
		#else
		, 1.0
		#endif
        #if (DOHELM)
        , gpu_eos_table
        #endif
    );
    */
    //if (safe_guess == 0) {
        // safe guess
        
        x_3d[0] = fabs(W_last);
        x_3d[1] = x1_of_x0(W_last, Bsq, Qtsq, QdotBsq);
        x_3d[2] = xTgas;

        /*x_3d[0] = MY_MAX(1.0, x_3d[0]);
        x_3d[0] = MY_MIN(1e2, x_3d[0]);
        
        x_3d[1] = MY_MAX(0.0, x_3d[1]);
        x_3d[1] = MY_MIN(0.95*W_TOO_BIG, x_3d[1]);
    
        x_3d[2] = MY_MAX(eos_temp_low, x_3d[2]);
        x_3d[2] = MY_MIN(eos_temp_up, x_3d[2]);*/
    //}

    /*i_increase = 0;
    while (((W_last * W_last * Qtsq + QdotBsq * (2. * W_last + Bsq)) >= W_last * W_last * (Bsq * Bsq + W_last) * (Bsq * Bsq + W_last)) && (i_increase < 10)) {
        W_last *= 10.;
        i_increase++;
    }*/
    //x_3d[0] = fabs(W_last);
    //x_3d[1] = x1_of_x0(W_last, Bsq, Qtsq, QdotBsq);
    //x_3d[2] = xTgas;

    retval = general_newton_raphson_3D_T(x_3d, Bsq, Qtsq, QdotBsq, Qdotn, D, tolerance
#if (DOHELM)
        , gpu_eos_table
#endif
#if (DO_YE)
        , ye
#endif
    );

    W = x_3d[0];
    vsq = x_3d[1];
    xTgas = x_3d[2];

    //Problem with solver, so return denoting error before doing anything further */
    if ((retval != 0) || (W == FAIL_VAL)) {
        retval = retval * 100 + 1;
        return(retval);
    }
    else {
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
    prim[RHO] = rho0;
    eos_mode_rhotemp_pres_u(gpu_eos_table, rho0, xTgas,
#if (DO_YE)
        ye,
#else 
        1.0,
#endif
        &p, &u
    );
#else
// 2. Ideal gas EOS
    p = (GAMMA - 1.) * (w - rho0) / GAMMA;
    u = w - (rho0 + p);
#endif

    // User may want to handle this case differently, e.g. do NOT return upon
    // a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
    if ((rho0 <= 0.)) {
        retval = 5;
        return(retval);
    }
    if ((u <= 0.) && (lim == BASIC)) {
        retval = 6;
        return(retval);
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


__device__ int general_newton_raphson_3D_T(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
) {
    double f[3], df[3], dx[3], x_old[3], error[3];
    double resid[3], jac[3][3];
    double errx;
    int n_iter, id, i_extra, doing_extra, keep_iterating;
    double x_lowlim[3];

    x_lowlim[0] = 0.0;
    x_lowlim[1] = 1e-15;
    x_lowlim[2] = eos_temp_low;

    // Initialize various parameters and variables:
    errx = 1.;
    f[0] = f[1] = f[2] = df[0] = df[1] = df[2] = 1.;
    i_extra = doing_extra = 0;
    for (id = 0; id < 3; id++) x_old[id] = x[id];

    n_iter = 0;

    //Start the Newton-Raphson iterations
    keep_iterating = 1;
    while (keep_iterating) {
        func_vsq_3D_T(x, dx, resid, jac, f, df, Bsq, Qtsq, QdotBsq, Qdotn, D
            #if (DOHELM)
            , gpu_eos_table
            #endif
            #if (DO_YE)
            , ye
            #endif
        );  /* returns with new dx, f, df */

        //Save old values before calculating the new
        errx = 0.;
        for (id = 0; id < 3; id++) {
            x_old[id] = x[id];
            x[id] += dx[id];
            //x[id] = fabs(x_old[id] + dx[id] - x_lowlim[id]) + x_lowlim[id];
        }

        /****************************************/
        /* Make sure that the new x[] is physical : */
        /****************************************/
        validate_x_3D_T(x, x_old);

        for (id = 0; id < 3; id++) {
            error[id] = (x[id] == 0.) ? fabs(x[id] - x_old[id]) : fabs((x[id] - x_old[id]) / x[id]);
        }

        /****************************************/
        /* Calculate the convergence criterion */
        /****************************************/
        errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);
        //errx = MY_MAX(error[0], MY_MAX(error[1], error[2]));

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
    if ((isfinite(f[0]) == 0) || (isfinite(f[1]) == 0) || (isfinite(f[2]) == 0)) return(2);

    //Depending on error return OK or error
    if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
    if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
    if (fabs(errx) <= tolerance)return(0);

    return(0);
}

__device__ void func_vsq_3D_T(double x[], double dx[], double resid[], double jac[][3], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
#endif
#if (DO_YE)
    , double ye
#endif
) {
    double J[3][3];
    double W = x[0];
    double vsq = x[1];
    double T = x[2];

    double E, E_EOS, P_EOS, dEdW, dEdvsq, dEdT, dpEOSdrho, dpEOSdT;

    // compute partial derivatives of specific internal energy: dEdW and dEdZ
    EP_dEdW_dEdZ_dEdT(&E_EOS, &P_EOS, &dEdvsq, &dEdW, &dEdT, &dpEOSdrho, &dpEOSdT, x, D
#if (DOHELM)
        , gpu_eos_table
#endif
#if (DO_YE)
        , ye
#endif
    );

    // compute specific internal energy from state vector x and conservatives
    double gamma = 1. / sqrt(1. - vsq);
    E = -1.0 + W / (D * gamma) - P_EOS * gamma / D;
    E = MY_MAX(E, 1e-30);

    // d/dW (1)
    J[0][0] = 2.0 * (W + Bsq) * vsq + 2. * (QdotBsq) * (1. / (W * W) + Bsq / (W * W * W));
    double a = J[0][0];

    // d/dvsq (1)
    J[0][1] = (W + Bsq) * (W + Bsq);
    double b = J[0][1];

    // d/dT (1)
    J[0][2] = 0.0;
    double c = J[0][2];

    // d/dW (2)
    J[1][0] = -1. - (QdotBsq) / (W * W * W);
    double d = J[1][0];

    // d/dvsq (2)
    J[1][1] = -0.5 * Bsq - 0.5 * D * dpEOSdrho * gamma;
    double e = J[1][1];

    // d/dT (2)
    J[1][2] = dpEOSdT;
    double fc = J[1][2];

    // d/dW (E-E(rho,T,Y_e))
    J[2][0] = dEdW;
    double g = J[2][0];

    // d/dvsq (E-E(rho,T,Y_e))
    J[2][1] = dEdvsq;
    double h = J[2][1];

    // d/dT (E-E(rho,T,Y_e))
    J[2][2] = dEdT;
    double k = J[2][2];

    // compute f(x) from (21), (22) and (28) in Cerda-Duran et al. 2008
    f[0] = (W + Bsq) * (W + Bsq) * vsq - Qtsq - (2.0 * W + Bsq) * (QdotBsq) / (W * W);
    f[1] = -Qdotn - W - 0.5 * Bsq * (1. + vsq) + (QdotBsq) / (2.0 * W * W) + P_EOS;
    f[2] = E - E_EOS;

    // Compute the determinant
    double A = e * k - fc * h;
    double B = fc * g - d * k;
    double C = d * h - e * g;
    double detJ = a * (A)+b * (B)+c * (C);

    //Compute the matrix inverse
    double Ji[3][3];
    Ji[0][0] = A / detJ;
    Ji[1][0] = B / detJ;
    Ji[2][0] = C / detJ;
    Ji[0][1] = (c * h - b * k) / detJ;
    Ji[1][1] = (a * k - c * g) / detJ;
    Ji[2][1] = (g * b - a * h) / detJ;
    Ji[0][2] = (b * fc - c * e) / detJ;
    Ji[1][2] = (c * d - a * fc) / detJ;
    Ji[2][2] = (a * e - b * d) / detJ;

    // Compute the step size
    dx[0] = -Ji[0][0] * f[0] - Ji[0][1] * f[1] - Ji[0][2] * f[2];
    dx[1] = -Ji[1][0] * f[0] - Ji[1][1] * f[1] - Ji[1][2] * f[2];
    dx[2] = -Ji[2][0] * f[0] - Ji[2][1] * f[1] - Ji[2][2] * f[2];
}

__device__ int get_safe_guess_NR_3D_T(double x[3], double D, double Bsq, double Qdotn, double ye
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
) {
#if (DOHELM && DOHELM_TEMPERATURE)
    double gamma_max = 1.0e2;
    double rho_max = MY_MIN(D, 0.95 * eos_dens_up);
    double ugas_max = rho_max * (-Qdotn - D - 0.5 * Bsq) / D;
    ugas_max = MY_MAX(1e-30, ugas_max);

    double tgas_max = 0.95 * eos_temp_up;
    double ugas_max_helm, Pmax;
    eos_mode_rhotemp_pres_u(gpu_eos_table, rho_max, tgas_max, ye, &Pmax, &ugas_max_helm);
    ugas_max = MY_MIN(ugas_max, ugas_max_helm);

    int flag = eos_mode_rhotemp_u_pres_floor(gpu_eos_table, rho_max, &tgas_max, ye, ugas_max, &Pmax);

    double W_max = -Qdotn + Pmax - 0.5 * Bsq;

    x[0] = W_max;
    x[1] = 1. - 1. / gamma_max;
    x[2] = tgas_max;
    return (flag);
#endif
}

__device__ void validate_x_3D_T(double x[3], double x0[3]) {
    /* Always take the absolute value of x[0] and check to see if it's too big:  */
    x[0] = fabs(x[0]);
    x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];

    x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
    x[1] = (x[1] > 1.) ? (1. - 1.e-15) : x[1];  /* if it's too big   */

    validate_T(&x[2]);
    return;
}
#endif