

__device__ int Utoprim_NM(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if(TWO_T)
    , double fel
    #endif
){
    double U_tmp[NPR_U], prim_tmp[NPR_HD];
    int i, ret;
    double alpha;
    #if(TWO_T)
    double S[NPR_2T];
    #endif
    #if (DO_YE)
    double ye_new = U[YE] / U[RHO];
    validate_ye(&prim[YE]);
    #endif
    #if (DONUCLEAR)
    double x_alpha_new = U[XALPHA] / U[RHO];
    double x_atm_new = U[XATM] / U[RHO];
    validate_abund(&x_alpha_new);
    validate_abund(&x_atm_new);
    #endif

    //If mass flux negative, return immediately
    if (U[0] <= 0.) return(-100);

    //First update the primitive B-fields
    for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

    //Set the geometry variables
    alpha = 1.0 / sqrt(-gcon[0]);

    //Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
    U_tmp[RHO] = alpha * U[RHO] / gdet; //W=ucon[0]*alpha
    U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
    for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
    for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

    //Transform the PRIMITIVE variables into the new system
    for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

    #if(TWO_T)
    S[0] = U[ENTRE] / U[RHO];
    S[1] = U[ENTRI] / U[RHO];
    #endif

    ret = Utoprim_NM_calc(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
        #if (DOHELM)
        , gpu_eos_table
        #endif
        #if (DO_YE)
        , ye_new
        #endif
        #if(TWO_T)
        , S
        , fel
        #endif
    );

    //Transform new primitive variables back if there was no problem : */
    if (ret == 0) {
        for (i = 0; i < BCON1; i++) {
            prim[i] = prim_tmp[i];
        }
        #if(TWO_T)
        prim[ENTRE] = S[0];
        prim[ENTRI] = S[1];
        #endif
    }

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

    return(ret);
}

__device__ int Utoprim_NM_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if(TWO_T)
    , double *S
    , double fel
    #endif
) {
    double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
    double rho0, u,  w,  gamma,   vsq, errx, gamma_eos;
    double Bsq, QdotBsq, Qtsq, Qdotn;
    int i;

    // Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
    Bcon[0] = 0.;
    for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

    lower(Bcon, gcov, Bcov);
    for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
    raise(Qcov, gcon, Qcon);

    Bsq = 0.;
    /*#pragma ivdepreduction(+:Bsq)*/
    for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

    QdotB = 0.;
    //#pragma ivdepreduction(+:QdotB)
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
    Qtsq = Qsq + Qdotn*Qdotn;

    //Start inversion scheme AKA Newman et al
    double a, d, z, phi, R, Wsq, p_array[3], epsilon, p_old, p_new;
    int iter = 0;
    int iter_tot = 0;
    int set_variables = 0;

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    // -- to get min. pressure for a given density, set T = T_min
    double xpres;
    #if (DOHELM_TEMPERATURE)
    eos_mode_rhotemp_pres(gpu_eos_table, prim[RHO], eos_temp_low,
        #if(DO_YE)
        ye,
        #else
        1.0, 
        #endif
        &xpres
        #if (DONUCLEAR)
        , &prim[XALPHA], &prim[XATM]
        #endif
    );
    #else 
    eos_mode_rhotemp_pres_min (gpu_eos_table, prim[RHO], 
        #if(DO_YE)
        ye,
        #else
        1.0,
        #endif
        &xpres);
    #endif
    p_array[0] = xpres;
    #else
        // Ideal gas EOS
        #if(TWO_T)
        gamma_eos = calc_gamma_gas_conserved(S, prim[RHO]);
        #else
        gamma_eos = GAMMA;
        #endif
    p_array[0] = (gamma_eos - 1.) * prim[UU];
    #endif
    
    p_new = p_array[0];
    d = 0.5*(Qtsq*Bsq - QdotBsq);
    if (d < 0.0) return(1);
    do {
        set_variables = 0;
        a = -Qdotn + p_new + 0.5 * Bsq;
        phi = acos(1. / a * sqrt((27. * d) / (4. * a)));
        epsilon = a / 3. - 2. / 3. * a * cos(2. / 3. * phi + 2. / 3. * M_PI);
        z = epsilon - Bsq;

        vsq = (Qtsq * z * z + QdotBsq * (Bsq + 2. * z)) / (z * z * pow(Bsq + z, 2.));

        // DANAT: add this - therefore rho0 is nan
        if (fabs(vsq) < 1e-15) vsq = 0.0;
        if (fabs(vsq) >= 1.0) return(1);

        Wsq = 1. / (1. - vsq);
        w = z * (1. - vsq);
        gamma = 1. / sqrt(1. - vsq);
        rho0 = U[RHO] / gamma;

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        #if (DOHELM_TEMPERATURE)
        eos_mode_rhotemp_w_pres_u(gpu_eos_table, rho0, &prim[UU], 
            #if(DO_YE)
            ye,
            #else
            1.0,
            #endif
            w - rho0, &xpres, &u
            #if (DONUCLEAR)
            , &prim[XALPHA], &prim[XATM]
            #endif
        );
        #else
        eos_mode_rhow_pres_u(gpu_eos_table, rho0, w-rho0, 
            #if (DO_YE)
			ye,
			#else 
			1.0,
			#endif
            &xpres, &u);
        #endif
        #else
            // Ideal gas EOS
            #if(TWO_T)
            gamma_eos = calc_gamma_gas_w(S, rho0, w, fel);
            if (isnan(gamma_eos))gamma_eos = GAMMA;
            #else
            gamma_eos = GAMMA;
            #endif
        u = (w - rho0) / gamma_eos;
        #endif

        iter++;
        iter_tot++;

        #if DOHELM
        // Helmholtz EOS
        p_array[iter] = xpres;
        #else
        // Ideal gas EOS
        p_array[iter] = (gamma_eos - 1.) * u;
        #endif

        p_old = p_array[iter - 1];
        p_new = p_array[iter];

        if (iter >= 2) {
            R = (p_array[iter] - p_array[iter - 1]) / (p_array[iter - 1] - p_array[iter - 2]);

            if (R < 1. && R>0.) {
                set_variables = 1;
                p_new = p_array[iter - 1] + (p_array[iter] - p_array[iter - 1]) / (1. - R);
                p_old = p_array[iter];
                iter = 0.;
                p_array[iter] = p_new;
            }
        }
        errx = fabs(p_new - p_old) / fabs(p_new + p_old);
    } while (errx > tolerance && iter_tot < MAX_NEWT_ITER);

    //Return in different ways depending on tolerance and minimum tolerance
    if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);

    if (set_variables == 1) {
        a = -Qdotn + p_new + 0.5 * Bsq;
        phi = acos(1. / a * sqrt((27. * d) / (4. * a)));
        epsilon = a / 3. - 2. / 3. * a * cos(2. / 3. * phi + 2. / 3. * M_PI);
        z = epsilon - Bsq;

        vsq = (Qtsq * z * z + QdotBsq * (Bsq + 2. * z)) / (z * z * pow(Bsq + z, 2.));
        Wsq = 1. / (1. - vsq);
        w = z / Wsq;
        if (vsq >= 1.0 || vsq<0. || z <= 0. || z > W_TOO_BIG || !isfinite(vsq) || !isfinite(z)) {
            return(4);
        }
        gamma = sqrt(Wsq);

        rho0 = U[RHO] / gamma; //Watch out you may need this for a more complicated EOS

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        #if (DOHELM_TEMPERATURE)
        eos_mode_rhotemp_w_pres_u(gpu_eos_table, rho0, &prim[UU],
            #if(DO_YE)
            ye, 
            #else
            1.0, 
            #endif
            w - rho0, &p_new, &u
            #if (DONUCLEAR)
            , &prim[XALPHA], &prim[XATM]
            #endif
        );
        #else
        eos_mode_rhow_pres_u (gpu_eos_table, rho0, w-rho0, 
            #if (DO_YE)
			ye,
			#else 
			1.0,
			#endif
            &p_new, &u);
        #endif
        #else
            #if(TWO_T)
            gamma_eos = set_S_w(S, rho0, w, fel);
            #else
            gamma_eos = GAMMA;
            #endif
        // Ideal gas EOS
        u = (w - rho0) / gamma_eos;
        p_new = (gamma_eos - 1.) * u;
        #endif
    }

    //If density or internal energy is negative return error code
    if ((rho0 < 0.0)) return(5);
    if ((p_new < 0.0) && (lim == BASIC)) return(6);

    prim[RHO] = rho0;
    #if (!DOHELM_TEMPERATURE)
    prim[UU] = u;
    #endif

    //Set 4-velocities
    for (i = 1; i < 4; i++) {
        Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
        prim[UTCON1 + i - 1] = gamma / (z + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / z);
    }

    /* done! */
    return(0);
}