
#if (DONUCLEAR)
__device__ void nuc_evol(const double* __restrict__ gpu_eos_table, double* ph) {
    double x_n, x_p, x_alpha;
    
    // Make sure that the initial abundances are not out of bounds
    double x_alpha_ini = ph[XALPHA];
    double x_atm = ph[XATM];
    x_alpha_ini = MY_MIN(1.0, x_alpha_ini);
    x_alpha_ini = MY_MAX(1e-10, x_alpha_ini);
    x_atm = MY_MIN(1.0, x_atm);
    x_atm = MY_MAX(1e-10, x_atm);

    // Initial values of temperature and gas internal energy
    double ugas_ini, dudt;
    double tgas_ini = ph[UU];
    eos_mode_rhotemp_u_dudt(gpu_eos_table, ph[RHO], tgas_ini, ph[YE], &ugas_ini, &dudt, x_alpha_ini, x_atm);
    
    // Start N-R iterations
    double tgas_new, ugas, xa_t, Fa, dFa;
    ugas = ugas_ini;
    nse_nucevol(ph[RHO], tgas_ini, ph[YE], &x_alpha, &x_atm, &xa_t);

    int i, iter_max = 50;
    for (i = 0; i < iter_max; i++) {
        // Do a N-R iteration
        Fa = (ugas - ugas_ini) - ph[RHO] * Qalpha / m_alpha * (x_alpha - x_alpha_ini) / (ENERGY_DENSITY_SCALE);
        dFa = dudt - ph[RHO] * Qalpha / m_alpha * xa_t / (ENERGY_DENSITY_SCALE);
        
        // Update Tgas
        tgas_new = tgas_ini - Fa / dFa;

        // Make sure Tgas doesn't go outside the EOS table bounds
        tgas_new = MY_MAX(eos_temp_low, tgas_new);
        tgas_new = MY_MIN(eos_temp_up, tgas_new);

        // Get new ugas, dudt 
        eos_mode_rhotemp_u_dudt(gpu_eos_table, ph[RHO], tgas_new, ph[YE], &ugas, &dudt, x_alpha_ini, x_atm);
        // Get new Xalpha, dXa_dT 
        nse_nucevol(ph[RHO], tgas_ini, ph[YE], &x_alpha, &x_atm, &xa_t);

        // Exit if desired tolerance is reached
        if (fabs(Fa / dFa) < 1e-8 * tgas_new) break;
    }

    // Update the variables
    ph[UU] = tgas_new;
    ph[XALPHA] = x_alpha;
    ph[XATM] = x_atm;

    return;
}
#endif

#if (DONUCLEAR)
__device__ double get_xp (double ye, double x_alpha) {
    return (ye - 0.5 * x_alpha);
}

__device__ double get_xn(double ye, double x_alpha) {
    return (1. - ye - 0.5 * x_alpha);
}

__device__ void nse_abundances(double rho, double tgas, double ye, double* x_n, double* x_p, double* x_alpha) {
    double n = rho / amu;
    double n_Q = pow((2.0 * M_PI * amu * BOLTZ_CGS * tgas / (PLANCK_CGS * PLANCK_CGS)), 1.5);
    double Fa, dFa;
    int i, max_iter = 50;

    *x_alpha = ye;
    for (i = 0; i < max_iter; i++) {
        *x_n = get_xn(ye, *x_alpha);
        *x_p = get_xp(ye, *x_alpha);

        Fa = pow((*x_n) * (*x_p), 2.0) - 0.5 * (*x_alpha) * pow((n_Q / n), 3.0) * exp(-Qalpha / (BOLTZ_CGS * tgas));
        dFa = - (*x_p) * pow((*x_n), 2.0) - (*x_n) * pow((*x_p), 2.0) - 0.5 * pow((n_Q / n), 3.0) * exp(-Qalpha / (BOLTZ_CGS * tgas));
        *x_alpha -= Fa / dFa;

        if (fabs(Fa / dFa) < 1e-8 * (*x_alpha)) break;
    }

    *x_n = get_xn(ye, *x_alpha);
    *x_p = get_xp(ye, *x_alpha);

    return;
}

__device__ void nse_derivatives(double rho, double tgas, double ye, double x_n, double x_p, double x_alpha, double* xn_d, double* xn_t, double* xn_y, double* xp_d, double* xp_t, double* xp_y, double* xa_d, double* xa_t, double* xa_y) {
    *xa_d = (3.0 / rho) * x_n * x_p * x_alpha / (x_n * x_p + x_n * x_alpha + x_p * x_alpha);
    *xa_t = -1.0 / tgas * (4.5 + Qalpha / BOLTZ_CGS / tgas) * x_n * x_p * x_alpha / (x_n * x_p + x_n * x_alpha + x_p * x_alpha);
    *xa_y = 2.0 * (x_n - x_p) * x_alpha / (x_n * x_p + x_n * x_alpha + x_p * x_alpha);

    *xn_d = -0.5 * (*xa_d);
    *xn_t = -0.5 * (*xa_t);
    *xn_y = -1.0 - 0.5 * (*xa_y);

    *xp_d = (*xn_d);
    *xp_t = (*xn_t);
    *xp_y = 1.0 - 0.5 * (*xa_y);

    return;
}

__device__ void nse_nucevol(double rho, double tgas, double ye, double* x_alpha, double* x_atm, double* xa_t) {
    double x_n, x_p, xn_d, xn_t, xn_y, xp_d, xp_t, xp_y, xa_d, xa_y;	
    double x_alpha_tmp = *x_alpha;
    double x_atm_tmp = *x_atm;
    // Compute abundances 
    if (x_atm_tmp < x_atm_cutoff && tgas > tgas_cutoff) {
        x_atm_tmp = 0.0;
        nse_abundances(rho * MASS_DENSITY_SCALE, tgas, ye, &x_n, &x_p, &x_alpha_tmp);
        nse_derivatives(rho * MASS_DENSITY_SCALE, tgas, ye, x_n, x_p, x_alpha_tmp, &xn_d, &xn_t, &xn_y, &xp_d, &xp_t, &xp_y, &xa_d, xa_t, &xa_y);
    }
    else {
        x_n = get_xn(ye, x_alpha_tmp);
        x_p = get_xp(ye, x_alpha_tmp);
        // set derivatives
        // normalize
        double x_sum = x_n + x_p + x_alpha_tmp + x_atm_tmp;
        if (x_sum > 1.0) {
            x_n = x_n / x_sum;
            x_p = x_p / x_sum;
            x_alpha_tmp = x_alpha_tmp / x_sum;
            x_atm_tmp = x_atm_tmp / x_sum;
        }
        *xa_t = 0.0;
    }

    x_alpha_tmp = MY_MAX(x_alpha_tmp, 1e-10);
    x_atm_tmp = MY_MAX(x_atm_tmp, 1e-10);
    
    x_alpha_tmp = MY_MIN(x_alpha_tmp, 1.0);
    x_atm_tmp = MY_MIN(x_atm_tmp, 1.0);

    *x_alpha = x_alpha_tmp;
    *x_atm = x_atm_tmp;

    return;
}
#endif

// EOS function calls
//__device__ void validate_ye(double* ye) {
//    if (*ye < 0.0) *ye = 0.0;
//    if (*ye > 1.0) *ye = 1.0;
//    return;
//}
