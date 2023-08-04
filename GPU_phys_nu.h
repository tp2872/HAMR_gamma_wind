// Neutrino functions
#if(NEUTRINOS_M1)
// U2P NU
__device__ int Rtoprim_nu(double* U, struct of_geom* geom, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim) {
    double U_tmp[NPR_NU], prim_tmp[NPR_NU];
    int i, ret = 0;
    double alpha;
    double ucon[NDIM], ucov[NDIM];
    ucon_calc(prim, geom, ucon);
    lower(ucon, geom->gcov, ucov);

    //Set the geometry variables
    alpha = 1.0 / sqrt(-gcon[0]);

    for (int sp = 0; sp < NU_SPECIES; sp++) {
        //Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
        for (i = 0; i < NPR_NU; i++) U_tmp[i] = alpha * U[i + index_nu(UU_NU, sp)] / gdet;

        //Transform the PRIMITIVE variables into the new system
        for (i = 0; i < NPR_NU; i++) prim_tmp[i] = prim[i + index_nu(UU_NU, sp)];

        //Do inversion
        // add a for loop for 3 neutrino fluids
        ret += Rtoprim_nu_calc(U_tmp, ucon, ucov, gcov, gcon, gdet, prim_tmp, y_max, lim);

        //Transform new primitive variables back if there was no problem
        for (i = 0; i < NPR_NU; i++) prim[i + index_nu(UU_NU, sp)] = prim_tmp[i];
    }

    return(ret);
}

#if (1)
__device__ int Rtoprim_nu_calc(double* U, double* ucon, double* ucov, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim) {

    double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
    double Uabs, qsq;
    double gammasq, y, pressure, f;
    double Tnu;
    int i, returnval = 0;

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

    if (y > y_max || y < 0. || isnan(Qdotn) || Qdotn > 0.0 || isnan(prim[1]) || isnan(prim[2]) || isnan(prim[3])) {
        Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
        for (i = 1; i < 4; i++)prim[i] = GAMMAMAX_NU * Qtcon[i] / Uabs;

        qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
            + 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
        if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
        gammasq = 1. + qsq;

        f = sqrt((GAMMAMAX_NU * GAMMAMAX_NU - 1.) / (gammasq - 1.));
        prim[1] *= f;
        prim[2] *= f;
        prim[3] *= f;

        if (lim == TYPE2) {
            /*
            // if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
            Qdotn = -(1e-30 + sqrt(fabs(Qtsq) / y_max));
            // }

            //Get gammasq
            gammasq = (2. - y_max + sqrt(4. - 3. * y_max)) / (4. - 4. * y_max);

            pressure = -Qdotn / (4. * gammasq - 1.);
            prim[0] = 1e-30 + pressure * 3.; // Erad = 3*p_rad	

            // utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
            for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

            returnval = 1;
            */

            if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
                Qdotn = -(1e-30 + sqrt(fabs(Qtsq) / y_max));
            }
            pressure = -Qdotn / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
            prim[0] = 1e-30 + pressure * 3.; // Erad = 3*p_rad	

            returnval = 1;
        }
        else if (lim == TYPE3) {
            // If energy density is negative, reset it to floor value
            if (Qdotn > 0.0) {
                prim[0] = 1.e-30;
                prim[1] = 0.;
                prim[2] = 0.;
                prim[3] = 0.;
            }
            // Causality violation: rescale!
            else if (y > y_max) {
                pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
                prim[0] = pressure * 3.; // Erad = 3*p_rad
                for (i = 1; i < 4; i++) prim[i] = Qtcon[i] / (4. * pressure * GAMMAMAX_NU);
            }
        }
        else {
            if (1) {
                // if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
                    // if (Qdotn > 0.0 || y < 0.0) {
                prim[0] = 1.e-30;
                prim[1] = 0.;
                prim[2] = 0.;
                prim[3] = 0.;
            }
            else {
                pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
                prim[0] = pressure * 3.;
            }
        }
        if (!isfinite(prim[0])) prim[0] = 1.e-30;
        if (!isfinite(prim[1])) prim[1] = 0.0;
        if (!isfinite(prim[2])) prim[2] = 0.0;
        if (!isfinite(prim[3])) prim[3] = 0.0;
        returnval = 1;
    }

    double ucon_nu[NDIM];
    qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
        + 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
    if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
    double gamma = sqrt(1. + qsq);
    ucon_nu[0] = gamma * ncon[0];
    for (i = 1; i < 4; i++) ucon_nu[i] = prim[i] + gamma * ncon[i];

    double u_dot_unu = 0.0;
    for (i = 0; i < 4; i++) u_dot_unu += ucov[i] * ucon_nu[i];

    prim[4] = U[4] * 3 * ncon[0] / (4. * ucon_nu[0] + ucon[0] / u_dot_unu);
    if (prim[4] < 0.0 || returnval) {
        Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
        prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
        prim[4] *= 3 * (-ncov * ucon[0]) / (4. * (ncov * ncov * ucon[0] * ucon[0]) - 1.);
        returnval = 1;
    }

    return(returnval);
}
#else
__device__ int Rtoprim_nu_calc(double* U, double* ucon, double* ucov, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim) {

    double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
    double Uabs, qsq;
    double gammasq, y, pressure, f;
    double Tnu;
    int i, returnval = 0;

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

    if (y > y_max || y < 0. || isnan(Qdotn) || Qdotn > 0.0 || isnan(prim[1]) || isnan(prim[2]) || isnan(prim[3])) {
        Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
        for (i = 1; i < 4; i++)prim[i] = GAMMAMAX_NU * Qtcon[i] / Uabs;

        qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
            + 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
        if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
        gammasq = 1. + qsq;

        f = sqrt((GAMMAMAX_NU * GAMMAMAX_NU - 1.) / (gammasq - 1.));
        prim[1] *= f;
        prim[2] *= f;
        prim[3] *= f;

        if (lim == TYPE2) {
            if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
                Qdotn = -(1e-30 + sqrt(fabs(Qtsq) / y_max));
            }
            pressure = -Qdotn / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
            prim[0] = 1e-30 + pressure * 3.; // Erad = 3*p_rad	

            returnval = 1;
        }
        else if (lim == TYPE3) {
            // If energy density is negative, reset it to floor value
            if (Qdotn > 0.0) {
                prim[0] = 1.e-30;
                prim[1] = 0.;
                prim[2] = 0.;
                prim[3] = 0.;
            }
            // Causality violation: rescale!
            else if (y > y_max) {
                pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
                prim[0] = pressure * 3.; // Erad = 3*p_rad
                for (i = 1; i < 4; i++) prim[i] = Qtcon[i] / (4. * pressure * GAMMAMAX_NU);
            }
        }
        else {
            if (y < 1. - 100. * NUMEPSILON || Qdotn > 0.0) {
                //if (Qdotn > 0.0 || y < 0.0) {
                prim[0] = 1.e-30;
                prim[1] = 0.;
                prim[2] = 0.;
                prim[3] = 0.;
            }
            else {
                //else if (y > y_max) {
                pressure = fabs(Qdotn) / (4. * GAMMAMAX_NU * GAMMAMAX_NU - 1.);
                prim[0] = pressure * 3.; // Erad = 3*p_rad
                //for (i = 1; i < 4; i++) prim[i] = Qtcon[i] / (4. * pressure * GAMMAMAX_NU);
                //prim[4] = U[4] / GAMMAMAX_NU;
            }
            prim[0] = 1.e-30;
            prim[1] = 0.;
            prim[2] = 0.;
            prim[3] = 0.;
        }
        if (!isfinite(prim[0]))prim[0] = 1.e-30;
        if (!isfinite(prim[1]))prim[1] = 0.0;
        if (!isfinite(prim[2]))prim[2] = 0.0;
        if (!isfinite(prim[3]))prim[3] = 0.0;

        //Floor on photon number+
        //prim[4] = U[4];
        returnval = 1;
    }

    double ucon_nu[NDIM];
    qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
        + 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
    if (qsq < 0. || fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
    double gamma = sqrt(1. + qsq);
    ucon_nu[0] = gamma * ncon[0];
    for (i = 1; i < 4; i++) ucon_nu[i] = prim[i] + gamma * ncon[i];

    double u_dot_unu = 0.0;
    for (i = 0; i < 4; i++) u_dot_unu += ucov[i] * ucon_nu[i];

    prim[4] = U[4] * 3 * ncon[0] / (4. * ucon_nu[0] + ucon[0] / u_dot_unu);
    if (prim[4] < 0.0 || returnval) {
        /*if (prim[0] > 1e-30) printf("\n\tNR<0: %e, E: %e, UN: %e, 4ucon_nu0: %e, ucon0/udotunu: %e\n", prim[4], prim[0], U[4], 4. * ucon_nu[0], ucon[0] / u_dot_unu);*/
        Tnu = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
        prim[4] = prim[0] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
        prim[4] *= 3 * (-ncov * ucon[0]) / (4. * (ncov * ncov * ucon[0] * ucon[0]) - 1.);
        returnval = 1;
    }
    return(returnval);
}
#endif

__device__ void primtoflux_nu(double* pr, struct of_state_nu* q_nu, int dir, struct of_geom* geom, double* flux) {
    int k;
    double ucon[NDIM], ucov[NDIM];
    ucon_calc(pr, geom, ucon);
    lower(ucon, geom->gcov, ucov);
    for (int sp = 0; sp < NU_SPECIES; sp++) {
        mhd_calc_nu(pr, dir, &q_nu[sp], &flux[index_nu(UU_NU, sp)], sp);
        for (k = UU_NU; k <= U3_NU; k++) flux[index_nu(k, sp)] *= geom->g;

        //Flux of photon number
        double u_dot_unu = 0.;
        for (int i = 0; i < NDIM; i++) u_dot_unu += ucov[i] * q_nu[sp].ucon[i];
        flux[index_nu(NUMBER_NU, sp)] = (geom->g) * pr[index_nu(NUMBER_NU, sp)] / 3. * (4. * q_nu[sp].ucon[dir] + ucon[dir] / u_dot_unu);
    }

    return;
}

// Neutrino energy-momentum tensor
__device__ void mhd_calc_nu(double* pr, int dir, struct of_state_nu* q_nu, double* mhd_nu, int species) {
    int j;
    /* single row of mhd stress tensor, first index up, second index down */
    DLOOPA mhd_nu[j] = 4. / 3. * pr[index_nu(UU_NU, species)] * q_nu->ucon[dir] * q_nu->ucov[j] + 1. / 3. * pr[index_nu(UU_NU, species)] * delta(dir, j);
}

__device__ void get_state_nu(double* pr, struct of_geom* geom, struct of_state_nu* q_nu, int species)
{
    /* get radiation ucon */
    ucon_calc_nu(pr, geom, q_nu->ucon, species);
    lower(q_nu->ucon, geom->gcov, q_nu->ucov);

    return;
}

__device__ void ucon_calc_nu(double* pr, struct of_geom* geom, double* ucon_nu, int species)
{
    double alpha, gamma;
    double beta[NDIM];
    int j;

    alpha = 1. / sqrt(-geom->gcon[0]);
#pragma unroll 4
    SLOOPA beta[j] = geom->gcon[j] * alpha * alpha;

    gamma_calc_nu(pr, geom, &gamma, species);

    ucon_nu[0] = gamma / alpha;
#if AMD
#pragma unroll 4
    SLOOPA ucon_nu[j] = fma(-gamma, beta[j] / alpha, pr[index_nu(U1_NU, species) + j - 1]);
#else
#pragma unroll 4
    SLOOPA ucon_nu[j] = pr[index_nu(U1_NU, species) + j - 1] - gamma * beta[j] / alpha;
#endif

    return;
}

__device__ int gamma_calc_nu(double* pr, struct of_geom* geom, double* gamma_nu, int species)
{
    double qsq_nu;
#if AMD
    qsq_nu = fma(geom->gcov[4], pr[index_nu(U1_NU, species)] * pr[index_nu(U1_NU, species)], fma(geom->gcov[7], pr[index_nu(U2_NU, species)] * pr[index_nu(U2_NU, species)], geom->gcov[9] * pr[index_nu(U3_NU, species)] * pr[index_nu(U3_NU, species)]))
        + 2. * fma(geom->gcov[5], pr[index_nu(U1_NU, species)] * pr[index_nu(U2_NU, species)], fma(geom->gcov[6], pr[index_nu(U1_NU, species)] * pr[index_nu(U3_NU, species)], geom->gcov[8] * pr[index_nu(U2_NU, species)] * pr[index_nu(U3_NU, species)]));
#else
    qsq_nu = geom->gcov[4] * pr[index_nu(U1_NU, species)] * pr[index_nu(U1_NU, species)] + geom->gcov[7] * pr[index_nu(U2_NU, species)] * pr[index_nu(U2_NU, species)] + geom->gcov[9] * pr[index_nu(U3_NU, species)] * pr[index_nu(U3_NU, species)]
        + 2. * (geom->gcov[5] * pr[index_nu(U1_NU, species)] * pr[index_nu(U2_NU, species)] + geom->gcov[6] * pr[index_nu(U1_NU, species)] * pr[index_nu(U3_NU, species)] + geom->gcov[8] * pr[index_nu(U2_NU, species)] * pr[index_nu(U3_NU, species)]);
#endif

    if (qsq_nu < 0.) {
        if (fabs(qsq_nu) > 1.E-10) { // then assume not just machine precision
            *gamma_nu = 1.;
            return (1);
        }
        else qsq_nu = 1.E-10; // set floor
    }

    *gamma_nu = sqrt(1. + qsq_nu);

    return(0);
}


__device__ void vchar_nu(double* pr, struct of_state* q, struct of_state_nu* q_nu, struct of_geom* geom, int dir, double* vmax, double* vmin, double dx, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table) {
    double discr, vp, vm, tau, kappa_tot, crad2, cmin_nu, cmax_nu, cmin_mhd, cmax_mhd, bsq, Tr;
    double Acon_0, Acon_js;
    double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;

    double vmax_tmp, vmin_tmp; //edit 4
    *vmax = 1.0;
    *vmin = 0.0;

    if (dir == 1) {
        Acon_0 = geom->gcon[1];
        Acon_js = geom->gcon[4];
    }
    else if (dir == 2) {
        Acon_0 = geom->gcon[2];
        Acon_js = geom->gcon[7];
    }
    else if (dir == 3) {
        Acon_0 = geom->gcon[3];
        Acon_js = geom->gcon[9];
    }

    double ener_nu_avg;
    for (int sp = 0; sp < NU_SPECIES; sp++) {
        /* find radiation wave speed at 1./3. speed of light (==isotropic in radiation frame) */
        crad2 = 1.0 / 3.0;
        /* now require that speed of wave measured by observer q->ucon is crad2 */
        Asq = Acon_js;
        Bsq = geom->gcon[0];// dot(Bcon, Bcov);
        Au = q_nu[sp].ucon[dir];
        Bu = q_nu[sp].ucon[0];
        AB = Acon_0;
        Au2 = Au * Au;
        Bu2 = Bu * Bu;
        AuBu = Au * Bu;

        A = Bu2 - (Bsq + Bu2) * crad2;
        B = 2. * (AuBu - (AB + AuBu) * crad2);
        C = Au2 - (Asq + Au2) * crad2;

        discr = B * B - 4. * A * C;
        if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
        else if (discr < -1.e-10)discr = 0.;

        discr = sqrt(discr);
        vp = -(-B + discr) / (2. * A);
        vm = -(-B - discr) / (2. * A);

        if (vp > vm) {
            cmax_nu = vp;
            cmin_nu = vm;
        }
        else {
            cmax_nu = vm;
            cmin_nu = vp;
        }

        /* find radiation wave speed in fluid frame based on optical depth */
        // calculate avg. energy of neutrinos
        calc_avg_neutrino_energy(pr, &ener_nu_avg, geom, sp);

        //Calculate optical depth
        kappa_tot = (calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, pr, ener_nu_avg, sp)) + (calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, pr, ener_nu_avg, sp)) + SMALL; // to make it non-zero
        tau = kappa_tot * sqrt(geom->gcov[(dir == 1) * 4 + (dir == 2) * 7 + (dir == 3) * 9]) * dx;
        crad2 = 16. / (9. * tau * tau);
        if (isnan(tau)) crad2 = 1.0; // given that tau << 1 in post-merger scenario, this simplifies things

        /* check on it! */
        if (crad2 < 0.) crad2 = SMALL;
        if (crad2 > 1.) crad2 = 1.;

        /* now require that speed of wave measured by observer q->ucon is crad2 */
        Au = q->ucon[dir];
        Bu = q->ucon[0];
        Au2 = Au * Au;
        Bu2 = Bu * Bu;
        AuBu = Au * Bu;

        A = Bu2 - (Bsq + Bu2) * crad2;
        B = 2. * (AuBu - (AB + AuBu) * crad2);
        C = Au2 - (Asq + Au2) * crad2;

        discr = B * B - 4. * A * C;
        if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
        else if (discr < -1.e-10) discr = 0.;

        discr = sqrt(discr);
        vp = -(-B + discr) / (2. * A);
        vm = -(-B - discr) / (2. * A);

        if (vp > vm) {
            cmax_mhd = vp;
            cmin_mhd = vm;
        }
        else {
            cmax_mhd = vm;
            cmin_mhd = vp;
        }

        vmax[sp] = MY_MIN(cmax_mhd, cmax_nu);
        vmin[sp] = MY_MAX(cmin_mhd, cmin_nu);
    }

    return;
}

// For now, assume that we are dealing with a single species of neutrinos
// Caveat 1: EOS should provide chemical potential value + gas temperature

__device__ double calc_nu_kappa_emiss(const double* __restrict__ gpu_nulib_table, double* ph, int sp) {
    double kappa_emiss;
    interp_nulib_check_bounds(gpu_nulib_table, ph, sp, NU_EMISSIVITY, &kappa_emiss);

    // Multiply by applicable units
#if (ZERO_TAU_MODE)
    return (0.0);
#else
    return kappa_emiss * R_G_CGS / C_CGS / (ENERGY_DENSITY_SCALE); // erg/cm^3/s --> erg/cm^4 * Rg
#endif
}

__device__ double calc_nu_kappa_abs(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp) {
    double kappa_abs, Tnu_over_Tgas;

    // kappa_abs = equilibrium absorption opacity (gas in equilibrium with neutrinos)
    interp_nulib_check_bounds(gpu_nulib_table, ph, sp, NU_ABSORPTION, &kappa_abs);

    // need to multiply kappa_abs by (T_nu/T_gas)^2
    calc_neutrino_temperature(gpu_eos_table, ph, ener_nu_avg, &Tnu_over_Tgas, sp);

    // Multiply by applicable units
#if (ZERO_TAU_MODE)
    return (0.0);
#else
    return (kappa_abs * pow(Tnu_over_Tgas, 2.0) * R_G_CGS); // 1/cm --> Rg/cm
#endif
}

__device__ double calc_nu_kappa_scatt(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp) {
    double kappa_scatt, Tnu_over_Tgas;

    // kappa_abs = equilibrium absorption opacity (gas in equilibrium with neutrinos)
    interp_nulib_check_bounds(gpu_nulib_table, ph, sp, NU_SCATTERING, &kappa_scatt);

    // need to multiply kappa_abs by (T_nu/T_gas)^2
    calc_neutrino_temperature(gpu_eos_table, ph, ener_nu_avg, &Tnu_over_Tgas, sp);

    // Multiply by applicable units
#if (ZERO_TAU_MODE)
    return (0.0);
#else
    return (kappa_scatt * pow(Tnu_over_Tgas, 2.0) * R_G_CGS);
#endif
}

/* for neutrino number density evolution: */
__device__ double calc_nu_number_emiss(const double* __restrict__ gpu_nulib_table, double* ph, int sp) {
    double kappa_emiss;
    interp_nulib_check_bounds(gpu_nulib_table, ph, sp, NU_EMISSIVITY_N, &kappa_emiss);

    // Multiply by applicable units
#if (ZERO_TAU_MODE)
    return (0.0);
#else
    return kappa_emiss * (R_G_CGS / C_CGS) / MASS_DENSITY_SCALE; // 1/cm^3
#endif
}

__device__ double calc_nu_number_abs(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp) {
    double kappa_emiss, kappa_number_emiss, kappa_abs, Tnu_over_Tgas;

    kappa_emiss = calc_nu_kappa_emiss(gpu_nulib_table, ph, sp);
    kappa_number_emiss = calc_nu_number_emiss(gpu_nulib_table, ph, sp);
    kappa_abs = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, sp);

    //calc_neutrino_temperature(gpu_eos_table, ph, ener_nu_avg, &Tnu_over_Tgas, sp);

    double mu_ele;
    eos_mode_rhotemp_etaele(gpu_eos_table, ph[RHO], ph[UU], ph[YE], &mu_ele
        #if (DONUCLEAR)
        , &ph[XALPHA], &ph[XATM]
        #endif
    );

    double mu_n, mu_p, mu_nu;
    calc_mu_np(ph[RHO], ph[UU], ph[YE], &mu_n, &mu_p
        #if (DONUCLEAR)
        , ph[XALPHA], ph[XATM]
        #endif
    );
    mu_nu = mu_p + mu_ele - mu_n + (MP_CGS + ME_CGS - MN_CGS) * C_CGS * C_CGS / (BOLTZ_CGS * ph[UU]);

    double F2, F3;

    // 0 == electron neutrino
    // 1 == electron antineutrino (mu_nua = - mu_nu)
    // 2 == heavy lepton neutrinos (mu_nux = 0)
    if (sp == 0) {
        F2 = calc_fermiint2(mu_nu);
        F3 = calc_fermiint3(mu_nu);
    }
    else if (sp == 1) {
        F2 = calc_fermiint2(-mu_nu);
        F3 = calc_fermiint3(-mu_nu);
    }
    else if (sp == 2) {
        F2 = calc_fermiint2(0.0);
        F3 = calc_fermiint3(0.0);
    }

#if (ZERO_TAU_MODE)
    return (0.0);
#else
    // Multiply by applicable units
    if (kappa_emiss == 0.) {
        return 0.;
    }
    else {
        /* Notes:
            1. 1 / MASS_DENSITY_SCALE factor is already in eta_N
            2. [kappa_N] = erg/cm^3 / energy density scale
        */
        //return kappa_abs * (kappa_number_emiss / kappa_emiss) * F3 / (F2 + 1e-30) * (BOLTZ_CGS * ph[UU]) / (C_CGS * C_CGS) / (MASS_DENSITY_SCALE);
        return kappa_abs * (kappa_number_emiss / kappa_emiss) * F3 / (F2 + 1e-30) * (BOLTZ_CGS * ph[UU]) / (ENERGY_DENSITY_SCALE);
    }
#endif
}

/* NULIB tables part */
__device__ void interp_nulib_check_bounds(const double* __restrict__ gpu_nulib_table, double* ph, int species, int quantity, double* opacity) {
    double factor = 1.0;
    double rho = ph[RHO] * MASS_DENSITY_SCALE;
    double Tgas = ph[UU];
    double ye = ph[YE];

    // Density bounds:
    double nulib_rho_low = pow(10., nulib_dlo);
    double nulib_rho_high = 0.5 * pow(10., nulib_dhi);
    if (rho < nulib_rho_low || rho > nulib_rho_high) {
        factor = 0.0;
        rho = nulib_rho_low;
        //*opacity = 0.0; // DINU: reset all opacities for now; later could add extrapolation from the lower bound of the table
        //return;
    }

    // Electron fraction bounds:
#if (NULIB_YE_CORRECTION)
    double correctionEtaNue = 1.0;
    double correctionEtaNua = 1.0;

    if (ye < nulib_yelo_threshold) {
        correctionEtaNue = (ye < nulib_ylo ? 0. : (ye - nulib_ylo) / (nulib_yelo_threshold - nulib_ylo));
    }
    if (ye > nulib_yehi_threshold) {
        correctionEtaNua = (ye > nulib_yhi ? 0. : (nulib_yhi - ye) / (nulib_yhi - nulib_yehi_threshold));
    }

    if (species == 0) {
        if (quantity == NU_EMISSIVITY) {
            factor *= correctionEtaNue * correctionEtaNua;
        }
        else if (quantity == NU_ABSORPTION) {
            factor *= correctionEtaNua;
        }
    }
    if (species == 1) {
        if (quantity == NU_EMISSIVITY) {
            factor *= correctionEtaNue * correctionEtaNua;
        }
        else if (quantity == NU_ABSORPTION) {
            factor *= correctionEtaNue;
        }
    }
    //#else
#endif
    if (ye < nulib_ylo || ye > nulib_yhi) {
        factor = 0.0; // DINU: reset for now; 
        ye = nulib_ylo;
    }
    //#endif

    // Temperature bounds:
    double nulib_temp_low = pow(10., 9.77); // corresponds to 0.511 MeV
    double nulib_temp_high = 0.5 * pow(10., nulib_thi);
    if (ph[UU] < nulib_temp_low) {
        // Based on quantity, get the correct scaling
        if (quantity == NU_EMISSIVITY) {
            factor *= pow((Tgas / nulib_temp_low), 6.0);
        }
        else if (quantity == NU_ABSORPTION || quantity == NU_SCATTERING) {
            factor *= pow((Tgas / nulib_temp_low), 2.0);
        }
        else { // DINU: idk, number emissivity
            factor *= pow((Tgas / nulib_temp_low), 6.0);
        }
        Tgas = nulib_temp_low;
    }
    if (ph[UU] > nulib_temp_high) {
        factor = 0.0;
        Tgas = nulib_temp_high;
    }

    if (factor == 0.0) {
        *opacity = 0.0;
        return;
    }

    interp_nulib_table(gpu_nulib_table, rho, Tgas, ye, species, quantity, opacity);
    *opacity *= factor;
}

__device__ void interp_nulib_table(const double* __restrict__ gpu_nulib_table, double rho, double Tgas, double ye, int species, int quantity, double* opacity) {
    int iat = (int)((log10(rho) - nulib_dlo) * (double)(NULIB_RHO - 1) / (nulib_dhi - nulib_dlo)) + 1;
    int jat = (int)((log10(Tgas) - nulib_tlo) * (double)(NULIB_TEMP - 1) / (nulib_thi - nulib_tlo)) + 1;
    int kat = (int)((ye - nulib_ylo) * (double)(NULIB_YE - 1) / (nulib_yhi - nulib_ylo)) + 1;
    iat = MY_MAX(1, MY_MIN(iat, NULIB_RHO - 1)) - 1;
    jat = MY_MAX(1, MY_MIN(jat, NULIB_TEMP - 1)) - 1;
    kat = MY_MAX(1, MY_MIN(kat, NULIB_YE - 1)) - 1;

    double dstp = (nulib_dhi - nulib_dlo) / (double)(NULIB_RHO - 1);
    double tstp = (nulib_thi - nulib_tlo) / (double)(NULIB_TEMP - 1);
    double ystp = (nulib_yhi - nulib_ylo) / (double)(NULIB_YE - 1);
    double nulib_d_iat = pow(10.0, (nulib_dlo + iat * dstp));
    double nulib_t_jat = pow(10.0, (nulib_tlo + jat * tstp));
    double nulib_y_kat = (nulib_ylo + kat * ystp);
    double nulib_dd_iat = pow(10.0, (nulib_dlo + (iat + 1) * dstp)) - pow(10.0, (nulib_dlo + iat * dstp));
    double nulib_dt_jat = pow(10.0, (nulib_tlo + (jat + 1) * tstp)) - pow(10.0, (nulib_tlo + jat * tstp));
    double nulib_dy_kat = ystp;

    double xd = MY_MAX((rho - nulib_d_iat) / nulib_dd_iat, 0.0);
    double xt = MY_MAX((Tgas - nulib_t_jat) / nulib_dt_jat, 0.0);
    double xy = MY_MAX((ye - nulib_y_kat) / nulib_dy_kat, 0.0);
    double mxd = 1.0 - xd;
    double mxt = 1.0 - xt;
    double mxy = 1.0 - xy;

    *opacity = gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * mxd * mxy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * mxt * xd * mxy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * mxd * mxy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat)*NU_SPECIES + species] * xt * xd * mxy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * mxd * xy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat)*NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * mxt * xd * xy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat)*NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * mxd * xy +
        gpu_nulib_table[(quantity - 1) * (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES) + (iat + 1) * NULIB_TEMP * NULIB_YE * NU_SPECIES + (jat + 1) * NULIB_YE * NU_SPECIES + (kat + 1) * NU_SPECIES + species] * xt * xd * xy;

    return;
}

// Neutrino temperature calculation: needs EOS
__device__ void calc_neutrino_temperature(const double* __restrict__ gpu_eos_table, double* ph, double ener_nu_avg, double* Tnu_over_Tgas, int species) {

    // Call EOS to get gas temperature and electron chemical potential
    double mu_ele;
    eos_mode_rhotemp_etaele(gpu_eos_table, ph[RHO], ph[UU], ph[YE], &mu_ele
#if (DONUCLEAR)
        , &ph[XALPHA], &ph[XATM]
#endif
    );

    double mu_n, mu_p, mu_nu;
    calc_mu_np(ph[RHO], ph[UU], ph[YE], &mu_n, &mu_p
        #if (DONUCLEAR)
        , ph[XALPHA], ph[XATM]
        #endif
    );
    mu_nu = mu_p + mu_ele - mu_n + (MP_CGS + ME_CGS - MN_CGS) * C_CGS * C_CGS / (BOLTZ_CGS * ph[UU]);

    double F2, F3;

#if (NU_SPECIES == 1) 
    species = 2;
#endif

    // 0 == electron neutrino
    // 1 == electron antineutrino (mu_nua = - mu_nu)
    // 2 == heavy lepton neutrinos (mu_nux = 0)
    if (species == 0) {
        F2 = calc_fermiint2(mu_nu);
        F3 = calc_fermiint3(mu_nu);
    }
    else if (species == 1) {
        F2 = calc_fermiint2(-mu_nu);
        F3 = calc_fermiint3(-mu_nu);
    }
    else if (species == 2) {
        F2 = calc_fermiint2(0.0);
        F3 = calc_fermiint3(0.0);
    }

    *Tnu_over_Tgas = (ener_nu_avg * C_CGS * C_CGS) * F2 / (F3 + 1e-30) / (BOLTZ_CGS * ph[UU]);

    // debugging:
    if (isnan(*Tnu_over_Tgas)) *Tnu_over_Tgas = 1.0;
    *Tnu_over_Tgas = MY_MAX(1.0, *Tnu_over_Tgas);
}

__device__ double calc_fermiint2(double x) {
    if (x > 0.001)
        return (x * x * x / 3.0 + 3.2899 * x) / (1.0 - exp(-1.8246 * x));
    else
        return 2.0 * exp(x) / (1.0 + 0.1092 * exp(0.8908 * x));
}

__device__ double calc_fermiint3(double x) {
    if (x > 0.001)
        return (x * x * x * x / 4.0 + 4.9348 * x * x + 11.3644) / (1.0 + exp(-1.9039 * x));
    else
        return 6.0 * exp(x) / (1.0 + 0.0559 * exp(0.9069 * x));
}

// Neutron-proton chemical potentials assuming ideal gas
// From: NuLib code
__device__ void calc_mu_np(double rho, double T_gas, double ye, double* mu_n, double* mu_p
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    #if (DONUCLEAR)
    double x_n = get_xn(ye, x_alpha);
    double x_p = get_xp(ye - x_atm, x_alpha);
    #else
    double x_n = MY_MAX(1. - ye, 1e-20);
    double x_p = MY_MAX(ye, 1e-20);
    #endif

    double n_n = x_n * rho * MASS_DENSITY_SCALE / MN_CGS;
    double n_p = x_p * rho * MASS_DENSITY_SCALE / MP_CGS;

    if (n_n > 0.0)
        *mu_n = log(0.5 * n_n * pow(PLANCK_CGS * PLANCK_CGS / (2.0 * M_PI * MN_CGS * BOLTZ_CGS * T_gas), 1.5));
    else
        *mu_n = 0.0;

    if (n_p > 0.0)
        *mu_p = log(0.5 * n_p * pow(PLANCK_CGS * PLANCK_CGS / (2.0 * M_PI * MP_CGS * BOLTZ_CGS * T_gas), 1.5));
    else
        *mu_p = 0.0;

    //coulomb correction for charged particles from Chabrier & Potkhin(1998)
    double A_1 = -0.9052;
    double A_2 = 0.6322;
    double A_3 = -sqrt(3.0) / 2.0 - A_1 / sqrt(A_2);
    double a_e = pow(4.0 / 3.0 * M_PI * rho * ye * avo, -1./3.);
    double Gamma_p = esqu / (BOLTZ_CGS * T_gas * a_e);

    double mu_p_coul = (A_1 * (sqrt(Gamma_p * (A_2 + Gamma_p)) - A_2 * log(sqrt(Gamma_p / A_2) + sqrt(1. + Gamma_p / A_2))) + 2. * A_3 * (sqrt(Gamma_p) - atan(sqrt(Gamma_p))));

    *mu_p = *mu_p + mu_p_coul;
}

#endif
