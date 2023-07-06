
/* find relative 4-velocity from 4-velocity (both in code coords) */
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon)
{
	double alpha, beta[NDIM], gamma;
	int j;

	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	alpha = 1. / sqrt(-geom->gcon[0]);
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;
	gamma = alpha*ucon[0];

	utcon[0] = 0;
	SLOOPA utcon[j] = ucon[j] + gamma*beta[j] / alpha;
}

__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
{
	double AA, BB, CC, DD, one_over_alpha_sq;
	
	//compute the Lorentz factor based on contravariant 3-velocity
	AA = geom->gcov[0];
	BB = 2.*(geom->gcov[1] * vcon[1] +geom->gcov[2] * vcon[2] +geom->gcov[3] * vcon[3]);
	CC = geom->gcov[4] * vcon[1] * vcon[1] +geom->gcov[7] * vcon[2] * vcon[2] +geom->gcov[9] * vcon[3] * vcon[3] +2.*(geom->gcov[5] * vcon[1] * vcon[2] +geom->gcov[6] * vcon[1] * vcon[3] + geom->gcov[8] * vcon[2] * vcon[3]);
	DD = -1. / (AA + BB + CC);
	one_over_alpha_sq = -geom->gcon[0];
	if (DD<one_over_alpha_sq) DD = one_over_alpha_sq;

	*ut = sqrt(DD);
}

/* add in geometrical and cooling source terms to equations of motion */
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *  dU, double Dt, const  double* __restrict__ conn_GPU, struct of_state *  q, double r
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (TWO_T)
    , double gamma_g
    #endif
) {
    double mhd[NDIM][NDIM];
    int k, j, dir;
    double conn, P, u, w, bsq, eta, ptot;
    #if(NSY)
    int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
    int global_id = icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
    #else
    int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
    int global_id = icurr*(BS_2 + 2 * N2G) + jcurr;
    #endif

    #if (DOHELM)
    // Helmholtz EOS
    #if (DOHELM_TEMPERATURE)
    eos_mode_rhotemp_pres_u(gpu_eos_table, ph[RHO], ph[UU],
        #if (DO_YE)
        ph[YE],
        #else 
        1.0,
        #endif
        &P, &u
        #if (DONUCLEAR)
        , &ph[XALPHA], &ph[XATM]
        #endif
    );
    #else
    eos_mode_rhou_pres(gpu_eos_table, ph[RHO], ph[UU]
		#if (DO_YE)
		, ph[YE]
		#else
		, 1.0
		#endif
		, &P);
    #endif
    #elif(TWO_T)
    P = (gamma_g - 1.) * ph[UU];
    #else
    // Ideal gas EOS
    P = (GAMMA - 1.)*ph[UU];
    #endif

    #if (!DOHELM_TEMPERATURE)
    u = ph[UU];
    #endif

    w = P + ph[RHO] + u;
    bsq = dot(q->bcon, q->bcov);
    eta = w + bsq;
    #if AMD
    ptot = fma(0.5, bsq, P);
    #else
    ptot = P + 0.5*bsq;
    #endif

    /* single row of mhd stress tensor,
    * first index up, second index down */
    for (dir = 0; dir < NDIM; dir++){
        #if AMD
        #pragma unroll 4
        DLOOPA mhd[dir][j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
        #else
        DLOOPA mhd[dir][j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
        #endif
    }

    /* contract mhd stress tensor with connection */
    #pragma unroll 9
    PLOOP dU[k] = 0.;

    #pragma unroll 4
    for (k = 0; k<NDIM; k++){
        #if(NSY)
        dU[UU] += mhd[0][k] * conn_GPU[0 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[1][k] * conn_GPU[4 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2] += mhd[2][k] * conn_GPU[7 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U3] += mhd[3][k] * conn_GPU[9 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        conn = conn_GPU[1 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[1][k] * conn;
        dU[U1] += mhd[0][k] * conn;
        conn = conn_GPU[2 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[2][k] * conn;
        dU[U2] += mhd[0][k] * conn;
        conn = conn_GPU[3 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[3][k] * conn;
        dU[U3] += mhd[0][k] * conn;
        conn = conn_GPU[5 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[2][k] * conn;
        dU[U2] += mhd[1][k] * conn;
        conn = conn_GPU[6 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[3][k] * conn;
        dU[U3] += mhd[1][k] * conn;
        conn = conn_GPU[8 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2] += mhd[3][k] * conn;
        dU[U3] += mhd[2][k] * conn;
        #else
        dU[UU] += mhd[0][k] * conn_GPU[0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[1][k] * conn_GPU[4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2] += mhd[2][k] * conn_GPU[7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U3] += mhd[3][k] * conn_GPU[9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        conn = conn_GPU[1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[1][k] * conn;
        dU[U1] += mhd[0][k] * conn;
        conn = conn_GPU[2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[2][k] * conn;
        dU[U2] += mhd[0][k] * conn;
        conn = conn_GPU[3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU] += mhd[3][k] * conn;
        dU[U3] += mhd[0][k] * conn;
        conn = conn_GPU[5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[2][k] * conn;
        dU[U2] += mhd[1][k] * conn;
        conn = conn_GPU[6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1] += mhd[3][k] * conn;
        dU[U3] += mhd[1][k] * conn;
        conn = conn_GPU[8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2] += mhd[3][k] * conn;
        dU[U3] += mhd[2][k] * conn;
        #endif
    }

    //Add cooling term if needed
    #if (COOL_DISK)
    misc_source(ph, icurr, jcurr, geom, q, dU, r, Dt
		#if (DOHELM)
		, gpu_eos_table
		#endif
	);
    #endif

    dU[UU] *= geom->g;
    dU[U1] *= geom->g;
    dU[U2] *= geom->g;
    dU[U3] *= geom->g;
    dU[KTOT] *= geom->g;

    //Add M1 radiation terms
    #if(RAD_M1)
    double mhd_rad[NDIM][NDIM];
    struct of_state_rad q_rad;
    get_state_rad(ph, geom, &q_rad);
    mhd_calc_rad(ph, 0, &q_rad, mhd_rad[0]);
    mhd_calc_rad(ph, 1, &q_rad, mhd_rad[1]);
    mhd_calc_rad(ph, 2, &q_rad, mhd_rad[2]);
    mhd_calc_rad(ph, 3, &q_rad, mhd_rad[3]);

    //contract radiation stress tensor with connection
    #pragma unroll 4	
    for (k = 0; k<NDIM; k++) {
        #if(NSY)
        dU[UU_RAD] += mhd_rad[0][k] * conn_GPU[0 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[1][k] * conn_GPU[4 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2_RAD] += mhd_rad[2][k] * conn_GPU[7 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U3_RAD] += mhd_rad[3][k] * conn_GPU[9 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        conn = conn_GPU[1 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[1][k] * conn;
        dU[U1_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[2 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[2][k] * conn;
        dU[U2_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[3 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[5 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[2][k] * conn;
        dU[U2_RAD] += mhd_rad[1][k] * conn;
        conn = conn_GPU[6 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[1][k] * conn;
        conn = conn_GPU[8 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[2][k] * conn;
        #else
        dU[UU_RAD] += mhd_rad[0][k] * conn_GPU[0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[1][k] * conn_GPU[4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2_RAD] += mhd_rad[2][k] * conn_GPU[7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U3_RAD] += mhd_rad[3][k] * conn_GPU[9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        conn = conn_GPU[1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[1][k] * conn;
        dU[U1_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[2][k] * conn;
        dU[U2_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[UU_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[0][k] * conn;
        conn = conn_GPU[5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[2][k] * conn;
        dU[U2_RAD] += mhd_rad[1][k] * conn;
        conn = conn_GPU[6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U1_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[1][k] * conn;
        conn = conn_GPU[8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
        dU[U2_RAD] += mhd_rad[3][k] * conn;
        dU[U3_RAD] += mhd_rad[2][k] * conn;
        #endif
    }
    dU[UU_RAD] *= geom->g;
    dU[U1_RAD] *= geom->g;
    dU[U2_RAD] *= geom->g;
    dU[U3_RAD] *= geom->g;
    #endif

    // Neutrinos source terms:
    #if(NEUTRINOS_M1)
    double mhd_nu[NDIM][NDIM];
    struct of_state_nu q_nu[NU_SPECIES];
    for (int sp = 0; sp < NU_SPECIES; sp++) {
        get_state_nu(ph, geom, &q_nu[sp], sp);
        mhd_calc_nu(ph, 0, &q_nu[sp], mhd_nu[0], sp);
        mhd_calc_nu(ph, 1, &q_nu[sp], mhd_nu[1], sp);
        mhd_calc_nu(ph, 2, &q_nu[sp], mhd_nu[2], sp);
        mhd_calc_nu(ph, 3, &q_nu[sp], mhd_nu[3], sp);

        //contract radiation stress tensor with connection
        #pragma unroll 4	
        for (k = 0; k < NDIM; k++) {
            #if(NSY)
            dU[index_nu(UU_NU, sp)] += mhd_nu[0][k] * conn_GPU[0 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[1][k] * conn_GPU[4 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U2_NU, sp)] += mhd_nu[2][k] * conn_GPU[7 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U3_NU, sp)] += mhd_nu[3][k] * conn_GPU[9 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            conn = conn_GPU[1 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[1][k] * conn;
            dU[index_nu(U1_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[2 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[2][k] * conn;
            dU[index_nu(U2_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[3 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[5 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[2][k] * conn;
            dU[index_nu(U2_NU, sp)] += mhd_nu[1][k] * conn;
            conn = conn_GPU[6 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[1][k] * conn;
            conn = conn_GPU[8 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U2_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[2][k] * conn;
            #else
            dU[index_nu(UU_NU, sp)] += mhd_nu[0][k] * conn_GPU[0 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[1][k] * conn_GPU[4 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U2_NU, sp)] += mhd_nu[2][k] * conn_GPU[7 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U3_NU, sp)] += mhd_nu[3][k] * conn_GPU[9 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            conn = conn_GPU[1 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[1][k] * conn;
            dU[index_nu(U1_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[2 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[2][k] * conn;
            dU[index_nu(U2_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[3 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(UU_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[0][k] * conn;
            conn = conn_GPU[5 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[2][k] * conn;
            dU[index_nu(U2_NU, sp)] += mhd_nu[1][k] * conn;
            conn = conn_GPU[6 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U1_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[1][k] * conn;
            conn = conn_GPU[8 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
            dU[index_nu(U2_NU, sp)] += mhd_nu[3][k] * conn;
            dU[index_nu(U3_NU, sp)] += mhd_nu[2][k] * conn;
            #endif
        }
        dU[index_nu(UU_NU, sp)] *= geom->g;
        dU[index_nu(U1_NU, sp)] *= geom->g;
        dU[index_nu(U2_NU, sp)] *= geom->g;
        dU[index_nu(U3_NU, sp)] *= geom->g;
    }
    #endif
    /* done! */
}


__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,  double r, double Dt
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
){
	#if (DOHELM_TEMPERATURE)
	double xpres, ugas;
	eos_mode_rhotemp_pres_u(gpu_eos_table, ph[RHO], ph[UU], ph[YE], &xpres, &ugas
		#if (DONUCLEAR)
		, &ph[XALPHA], &ph[XATM]
		#endif
	);
	#else
	double ugas = ph[UU];
	#endif
	double epsilon = ugas / ph[RHO];
	double om_kepler = 1. / (pow(r, 3. / 2.) + BH_SPIN);
	double T_target = M_PI / 2.*pow(H_OVER_R*r*om_kepler, 2.);
	double Y = (GAMMA - 1.)*epsilon / T_target; // HELMEOS
	double lambda = om_kepler*ugas * sqrt(Y - 1. + fabs(Y - 1.));
	double int_energy = q->ucov[0] * q->ucon[0] * ugas;
	double bsq = dot(q->bcon,q->bcov);
	#if(WHICHPROBLEM==TRUNC_PROBLEM)
	if (r > 40.) {
		if (fabs(q->ucov[0] * lambda) * Dt < 0.1 * fabs(int_energy)) {
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) * (GAMMA - 1.) * lambda;
		}
		else {
			lambda *= (0.1 * fabs(int_energy)) / (fabs(q->ucov[0] * lambda) * Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) * (GAMMA - 1.) * lambda;
		}
	}
	#else
	//if (bsq / ph[RHO]<1. || r<10.){
		if (fabs(q->ucov[0] * lambda)*Dt<0.1*fabs(int_energy)){
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda; // HELMEOS
		}
		else{
			lambda *= (0.1*fabs(int_energy)) / (fabs(q->ucov[0] * lambda)*Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda; // HELMEOS
		}
	//}
	#endif
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	int j;
	double r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];

    // EOS-specific calls:
	#if (DOHELM)
    // 1. Helmholtz EOS
		#if (DOHELM_TEMPERATURE)
		eos_mode_rhotemp_pres_u(gpu_eos_table, r, pr[UU],
			#if (DO_YE)
			pr[YE],
			#else 
			1.0,
			#endif
			&P, &u
			#if (DONUCLEAR)
			, &pr[XALPHA], &pr[XATM]
			#endif
		);
		#else
		eos_mode_rhou_pres (gpu_eos_table, r, u
			#if (DO_YE)
			, pr[YE]
			#else
			, 1.0
			#endif
			, &P);
		#endif
    #elif(TWO_T)
	P = (gamma_g - 1.) * u;
    #else
    // 2. Ideal gas EOS
	P = (GAMMA - 1.)*u;
    #endif

	w = P + r + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
}

__device__ void primtoflux(double *  pr, struct of_state *  q,  int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if (TWO_T)
	, double gamma_g
	#endif
) {
	int j, k;
	double mhd[NDIM];
	double P, u, w, bsq, eta, ptot;
	u = pr[UU];

	/*Calculate misc quantities*/
    // EOS-specific calls:
    #if (DOHELM)
		double cs2_helm;
		#if (DOHELM_TEMPERATURE)
		eos_mode_rhotemp_pres_u_cs2 (gpu_eos_table, pr[RHO], pr[UU], 
			#if (DO_YE)
			pr[YE],
			#else 
			1.0,
			#endif
			&P, &u, &cs2_helm
			#if (DONUCLEAR)
			, &pr[XALPHA], &pr[XATM]
			#endif
		);
		#else
		eos_mode_rhou_pres_cs2 (gpu_eos_table, pr[RHO], pr[UU], 
			#if (DO_YE)
			pr[YE], 
			#else
			1.0,
			#endif
			&P, &cs2_helm);
		#endif
    #elif(TWO_T)
	P = (gamma_g - 1.) * u;
    #else
	P = (GAMMA - 1.) * u;
    #endif

	w = pr[RHO] + P + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	#if(AMD)
	ptot = fma(0.5, bsq, P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* particle number flux */
	flux[RHO] = pr[RHO] * q->ucon[dir];

	/* single row of mhd stress tensor,
	* first index up, second index down */
	#if(AMD)
	#pragma unroll 4
	DLOOPA mhd[j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
	#else
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
	#endif

	/* MHD stress-energy tensor w/ first index up,
	* second index down. */
	flux[UU] = mhd[0] + flux[RHO];
	flux[U1] = mhd[1];
	flux[U2] = mhd[2];
	flux[U3] = mhd[3];

	/* dual of Maxwell tensor */
	#if AMD
	flux[B1] = fma(q->bcon[1], q->ucon[dir], -q->bcon[dir] * q->ucon[1]);
	flux[B2] = fma(q->bcon[2], q->ucon[dir], -q->bcon[dir] * q->ucon[2]);
	flux[B3] = fma(q->bcon[3], q->ucon[dir], -q->bcon[dir] * q->ucon[3]);
	#else
	flux[B1] = q->bcon[1] * q->ucon[dir] - q->bcon[dir] * q->ucon[1];
	flux[B2] = q->bcon[2] * q->ucon[dir] - q->bcon[dir] * q->ucon[2];
	flux[B3] = q->bcon[3] * q->ucon[dir] - q->bcon[dir] * q->ucon[3];
	#endif

	//Flux of electron and ion entropies
	#if(TWO_T)
	flux[ENTRE] = flux[RHO] * pr[ENTRE];
	flux[ENTRI] = flux[RHO] * pr[ENTRI];
	#endif

	#if(DOKTOT)
	flux[KTOT] = flux[RHO] * calc_entropy(pr
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	#endif

	#pragma unroll 9
	for (k = 0; k < NPR_U;k++) flux[k] *= geom->g;
	#if(TWO_T)
	flux[ENTRE] *= geom->g;
	flux[ENTRI] *= geom->g;
	#endif

	//Calculate wavespeed
	if (dir != 0){
		double discr, vp, vm, va2, cs2, cms2;
		double Acon_0, Acon_js;
		double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
		if (dir == 1){
			Acon_0 = geom->gcon[1];
			Acon_js = geom->gcon[4];
		}
		else if (dir == 2){
			Acon_0 = geom->gcon[2];
			Acon_js = geom->gcon[7];
		}
		else if (dir == 3){
			Acon_0 = geom->gcon[3];
			Acon_js = geom->gcon[9];
		}

		/* find fast magnetosonic speed */

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        // cs2 was already calculated above
        cs2 = cs2_helm;
		#elif(TWO_T)
		cs2 = gamma_g * (gamma_g - 1.) * pr[UU] / w;
        #else
        // 2. Ideal gas EOS
		cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / w;
        #endif

		va2 = bsq / eta;
		cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

		//check on it!
		if (cms2 < 0.) cms2 = SMALL;
		if (cms2 > 1.) cms2 =1.;

		//now require that speed of wave measured by observer q->ucon is cms2
		Asq = Acon_js;
		Bsq = geom->gcon[0];// dot(Bcon, Bcov);
		Au = q->ucon[dir];
		Bu = q->ucon[0];
		AB = Acon_0;
		Au2 = Au*Au;
		Bu2 = Bu*Bu;
		AuBu = Au*Bu;
		#if AMD
		A = fma(-(Bsq + Bu2), cms2, Bu2);
		B = 2.* fma(-(AB + AuBu), cms2, AuBu);
		C = fma(-(Asq + Au2), cms2, Au2);
		discr = fma(B, B, -4.*A*C);
		#else
		A = Bu2 - (Bsq + Bu2)*cms2;
		B = 2.*(AuBu - (AB + AuBu)*cms2);
		C = Au2 - (Asq + Au2)*cms2;
		discr = B*B - 4.*A*C;
		#endif

		if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
		else if (discr < -1.e-10) discr = 0.;
		discr = sqrt(discr);
		vp = -(-B + discr) / (2.*A);
		vm = -(-B - discr) / (2.*A);

		*vmax = MY_MAX(vp, vm);
		*vmin = MY_MIN(vp, vm);
	}

	#if (DO_YE)
    flux[YE] = flux[RHO] * pr[YE];
    #endif
    #if (DONUCLEAR)
    flux[XALPHA] = flux[RHO] * pr[XALPHA];
    flux[XATM] = flux[RHO] * pr[XATM];
    #endif

	return;
}

//Calculate gas entropy
__device__ double calc_entropy(double* pr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	double entr;
	#if(DOHELM)
		#if(DOHELM_TEMPERATURE)
		eos_mode_rhotemp_entr(gpu_eos_table, pr[RHO], pr[UU],
			#if (DO_YE)
			pr[YE],
			#else 
			1.0,
			#endif
			&entr
			#if (DONUCLEAR)
			, &pr[XALPHA], &pr[XATM]
			#endif
		);
		#else
			eos_mode_rhou_entr(gpu_eos_table, pr[RHO], pr[UU],
				#if (DO_YE)
				pr[YE],
				#else 
				1.0,
				#endif
				&entr);
		#endif
	#elif(TWO_T)
		#if(0)
		double Theta;
		//For variable entropy
		Theta = (gamma_g - 1.0) * pr[UU] / pr[RHO] * MU_G;
			#if(FULL_ENTROPY)
			entr = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO]);
			#else
			entr = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO];
			#endif
		#else
		double P = (GAMMA - 1.0) * pr[UU];
			#if(FULL_ENTROPY)
			entr = 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
			#else
			entr = P * pow(pr[RHO], -GAMMA);
			#endif
		#endif
	#else 
	double P = (GAMMA - 1.0) * pr[UU];
		#if(FULL_ENTROPY)
		entr = 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
		#else
		entr = P * pow(pr[RHO], -GAMMA);
		#endif
	#endif

	return entr;
}

__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q
	#if(CALC_MDOT)
	, double magnetic_density_scale
	#endif
)
{
	/* get ucon */
	ucon_calc(pr, geom, q->ucon);
	lower(q->ucon, geom->gcov, q->ucov);
	bcon_calc(pr, q->ucon, q->ucov, q->bcon);
	lower(q->bcon, geom->gcov, q->bcov);
	#if(CALC_MDOT)
	int k;
	for (k = 0; k < NDIM; k++) {
		q->bcon[k] *= magnetic_density_scale;
		q->bcov[k] *= magnetic_density_scale;
	}
	#endif
	return;
}

/* find contravariant four-velocity */
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon)
{
	double alpha, gamma;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0]);
	#pragma unroll 4
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;

	gamma_calc(pr, geom, &gamma);

	ucon[0] = gamma / alpha;
	#if AMD
	#pragma unroll 4
	SLOOPA ucon[j] = fma(-gamma, beta[j] / alpha, pr[U1 + j - 1]);
	#else
	#pragma unroll 4
	SLOOPA ucon[j] = pr[U1 + j - 1] - gamma*beta[j] / alpha;
	#endif

	return;
}

__device__ void bcon_calc(double *  pr, double *  ucon, double *  ucov, double *  bcon)
{
	int j;

	#if AMD
	bcon[0] = fma(pr[B1], ucov[1], fma(pr[B2], ucov[2], pr[B3] * ucov[3]));
	#pragma unroll 3
	for (j = 1; j<4; j++)
		bcon[j] = (fma(bcon[0], ucon[j], pr[B1 - 1 + j])) / ucon[0];
	#else
	bcon[0] = pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3];
	#pragma unroll 3
	for (j = 1; j<4; j++) bcon[j] = (pr[B1 - 1 + j] + bcon[0] * ucon[j]) / ucon[0];
	#endif
	return;
}

__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma)
{
	double qsq;
	#if AMD
	qsq = fma(geom->gcov[4], pr[U1] * pr[U1], fma(geom->gcov[7], pr[U2] * pr[U2],geom->gcov[9] * pr[U3] * pr[U3]))
		+ 2.*fma(geom->gcov[5], pr[U1] * pr[U2], fma(geom->gcov[6], pr[U1] * pr[U3],geom->gcov[8] * pr[U2] * pr[U3]));
	#else
	qsq = geom->gcov[4] * pr[U1] * pr[U1]+ geom->gcov[7] * pr[U2] * pr[U2]+ geom->gcov[9] * pr[U3] * pr[U3]
		+ 2.*(geom->gcov[5] * pr[U1] * pr[U2]+ geom->gcov[6] * pr[U1] * pr[U3] + geom->gcov[8] * pr[U2] * pr[U3]);
	#endif

	if (qsq < 0.){
		if (fabs(qsq) > 1.E-10){ // then assume not just machine precision
			*gamma = 1.;
			return (1);
		}
		else qsq = 1.E-10; // set floor
	}

	*gamma = sqrt(1. + qsq);

	return(0);
}

__device__ void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *vmax, double *vmin
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
{
	double discr, vp, vm, va2, cs2, cms2;
	double bsq, eta, w;
	double Acon_0, Acon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
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

	/* find fast magnetosonic speed */

    // EOS-specific calls:
     // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    double xpres;
		#if (DOHELM_TEMPERATURE)
		double u;
		eos_mode_rhotemp_pres_u_cs2 (gpu_eos_table, pr[RHO], pr[UU], 
			#if (DO_YE)
			pr[YE],
			#else 
			1.0,
			#endif
			&xpres, &u, &cs2
			#if (DONUCLEAR)
			, &pr[XALPHA], &pr[XATM]
			#endif
		);
		w = pr[RHO] + u + xpres;
		#else
		eos_mode_rhou_pres_cs2 (gpu_eos_table, pr[RHO], pr[UU], 
			#if (DO_YE)
			pr[YE], 
			#else
			1.0,
			#endif	
			&xpres, &cs2);
		w = pr[RHO] + pr[UU] + xpres;
		#endif
    #else
    // 2. Ideal gas EOS
	#if AMD
	w = fma(GAMMA, pr[UU], pr[RHO]);
	#else
	w = pr[RHO] + GAMMA*pr[UU];
	#endif
    cs2 = GAMMA*(GAMMA - 1.)*pr[UU] / w;
    #endif

	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	va2 = bsq / eta;
	cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

	/* check on it! */
	if (cms2 < 0.) {
		//fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL;
	}
	if (cms2 > 1.) {
		//fail(FAIL_COEFF_SUP) ;
		cms2 = 1.;
	}

	/* now require that speed of wave measured by observer
	q->ucon is cms2 */
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	AB = Acon_0;
	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;
	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2.* fma(-(AB + AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4.*A*C);
	#else
	A = Bu2 - (Bsq + Bu2)*cms2;
	B = 2.*(AuBu - (AB + AuBu)*cms2);
	C = Au2 - (Asq + Au2)*cms2;
	discr = B*B - 4.*A*C;
	#endif
	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.;

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	*vmax = MY_MAX(vp, vm);
	*vmin = MY_MIN(vp, vm);

	return;
}
