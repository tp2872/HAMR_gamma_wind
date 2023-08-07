
#if (NEUTRINOS_M1)
__device__ void semiimplicit_solve_nu_init(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int *pflag_nu
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(NU_INNER_STOP)
	, double r
	#endif
);
__device__ void calc_avg_neutrino_energy(double* pb, double* ener_nu_avg, struct of_geom* geom, int sp);
// Predictor step
__device__ void get_ye_predictor(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, struct of_geom* geom, double* ucon, double* ucov, double* ener_nu_avg, double* eta, double* kappa_abs, double* kappa_s, double Dt);
// Functions
__device__ int semiimplicit_solve_nu(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_nu, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table
    #if (NU_INNER_STOP)
    , double radius
    #endif
    #if(NEUTRINOS_DEBUG)
    , double* error_nu0, double* error_nu1, double* error_nu2
    #endif
) {
    int k, sp;
    double U_ft[NPR], pb_i[NPR], U_n_temp[NPR], U_i_temp[NPR], U_prev[NPR], error_t[1];

    //Initialize temporary variables
    PLOOP {
        U_n_temp[k] = U_n[k];
        U_i_temp[k] = U_i[k];
    }

    PLOOP{
        dU[k] = 0.;
        U_ft[k] = U_i[k]; // U_i_temp[k];
        pb_i[k] = pb[k];
    }

    semiimplicit_solve_nu_init(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, geom, dU, Dt, error_t, cell_size, y_max, pflag_nu, gpu_eos_table
    	#if (NU_INNER_STOP)
    	, radius
    	#endif
    );

    /*
    #if (NU_EXPLICIT)		
    // Explicit step
    source_nu(pb_i, geom, dU, U_i_temp, U_ft, Dt, y_max, gpu_eos_table, gpu_nulib_table);

    
    // Invert conserved to primitives
    #if (USE_3D_INV)
    *pflag = Utoprim_3D_T(U_ft, geom->gcov, geom->gcon, geom->g, pb_i, NEWT_TOL, BASIC, gpu_eos_table);
    if (*pflag) {
        *pflag = Utoprim_2d(U_ft, geom->gcov, geom->gcon, geom->g, pb_i, NEWT_TOL, BASIC, gpu_eos_table);
    }
    #else
    *pflag = Utoprim_2d(U_ft, geom->gcov, geom->gcon, geom->g, pb_i, NEWT_TOL, BASIC, gpu_eos_table);
    #endif
    #if(DO_FONT_FIX)
    if (*pflag) {
        *pflag = Utoprim_1dvsq2fix1(U_ft, geom->gcov, geom->gcon, geom->g, pb_i, NEWT_TOL, BASIC, 1, gpu_eos_table);
    }
    #endif
    *pflag_nu = Rtoprim_nu(U_ft, geom, geom->gcov, geom->gcon, geom->g, pb_i, y_max, BASIC);

    #else				
    */

    if (error_t[0] <= 1e-12) {
        PLOOP {
            U_f[k] = U_ft[k];
            dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
            pb[k] = pb_i[k];
        }
    }
    else {
        // Implicit step
        implicit_solve_nu(pb_i, U_n_temp, U_i_temp, U_ft, U_prev, pflag, pflag_nu, geom, dU, Dt, cell_size, y_max, gpu_eos_table, gpu_nulib_table
            #if (NEUTRINOS_DEBUG)
            , error_nu0, error_nu1, error_nu2
            #endif
            #if (NU_INNER_STOP)
            , radius
            #endif
        );

        //If used TYPE2 limiter for inversion radiative quantities, redo with BASIC limiter
		if(pflag_nu[0]){
            Rtoprim_nu(U_prev, geom, geom->gcov, geom->gcon, geom->g, pb_i, y_max, BASIC);
            struct of_state_nu q_nu[NU_SPECIES];
            for (sp = 0; sp < NU_SPECIES; sp++) get_state_nu(pb_i, geom, &q_nu[sp], sp);
            primtoflux_nu(pb_i, q_nu, 0, geom, U_ft);
		}

        //Set final quantities
        PLOOP {
            U_f[k] = U_ft[k];
            dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
            pb[k] = pb_i[k];
        }
    }
    /*
    #endif
    */
}


__device__ int implicit_solve_nu(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_nu, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table
    #if (NEUTRINOS_DEBUG)
    , double* error_nu0, double* error_nu1, double* error_nu2
    #endif
    #if (NU_INNER_STOP)
    , double radius
    #endif
) {
    double U_new[NPR], U_old[NPR], U_prev_old[NPR], pb_new[NPR], pb_old[NPR];
    double p_h[NPR], U_2[NPR], U_h[NPR];
    double kappa_abs, kappa_emmit, kappa_es, tau = 0.0;
    struct of_state q;
    struct of_state_nu q_nu[NU_SPECIES];
    int k, sp, flag = 0, flag_nu = 0;

    // Set temporary variables to their initial values
    for (k = 0; k < NPR; k++) {
        pb_old[k] = pb[k];
        U_old[k] = U_f[k];
        dU[k] = 0.;
        pb_new[k] = pb_old[k];
        U_new[k] = U_old[k];

        p_h[k] = pb_old[k];
        U_2[k] = U_old[k];
        U_h[k] = U_old[k];
    }
    
    /*int is_timestep_explicit = 0;
    if (1) {
        source_nu(pb_old, geom, dU, U_old, U_new, Dt, y_max, gpu_eos_table, gpu_nulib_table 
	#if (NU_KEEP_COEFF_CONST)
            , eta_0, kappa_abs0, kappa_s0, eta_N0, kappa_N0
	#endif
        );
        is_timestep_explicit = 1;
    }*/
    // Compute n_0, n^mu
    double ncon[NDIM], ncov0;
    ncov0 = -sqrt(-1. / geom->gcon[0]);
    ncon[0] = geom->gcon[0] * ncov0;
    ncon[1] = geom->gcon[1] * ncov0;
    ncon[2] = geom->gcon[2] * ncov0;
    ncon[3] = geom->gcon[3] * ncov0;

    // Do a single implicit step for now:
    /*
        U_old:	post-explicit step cons. variables
        U_new:	post-implicit step cons. variables
    */

    // Use a predictor step
    /*
	#if (NU_PREDICTOR)
    ucon_calc(pb, geom, ucon);
    lower(ucon, geom->gcov, ucov);
    double pb_ye = pb[YE];
    get_ye_predictor(gpu_eos_table, gpu_nulib_table, pb, geom, ucon, ucov, ener_nu_avg, eta_0, kappa_abs0, kappa_s0, Dt);
    for (sp = 0; sp < NU_SPECIES; sp++) {
	#if (NU_INNER_STOP)
        if (radius < RAD_NU_STOP) {
            eta_0[sp] = eta_N0[sp] = kappa_abs0[sp] = kappa_s0[sp] = kappa_N0[sp] = 1e-30;
        }
        else {
            eta_0[sp] = calc_nu_kappa_emiss(gpu_nulib_table, pb, sp);
            eta_N0[sp] = calc_nu_number_emiss(gpu_nulib_table, pb, sp);
            kappa_abs0[sp] = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
            kappa_s0[sp] = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
            kappa_N0[sp] = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
            tau = MY_MAX(tau, (kappa_abs0[sp] + kappa_s0[sp]) * cell_size);
        }
	#else 
        eta_0[sp] = calc_nu_kappa_emiss(gpu_nulib_table, pb, sp);
        eta_N0[sp] = calc_nu_number_emiss(gpu_nulib_table, pb, sp);
        kappa_abs0[sp] = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        kappa_s0[sp] = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        kappa_N0[sp] = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
	#endif
    }
	#endif
    */

    /*
	#if (NU_COOLING)
    double factor_Dt_cool = 1.;
    double dJ[NU_SPECIES];
    double Eint, xP;
    double nu_gas_timescale = 0.;
    eos_mode_rhotemp_pres_u(gpu_eos_table, pb[RHO], pb[UU], pb[YE], &xP, &Eint
        #if (DONUCLEAR)
        , &pb[XALPHA], &pb[XATM]
        #endif
    );
    for (sp = 0; sp < NU_SPECIES; sp++) {
        dJ[sp] = (eta_0[sp] / (kappa_abs0[sp] + 1e-30) - J0[sp]) * (1. - exp(-kappa_abs0[sp] * Dt)) + 1e-30;
        nu_gas_timescale = MY_MAX(nu_gas_timescale, 1. / sqrt(1e-30 + kappa_abs0[sp] * (kappa_abs0[sp] + kappa_s0[sp])));
    }
    double ratio_cooling = Eint / (dJ[0] + dJ[1] + dJ[2]);
    double tau_cooling = Eint / fabs((dJ[0] + dJ[1] + dJ[2]) / Dt);
    factor_Dt_cool = MY_MIN(0.5 * tau_cooling / Dt, 1.);
    //double factor_cooling = exp(-Dt / tau_cooling);
	#endif
    */

    int is_backup_inv = 0;
    double factor_Dt = 1.;
    double xentr, Tg;
    // Update conserved variables given the NU source term
    // -- update of the cons. variables happens inside this function:
    
	#if (NU_SUBCYCLING)
    int n_iter, count_increase=0, n_iter_fail=0;
    double remainder_Dt = 1.;

    double norm = 0., err1 = 0., err2 = 0., dQtcov[NDIM], dQtcon[NDIM], error_tmp[NU_SPECIES];

    for (n_iter = 0; n_iter < 1; n_iter++) {        
    	PLOOP {
    		pb_new[k] = pb_old[k];
    		U_new[k] = U_old[k];
    		dU[k] = 0.0;
    	}
    	factor_Dt = 1.0 / pow(2.0, (double)n_iter_fail);

        for (sp = 0; sp < NU_SPECIES; sp++) {
            flag += source_linearized_nu(pb_old, geom, &ncon[0], ncov0, U_old, U_new, factor_Dt * Dt, gpu_eos_table, gpu_nulib_table, sp
				#if (NU_KEEP_COEFF_CONST)
                , eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
				#endif
            );

            source_linearized_nu(pb_old, geom, &ncon[0], ncov0, U_old, U_h, 0.5 * factor_Dt * Dt, gpu_eos_table, gpu_nulib_table, sp
				#if (NU_KEEP_COEFF_CONST)
                , eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
				#endif
            );

            flag_nu = Rtoprim_nu(U_h, geom, geom->gcov, geom->gcon, geom->g, p_h, y_max, TYPE2); // danat: TYPE2

            source_linearized_nu(p_h, geom, &ncon[0], ncov0, U_h, U_2, 0.5 * factor_Dt * Dt, gpu_eos_table, gpu_nulib_table, sp
				#if (NU_KEEP_COEFF_CONST)
                , eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
				#endif
            );

            norm = fabs(U_old[index_nu(UU_NU, sp)]);
            err1 = fabs(U_2[index_nu(UU_NU, sp)] - U_new[index_nu(UU_NU, sp)]) / norm;
            
            dQtcov[0] = 0.0;
            for (int i = 1; i < NDIM; i++) dQtcov[i] = fabs(U_2[index_nu(UU_NU + i, sp)] - U_new[index_nu(UU_NU + i, sp)]);
            raise(dQtcov, geom->gcon, dQtcon);
            dQtcon[0] = 0.0;
            err2 = (dQtcon[1] * dQtcov[1] + dQtcon[2] * dQtcov[2] + dQtcon[3] * dQtcov[3]) / norm;
            error_tmp[sp] = MY_MAX(err1, err2);
        }

        #if (NEUTRINOS_DEBUG)
        *error_nu0 = error_tmp[0];
        *error_nu1 = error_tmp[1];
        *error_nu2 = error_tmp[2];
	    #endif

        for (sp = 0; sp < NU_SPECIES; sp++) {
	        // UU --> U3 source terms
	        dU[UU] -= (U_new[index_nu(UU_NU, sp)] - U_old[index_nu(UU_NU, sp)]);
	        dU[U1] -= (U_new[index_nu(U1_NU, sp)] - U_old[index_nu(U1_NU, sp)]);
	        dU[U2] -= (U_new[index_nu(U2_NU, sp)] - U_old[index_nu(U2_NU, sp)]);
	        dU[U3] -= (U_new[index_nu(U3_NU, sp)] - U_old[index_nu(U3_NU, sp)]);
	    }

	    U_new[UU] = U_old[UU] + dU[UU];
	    U_new[U1] = U_old[U1] + dU[U1];
	    U_new[U2] = U_old[U2] + dU[U2];
	    U_new[U3] = U_old[U3] + dU[U3];

	    /* compute Ye evolution: note that U_new and U_old are already divided by MASS_DENSITY_SCALE */
	    for (sp = 0; sp < NU_SPECIES - 1; sp++) {
	        if (sp == 0)
	            dU[YE] += -(U_new[index_nu(NUMBER_NU, sp)] - U_old[index_nu(NUMBER_NU, sp)]) * MP_CGS;
	        else if (sp == 1)
	            dU[YE] += (U_new[index_nu(NUMBER_NU, sp)] - U_old[index_nu(NUMBER_NU, sp)]) * MP_CGS;
	    }
	    U_new[YE] = U_old[YE] + dU[YE];

	    get_state(pb_old, geom, &q);
		#if(DOKTOT)
	    Tg = pb_old[UU];
	    dU[KTOT] = -(dU[UU] * q.ucon[0] + dU[U1] * q.ucon[1] + dU[U2] * q.ucon[2] + dU[U3] * q.ucon[3]) / (BOLTZ_CGS * Tg);
	    U_new[KTOT] += dU[KTOT];
		#endif

        // Compute new MHD primitive variables
        flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
			#if (DOHELM)
            , gpu_eos_table
			#endif
        );

        if (flag == 0) {
	        flag_nu = Rtoprim_nu(U_new, geom, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2); // danat: TYPE2
			PLOOP U_prev_old[k] = U_new[k];

	        //Recompute T_t^mu for consistency
	        U_new[RHO] = U_i[RHO];
	        get_state(pb_new, geom, &q);
	        mhd_calc(pb_new, 0, &q, &U_new[UU], gpu_eos_table);
	        for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
	        U_new[UU] += U_new[RHO];

	        //Recompute entropy for consistency
	        eos_mode_rhotemp_entr(gpu_eos_table, pb_new[RHO], pb_new[UU], pb_new[YE], &xentr
				#if (DONUCLEAR)
	            , &pb_new[XALPHA], &pb_new[XATM]
				#endif
	        );
	        U_new[KTOT] = U_new[RHO] * xentr;
	        // Recompute Ye
	        U_new[YE] = U_new[RHO] * pb_new[YE];

	        /*
	        //Recompute R_t^mu for consistency
	        for (sp = 0; sp < NU_SPECIES; sp++) {
	            get_state_nu(pb_new, geom, &q_nu[sp], sp);
	            mhd_calc_nu(pb_new, 0, &q_nu[sp], &U_new[index_nu(UU_NU, sp)], sp);
	            for (k = UU_NU; k <= NUMBER_NU; k++) U_new[index_nu(k, sp)] *= geom->g;
	        }
	        primtoflux_nu(pb_new, q_nu, 0, geom, U_new);
	        */

	        // Update quantities
	        PLOOP {
	            pb_old[k] = pb_new[k];
	            U_old[k] = U_new[k];
	        }

            if (flag_nu) {
                for (sp = 0; sp < NU_SPECIES; sp++) {
                    for (k = index_nu(UU_NU, sp); k <= index_nu(NUMBER_NU, sp); k++) U_prev[k] = U_prev_old[k];
                }
			}
			pflag_nu[0] = flag_nu;

            for (k = 0; k < NPR; k++) {
                U_f[k] = U_new[k];
                dU[k] = (U_new[k] - U_i[k]) / Dt;
                pb[k] = pb_new[k];
            }

	        remainder_Dt -= factor_Dt;
	        if(remainder_Dt < 1e-4) break;
		}
		else {
			n_iter_fail++;
			if (n_iter_fail == 5) break;
        }
    }

	#else 
    // Compute tau
    double mhd_nu[NDIM][NDIM], ener_nu_avg[NU_SPECIES], ucon[NDIM], ucov[NDIM];
    double kappa_abs0[NU_SPECIES], kappa_s0[NU_SPECIES], eta_0[NU_SPECIES], eta_N0[NU_SPECIES], kappa_N0[NU_SPECIES], J0[NU_SPECIES];
    double R_dot_ucon[NDIM];
    ucon_calc(pb, geom, ucon);
    lower(ucon, geom->gcov, ucov);

    for (sp = 0; sp < NU_SPECIES; sp++) {
        get_state_nu(pb, geom, &q_nu[sp], sp);
        mhd_calc_nu(pb, 0, &q_nu[sp], mhd_nu[0], sp);
        // compute avg neutrino energy in fluid frame
        calc_avg_neutrino_energy(pb, &ener_nu_avg[sp], geom, sp);
        // Idea: Use eta, kappa values throughout the neutrino step
        #if (NU_INNER_STOP)
        if (radius < RAD_NU_STOP) {
            eta_0[sp] = eta_N0[sp] = kappa_abs0[sp] = kappa_s0[sp] = kappa_N0[sp] = 1e-30;
        }
        else {
            eta_0[sp] = calc_nu_kappa_emiss(gpu_nulib_table, pb, sp);
            eta_N0[sp] = calc_nu_number_emiss(gpu_nulib_table, pb, sp);
            kappa_abs0[sp] = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
            kappa_s0[sp] = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
            kappa_N0[sp] = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        }
        #else 
        eta_0[sp] = calc_nu_kappa_emiss(gpu_nulib_table, pb, sp);
        eta_N0[sp] = calc_nu_number_emiss(gpu_nulib_table, pb, sp);
        kappa_abs0[sp] = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        kappa_s0[sp] = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        kappa_N0[sp] = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, pb, ener_nu_avg[sp], sp);
        #endif
        tau = MY_MAX(tau, (kappa_abs0[sp] + kappa_s0[sp]) * cell_size);

        #if (NU_COOLING)
        mhd_calc_nu(pb, 1, &q_nu[sp], mhd_nu[1], sp);
        mhd_calc_nu(pb, 2, &q_nu[sp], mhd_nu[2], sp);
        mhd_calc_nu(pb, 3, &q_nu[sp], mhd_nu[3], sp);
        for (int i = 0; i < NDIM; i++)
            R_dot_ucon[i] = (mhd_nu[i][0] * ucon[0] + mhd_nu[i][1] * ucon[1] + mhd_nu[i][2] * ucon[2] + mhd_nu[i][3] * ucon[3]);
        J0[sp] = (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]);
        #endif
    }
    double error = 0.0, error_tmp[NU_SPECIES];
    int iter = 0, err_type;
    for (sp = 0; sp < NU_SPECIES; sp++) {
        // One full dt timestep
        flag += source_linearized_nu(pb_old, geom, &ncon[0], ncov0, U_old, U_new, factor_Dt * Dt, gpu_eos_table, gpu_nulib_table, sp
			#if (NU_KEEP_COEFF_CONST)
            , eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
			#endif
        );

        //// Two 0.5dt timesteps
        //source_linearized_nu(pb_old, geom, &ncon[0], ncov0, U_old, Uh, 0.5 * Dt, gpu_eos_table, gpu_nulib_table, sp
        //	#if (NU_KEEP_COEFF_CONST)
        //	, eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
        //	#endif
        //);

        //// ... note the same primitives
        //source_linearized_nu(pb_old, geom, &ncon[0], ncov0, Uh, U_new, 0.5 * Dt, gpu_eos_table, gpu_nulib_table, sp
        //	#if (NU_KEEP_COEFF_CONST)
        //	, eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
        //	#endif
        //);		

        ///* compute error here & update cons. quantities */
        //err_type = calc_linearized_error2(ncon, ncov0, geom->gcon, U_1, U_2, U_old, U_new, sp, y_max, &error_tmp[sp]);
    }

	#if (NEUTRINOS_DEBUG)
    //*error_nu0 = fabs(U_new[UU_NU] - U_old[UU_NU]) / U_old[UU_NU]; // error_tmp[0];
    //*error_nu1 = fabs(U_new[UU_NU + 5] - U_old[UU_NU + 5]) / U_old[UU_NU + 5]; // error_tmp[1];
    //*error_nu2 = fabs(U_new[UU_NU + 10] - U_old[UU_NU + 10]) / U_old[UU_NU + 10]; // error_tmp[2];
	#endif

    for (sp = 0; sp < NU_SPECIES; sp++) {
        // UU --> U3 source terms
        dU[UU] -= (U_new[index_nu(UU_NU, sp)] - U_old[index_nu(UU_NU, sp)]);
        dU[U1] -= (U_new[index_nu(U1_NU, sp)] - U_old[index_nu(U1_NU, sp)]);
        dU[U2] -= (U_new[index_nu(U2_NU, sp)] - U_old[index_nu(U2_NU, sp)]);
        dU[U3] -= (U_new[index_nu(U3_NU, sp)] - U_old[index_nu(U3_NU, sp)]);
    }

    U_new[UU] = U_old[UU] + dU[UU];
    U_new[U1] = U_old[U1] + dU[U1];
    U_new[U2] = U_old[U2] + dU[U2];
    U_new[U3] = U_old[U3] + dU[U3];

    /* compute Ye evolution: note that U_new and U_old are already divided by MASS_DENSITY_SCALE */
    for (sp = 0; sp < NU_SPECIES - 1; sp++) {
        if (sp == 0)
            dU[YE] += -(U_new[index_nu(NUMBER_NU, sp)] - U_old[index_nu(NUMBER_NU, sp)]) * MP_CGS;
        else if (sp == 1)
            dU[YE] += (U_new[index_nu(NUMBER_NU, sp)] - U_old[index_nu(NUMBER_NU, sp)]) * MP_CGS;
    }
    U_new[YE] = U_old[YE] + dU[YE];

    get_state(pb_old, geom, &q);
	#if(DOKTOT)
    Tg = pb_old[UU];
    dU[KTOT] = -(dU[UU] * q.ucon[0] + dU[U1] * q.ucon[1] + dU[U2] * q.ucon[2] + dU[U3] * q.ucon[3]) / (BOLTZ_CGS * Tg);
    U_new[KTOT] += dU[KTOT];
	#endif

    // Compute new MHD primitive variables
    /*
    #if(0)
    *pflag = Utoprim_NM(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
        #if (DOHELM)
        , gpu_eos_table
        #endif
    );
    #else
    #if (USE_3D_INV)
    *pflag = Utoprim_3D_T(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
        #if (DOHELM)
        , gpu_eos_table
        #endif
    );
    if (*pflag) {
        *pflag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
            #if (DOHELM)
            , gpu_eos_table
            #endif
        );
    }
    is_backup_inv = *pflag;
    #else
    */
    flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
        #if (DOHELM)
        , gpu_eos_table
        #endif
    );
    is_backup_inv = flag;
    // #endif
    // #endif
    /*
    #if(DO_FONT_FIX)
    if (flag) {
        flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC, 1
            #if (DOHELM)
            , gpu_eos_table
            #endif
        );
    }
    #endif
    */

    // In case MHD inversion fails, terminate with this error message
    if (flag == 0) {
        // Compute new NU primitive variables
        flag_nu = Rtoprim_nu(U_new, geom, geom->gcov, geom->gcon, geom->g, pb_new, y_max, BASIC);

        //Recompute T_t^mu for consistency
        U_new[RHO] = U_i[RHO];
        get_state(pb_new, geom, &q);
        mhd_calc(pb_new, 0, &q, &U_new[UU], gpu_eos_table);
        for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
        U_new[UU] += U_new[RHO];

        //Recompute entropy for consistency
        eos_mode_rhotemp_entr(gpu_eos_table, pb_new[RHO], pb_new[UU], pb_new[YE], &xentr
            #if (DONUCLEAR)
            , &pb_new[XALPHA], &pb_new[XATM]
            #endif
        );
        U_new[KTOT] = U_new[RHO] * xentr;

        // Recompute Ye
        U_new[YE] = U_i[RHO] * pb_new[YE];

        //Recompute R_t^mu for consistency
        for (sp = 0; sp < NU_SPECIES; sp++) {
            get_state_nu(pb_new, geom, &q_nu[sp], sp);
            mhd_calc_nu(pb_new, 0, &q_nu[sp], &U_new[index_nu(UU_NU, sp)], sp);
            for (k = UU_NU; k <= NUMBER_NU; k++) U_new[index_nu(k, sp)] *= geom->g;
        }
        primtoflux_nu(pb_new, q_nu, 0, geom, U_new);

        // Finish the NU implicit timestep
        for (k = 0; k < NPR; k++) {
            U_f[k] = U_new[k];
            dU[k] = (U_new[k] - U_i[k]) / Dt;
            pb[k] = pb_new[k];
        }
    }
    else {
        #if(NU_DEBUG)
            printf("\n[post-nu step: con2prim failed, flag %d (prev. %d)] \n\tr,t,y: (%e %e %e) xa,xamb: (%e %e) (dU[UU->U3]: %e, %e %e %e), \n\topt.depth = %e, <e> = (%e %e %e), E/dJ=%e, Dt,dt_cool,dt_beta: (%e, %e, %e), Eint,J_ini,dJ: (%e, %e, %e), \n\tkappaA = (%e %e %e), r = %e\n", 
                *pflag, is_backup_inv, 
                pb_new[RHO], pb_new[UU], pb_new[YE], pb_new[XALPHA], pb_new[XATM],
                fabs(dU[UU]) / U_old[UU], fabs(dU[U1]) / U_old[U1], fabs(dU[U2]) / U_old[U2], fabs(dU[U3]) / U_old[U3], 
                tau, ener_nu_avg[0], ener_nu_avg[1], ener_nu_avg[2]
				#if (NU_COOLING)
                , ratio_cooling, Dt, tau_cooling, nu_gas_timescale, Eint, fabs(dJ[0] + dJ[1] + dJ[2]), (J0[0]+J0[1]+J0[2]), kappa_abs0[0], kappa_abs0[1], kappa_abs0[2]
				#endif
				#if (NU_INNER_STOP)
                , radius
				#endif
            );
        #endif
        return(1);
    }
    #endif // end of #if (NU_SUBCYCLING)
}


__device__ int source_linearized_nu(double* ph, struct of_geom* geom, double* ncon, double ncov0, double* U_old, double* U_new, double Dt, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, int species
    #if (NU_KEEP_COEFF_CONST)
    , double eta_0, double kappa_abs0, double kappa_s0, double eta_N0, double kappa_N0
    #endif
) {
    int i, j, flag=0;
    double mhd_nu[NDIM][NDIM];
    double U_i[NDIM], U_h[NDIM], U_f[NDIM];
    double ucon[NDIM], ucov[NDIM], W;

    /*
        Timesteps:
            Dt is the coordinate time;
            dt is in the fluid frame; (Dt = 1/alpha * dt)
    */
    double alpha = fabs(ncov0);
    double dt = alpha * Dt;

    // Set ucon and ucov
    ucon_calc(ph, geom, ucon);
    lower(ucon, geom->gcov, ucov);
    W = alpha * ucon[0];
    double sqrt_gamma = geom->g / alpha;

    // Convert the conserved quantities from HAMR (U_old) to SPEC (U_i) 
    U_i[0] = U_old[index_nu(UU_NU, species)];
    U_i[1] = U_old[index_nu(U1_NU, species)];
    U_i[2] = U_old[index_nu(U2_NU, species)];
    U_i[3] = U_old[index_nu(U3_NU, species)];

    // Calculate the radiation tensor R^mu_nu
    struct of_state_nu q_nu;
    get_state_nu(ph, geom, &q_nu, species);
    mhd_calc_nu(ph, 0, &q_nu, mhd_nu[0], species);
    mhd_calc_nu(ph, 1, &q_nu, mhd_nu[1], species);
    mhd_calc_nu(ph, 2, &q_nu, mhd_nu[2], species);
    mhd_calc_nu(ph, 3, &q_nu, mhd_nu[3], species);

    // Calculate the pressure tensor P^i_j / E, P^i_j u_i / E, P^i_j u_i u^j / E
    double P_dot_ucov[NDIM], J, n_dot_unu, u_dot_unu, W_v_dot_unu;
    n_dot_unu = q_nu.ucon[0] * ncov0;
    u_dot_unu = q_nu.ucon[0] * ucov[0] + q_nu.ucon[1] * ucov[1] + q_nu.ucon[2] * ucov[2] + q_nu.ucon[3] * ucov[3];
    W_v_dot_unu = u_dot_unu - W * n_dot_unu;

    double Q_dot_n, Q_dot_u;
    Q_dot_n = -ph[index_nu(UU_NU, species)] / 3. * (4. * n_dot_unu * n_dot_unu - 1.);
    for (i = 0; i < NDIM; i++) 
        P_dot_ucov[i] = ph[index_nu(UU_NU, species)] / 3. * (ucov[i] + ncov0 * delta(0, i) * (4. * n_dot_unu * W_v_dot_unu - W) + q_nu.ucov[i] * 4. * W_v_dot_unu) / (-Q_dot_n);

    double Puu = ph[index_nu(UU_NU, species)] / 3. * (4. * W_v_dot_unu * W_v_dot_unu - (1. - W * W)) / (-Q_dot_n);

    // Set the opacities and the emissivity given (rho, T, Ye)
    double kappa_abs, eta, kappa_es;
    #if (NU_KEEP_COEFF_CONST)
    kappa_abs = kappa_abs0;
    kappa_es = kappa_s0;
    eta = eta_0;
    #else
    double ener_nu_avg;
    calc_avg_neutrino_energy(ph, &ener_nu_avg, geom, species);

    kappa_abs = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);
    kappa_es = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);
    eta = calc_nu_kappa_emiss(gpu_nulib_table, ph, species);
    #endif

    // ====
    double implicit_RHS[NDIM];
    // Set implicit_RHS
    for (i = 0; i < NDIM; i++)
        implicit_RHS[i] = geom->g * Dt * eta * ucov[i] + U_i[i];

    double implicit_A[NDIM][NDIM], implicit_A_inv[NDIM][NDIM];
    // Start to fill the matrix
    for (i = 0; i < NDIM; i++) for (j = 0; j < NDIM; j++) {
        implicit_A[i][j] =
            delta(i, j) * (1. + dt * (kappa_abs + kappa_es) * W) +
            dt * ncon[j] * (kappa_es * ucov[i] * (Puu - W * W) + (kappa_abs + kappa_es) * (-W * alpha * delta(0, i) + P_dot_ucov[i])) +
            dt * ucon[j] * ((kappa_abs + kappa_es) * alpha * delta(0, i) + 2 * kappa_es * W * ucov[i]);
    }

    // Invert implicit_A matrix
    flag = invert_matrix_4D(implicit_A, implicit_A_inv);
    if (flag) {
    /*
        #if(NU_DEBUG)
        printf("FF 4d inversion: %d\t(r,t,y=%e %e %e), (%e %e %e)\n- %e %e %e %e\n- %e %e %e %e\n- %e %e %e %e\n- %e %e %e %e\n", flag, ph[RHO], ph[UU], ph[YE], eta, kappa_abs, kappa_es, implicit_A[0][0], implicit_A[0][1], implicit_A[0][2], implicit_A[0][3], implicit_A[1][0], implicit_A[1][1], implicit_A[1][2], implicit_A[1][3], implicit_A[2][0], implicit_A[2][1], implicit_A[2][2], implicit_A[2][3], implicit_A[3][0], implicit_A[3][1], implicit_A[3][2], implicit_A[3][3]);
        #endif
    */
        for (int k = 0; k < 5; k++) U_new[k + index_nu(UU_NU, species)] = U_old[k + index_nu(UU_NU, species)];
        return (flag);
    }

    // Calculate the updated conserved variables 
    for (i = 0; i < NDIM; i++)
        U_f[i] = implicit_A_inv[i][0] * implicit_RHS[0] + implicit_A_inv[i][1] * implicit_RHS[1] + implicit_A_inv[i][2] * implicit_RHS[2] + implicit_A_inv[i][3] * implicit_RHS[3];

    // Convert the conserved variables from SPEC (U_f) to HAMR (U_new)
    U_new[index_nu(UU_NU, species)] = U_f[0];
    U_new[index_nu(U1_NU, species)] = U_f[1];
    U_new[index_nu(U2_NU, species)] = U_f[2];
    U_new[index_nu(U3_NU, species)] = U_f[3];

    // **** Evolve neutrino number IMPLICITLY ****
    #if (NU_KEEP_COEFF_CONST)
    double eta_N = eta_N0;
    double kappa_N = kappa_N0;
    #else
    double eta_N = calc_nu_number_emiss(gpu_nulib_table, ph, species);
    double kappa_N = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);
    #endif

    Q_dot_u = (U_f[0] * ucon[0] + U_f[1] * ucon[1] + U_f[2] * ucon[2] + U_f[3] * ucon[3]) / sqrt_gamma;
    Q_dot_n = (U_f[0] * ncon[0] + U_f[1] * ncon[1] + U_f[2] * ncon[2] + U_f[3] * ncon[3]) / sqrt_gamma;

    // implicit change
    U_new[index_nu(NUMBER_NU, species)] = (U_old[index_nu(NUMBER_NU, species)] + geom->g * eta_N * Dt) / (1. + geom->g * Dt * kappa_N * (2. * W * Q_dot_u + (Puu - W * W) * Q_dot_n) / (sqrt_gamma * Q_dot_u));
    // explicit change
    //U_new[index_nu(NUMBER_NU, species)] = U_old[index_nu(NUMBER_NU, species)] + geom->g * Dt * (eta_N - kappa_N * (2. * W * Q_dot_u + (Puu - W * W) * Q_dot_n) / (sqrt_gamma * Q_dot_u));

    return (0);
}


__device__ int calc_linearized_error2(double* ncon, double ncov0, double gcon[10], double* U_1, double* U_2, double* U_old, double* U_new, int species, double y_max, double* error_tmp) {
    int i, returnval = 0;
    /* convert conserved vars from HAMR to SPEC */
    double E_1, E_2, F_1[NDIM], F_2[NDIM], E_f, F_f[NDIM];
    double alphaRel = 1e-2;
    E_1 = -(ncon[0] * U_1[index_nu(UU_NU, species)] + ncon[1] * U_1[index_nu(U1_NU, species)] + ncon[2] * U_1[index_nu(U2_NU, species)] + ncon[3] * U_1[index_nu(U3_NU, species)]);

    F_1[0] = U_1[index_nu(UU_NU, species)] - ncov0 * E_1;
    F_1[1] = U_1[index_nu(U1_NU, species)];
    F_1[2] = U_1[index_nu(U2_NU, species)];
    F_1[3] = U_1[index_nu(U3_NU, species)];

    E_2 = -(ncon[0] * U_2[index_nu(UU_NU, species)] + ncon[1] * U_2[index_nu(U1_NU, species)] + ncon[2] * U_2[index_nu(U2_NU, species)] + ncon[3] * U_2[index_nu(U3_NU, species)]);

    F_2[0] = U_2[index_nu(UU_NU, species)] - ncov0 * E_2;
    F_2[1] = U_2[index_nu(U1_NU, species)];
    F_2[2] = U_2[index_nu(U2_NU, species)];
    F_2[3] = U_2[index_nu(U3_NU, species)];

    E_f = 2 * E_2 - E_1;
    for (i = 0; i < NDIM; i++) F_f[i] = 2 * F_2[i] - F_1[i];

    // Compute error due to linearization
    double errE, errF;
    double dE, dFcov[NDIM], dFcon[NDIM], dFsq = 0.;
    dE = fabs(E_2 - E_1);
    errE = fabs(E_2 - E_1) / E_f / alphaRel;

    for (i = 0; i < NDIM; i++) dFcov[i] = fabs(F_2[i] - F_1[i]);
    raise(dFcov, gcon, dFcon);
    for (i = 0; i < NDIM; i++) dFsq = dFcov[i] * dFcon[i];
    errF = dFsq / E_f / alphaRel;

    *error_tmp = MY_MAX(errE, errF);

    // Check if energy density is negative:
    if (E_f < 0.0) {
        E_f = 1e-30;
        F_f[0] = 0.0;
        F_f[1] = 0.0;
        F_f[2] = 0.0;
        F_f[3] = 0.0;

        returnval += 1;
    }

    // Check if causality is satisfied:
    double errFsq = 0.;
    double Fcon[NDIM], Fsq = 0.;
    raise(F_f, gcon, Fcon);
    for (i = 0; i < NDIM; i++) Fsq = F_f[i] * Fcon[i];
    if (Fsq > E_f * E_f) {
        errFsq = (Fsq - E_f * E_f) / E_f / alphaRel;
        returnval += 20;
    }

    if (Fsq / (E_f * E_f) > y_max) {
        E_f = 1e-30;
        F_f[0] = 0.0;
        F_f[1] = 0.0;
        F_f[2] = 0.0;
        F_f[3] = 0.0;
    }
    *error_tmp = MY_MAX(*error_tmp, errFsq);

    // Update U_new
    U_new[index_nu(UU_NU, species)] = 2 * U_2[index_nu(UU_NU, species)] - U_1[index_nu(UU_NU, species)];
    U_new[index_nu(U1_NU, species)] = 2 * U_2[index_nu(U1_NU, species)] - U_1[index_nu(U1_NU, species)];
    U_new[index_nu(U2_NU, species)] = 2 * U_2[index_nu(U2_NU, species)] - U_1[index_nu(U2_NU, species)];
    U_new[index_nu(U3_NU, species)] = 2 * U_2[index_nu(U3_NU, species)] - U_1[index_nu(U3_NU, species)];
    U_new[index_nu(NUMBER_NU, species)] = 2 * U_2[index_nu(NUMBER_NU, species)] - U_1[index_nu(NUMBER_NU, species)];

    return(returnval);
}


__device__ void get_ye_predictor(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, struct of_geom* geom, double* ucon, double* ucov, double* ener_nu_avg, double* eta, double* kappa_abs, double* kappa_s, double Dt)
{
    struct of_state_nu q_nu;
    double mhd_nu[NDIM][NDIM];
    double R_dot_ucon[NDIM];
    double delJ[NU_SPECIES], delH[NDIM], delHcov[NDIM], delR_cont_covi[NU_SPECIES], J, H[NDIM];
    double wold, wnew, delw, pres, u;
    int ii;

    for (int species = 0; species < NU_SPECIES; species++) {
        get_state_nu(ph, geom, &q_nu, species);
        mhd_calc_nu(ph, 0, &q_nu, mhd_nu[0], species);
        mhd_calc_nu(ph, 1, &q_nu, mhd_nu[1], species);
        mhd_calc_nu(ph, 2, &q_nu, mhd_nu[2], species);
        mhd_calc_nu(ph, 3, &q_nu, mhd_nu[3], species);

        for (int i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_nu[i][0] * ucon[0] + mhd_nu[i][1] * ucon[1] + mhd_nu[i][2] * ucon[2] + mhd_nu[i][3] * ucon[3]);

        // R_dot_ucon[0] = -(J u^t + H^t)
        // Note that Ju^t+H^t is based on updated cons. variables

        // Energy density in the fluid frame
        J = (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]);

        // Fluxes in the fluid frame
        H[0] = -(J * ucon[0] + R_dot_ucon[0]);
        H[1] = -(J * ucon[1] + R_dot_ucon[1]);
        H[2] = -(J * ucon[2] + R_dot_ucon[2]);
        H[3] = -(J * ucon[3] + R_dot_ucon[3]);

        delJ[species] = (eta[species] / (kappa_abs[species] + 1e-30) - J) * (1. - exp(-kappa_abs[species] * Dt));
        delH[0] = -H[0] * (1. - exp(-(kappa_abs[species] + kappa_s[species]) * Dt));
        delH[1] = -H[1] * (1. - exp(-(kappa_abs[species] + kappa_s[species]) * Dt));
        delH[2] = -H[2] * (1. - exp(-(kappa_abs[species] + kappa_s[species]) * Dt));
        delH[3] = -H[3] * (1. - exp(-(kappa_abs[species] + kappa_s[species]) * Dt));

        lower(delH, geom->gcov, delHcov);
        ii = 1;
        delR_cont_covi[species] = delJ[species] * (4. / 3. * ucon[0] * ucov[ii]) + delH[0] * ucov[ii] + delHcov[ii] * ucon[0];
    }
    // Find old value of enthalpy
    eos_mode_rhotemp_pres_u(gpu_eos_table, ph[RHO], ph[UU], ph[YE], &pres, &u
    #if (DONUCLEAR)
        , &ph[XALPHA], &ph[XATM]
        #endif
    );
    wold = pres + u;

    // Change in enthalpy
    delw = -(delR_cont_covi[0] + delR_cont_covi[1] + delR_cont_covi[2]) / (ucon[0] * ucov[ii]);
    
    // Update in Ye:
    ph[YE] += (-delJ[0] + delJ[1]) * (MP_CGS) / (ph[RHO] * ucon[0]);

    // Check if Ye is in the valid range:
    ph[YE] = MY_MIN(1.0, MY_MAX(1e-10, ph[YE]));

    // Find new value of temperature based on new enthalpy
    wnew = wold + delw;
    eos_mode_rhotemp_w_pres_u(gpu_eos_table, ph[RHO], &ph[UU], ph[YE], wnew, &pres, &u
        #if (DONUCLEAR)
        , &ph[XALPHA], &ph[XATM]
        #endif
    );
}

// For explicit step:
__device__ void source_nu(double* ph, struct of_geom* geom, double* dU, double *U_i, double *U_f, double Dt, double y_max, const  double* __restrict__ gpu_eos_table, const  double* __restrict__ gpu_nulib_table
    #if (NU_KEEP_COEFF_CONST)
    , double eta_0[NU_SPECIES], double kappa_abs0[NU_SPECIES], double kappa_s0[NU_SPECIES], double eta_N0[NU_SPECIES], double kappa_N0[NU_SPECIES]
    #endif
)
{
    double mhd_nu[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM], Tg, dK_dS, bsq;
    int k, sp;
    struct of_state_nu q_nu[NU_SPECIES];

    PLOOP dU[k] = 0.;

    //Add M1 radiation terms
    for (sp = 0; sp < NU_SPECIES; sp++) {
        get_state_nu(ph, geom, &q_nu[sp], sp);
        mhd_calc_nu(ph, 0, &q_nu[sp], mhd_nu[0], sp);
        mhd_calc_nu(ph, 1, &q_nu[sp], mhd_nu[1], sp);
        mhd_calc_nu(ph, 2, &q_nu[sp], mhd_nu[2], sp);
        mhd_calc_nu(ph, 3, &q_nu[sp], mhd_nu[3], sp);

        //Add radiation 4-force
        ucon_calc(ph, geom, ucon);
        lower(ucon, geom->gcov, ucov);

        double Ncon0 = ph[index_nu(NUMBER_NU, sp)] * q_nu[sp].ucon[0];

        calc_Gcon_nu(ph, Gcon, ucon, q_nu[sp].ucon, ucov, mhd_nu, Ncon0, gpu_eos_table, gpu_nulib_table, &(dU[index_nu(NUMBER_NU, sp)]), &(dU[YE]), sp
            #if (NU_KEEP_COEFF_CONST)
            , eta_0[sp], kappa_abs0[sp], kappa_s0[sp], eta_N0[sp], kappa_N0[sp]
            #endif
        );
        lower(Gcon, geom->gcov, Gcov);

        dU[UU] += Gcov[0];
        dU[U1] += Gcov[1];
        dU[U2] += Gcov[2];
        dU[U3] += Gcov[3];

        dU[index_nu(UU_NU, sp)] = -Gcov[0];
        dU[index_nu(U1_NU, sp)] = -Gcov[1];
        dU[index_nu(U2_NU, sp)] = -Gcov[2];
        dU[index_nu(U3_NU, sp)] = -Gcov[3];
    }

    //Entropy source term
    #if(DOKTOT)
    Tg = ph[UU];
    dU[KTOT] = -(dU[UU] * ucon[0] + dU[U1] * ucon[1] + dU[U2] * ucon[2] + dU[U3] * ucon[3]) / (BOLTZ_CGS * Tg);
    #endif

    #pragma ivdep
    PLOOP{
        dU[k] *= geom->g; 
        U_f[k] = U_i[k] + dU[k] * Dt;
    }

    #if (!NU_EXPLICIT)
    int pflag, pflag_nu;
    // Invert conserved to primitives
    pflag = Utoprim_2d(U_f, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC, gpu_eos_table);
    #if(DO_FONT_FIX)
    if (pflag) {
        pflag = Utoprim_1dvsq2fix1(U_f, geom->gcov, geom->gcon, geom->g, ph, NEWT_TOL, BASIC, 1, gpu_eos_table);
    }
    #endif
    pflag_nu = Rtoprim_nu(U_f, geom, geom->gcov, geom->gcon, geom->g, ph, y_max, BASIC);
    #endif
}


__device__ void calc_Gcon_nu(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucon_nu[NDIM], double ucov[NDIM], double mhd_nu[NDIM][NDIM], double Ncon0, const  double* __restrict__ gpu_eos_table, const  double* __restrict__ gpu_nulib_table, double* source_number_nu, double *source_ye, int species
    #if (NU_KEEP_COEFF_CONST)
    , double eta_0, double kappa_abs0, double kappa_s0, double eta_N0, double kappa_N0
    #endif
)
{
    int i;
    double R_dot_ucon[NDIM];

    double u_dot_unu = ucov[0] * ucon_nu[0] + ucov[1] * ucon_nu[1] + ucov[2] * ucon_nu[2] + ucov[3] * ucon_nu[3];
    double ener_nu_avg = -u_dot_unu * ph[index_nu(UU_NU, species)] / ph[index_nu(NUMBER_NU, species)];
    #if (NU_KEEP_COEFF_CONST)
    double nu_emiss = eta_0;
    double kappa_abs = kappa_abs0;
    double kappa_scatt = kappa_s0;
    double eta_N = eta_N0;
    double kappa_N = kappa_N0;
    #else
    double nu_emiss = calc_nu_kappa_emiss(gpu_nulib_table, ph, species);
    double kappa_abs = calc_nu_kappa_abs(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);
    double kappa_scatt = calc_nu_kappa_scatt(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);    
    double eta_N = calc_nu_number_emiss(gpu_nulib_table, ph, species);
    double kappa_N = calc_nu_number_abs(gpu_eos_table, gpu_nulib_table, ph, ener_nu_avg, species);
    #endif

    for (i = 0; i < NDIM; i++) 
        R_dot_ucon[i] = (mhd_nu[i][0] * ucon[0] + mhd_nu[i][1] * ucon[1] + mhd_nu[i][2] * ucon[2] + mhd_nu[i][3] * ucon[3]);
    double J = (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]);

    for (i = 0; i < NDIM; i++) {
        Gcon[i] = -(kappa_abs * R_dot_ucon[i] + nu_emiss * ucon[i]) - kappa_scatt * (R_dot_ucon[i] + J * ucon[i]);
    }

    // number density source term
    *source_number_nu = (eta_N - kappa_N * J / ener_nu_avg);

    if (species == 0)
        *source_ye += -(*source_number_nu) * MP_CGS;
    else if (species == 1)
        *source_ye += (*source_number_nu) * MP_CGS;
}


__device__ void semiimplicit_solve_nu_init(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int *pflag_nu
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(NU_INNER_STOP)
	, double r
	#endif
) {
	double norm, dK_dS;
	int k, sp, pflag, do_entropy = 0;
	struct of_state q;
    struct of_state_nu q_nu[NU_SPECIES];


	//Set guess values for primitives after implicit step based on optical depth
	#if(NEWMAN)
	pflag = Utoprim_NM(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
	);
	#else
	pflag = Utoprim_2d(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
	);
	#endif
	#if(DO_FONT_FIX)
	if (pflag) {
		pflag = Utoprim_1dvsq2fix1(U_i, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		if (!pflag) do_entropy = 1;
	}
	#endif

	//Even if MHD inversion fails, use updated value of radiation variable as guess
    pflag_nu[0] = Rtoprim_nu(U_i, geom, geom->gcov, geom->gcon, geom->g, pb, y_max, BASIC);

	if (pflag_nu[0]) {
		for (sp = 0; sp < NU_SPECIES; sp++) {
			for (k = index_nu(UU_NU, sp); k <= index_nu(NUMBER_NU, sp); k++) U_prev[k] = U_i[k];
		}
	}
	
	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q);
	mhd_calc(pb, 0, &q, &U_f[UU], gpu_eos_table);

	for (k = UU; k <= U3; k++)U_f[k] *= geom->g;
	U_f[UU] += U_f[RHO];
	//Recompute entropy for consistency
	#if(DOKTOT)
	U_f[KTOT] = U_f[RHO] * calc_entropy(pb
		#if (DOHELM)
		, gpu_eos_table
		#endif
	);
	U_i[KTOT] = U_f[KTOT];
	#endif
	// Recompute Ye
    U_f[YE] = U_i[RHO] * pb[YE];

	//Reset inverted variables (both in case of success and failure)
	U_i[RHO] = U_f[RHO];
	U_i[UU] = U_f[UU];
	U_i[U1] = U_f[U1];
	U_i[U2] = U_f[U2];
	U_i[U3] = U_f[U3];

	for (sp = 0; sp < NU_SPECIES; sp++) {
		for (k = index_nu(UU_NU, sp); k <= index_nu(NUMBER_NU, sp); k++) U_f[k] = U_i[k];
	}
    
    error_t[0] = 1.0;
    #if(NU_INNER_STOP)
    if (r < RAD_NU_STOP) error_t[0] = 0.0;
	#endif
}

__device__ void calc_avg_neutrino_energy(double* pb, double* ener_nu_avg, struct of_geom* geom, int sp) {
    double mhd_nu[1][NDIM];
    double ucon[NDIM], ucov[NDIM];
    double u_dot_unu;
    struct of_state_nu q_nu;
    ucon_calc(pb, geom, ucon);
    lower(ucon, geom->gcov, ucov);

    get_state_nu(pb, geom, &q_nu, sp);
    mhd_calc_nu(pb, 0, &q_nu, mhd_nu[0], sp);
    u_dot_unu = ucov[0] * q_nu.ucon[0] + ucov[1] * q_nu.ucon[1] + ucov[2] * q_nu.ucon[2] + ucov[3] * q_nu.ucon[3];
    // compute avg neutrino energy in fluid frame
    *ener_nu_avg = -u_dot_unu * pb[index_nu(UU_NU, sp)] / pb[index_nu(NUMBER_NU, sp)];
}
#endif