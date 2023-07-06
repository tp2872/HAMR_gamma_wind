// EOS function calls

__device__ void validate_ye(double* ye) {
    if (*ye < 0.0) *ye = 1e-10;
    if (*ye > 1.0) *ye = 1.0;
    return;
}

__device__ void validate_abund(double* xa) {
    if (*xa < 0.0) *xa = 1e-10;
    if (*xa > 1.0) *xa = 1.0;
    return;
}

#if (DOHELM)
__device__ void validate_T(double* temp) {
	if (*temp < eos_temp_low) *temp = eos_temp_low;
	if (*temp > eos_temp_up) *temp = eos_temp_up;
	return;
}

__device__ int eos_check_input_u(double rho, double u) {
	if (u <= 0.) return 1;
	else return 0;
}

__device__ void eos_NR_temp_guess(double rho, double u, double* temp) {
	double gam = 5. / 3.;

	if (u < 0. || rho < 0.) {
		*temp = eos_temp_low;
		return;
	}

	#if (RAD_M1)
	//*temp = fabs(MMW * MH_CGS * (gam - 1.) * (u * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * rho * MASS_DENSITY_SCALE));
	*temp = pow(u * PRESSURE_SCALE / ARAD, 0.25);
	#else
	*temp = pow(u * PRESSURE_SCALE / ARAD, 0.25);
	#endif

	validate_T(temp);
	return;
}

#if (EOS_LINEAR)
__device__ void interp_eostable_linear(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double* free, double* df_d, double* df_t, double* df_tt, double* df_dt, double* dpepdd, double* etaele) {
    int iat, jat;
    double xt, xd, mxt, mxd;
    int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;

    //  hash locate this temperature and density
    jat = (int)((log10(btemp) - eos_tlo) * (double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
    jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
    iat = (int)((log10(din) - eos_dlo) * (double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
    iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

    double tstp = (eos_thi - eos_tlo) / (double)(EOSJMAX - 1);
    double dstp = (eos_dhi - eos_dlo) / (double)(EOSIMAX - 1);
    double eos_t_jat = pow(10.0, (eos_tlo + jat * tstp));
    double eos_d_iat = pow(10.0, (eos_dlo + iat * dstp));
    double eos_dt_jat = pow(10.0, (eos_tlo + (jat + 1) * tstp)) - pow(10.0, (eos_tlo + jat * tstp));
    double eos_dd_iat = pow(10.0, (eos_dlo + (iat + 1) * dstp)) - pow(10.0, (eos_dlo + iat * dstp));

    //  various differences
    xt = MY_MAX((btemp - eos_t_jat) / eos_dt_jat, 0.0); 
    xd = MY_MAX((din - eos_d_iat) / eos_dd_iat, 0.0); 
    mxt = 1.0 - xt;
    mxd = 1.0 - xd;

    // the free energy
    *free = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
            gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
            gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
            gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // derivative with respect to density
    *df_d = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
            gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
            gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
            gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // derivative with respect to temperature
    *df_t = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
            gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
            gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
            gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // second derivative with respect to temperature
    *df_tt =	gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
                gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
                gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
                gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    //  second derivative with respect to temperature and density
    *df_dt =	gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
                gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
                gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
                gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions

    //  pressure derivative with density
    *dpepdd =	gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    *dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

    //  electron chemical potential etaele
    *etaele =	gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;
}

#else

__device__ void interp_eostable(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele) {
    int iat, jat;
    double fi[36];
    double xt, xd, mxt, mxd;
    double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
    double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;
    int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;

    //  hash locate this temperature and density
    jat = (int)((log10(btemp) - eos_tlo) * (double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
    jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
    iat = (int)((log10(din) - eos_dlo) * (double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
    iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

    //  access the table locations only once
    fi[0] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[1] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[2] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[3] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[4] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[5] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[6] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[7] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[8] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[9] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[10] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[11] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[12] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat) ];
    fi[13] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[14] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[15] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[16] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[17] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[18] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[19] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[20] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[21] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[22] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[23] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[24] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[25] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[26] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[27] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[28] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[29] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[30] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[31] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[32] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[33] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[34] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[35] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    double tstp = (eos_thi - eos_tlo) / (double)(EOSJMAX - 1);
    double dstp = (eos_dhi - eos_dlo) / (double)(EOSIMAX - 1);
    double eos_t_jat = pow (10.0, (eos_tlo + jat * tstp));
    double eos_d_iat = pow (10.0, (eos_dlo + iat * dstp));
    double eos_dt_jat = pow (10.0, (eos_tlo + (jat + 1) * tstp)) - pow (10.0, (eos_tlo + jat * tstp));
    double eos_dd_iat = pow (10.0, (eos_dlo + (iat + 1) * dstp)) - pow (10.0, (eos_dlo + iat * dstp));

    //  various differences
    xt = MY_MAX((btemp - eos_t_jat) / eos_dt_jat, 0.0); // fix here
    xd = MY_MAX((din - eos_d_iat) / eos_dd_iat, 0.0); // fix here
    mxt = 1.0 - xt;
    mxd = 1.0 - xd;

    //  the density and temperature basis functions
    si0t = psi0(xt);
    si1t = psi1(xt)*eos_dt_jat; // fix here
    si2t = psi2(xt)*eos_dt_jat * eos_dt_jat; // fix here

    si0mt = psi0(mxt);
    si1mt = -psi1(mxt)*eos_dt_jat; // fix here
    si2mt = psi2(mxt)*eos_dt_jat * eos_dt_jat; // fix here

    si0d = psi0(xd);
    si1d = psi1(xd)*eos_dd_iat; // fix here
    si2d = psi2(xd)*eos_dd_iat * eos_dd_iat; // fix here

    si0md = psi0(mxd);
    si1md = -psi1(mxd)*eos_dd_iat; // fix here
    si2md = psi2(mxd)*eos_dd_iat * eos_dd_iat; // fix here

    // the free energy
    *free = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    // the first derivatives of the basis functions
    dsi0d = dpsi0(xd) / eos_dd_iat; // fix here
    dsi1d = dpsi1(xd);
    dsi2d = dpsi2(xd)*eos_dd_iat; // fix here

    dsi0md = -dpsi0(mxd) / eos_dd_iat; // fix here
    dsi1md = dpsi1(mxd);
    dsi2md = -dpsi2(mxd)*eos_dd_iat; // fix here

    // derivative with respect to density
    *df_d = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

    // the first derivatives of the basis functions
    dsi0t = dpsi0(xt) / eos_dt_jat; // fix here
    dsi1t = dpsi1(xt);
    dsi2t = dpsi2(xt)*eos_dt_jat; // fix here

    dsi0mt = -dpsi0(mxt) / eos_dt_jat; // fix here
    dsi1mt = dpsi1(mxt);
    dsi2mt = -dpsi2(mxt)*eos_dt_jat; // fix here

    // derivative with respect to temperature
    *df_t = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    // the second derivatives of the basis functions
    ddsi0t = ddpsi0(xt) / (eos_dt_jat * eos_dt_jat); // fix here
    ddsi1t = ddpsi1(xt) / eos_dt_jat; // fix here
    ddsi2t = ddpsi2(xt);
    ddsi0mt = ddpsi0(mxt) / (eos_dt_jat * eos_dt_jat); // fix here
    ddsi1mt = -ddpsi1(mxt) / eos_dt_jat; // fix here
    ddsi2mt = ddpsi2(mxt);

    // second derivative with respect to temperature
    *df_tt = h5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    //  second derivative with respect to temperature and density
    *df_dt = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    si0t = xpsi0(xt);
    si1t = xpsi1(xt)*eos_dt_jat; // fix here

    si0mt = xpsi0(mxt);
    si1mt = -xpsi1(mxt)*eos_dt_jat; // fix here

    si0d = xpsi0(xd);
    si1d = xpsi1(xd)*eos_dd_iat; // fix here

    si0md = xpsi0(mxd);
    si1md = -xpsi1(mxd)*eos_dd_iat; // fix here

    //  pressure derivative with density
    *dpepdd =   gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si0t +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si0t +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si0mt +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si0mt +

                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si1t +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si1t +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si1mt +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si1mt +

                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si0t +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si0t +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si0mt +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si0mt +

                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si1t +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si1t +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si1mt +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si1mt;

    // h3dpd(iat,jat,
    //      si0t,   si1t,   si0mt,   si1mt,
    //      si0d,   si1d,   si0md,   si1md,
    //      eos_dpdf, eos_dpdft, eos_dpdfd, eos_dpdfdt);

    *dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

    //  electron chemical potential etaele
    *etaele =   gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si0t +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si0t +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si0mt +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si0mt +

                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si1t +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si1t +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si1mt +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si1mt +

                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si0t +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si0t +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si0mt +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si0mt +

                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si1t +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si1t +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si1mt +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si1mt;

    // h3e(iat,jat,
    //                si0t,   si1t,   si0mt,   si1mt,
    //                si0d,   si1d,   si0md,   si1md);

    //  electron + positron number densities
    /*xnefer = eos_xf[(iat)*EOSJMAX + jat] * si0d*si0t + eos_xf[(iat + 1)*EOSJMAX + jat] * si0md*si0t
     + eos_xf[(iat)*EOSJMAX + jat + 1] * si0d*si0mt + eos_xf[(iat + 1)*EOSJMAX + jat + 1] * si0md*si0mt
     + eos_xft[(iat)*EOSJMAX + jat] * si0d*si1t + eos_xft[(iat + 1)*EOSJMAX + jat] * si0md*si1t
     + eos_xft[(iat)*EOSJMAX + jat + 1] * si0d*si1mt + eos_xft[(iat + 1)*EOSJMAX + jat + 1] * si0md*si1mt
     + eos_xfd[(iat)*EOSJMAX + jat] * si1d*si0t + eos_xfd[(iat + 1)*EOSJMAX + jat] * si1md*si0t
     + eos_xfd[(iat)*EOSJMAX + jat + 1] * si1d*si0mt + eos_xfd[(iat + 1)*EOSJMAX + jat + 1] * si1md*si0mt
     + eos_xfdt[(iat)*EOSJMAX + jat] * si1d*si1t + eos_xfdt[(iat + 1)*EOSJMAX + jat] * si1md*si1t
     + eos_xfdt[(iat)*EOSJMAX + jat + 1] * si1d*si1mt + eos_xfdt[(iat + 1)*EOSJMAX + jat + 1] * si1md*si1mt;*/

    // h3x(iat,jat,
    //              si0t,   si1t,   si0mt,   si1mt,
    //              si0d,   si1d,   si0md,   si1md);
}
#endif


__device__ void eos_helm_backup_nondegenerate(int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele)
{
    // Local variables
    double prad, dpraddt, erad, deraddt, srad, dsraddt;
    double pion, dpiondt, eion, deiondt, sion, dsiondt;

    // Danat: out of all derivatives w.r.t. density we only need dpdrho so far; commented out the others for the sake of optimizing the code
    double dpraddd, dpiondd;
    double deraddd, deiondd;

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;

    double deni = 1.0 / den;
    double tempi = 1.0 / btemp;

    /*
        ye = zbar/abar
        abar = zbar / ye
        ytot1 = 1/abar

        Xp = np/(np+nn)
        Ye = np/(np+nn) = Xp
        Xn = 1-Xp
        abar = Xp/1 + Xn/1 + Xa/4 = 1 - 3Xa/4
    */
    double abar, ytot1;
    // Alpha particle recombination part

    // Useful relations
    //double ytot1 = 1.0 / abar;
    //double ye = ytot1 * zbar;
    abar = 1.0; // since we only have protons and neutrons
    ytot1 = 1.0; // since we only have protons and neutrons#endif
    double kt = kerg * btemp;
    double din = ye * den;
    double kavoy = kergavo * ytot1;

    // ion portion of the gas:
    double xni = avo * ytot1 * den;
    pion = xni * kt;
    eion = 1.5 * pion * deni;
    sion = kavoy * (2.5 + log(pow(abar, 2.5) * deni * avoinv * pow(sioncon * btemp, 1.5)));

    // radiation section:
#if (RAD_M1)
    prad = erad = srad = 0.;
#else
    prad = asoli3 * btemp * btemp * btemp * btemp;
    double x1 = prad * deni;
    erad = 3.0 * x1;
    srad = (x1 + erad) * tempi;
#endif

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
    *pres = prad + pion;
    *ener = erad + eion;
    *entr = srad + sion;

    if (calc_derivatives) {
        // Calculate pressure derivatives
        dpraddt = 4.0 * prad * tempi;
        dpraddd = 0.0;
        dpiondd = avo * ytot1 * kt;
        dpiondt = xni * kerg;

        *dpresdd = dpraddd + dpiondd; // pressure derivative vs density
        *dpresdt = dpraddt + dpiondt; // pressure derivative vs temperature

        // Calculate energy derivatives
        deiondd = 0.0;
        deiondt = 1.5 * xni * kerg * deni;

        deraddd = -erad * deni;
        deraddt = 4.0 * erad * tempi;

        *denerdd = deraddd + deiondd;  // energy derivative vs density
        *denerdt = deraddt + deiondt; // energy derivative vs temperature

        // Calculate entropy derivatives
        //dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt * deni + deraddt - srad) * tempi;
        //dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
        dsiondt = (dpiondt * deni + deiondt) * tempi - (pion * deni + eion) * tempi * tempi + 1.5 * kavoy * tempi;
        //dentrdd = dsraddd + dsiondd + dsepdd; // entropy derivative vs density and density
        * dentrdt = dsraddt + dsiondt; // entropy derivative vs density and time

        // calculate relativistic soundspeeds
        double chit, z;
        chit = btemp / (*pres) * (*dpresdt);
        z = 1.0 + ((*ener) + (c_light * c_light)) * den / (*pres);
        *cs2 = (chit * chit * (*pres) * deni * tempi / (*denerdt) + (*dpresdd) * den / (*pres)) / z; // already in the units of the code (c = 1)
    }

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;
    *entr *= conv_entr_CGS2CODE;

    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
    *dentrdt *= conv_entr_CGS2CODE;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;
    *denerdd *= conv_ener_CGS2CODE * conv_dens_CODE2CGS;

    *etaele = 0.511 * mev2k / btemp;

    return;
}

__device__ void eos_helm(const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
)
{
    // Local variables
    double prad, dpraddt, erad, deraddt, srad, dsraddt;
    double pion, dpiondt, eion, deiondt, sion, dsiondt;
    double pele, dpepdt, eele, deepdt, sele, dsepdt;

    // Danat: out of all derivatives w.r.t. density we only need dpdrho so far; commented out the others for the sake of optimizing the code
    double dpraddd, dpiondd, dpepdd;
    double deraddd, deiondd, deepdd;
    double dsepdd; //dentrdd, dsraddd, dsiondd, ;

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;

    int LOWDENS_CORR = 0;
    if (den < eos_dens_low) LOWDENS_CORR = 1;

    double deni = 1.0 / den;
    double tempi = 1.0 / btemp;

    /* 
        ye = zbar/abar
        abar = zbar / ye
        ytot1 = 1/abar

        Xp = np/(np+nn)
        Ye = np/(np+nn) = Xp
        Xn = 1-Xp
        abar = Xp/1 + Xn/1 + Xa/4 = 1 - 3Xa/4
    */
    double abar, ytot1;
    // Alpha particle recombination part
    #if (DONUCLEAR)
    double x_n, x_p, x_alpha_tmp, x_atm_tmp, xn_d, xn_t, xn_y, xp_d, xp_t, xp_y, xa_d, xa_t, xa_y;
    x_alpha_tmp = *x_alpha;
    x_atm_tmp = *x_atm;
    // Compute abundances 
    if (x_atm_tmp < x_atm_cutoff && btemp > tgas_cutoff) {
        x_atm_tmp = 0.0;
        nse_abundances(den, btemp, ye, &x_n, &x_p, &x_alpha_tmp, &x_atm_tmp);
        nse_derivatives(den, btemp, ye, x_n, x_p, x_alpha_tmp, &xn_d, &xn_t, &xn_y, &xp_d, &xp_t, &xp_y, &xa_d, &xa_t, &xa_y);
    }
    else {
        x_n = get_xn(ye, x_alpha_tmp);
        x_p = get_xp(ye - x_atm_tmp, x_alpha_tmp);
        // set derivatives
        // normalize
        double x_sum = x_n + x_p + x_alpha_tmp + x_atm_tmp;
        if (x_sum > 1.0) {
            x_n = x_n / x_sum;
            x_p = x_p / x_sum;
            x_alpha_tmp = x_alpha_tmp / x_sum;
            x_atm_tmp = x_atm_tmp / x_sum;
        }
        xn_d = xn_t = xn_y = xp_d = xp_t = xp_y = xa_d = xa_t = xa_y = 0.0;
    }
    *x_alpha = x_alpha_tmp;
    *x_atm = x_atm_tmp;

    ytot1 = x_n + x_p + x_atm_tmp + 0.25 * x_alpha_tmp;
    abar = 1. / ytot1;
    //zbar = ye * abar;
    #else
    // Useful relations
    //double ytot1 = 1.0 / abar;
    //double ye = ytot1 * zbar;
    abar = 1.0; // since we only have protons and neutrons
    ytot1 = 1.0; // since we only have protons and neutrons
    #endif
    double kt = kerg * btemp;
    double din = ye * den;
    double kavoy = kergavo * ytot1;
    double xni = avo * ytot1 * den;

    // ion portion of the gas:
    pion = xni * kt;
#if (DONUCLEAR)
    eion = 1.5 * pion * deni - 0.25 * Qalpha * avo * x_alpha_tmp;
#else
    eion = 1.5 * pion * deni;
#endif
    sion = kavoy * (2.5 + log(pow(abar, 2.5) * deni * avoinv * pow(sioncon * btemp, 1.5)));

    //Look up the desired quantities in the eos table
    double free, df_d, df_t, df_tt, df_dt;
    #if (DOHELM_LOWTEMP)
    double temp_low = pow(10., eos_tlo);
    // the desired electron-positron thermodynamic quantities
    if (btemp < temp_low) {
        #if (EOS_LINEAR)
        interp_eostable_linear(gpu_eos_table, den, temp_low, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #else
        interp_eostable(gpu_eos_table, den, temp_low, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #endif
        dpepdd = 1e-30;
        //free = df_d = df_t = df_tt = df_dt = 0.0;
        pele = din * din * df_d + din * din * df_dt * (btemp - temp_low);
        sele = -df_t * ye + (-df_tt * ye) * (btemp - temp_low);
        eele = ye * free + temp_low * sele + temp_low * (-df_tt * ye) * (btemp - temp_low);
    }
    else {
        #if (EOS_LINEAR)
        interp_eostable_linear(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #else
        interp_eostable(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #endif
        // the desired electron-positron thermodynamic quantities
        pele = din * din * df_d;
        sele = -df_t * ye;
        eele = ye * free + btemp * sele;
    }
    #else
        #if (EOS_LINEAR)
        interp_eostable_linear(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #else
        interp_eostable(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        #endif
        // the desired electron-positron thermodynamic quantities
        pele = din * din * df_d * (LOWDENS_CORR == 0) + pion * ye * (LOWDENS_CORR == 1);
        sele = -df_t * ye * (LOWDENS_CORR == 0) + ye * kavoy * (2.5 + log(pow(abar, 2.5) * deni * avoinv * pow(selecon * btemp, 1.5))) * (LOWDENS_CORR == 1);
        eele = (ye * free + btemp * sele) * (LOWDENS_CORR == 0) + eion * ye * (LOWDENS_CORR == 1);
    #endif

    // radiation section:
    #if (RAD_M1)
    prad = erad = srad = 0.;
    #else
    prad = asoli3 * btemp * btemp * btemp * btemp;
    double x1 = prad * deni;
    erad = 3.0 * x1;
    srad = (x1 + erad)*tempi;
    #endif

    #if (EOS_COULOMB_CORR_GPU)
    double s1 = 4.0 / 3.0 * M_PI * xni;
    double lami = 1.0 / pow(s1, third);
    double plasg = (ye * abar) * (ye * abar) * esqu / (kt * lami);
    double plasgdd = -plasg / lami * (-third * lami / s1 * 4.0 / 3.0 * M_PI * avo * ytot1);
    double plasgdt = -plasg / btemp;

    double ecoul, pcoul, scoul, x4, x5, y3, z4, z5;
    if (plasg >= 1.0) {
        // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        x4 = pow(plasg, 0.25);
        z4 = eos_c1 / x4;
        ecoul = avo * ytot1 * kt * (eos_a1 * plasg + eos_b1 * x4 + z4 + d1cc);
        pcoul = third * den * ecoul;
        scoul = -kavoy * (3.0 * eos_b1 * x4 - 5.0 * z4 + d1cc * (log(plasg) - 1.0) - e1cc);
    }
    else if (plasg < 1.0) {
        // yakovlev & shalybkov 1989 equations 102, 103, 104
        x5 = plasg * sqrt(plasg);
        y3 = pow(plasg, eos_b2);
        z5 = eos_c2 * x5 - third * eos_a2 * y3;
        pcoul = -pion * z5;
        ecoul = 3.0 * pcoul * deni;
        scoul = -kavoy * (eos_c2 * x5 - eos_a2 * (eos_b2 - 1.0) / eos_b2 * y3);
    }
    #endif

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
    *pres = prad + pion + pele;
    *ener = erad + eion + eele;
    *entr = srad + sion + sele;
    #if (EOS_COULOMB_CORR_GPU)
    double prestot = *pres + pcoul * eos_coulombMult;

    double local_coulombMult = eos_coulombMult;
    if (prestot < 0.) local_coulombMult = 0.;

    *pres += pcoul * local_coulombMult;
    *ener += ecoul * local_coulombMult;
    *entr += scoul * local_coulombMult;
    #endif

    if (calc_derivatives) {
        // Calculate pressure derivatives
        dpraddt = 4.0 * prad * tempi;
        dpraddd = 0.0;
#if (DONUCLEAR)
        dpiondd = avo * ytot1 * kt + den * kt * avo * (xn_d + xp_d + 0.25 * xa_d);
        dpiondt = xni * kerg + den * kt * avo * (xn_t + xp_t + 0.25 * xa_t);
#else
        dpiondd = avo * ytot1 * kt;
        dpiondt = xni * kerg;
#endif

        #if (DOHELM_LOWTEMP)
        if (btemp < temp_low) {
            dpepdt = 1e-30;
        }
        else {
            dpepdt = din * din * df_dt;
        }
        #else
        dpepdt = din * din * df_dt;
        #endif

        *dpresdd = dpraddd + dpiondd * (1.+LOWDENS_CORR*ye) + dpepdd * (LOWDENS_CORR == 0); // pressure derivative vs density
        *dpresdt = dpraddt + dpiondt * (1.+LOWDENS_CORR*ye) + dpepdt * (LOWDENS_CORR == 0); // pressure derivative vs temperature
        

        // Calculate energy derivatives
#if (DONUCLEAR)
        deiondd = avo * kt * 1.5 * (xn_d + xp_d + 0.25 * xa_d) - 0.25 * xa_d * Qalpha * avo;
        deiondt = 1.5 * xni * kerg * deni + avo * kt * 1.5 * (xn_t + xp_t + 0.25 * xa_t) - 0.25 * xa_t * Qalpha * avo;
#else
        deiondd = 0.0;
        deiondt = 1.5 * xni * kerg * deni;
#endif

        deraddd = -erad * deni;
        deraddt = 4.0 * erad * tempi;
        
        #if (DOHELM_LOWTEMP)
        if (btemp < temp_low) {
            dsepdt = 1e-30;
            dsepdd = 1e-30;
            deepdt = 1e-30;
            deepdd = 1e-30;
        }
        else {
            dsepdt = -df_tt * ye;
            dsepdd = -df_dt * ye * ye;
            deepdt = btemp * dsepdt;
            deepdd = ye * ye * df_d + btemp * dsepdd;
        }
        #else
        dsepdt = -df_tt * ye;
        dsepdd = -df_dt * ye * ye;
        deepdt = btemp * dsepdt;
        deepdd = ye*ye*df_d + btemp*dsepdd;
        #endif

        *denerdd = deraddd + deiondd * (1.+LOWDENS_CORR*ye) + deepdd * (LOWDENS_CORR == 0);  // energy derivative vs density
        *denerdt = deraddt + deiondt * (1.+LOWDENS_CORR*ye) + deepdt * (LOWDENS_CORR == 0); // energy derivative vs temperature

        // Calculate entropy derivatives
        //dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt*deni + deraddt - srad)*tempi;
        //dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
#if (DONUCLEAR)
        double dadt = -(abar * abar) * (xn_t + xp_t + 0.25 * xa_t);
        dsiondt = 1.5 * kergavo * ytot1 * tempi + (2.5 * kergavo * ytot1 - sion) * ytot1 * dadt;
#else
        dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion) * tempi*tempi + 1.5 * kavoy * tempi;
#endif

        //dentrdd = dsraddd + dsiondd + dsepdd; // entropy derivative vs density and density
        *dentrdt = dsraddt + dsiondt * (1. + LOWDENS_CORR * ye) + dsepdt * (LOWDENS_CORR == 0); // entropy derivative vs density and time

        #if (EOS_COULOMB_CORR_GPU)
        double decouldd, decouldt, dpcouldd, dpcouldt, dscouldt, y1, y2, s2, s3;
        //double dscouldd;
        if (plasg >= 1.0) {
            // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
            y1 = avo * ytot1 * kt * (eos_a1 + 0.25 / plasg * (eos_b1 * x4 - z4));
            decouldd = y1 * plasgdd;
            decouldt = y1 * plasgdt + ecoul * tempi;
            dpcouldd = third * (ecoul + den * decouldd);
            dpcouldt = third * den * decouldt;
            // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
            y2 = -kavoy / plasg * (0.75 * eos_b1 * x4 + 1.25 * z4 + d1cc);
            //dscouldd = y2 * plasgdd;
            dscouldt = y2 * plasgdt;
        }
        else if (plasg < 1.0) {
            // yakovlev & shalybkov 1989 equations 102, 103, 104
            s2 = (1.5 * eos_c2 * x5 - third * eos_a2 * eos_b2 * y3) / plasg;
            dpcouldd = -dpiondd * z5 - pion * s2 * plasgdd;
            dpcouldt = -dpiondt * z5 - pion * s2 * plasgdt;
            decouldd = 3.0 * dpcouldd * deni - ecoul * deni;
            decouldt = 3.0 * dpcouldt * deni;
            // yakovlev & shalybkov 1989 equations 102, 103, 104
            s3 = -kavoy / plasg * (1.5 * eos_c2 * x5 - eos_a2 * (eos_b2 - 1.0) * y3);
            //dscouldd = s3 * plasgdd;
            dscouldt = s3 * plasgdt;
        }

        *dpresdd += dpcouldd * local_coulombMult;
        *denerdd += decouldd * local_coulombMult;
        *dpresdt += dpcouldt * local_coulombMult;
        *denerdt += decouldt * local_coulombMult;
        *dentrdt += dscouldt * local_coulombMult;
        #endif

        // calculate relativistic soundspeeds
        double chit, z;
        chit = btemp / (*pres) * (*dpresdt);
        z = 1.0 + ((*ener) + (c_light * c_light)) * den / (*pres);
        *cs2 = (chit * chit * (*pres) * deni  * tempi / (*denerdt) + (*dpresdd) * den / (*pres)) / z; // already in the units of the code (c = 1)
    }

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;
    *entr *= conv_entr_CGS2CODE;

    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
    *dentrdt *= conv_entr_CGS2CODE;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;
    *denerdd *= conv_ener_CGS2CODE * conv_dens_CODE2CGS;

    //*etaele += 0.511 * mev2k / btemp;

    return;
}


__device__ void eos_mode_rhou_pres (const  double* __restrict__ gpu_eos_table, double den, double u_goal, double ye, double *pres) {
    #if (DOHELM_TEMPERATURE == 0)
    double ener_goal = u_goal / den;
    
    // check if the input is valid:
    #if (enable_input_check)
    if (u_goal < 0.0) {
        u_goal = fabs(u_goal);
        *pres = u_goal * (GAMMA - 1.);
        return;
    }
    #endif
    
    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double entr, dsdt, dedrho;
    double cs2, etaele;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;
    int flag = 1;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif 

    #if (revert_gamma)
    if (error_e > EOS_TOL) {
        // Use GAMMA EOS in this case
        *pres = (GAMMA - 1.0) * u_goal;
        error_e = 10.0 * EOS_TOL;
    }
    #endif	

    #if (eos_nr_debug)
    if (error_e > EOS_TOL) printf("1 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);
    #endif
    #endif
}

__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double ye, double *pres, double *cs2) {
    #if (DOHELM_TEMPERATURE == 0)
    double ener_goal = u_goal / den;
    
    // check if the input is valid:
    #if (enable_input_check)
    if (u_goal < 0.0) {
        u_goal = fabs(u_goal);
        *pres = u_goal * (GAMMA - 1.);
        *cs2 = GAMMA * (GAMMA - 1.0) * u_goal / (den + GAMMA * u_goal);
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double entr, dsdt, dedrho;
    double etaele; 

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;
    int flag = 1;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif

    #if (revert_gamma)
    if (error_e > EOS_TOL) {
        *pres = (GAMMA - 1.0) * u_goal;
        *cs2 = GAMMA * (GAMMA - 1.0) * u_goal / (den + GAMMA * u_goal);
        error_e = 10.0 * EOS_TOL;
    }
    #endif

    #if (eos_nr_debug)
    if (error_e > EOS_TOL) printf("2 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);
    #endif
    #endif
}

__device__ void eos_mode_rhow_pres_dpdrho_dpde_d (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double ye, double *pres, double *dpdrho, double *dpde_d) {
    #if (DOHELM_TEMPERATURE == 0)
    double deni = 1.0 / den;
    // w_goal is w - rho
    double xenth = w_goal * deni; // Helmholtz EOS takes non-relativistic enthalpy
    // check if the input is valid:
    #if (enable_input_check)
    if (prim[UU] < 0.0) {
        *pres = (GAMMA - 1.0) * fabs(prim[UU]) / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0) * den;
        return;
    }
    #endif
 
    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, w_goal, &temp_ini_guess);
        
    double temp_new, temp_old;
    double dpdt, dedt, dhdt;
    double entr, dsdt, dedrho;
    double h_tmp;
    double cs2, etaele;

    double error, error_h;
    int i;

    double xener = 0.0;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);

        h_tmp = xener + (*pres) * deni;
        dhdt = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdt * xenth;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
        validate_T(&temp_new);

        temp_old = temp_new;
        if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_h > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC + presC * deni - xenth;
            error_h = fabs(fC / xenth);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
        *pres = presC;
        xener = enerC;
    }
    #endif

    *dpde_d = dpdt / dedt;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener * deni * (*dpde_d);
    #endif

    #if (revert_gamma)
    if (error_h > EOS_TOL) {
        *pres = (GAMMA - 1.0) * w_goal / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0) * den;
        error_h = 10.0 * EOS_TOL;
    }
    #endif

    #if (eos_nr_debug)
    if (error_h > EOS_TOL) printf("3 %g %g %g %g %g\n", error_h, temp_old, den, xenth, temp_ini_guess);
    #endif
    #endif
}

__device__ void eos_mode_rhow_pres_u (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double ye, double *pres, double *u) {
    #if (DOHELM_TEMPERATURE == 0)
    // implementation in Newman-Hamlin inversion
    double deni = 1.0 / den;
    // w_goal is w - rho
    double xenth = w_goal * deni; // Helmholtz EOS takes non-relativistic enthalpy
    // check if the input is valid:
    #if (enable_input_check)
    if (prim[UU] < 0.0) {
        *u = fabs(prim[UU]) / GAMMA;
        *pres = *u * (GAMMA - 1.);
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, w_goal, &temp_ini_guess);
    
    double temp_new, temp_old;
    double dhdtemp;
    double h_tmp;
    double cs2, etaele;
    double dpdrho, dpdt, dedt;
    double entr, dsdt, dedrho;

    double error, error_h;
    int i;

    double xener;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);

        h_tmp = xener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
        validate_T(&temp_new);

        temp_old = temp_new;
        if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
    *u = xener * den;

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_h > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC + presC * deni - xenth;
            error_h = fabs(fC / xenth);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;

            i++;
        }
        *pres = presC;
        *u = enerC * den;
    }
    #endif 

    #if (revert_gamma)
    if (error_h > EOS_TOL) {
        *u = (w_goal) / GAMMA;
        *pres = *u * (GAMMA - 1.);
        error_h = 10.0 * EOS_TOL;
    }
    #endif
    
    #if (eos_nr_debug)
    if (error_h > EOS_TOL) printf("4 %g %g %g %g %g\n", error_h, temp_old, den, xenth, temp_ini_guess);
    #endif
    #endif
}

__device__ void eos_mode_rhotemp_pres_min (const  double* __restrict__ gpu_eos_table, double den, double ye, double *pres) {
    #if (DOHELM_TEMPERATURE == 0)
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = eos_temp_low;
    double ener, dpdt, dedt, dpdrho;
    double entr, dsdt, dedrho;
    double cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, den, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
    #endif
}


// ENTROPY PART:
__device__ void get_sackur_tetrode_entropy(double den, double temp, double abar, double* entr);

__device__ void get_sackur_tetrode_entropy(double den, double temp, double abar, double *entr) {
    double deni = 1. / den;
    *entr = kergavo / abar * log(pow(abar, 2.5) * deni * avoinv * pow(sioncon * temp, 1.5));
}

// Entropy inversion
__device__ void eos_mode_rhos_upres(const double* __restrict__ gpu_eos_table, double den, double entr_goal, double ye, double *pres, double* u, double *dpdrho, double *dudrho) {
    #if (DOHELM_TEMPERATURE == 0)
    double deni = 1.0 / den;
    // check if the input is valid:
    #if (enable_input_check)
    /*if (prim[UU] < 0.0) {
        *pres = (GAMMA - 1.0) * fabs(prim[UU]) / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0);
        return;
    }*/
    #endif

    // Convert kappa to entropy
    #if (!DOHELM_FULLENTROPY)
    entr_goal = log(entr_goal) / KTOT_FACTOR;
    #endif

    // initial guess : temperature
    double temp_ini_guess;
    // Danat: if the target entropy is somehow below 0, initial guess will be screwed; setting the temp to a random value then
    if (entr_goal <= 0.0) {
        temp_ini_guess = 1e9;
    }
    else {
        temp_ini_guess = pow(den * entr_goal * conv_entr_CODE2CGS * conv_dens_CODE2CGS / asol, 1. / 3.);
    }

    double temp_new, temp_old;
    double cs2, etaele;
    double xener, xentr;
    double dedrho;
    double error, error_p;
    int i;
    double dpdt, dedt, dsdt;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, pres, &xener, &xentr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);

        temp_new = temp_old - (xentr - entr_goal) / dsdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((xentr - entr_goal) / entr_goal);
        validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_p < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double entrA, entrB, entrC;
    double fA, fB, fC;
    int flag = 1;

    if (error_p > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, pres, &xener, &entrA, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fA = entrA - entr_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, pres, &xener, &entrB, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fB = entrB - entr_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, pres, &xener, &entrC, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
            fC = entrC - entr_goal;
            error_p = fabs(fC / entr_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif

    *u = xener * den;
    *dudrho = dedrho * den + xener;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener / den * (dpdt / dedt);
    #endif

    #if (eos_nr_debug)
    if (error_p > EOS_TOL) printf("6 %g %g %g %g %g\n", error_p, temp_old, den, entr_goal, temp_ini_guess);
    #endif
    //if (isnan(entr_goal)) printf("[helm] u: %g, iters: %d, T: %g err: %g den: %g s: %g\n", (*u), i, temp_old, error_p, den, entr_goal);
    #endif
}


__device__ void eos_mode_rhou_entr(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double ye, double* entr) {
    #if (DOHELM_TEMPERATURE == 0)
    double ener_goal = u_goal / den;
    
    // check if the input is valid:
    #if (0)
    if (u_goal < 0.0) {
        u_goal = fabs(u_goal);
        *pres = u_goal * (GAMMA - 1.);
        *cs2 = GAMMA * (GAMMA - 1.0) * u_goal / (den + GAMMA * u_goal);
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double dsdt, dedrho;
    double pres, cs2, etaele;
    double error, error_e;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, &pres, &ener_tmp, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;
    int flag = 1;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, &pres, &enerA, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, &pres, &enerB, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, &pres, &enerC, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif

    #if (revert_gamma) 
    if (error_e > EOS_TOL) {
        get_sackur_tetrode_entropy(den, temp_ini_guess, 1.0, entr);
        *entr += GAMMA * ener_goal * conv_ener_CODE2CGS / temp_ini_guess;
        *entr *= conv_entr_CGS2CODE;
        error_e = EOS_TOL * 10.0;
    }
    #endif

    // Convert entropy to kappa
    #if (!DOHELM_FULLENTROPY)
    *entr = exp((*entr) * KTOT_FACTOR);
    #endif

    #if (eos_nr_debug)
    if (error_e > EOS_TOL) printf("7 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);
    #endif
    #endif
}

__device__ void eos_mode_rhou_temp(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double ye, double* temp) {
    #if (DOHELM_TEMPERATURE == 0)
    double ener_goal = u_goal / den;
    
    // check if the input is valid:
    #if (enable_input_check)
    if (u_goal < 0.0) {
        ener_goal = fabs(ener_goal);
        *temp = fabs(MMW * MH_CGS * (GAMMA - 1.) * (ener_goal * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * MASS_DENSITY_SCALE));
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess;
    eos_NR_temp_guess(den, u_goal, &temp_ini_guess);
  
    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double dsdt, dedrho;
    double pres, cs2, entr, etaele;
    double error, error_e;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(gpu_eos_table, 1, temp_old, den, ye, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);	
        validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
    *temp = temp_old;

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;
    int flag = 1;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, den, ye, &pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, den, ye, &pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, den, ye, &pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    *temp = tempC;
    #endif

    #if (revert_gamma)
    if (error_e > EOS_TOL) {
        // Revert back to GAMMA law
        *temp = fabs(MMW * MH_CGS * (GAMMA - 1.) * (ener_goal * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * MASS_DENSITY_SCALE));
        error_e = 10.0 * EOS_TOL;
    }
    #endif

    #if (eos_nr_debug)
    if (error_e > EOS_TOL) printf("8 %g %g %g %g %g\n", error_e, *temp, den, u_goal, temp_ini_guess);
    #endif
    #endif
}

// DITEMP: eos wrapper functions 
#if (DOHELM_TEMPERATURE)
__device__ void eos_mode_rhotemp_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* u
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double ener;
    double entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
    #if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *u = dens * ener;
}

__device__ void eos_mode_rhotemp_u_dudt(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* u, double* dudt
    #if (DONUCLEAR)
    , double* x_alpha, double* x_atm
    #endif
) {
    double ener, pres;
    double entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, &pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    #if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *u = dens * ener;
    *dudt = dens * dedt;
}

__device__ void eos_mode_rhotemp_pres_u_cs2(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* u, double* cs2
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double ener;
    double entr, dpdt, dedt, dsdt, dpdrho, dedrho, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
    #if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *u = dens * ener;
}

__device__ void EP_dEdW_dEdZ_dEdT(double* Eprim, double* Pprim, double* dEdvsq, double* dEdW, double* dEdT, double* dpdrho, double* dpdT, double* x, double D
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
) {
    // Compute partial derivatives of specific internal energy and pressure with respect
    // to W and Z
    // Note: E = eps - eps_EOS

    double W = x[0];
    double vsq = x[1];
    double T = x[2];

    double epsEOS, pEOS, depsEOSdrho, dpEOSdrho, depsEOSdt, dpEOSdt;
    epsEOS = 0.0;
    pEOS = 0.0;
    depsEOSdrho = 0.0;
    dpEOSdrho = 0.0;
    dpEOSdt = 0.0;

    /* need partial derivatives of specific internal energy and pressure wrt density and
    * temperature. Those need to be based on primitives computed from Newton-Raphson state
    * vector x and conservatives
    */
    double gamma = 1. / sqrt(1. - vsq);
    double rho0 = D / gamma;
    eos_mode_rhotemp_pres_u_3D_T(gpu_eos_table, rho0, T, 
        #if (DO_YE)
        ye,
        #else
        1.0,
        #endif
        &pEOS, &epsEOS, &dpEOSdrho, &dpEOSdt, &depsEOSdrho, &depsEOSdt);

    // Further partial derivatives
    double depsdW = 1.0 / (D * gamma);
    double depsdvsq = -0.5 / D * gamma * (W + pEOS / (1. - vsq)) + dpEOSdrho / (1. - vsq);
    double depsdP = -gamma / D;

    *Eprim = epsEOS;
    *Pprim = pEOS;
    *dEdvsq = depsdvsq - depsEOSdrho * 0.5 * D * gamma;
    *dEdW = depsdW;
    *dEdT = depsdP * dpEOSdt - depsEOSdt;
    *dpdrho = dpEOSdrho;
    *dpdT = dpEOSdt;
}

#if (USE_3D_INV)
__device__ void eos_mode_rhotemp_pres_u_3D_T(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* ener, double* dPdrho, double* dPdT, double* dEdrho, double* dEdT) {
    double entr, dsdt, etaele, cs2;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, pres, ener, &entr, dPdT, dEdT, &dsdt, dPdrho, dEdrho, &cs2, &etaele);
}
#endif

__device__ void eos_mode_rhotemp_pres(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double ener, entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
#if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
#endif
}

__device__ void eos_mode_rhotemp_entr(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* entr
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double pres, ener, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, &pres, &ener, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
#if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
#endif
}

#if (NEUTRINOS_M1)
__device__ void eos_mode_rhotemp_etaele(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* mu_ele
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double pres, ener, entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(gpu_eos_table, 1, temp, dens, ye, &pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
#if (DONUCLEAR)
    ener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
#endif
    // eta_ele to mu_ele
    *mu_ele = etaele - 0.511 * mev2k / temp;
}
#endif

__device__ void eos_mode_rhotemp_s_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double entr, double* pres, double* u, double* dpdrho, double* dudrho
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double temp_new, temp_old;
    double dpdt, dedt, dedrho, dsdt;
    double xener, xentr, cs2, etaele;
    double error, error_p;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    // initial guess : temperature
    double temp_ini_guess = *temp;

    double deni = 1.0 / dens;
    double entr_goal = entr;
    #if (DONUCLEAR)
    double x_alpha_ini = *x_alpha;
    double x_atm_ini = *x_atm;
    #endif
    // check if the input is valid:
    int is_valid_input = 1;
    #if (HELMEOS_INPUT_CHECK)
    double s_low, s_high;
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &xener, &s_low, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , *x_alpha, *x_atm
        #endif
    );
    // Highest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_up, dens, ye, pres, &xener, &s_high, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , *x_alpha, *x_atm
        #endif
    );
    
    if (entr_goal < s_low) {
        is_valid_input = 0;
        * u = xener * dens;
        *dudrho = dedrho * dens + xener;
        #if (inversion_w_edits)
        *dpdrho = *dpdrho - xener / dens * (dpdt / dedt);
        #endif
        *temp = eos_temp_low;
        return;
    }
    //else if (entr_goal > s_high) {
    //    is_valid_input = 0;
    //    *temp = 0.5*eos_temp_up;
    //    *u = xener * dens;
    //    *dudrho = dedrho * dens + xener;
    //    #if (inversion_w_edits)
    //    *dpdrho = *dpdrho - xener / dens * (dpdt / dedt);
    //    #endif
    //    return;
    //}
    #endif

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        if (is_valid_input) {
            eos_helm(gpu_eos_table, 1, temp_old, dens, ye, pres, &xener, &xentr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , x_alpha, x_atm
                #endif
            );
        }
        else {
            eos_helm_backup_nondegenerate(1, temp_old, dens, ye, pres, &xener, &xentr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        }

        temp_new = temp_old - (xentr - entr_goal) / dsdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((xentr - entr_goal) / entr_goal);
        validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_p < EOS_TOL) {
            more_iterations -= 1;
            *temp = temp_old;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double entrA, entrB, entrC;
    double fA, fB, fC;
    int flag = 1;

    if (error_p > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, dens, ye, pres, &xener, &entrA, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fA = entrA - entr_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, dens, ye, pres, &xener, &entrB, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        fB = entrB - entr_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, dens, ye, pres, &xener, &entrC, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
            fC = entrC - entr_goal;
            error_p = fabs(fC / entr_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
                *temp = temp_old;
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif
    #if (DONUCLEAR)
    xener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *temp = temp_new;
    *u = xener * dens;
    *dudrho = dedrho * dens + xener;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener / dens * (dpdt / dedt);
    #endif

    #if (revert_gamma)
    if (error_p > EOS_TOL) {
#if (EOS_DEBUG)
            printf("\n [EOS: s-mode (backup), too many iterations, err = %e]\nr,t,y,s: %e %e %e [%e, (min): %e] dT = %e\n", error_p, dens, *temp, ye, entr, s_low, -(xentr - entr_goal) / dsdt);
#endif
        #if (DONUCLEAR)
        *x_alpha = x_alpha_ini;
        *x_atm = x_atm_ini;
        #endif
    }
    #endif
}

__device__ void eos_mode_rhotemp_w_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double w, double* pres, double* u
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    // implementation in Newman-Hamlin inversion
    double deni = 1.0 / dens;
    // w is w - rho for this function only
    double xenth = w * deni; // Helmholtz EOS takes non-relativistic enthalpy

    // check if the input is valid:
    #if (enable_input_check)
    if (w < 0.0) {
        *u = fabs(w) / GAMMA;
        *pres = *u * (GAMMA - 1.);
        return;
    }
    #endif
    #if (DONUCLEAR)
    xenth -= 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    double x_alpha_ini = *x_alpha;
    double x_atm_ini = *x_atm;
    #endif
    // initial guess : temperature
    double temp_ini_guess = *temp;

    double temp_new, temp_old;
    double dhdtemp;
    double h_tmp;
    double cs2, etaele;
    double dpdrho, dpdt, dedt;
    double entr, dsdt, dedrho;

    double error, error_h;
    int i;

    double xener;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    // check if the input is valid:
    int is_valid_input = 1;
#if (HELMEOS_INPUT_CHECK)
    double h_low, h_high;
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
    h_low = xener + *pres * deni;
    // Highest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_up, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );
    h_high = xener + *pres * deni;

    if (xenth < h_low) {
        is_valid_input = 0;
#if (revert_gamma)
        *pres = (GAMMA - 1.0) * fabs(w) / (GAMMA);
        *temp = eos_temp_low;
        error_h = 10.0 * EOS_TOL;
#endif
        return;
    }
    //else if (xenth > h_high) {
    //    *temp = 0.5*eos_temp_up;
    //    *dpde_d = dpdt / dedt;
    //    #if (inversion_w_edits)
    //    *dpdrho = *dpdrho - xener * deni * (*dpde_d);
    //    #endif
    //    return;
    //}
#endif

    // Check if the guess temperature is already enough
    temp_old = temp_ini_guess;
    eos_helm(gpu_eos_table, 1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
#if (DONUCLEAR)
        , x_alpha, x_atm
#endif
    );

    h_tmp = xener + (*pres) * deni;
    error_h = fabs((h_tmp - xenth) / xenth);
    if (error_h < EOS_TOL) {
        *temp = temp_old;
    }
    else {
        for (i = 0; i < EOS_ITERATIONS; i++) {
            eos_helm(gpu_eos_table, 1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
    #if (DONUCLEAR)
                , x_alpha, x_atm
    #endif
            );

            h_tmp = xener + (*pres) * deni;
            dhdtemp = dedt + dpdt * deni;
            temp_new = temp_old - (h_tmp - xenth) / dhdtemp;

            // do not allow temp to change more than 2 times in one iteration
            if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
            if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

            error = fabs((temp_new - temp_old) / temp_old);
            error_h = fabs((h_tmp - xenth) / xenth);
            validate_T(&temp_new);

            temp_old = temp_new;
            if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
                more_iterations -= 1;
                *temp = temp_old;
                if (more_iterations == 0) break;
            }
        }	
    }
    #if (DONUCLEAR)
    xener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *u = xener * dens;

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_h > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, dens, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, dens, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, dens, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC + presC * deni - xenth;
            error_h = fabs(fC / xenth);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
                *temp = temp_old;
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;

            i++;
        }
        *pres = presC;
        *u = enerC * dens;
    }
    #endif

    #if (revert_gamma)
    if (error_h > EOS_TOL) {
        *u = (w) / GAMMA;
        *pres = *u * (GAMMA - 1.);
        error_h = 10.0 * EOS_TOL;
        #if (DONUCLEAR)
        *x_alpha = x_alpha_ini;
        *x_atm = x_atm_ini;
        #endif
    }
    #endif
} 

__device__ void eos_mode_rhotemp_w_pres_dpdrho_dpde_d(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double w, double* pres, double* dpdrho, double* dpde_d
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double temp_new, temp_old;
    double dpdt, dedt, dedrho, dsdt, dhdt;
    double xener, h_tmp, entr, cs2, etaele;
    double error, error_h;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    double temp_ini_guess = *temp;

    double deni = 1.0 / dens;
    // w is w - rho for this function
    double xenth = w * deni; // Helmholtz EOS takes non-relativistic enthalpy
    #if (DONUCLEAR)
    xenth -= 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    double x_alpha_ini = *x_alpha;
    double x_atm_ini = *x_atm;
    #endif
    // check if the input is valid:
    int is_valid_input = 1;
    #if (HELMEOS_INPUT_CHECK)
    double h_low, h_high;
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    h_low = xener + *pres * deni;
    // Highest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_up, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    h_high = xener + *pres * deni;

    if (xenth < h_low) {
        is_valid_input = 0;
        #if (revert_gamma)
        *pres = (GAMMA - 1.0) * fabs(w) / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0) * dens;
        *temp = eos_temp_low;
        error_h = 10.0 * EOS_TOL;
        #endif
        return;
    }
    //else if (xenth > h_high) {
    //    *temp = 0.5*eos_temp_up;
    //    *dpde_d = dpdt / dedt;
    //    #if (inversion_w_edits)
    //    *dpdrho = *dpdrho - xener * deni * (*dpde_d);
    //    #endif
    //    return;
    //}
    #endif
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    double w_low = dens * xener + *pres;

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        if (is_valid_input) {
            eos_helm(gpu_eos_table, 1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , x_alpha, x_atm
                #endif
            );
        }
        else {
            eos_helm_backup_nondegenerate(1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele);
        }
        
        h_tmp = xener + (*pres) * deni;
        dhdt = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp - xenth) / dhdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;


        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
        validate_T(&temp_new);

        temp_old = temp_new;
        if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
            more_iterations -= 1;
            *temp = temp_old;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_h > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, dens, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, dens, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, dens, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , x_alpha, x_atm
                #endif
            );
            fC = enerC + presC * deni - xenth;
            error_h = fabs(fC / xenth);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
                *temp = temp_old;
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
        *pres = presC;
        xener = enerC;
    }
    #endif

    #if (DONUCLEAR)
    xener += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    #endif
    *dpde_d = dpdt / dedt;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener * deni * (*dpde_d);
    #endif

    #if (revert_gamma)
    double gamma_2;
    if (error_h > EOS_TOL) {
#if (EOS_DEBUG)
        if (w > 0) printf("\n [EOS: w-mode, too many iterations, err = %e]\nr,t,y,w: %e %e %e [%e (min): %e] dT = %e\n", error_h, dens, *temp, ye, w, w_low, -(h_tmp - xenth) / dhdt);
#endif
        // Lowest Tgas:
        /*
        *temp = temp_new;
        eos_helm(gpu_eos_table, 1, *temp, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );
        */
        gamma_2 = GAMMA; //(*pres) / (dens * xener) + 1.;

        *pres = (gamma_2 - 1.) * w / gamma_2;
        *dpde_d = dens * (gamma_2 - 1.);
        #if (inversion_w_edits)
        *dpdrho -= (w / dens) * (gamma_2 - 1.) / gamma_2;
        #endif
        error_h = 10.0 * EOS_TOL;
        #if (DONUCLEAR)
        *x_alpha = x_alpha_ini;
        *x_atm = x_atm_ini;
        #endif
    }
    #endif
}

__device__ int eos_mode_rhotemp_u_pres_floor(const  double* gpu_eos_table, double dens, double* temp, double ye, double u, double* pres
#if (DONUCLEAR)
    , double* x_alpha, double* x_atm
#endif
) {
    double temp_new, temp_old;
    double dpdt, dedt, dsdt, dpdrho, dedrho;
    double ener_tmp, entr, cs2, etaele;
    double error, error_e;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    int is_converged = 0;

    // initial guess : temperature
    double temp_ini_guess = *temp;

    double ener_goal = u / dens;
    #if (DONUCLEAR)
    ener_goal -= 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
    double x_alpha_ini = *x_alpha;
    double x_atm_ini = *x_atm;
    #endif
    // check if the input is valid:
    int is_valid_input = 1;
    #if (HELMEOS_INPUT_CHECK)
    double e_low, e_high;
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &e_low, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    // Highest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_up, dens, ye, pres, &e_high, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    if (ener_goal < e_low) {
        is_valid_input = 0;
        #if (revert_gamma)
        // Use GAMMA EOS in this case
        *pres = (GAMMA - 1.0) * u;
        *temp = eos_temp_low;
        error_e = 10.0 * EOS_TOL;
        #endif 
        return;
    }
    //else if (ener_goal > e_high) {
    //    *temp = 0.5*eos_temp_up;
    //    return (0);
    //}
    #endif
    double e_low;
    // Lowest Tgas:
    eos_helm(gpu_eos_table, 1, eos_temp_low, dens, ye, pres, &e_low, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        if (is_valid_input) {
            eos_helm(gpu_eos_table, 1, temp_old, dens, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , x_alpha, x_atm
                #endif
            );
        }
        else {
            eos_helm_backup_nondegenerate(1, temp_old, dens, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        }
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            is_converged = 1;
            more_iterations -= 1;
            *temp = temp_old;
            if (more_iterations == 0) break;
        }
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;
    int flag = 1;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(gpu_eos_table, 1, tempA, dens, ye, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(gpu_eos_table, 1, tempB, dens, ye, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(gpu_eos_table, 1, tempC, dens, ye, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                *temp = tempC;
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif 

    #if (revert_gamma)
    double gamma_2;
    if (error_e > EOS_TOL) {
#if (EOS_DEBUG)
        printf("\n [EOS: u-mode (floor), too many iterations, err = %e]\nr,t,y,u: %e %e %e [%e (min): %e] dT = %e\n", error_e, dens, *temp, ye, u, dens*e_low, - (ener_tmp - ener_goal) / dedt);
#endif
        // Use GAMMA EOS in this case
        // Lowest Tgas:
        /*
        *temp = temp_new;
        eos_helm(gpu_eos_table, 1, *temp, dens, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );
        */
        #if (DONUCLEAR)
        ener_tmp += 0.25 * Qalpha * avo * (*x_alpha) / ENERGY_DENSITY_SCALE;
        #endif
        gamma_2 = GAMMA; //(*pres) / (dens * ener_tmp) + 1.;

        *pres = (gamma_2 - 1.) * u;
        error_e = 10.0 * EOS_TOL;
        #if (DONUCLEAR)
        *x_alpha = x_alpha_ini;
        *x_atm = x_atm_ini;
        #endif
    }
    #endif  

    return (is_converged);
}

#endif // DOHELM_TEMPERATURE
#endif //DOHELM

