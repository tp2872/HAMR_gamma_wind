#include "include.h"
#include "decs_MPI.h"

#if (DOHELM)

/*	
	Initialize Helmholtz EOS:
		read values from the table at helm_table.dat
		store the values in global arrays to be accessed by eos_helm() later 
*/
void eos_init (void) {

    FILE *fp;
    char fname_eos_table[] = "helm_table.dat";
    char buf[MAXLEN], *ptr;
    int nitems_read, nitems_expected;

    int i, j;
    double tstp, dstp;
    double eos_tstpi, eos_dstpi;

    fp = fopen(fname_eos_table, "r");
    if (NULL == fp) {
        fprintf(stderr, "Couldn't open %s for reading, exiting\n", fname_eos_table);
        exit(1234);
    }

    //..read the helmholtz free energy table
	for (j = 0; j<EOSJMAX; j++) for (i = 0; i<EOSIMAX; i++)  {
		ptr = fgets(buf, MAXLEN, fp);
        if (NULL == ptr) break;
        nitems_read = sscanf(ptr,  "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",&eos_f[i * EOSJMAX + j], &eos_fd[i * EOSJMAX + j], &eos_ft[i * EOSJMAX + j], &eos_fdd[i * EOSJMAX + j], &eos_ftt[i * EOSJMAX + j], &eos_fdt[i * EOSJMAX + j], &eos_fddt[i * EOSJMAX + j], &eos_fdtt[i * EOSJMAX + j], &eos_fddtt[i * EOSJMAX + j]);
        nitems_expected = 9;
        if (nitems_expected != nitems_read) break;
    }

    //..read the pressure derivative with density table
	for (j = 0; j<EOSJMAX; j++) for (i = 0; i<EOSIMAX; i++)  {
        ptr = fgets(buf, MAXLEN, fp);
        if (NULL == ptr) break;
        nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n",&eos_dpdf[i * EOSJMAX + j], &eos_dpdfd[i * EOSJMAX + j], &eos_dpdft[i * EOSJMAX + j], &eos_dpdfdt[i * EOSJMAX + j]);
        nitems_expected = 4;
        if (nitems_expected != nitems_read) break;
    }

    //..read the electron chemical potential table
	for (j = 0; j<EOSJMAX; j++) for (i = 0; i<EOSIMAX; i++)  {
		ptr = fgets(buf, MAXLEN, fp);
        if (NULL == ptr) break;
        nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n",&eos_ef[i * EOSJMAX + j], &eos_efd[i * EOSJMAX + j], &eos_eft[i * EOSJMAX + j], &eos_efdt[i * EOSJMAX + j]);
        nitems_expected = 4;
        if (nitems_expected != nitems_read) break;
    }

    //..read the number density table
	for (j = 0; j<EOSJMAX; j++) for (i = 0; i<EOSIMAX; i++)  {
		ptr = fgets(buf, MAXLEN, fp);
        if (NULL == ptr) break;
        nitems_read = sscanf(ptr,  "%lf %lf %lf %lf \n", &eos_xf[i * EOSJMAX + j], &eos_xfd[i * EOSJMAX + j], &eos_xft[i * EOSJMAX + j], &eos_xfdt[i * EOSJMAX + j]);
        nitems_expected = 4;
        if (nitems_expected != nitems_read) break;
    }

    fclose(fp);

    tstp  = (eos_thi - eos_tlo)/(double)(EOSJMAX-1);
	eos_tstpi = 1.0 /tstp;
    dstp  = (eos_dhi - eos_dlo)/(double)(EOSIMAX-1);
    eos_dstpi = 1.0/dstp;
    for (j=0; j<EOSJMAX; j++) eos_t[j] = pow(10.0, (eos_tlo + j*tstp));
    for (i=0; i<EOSIMAX; i++) eos_d[i] = pow(10.0, (eos_dlo + i*dstp));

    //..store the temperature and density differences and their inverses
    for (i=0; i<EOSIMAX-1; i++) eos_dd[i]   = eos_d[i+1] - eos_d[i];
    for (j=0; j<EOSJMAX-1; j++) eos_dt[j]   = eos_t[j+1] - eos_t[j];

    return;
}


/*
	Interpolation of the EOS table
		given the values of density and temperature (given that they are within the EOS bounds) computes interpolated values of thermodynamic quantities
*/

#if (EOS_LINEAR)
void interp_eostable(double den, double btemp, double din, double ye, double* free, double* df_d, double* df_t, double* df_tt, double* df_dt, double* dpepdd, double* etaele) {
    int iat, jat;
    double xt, xd, mxt, mxd;

    //  hash locate this temperature and density
    jat = (int)((log10(btemp) - eos_tlo) * (double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
    jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
    iat = (int)((log10(din) - eos_dlo) * (double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
    iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

    //  various differences
    xt = MY_MAX((btemp - eos_t[jat]) / eos_dt[jat], 0.0);
    xd = MY_MAX((din - eos_d[iat]) / eos_dd[iat], 0.0);
    mxt = 1.0 - xt;
    mxd = 1.0 - xd;

    // the free energy
    *free = eos_f[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_f[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_f[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_f[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // derivative with respect to density
    *df_d = eos_fd[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_fd[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_fd[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_fd[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // derivative with respect to temperature
    *df_t = eos_ft[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_ft[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_ft[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_ft[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // second derivative with respect to temperature
    *df_tt = eos_ftt[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_ftt[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_ftt[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_ftt[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    //  second derivative with respect to temperature and density
    *df_dt = eos_fdt[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_fdt[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_fdt[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_fdt[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions

    //  pressure derivative with density
    *dpepdd = eos_dpdf[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_dpdf[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_dpdf[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_dpdf[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

    *dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

    //  electron chemical potential etaele
    *etaele = eos_ef[(iat)*EOSJMAX + (jat)] * mxt * mxd +
        eos_ef[(iat + 1) * EOSJMAX + (jat)] * mxt * xd +
        eos_ef[(iat)*EOSJMAX + (jat + 1)] * xt * mxd +
        eos_ef[(iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;
}

#else
void interp_eostable(double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele) {
	int iat, jat;
	double fi[36];
	double xt, xd, mxt, mxd;
	double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
	double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;

	//  hash locate this temperature and density
	jat = (int)((log10(btemp) - eos_tlo)*(double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
	jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
	iat = (int)((log10(din) - eos_dlo)*(double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
	iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

	//  access the table locations only once
	fi[0] = eos_f[(iat)*EOSJMAX + (jat)];
	fi[1] = eos_f[(iat + 1)*EOSJMAX + (jat)];
	fi[2] = eos_f[(iat)*EOSJMAX + (jat + 1)];
	fi[3] = eos_f[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[4] = eos_ft[(iat)*EOSJMAX + (jat)];
	fi[5] = eos_ft[(iat + 1)*EOSJMAX + (jat)];
	fi[6] = eos_ft[(iat)*EOSJMAX + (jat + 1)];
	fi[7] = eos_ft[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[8] = eos_ftt[(iat)*EOSJMAX + (jat)];
	fi[9] = eos_ftt[(iat + 1)*EOSJMAX + (jat)];
	fi[10] = eos_ftt[(iat)*EOSJMAX + (jat + 1)];
	fi[11] = eos_ftt[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[12] = eos_fd[(iat)*EOSJMAX + (jat)];
	fi[13] = eos_fd[(iat + 1)*EOSJMAX + (jat)];
	fi[14] = eos_fd[(iat)*EOSJMAX + (jat + 1)];
	fi[15] = eos_fd[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[16] = eos_fdd[(iat)*EOSJMAX + (jat)];
	fi[17] = eos_fdd[(iat + 1)*EOSJMAX + (jat)];
	fi[18] = eos_fdd[(iat)*EOSJMAX + (jat + 1)];
	fi[19] = eos_fdd[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[20] = eos_fdt[(iat)*EOSJMAX + (jat)];
	fi[21] = eos_fdt[(iat + 1)*EOSJMAX + (jat)];
	fi[22] = eos_fdt[(iat)*EOSJMAX + (jat + 1)];
	fi[23] = eos_fdt[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[24] = eos_fddt[(iat)*EOSJMAX + (jat)];
	fi[25] = eos_fddt[(iat + 1)*EOSJMAX + (jat)];
	fi[26] = eos_fddt[(iat)*EOSJMAX + (jat + 1)];
	fi[27] = eos_fddt[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[28] = eos_fdtt[(iat)*EOSJMAX + (jat)];
	fi[29] = eos_fdtt[(iat + 1)*EOSJMAX + (jat)];
	fi[30] = eos_fdtt[(iat)*EOSJMAX + (jat + 1)];
	fi[31] = eos_fdtt[(iat + 1)*EOSJMAX + (jat + 1)];
	fi[32] = eos_fddtt[(iat)*EOSJMAX + (jat)];
	fi[33] = eos_fddtt[(iat + 1)*EOSJMAX + (jat)];
	fi[34] = eos_fddtt[(iat)*EOSJMAX + (jat + 1)];
	fi[35] = eos_fddtt[(iat + 1)*EOSJMAX + (jat + 1)];

	//  various differences
	xt = MY_MAX((btemp - eos_t[jat]) / eos_dt[jat], 0.0);
	xd = MY_MAX((din - eos_d[iat]) / eos_dd[iat], 0.0);
	mxt = 1.0 - xt;
	mxd = 1.0 - xd;

	//  the density and temperature basis functions
	si0t = psi0(xt);
	si1t = psi1(xt)*eos_dt[jat];
	si2t = psi2(xt)*eos_dt[jat] * eos_dt[jat];

	si0mt = psi0(mxt);
	si1mt = -psi1(mxt)*eos_dt[jat];
	si2mt = psi2(mxt)*eos_dt[jat] * eos_dt[jat];

	si0d = psi0(xd);
	si1d = psi1(xd)*eos_dd[iat];
	si2d = psi2(xd)*eos_dd[iat] * eos_dd[iat];

	si0md = psi0(mxd);
	si1md = -psi1(mxd)*eos_dd[iat];
	si2md = psi2(mxd)*eos_dd[iat] * eos_dd[iat];

	// the free energy
	*free = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

	// the first derivatives of the basis functions
	dsi0d = dpsi0(xd) / eos_dd[iat];
	dsi1d = dpsi1(xd);
	dsi2d = dpsi2(xd)*eos_dd[iat];

	dsi0md = -dpsi0(mxd) / eos_dd[iat];
	dsi1md = dpsi1(mxd);
	dsi2md = -dpsi2(mxd)*eos_dd[iat];

	// derivative with respect to density
	*df_d = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

	// the first derivatives of the basis functions
	dsi0t = dpsi0(xt) / eos_dt[jat];
	dsi1t = dpsi1(xt);
	dsi2t = dpsi2(xt)*eos_dt[jat];

	dsi0mt = -dpsi0(mxt) / eos_dt[jat];
	dsi1mt = dpsi1(mxt);
	dsi2mt = -dpsi2(mxt)*eos_dt[jat];

	// derivative with respect to temperature
	*df_t = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

	// the second derivatives of the basis functions
	ddsi0t = ddpsi0(xt) / (eos_dt[jat] * eos_dt[jat]);
	ddsi1t = ddpsi1(xt) / eos_dt[jat];
	ddsi2t = ddpsi2(xt);
	ddsi0mt = ddpsi0(mxt) / (eos_dt[jat] * eos_dt[jat]);
	ddsi1mt = -ddpsi1(mxt) / eos_dt[jat];
	ddsi2mt = ddpsi2(mxt);

	// second derivative with respect to temperature
	*df_tt = h5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

	//  second derivative with respect to temperature and density
	*df_dt = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

	// now get the pressure derivative with density, chemical potential, and
	// electron positron number densities
	// get the interpolation weight functions
	si0t = xpsi0(xt);
	si1t = xpsi1(xt)*eos_dt[jat];

	si0mt = xpsi0(mxt);
	si1mt = -xpsi1(mxt)*eos_dt[jat];

	si0d = xpsi0(xd);
	si1d = xpsi1(xd)*eos_dd[iat];

	si0md = xpsi0(mxd);
	si1md = -xpsi1(mxd)*eos_dd[iat];

	//  pressure derivative with density
	*dpepdd = eos_dpdf[(iat)*EOSJMAX + jat] * si0d*si0t + eos_dpdf[(iat + 1)*EOSJMAX + jat] * si0md*si0t
		+ eos_dpdf[(iat)*EOSJMAX + jat + 1] * si0d*si0mt + eos_dpdf[(iat + 1)*EOSJMAX + jat + 1] * si0md*si0mt
		+ eos_dpdft[(iat)*EOSJMAX + jat] * si0d*si1t + eos_dpdft[(iat + 1)*EOSJMAX + jat] * si0md*si1t
		+ eos_dpdft[(iat)*EOSJMAX + jat + 1] * si0d*si1mt + eos_dpdft[(iat + 1)*EOSJMAX + jat + 1] * si0md*si1mt
		+ eos_dpdfd[(iat)*EOSJMAX + jat] * si1d*si0t + eos_dpdfd[(iat + 1)*EOSJMAX + jat] * si1md*si0t
		+ eos_dpdfd[(iat)*EOSJMAX + jat + 1] * si1d*si0mt + eos_dpdfd[(iat + 1)*EOSJMAX + jat + 1] * si1md*si0mt
		+ eos_dpdfdt[(iat)*EOSJMAX + jat] * si1d*si1t + eos_dpdfdt[(iat + 1)*EOSJMAX + jat] * si1md*si1t
		+ eos_dpdfdt[(iat)*EOSJMAX + jat + 1] * si1d*si1mt + eos_dpdfdt[(iat + 1)*EOSJMAX + jat + 1] * si1md*si1mt;

	// h3dpd(iat,jat,
	//      si0t,   si1t,   si0mt,   si1mt,
	//      si0d,   si1d,   si0md,   si1md,
	//      eos_dpdf, eos_dpdft, eos_dpdfd, eos_dpdfdt);

	*dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

	//  electron chemical potential etaele
	*etaele = eos_ef[(iat)*EOSJMAX + jat] * si0d*si0t + eos_ef[(iat + 1)*EOSJMAX + jat] * si0md*si0t
		+ eos_ef[(iat)*EOSJMAX + jat + 1] * si0d*si0mt + eos_ef[(iat + 1)*EOSJMAX + jat + 1] * si0md*si0mt
		+ eos_eft[(iat)*EOSJMAX + jat] * si0d*si1t + eos_eft[(iat + 1)*EOSJMAX + jat] * si0md*si1t
		+ eos_eft[(iat)*EOSJMAX + jat + 1] * si0d*si1mt + eos_eft[(iat + 1)*EOSJMAX + jat + 1] * si0md*si1mt
		+ eos_efd[(iat)*EOSJMAX + jat] * si1d*si0t + eos_efd[(iat + 1)*EOSJMAX + jat] * si1md*si0t
		+ eos_efd[(iat)*EOSJMAX + jat + 1] * si1d*si0mt + eos_efd[(iat + 1)*EOSJMAX + jat + 1] * si1md*si0mt
		+ eos_efdt[(iat)*EOSJMAX + jat] * si1d*si1t + eos_efdt[(iat + 1)*EOSJMAX + jat] * si1md*si1t
		+ eos_efdt[(iat)*EOSJMAX + jat + 1] * si1d*si1mt + eos_efdt[(iat + 1)*EOSJMAX + jat + 1] * si1md*si1mt;

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

void eos_helm_backup_nondegenerate(int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele)
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
    * pres = prad + pion;
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
        *dentrdt = dsraddt + dsiondt; // entropy derivative vs density and time

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

/*
	Helmholtz EOS main function
		uses global arrays containing data from the table, which are filled in eos_init()
		interpolation is handled by interp_eos(): biquintic Hermite polynomials
		pressure, specific internal energy, entropy and their derivatives are outputs
*/
void eos_helm(int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
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

    // For the coulomb corrections
#if (EOS_COULOMB_CORR)
    double x4, x5, y1, y2, y3, z4, z5;
    double ktinv, dxnidd, dsdd, lami, inv_lami, lamidd, s1, s2, s3, plasg, plasg_inv, plasgdd, plasgdt;
    double ecoul, decouldd, decouldt, pcoul, dpcouldd, dpcouldt, scoul, dscouldd, dscouldt;
#endif

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;
    
    int LOWDENS_CORR = 0;
    if (den < eos_dens_low) LOWDENS_CORR = 1;

    double deni = 1.0 / den;
    double tempi = 1.0 / btemp;

    double abar, ytot1;
    // Alpha particle recombination part
    #if (DONUCLEAR)
    double x_n, x_p, x_alpha_tmp, x_atm_tmp, xn_d, xn_t, xn_y, xp_d, xp_t, xp_y, xa_d, xa_t, xa_y;
    x_alpha_tmp = x_alpha;
    x_atm_tmp = x_atm;
    // Compute abundances 
    if (x_atm_tmp < x_atm_cutoff && btemp > tgas_cutoff) {
        x_atm_tmp = 0.0;
        nse_abundances(den, btemp, ye, &x_n, &x_p, &x_alpha_tmp);
        nse_derivatives(den, btemp, ye, x_n, x_p, x_alpha_tmp, &xn_d, &xn_t, &xn_y, &xp_d, &xp_t, &xp_y, &xa_d, &xa_t, &xa_y);
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
        xn_d = xn_t = xn_y = xp_d = xp_t = xp_y = xa_d = xa_t = xa_y = 0.0;
    }

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
    pion = xni * kt;
    #if (DONUCLEAR)
    eion = 1.5 * pion * deni - 0.25 * Qalpha * avo * x_alpha_tmp;
    //if (eion < 0.0) fprintf(stderr, "\n\t [eoshelm, eion negative %g]: %g, %g, %g (%e %e %e, xatm: %e, ytot1: %e)\n", eion, 1.5 * pion * deni, 0.25 * Qalpha * avo * x_alpha_tmp, x_alpha_tmp, den, btemp, ye, x_atm_tmp, ytot1);
    #else
    eion = 1.5 * pion * deni;
    #endif
    sion = kavoy * (2.5 + log(pow(abar, 2.5) * deni * avoinv * pow(sioncon * btemp, 1.5)));

    //Look up the desired quantities in the eos table
    double free, df_d, df_t, df_dd, df_tt, df_dt;
    #if (DOHELM_LOWTEMP)
    double pele1, sele1, eele1, dpepdt_lowtemp, dsepdt_lowtemp, deepdt_lowtemp;
    double temp_low = pow(10., eos_tlo);
    if (btemp < temp_low) {
        // the desired electron-positron thermodynamic quantities
        interp_eostable(den, temp_low *(1. + 1e-5), din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        pele1 = din * din * df_d + din * din * df_dt * (btemp - temp_low);
        sele1 = -df_t * ye + (-df_tt * ye) * (btemp - temp_low);
        eele1 = ye * free + temp_low * sele + temp_low * (-df_tt * ye) * (btemp - temp_low);

        interp_eostable(den, temp_low, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);
        dpepdd = 1e-30;
        //free = df_d = df_t = df_dd = df_tt = df_dt = 0.0;
        pele = din * din * df_d + din * din * df_dt * (btemp - temp_low);
        sele = -df_t * ye + (-df_tt * ye) * (btemp - temp_low);
        eele = ye * free + temp_low * sele + temp_low * (-df_tt * ye) * (btemp - temp_low);

        dpepdt_lowtemp = (pele1 - pele) / (temp_low * 1e-5);
        dsepdt_lowtemp = (sele1 - sele) / (temp_low * 1e-5);
        deepdt_lowtemp = (eele1 - eele) / (temp_low * 1e-5);
    }
    else {
        interp_eostable(den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);

        // the desired electron-positron thermodynamic quantities
        pele = din * din * df_d;
        sele = -df_t * ye;
        eele = ye * free + btemp * sele;
    }
    #else
    interp_eostable(den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);

    // the desired electron-positron thermodynamic quantities
    pele = din * din * df_d * (LOWDENS_CORR == 0) + pion * ye * (LOWDENS_CORR == 1);
    sele = -df_t * ye * (LOWDENS_CORR == 0) + ye * kavoy * (2.5 + log(pow(abar, 2.5) * deni * avoinv * pow(selecon * btemp, 1.5))) * (LOWDENS_CORR == 1);
    eele = (ye * free + btemp * sele) * (LOWDENS_CORR == 0) + eion * ye * (LOWDENS_CORR == 1);
    #endif


    // uniform background corrections & only the needed parts for speed
    // plasg is the plasma coupling parameter
    // split up calculations below -- they all used to depend upon a redefined z
#if (EOS_COULOMB_CORR)
    s1 = 4.0 / 3.0 * M_PI * xni;
    lami = 1.0 / pow(s1, third);
    inv_lami = 1.0 / lami;
    dxnidd = avo * ytot1;
    ktinv = 1.0 / kt;
    plasg = zbar * zbar * esqu * ktinv * inv_lami;
    if (plasg >= 1.0) {
        // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        x4 = pow(plasg, 0.25);
        z4 = eos_c1 / x4;
        ecoul = dxnidd * kt * (eos_a1 * plasg + eos_b1 * x4 + z4 + d1cc);
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
#if (EOS_COULOMB_CORR)
    *pres = prad + pion + pele + pcoul * eos_coulombMult;
    *ener = erad + eion + eele + ecoul * eos_coulombMult;
    *entr = srad + sion + sele + scoul * eos_coulombMult;
#else
    *pres = prad + pion + pele;
    *ener = erad + eion + eele;
    *entr = srad + sion + sele;
#endif

    //if (*ener < 0.0) fprintf(stderr, "\n\t [eoshelm, ener negative %g]: eion: %g, xa: %g, (%e %e %e, xatm: %e, ytot1: %e)\n", *ener, eion, x_alpha_tmp, den, btemp, ye, x_atm_tmp, ytot1);

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
            //dpepdt = 0.0;
            dpepdt = din * din * df_dt + (btemp - temp_low) * dpepdt_lowtemp;
        }
        else {
            dpepdt = din * din * df_dt;
        }
        #else
        dpepdt = din * din * df_dt;
        #endif
#if (EOS_COULOMB_CORR)
        plasg_inv = 1.0 / plasg;
        if (plasg >= 1.0) {
            // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
            y1 = dxnidd * kt * (eos_a1 + 0.25 * plasg_inv * (eos_b1 * x4 - z4));
            dsdd = 4.0 / 3.0 * M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            decouldd = y1 * plasgdd;
            decouldt = y1 * plasgdt + ecoul * tempi;
            dpcouldd = third * (ecoul + den * decouldd);
            dpcouldt = third * den * decouldt;
        }
        else if (plasg < 1.0) {
            // yakovlev & shalybkov 1989 equations 102, 103, 104
            s2 = (1.5 * eos_c2 * x5 - third * eos_a2 * eos_b2 * y3) * plasg_inv;
            dxnidd = avo * ytot1;
            dsdd = 4.0 / 3.0 * M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            dpcouldd = -dpiondd * z5 - pion * s2 * plasgdd;
            dpcouldt = -dpiondt * z5 - pion * s2 * plasgdt;
            decouldd = 3.0 * dpcouldd * deni - ecoul * deni;
            decouldt = 3.0 * dpcouldt * deni;
        }
#endif

#if (EOS_COULOMB_CORR)
        *dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * eos_coulombMult; // pressure derivative vs density
        *dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * eos_coulombMult; // pressure derivative vs temperature
#else
        *dpresdd = dpraddd + dpiondd * (1. + LOWDENS_CORR * ye) + dpepdd * (LOWDENS_CORR == 0); // pressure derivative vs density
        *dpresdt = dpraddt + dpiondt * (1. + LOWDENS_CORR * ye) + dpepdt * (LOWDENS_CORR == 0); // pressure derivative vs temperature
#endif
        // Calculate energy derivatives
        #if (DONUCLEAR)
        deiondd = avo * kt * 1.5 * (xn_d + xp_d + 0.25 * xa_d) - 0.25 * xa_d * Qalpha * avo;
        deiondt = 1.5 * xni * kerg * deni + avo * kt * 1.5 * (xn_t + xp_t + 0.25 * xa_t) - 0.25 * xa_t * Qalpha * avo;
        #else
        deiondd = 0.0;
        deiondt = 1.5 * xni * kerg * deni;
        #endif

        deraddd = -erad*deni;
        deraddt = 4.0 * erad * tempi;

        #if (DOHELM_LOWTEMP)
        if (btemp < temp_low) {
            //dsepdt = 0.0;
            dsepdt = -df_tt * ye + (btemp - temp_low) * dsepdt_lowtemp;
            dsepdd = 0.0;
            //deepdt = 0.0;
            deepdt = btemp * dsepdt + (btemp - temp_low) * deepdt_lowtemp;
            deepdd = 0.0;
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

#if (EOS_COULOMB_CORR)
        *denerdd = deraddd + deiondd + deepdd + decouldd * eos_coulombMult;  // energy derivative vs density
        *denerdt = deraddt + deiondt + deepdt + decouldt * eos_coulombMult; // energy derivative vs temperature
#else 
        *denerdd = deraddd + deiondd * (1. + LOWDENS_CORR * ye) + deepdd * (LOWDENS_CORR == 0);  // energy derivative vs density
        *denerdt = deraddt + deiondt * (1. + LOWDENS_CORR * ye) + deepdt * (LOWDENS_CORR == 0); // energy derivative vs temperature
#endif

        // Calculate entropy derivatives
        //dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt * deni + deraddt - srad) * tempi;
        //dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
        #if (DONUCLEAR)
        double dadt = -(abar * abar) * (xn_t + xp_t + 0.25 * xa_t);
        dsiondt = 1.5 * kergavo * ytot1 * tempi + (2.5 * kergavo * ytot1 - sion) * ytot1 * dadt;
        #else
        dsiondt = (dpiondt * deni + deiondt) * tempi - (pion * deni + eion) * tempi * tempi + 1.5 * kavoy * tempi;
        #endif

#if (EOS_COULOMB_CORR)
        if (plasg >= 1.0) {
            // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
            y2 = -kavoy * plasg_inv * (0.75 * eos_b1 * x4 + 1.25 * z4 + d1cc);
            dscouldd = y2 * plasgdd;
            dscouldt = y2 * plasgdt;
        }
        else if (plasg < 1.0) {
            // yakovlev & shalybkov 1989 equations 102, 103, 104
            s3 = -kavoy * plasg_inv * (1.5 * eos_c2 * x5 - eos_a2 * (eos_b2 - 1.0) * y3);
            dscouldd = s3 * plasgdd;
            dscouldt = s3 * plasgdt;
        }
#endif

#if (EOS_COULOMB_CORR)
        //dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * eos_coulombMult; // entropy derivative vs density and density
        *dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * eos_coulombMult; // entropy derivative vs density and time
#else
        //dentrdd = dsraddd + dsiondd + dsepdd; // entropy derivative vs density and density
        *dentrdt = dsraddt + dsiondt * (1. + LOWDENS_CORR * ye) + dsepdt * (LOWDENS_CORR == 0); // entropy derivative vs density and time
#endif 
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
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;

    //*etaele += 0.511 * mev2k / btemp;

    return;
}

void validate_T(double* temp);

void validate_T(double* temp) {
    //#if (DOHELM_LOWTEMP)
    //if (*temp < 1e-10) *temp = 1e-10;
    //#else
    if (*temp < eos_temp_low) *temp = eos_temp_low;
    //#endif
    if (*temp > eos_temp_up) *temp = eos_temp_up;
    return;
}

// Entropy inversion
void eos_mode_rhou_entr(double* prim, double* entr) {
    // Parameters of Newton-Raphson iterations
    double tolerance_q = EOS_TOL;
    
    double den = prim[RHO];
    double u_goal = prim[UU];
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    double ener_goal = u_goal / den;
    
    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);

    double temp_new, temp_old, ener_tmp, ener_old, dpdt, dedt, dpdrho, pres, cs2;
    double dsdt, dedrho, etaele;
    double error, error_q;
    int i = 0;

    int more_iterations = 1; 
    int addtnl_iters = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    while (i < EOS_ITERATIONS && more_iterations)
    {
        eos_helm(1, temp_old, den, ye, &pres, &ener_tmp, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );

        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        temp_new = MY_MAX(0.5*temp_old, temp_new);
        temp_new = MY_MIN(2.0*temp_old, temp_new);

        error = fabs((temp_new - temp_old) / temp_old);
        error_q = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_q < tolerance_q) {
            addtnl_iters -= 1;
            if (addtnl_iters == 0) more_iterations = 0;
        }

        i++;
    }

    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;

    if (error_q > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, den, ye, &pres, &enerA, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, den, ye, &pres, &enerB, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );

        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) return;

        i = 0;
        while (i < 2 * EOS_ITERATIONS) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, den, ye, &pres, &enerC, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );

            fC = enerC - ener_goal;
            error_q = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_q < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }

    #if (!DOHELM_FULLENTROPY)
    * entr = exp((*entr) * KTOT_FACTOR);
    #endif

    if (*entr != *entr)
        fprintf(stderr, "Entr = %g\n", *entr);
}


void eos_mode_rhou_pres(double* prim, double *pres) {
    // Parameters of Newton-Raphson iterations
    double tolerance_q = EOS_TOL;
    
    double den = prim[RHO];
    double u_goal = prim[UU];
    double ener_goal = u_goal / den;
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);

    double temp_new, temp_old, ener_tmp, ener_old, dpdt, dedt, dpdrho, entr, cs2;
    double dsdt, dedrho, etaele;
    double error, error_q;
    int i = 0;

    int more_iterations = 1;
    int addtnl_iters = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    while (i < EOS_ITERATIONS && more_iterations)
    {
        eos_helm(1, temp_old, den, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        temp_new = MY_MAX(0.5 * temp_old, temp_new);
        temp_new = MY_MIN(2.0 * temp_old, temp_new);

        error = fabs((temp_new - temp_old) / temp_old);
        error_q = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_q < tolerance_q) {
            addtnl_iters -= 1;
            if (addtnl_iters == 0) more_iterations = 0;
        }

        i++;
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;

    if (error_q > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, den, ye, pres, &enerA, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, den, ye, pres, &enerB, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) return;

        i = 0;
        while (i < 2 * EOS_ITERATIONS) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, den, ye, pres, &enerC, &entr, &dpdt, &dedt, &dpdrho, &cs2
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );
            fC = enerC - ener_goal;
            error_q = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_q < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif
}

void eos_mode_rhou_pres_cs2(double* prim, double *pres, double *cs2) {
    // Parameters of Newton-Raphson iterations
    double tolerance_q = EOS_TOL;

    double den = prim[RHO];
    double u_goal = prim[UU];
    double ener_goal = u_goal / den;
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);

    double temp_new, temp_old, ener_tmp, ener_old, dpdt, dedt, dpdrho, entr;
    double dsdt, dedrho, etaele;
    double error, error_q;
    int i = 0;

    int more_iterations = 1;
    int addtnl_iters = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    while (i < EOS_ITERATIONS && more_iterations)
    {
        eos_helm(1, temp_old, den, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );

        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        temp_new = MY_MAX(0.5 * temp_old, temp_new);
        temp_new = MY_MIN(2.0 * temp_old, temp_new);

        error = fabs((temp_new - temp_old) / temp_old);
        error_q = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_q < tolerance_q) {
            addtnl_iters -= 1;
            if (addtnl_iters == 0) more_iterations = 0;
        }

        i++;
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;

    if (error_q > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, den, ye, pres, &enerA, &entr, &dpdt, &dedt, &dpdrho, cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, den, ye, pres, &enerB, &entr, &dpdt, &dedt, &dpdrho, cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) return;

        i = 0;
        while (i < 2 * EOS_ITERATIONS) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, den, ye, pres, &enerC, &entr, &dpdt, &dedt, &dpdrho, cs2
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );
            fC = enerC - ener_goal;
            error_q = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_q < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif
}

void eos_mode_rhow_pres_dpdrho_dpde_d (double* prim, double *pres, double *dpdrho, double *dpde_d) {
    // Parameters of Newton-Raphson iterations
    double tolerance_q = EOS_TOL;
    
    double den = prim[RHO];
    double deni = 1.0 / den;
    // prim[UU] is w - rho for this function only
    double xenth = prim[UU] * deni; // Helmholtz EOS takes non-relativistic enthalpy
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (xenth < 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(den * xenth * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);
    
    double xener = 0.0;
    double h_tmp;

    double temp_new, temp_old, ener_tmp, ener_old, dpdt, dedt, dhdt, entr, cs2;
    double dsdt, dedrho, etaele;
    double error, error_q;
    int i = 0;

    int more_iterations = 1;
    int addtnl_iters = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    while (i < EOS_ITERATIONS && more_iterations)
    {
        eos_helm(1, temp_old, den, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        h_tmp = xener + (*pres) * deni;
        dhdt = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdt * xenth;

        //do not allow temp to change more than 2. times in one iteration
        temp_new = MY_MAX(0.5 * temp_old, temp_new);
        temp_new = MY_MIN(2.0 * temp_old, temp_new);

        error = fabs((temp_new - temp_old) / temp_old);
        error_q = fabs((h_tmp - xenth) / xenth);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_q < tolerance_q) {
            addtnl_iters -= 1;
            if (addtnl_iters == 0) more_iterations = 0;
        }

        i++;
    }
    
    *dpde_d = dpdt / dedt;
}

void eos_mode_rhow_pres_u (double* prim, double *pres, double *u) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double tolerance_h = EOS_TOL;
    
    double den = prim[RHO];
    double deni = 1.0 / den;
    // prim[UU] is w - rho for this function only
    double xenth = prim[UU] * deni; // Helmholtz EOS takes non-relativistic enthalpy
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (xenth < 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(den * xenth * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);
    
    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp;
    double h_tmp;
    double entr, cs2;
    double dpdrho, dpdt, dedt, dpde_d;
    double dsdt, dedrho, etaele;
    
    double error, error_h;
    int i;
    
    double xener;
    
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    
    temp_old = temp_ini_guess;
    
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(1, temp_old, den, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        
        h_tmp = xener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;
        
        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;
        
        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
        validate_T(&temp_new);
        
        temp_old = temp_new;
        if(error < EOS_TEMP_TOL && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
    
    *u = xener * den;
}

void eos_mode_rhotemp_pres_min(double den, double ye, double* pres) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = eos_temp_low;
    double ener, dpdt, dedt, dpdrho;
    double entr, cs2, etaele;
    double dsdt, dedrho;
    eos_helm(1, temp, den, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , 0., 0.
        #endif
    );
}

void eos_mode_rhopres_u (double* prim) {
    // Parameters of Newton-Raphson iterations
    double tolerance_p = EOS_TOL;
    
    double den = prim[RHO];
    double deni = 1.0 / den;
    // prim[UU] is p_goal for this function only
    double p_goal = prim[UU];
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (p_goal <= 0.0) temp_ini_guess = eos_temp_low;
    temp_ini_guess = pow(p_goal * conv_pres_CODE2CGS * asoli3_inv, 0.25);

#if (DOHELM_LOWTEMP)
    temp_ini_guess = p_goal / (BOLTZ_CGS * avo * den) * C_CGS * C_CGS;
#endif
    validate_T(&temp_ini_guess);
    
    double temp_new, temp_old;
    double p_tmp;
    double entr, cs2;
    double xener;
    
    double error, error_p;
    int i;
    
    double dpdt, dedt, dpdrho;
    double dsdt, dedrho, etaele;
    
    // check if the input is valid:
    int is_valid_input = 1;
    #if (HELMEOS_INPUT_CHECK)
    double p_low, p_high;
    // Lowest Tgas:
    eos_helm(1, eos_temp_low, den, ye, &p_low, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , prim[XALPHA], prim[XATM]
        #endif
    );
    // Highest Tgas:
    eos_helm(1, eos_temp_up, den, ye, &p_high, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , prim[XALPHA], prim[XATM]
        #endif
    );

    if (p_goal < p_low || p_goal > p_high) is_valid_input = 0;
    #endif

    temp_old = temp_ini_guess;
    
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    for(i = 0; i < EOS_ITERATIONS; i++){
        if (is_valid_input) {
            eos_helm(1, temp_old, den, ye, &p_tmp, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );
        }
        else {
            eos_helm_backup_nondegenerate(1, temp_old, den, ye, &p_tmp, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        }
        temp_new = temp_old - (p_tmp - p_goal) / dpdt;
        
        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;
        
        error = fabs((temp_new - temp_old) / temp_new);
        error_p = fabs((p_tmp - p_goal) / p_goal);
        validate_T(&temp_new);
        
        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < EOS_TEMP_TOL && error_p < tolerance_p) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }

    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_p > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, den, ye, &presA, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fA = presA - p_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, den, ye, &presB, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fB = presB - p_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, den, ye, &presC, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );
            fC = presC - p_goal;
            error_p = fabs(fC / p_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif
    
    if (error_p > EOS_TOL || error > EOS_TEMP_TOL) {
        fprintf(stderr, "5 errP: %g T_ini: %g T_fin: %g rho: %g pG: %g\n", error_p, temp_ini_guess, temp_old, den, p_goal);

    }

    prim[UU] = xener * den;
}

void eos_mode_rhou_temp(double* prim, double* temp) {
    double den = prim[RHO];
    double u_goal = prim[UU];
    double ener_goal = u_goal / den;
    #if(DO_YE)
    double ye = prim[YE];
    #else
    double ye = 1.0;
    #endif
    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(u_goal * conv_pres_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double dsdt, dedrho, etaele;
    double pres, cs2, entr;
    double error, error_e;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    // DIMARK: testing below rho_low
    double rho_f = 1.0;
    //if (den < eos_dens_low) {
    //    den = eos_dens_low;
    //    rho_f = den / eos_dens_low;
    //    *temp = fabs(MMW * MH_CGS * (5. / 3. - 1.) * (u_goal * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * den * MASS_DENSITY_SCALE));
    //    return;
    //}

    // DIMARK: end of the code snippet

    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, den, ye, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

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

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, den, ye, &pres, &enerA, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, den, ye, &pres, &enerB, &entr, &dpdt, &dedt, &dpdrho, &cs2
            #if (DONUCLEAR)
            , prim[XALPHA], prim[XATM]
            #endif
        );
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) return;

        i = 0;
        while (i < 2 * EOS_ITERATIONS) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, den, ye, &pres, &enerC, &entr, &dpdt, &dedt, &dpdrho, &cs2
                #if (DONUCLEAR)
                , prim[XALPHA], prim[XATM]
                #endif
            );
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
        *temp = tempC;
    }
    #endif
}

void test_eos(void) {

}


// DITEMP: eos wrapper functions 
#if (DOHELM_TEMPERATURE)

void eos_mode_rhotemp_pres_u(double dens, double temp, double ye, double* pres, double* u
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double ener;
    double entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    *u = dens * ener;
}

void eos_mode_rhotemp_pres_u_cs2(double dens, double temp, double ye, double* pres, double* u, double* cs2
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double ener;
    double entr, dpdt, dedt, dsdt, dpdrho, dedrho, etaele;
    eos_helm(1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    *u = dens * ener;
}

void eos_mode_rhotemp_pres(double dens, double temp, double ye, double* pres
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double ener, entr, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(1, temp, dens, ye, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
}

void eos_mode_rhotemp_entr(double dens, double temp, double ye, double* entr
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double pres, ener, dpdt, dedt, dsdt, dpdrho, dedrho, cs2, etaele;
    eos_helm(1, temp, dens, ye, &pres, &ener, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );

    // Convert entropy to kappa
    #if (!DOHELM_FULLENTROPY)
    * entr = exp((*entr) * KTOT_FACTOR);
    #endif
}


void eos_mode_rhou_temp_init(double dens, double* temp, double ye, double u_goal
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double ener_goal = u_goal / dens;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = eos_temp_low;
    else temp_ini_guess = pow(u_goal * conv_pres_CODE2CGS / asol, 0.25);
    temp_ini_guess = MY_MIN(eos_temp_up, temp_ini_guess);

#if (DOHELM_LOWTEMP)
    temp_ini_guess = (GAMMA - 1.) * ener_goal / (BOLTZ_CGS * avo) * C_CGS * C_CGS;
    validate_T(&temp_ini_guess);
#endif

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double dsdt, dedrho, etaele;
    double pres, cs2, entr;
    double error, error_e;
    int i;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    // check if the input is valid:
    int is_valid_input = 1;
    #if (HELMEOS_INPUT_CHECK)
    double e_low, e_high;
    // Lowest Tgas:
    eos_helm(1, eos_temp_low, dens, ye, &pres, &e_low, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );
    // Highest Tgas:
    eos_helm(1, eos_temp_up, dens, ye, &pres, &e_high, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
        #if (DONUCLEAR)
        , x_alpha, x_atm
        #endif
    );

    if (ener_goal < e_low || ener_goal > e_high) is_valid_input = 0;
    #endif

    temp_old = temp_ini_guess;

    for (i = 0; i < EOS_ITERATIONS; i++) {
        if (is_valid_input) {
            eos_helm(1, temp_old, dens, ye, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
                #if (DONUCLEAR)
                , x_alpha, x_atm
                #endif
            );
        }
        else {
            eos_helm_backup_nondegenerate(1, temp_old, dens, ye, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele);
        }
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_new);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            *temp = temp_old;
            if (more_iterations == 0) break;
        }
    }
    *temp = temp_old;

    if (error_e > EOS_TOL || error > EOS_TEMP_TOL) {
        //*temp = temp_ini_guess;
        fprintf(stderr, "6 errE: %g, errT: %e T_ini: %g T_fin: %g rho: %g uG: %g ye: %g)\n", error_e, error, temp_ini_guess, temp_old, dens, u_goal, ye);
    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double enerA, enerB, enerC;
    double fA, fB, fC;

    if (error_e > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, dens, ye, &pres, &enerA, &entr, &dpdt, &dedt, &dpdrho, &cs2);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, &pres, &enerB, &entr, &dpdt, &dedt, &dpdrho, &cs2);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) return;

        i = 0;
        while (i < 2 * EOS_ITERATIONS) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, &pres, &enerC, &entr, &dpdt, &dedt, &dpdrho, &cs2);
            fC = enerC - ener_goal;
            error_e = fabs(fC / ener_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
        *temp = tempC;
    }
    #endif
}


void eos_mode_rhopres_temp_init(double dens, double* temp, double ye, double p_goal
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    // Parameters of Newton-Raphson iterations
    double tolerance_p = EOS_TOL;
    double deni = 1.0 / dens;

    // initial guess : temperature
    double temp_ini_guess;
    if (p_goal <= 0.0) temp_ini_guess = eos_temp_low;
    temp_ini_guess = pow(p_goal * conv_pres_CODE2CGS * asoli3_inv, 0.25);
    if (temp_ini_guess > eos_temp_up) temp_ini_guess = eos_temp_up;

    double temp_new, temp_old;
    double p_tmp;
    double entr, cs2;
    double xener;
    double dsdt, dedrho, etaele;

    double error, error_p;
    int i;

    double dpdt, dedt, dpdrho;

    temp_old = temp_ini_guess;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, dens, ye, &p_tmp, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );

        temp_new = temp_old - (p_tmp - p_goal) / dpdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((p_tmp - p_goal) / p_goal);
        validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if (error < EOS_TEMP_TOL && error_p < tolerance_p) {
            more_iterations -= 1;
            *temp = temp_old;
            if (more_iterations == 0) break;
        }

    }

    #if (EOS_BISECTION)
    // Bisection method as backup rootfinder
    double tempA, tempB, tempC;
    double presA, presB, presC;
    double fA, fB, fC;
    int flag = 1;

    if (error_p > EOS_TOL) {
        tempA = eos_temp_low;
        eos_helm(1, tempA, dens, ye, &presA, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2);
        fA = presA - p_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, &presB, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2);
        fB = presB - p_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, &presC, &xener, &entr, &dpdt, &dedt, &dpdrho, &cs2);
            fC = presC - p_goal;
            error_p = fabs(fC / p_goal);

            if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
                *temp = tempC;
                break;
            }

            if (fC * fA >= 0.0) tempA = tempC;
            else tempB = tempC;
            i++;
        }
    }
    #endif
}


void eos_mode_rhotemp_s_pres_u(double dens, double* temp, double ye, double entr, double* pres, double* u, double* dpdrho, double* dudrho
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double deni = 1.0 / dens;
    // prim[UU] is K_atm for this function only
    double entr_goal = entr;

    //#if (enable_input_check)
    #if (0)
    // Danat: if the target entropy is somehow below 0, initial guess will be screwed; setting the temp to a random value then
    if (entr_goal <= 0.0) {
        temp_ini_guess = 1e9;
    }
    else {
        temp_ini_guess = pow(dens * entr_goal * conv_entr_CODE2CGS * conv_dens_CODE2CGS / asol, 1. / 3.);
    }
    #endif

    // Convert kappa to entropy
    #if (!DOHELM_FULLENTROPY)
    entr_goal = log(entr_goal) / KTOT_FACTOR;
    #endif

    // initial guess : temperature
    double temp_ini_guess = *temp;


    double temp_new, temp_old;
    double cs2;
    double xener, xentr;
    double dedrho;
    double error, error_p;
    int i;
    double dpdt, dedt, dsdt, etaele;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, dens, ye, pres, &xener, &xentr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );

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
        eos_helm(1, tempA, dens, ye, pres, &xener, &entrA, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
        fA = entrA - entr_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, pres, &xener, &entrB, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
        fB = entrB - entr_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, pres, &xener, &entrC, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
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

    * u = xener * dens;
    *dudrho = dedrho * dens + xener;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener / dens * (dpdt / dedt);
    #endif
}

void eos_mode_rhotemp_w_pres_u(double dens, double* temp, double ye, double w, double* pres, double* u
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
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

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );

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
            *temp = temp_old;
            if (more_iterations == 0) break;
        }
    }
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
        eos_helm(1, tempA, dens, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
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
    }
    #endif
}

void eos_mode_rhotemp_w_pres_dpdrho_dpde_d(double dens, double* temp, double ye, double w, double* pres, double* dpdrho, double* dpde_d
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double deni = 1.0 / dens;
    // w is w - rho for this function
    double xenth = w * deni; // Helmholtz EOS takes non-relativistic enthalpy

    // check if the input is valid:
    #if (enable_input_check)
    if (w < 0.0) {
        *pres = (GAMMA - 1.0) * fabs(w) / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0);
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess = *temp;

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
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, dens, ye, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );

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
        eos_helm(1, tempA, dens, ye, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
        fA = enerA + presA * deni - xenth;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
        fB = enerB + presB * deni - xenth;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
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

    * dpde_d = dpdt / dedt;
    #if (inversion_w_edits)
    *dpdrho = *dpdrho - xener * deni * (*dpde_d);
    #endif

    #if (revert_gamma)
    if (error_h > EOS_TOL) {
        *pres = (GAMMA - 1.0) * (w) / (GAMMA);
        *dpdrho = 0.0;
        *dpde_d = (GAMMA - 1.0);
        error_h = 10.0 * EOS_TOL;
    }
    #endif
}


void eos_mode_rhotemp_u_pres_floor(double dens, double* temp, double ye, double u, double* pres
    #if(DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
) {
    double ener_goal = u / dens;

    // check if the input is valid:
    //#if (enable_input_check)
    #if (0)
    if (u_goal < 0.0) {
        u_goal = fabs(u_goal);
        *pres = u_goal * (GAMMA - 1.);
        return;
    }
    #endif

    // initial guess : temperature
    double temp_ini_guess = *temp;

    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
    double entr, dsdt, dedrho;
    double cs2, etaele;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for (i = 0; i < EOS_ITERATIONS; i++) {
        eos_helm(1, temp_old, dens, ye, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2, &etaele
            #if (DONUCLEAR)
            , x_alpha, x_atm
            #endif
        );
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
        eos_helm(1, tempA, dens, ye, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
        fA = enerA - ener_goal;

        tempB = eos_temp_up;
        eos_helm(1, tempB, dens, ye, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
        fB = enerB - ener_goal;

        if (fA * fB >= 0.0) flag = 0;

        i = 0;
        while (i < 2 * EOS_ITERATIONS && flag) {
            tempC = 0.5 * ((tempA)+(tempB));

            eos_helm(1, tempC, dens, ye, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
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
    if (error_e > EOS_TOL) {
        // Use GAMMA EOS in this case
        *pres = (GAMMA - 1.0) * u;
        error_e = 10.0 * EOS_TOL;
    }
    #endif	
}
#endif // DOHELM_TEMPERATURE
#endif


#if (DONUCLEAR)
double get_xp(double ye, double x_alpha) {
    return (ye - 0.5 * x_alpha);
}

double get_xn(double ye, double x_alpha) {
    return (1. - ye - 0.5 * x_alpha);
}

void nse_abundances(double rho, double tgas, double ye, double* x_n, double* x_p, double* x_alpha) {
    double n = rho / amu;
    double n_Q = pow((2.0 * M_PI * amu * BOLTZ_CGS * tgas / (PLANCK_CGS * PLANCK_CGS)), 1.5);
    double Fa, dFa;
    int i, max_iter = 50;

    *x_alpha = ye;
    for (i = 0; i < max_iter; i++) {
        *x_n = get_xn(ye, *x_alpha);
        *x_p = get_xp(ye, *x_alpha);

        Fa = pow((*x_n) * (*x_p), 2.0) - 0.5 * (*x_alpha) * pow((n_Q / n), 3.0) * exp(-Qalpha / (BOLTZ_CGS * tgas));
        dFa = -(*x_p) * pow((*x_n), 2.0) - (*x_n) * pow((*x_p), 2.0) - 0.5 * pow((n_Q / n), 3.0) * exp(-Qalpha / (BOLTZ_CGS * tgas));
        *x_alpha -= Fa / dFa;

        fprintf(stderr, "\n\t\t(iter = %d, r,t,y=%e %e %e) dx=%e, xa=%e, xn=%e, xp=%e", i, rho, tgas, ye, fabs(Fa/dFa), *x_alpha, *x_n, *x_p);

        if (fabs(Fa / dFa) < 1e-8 * (*x_alpha)) break;
    }

    *x_n = get_xn(ye, *x_alpha);
    *x_p = get_xp(ye, *x_alpha);

    return;
}

void nse_derivatives(double rho, double tgas, double ye, double x_n, double x_p, double x_alpha, double* xn_d, double* xn_t, double* xn_y, double* xp_d, double* xp_t, double* xp_y, double* xa_d, double* xa_t, double* xa_y) {
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

#endif