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

    tstp  = (11.0 - eos_tlo)/(double)(EOSJMAX-1);
	eos_tstpi = 1.0 / t;
    dstp  = (11.0 - eos_dlo)/(double)(EOSIMAX-1);
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
void interp_eostable(double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele) {
	int iat, jat;
	double fi[36];
	double xt, xd, mxt, mxd;
	double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
	double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;

	//  hash locate this temperature and density
	jat = (int)((log10(btemp) - eos_tlo)*(double)(EOSJMAX - 1) / (11.0 - eos_tlo)) + 1;
	jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
	iat = (int)((log10(din) - eos_dlo)*(double)(EOSIMAX - 1) / (11.0 - eos_dlo)) + 1;
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


/*
	Helmholtz EOS main function
		uses global arrays containing data from the table, which are filled in eos_init()
		interpolation is handled by interp_eos(): biquintic Hermite polynomials
		pressure, specific internal energy, entropy and their derivatives are outputs
*/
void eos_helm (int calc_derivatives, double btemp, double den, double abar, double zbar, double *pres, double *ener, double *entr, double *dpresdt, double *denerdt, double *dpresdd, double *dentrdt, double *dentrdd)
{
    // Local variables
    double ytot1, ye, local_coulombMult;
	double x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, deni, tempi, kt, prad, dpraddd, dpraddt, erad, deraddd, deraddt, srad, dsraddd, dsraddt;
	double xni, pion, dpiondd, dpiondt, eion, deiondd, deiondt;
	double sion, dsiondd, dsiondt;
	double pele, dpepdd, dpepdt, eele, deepdd, deepdt;
	double sele, dsepdd, dsepdt;
	double presi, gamc, kavoy;
    double etaele, xnefer, denerdd;

    // For the interpolations
    double free,df_d,df_t,df_dd,df_tt,df_dt;
	double zFunc, z0, z1, z2, z3, z4, z5, z6, din;

    // For the coulomb corrections
    double ktinv,dxnidd,dsdd,lami,inv_lami,lamidd,s0,s1,s2,s3,s4,plasg,plasg_inv,plasgdd,plasgdt,ecoul,decouldd,decouldt,pcoul,dpcouldd,dpcouldt,scoul,dscouldd,dscouldt;

    // Added by Calhoun for calculations for the Aprox13t network
    double deradda,dxnida,dpionda,deionda,dsepda,deepda,decoulda,dsda,dsdda,lamida,plasgda,denerda,deraddz,deiondz,deepdz,decouldz,dsepdz,plasgdz,denerdz;

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;

	// Useful relations
    ytot1 = 1.0 / abar;
    ye = ytot1 * zbar;
	kt = kerg * btemp;
	ktinv = 1.0 / kt;
	din = zbar * ytot1 * den;

	#if (bAprox13t)
	dsda = 4.0 / 3.0*M_PI * dxnida; //Calhoun
	lamida   = z2 * dsda / s1;      //Calhoun
	plasgda  = z3 * lamida;         //Calhoun
	plasgdz  = 2.0 * plasg/zbar;    //Calhoun
	#endif

	//Look up the desired quantities in the eos table
	interp_eostable(den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, &etaele);

	// the desired electron-positron thermodynamic quantities
	x3 = din * din;
	pele = x3 * df_d;
	sele = -df_t * ye;
	eele = ye * free + btemp * sele;

    dxnidd = avo * ytot1;
	xni = dxnidd * den;
	pion = xni * kt;

	// uniform background corrections & only the needed parts for speed
	// plasg is the plasma coupling parameter
	// split up calculations below -- they all used to depend upon a redefined z
	s1 = 4.0 / 3.0 * M_PI * xni;
	lami = 1.0 / pow(s1, third);
	inv_lami = 1.0 / lami;
	plasg = zbar*zbar*esqu*ktinv*inv_lami;

    //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
	deni = 1.0 / den;
	tempi = 1.0 / btemp;
	if (plasg >= 1.0) {
		x4 = pow(plasg, 0.25);
		z4 = eos_c1 / x4;
		ecoul = dxnidd * kt * (eos_a1*plasg + eos_b1*x4 + z4 + d1cc);
		pcoul = third * den * ecoul;
		kavoy = kergavo * ytot1;
		scoul = -kavoy*(3.0*eos_b1*x4 - 5.0*z4 + d1cc*(log(plasg) - 1.0) - e1cc);
    }
    //yakovlev & shalybkov 1989 equations 102, 103, 104
	else if (plasg < 1.0) {
		x5 = plasg * sqrt(plasg);
		y3 = pow(plasg, eos_b2);
		z5 = eos_c2 * x5 - third * eos_a2 * y3;
		pcoul = -pion * z5;
		ecoul = 3.0 * pcoul * deni;
		kavoy = kergavo * ytot1;
		scoul = -kavoy*(eos_c2*x5 - eos_a2*(eos_b2 - 1.0) / eos_b2*y3);
    }

	//  radiation section:
	prad = asoli3 * btemp * btemp * btemp * btemp;
	x1 = prad * deni;
	erad = 3.0 * x1;
	srad = (x1 + erad)*tempi;

	// Set the coulomb multiplier to a local value -- we might change it only within this call
	local_coulombMult = eos_coulombMult;

    // assume that NaN always compares as false in an inequality
	if (!(prad + pion + pele + pcoul*eos_coulombMult > 0.0)) {
        printf("[eos_helm] Negative total pressure.\n");
        printf("%s %e %e\n", " values: dens,temp: ",den,btemp);
        printf("%s %e %e\n", " values: abar,zbar: ",abar,zbar);
        printf("%s %e\n", " coulomb coupling parameter Gamma: ",plasg);

        if ( !(abar > 0.0) ) {
            printf("%s %e\n", "  However, abar is negative, abar=",abar);
            printf("%s\n", "      It is possible that the mesh is of low quality.");
            printf("%s\n", "[eos_helm] ERROR: abar is negative.");
        }

		if (prad + pion + pele > 0.0) {
			printf("%s %e %e\n", " nonpositive P caused by coulomb correction: Pnocoul,Pwithcoul: ", prad + pion + pele, prad + pion + pele + pcoul*eos_coulombMult);

            if (eos_coulombMult > 0.0) {
                printf("  set runtime parameter eos_coulombMult to zero if plasma Coulomb corrections not important\n");
            }
            if (eos_coulombAbort) {
                printf("[eos_helm] ERROR: coulomb correction causing negative total pressure.\n");
            }
            else {
                printf("Setting coulombMult to zero for this call, eos_coulombAbort=false\n");
                local_coulombMult = 0.0;
            }
        }
        else {
			printf("Prad %e\nPion %e\nPele %e\nPcoul %e\nPtot %e\ndf_d %e\n", prad, pion, pele, pcoul*eos_coulombMult, prad + pion + pele + pcoul*eos_coulombMult, df_d);
            printf("[eos_helm] ERROR: negative total pressure.\n");
        }
    }

	#if (bAprox13t)
	dxnida = -xni*ytot1; //Calhoun
	dpionda = dxnida * kt; //Calhoun
	deionda = 1.5 * dpionda*deni;  //Calhoun
	deiondz = 0.0; //Calhoun
	#endif

	//  sackur-tetrode equation for the ion entropy of
	//  a single ideal gas characterized by abar
	*pres = prad + pion + pele + pcoul * local_coulombMult;

    eion = 1.5 * pion * deni;

	*ener = erad + eion + eele + ecoul * local_coulombMult;

	sion = (pion*deni + eion)*tempi + kavoy*log(pow(abar, 2.5) * deni*avoinv *pow(sioncon * btemp, 1.5));

	*entr = srad + sion + sele + scoul * local_coulombMult;

	if (calc_derivatives){
		//Calculate pressure derivatives
		dpraddt = 4.0 * prad * tempi;
		dpraddd = 0.0;

		dpiondd = avo * ytot1 * kt;
		dpiondt = xni * kerg;

		dpepdt = x3 * df_dt;

		//yakovlev & shalybkov 1989 equations 82, 85, 86, 87
		if (plasg >= 1.0) {
			plasg_inv = 1.0 / plasg;
			y1 = dxnidd * kt * (eos_a1 + 0.25*plasg_inv*(eos_b1*x4 - z4));
			dsdd = 4.0 / 3.0*M_PI * dxnidd;
			lamidd = -third * lami / s1 * dsdd;
			plasgdd = -plasg * inv_lami * lamidd;
			plasgdt = -plasg * ktinv * kerg;
			decouldd = y1 * plasgdd;
			decouldt = y1 * plasgdt + ecoul * tempi;
			dpcouldd = third * (ecoul + den * decouldd);
			dpcouldt = third * den  * decouldt;
		}
		//yakovlev & shalybkov 1989 equations 102, 103, 104
		else if (plasg < 1.0) {
			plasg_inv = 1.0 / plasg;
			s2 = (1.5*eos_c2*x5 - third*eos_a2*eos_b2*y3)*plasg_inv;
			dxnidd = avo * ytot1;
			dsdd = 4.0 / 3.0*M_PI * dxnidd;
			lamidd = -third * lami / s1 * dsdd;
			plasgdd = -plasg * inv_lami * lamidd;
			plasgdt = -plasg * ktinv * kerg;
			dpcouldd = -dpiondd*z5 - pion*s2*plasgdd;
			dpcouldt = -dpiondt*z5 - pion*s2*plasgdt;
			decouldd = 3.0*dpcouldd*deni - ecoul*deni;
			decouldt = 3.0*dpcouldt*deni;
		}
		*dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * local_coulombMult; //pressure derivative vs density
		*dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * local_coulombMult; //pressure derivative vs temperature

		//Calculate energy derivatives
		deiondd = (1.5 * dpiondd - eion)*deni;
        deiondt = 1.5 * xni * kerg *deni;

		deraddd = -erad*deni;
		deraddt = 4.0 * erad * tempi;
		dsepdt = -df_tt * ye;
		dsepdd = -df_dt * ye * ye;
		deepdt = btemp * dsepdt;
		deepdd = ye*ye*df_d + btemp*dsepdd;
		#if (bAprox13t)
		//  Calhoun next two lines
		deradda = 0.0;
		deraddz = 0.0;
		#endif
		denerdd = deraddd + deiondd + deepdd + decouldd * local_coulombMult; //energy derivative vs density
		*denerdt = deraddt + deiondt + deepdt + decouldt * local_coulombMult; //energy derivative vs temperature

		//Calculate entropy derivatives
		dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
		dsraddt = (dpraddt*deni + deraddt - srad)*tempi;

		dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
		dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion) * tempi*tempi + 1.5 * kavoy * tempi;

		//yakovlev & shalybkov 1989 equations 82, 85, 86, 87
		if (plasg >= 1.0) {
			y2 = -kavoy*plasg_inv*(0.75*eos_b1*x4 + 1.25*z4 + d1cc);
			dscouldd = y2 * plasgdd;
			dscouldt = y2 * plasgdt;
		}
		//yakovlev & shalybkov 1989 equations 102, 103, 104
		else if (plasg < 1.0) {
			s3 = -kavoy*plasg_inv*(1.5*eos_c2*x5 - eos_a2*(eos_b2 - 1.0)*y3);
			dscouldd = s3 * plasgdd;
			dscouldt = s3 * plasgdt;
		}
		*dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * local_coulombMult;//entropy derivative vs density and density
		*dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * local_coulombMult;//entropy derivative vs density and time

        // calculate relativistic soundspeeds
		double zz, zzi, chit, chid, cv, cp, x, gam1, gam2, gam3, nabad, z, c_sound;
		zz = (*pres) * deni;
		zzi = den / (*pres);
		chit = btemp / (*pres) * (*dpresdt);
		chid = (*dpresdd) * zzi;
		cv = (*denerdt);
		x = zz * chit / (btemp * cv);
		gam3 = x + 1.0;
		gam1 = chit * x + chid;
		nabad = x / gam1;
		gam2 = 1.0 / (1.0 - nabad);
		cp = cv * gam1 / chid;
		z = 1.0 + ((*ener) + (c_light * c_light)) * zzi;
		c_sound = c_light * sqrt(gam1 / z);
	}

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;
    *entr *= conv_entr_CGS2CODE;


    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;

    return;
}

#if (DONUCLEAR)
// DANAT: When DONUCLEAR = 1; abar stands for Ye and zbar - for xx_atm
void eos_helm_nuclear(int calc_derivatives, double btemp, double den, double ye, double *xx_atm, double *xxn, double *xxp, double *xxa, double *etaele, double *pres, double *ener, double *entr, double *dpresdt, double *denerdt, double *dpresdd, double *dentrdt, double *dentrdd)
{
    // Local variables
    double abar, zbar;
    double ytot1, local_coulombMult;
    double  x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, deni, tempi, kt, prad, dpraddd, dpraddt, erad, deraddd, deraddt, srad, dsraddd, dsraddt;
    double  xni, pion, dpiondd, dpiondt, eion, deiondd, deiondt;
    double sion, dsiondd, dsiondt;
    double pele, dpepdd, dpepdt, eele, deepdd, deepdt;
    double sele, dsepdd, dsepdt;
    double presi, chit, chid, gamc, kavoy;
    double cv, cp, xnefer,denerdd;

    // For the interpolations
    double free,df_d,df_t,df_dd,df_tt,df_dt;
    //double h3e, h3x;
    double zFunc, z0, z1, z2, z3, z4, z5, z6, din;

    // For the coulomb corrections
    double  ktinv,dxnidd,dsdd,lami,inv_lami,lamidd,s0,s1,s2,s3,s4,plasg,plasg_inv,plasgdd,plasgdt,ecoul,decouldd,decouldt,pcoul,dpcouldd,dpcouldt,scoul,dscouldd,dscouldt;

    // Added by Calhoun for calculations for the Aprox13t network
    double  deradda,dxnida,dpionda,deionda,dsepda,deepda,decoulda,dsda,dsdda,lamida,plasgda,denerda,deraddz,deiondz,deepdz,decouldz,dsepdz,plasgdz,denerdz;

    // Nuclear: variables for NSE corrections
    double xxn_r, xxn_t, xxn_y, xxp_r, xxp_t, xxp_y, xxa_r, xxa_t, xxa_y;
    double dadd, dadt, dady;

    // Convert from code units to cgs
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;
    
    double xcutoff = 0.1;
    // Nuclear: Compute abundances in NSE
    // From RF:
    if (*xx_atm < xcutoff && btemp > 5.0e9) {
        *xx_atm = 0.0;

        nse_abundance (den, btemp, ye, xxn, xxp, xxa);
        nse_derivatives (den, btemp, *xxn, *xxp, *xxa, &xxa_r, &xxa_t, &xxa_y, &xxn_r, &xxn_t, &xxn_y, &xxp_r, &xxp_t, &xxp_y);

        if (xxa_r != xxa_r) {
            1==1;
        }
        //RF: Prevent anomalous aboundances (for testing purposes)
        *xxn = max (*xxn, 1.0e-10);
        *xxn = min (*xxn, 1.0);
        *xxp = max (*xxp, 1.0e-10);
        *xxp = min (*xxp, 1.0);
        *xxa = max (*xxa, 1.0e-10);
        *xxa = min (*xxa, 1.0);

        if (*xxn > 1.0 || *xxp > 1.0 || *xxa > 1.0) {
            printf ("ld, lT, ye = %e, %e, %e\n", log10(den), log10(btemp), ye);
            printf ("xn, xp, xa = %e, %e, %e\n", *xxn, *xxp, *xxa);
            printf ("eos_helm: wrong abundances out of nse\n");
        }
    }
    else {
        //Freeze mass fractions
        xxa_r=0.; xxa_t=0.; xxa_y=0.; xxn_r=0.; xxn_t=0.; xxn_y=0.; xxp_r=0.; xxp_t=0.; xxp_y=0.;

        // Make sure things remain normalized
        if (fabs (*xx_atm + *xxn + *xxp + *xxa - 1.0) > 1e-8) {
            *xx_atm = *xx_atm / (*xx_atm + *xxn + *xxp + *xxa);
            *xxn    = *xxn    / (*xx_atm + *xxn + *xxp + *xxa);
            *xxp    = *xxp    / (*xx_atm + *xxn + *xxp + *xxa);
            *xxa    = *xxa    / (*xx_atm + *xxn + *xxp + *xxa);
        }
    }

    abar  = 1.0 / (*xx_atm + *xxn + *xxp + *xxa / 4.0);
    if (abar < 0.0) {
        1+2==3;
    }
    zbar  = abar * ye;
    ytot1 = 1.0 / abar;

    kt = kerg * btemp;
    ktinv = 1.0 / kt;

#if (bAprox13t)
    dsda = 4.0 / 3.0*M_PI * dxnida;          //Calhoun
    lamida   = z2 * dsda / s1;      //Calhoun
    plasgda  = z3 * lamida;         //Calhoun
    plasgdz  = 2.0 * plasg/zbar;   //Calhoun
#endif

    //frequent combinations
    din = ye * den;

    //Look up the desired quantities in the eos table
    interp_eostable(den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, etaele);

    //  the desired electron-positron thermodynamic quantities
    x3 = din * din;
    pele = x3 * df_d;
    sele = -df_t * ye;
    eele = ye * free + btemp * sele;

    dxnidd = avo * ytot1;
    xni = dxnidd * den;
    pion = xni * kt;

    //  uniform background corrections & only the needed parts for speed
    //  plasg is the plasma coupling parameter
    //  split up calculations below -- they all used to depend upon a redefined z
    s1 = 4.0 / 3.0*M_PI * xni;
    lami = 1.0 / pow(s1, third);
    inv_lami = 1.0 / lami;
    plasg = zbar*zbar*esqu*ktinv*inv_lami;

    //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    deni = 1.0 / den;
    tempi = 1.0 / btemp;
    if (plasg >= 1.0) {
        x4 = pow(plasg, 0.25);
        z4 = eos_c1 / x4;
        ecoul = dxnidd * kt * (eos_a1*plasg + eos_b1*x4 + z4 + d1cc);
        pcoul = third * den * ecoul;
        kavoy = kergavo * ytot1;
        scoul = -kavoy*(3.0*eos_b1*x4 - 5.0*z4 + d1cc*(log(plasg) - 1.0) - e1cc);
    }
    //yakovlev & shalybkov 1989 equations 102, 103, 104
    else if (plasg < 1.0) {
        x5 = plasg * sqrt(plasg);
        y3 = pow(plasg, eos_b2);
        z5 = eos_c2 * x5 - third * eos_a2 * y3;
        pcoul = -pion * z5;
        ecoul = 3.0 * pcoul * deni;
        kavoy = kergavo * ytot1;
        scoul = -kavoy*(eos_c2*x5 - eos_a2*(eos_b2 - 1.0) / eos_b2*y3);
    }

    //  radiation section:
    prad = asoli3 * btemp * btemp * btemp * btemp;
    x1 = prad * deni;
    erad = 3.0 * x1;
    srad = (x1 + erad)*tempi;

    // Set the coulomb multiplier to a local value -- we might change it only within this call
    local_coulombMult = eos_coulombMult;

    // assume that NaN always compares as false in an inequality
    if (!(prad + pion + pele + pcoul*eos_coulombMult > 0.0)) {
        printf("[eos_helm] Negative total pressure.\n");
        printf("%s %e %e\n", " values: dens,temp: ",den,btemp);
        printf("%s %e %e\n", " values: abar,zbar: ",abar,zbar);
        printf("%s %e\n", " coulomb coupling parameter Gamma: ",plasg);

        if ( !(abar > 0.0) ) {
            printf("%s %e\n", "  However, abar is negative, abar=",abar);
            printf("%s\n", "      It is possible that the mesh is of low quality.");
            printf("%s\n", "[eos_helm] ERROR: abar is negative.");
        }

        if (prad + pion + pele > 0.0) {
            printf("%s %e %e\n", " nonpositive P caused by coulomb correction: Pnocoul,Pwithcoul: ", prad + pion + pele, prad + pion + pele + pcoul*eos_coulombMult);

            if (eos_coulombMult > 0.0) {
                printf("  set runtime parameter eos_coulombMult to zero if plasma Coulomb corrections not important\n");
            }
            if (eos_coulombAbort) {
                printf("[eos_helm] ERROR: coulomb correction causing negative total pressure.\n");
            }
            else {
                printf("Setting coulombMult to zero for this call, eos_coulombAbort=false\n");
                local_coulombMult = 0.0;
            }
        }
        else {
            printf("Prad %e\nPion %e\nPele %e\nPcoul %e\nPtot %e\ndf_d %e\n", prad, pion, pele, pcoul*eos_coulombMult, prad + pion + pele + pcoul*eos_coulombMult, df_d);
            printf("[eos_helm] ERROR: negative total pressure.\n");
        }
    }

#if (bAprox13t)
    dxnida = -xni*ytot1; //Calhoun
    dpionda = dxnida * kt; //Calhoun
    deionda = 1.5 * dpionda*deni;  //Calhoun
    deiondz = 0.0; //Calhoun
#endif

    //  sackur-tetrode equation for the ion entropy of
    //  a single ideal gas characterized by abar
    *pres = prad + pion + pele + pcoul * local_coulombMult;

    // Nuclear: edited
    eion = 1.5 * pion * deni - (Qalpha / 4.0) * avo * (*xxa);

    *ener = erad + eion + eele + ecoul * local_coulombMult;

    // Nuclear: edited
    sion    = kergavo * ytot1 * (2.5 + log(pow (abar, 2.5) * deni * avoinv * pow(sioncon * btemp, 1.5)));
    dadd    = -abar * abar * (xxn_r + xxp_r + xxa_r / 4.0);
    dadt    = -abar * abar * (xxn_t + xxp_t + xxa_t / 4.0);

    *entr = srad + sion + sele + scoul * local_coulombMult;

    if (calc_derivatives){
        //Calculate pressure derivatives
        dpraddt = 4.0 * prad * tempi;
        dpraddd = 0.0;

        // Nuclear: edited
        dpiondd = dxnidd * kt + den * kt * avo * (xxn_r + xxp_r + xxa_r / 4.0);
        dpiondt = xni * kerg  + den * kt * avo * (xxn_t + xxp_t + xxa_t / 4.0);

        dpepdt = x3 * df_dt;

        //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        if (plasg >= 1.0) {
            plasg_inv = 1.0 / plasg;
            y1 = dxnidd * kt * (eos_a1 + 0.25*plasg_inv*(eos_b1*x4 - z4));
            dsdd = 4.0 / 3.0*M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            decouldd = y1 * plasgdd;
            decouldt = y1 * plasgdt + ecoul * tempi;
            dpcouldd = third * (ecoul + den * decouldd);
            dpcouldt = third * den  * decouldt;
        }
        //yakovlev & shalybkov 1989 equations 102, 103, 104
        else if (plasg < 1.0) {
            plasg_inv = 1.0 / (plasg + SMALL); // Danat: avoid NaN when ye = 0.0
            s2 = (1.5*eos_c2*x5 - third*eos_a2*eos_b2*y3)*plasg_inv;
            dxnidd = avo * ytot1;
            dsdd = 4.0 / 3.0*M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            dpcouldd = -dpiondd*z5 - pion*s2*plasgdd;
            dpcouldt = -dpiondt*z5 - pion*s2*plasgdt;
            decouldd = 3.0*dpcouldd*deni - ecoul*deni;
            decouldt = 3.0*dpcouldt*deni;
        }
        *dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * local_coulombMult; //pressure derivative vs density
        *dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * local_coulombMult; //pressure derivative vs temperature

        //Calculate energy derivatives

        // Nuclear: edited
        deiondd = avo * kt * (1.5 * xxn_r + 1.5 * xxp_r + xxa_r / 4.0 * (1.5 - Qalpha / kt));
        deiondt = 1.5 * avo * ytot1 * kerg + avo * kt * (1.5 * xxn_t + 1.5 * xxp_t + xxa_t / 4.0 * (1.5 - Qalpha / kt));
        // Nuclear: end

        deraddd = -erad*deni;
        deraddt = 4.0 * erad * tempi;
        dsepdt = -df_tt * ye;
        dsepdd = -df_dt * ye * ye;
        deepdt = btemp * dsepdt;
        deepdd = ye*ye*df_d + btemp*dsepdd;
#if (bAprox13t)
        //  Calhoun next two lines
        deradda = 0.0;
        deraddz = 0.0;
#endif
        denerdd = deraddd + deiondd + deepdd + decouldd * local_coulombMult; //energy derivative vs density
        *denerdt = deraddt + deiondt + deepdt + decouldt * local_coulombMult; //energy derivative vs temperature
        
        if (*denerdt != *denerdt || *dpresdt != *dpresdt) {
            printf ("Danat: something is wrong with the derivatives of pressure/energy\n");
            printf ("%e %e %e %e\n", dpraddt, dpiondt, dpepdt, dpcouldt);
            printf ("%e %e %e %e\n", deraddt, deiondt, deepdt, decouldt);
        }

        //Calculate entropy derivatives
        dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt*deni + deraddt - srad)*tempi;

        // Nuclear: edited
        dsiondd = -kergavo * ytot1 * deni + (2.5 * kergavo * ytot1 - sion) * ytot1 * dadd;
        dsiondt = 1.5 * kergavo * ytot1 * tempi + (2.5 * kergavo * ytot1 - sion) * ytot1 * dadt;

        //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        if (plasg >= 1.0) {
            y2 = -kavoy*plasg_inv*(0.75*eos_b1*x4 + 1.25*z4 + d1cc);
            dscouldd = y2 * plasgdd;
            dscouldt = y2 * plasgdt;
        }
        //yakovlev & shalybkov 1989 equations 102, 103, 104
        else if (plasg < 1.0) {
            s3 = -kavoy*plasg_inv*(1.5*eos_c2*x5 - eos_a2*(eos_b2 - 1.0)*y3);
            dscouldd = s3 * plasgdd;
            dscouldt = s3 * plasgdt;
        }
        *dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * local_coulombMult;//entropy derivative vs density and density
        *dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * local_coulombMult;//entropy derivative vs density and time

        // DANAT: calc soundspeeds

        //  form gamma_1
        //presi = 1.0/pres;
        //chit  = btemp*presi * dpresdt;
        //chid  = dpresdd * den*presi;
        //x7     = pres * deni * chit/(btemp * denerdt);
        //gamc  = chit*x7 + chid;
        //cv    = denerdt;
        //cp    = cv*gamc/chid;

        //  store the output -- note that many of these are not used by the calling program!
        // ptotRow(j)   = pres;   //used by Eos as EOS_PRES = PRES_VAR
        // etotRow(j)   = ener;   //used by Eos as EOS_EINT = EINT_VAR
        // stotRow(j)   = entr;   //this is entropy, used by Eos as EOS_ENTR (masked)

        // dpdRow(j)    = dpresdd;  // used as EOS_DPD
        // dptRow(j)    = dpresdt;  // used as EOS_DPT ALWAYS used by MODE_DENS_PRES in Eos.F90
        //
        // dedRow(j)    = denerdd;  // used as EOS_DED
        // detRow(j)    = denerdt;  // used as EOS_DET  ALWAYS used by MODE_DENS_EI in Eos.F90
        //
        // if (bAprox13t) {
        //        dsepda = ytot1 * (ye * df_dt * din - sele); //Calhoun
        //        dsepdz = -ytot1 * (ye * df_dt * den + df_t); //Calhoun
        //        deepda = -ye * ytot1 * (free + df_d * din) + btemp * dsepda; //Calhoun
        //        deepdz = ytot1* (free + ye * df_d * den) + btemp * dsepdz; //Calhoun
        //     denerda = deradda + deionda + deepda + decoulda;  //Calhoun
        //     denerdz = deraddz + deiondz + deepdz + decouldz;  //Calhoun
        //     deaRow(j)    = denerda;  //Calhoun EOS_DEA
        //     dezRow(j)    = denerdz;  //Calhoun EOS_DEZ
        // }
        //
        // dsdRow(j)    = dentrdd;  // used as EOS_DSD
        // dstRow(j)    = dentrdt;  // used as EOS_DST
        //
        // pelRow(j)   = pele;     // used as EOS_PEL
        //
        // neRow(j)    = xnefer;   // used as EOS_NE
        // etaRow(j) = etaele;     // used as EOS_ETA
        //
        // gamcRow(j)   = gamc;    // used as EOS_GAMC = GAMC_VAR
        //
        // cvRow(j)     = cv;      // EOS_CV
        // cpRow(j)     = cp;      // EOS_CP
    }

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;
    *entr *= conv_entr_CGS2CODE;

    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;

    return;
}

void eos_mode_dens_ener_nuclear_nucevol(double ener_goal, double den, double *btemp, double ye, double *xx_atm, double *xxn, double *xxp, double *xxa, double *etaele) {
    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double temp_new, temp_old;
    double ener_tmp, ener_old, entr, pres;
    double dpdt, dedt, dpdrho, dsdt, dsdd;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        if (temp_old != temp_old){
            1+2==3;
        }
        eos_helm_nuclear(1, temp_old, den, ye, xx_atm, xxn, xxp, xxa, etaele, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *btemp = temp_new;
}

void eos_mode_dens_ener_nuclear(double ener_goal, double den, double ye, double xx_atm, double xxn, double xxp, double xxa, double *pres) {
    // Specific energy without the alpha particle contribution
    ener_goal = ener_goal - 6.8e18 / 9e20 * xxa;

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double temp_new, temp_old;
    double ener_tmp, ener_old, entr;
    double dpdt, dedt, dpdrho, dsdt, dsdd;
    double etaele;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm_nuclear(1, temp_old, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, pres, &ener_tmp, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
}

void eos_calc_soundspeed_nuclear (double ener_goal, double den, double ye, double xx_atm, double xxn, double xxp, double xxa, double *pres, double *cs2) {
    // Specific energy without the alpha particle contribution
    ener_goal = ener_goal - 6.8e18 / 9e20 * xxa;

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double deni = 1.0 / den;
    double temp_new, temp_old;
    double ener_tmp, ener_old, entr;
    double dpdt, dedt, dpdrho, dsdt, dsdd;
    double etaele;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm_nuclear(1, temp_old, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, pres, &ener_tmp, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

//    *cs2 = (dpdrho - dpdt * dsdd / dsdt) / (1.0 + ener_goal + (*pres) * deni);
    *cs2 = ((*pres) * temp_old * (deni * deni) * dpdt * (dpdt / dedt) + dpdrho) / (1.0 + ener_goal + (*pres) * deni);
    if (*cs2 < 0.0) {
        printf ("Alert! cs2 < 0.0\n");
    }
}

void eos_mode_dens_enth_nuclear (double den, double ye, double xx_atm, double xxn, double xxp, double xxa, double *pres, double h_goal, double *dpdrho, double *dpdt, double *dedt, double *dpde_d) {

    // Specific energy (enthropy) without the alpha particle contribution
    h_goal = h_goal - 6.8e18 / 9e20 * xxa;

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp, dsdt, dsdd;
    double h_tmp, entr;

    double error, error_h;
    int i;
    double etaele;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xpres = *pres;
    double xener = 0.0;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm_nuclear(1, temp_old, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, pres, &xener, &entr, dpdt, dedt, dpdrho, &dsdt, &dsdd);

        h_tmp = xener + (*pres) * deni;
        dhdtemp = (*dedt) + (*dpdt) * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, xener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *dpde_d = (*dpdt) / (*dedt);
}

void eos_mode_dens_enth_NH_nuclear (double den, double ye, double xx_atm, double xxn, double xxp, double xxa, double *pres, double *ener, double h_goal) {
    // implementation in Newman-Hamlin inversion

    // Specific energy (enthropy) without the alpha particle contribution
    h_goal = h_goal - 6.8e18 / 9e20 * xxa;

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp;
    double h_tmp, entr;
    double dpdrho, dpdt, dedt, dpde_d, dsdt, dsdd;

    double error, error_h;
    int i;
    double etaele;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm_nuclear(1, temp_old, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, pres, ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);

        h_tmp = *ener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, *ener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    // Returning the energy with alpha particle recombination component
    *ener = *ener + 6.8e18 / 9e20 * xxa;
}

void eos_get_min_pres_NH_nuclear (double den, double ye, double xx_atm, double xxn, double xxp, double xxa, double *pres) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = 1.0e4;
    double ener, entr, dpdt, dedt, dpdrho, dsdt, dsdd;
    double etaele;
    eos_helm_nuclear(1, temp, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, pres, &ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
}

void eos_mode_dens_pres_nuclear(double *ener, double den, double ye, double p_goal, double *xx_atm, double *xxn, double *xxp, double *xxa) {
    // Specific energy without the alpha particle contribution
    *ener = *ener - 6.8e18 / 9e20 * (*xxa);

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_p = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (p_goal <= 0.0) temp_ini_guess = 1.0e4;
    temp_ini_guess = pow(p_goal * conv_pres_CODE2CGS * asoli3_inv, 0.25);
    if (temp_ini_guess > 1.0e11) temp_ini_guess = 1.0e10;

    double temp_new, temp_old;
    double p_tmp, entr;

    double error, error_p;
    int i;

    double dpdt, dedt, dpdrho, dsdt, dsdd, etaele;

    temp_old = temp_ini_guess;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    for(i = 0; i < max_iterations; i++){
        eos_helm_nuclear(1, temp_old, den, ye, xx_atm, xxn, xxp, xxa, &etaele, &p_tmp, ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);

        temp_new = temp_old - (p_tmp - p_goal) / dpdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((p_tmp - p_goal) / p_goal);

        //printf("num = %d, T = %e, err in T = %e, err in p = %e, p = %e, dpdt = %e\n", i, temp_new, error, error_p, p_tmp, dpdt);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_p < tolerance_p) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
    
    *ener = *ener + 6.8e18 / 9e20 * (*xxa);
}


// End of DONUCLEAR
#endif

void eos_calc_soundspeed(double ener_goal, double den, double abar, double zbar, double *pres, double *cs2) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);

    double deni = 1.0 / den;
    double temp_new, temp_old;
    double ener_tmp, ener_old, entr;
    double dpdt, dedt, dpdrho, dsdt, dsdd;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm(1, temp_old, den, abar, zbar, pres, &ener_tmp, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

//    *cs2 = (dpdrho - dpdt * dsdd / dsdt) / (1.0 + ener_goal + (*pres) * deni);
    *cs2 = ((*pres) * temp_old * (deni * deni) * dpdt * (dpdt / dedt) + dpdrho) / (1.0 + ener_goal + (*pres) * deni);
}

void eos_mode_dens_ener(double ener_goal, double den, double abar, double zbar, double *pres) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    if (ener_goal <= 0.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);

    double temp_new, temp_old;
    double ener_tmp, ener_old, entr;
    double dpdt, dedt, dpdrho, dsdt, dsdd;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm(1, temp_old, den, abar, zbar, pres, &ener_tmp, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
}

void eos_mode_dens_enth(double den, double abar, double zbar, double *pres, double h_goal, double *dpdrho, double *dpdt, double *dedt, double *dpde_d) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp, dsdt, dsdd;
    double h_tmp, entr;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xpres = *pres;
    double xener = 0.0;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm(1, temp_old, den, abar, zbar, pres, &xener, &entr, dpdt, dedt, dpdrho, &dsdt, &dsdd);

        h_tmp = xener + (*pres) * deni;
        dhdtemp = (*dedt) + (*dpdt) * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, xener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *dpde_d = (*dpdt) / (*dedt);
}

void eos_mode_dens_enth_NH (double den, double abar, double zbar, double *pres, double *ener, double h_goal) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp;
    double h_tmp, entr;
    double dpdrho, dpdt, dedt, dpde_d, dsdt, dsdd;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm(1, temp_old, den, abar, zbar, pres, ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);

        h_tmp = *ener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, *ener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
}

void eos_get_min_pres_NH (double den, double abar, double zbar, double *pres) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = 1.0e4;
    double ener, entr, dpdt, dedt, dpdrho, dsdt, dsdd;
    eos_helm(1, temp, den, abar, zbar, pres, &ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
}

void eos_mode_dens_pres(double *ener, double den, double abar, double zbar, double p_goal) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_p = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (p_goal <= 0.0) temp_ini_guess = 1.0e4;
    temp_ini_guess = pow(p_goal * conv_pres_CODE2CGS * asoli3_inv, 0.25);

    double temp_new, temp_old;
    double p_tmp, entr;

    double error, error_p;
    int i;

    double dpdt, dedt, dpdrho, dsdt, dsdd;

    temp_old = temp_ini_guess;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    for(i = 0; i < max_iterations; i++){

        eos_helm(1, temp_old, den, abar, zbar, &p_tmp, ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);

        temp_new = temp_old - (p_tmp - p_goal) / dpdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((p_tmp - p_goal) / p_goal);

        //printf("num = %d, T = %e, err in T = %e, err in p = %e, p = %e, dpdt = %e\n", i, temp_new, error, error_p, p_tmp, dpdt);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_p < tolerance_p) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
}

void test_eos(void) {
	double temp, den;
	double abar=1.0, zbar=1.0;
	double pres, ener, entr, dedt, dpdt, dpdrho, dpresdener, dsdt, dsdd, etaele, ye, xx_atm, xxn, xxp, xxa;
    double h_goal = 0.0;
    eos_init();
    
#if 0
    ye = 0.9997513515148767;
    xxp = 0.59236411574360348;
    xxn = 0.0000000001;
    xxa = 0.81477447154254645;
    xx_atm = 0.0;
    eos_helm_nuclear(1, btemp, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, &pres, &ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
    exit(1);
#endif
#if 0
    FILE *f_testeos;
    
    f_testeos = fopen("./test_eos.bdat", "wb");
    ye = 1.0;
    xxp = 0.0;
    xxn = 0.0;
    xxa = 0.0;
    xx_atm = 1.0;
    int N = 1000;
    double cs2, cs2_RF;
    
    for (int i = 0; i < N; i++) {
        
        den = pow (10.0, -10.0 + i * (21.0) / (N - 1)) * conv_dens_CGS2CODE;
        for (int j = 0; j < N; j++) {
            
            temp = pow (10.0, 4.0 + j * (7.0) / (N - 1));
            
            eos_helm_nuclear(1, temp, den, ye, &xx_atm, &xxn, &xxp, &xxa, &etaele, &pres, &ener, &entr, &dpdt, &dedt, &dpdrho, &dsdt, &dsdd);
            
            cs2 = (dpdrho - dpdt * dsdd / dsdt) / (1.0 + ener + pres / den);
            cs2_RF = (pres * temp / (den * den) * dpdt * (dpdt / dedt) + dpdrho) / (1.0 + ener + pres / den);
            
            fwrite(&den, 1, sizeof(double), f_testeos);
            fwrite(&temp, 1, sizeof(double), f_testeos);
            fwrite(&cs2, 1, sizeof(double), f_testeos);
            fwrite(&cs2_RF, 1, sizeof(double), f_testeos);
        }
    }
    
    fclose(f_testeos);
    exit(0);
#endif
}

#if DONUCLEAR
//!RF:
//!------------------------------------------------
//!NSE abundance (n,p,alphas)

void nse_abundance (double dens, double temp, double ye, double *xn, double *xp, double *xa) {

    int i, j;
    double x_n, x_p, n_Q, n;
    double Fa, dFa;

    n   = dens / amu;
    n_Q = pow((amu * kerg * temp / (M_PI_2 * hbar_planck * hbar_planck)), 1.5);

    *xa = ye;

    for (i = 0; i < 50; i++) {
        x_n = 1.0 - ye - 0.5 * (*xa);
        x_p = ye - 0.5 * (*xa);
        Fa  = x_n * x_n * x_p * x_p - 0.5 * (*xa) * pow(n_Q / n, 3.0) * exp (-Qa / kerg / temp);
        dFa = -x_p * x_n * x_n - x_n * x_p * x_p - 0.5 * pow(n_Q / n, 3.0) * exp (-Qa / kerg / temp);
        *xa  = *xa - Fa / dFa;

        if (fabs (Fa / dFa) < 1e-8 * (*xa)) break;
    }

    *xn = 1.0 - ye - 0.5 * (*xa);
    *xp = ye - 0.5 * (*xa);

    return;
}


//!---------------------------------------------------------------------------------
//!Derivative of abundances in NSE. Not affected by constant atmosphere (as written)

void nse_derivatives (double dens, double temp, double xn, double xp, double xa, double *xa_r, double *xa_t, double *xa_y, double *xn_r, double *xn_t, double *xn_y, double *xp_r, double *xp_t, double *xp_y) {

    *xa_r = (3.0 / dens) * xn * xp * xa / (xn * xp + xn * xa + xp * xa);
    *xa_t = -1.0 / temp * (4.5 + Qa / kerg / temp) * xn * xp * xa / (xn * xp + xn * xa + xp * xa);
    *xa_y = 2.0 * (xn - xp) * xa / (xn * xp + xn * xa + xp * xa);

    *xn_r = -0.5 * (*xa_r);
    *xn_t = -0.5 * (*xa_t);
    *xn_y = -1.0 - 0.5 * (*xa_y);

    *xp_r = *xn_r;
    *xp_t = *xn_t;
    *xp_y = 1.0 - 0.5 * (*xa_y);

    return;
}
// End fo DONUCLEAR
#endif

#endif