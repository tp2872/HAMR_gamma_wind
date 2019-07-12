#include "decs_MPI.h"

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

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

void eos_copy_gpu(void) {


}


void eos_helm(int calc_derivatives, double btemp, double den, double abar, double zbar, double *pres, double *ener, double *entr, double *denerdt)
{
    // Local variables
    double ytot1, ye, local_coulombMult;
	double  x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, deni, tempi, kt, prad, dpraddd, dpraddt, erad, deraddd, deraddt, srad, dsraddd, dsraddt;
	double  xni, pion, dpiondd, dpiondt, eion, deiondd, deiondt;
	double sion, dsiondd, dsiondt;
	double pele, dpepdd, dpepdt, eele, deepdd, deepdt;
	double sele, dsepdd, dsepdt;
	double dpresdd, dpresdt;
	double presi, chit, chid, gamc, kavoy;
    double cv, cp, etaele, xnefer,denerdd, dentrdd ,dentrdt;

    // For the interpolations
    double free,df_d,df_t,df_dd,df_tt,df_dt;
    //double h3e, h3x;
	double zFunc, z0, z1, z2, z3, z4, z5, z6, din;

    // For the coulomb corrections
    double  ktinv,dxnidd,dsdd,lami,inv_lami,lamidd,s0,s1,s2,s3,s4,plasg,plasg_inv,plasgdd,plasgdt,ecoul,decouldd,decouldt,pcoul,dpcouldd,dpcouldt,scoul,dscouldd,dscouldt;

    // Added by Calhoun for calculations for the Aprox13t network
    double  deradda,dxnida,dpionda,deionda,dsepda,deepda,decoulda,dsda,dsdda,lamida,plasgda,denerda,deraddz,deiondz,deepdz,decouldz,dsepdz,plasgdz,denerdz;

	kt = kerg * btemp;
	ktinv = 1.0 / kt;

	#if (bAprox13t)
	dsda = 4.0 / 3.0*M_PI * dxnida;          //Calhoun
	lamida   = z2 * dsda / s1;      //Calhoun
	plasgda  = z3 * lamida;         //Calhoun
	plasgdz  = 2.0 * plasg/zbar;   //Calhoun
	#endif

	//frequent combinations
	din = zbar / abar*den;
	ytot1 = 1.0 / abar;
	ye = ytot1 * zbar;
	
	//Look up the desired quantities in the eos table
	interp_eostable(den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd);

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
		dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * local_coulombMult; //pressure derivative vs density and density
		dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * local_coulombMult; //pressure derivative vs density and time

		//Calculate energy derivatives
		deiondd = (1.5 * dpiondd - eion)*deni;
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
		denerdd = deraddd + deiondd + deepdd + decouldd * local_coulombMult; //energy derivative vs density and density
		deiondt = 0.0;
		*denerdt = deraddt + deiondt + deepdt + decouldt * local_coulombMult; //energy derivative vs density and time

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
		dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * local_coulombMult;//entropy derivative vs density and density
		dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * local_coulombMult;//entropy derivative vs density and time

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
		//		dsepda = ytot1 * (ye * df_dt * din - sele); //Calhoun
		//		dsepdz = -ytot1 * (ye * df_dt * den + df_t); //Calhoun
		//		deepda = -ye * ytot1 * (free + df_d * din) + btemp * dsepda; //Calhoun
		//		deepdz = ytot1* (free + ye * df_d * den) + btemp * dsepdz; //Calhoun
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
    return;
}

void test_eos(void) {
	double btemp=2.0e8, den=1.0e7;
	double abar=1.0, zbar=1.0;
	double pres, ener, entr, denerdtemp;

    
    // START decs for output table: just for a check:
//    int ix, jx;
//    int rho_powmin = -10;
//    int rho_powmax = 11;
//    int temp_powmin = 4;
//    int temp_powmax = 11;
//    FILE *fp_checking_eos;
//    fp_checking_eos = fopen("./test.txt", "w+");
    // END of decs

    // Reading the table and writing into arrays
    eos_init();  
	eos_helm(1, btemp,den,abar, zbar, &pres,  &ener, &entr, &denerdtemp);
    
    printf("d=%21.15e, T=%21.15e, Pressure = %21.15e, Energy = %21.15e,  Entr = %21.15e\n", den, btemp, pres, ener, entr);
    exit(1);

/*
    // START output table
    // Output a table with different rho,T --> P,u,s
    // Physical range per FLASH manual: rho = (1e-10, 1e11) [g/cm3]; T = (1e4, 1e11) [K]
	fprintf(fp_checking_eos, "# Density, Temperature, Pressure, Energy, Entropy \n");
	int Num_max = 1000;
	for (ix = 0; ix < Num_max; ix++){
	   den = pow(10.0, rho_powmin + ix * (rho_powmax - rho_powmin) / (float) Num_max);
	   for (jx = 0; jx < Num_max; jx++){
	       btemp = pow(10.0, temp_powmin + jx * (temp_powmax - temp_powmin) / (float) Num_max);
	       eos_helm(1, btemp,den,abar, zbar, &pres,  &ener, &entr);
	       fprintf(fp_checking_eos, "%e %e %e %e %e\n", den, btemp, pres, ener, entr);
	   }
	}
	fclose(fp_checking_eos);
    // END output table
*/
/* 
    // START eos mode dens+ener instead of dens+temp
    int max_iterations = 50;
    int iter_num = 0;
    double tolerance = 1.0e-5;
    double ener_goal = ener;
    // initial guess : temperature
    double temp_ini_guess = 1.1e8;//(gam - 1.0) * ener_goal * 1.211475197e-8;
    double temp_new, temp_old;
    double ener_old;
    double error;
    int i;
    
    for(i = 0; i < max_iterations; i++){
        temp_old = temp_ini_guess;
        eos_helm(1, temp_old, den, abar, zbar, &pres,  &ener_old, &entr, &denerdtemp);
        temp_new = temp_old - (ener_old - ener_goal) / denerdtemp;
        
        //do not allow temp to change more than 10. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;
        
        error = fabs((temp_new - temp_old) / temp_old);
        iter_num++;
        if(error < tolerance) break;
    }
    printf("Error = %e; ener = %e, dens = %e, temp = %e, iter = %d", error, ener_goal, den, temp_new, iter_num);
*/
    
}


void interp_eostable(double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd){
	int iat, jat;
	double fi[36];
	double xt, xd, mxt, mxd;
	double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
	double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;

	//  hash locate this temperature and density
	jat = (int)((log10(btemp) - eos_tlo)*(double)(EOSJMAX - 1) / (11.0 - eos_tlo)) + 1;
	jat = max(1, min(jat, EOSJMAX - 1)) - 1;
	iat = (int)((log10(din) - eos_dlo)*(double)(EOSIMAX - 1) / (11.0 - eos_dlo)) + 1;
	iat = max(1, min(iat, EOSIMAX - 1)) - 1;

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
	xt = max((btemp - eos_t[jat]) / eos_dt[jat], 0.0);
	xd = max((din - eos_d[iat]) / eos_dd[iat], 0.0);
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

	*dpepdd = max(ye * (*dpepdd), 0.0);

	//  electron chemical potential etaele
	/*etaele = eos_ef[(iat)*EOSJMAX + jat] * si0d*si0t + eos_ef[(iat + 1)*EOSJMAX + jat] * si0md*si0t
	+ eos_ef[(iat)*EOSJMAX + jat + 1] * si0d*si0mt + eos_ef[(iat + 1)*EOSJMAX + jat + 1] * si0md*si0mt
	+ eos_eft[(iat)*EOSJMAX + jat] * si0d*si1t + eos_eft[(iat + 1)*EOSJMAX + jat] * si0md*si1t
	+ eos_eft[(iat)*EOSJMAX + jat + 1] * si0d*si1mt + eos_eft[(iat + 1)*EOSJMAX + jat + 1] * si0md*si1mt
	+ eos_efd[(iat)*EOSJMAX + jat] * si1d*si0t + eos_efd[(iat + 1)*EOSJMAX + jat] * si1md*si0t
	+ eos_efd[(iat)*EOSJMAX + jat + 1] * si1d*si0mt + eos_efd[(iat + 1)*EOSJMAX + jat + 1] * si1md*si0mt
	+ eos_efdt[(iat)*EOSJMAX + jat] * si1d*si1t + eos_efdt[(iat + 1)*EOSJMAX + jat] * si1md*si1t
	+ eos_efdt[(iat)*EOSJMAX + jat + 1] * si1d*si1mt + eos_efdt[(iat + 1)*EOSJMAX + jat + 1] * si1md*si1mt;*/

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
