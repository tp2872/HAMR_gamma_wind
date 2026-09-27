#include "include.h"
#include "decs_MPI.h"

void get_rho_u_floor(double r, double th, double phi, double *rho_floor, double *u_floor);

#define FLOOP for(k=0;k<B1;k++)

/* apply floors to density, internal energy */

void fixup(double((* restrict pv[NB_LOCAL])[NPR]), int n)
{
	#if(!CARTESIAN)
	int i, j, z;
	#pragma omp parallel shared(n,pv, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z)
	{
		#pragma omp for collapse(3) schedule(static,BS_1*BS_2*BS_3/nthreads)
		for (i = N1_GPU_offset[n]; i<N1_GPU_offset[n] + BS_1; i++)for (j = N2_GPU_offset[n]; j<N2_GPU_offset[n] + BS_2; j++)for (z = N3_GPU_offset[n]; z<N3_GPU_offset[n] + BS_3; z++){	
			//printf("floors: %d %d %d %e %e\n", i, j, z, pv[nl[n]][index_3D(n, i, j, z)][RHO], pv[nl[n]][index_3D(n, i, j, z)][UU]);
			fixup1zone(i, j, z, n, pv[nl[n]][index_3D(n, i, j, z)]);
		}
	}
	#endif
}

void fixup1zone( int i, int j, int z, int n, double pv[NPR] ) 
{
	double r,th, phi, X[NDIM],uuscal,rhoscal, rhoflr, uuflr;
        double f,gamma, bsq;
        double pv_prefloor[NPR], dpv[NPR], U_prefloor[NPR], dU[NPR], U[NPR], U_ent;
        double trans, betapar, betasq, betasqmax, one_over_ucondr_, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut, u;
        double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];
        int m;
        int k, flag, dofloor=0;
        #if(RESISTIVE)
        struct of_state_res q;
        #else
        struct of_state q;
        #endif
        #if(TWO_T)
        double ue, ui, Theta, gam, C, dis;
        #endif
        #if (NEUTRON_STAR)
        double Rlc, rho_b, rho_g, smooth, smooth_geom;
        #endif
        struct of_geom geom;



        coord(n, i,j, z, CENT,X);
        bl_coord(X,&r,&th, &phi);

	if (r<radfix_dens) {
		pv[RHO] = 1.0;
		pv[UU] = 0.1;
	}

#if 0
	double r,th, phi, X[NDIM],uuscal,rhoscal, rhoflr, uuflr;
	double f,gamma, bsq;
	double pv_prefloor[NPR], dpv[NPR], U_prefloor[NPR], dU[NPR], U[NPR], U_ent;
	double trans, betapar, betasq, betasqmax, one_over_ucondr_, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut, u;
	double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];
	int m;
	int k, flag, dofloor=0;
	#if(RESISTIVE)
	struct of_state_res q;
	#else
	struct of_state q;
	#endif
	#if(TWO_T)
	double ue, ui, Theta, gam, C, dis;
	#endif
	#if (NEUTRON_STAR)
	double Rlc, rho_b, rho_g, smooth, smooth_geom;
	#endif
	struct of_geom geom;

	

	coord(n, i,j, z, CENT,X) ;
	bl_coord(X,&r,&th, &phi) ;
	#if(CARTESIAN_GR)
	r = MY_MAX(r, 1.0);
	#endif

	// Danat addition: 11/18/19 - avoid rhoflr too large`
	get_rho_u_floor (r, th, phi, &rhoflr, &uuflr); 

	rhoscal = pow(MY_MAX(r, 1.0), -POWRHO);
	uuscal = pow(rhoscal, GAMMA);

	rhoflr = RHOMINLIMIT; //1.e-5 * RHOMIN * rhoscal;
	uuflr = UUMINLIMIT; // 1.e-5 * UUMIN * uuscal;
    
	//compute the square of fluid frame magnetic field (twice magnetic pressure)
	get_geometry(n,i,j,z,CENT,&geom) ;
	#if(RESISTIVE)
	bsq = bsq_calc_res(pv, &geom);
	#else
	bsq = bsq_calc(pv, &geom);
	#endif


	#if (DOHELM)
	double xP;
	#if (DOHELM_TEMPERATURE == 2)
	// Making sure that temperature is not below the threshold of the table
	if (pv[UU] < eos_temp_low) pv[UU] = eos_temp_low;
	eos_mode_rhotemp_pres_u(pv[RHO], pv[UU], &xP, &u);
	double prefloor_u = u;
	#else 
	u = pv[UU];
	#endif
	#else
	u = pv[UU];
	#endif

	//floor on density and internal energy density (momentum *not* conserved) 
	//printf("floors: %e %e %e\n", pv[RHO], pv[UU], bsq);
	for (k = 0; k < (NPR_U + NEUTRON_STAR * (1 + DOFLR + USE_PS1START)); k++) pv_prefloor[k] = pv[k];

#if (NEUTRON_STAR)
#if(NS_TAPERED_FLOORS)
	if (OMEGA_NS > 0.0)
		Rlc = 1.0 / OMEGA_NS;
	else
		Rlc = 50.*R_NS;

	double rho0 = RHO0_HYDROSTAT_ATM_NS * pow(MU_NS / 10.0, 2.0);
	double alpha1_NS, alpha2_NS, n_NS, rb_NS, constant_NS;
	rb_NS = 1.0 * Rlc; // break radius for smooth broken power law
	n_NS = 2.0;       // smaller n = smoother break
	alpha2_NS = 6.0;    // At large r, want rho, u to be propto r^(-alpha2)
					  // For hydrostatic atmosphere to large radii, set alpha2 = alpha1
					  // in each case below.

	/* Density */
	alpha1_NS = 1.0 / (GAMMA - 1.0);
	constant_NS = rho0 * pow(R_NS, alpha1_NS);
	/* Function for smooth broken power law */
	rhoflr = constant_NS * pow(rb_NS, -alpha1_NS) * pow(pow(r / rb_NS, alpha1_NS * n_NS) + pow(r / rb_NS, alpha2_NS * n_NS), -1.0 / n_NS);

	/* Internal energy */
	alpha1_NS = GAMMA / (GAMMA - 1.0);
	constant_NS = (rho0 / GAMMA) * (1.0 / R_NS) * pow(R_NS, alpha1_NS);
	uuflr = constant_NS * pow(rb_NS, -alpha1_NS) * pow(pow(r / rb_NS, alpha1_NS * n_NS) + pow(r / rb_NS, alpha2_NS * n_NS), -1.0 / n_NS);
	double mod_bsq_over_rho_max, mod_bsq_over_uu_max;
	double log_mod_bsq_rho, log_mod_bsq_uu, log_bsq_rho, log_bsq_uu, log_profile;
	/*** Dynamic floors ***/
	/* Use dynamis floor everywhere, but increase acceptable bsq/rho and bsq/u inside the light-sphere,
	 * so that the hydrostatic region doesn't trigger this unless bsq increases enormously              */
	 //double bsq_over_rho_surf = 1e6;
	 //double bsq_over_uu_surf  = 1e7;
	if (r < Rlc)
	{
		log_bsq_rho = log10(MAX_BSQ_OVER_RHO);
		log_bsq_uu = log10(MAX_BSQ_OVER_UINT);
		log_profile = pow((Rlc - r) / (Rlc - R_NS), 2.0);
		log_mod_bsq_rho = log_bsq_rho + (SURF_MAX_BSQ_RHO_LOG - log_bsq_rho) * log_profile;
		log_mod_bsq_uu = log_bsq_uu + (SURF_MAX_BSQ_UINT_LOG - log_bsq_uu) * log_profile;
		mod_bsq_over_rho_max = pow(10.0, log_mod_bsq_rho);
		mod_bsq_over_uu_max = pow(10.0, log_mod_bsq_uu);
	}
	else
	{
		mod_bsq_over_rho_max = MAX_BSQ_OVER_RHO;
		mod_bsq_over_uu_max = MAX_BSQ_OVER_UINT;
	}
	/* tie floors to the local values of magnetic field and internal energy density */
	if (rhoflr < bsq / mod_bsq_over_rho_max)
		rhoflr = bsq / mod_bsq_over_rho_max;

	if (uuflr < bsq / mod_bsq_over_uu_max)
		uuflr = bsq / mod_bsq_over_uu_max;

#else
	if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
	if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;
#endif
#else
	//tie floors to the local values of magnetic field and internal energy density
	if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);
	#if(RAD_M1)
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	if (rhoflr < (u + pv[UU_RAD]) / UORHOMAX)  rhoflr = (u + pv[UU_RAD]) / (UORHOMAX);
	#elif(NEUTRINOS_M1)
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	#if (NU_SPECIES > 1)
	if (rhoflr < (u + pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)]) / UORHOMAX)  rhoflr = (u + pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)]) / (UORHOMAX);
	#else
	if (rhoflr < (u + pv[UU_NU]) / UORHOMAX)  rhoflr = (u + pv[UU_NU]) / (UORHOMAX); // DINU: 3 species
	#endif
	#else
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	if (rhoflr < u / UORHOMAX) rhoflr = u / (UORHOMAX);
	#endif

	//printf("floors: %e %e\n", rhoflr, uuflr);
	if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
	if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;
	//printf("2 floors: %e %e\n", rhoflr, uuflr);
#endif



#if(NEUTRON_STAR)
	if (OMEGA_NS > 0.0)
		Rlc = 1.0 / OMEGA_NS;
	else
		Rlc = 50. * R_NS;

	if (pv[FLRFRAC] < 0.0)
		pv[FLRFRAC] = 0.0;
	if (pv[FLRFRAC] > 1.0)
		pv[FLRFRAC] = 1.0;

	rho_b = pv[RHO] * pv[FLRFRAC];
	rho_g = pv[RHO] - rho_b;

	if (pv[RHO] < 1.0001 * rhoflr)
		pv[FLRFRAC] = 1.0;
	else
		pv[FLRFRAC] = 0.0;

	if (r > Rlc)
		smooth_geom = 1.0;
	else if (r < R_NS)
		smooth_geom = 0.0;
	else
		smooth_geom = pow((1.0) * 0.5 * (1.0 - cos(M_PI * (r - R_NS) / (Rlc - R_NS))), 2.0);
	smooth_geom = pow(smooth_geom, 0.5); //fixupWeight=0.5 always
	/* This variable is 1 beyond Rlc, or if FLRFRAC = 0 */
	smooth = 1.0 - pv[FLRFRAC] * (1.0 - smooth_geom);  // Don't want to do anything to real gas
#endif



	#if(TWO_T)
	pv_prefloor[ENTRE] = pv[ENTRE];
	pv_prefloor[ENTRI] = pv[ENTRI];
	#endif

	if (pv[RHO] < rhoflr) {
#if(NEUTRON_STAR)
		rho_b = rhoflr - rho_g;
#endif
		pv[RHO] = rhoflr;
		dofloor = 1;
	}
#if(NEUTRON_STAR)
	else if (rho_b > rhoflr && r < Rlc)
	{
		pv[RHO] = rhoflr + smooth_geom * (rho_b - rhoflr) + rho_g;
	}
	if (r < R_NS) {
		pv[RHO] = bsq * pow(10.0, -SURF_MAX_BSQ_RHO_LOG);
	}
#endif
	
	//Internal energy floor

#if(NEUTRON_STAR)
	/*** Cool rapidly in high-floor zones inside the LC ***/
	if (pv[UU] > uuflr && r < Rlc)
		pv[UU] = uuflr + smooth * (pv[UU] - uuflr); // Full smoothing fn: don't cool good gas

	//printf("floors: %e %e %e %e %e %e\n", rhoflr, pv[RHO], pv_prefloor[RHO], uuflr, pv[UU], pv_prefloor[UU]);
#endif
#if(!NS_TAPERED_FLOORS)
	pv[RHO] = bsq / FREEZE_BSQORHO;
	pv[UU] = 0.2 * pv[RHO];
#endif
	#if(RAD_M1)
	if (u + pv[UU_RAD] < uuflr) {
		u = uuflr - pv[UU_RAD];
		dofloor = 1;
	}
	if (pv[UU] < 0.0001 * uuflr) {
		pv[UU] = 0.0001 * uuflr;
		dofloor = 1;
	}
	#elif(NEUTRINOS_M1)
		#if (NU_SPECIES > 1)
		if (u + pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)] < uuflr) {
			u = uuflr - (pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)]);
			#if (!(DOHELM_TEMPERATURE == 2))
			pv[UU] = u;
			#endif
			dofloor = 1;
		}
		#else
		if (u + pv[UU_NU] < uuflr) {
			u = uuflr - pv[UU_NU];
			#if (!(DOHELM_TEMPERATURE == 2))
			pv[UU] = u;
			#endif
			dofloor = 1;
		}
		#endif
	#else
	if (u < uuflr) {
		u = uuflr;
		#if (!(DOHELM_TEMPERATURE == 2))
		pv[UU] = u;
		#endif
		dofloor = 1;
	}
	#endif
	//printf("3 floors: %e %e\n", rhoflr, uuflr);
#if(NEUTRON_STAR)
	if (r < R_NS) {
		pv[UU] = bsq * pow(10.0, -SURF_MAX_BSQ_UINT_LOG);
	}
#endif
	// Floor on Ye
	#if (DO_YE)
	pv[YE] = MY_MAX(nulib_ylo, pv[YE]);
	pv[YE] = MY_MIN(1.0, pv[YE]);
	#endif
	
	#if (DONUCLEAR)
	pv[XALPHA] = MY_MAX(1e-10, pv[XALPHA]);
	pv[XALPHA] = MY_MIN(1.0, pv[XALPHA]);
	
	pv[XATM] = MY_MAX(1e-10, pv[XATM]);
	pv[XATM] = MY_MIN(1.0, pv[XATM]);
	#endif

	//Floor on radiation internal energy
	#if(RAD_M1)
	if (pv[UU_RAD] < pow(10., -30.)) {
		pv[UU_RAD] = pow(10., -30.);

		//Floor on photon number+
		#if(P_NUM)
		double Tr;
		Tr = pow(pv[UU_RAD] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		pv[PHOTON] = pv[UU_RAD] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif
	}
	#endif
	
	#if(NEUTRINOS_M1)
	double Tnu;
	for (int sp = 0; sp < NU_SPECIES; sp++)
	{
		if (pv[index_nu(UU_NU, sp)] < pow(10., -30.)) {
			pv[index_nu(UU_NU, sp)] = pow(10., -30.);

			//Floor on photon number+
			//pv[NUMBER_NU] = 1e-30;
			Tnu = pow(pv[index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
			pv[index_nu(NUMBER_NU, sp)] = pv[index_nu(UU_NU, sp)] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tnu);
		}
	}
	#endif

	//Divide internal energy inject between electrons and ions 1:1
	#if(TWO_T)
	if(dofloor) {	
		#if(CONSTANTGAMMA || FIXEDGAMMA)
		//Calculate electron entropy
		ue = pv[ENTRE] * pow(pv[RHO], GAMMAE) / (GAMMAE - 1.0);

		//Calculate ion entropy
		ui = pv[ENTRI] * pow(pv[RHO], GAMMA) / (GAMMA - 1.0);
		
		//Calculate total dissipation
		dis = pv[UU] - (ue + ui);
		ue += 0.5 * dis;

		//Check limits
		if (ue > (1.0 - FLOOR_ENTROPY) * pv[UU]) {
			ue = (1.0 - FLOOR_ENTROPY) * pv[UU];
		}
		if (ue < FLOOR_ENTROPY * pv[UU]) {
			ue = FLOOR_ENTROPY * pv[UU];
		}
		ui = pv[UU] - ue;

			//Calculate electron and ion entropies
			#if(FULL_ENTROPY)
			pv[ENTRE] = 1. / (GAMMAE - 1.) * log(0.5 * (GAMMAE - 1.0) * pv[UU] * pow(pv[RHO], -GAMMAE));
			pv[ENTRI] = 1. / (GAMMA - 1.) * log(0.5 * (GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA));
			#else
			pv[ENTRE] = 0.5 * (GAMMAE - 1.0) * pv[UU] * pow(pv[RHO], -GAMMAE);
			pv[ENTRI] = 0.5 * (GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA);
			#endif
		#elif(VARGAMMA)
		//Calculate ue
		#if(FULL_ENTROPY_VARGAMMA)
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pv[RHO] * exp(pv[ENTRE])), 2. / 3.)) - 1.0));
		#else
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pv[RHO], 2. / 3.) * fabs(pv[ENTRE])) - 1.0));
		#endif
		gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		ue = Theta / (MU_E * MASS_RATIO) * pv[RHO] / (gam - 1.0);

		//Calculate ui
		#if(FULL_ENTROPY_VARGAMMA)
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pv[RHO] * exp(pv[ENTRI])), 2. / 3.)) - 1.0));
		#else
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pv[RHO], 2. / 3.) * fabs(pv[ENTRI])) - 1.0));
		#endif
		gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		ui = Theta / (MU_I) * pv[RHO] / (gam - 1.0);

		//Calculate total dissipation
		dis = pv[UU] - (ue + ui);
		ue += 0.5 * dis;
		ue = 0.5 * pv[UU];

		//Check limits
		if (ue > (1.0 - FLOOR_ENTROPY) * pv[UU]) {
			ue = (1.0 - FLOOR_ENTROPY) * pv[UU];
		}
		if (ue < FLOOR_ENTROPY * pv[UU]) {
			ue = FLOOR_ENTROPY * pv[UU];
		}
		ui = pv[UU] - ue;

		//Calculate electron entropy
		C = ue / pv[RHO] * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pv[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pv[RHO]);
			#else
			pv[ENTRE] = Theta * (Theta + 0.4) / pow(pv[RHO], 2. / 3.);
			#endif

		//Calculate ion entropy
		C = ui / pv[RHO] * MU_I;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pv[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pv[RHO]);
			#else
			pv[ENTRI] = Theta * (Theta + 0.4) / pow(pv[RHO], 2. / 3.);
			#endif
		#endif
	}
	#endif

	#if(NEUTRON_STAR)
	//kyles_unified_4D_velocityAdjust(pv_prefloor,  pv, &geom, smooth)
	double etacon[NDIM], bhatcon[NDIM], bhatcov[NDIM];
	double uprllcon[NDIM], uperpcon[NDIM], uprllcov[NDIM], uperpcov[NDIM];
	double uDotEta, bDotEta, bhatsq, uDotBhat, uprllsq, uperpsq;
	double rhopre, uintpre, rhopost, uintpost;
	double Kprll, Kcal, K2, uprllDotBhat;
	double XNS, YNS, ZNS, aminusNS, aplusNS, aNS, normalizerNS;
	double u1con[NDIM], beta[NDIM];
	get_state(pv_prefloor, &geom, &q);
	if (pv_prefloor[RHO] < pv[RHO] || pv_prefloor[UU] < pv[UU] || smooth < 1.0) {
		/* 4-velocity of observer which is static wrt the coordinates */
		etacon[0] = sqrt(-1.0 / geom.gcov[0][0]);
		SLOOPA etacon[j] = 0.0;

		uDotEta = dot(q.ucov, etacon);
		bDotEta = dot(q.bcov, etacon);

		/* Construct B-field 4-vector in coordinate-static frame */
		DLOOPA bhatcon[j] = bDotEta * q.ucon[j] - uDotEta * q.bcon[j];
		lower(bhatcon, &geom, bhatcov);
		bhatsq = dot(bhatcov, bhatcon);

		/* Construct parallel and perpendicular (unnormalized) 4-velocities by projection */
		uDotBhat = dot(q.ucov, bhatcon);
		DLOOPA
		{
			uprllcon[j] = (uDotBhat / bhatsq) * bhatcon[j];
			uperpcon[j] = q.ucon[j] - uprllcon[j]; // identical to using projection tensor
		}
		lower(uprllcon, &geom, uprllcov);
		lower(uperpcon, &geom, uperpcov);
		uprllsq = dot(uprllcov, uprllcon);
		uperpsq = dot(uperpcov, uperpcon);

		rhopre = pv_prefloor[RHO];
		uintpre = pv_prefloor[UU];

		/***** Modification due to flooring ******/
		if (pv[RHO] >= rhopre || pv[UU] >= uintpre)
		{
			/* Find final enthalpy */
			if (pv[RHO] > rhopre)
				rhopost = pv[RHO];
			else
				rhopost = rhopre;

			if (pv[UU] > uintpre)
				uintpost = pv[UU];
			else
				uintpost = uintpre;

			// Conserved mom'm along coord-static frame magnetic field
			Kprll = (rhopre + uintpre + (GAMMA - 1.0) * uintpre) * q.ucon[0] * dot(bhatcon, q.ucov) + (GAMMA - 1.0) * uintpre * bhatcon[0];

			// Calligraphic K from notes: subtract off pressure term with *post-floor* value
			Kcal = Kprll - (GAMMA - 1.0) * uintpost * bhatcon[0];

			uprllDotBhat = dot(uprllcov, bhatcon);

			// Final form: K_2
			K2 = Kcal / ((rhopost + uintpost + (GAMMA - 1.0) * uintpost) * uprllDotBhat);

			/* X a^2 + Y a + Z = 0 */
			XNS = uprllcon[0] + K2 * uprllsq;
			YNS = uperpcon[0];
			ZNS = K2 * uperpsq;

			aminusNS = (-YNS - sqrt(YNS * YNS - 4 * XNS * ZNS)) / (2 * XNS);
			aplusNS = (-YNS + sqrt(YNS * YNS - 4 * XNS * ZNS)) / (2 * XNS);

			// Want a to be positive
			if ((aminusNS >= 0.0 && aminusNS <= 1.0) && (aplusNS < 0 || aplusNS > 1))
				aNS = aminusNS;
			else if ((aplusNS >= 0.0 && aplusNS <= 1.0) && (aminusNS < 0 || aminusNS > 1))
				aNS = aplusNS;
			else            // Both or neither values in range --- always seems to be neither
				aNS = 1.0;   // Seems to happen when the "true" solution is just over 1 due to numerical errors
				//fprintf(stderr, "BOTH or NEITHER A VALUES IN RANGE: %e   %e\n", aminus, aplus);
		}
		else
			aNS = 1.0;

		/***** Modification due to reduce parallel velocity inside the light-sphere *****/
		if (smooth < 1.0)
			aNS *= smooth;

		/***** Construct final 4-velocity *****/
		normalizerNS = sqrt(-(aNS * aNS * uprllsq + uperpsq));
		DLOOPA ucon[j] = (aNS * uprllcon[j] + uperpcon[j]) / normalizerNS;

		SLOOPA beta[j] = - geom.gcon[0][j]/ geom.gcon[0][0];
		SLOOPA	pv[j + U1 -1] = ucon[j] + beta[j] * ucon[0];

	}
	#elif(DRIFT_FLOOR)
	if (dofloor && (trans = 10.*bsq / MY_MIN(pv[RHO], u) - 1.) > 0.) {
		#if(RESISTIVE)
		get_state_res(pv_prefloor, &geom, &q);
		#else
		get_state(pv_prefloor, &geom, &q);
		#endif
		if (trans > 1.) {
			trans = 1.;
		}

		//set velocity to drift velocity
		betapar = -q.bcon[0] / ((bsq + SMALL)*q.ucon[0]);
		betasq = betapar*betapar*bsq;
		betasqmax = 1. - 1. / (GAMMAMAX*GAMMAMAX);
		if (betasq > betasqmax) {
			betasq = betasqmax;
		}
		gamma = 1. / sqrt(1 - betasq);
		for (m = 0; m < NDIM; m++) {
			ucondr[m] = gamma*(q.ucon[m] + betapar*q.bcon[m]);
		}

		Bcon[0] = 0.;
		#if(CALC_MDOT)
		for (m = 1; m < NDIM; m++) Bcon[m] = magnetic_density_scale_cpu*pv[B1 - 1 + m];
		#else
		for (m = 1; m < NDIM; m++) Bcon[m] = pv[B1 - 1 + m];
		#endif

		lower(Bcon, &geom, Bcov);
		udotB = dot(q.ucon, Bcov);
		Bsq = dot(Bcon, Bcov);
		B = sqrt(Bsq);

		//enthalpy before the floors
		#if (DOHELM)
			#if (DOHELM_TEMPERATURE == 2)
			wold = pv_prefloor[RHO] + prefloor_u + xP;
			#else 
			eos_mode_rhou_pres(pv_prefloor, &xP);
			wold = pv_prefloor[RHO] + pv_prefloor[UU] + xP;
			#endif
		#elif(TWO_T)
		double gamma_g;
		gamma_g = calc_gamma_gas_prim(pv_prefloor);
		wold = pv_prefloor[RHO] + pv_prefloor[UU] * gamma_g;
		#else
		wold = pv_prefloor[RHO] + pv_prefloor[UU] * GAMMA;
		#endif
		//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
		QdotB = udotB * wold * q.ucon[0];

		//enthalpy after the floors
		#if (DOHELM)
			#if (DOHELM_TEMPERATURE == 2)
			eos_mode_rhotemp_u_pres_floor(pv[RHO], &pv[UU], u, &xP);
			wnew = pv[RHO] + u + xP;
			#else 
			eos_mode_rhou_pres(pv, &xP);
			wnew = pv[RHO] + u + xP;
			#endif
		#elif(TWO_T)
		gamma_g = calc_gamma_gas_prim(pv);
		wnew = pv[RHO] + pv[UU] * gamma_g;
		#else
		wnew = pv[RHO] + pv[UU] * GAMMA;
		#endif

		x = 2.*QdotB / (B*wnew*ucondr[0] + SMALL);

		//new parallel velocity
		vpar = x / (ucondr[0] * (1. + sqrt(1. + x*x)));

		one_over_ucondr_t = 1. / ucondr[0];

		//new contravariant 3-velocity, v^i
		vcon[0] = 1.;
		for (m = 1; m < NDIM; m++) {
			//parallel (to B) plus perpendicular (to B) velocities
			vcon[m] = vpar*Bcon[m] / (B + SMALL) + ucondr[m] * one_over_ucondr_t;
		}

		//compute u^t corresponding to the new v^i
		ut_calc_3vel(vcon, &geom, &ut);

		for (m = 0; m < NDIM; m++) {
			ucon[m] = ut*vcon[m];
		}
		ucon_to_utcon(ucon, &geom, utcon);

		//now convert 3-vel to relative 4-velocity and put it into pv[U1..U3]
		//\tilde u^i = u^t(v^i-g^{ti}/g^{tt})
		for (m = 1; m < NDIM; m++) {
			pv[m + UU] = utcon[m] * trans + pv_prefloor[m + UU] * (1. - trans);
		}
	}
	#endif

#if(NEUTRON_STAR)
	/*** Reset floor fraction following either flooring or draining ***/
	pv[FLRFRAC] = rho_b / (rho_b + rho_g);

	/*** Enforce some sanity ***/
	if (pv[FLRFRAC] < 0.0)
		pv[FLRFRAC] = 0.0;
	else if (pv[FLRFRAC] > 1.0)
		pv[FLRFRAC] = 1.0;
#endif

	/* limit gamma wrt normal observer */
	if(gamma_calc(pv,&geom,&gamma) ) { 
		/* Treat gamma failure here as "fixable" for fixup_utoprim() */
			fprintf(stderr, "Gamma fail: %d %d %d %d \n",n, i, j, z);
		pflag[nl[n]][index_3D(n ,i,j,z)] = -333;
		pflag[nl[n]][index_3D(n ,N1_GPU_offset[n] - N1G, N2_GPU_offset[n] - N2G, N3_GPU_offset[n] - N3G)] = 100;
		failimage[nl[n]][index_3D(n ,i,j,z)][3]++ ;
	}
	else { 
		if(gamma > GAMMAMAX) {
			f = sqrt((GAMMAMAX*GAMMAMAX - 1.)/(gamma*gamma - 1.)) ;
			pv[U1] *= f ;	
			pv[U2] *= f ;	
			pv[U3] *= f ;	
		}
	}
#endif
	return;
}

/* find relative 4-velocity from 4-velocity (both in code coords) */
void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon)
{
	double alpha, beta[NDIM], gamma;
	int j;

	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	alpha = 1. / sqrt(-geom->gcon[0][0]);
	SLOOPA beta[j] = geom->gcon[0][j] * alpha*alpha;
	gamma = alpha*ucon[0];


	utcon[0] = 0;
	SLOOPA utcon[j] = ucon[j] + gamma*beta[j] / alpha;
}

void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
{
	double AA, BB, CC, DD, one_over_alpha_sq;
	//compute the Lorentz factor based on contravariant 3-velocity
	AA = geom->gcov[0][0];
	BB = 2.*(geom->gcov[0][1] * vcon[1] +
		geom->gcov[0][2] * vcon[2] +
		geom->gcov[0][3] * vcon[3]);
	CC = geom->gcov[1][1] * vcon[1] * vcon[1] +
		geom->gcov[2][2] * vcon[2] * vcon[2] +
		geom->gcov[3][3] * vcon[3] * vcon[3] +
		2.*(geom->gcov[1][2] * vcon[1] * vcon[2] +
		geom->gcov[1][3] * vcon[1] * vcon[3] +
		geom->gcov[2][3] * vcon[2] * vcon[3]);

	DD = -1. / (AA + BB + CC);

	one_over_alpha_sq = -geom->gcon[0][0];

	if (DD<one_over_alpha_sq) {
		DD = one_over_alpha_sq;
	}

	*ut = sqrt(DD);

}
/**************************************************************************************
 INTERPOLATION STENCILS:  
 ------------------------
   -- let the stencils be characterized by the following numbering convention:

           1 2 3 
           8 x 4      where x is the point at which we are interpolating 
           7 6 5
*******************************************************************************************/

/* 12345678 */
#define AVG8(pr,i,j,z,k, n)  \
        (0.125*(pr[nl[n]][index_3D(n ,i-1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i,j+1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j,z)][k]+pr[nl[n]][index_3D(n ,i+1,j-1,z)][k]+pr[nl[n]][index_3D(n ,i,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j,z)][k])) 

/* 2468  */
#define AVG4_1(pr,i,j,z,k, n) (0.25*(pr[nl[n]][index_3D(n ,i,j+1,z)][k]+pr[nl[n]][index_3D(n ,i,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j,z)][k]+pr[nl[n]][index_3D(n ,i+1,j,z)][k]))

/* 1357  */
#define AVG4_2(pr,i,j,z, k, n) (0.25*(pr[nl[n]][index_3D(n ,i+1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j-1,z)][k]))

/* 2468+cells in 3rd dimension  */
#define AVG6_1(pr,i,j,z,k, n) (1./6.*(pr[nl[n]][index_3D(n ,i,j+1,z)][k]+pr[nl[n]][index_3D(n ,i,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j,z)][k]+pr[nl[n]][index_3D(n ,i+1,j,z)][k] +pr[nl[n]][index_3D(n ,i,j,z+1)][k]+pr[nl[n]][index_3D(n ,i,j,z-1)][k]))

/* 1357+cells in 3rd dimension  */
#define AVG6_2(pr,i,j,z,k, n) (1./6.*(pr[nl[n]][index_3D(n ,i+1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j-1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i-1,j-1,z)][k] +pr[nl[n]][index_3D(n ,i,j,z+1)][k]+pr[nl[n]][index_3D(n ,i,j,z-1)][k]))

/* + shaped,  Linear interpolation in X1 or X2 directions using only neighbors in these direction */
/* 48  */
#define AVG2_X1(pr,i,j,z,k, n) (0.5*(pr[nl[n]][index_3D(n ,i-1,j,z)][k]+pr[nl[n]][index_3D(n ,i+1,j,z)][k]))
/* 26  */
#define AVG2_X2(pr,i,j,z,k, n) (0.5*(pr[nl[n]][index_3D(n ,i,j-1,z)][k]+pr[nl[n]][index_3D(n ,i,j+1,z)][k]))
/*910*/
#define AVG2_X3(pr,i,j,z,k, n) (0.5*(pr[nl[n]][index_3D(n ,i,j,z-1)][k]+pr[nl[n]][index_3D(n ,i,j,z+1)][k]))

/* x shaped,  Linear interpolation diagonally along both X1 and X2 directions "corner" neighbors */
/* 37  */
#define AVG2_1_X1X2(pr,i,j,z,k, n) (0.5*(pr[nl[n]][index_3D(n ,i-1,j-1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j+1,z)][k]))
/* 15  */
#define AVG2_2_X1X2(pr,i,j,z,k, n) (0.5*(pr[nl[n]][index_3D(n ,i-1,j+1,z)][k]+pr[nl[n]][index_3D(n ,i+1,j-1,z)][k]))

/*******************************************************************************************
  fixup_utoprim(): 

    -- figures out (w/ pflag[]) which stencil to use to interpolate bad point from neighbors;

    -- here we use the following numbering scheme for the neighboring cells to i,j:  

                      1  2  3 
                      8  x  4        where "x" is the (i,j) cell or the cell to be interpolated
                      7  6  5

 *******************************************************************************************/

void fixup_utoprim(double((* restrict pv[NB_LOCAL])[NPR]), int n)
{
  int i, j, z, k;
  int pf[11];

  /* Fix the interior points first */ 
	#pragma omp parallel shared(pflag, pv) private(i,j,z,k, pf)
	{
		#pragma omp for schedule(static,(BS_1+D1)*(BS_2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) 	{
			if (pflag[nl[n]][index_3D(n ,i, j, z)] != 0) {
				//fprintf(stderr, "i: %d j: %d, pflag: %d \n", i, j, pflag[i][j]);
				pf[1] = !pflag[nl[n]][index_3D(n ,i - 1, j + 1, z)];   pf[2] = !pflag[nl[n]][index_3D(n ,i, j + 1, z)];  pf[3] = !pflag[nl[n]][index_3D(n ,i + D1, j + D2, z)];
				pf[8] = !pflag[nl[n]][index_3D(n ,i - 1, j, z)];                           pf[4] = !pflag[nl[n]][index_3D(n ,i + 1, j, z)];
				pf[7] = !pflag[nl[n]][index_3D(n ,i - 1, j - 1, z)];   pf[6] = !pflag[nl[n]][index_3D(n ,i, j - 1, z)];  pf[5] = !pflag[nl[n]][index_3D(n ,i + D1, j - D2, z)];
				pf[9] = !pflag[nl[n]][index_3D(n ,i, j, z + D3)]; pf[10] = !pflag[nl[n]][index_3D(n ,i, j, z - D3)];
				/* Now the pf's  are true if they represent good points */

				//      if(      pf[1]&&pf[2]&&pf[3]&&pf[4]&&pf[5]&&pf[6]&&pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = AVG8(            pv,i,j,k)                   ; }
				//      else if(        pf[2]&&       pf[4]&&       pf[6]&&       pf[8] ){ FLOOP pv[i][j][k] = AVG4_1(          pv,i,j,k)                   ; }
				//      else if( pf[1]&&       pf[3]&&       pf[5]&&       pf[7]        ){ FLOOP pv[i][j][k] = AVG4_2(          pv,i,j,k)                   ; }
				//      else if(               pf[3]&&pf[4]&&              pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
				//      else if(        pf[2]&&pf[3]&&              pf[6]&&pf[7]        ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
				//      else if( pf[1]&&              pf[4]&&pf[5]&&              pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
				//      else if( pf[1]&&pf[2]&&              pf[5]&&pf[6]               ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
				//      else if(               pf[3]&&                     pf[7]        ){ FLOOP pv[i][j][k] = AVG2_1_X1X2(     pv,i,j,k)                   ; }
				//      else if( pf[1]&&                     pf[5]                      ){ FLOOP pv[i][j][k] = AVG2_2_X1X2(     pv,i,j,k)                   ; }
				//      else if(        pf[2]&&                     pf[6]               ){ FLOOP pv[i][j][k] = AVG2_X2(         pv,i,j,k)                   ; }
				//      else if(                      pf[4]&&                     pf[8] ){ FLOOP pv[i][j][k] = AVG2_X1(         pv,i,j,k)                   ; }

				// Old way:
				if (pf[2] && pf[4] && pf[6] && pf[8] && pf[9] && pf[10]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG6_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7] && pf[9] && pf[10]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG6_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[4] && pf[6] && pf[8]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG4_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG4_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[6]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X1(pv, i, j, z, k, n);
				}
				else if (pf[4] && pf[8]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X2(pv, i, j, z, k, n);
				}
				else if (pf[9] && pf[10]){
					FLOOP pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X3(pv, i, j, z, k, n);
				}
				else{
					failimage[nl[n]][index_3D(n ,i, j, z)][4]++;
					/* if nothing better to do, then leave densities and B-field unchanged, set v^i = 0 */
					for (k = RHO; k <= UU; k++) { pv[nl[n]][index_3D(n, i, j, z)][k] = 0.5*(AVG4_1(pv, i, j, z, k, n) + AVG4_2(pv, i, j, z, k, n)); }
					pv[nl[n]][index_3D(n ,i, j, z)][U1] = pv[nl[n]][index_3D(n ,i, j, z)][U2] = pv[nl[n]][index_3D(n ,i, j, z)][U3] = 0.;
				}
				pflag[nl[n]][index_3D(n ,i, j, z)] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
				//fixup1zone(i, j,z, pv[nl[n]][index_3D(n ,i,j,z)]);  /* Floor and limit gamma the interpolated value */
			}
		}
	}
  return;
}

#if(RAD_M1)
void fixup_utoprim_rad(double((*restrict pv[NB_LOCAL])[NPR]), int n)
{
	int i, j, z, k;
	int pf[11];

	/* Fix the interior points first */
	#pragma omp parallel shared(pflag, pv) private(i,j,z,k, pf)
	{
		#pragma omp for schedule(static,(BS_1+D1)*(BS_2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1) {
			if (pflag_rad[nl[n]][index_3D(n, i, j, z)] != 0) {
				//fprintf(stderr, "i: %d j: %d, pflag: %d \n", i, j, pflag[i][j]);
				pf[1] = !pflag_rad[nl[n]][index_3D(n, i - 1, j + 1, z)];   pf[2] = !pflag_rad[nl[n]][index_3D(n, i, j + 1, z)];  pf[3] = !pflag[nl[n]][index_3D(n, i + D1, j + D2, z)];
				pf[8] = !pflag_rad[nl[n]][index_3D(n, i - 1, j, z)];                           pf[4] = !pflag_rad[nl[n]][index_3D(n, i + 1, j, z)];
				pf[7] = !pflag_rad[nl[n]][index_3D(n, i - 1, j - 1, z)];   pf[6] = !pflag_rad[nl[n]][index_3D(n, i, j - 1, z)];  pf[5] = !pflag[nl[n]][index_3D(n, i + D1, j - D2, z)];
				pf[9] = !pflag_rad[nl[n]][index_3D(n, i, j, z + D3)]; pf[10] = !pflag_rad[nl[n]][index_3D(n, i, j, z - D3)];


				/* Now the pf's  are true if they represent good points */

				//      if(      pf[1]&&pf[2]&&pf[3]&&pf[4]&&pf[5]&&pf[6]&&pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = AVG8(            pv,i,j,k)                   ; }
				//      else if(        pf[2]&&       pf[4]&&       pf[6]&&       pf[8] ){ FLOOP pv[i][j][k] = AVG4_1(          pv,i,j,k)                   ; }
				//      else if( pf[1]&&       pf[3]&&       pf[5]&&       pf[7]        ){ FLOOP pv[i][j][k] = AVG4_2(          pv,i,j,k)                   ; }
				//      else if(               pf[3]&&pf[4]&&              pf[7]&&pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
				//      else if(        pf[2]&&pf[3]&&              pf[6]&&pf[7]        ){ FLOOP pv[i][j][k] = 0.5*(AVG2_1_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
				//      else if( pf[1]&&              pf[4]&&pf[5]&&              pf[8] ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X1(pv,i,j,k)); }
				//      else if( pf[1]&&pf[2]&&              pf[5]&&pf[6]               ){ FLOOP pv[i][j][k] = 0.5*(AVG2_2_X1X2(pv,i,j,k)+AVG2_X2(pv,i,j,k)); }
				//      else if(               pf[3]&&                     pf[7]        ){ FLOOP pv[i][j][k] = AVG2_1_X1X2(     pv,i,j,k)                   ; }
				//      else if( pf[1]&&                     pf[5]                      ){ FLOOP pv[i][j][k] = AVG2_2_X1X2(     pv,i,j,k)                   ; }
				//      else if(        pf[2]&&                     pf[6]               ){ FLOOP pv[i][j][k] = AVG2_X2(         pv,i,j,k)                   ; }
				//      else if(                      pf[4]&&                     pf[8] ){ FLOOP pv[i][j][k] = AVG2_X1(         pv,i,j,k)                   ; }

				// Old way:
				if (pf[2] && pf[4] && pf[6] && pf[8] && pf[9] && pf[10]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG6_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7] && pf[9] && pf[10]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG6_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[4] && pf[6] && pf[8]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG4_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG4_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[6]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X1(pv, i, j, z, k, n);
				}
				else if (pf[4] && pf[8]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X2(pv, i, j, z, k, n);
				}
				else if (pf[9] && pf[10]) {
					for (k = UU_RAD; k <= U3_RAD; k++) pv[nl[n]][index_3D(n, i, j, z)][k] = AVG2_X3(pv, i, j, z, k, n);
				}
				else {
					failimage[nl[n]][index_3D(n, i, j, z)][4]++;
					
					/* if nothing better to do, then floor values*/
					pv[nl[n]][index_3D(n, i, j, z)][UU_RAD] = pow(10.,-36.);
					pv[nl[n]][index_3D(n, i, j, z)][U1_RAD] = 0.;
					pv[nl[n]][index_3D(n, i, j, z)][U2_RAD] = 0.;
					pv[nl[n]][index_3D(n, i, j, z)][U3_RAD] = 0.;
				}
				pflag_rad[nl[n]][index_3D(n, i, j, z)] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
			}
		}
	}
	return;
}
#endif

void fix_flux(double(*restrict F1[NB_LOCAL])[NPR], double(*restrict F2[NB_LOCAL])[NPR], double(*restrict F3[NB_LOCAL])[NPR], int n)
{
	int i, j, z, k;
	double test;
	if (block[n][AMR_NBR1] == -1){
		#pragma omp parallel shared(block, n,n_ord,F1, F2, F3) private(i,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_GPU_offset[n] - D1; i < N1_GPU_offset[n] + BS_1 + D1; i++){
				#pragma ivdep
				for (z = N3_GPU_offset[n] - D3; z < N3_GPU_offset[n] + BS_3 + D3; z++){
					F1[nl[n]][index_3D(n, i, -1, z)][B2] = -F1[nl[n]][index_3D(n, i, 0, z)][B2];
					F3[nl[n]][index_3D(n, i, -1, z)][B2] = -F3[nl[n]][index_3D(n, i, 0, z)][B2];
					#if INFLOW==0
					PLOOP F2[nl[n]][index_3D(n, i, 0, z)][k] = 0.;
					#endif	
				}
			}
		}
	}

	if (block[n][AMR_NBR3] == -1){
		#pragma omp parallel shared(block,n,n_ord,F1, F2, F3) private(i,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_GPU_offset[n] - D1; i < N1_GPU_offset[n] + BS_1 + D1; i++){
				#pragma ivdep
				for (z = N3_GPU_offset[n] - D3; z < N3_GPU_offset[n] + BS_3 + D3; z++){
					F1[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL]), z)][B2] = -F1[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z)][B2];
					F3[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL]), z)][B2] = -F3[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL]) - 1, z)][B2];
				}
				#if INFLOW==0
				PLOOP F2[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL]), z)][k] = 0.;
				#endif	
			}
		}
	}
		if (INFLOW == 0){
		if (block[n][AMR_NBR4] == -1){
			#pragma omp parallel shared(block,n,n_ord,F1) private(j,z)
			{
				#pragma omp for schedule(static,1)
				for (j = N2_GPU_offset[n] - D2; j < N2_GPU_offset[n] + BS_2 + D2; j++){
					#pragma ivdep
					for (z = N3_GPU_offset[n] - D3; z < N3_GPU_offset[n] + BS_3 + D3; z++){
						if (F1[nl[n]][index_3D(n, 0, j, z)][RHO] > 0.) F1[nl[n]][index_3D(n, 0, j, z)][RHO] = 0.;
					}
				}
			}
		}
		if (block[n][AMR_NBR2] == -1){
			#pragma omp parallel shared(block,n,n_ord,F1) private(j,z)
			{
				#pragma omp for schedule(static,1)
				for (j = N2_GPU_offset[n] - D2; j < N2_GPU_offset[n] + BS_2 + D2; j++){
					#pragma ivdep
					for (z = N3_GPU_offset[n] - D3; z < N3_GPU_offset[n] + BS_3 + D3; z++){
						if (F1[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL]), j, z)][RHO] < 0.) F1[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL]), j, z)][RHO] = 0.;
					}
				}
			}
		}
	}
	return;
}

