/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/
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
	struct of_geom geom;

	coord(n, i,j, z, CENT,X) ;
	bl_coord(X,&r,&th, &phi) ;

	// Danat addition: 11/18/19 - avoid rhoflr too large`
	get_rho_u_floor (r, th, phi, &rhoflr, &uuflr); 
    
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
	#endif

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

	//floor on density and internal energy density (momentum *not* conserved) 
	for (k = 0; k < NPR_U; k++) pv_prefloor[k] = pv[k];
	if (pv[RHO] < rhoflr) {
		pv[RHO] = rhoflr;
		dofloor = 1;
	}

	//Internal energy floor
	#if(RAD_M1)
	if (u + pv[UU_RAD] < uuflr) {
		u = uuflr - pv[UU_RAD];
		dofloor = 1;
	}
	#elif(NEUTRINOS_M1)
	#if (NU_SPECIES > 1)
	if (u + pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)] < uuflr) {
		u = uuflr - (pv[UU_NU] + pv[index_nu(UU_NU, 1)] + pv[index_nu(UU_NU, 2)]);
	#else
	if (u + pv[UU_NU] < uuflr) {
		u = uuflr - pv[UU_NU];
	#endif
		#if (!(DOHELM_TEMPERATURE == 2))
		pv[UU] = u;
		#endif
		dofloor = 1;
	}

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

	// Floor on Ye
	#if (DO_YE)
	pv[YE] = MY_MAX(nulib_ylo, pv[YE]);
	pv[YE] = MY_MIN(nulib_yhi, pv[YE]);
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

	#if(DRIFT_FLOOR)
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
		for (m = 1; m < NDIM; m++) {
			Bcon[m] = pv[B1 - 1 + m];
		}

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
		#else
		wnew = pv[RHO] + pv[UU] * gam;
		//wnew = wold;
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

	if (dofloor) {
		#if(TWO_T)
			#if(FIXEDGAMMA)
				#if(FULL_ENTROPY)
				pv[ENTRE] = 1. / (GAMMAE - 1.) * log(0.5 * (GAMMAE - 1.0) * pv[UU] * pow(pv[RHO], -GAMMAE));
				pv[ENTRI] = 1. / (GAMMA - 1.) * log(0.5 * (GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA));
				#else
				pv[ENTRE] = 0.5 * (GAMMAE - 1.0) * pv[UU] * pow(pv[RHO], -GAMMAE);
				pv[ENTRI] = 0.5 * (GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA);
				#endif
			#else


			#endif
		#endif
	}

	#if DOKTOT
	#if (DOHELM)
	double xentr;
	#if (DOHELM_TEMPERATURE == 2)
	eos_mode_rhotemp_entr(pv[RHO], pv[UU], &xentr);
	#else
	eos_mode_rhou_entr(pv, &xentr);
	#endif
	pv[KTOT] = xentr;
	#else 
	#if(FULL_ENTROPY)
	pv[KTOT] = 1. / (GAMMA - 1.) * log((GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA));
	#else
	pv[KTOT] = (GAMMA - 1.0) * pv[UU] * pow(pv[RHO], -GAMMA);
	#endif
	#endif
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
