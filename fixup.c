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

#define FLOOP for(k=0;k<B1;k++)

/* apply floors to density, internal energy */

void fixup(double((* restrict pv[NB])[NPR]), int n)
{
	int i, j, z;
	#pragma omp parallel shared(n,pv,N1_GPU, N2_GPU, N3_GPU, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z)
	{
		#pragma omp for collapse(2) schedule(static,N1_GPU[n]*N2_GPU[n]/nthreads)
		for (i = N1_GPU_offset[n]; i<N1_GPU_offset[n] + N1_GPU[n]; i++)for (j = N2_GPU_offset[n]; j<N2_GPU_offset[n] + N2_GPU[n]; j++)for (z = N3_GPU_offset[n]; z<N3_GPU_offset[n] + N3_GPU[n]; z++){			
			fixup1zone(i, j, z, n, pv[n][index(n, i, j, z)]);
		}
	}
}

void fixup1zone( int i, int j, int z, int n, double pv[NPR] ) 
{
  double r,th, phi, X[NDIM],uuscal,rhoscal, rhoflr,uuflr;
  double f,gamma, bsq;
  double pv_prefloor[NPR], dpv[NPR], U_prefloor[NPR], dU[NPR], U[NPR], U_ent;
  double trans, betapar, betasq, betasqmax, one_over_ucondr_, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut;
  double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];
  int m;
  int k, flag, dofloor;
  struct of_state q;
  struct of_geom geom;

  coord(n, i,j, z, CENT,X) ;
  bl_coord(X,&r,&th, &phi) ;

  rhoscal = pow(r,-POWRHO) ;
  uuscal = pow(rhoscal,gam);

  rhoflr = RHOMIN*rhoscal;
  uuflr  = UUMIN*uuscal;

  //compute the square of fluid frame magnetic field (twice magnetic pressure)
  get_geometry(n,i,j,z,CENT,&geom) ;
  bsq = bsq_calc(pv,&geom) ;
  
  //tie floors to the local values of magnetic field and internal energy density
#if(1)
  if( rhoflr < bsq / BSQORHOMAX ) rhoflr = bsq / BSQORHOMAX;
  if( uuflr < bsq / BSQOUMAX ) uuflr = bsq / BSQOUMAX;
  if( rhoflr < pv[UU] / UORHOMAX ) rhoflr = pv[UU] / UORHOMAX;
#endif

  if( rhoflr < RHOMINLIMIT ) rhoflr = RHOMINLIMIT;
  if( uuflr  < UUMINLIMIT  ) uuflr  = UUMINLIMIT;

  /* floor on density and internal energy density (momentum *not* conserved) */
   #pragma ivdep
  PLOOP pv_prefloor[k] = pv[k];
	if (pv[RHO] < rhoflr){
		pv[RHO] = rhoflr;
		dofloor = 1;
	}
	if (pv[UU] < uuflr){
		pv[UU] = uuflr;
		dofloor = 1;
	}

	#if( ZAMO_FLOOR )
	if (dofloor && (trans = 10.*bsq / MY_MIN(pv[RHO], pv[UU]) - 1.) > 0.) {
		get_state(pv_prefloor, &geom, &q);
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
		wold = pv_prefloor[RHO] + pv_prefloor[UU] * gam;

		//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
		QdotB = udotB*wold*q.ucon[0];

		//enthalpy after the floors
		wnew = pv[RHO] + pv[UU] * gam;
		//wnew = wold;

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

	#if DOKTOT
	pv[KTOT] = (gam - 1.)*pv[UU] * pow(pv[RHO], -gam);
	#endif
  /* limit gamma wrt normal observer */

  if( gamma_calc(pv,&geom,&gamma) ) { 
    /* Treat gamma failure here as "fixable" for fixup_utoprim() */
	  fprintf(stderr, "Gamma fail: %d %d %d %d \n",n, i, j, z);
    pflag[n][index(n ,i,j,z)] = -333;
	pflag[n][index(n ,N1_GPU_offset[n] - N1G, N2_GPU_offset[n] - N2G, N3_GPU_offset[n] - N3G)] = 100;
    failimage[n][index(n ,i,j,z)][3]++ ;
  }
  else { 
    if(gamma > GAMMAMAX) {
      f = sqrt(
	       (GAMMAMAX*GAMMAMAX - 1.)/
	       (gamma*gamma - 1.)
	       ) ;
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
	alpha = 1. / sqrt(-geom->gcon[TT][TT]);
	SLOOPA beta[j] = geom->gcon[TT][j] * alpha*alpha;
	gamma = alpha*ucon[TT];


	utcon[0] = 0;
	SLOOPA utcon[j] = ucon[j] + gamma*beta[j] / alpha;
}

void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
{
	double AA, BB, CC, DD, one_over_alpha_sq;
	//compute the Lorentz factor based on contravariant 3-velocity
	AA = geom->gcov[TT][TT];
	BB = 2.*(geom->gcov[TT][1] * vcon[1] +
		geom->gcov[TT][2] * vcon[2] +
		geom->gcov[TT][3] * vcon[3]);
	CC = geom->gcov[1][1] * vcon[1] * vcon[1] +
		geom->gcov[2][2] * vcon[2] * vcon[2] +
		geom->gcov[3][3] * vcon[3] * vcon[3] +
		2.*(geom->gcov[1][2] * vcon[1] * vcon[2] +
		geom->gcov[1][3] * vcon[1] * vcon[3] +
		geom->gcov[2][3] * vcon[2] * vcon[3]);

	DD = -1. / (AA + BB + CC);

	one_over_alpha_sq = -geom->gcon[TT][TT];

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
        (0.125*(pr[n][index(n ,i-1,j+1,z)][k]+pr[n][index(n ,i,j+1,z)][k]+pr[n][index(n ,i+1,j+1,z)][k]+pr[n][index(n ,i+1,j,z)][k]+pr[n][index(n ,i+1,j-1,z)][k]+pr[n][index(n ,i,j-1,z)][k]+pr[n][index(n ,i-1,j-1,z)][k]+pr[n][index(n ,i-1,j,z)][k])) 

/* 2468  */
#define AVG4_1(pr,i,j,z,k, n) (0.25*(pr[n][index(n ,i,j+1,z)][k]+pr[n][index(n ,i,j-1,z)][k]+pr[n][index(n ,i-1,j,z)][k]+pr[n][index(n ,i+1,j,z)][k]))

/* 1357  */
#define AVG4_2(pr,i,j,z, k, n) (0.25*(pr[n][index(n ,i+1,j+1,z)][k]+pr[n][index(n ,i+1,j-1,z)][k]+pr[n][index(n ,i-1,j+1,z)][k]+pr[n][index(n ,i-1,j-1,z)][k]))

/* 2468+cells in 3rd dimension  */
#define AVG6_1(pr,i,j,z,k, n) (1./6.*(pr[n][index(n ,i,j+1,z)][k]+pr[n][index(n ,i,j-1,z)][k]+pr[n][index(n ,i-1,j,z)][k]+pr[n][index(n ,i+1,j,z)][k] +pr[n][index(n ,i,j,z+1)][k]+pr[n][index(n ,i,j,z-1)][k]))

/* 2468+cells in 3rd dimension  */
#define AVG6_2(pr,i,j,z,k, n) (1./6.*(pr[n][index(n ,i+1,j+1,z)][k]+pr[n][index(n ,i+1,j-1,z)][k]+pr[n][index(n ,i-1,j+1,z)][k]+pr[n][index(n ,i-1,j-1,z)][k] +pr[n][index(n ,i,j,z+1)][k]+pr[n][index(n ,i,j,z-1)][k]))

/* + shaped,  Linear interpolation in X1 or X2 directions using only neighbors in these direction */
/* 48  */
#define AVG2_X1(pr,i,j,z,k, n) (0.5*(pr[n][index(n ,i-1,j,z)][k]+pr[n][index(n ,i+1,j,z)][k]))
/* 26  */
#define AVG2_X2(pr,i,j,z,k, n) (0.5*(pr[n][index(n ,i,j-1,z)][k]+pr[n][index(n ,i,j+1,z)][k]))
/*910*/
#define AVG2_X3(pr,i,j,z,k, n) (0.5*(pr[n][index(n ,i,j,z-1)][k]+pr[n][index(n ,i,j,z+1)][k]))

/* x shaped,  Linear interpolation diagonally along both X1 and X2 directions "corner" neighbors */
/* 37  */
#define AVG2_1_X1X2(pr,i,j,z,k, n) (0.5*(pr[n][index(n ,i-1,j-1,z)][k]+pr[n][index(n ,i+1,j+1,z)][k]))
/* 15  */
#define AVG2_2_X1X2(pr,i,j,z,k, n) (0.5*(pr[n][index(n ,i-1,j+1,z)][k]+pr[n][index(n ,i+1,j-1,z)][k]))

/*******************************************************************************************
  fixup_utoprim(): 

    -- figures out (w/ pflag[]) which stencil to use to interpolate bad point from neighbors;

    -- here we use the following numbering scheme for the neighboring cells to i,j:  

                      1  2  3 
                      8  x  4        where "x" is the (i,j) cell or the cell to be interpolated
                      7  6  5

 *******************************************************************************************/

void fixup_utoprim(double((* restrict pv[NB])[NPR]), int n)
{
  int i, j, z, k;
  int pf[11];

  /* Fix the interior points first */ 
	#pragma omp parallel shared(pflag, pv) private(i,j,z,k, pf)
	{
		#pragma omp for schedule(static,1)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + N1_GPU[n] - 1, N2_GPU_offset[n], N2_GPU_offset[n] + N2_GPU[n] - 1, N3_GPU_offset[n], N3_GPU_offset[n] + N3_GPU[n] - 1) 	{
			if (pflag[n][index(n ,i, j, z)] != 0) {
				//printf("i: %d j: %d, pflag: %d \n", i, j, pflag[i][j]);
				pf[1] = !pflag[n][index(n ,i - 1, j + 1, z)];   pf[2] = !pflag[n][index(n ,i, j + 1, z)];  pf[3] = !pflag[n][index(n ,i + 1, j + 1, z)];
				pf[8] = !pflag[n][index(n ,i - 1, j, z)];                           pf[4] = !pflag[n][index(n ,i + 1, j, z)];
				pf[7] = !pflag[n][index(n ,i - 1, j - 1, z)];   pf[6] = !pflag[n][index(n ,i, j - 1, z)];  pf[5] = !pflag[n][index(n ,i + 1, j - 1, z)];
				#if(N3>1)
				pf[9] = !pflag[n][index(n ,i, j, z + 1)]; pf[10] = !pflag[n][index(n ,i, j, z - 1)];
				#else
				pf[9]=0;						      pf[10]=0;
				#endif
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
					FLOOP pv[n][index(n, i, j, z)][k] = AVG6_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7] && pf[9] && pf[10]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG6_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[4] && pf[6] && pf[8]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG4_1(pv, i, j, z, k, n);
				}
				else if (pf[1] && pf[3] && pf[5] && pf[7]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG4_2(pv, i, j, z, k, n);
				}
				else if (pf[2] && pf[6]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG2_X1(pv, i, j, z, k, n);
				}
				else if (pf[4] && pf[8]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG2_X2(pv, i, j, z, k, n);
				}
				else if (pf[9] && pf[10]){
					FLOOP pv[n][index(n, i, j, z)][k] = AVG2_X3(pv, i, j, z, k, n);
				}
				else{
					failimage[n][index(n ,i, j, z)][4]++;
					/* if nothing better to do, then leave densities and B-field unchanged, set v^i = 0 */
					for (k = RHO; k <= UU; k++) { pv[n][index(n, i, j, z)][k] = 0.5*(AVG4_1(pv, i, j, z, k, n) + AVG4_2(pv, i, j, z, k, n)); }
					pv[n][index(n ,i, j, z)][U1] = pv[n][index(n ,i, j, z)][U2] = pv[n][index(n ,i, j, z)][U3] = 0.;
				}
				pflag[n][index(n ,i, j, z)] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
				//fixup1zone(i, j,z, pv[n][index(n ,i,j,z)]);  /* Floor and limit gamma the interpolated value */
			}
		}
	}
  return;
}


#if( DO_FONT_FIX ) 
/***********************************************************************
   set_Katm():

       -- sets the EOS constant used for Font's fix. 

       -- see utoprim_1dfix1.c and utoprim_1dvsq2fix1.c  for more
           information. 

       -- uses the initial floor values of rho/u determined by fixup1zone()

       -- we assume here that Constant X1,r is independent of theta,X2

***********************************************************************/
void set_Katm( void )
{
  int i, j, k, G_type, n ;
  double prim[NPR], G_tmp;

  G_type = get_G_ATM( &G_tmp );

  if (rank == 0){
	  fflush(stdout);
	  fprintf(stdout, "G_tmp = %26.20e \n", G_tmp);
	  fflush(stdout);
  }

  for (n = 0; n < n_active; n++){
	  j = N2_GPU_offset[n_ord[n]];
	  for (i = N1_GPU_offset[n_ord[n]]-N1G; i < N1_GPU_offset[n_ord[n]] + N1_GPU[n_ord[n]]+N1G; i++) {
		  PLOOP prim[k] = 0.;
		  prim[RHO] = prim[UU] = -1.;

		  fixup1zone(i, j, N3_GPU_offset[n_ord[n]], n_ord[n], prim);
		  Katm[n_ord[n]][i - (N1_GPU_offset[n_ord[n]]-N1G)] = (gam - 1.) * prim[UU] / pow(prim[RHO], G_tmp);

		  //fflush(stdout);
		  //fprintf(stdout,"Katm[%d] = %26.20e \n", i, Katm[i] );
		  //fflush(stdout); 
	  }
  }
  return;
}
#endif


#undef FLOOP 
