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

#include "decs.h"

/** 
 *
 * this file contains all the coordinate dependent
 * parts of the code, except the initial and boundary
 * conditions 
 *
 **/

/* should return boyer-lindquist coordinte of point */
void bl_coord(double *X, double *r, double *th, double *phi)
{
	*r = exp(X[1]) + R0 ;

	double A1 = 1. / (1. + pow(CHARLIE*log(*r) / log(10.), DELTA));
	double A2 = (0.5 - BRAVO*0.5) / pow(0.5, QUEBEC);
	double Xc = sqrt(pow(X[2], 2.));
	double sign=1.;
	
	if (X[2] < 0.0){
		sign = -1.;
	}
	if (X[2] > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X[2] < 0.5){
		*th = sign*(A1* M_PI*Xc + M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))*(1. - A1) + 0.50*(1. - A1)*sin(M_PI + 2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
	}
	else{
		*th = M_PI - sign*(A1* M_PI*(1. - Xc) + M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))*(1. - A1) + 0.50*(1. - A1)*sin(M_PI + 2.*M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))));
	}

	*phi = X[3];

	// avoid singularity at polar axis
	#if(COORDSINGFIX)
	if(fabs(*th)<SINGSMALL){
	  if((*th)>=0) *th =  SINGSMALL;
	  if((*th)<0)  *th = -SINGSMALL;
	}
	if(fabs(M_PI - (*th)) < SINGSMALL){
	  if((*th)>=M_PI) *th = M_PI+SINGSMALL;
	  if((*th)<M_PI)  *th = M_PI-SINGSMALL;
	}
	#endif
	
	return ;
}

/* insert metric here */
void gcov_func(double *X, double gcov[][NDIM])
{
	int j,k ;
	double sth,cth,s2,rho2 ;
	double r,th, phi ;
	double tfac,rfac,hfac1, hfac2 ,pfac, A1, A2, A3, Xc, sign ;

	DLOOP gcov[j][k] = 0. ;

	bl_coord(X,&r,&th, &phi) ;

	cth = cos(th) ;
	sth = sin(th) ;
	s2 = sth*sth ;
	rho2 = r*r + a*a*cth*cth ;
	tfac = 1. ;
	rfac = r - R0 ;
	A1 = CHARLIE*DELTA*pow(log(10.), -DELTA)*pow(CHARLIE*log(r), -1. + DELTA)/pow(1. + pow(log(10.), -DELTA)*pow(CHARLIE*log(r), DELTA), 2.);
	A2 = (0.5 - BRAVO*0.5) / pow(0.5, QUEBEC);
	A3 = 1. / (1. + pow(log(10.), -DELTA)*pow(CHARLIE*log(r), DELTA));
	Xc = sqrt(pow(X[2], 2.));
	sign = 1.;
	if (X[2] < 0.0){
		sign = -1.;
	}
	if (X[2] > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X[2] < 0.5){
		hfac1 = sign*(-M_PI*A1*Xc + M_PI*A1*(BRAVO*Xc + A2*pow(Xc, QUEBEC)) + 0.5*A1*sin(2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
		hfac2 = M_PI * A3 + M_PI*(BRAVO + A2*QUEBEC*pow(Xc, -1. + QUEBEC))*(1. - A3) *(1. - cos(2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
	}
	else{
		hfac1 = sign*(-M_PI*A1*(1. - Xc) + M_PI*A1*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC)) + 0.5*A1*sin(2.*M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))));
		hfac2 = M_PI * A3 + M_PI*(BRAVO + A2*QUEBEC*pow(1. - Xc, -1. + QUEBEC))*(1. - A3) *(1. - cos(2.*M_PI*(BRAVO*(1. - Xc) + A2*pow(1. - Xc, QUEBEC))));
	}

	pfac = 1. ;

	gcov[TT][TT] = (-1. + 2.*r/rho2)      * tfac*tfac ;
	gcov[TT][1] = (2.*r/rho2)             * tfac*rfac ;
	gcov[TT][3] = (-2.*a*r*s2/rho2)       * tfac*pfac ;

	gcov[1][TT] = gcov[TT][1] ;
	gcov[1][1] = (1. + 2.*r/rho2)         * rfac*rfac + rho2*hfac1*hfac1;
	gcov[1][2] = rho2					  * hfac1*hfac2;
	gcov[1][3] = (-a*s2*(1. + 2.*r/rho2)) * rfac*pfac ;

	gcov[2][1] = gcov[1][2];
	gcov[2][2] = rho2                     * hfac2*hfac2 ;

	gcov[3][TT] = gcov[TT][3] ;
	gcov[3][1]  = gcov[1][3] ;
	gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r/rho2)) * pfac*pfac ;
}

/* some grid location, dxs */
void set_points()
{
        startx[1] = log(Rin - R0) ;
        startx[2] = 0.+0.5*(1.-fractheta) ;
		startx[3] = 0.;
        dx[1] = log((Rout - R0)/(Rin - R0))/N1 ;
        dx[2] = fractheta/(N2) ;
		dx[3] = 2.*M_PI / (N3);
}

void fix_flux(double(*F1)[NPR], double(*F2)[NPR], double(*F3)[NPR])
{
	int i,j,z,k ;
	double test;
	if (N2_MPI_offset == 0){
		#pragma omp parallel shared(F1, F2, F3) private(i,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - D1; i < N1_MPI_offset + N1_MPI+D1; i++){
				//#pragma omp simd
				for (z = N3_MPI_offset - D3; z < N3_MPI_offset + N3_MPI + D3; z++){
					test = F1[index(i, -1, z)][B2];
					F1[index(i, -1, z)][B2] = -F1[index(i, 0, z)][B2];
					F3[index(i, -1, z)][B2] = -F3[index(i, 0, z)][B2];
				//	if ((F1[index(i, -1, z)][B2]/test>1.00001 ||F1[index(i, -1, z)][B2]/test<0.99999)) printf("F1: %f\n", F1[index(i, -1, z)][B2]/test);
					#if INFLOW==0
					PLOOP F2[index(i, 0, z)][k] = 0.;
					#endif	
				}	
			}
		}
	}

	if (N2_MPI_offset + N2_MPI==N2){
		#pragma omp parallel shared(F1, F2, F3) private(i,z,k)
		{
			#pragma omp for schedule(static,1)
			for (i = N1_MPI_offset - D1; i < N1_MPI_offset + N1_MPI+D1; i++){
				//#pragma omp simd
				for (z = N3_MPI_offset - D3; z < N3_MPI_offset + N3_MPI + D3; z++){
					F1[index(i, N2, z)][B2] = -F1[index(i, N2-1, z)][B2];
					F3[index(i, N2, z)][B2] = -F3[index(i, N2-1, z)][B2];
				}
				#if INFLOW==0
				PLOOP F2[index(i, N2, z)][k] = 0.;
				#endif	
			}
		}
	}
	if (INFLOW == 0){
		if (N1_MPI_offset == 0){
			#pragma omp parallel shared(F1) private(j,z)
			{
				#pragma omp for schedule(static,1)
				for (j = N2_MPI_offset - D2; j < N2_MPI_offset + N2_MPI + D2; j++){
					//#pragma omp simd
					for (z = N3_MPI_offset - D3; z < N3_MPI_offset + N3_MPI + D3; z++){
						if (F1[index(0, j, z)][RHO] > 0.) F1[index(0, j, z)][RHO] = 0.;
					}
				}
			}
		}
		if (N1_MPI_offset + N1_MPI == N1){
			#pragma omp parallel shared(F1) private(j,z)
			{
				#pragma omp for schedule(static,1)
				for (j = N2_MPI_offset - D2; j < N2_MPI_offset + N2_MPI + D2; j++){
					//#pragma omp simd
					for (z = N3_MPI_offset - D3; z < N3_MPI_offset + N3_MPI + D3; z++){
						if (F1[index(N1, j, z)][RHO] < 0.) F1[index(N1, j, z)][RHO] = 0.;
					}
				}
			}
		}
	}
	return;
}

void rescale(double *pr, int which, int dir, int ii, int jj, int zz, int face, struct of_geom *geom)
{
	double scale[NPR], r, th, phi, X[NDIM];
	int k;

	coord(ii, jj, zz, face, X);
	bl_coord(X, &r, &th, &phi);

	if (dir == 1) {
		// optimized for pole
		scale[RHO] = pow(r, 1.5);
		scale[UU] = scale[RHO] * r;
		scale[U1] = scale[RHO];
		scale[U2] = 1.0;
		scale[U3] = r * r;
		scale[B1] = r * r;
		scale[B2] = r * r;
		scale[B3] = r * r;
	}
	else if (dir == 2) {
		scale[RHO] = 1.0;
		scale[UU] = 1.0;
		scale[U1] = 1.0;
		scale[U2] = 1.0;
		scale[U3] = 1.0;
		scale[B1] = 1.0;
		scale[B2] = 1.0;
		scale[B3] = 1.0;
	}
	else if (dir == 3) {
		scale[RHO] = 1.0;
		scale[UU] = 1.0;
		scale[U1] = 1.0;
		scale[U2] = 1.0;
		scale[U3] = 1.0;
		scale[B1] = 1.0;
		scale[B2] = 1.0;
		scale[B3] = 1.0;
	}

	if (which == FORWARD) {	// rescale before interpolation
		PLOOP pr[k] *= scale[k];
	} else if (which == REVERSE) {	// unrescale after interpolation
		PLOOP pr[k] /= scale[k];
	} else {
		if (rank == 0){
			fprintf(stderr, "no such rescale type!\n");
		}
		exit(100) ;
	}
}
