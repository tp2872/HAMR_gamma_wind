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

/*************************************************************************************/
/*************************************************************************************/
/*************************************************************************************

utoprim_2d.c:
---------------

Uses the 2D method:
-- solves for two independent variables (W,v^2) via a 2D
Newton-Raphson method
-- can be used (in principle) with a general equation of state.

-- Currently returns with an error state (>0) if a negative rest-mass
density or internal energy density is calculated.  You may want
to change this aspect of the code so that it still calculates the
velocity and so that you can floor the densities.  If you want to
change this aspect of the code please comment out the "return(retval)"
statement after "retval = 5;" statement in Utoprim_new_body();

******************************************************************************/
#include "u2p_util.h"
#include "decs.h"

/* these variables need to be shared between the functions
Utoprim_1D, residual, and utsq */
//double Bsq, QdotBsq, Qtsq, Qdotn, D;
//#pragma omp threadprivate(Bsq, QdotBsq, Qtsq, Qdotn, D)


int Utoprim_3d_res(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim){
	/*double U_tmp[NPR],  prim_tmp[NPR_HD];
	int i, j, ret=0;
	double alpha;

	if (U[0] <= 0.) {
		return(-100);
	}

	//First update the primitive B-fields
	 #pragma ivdep
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//First update the primitive E-fields
	#pragma ivdep
	for (i = E1; i <= E3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);

	//Transform the CONSERVED variables to 3+1
	U_tmp[RHO] = alpha * U[RHO] / gdet;

	//Energy to 3+1
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	
	//Momentum to 3+1
	#pragma ivdep
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	
	//Magnetic field to 3+1
	#pragma ivdep
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Electric field to 3+1
	#pragma ivdep
	for (i = E1; i <= E3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	#pragma ivdep
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	//ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim);

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		 #pragma ivdep
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);*/
}