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
int invert_3DU(double D, double Dt_times_alpha, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
void res_3du_der(double D, double Dt_times_alpha, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);
void getE_resistive(double Enew[3], double E[3], double vD[3], double B_D[3], double Dt_times_alpha, double etares, double ggammainv[3][3], double sqrtgamma, double lfac);
void getdEdu_resistive(double Enew[3], double E[3], double vD[3], double B_D[3], double Dt_times_alpha, double etares, double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]);

int Utoprim_3d_res(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim, double Dt){
	double D, tau, S[3], B_guess[3], E_guess[3], ncov[NDIM], ncon[NDIM], U_tmp[NPR], rho , ug;
	int i, j, k, retval = 0;
	double alpha, utsq, etares, u, p, enth, sqrtgamma;
	double vD_guess[3], ExB[3], xi_guess, ggamma[3][3], ggammainv[3][3];

	//Return if rho*gamma is negative
	if (U[0] <= 0.) {
		return(-100);
	}

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);
	sqrtgamma = gdet / alpha; //determinant for spatial part of metric
	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);

	//Calculate covariant and contravariant 3+1 metrics
	for (i=0;i<3; i++){
		 for (j=0;j<3;j++){
			ggamma[i][j] = gcov[i + 1][j + 1] + ncov[i + 1] * ncov[j + 1]; //gamma_{ij}
			ggammainv[i][j] = gcon[i + 1][j + 1] + ncon[i + 1] * ncon[j + 1]; //gamma^{ij}
		 }
	}

	//Transform the CONSERVED variables to 3+1
	D = alpha * U[RHO] / gdet;

	//Energy to 3+1
	tau = ncov[0] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3]) / gdet;
	
	//Momentum to 3+1
	#pragma ivdep
	for (i = 0; i < 3; i++) S[i] = ncov[0] * (delta(i + 1, 1) * U[U1] + delta(i + 1, 2) * U[U2] + delta(i + 1, 3) * U[U3] + ncov[i+1] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3])) / gdet;
	
	//Magnetic field to 3+1
	#pragma ivdep
	for (i = 0; i < 3; i++) B_guess[i] = alpha * U[B1+i] / gdet;

	//Electric field to 3+1
	#pragma ivdep
	for (i = 1; i < 3; i++) E_guess[i] = alpha * U[E1+i] / gdet;

	//guesses for NR 
	u = U[UU] / gdet;
	p = pressure_rho0_u(U[RHO] / gdet, u);
	enth = (1.0 + u + p) / (U[RHO] / gdet);
	xi_guess = D * enth;
  	  	
  	for (i = 0; i < 3; i++){
		ExB[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++){
			ExB[i] = ExB[i] + sqrtgamma*lvc3u(i,j,k)*E_guess[j]*B_guess[k];
		}
		vD_guess[i] = (S[i] - ExB[i]) / xi_guess; //gamma*v_i-->vD_guess (eq. 57)
	}
	
	//NR Step, you get back gamma*v_i and E
	retval=invert_3DU(D, Dt*alpha, ETA, tau, S, ggamma, ggammainv, sqrtgamma, &rho, &ug, B_guess, E_guess, vD_guess, tolerance);

	//Transform new primitive variables back if there was no problem
	if (retval == 0) {
		prim[RHO] = rho;
		prim[UU] = ug;
		prim[U1] = vD_guess[0];
		prim[U2] = vD_guess[1];
		prim[U3] = vD_guess[2];
		prim[E1] = E_guess[0] / alpha;
		prim[E2] = E_guess[1] / alpha;
		prim[E3] = E_guess[2] / alpha;
	}

	//Update B fields regardless to preserve Div.B==0
	#pragma ivdep
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	return(retval);
}

//gives back E and gamma*v_i
int invert_3DU(double D, double Dt_times_alpha, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double *rho, double *ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance){
	double vD[3], vU[3], vDprev[3], xk_3du[3], J_3du[3][3], J_3du_inv[3][3], f_3du[3], Enew[3], B_D[3], alpha, lfac;
	double Enew_D[3], Stilde_j[3], ExB[3], Stilde_uj[3], Ssqr, bsqr, esqr, tautilde, z, eps;
	int i,j,k, nit, maxitnr, ii;
	double er, er1, normV, half;
	int maxitr = 100;
	int retval = 1;

	for (i=0;i<3; i++) vD[i] = vD_guess[i];

    //Newton-Raphson iteration
    er = 1.0;
    er1 = 1.0;
    nit = 0;

    for (i=0;i<3;i++) vDprev[i] = vD[i];
	ii = 1;

	// Start of the Newton RAphson loop
    do{ 
    	nit = nit + 1;
        if(nit>maxitnr/2){
        	// mix the last  value for convergence
			for (i = 0; i < 3; i++) vD[i]=0.5*(vD[i]+vDprev[i]);
          	
			// relax accuracy requirement
          	er1=10.0*er1;
          	
			// following avoids decrease of accuracy requirement every iteration step beyond maxitnr/2
          	nit = nit - maxitnr/10;
		}
        
        //Compute residual and derivatives
        for (i=0;i<3;i++) xk_3du[i] = vD[i];   
        
		//Calculate jacobian and residuals
        res_3du_der(D, Dt_times_alpha, etares, tau, S, xk_3du, gcov, gcon, sqrtgamma, B_guess, E_guess, J_3du, f_3du); //MATTHEW: Why don't you use J_3du?

        //Store previous ucov_tilde
        for (i=0;i<3;i++) vDprev[i] = vD[i];
        
		//Find inverse of Jacobian 
		invert_matrix_3D(J_3du, J_3du_inv);

        //Update ucov_tilde
        for (i=0;i<3;i++) vD[i] = vD[i] - (J_3du_inv[i][0]*f_3du[0]+ J_3du_inv[i][1] * f_3du[1]+ J_3du_inv[i][1] * f_3du[1]); //MATTHEW: This must be wrong; You want to divide residuals by jacobian
        
        //check convergence of ucov to exit loop
  		er = 0.0;
  		normV=0.0;
  		for(i=0;i<3;i++)for(j=0;j<3;j++){
			er += (ggammainv[i][j] * f_3du[i] * f_3du[j]);
			normV += (ggammainv[i][j] * vD[i] * vD[j]);
  		}
  		
        if ((er < tolerance) || (er/(normV+1.e-16) <= er1*tolerance)){
        	retval=0;
        	break; //solution found!!
        }
        retval=1;
        
		ii++;
	} while (ii < maxitnr); // End of the Newton cycle

	//Recompute electric field
	for (i=0;i<3;i++){
		B_D[i] = 0.;
		vU[i] = 0.;
		for (j=0;j<3;j++){
			B_D[i] = B_D[i] + ggammainv[i][j] * B_guess[j];
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}

	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//Get E and ucov_tilde
	getE_resistive(Enew, E_guess, vD, B_D, Dt_times_alpha, etares, ggammainv, sqrtgamma, lfac);

	//lower Enew 
	for (i = 0; i < 3; i++) {
		Enew_D[i] = 0.;
		for (j = 0; j < 3; j++) {
			Enew_D[i] = Enew_D[i] + ggamma[i][j] * Enew[j];
		}
	}

	//calculate bsqr and esqr
	bsqr = B_guess[0] * B_D[0] + B_guess[1] * B_D[1] + B_guess[2] * B_D[2];
	esqr = Enew[0] * Enew_D[0] + Enew[1] * Enew_D[1] + Enew[2] * Enew_D[2];

	//calculate tautilde and stilde
	tautilde = tau - (bsqr + esqr) * 0.5;

	for (i = 0; i < 3; i++){
		ExB[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			ExB[i] = ExB[i] + sqrtgamma * lvc3u(i, j, k) * Enew[j] * B_guess[k];
		}
	}
	for (i = 0; i < 3; i++) Stilde_j[i] = S[i] - ExB[i];

	//raise Stilde
	for (i = 0; i < 3; i++) {
		Stilde_uj[i] = 0.;
		for (j = 0; j < 3; j++) {
			Stilde_uj[i] = Stilde_uj[i] + ggammainv[i][j] * Stilde_j[j];
		}
	}

	//Calculate Stilde^2
	Ssqr = Stilde_uj[0] * Stilde_j[0] + Stilde_uj[1] * Stilde_j[1] + Stilde_uj[2] * Stilde_j[2];

	//compute z and epsilon (eq 58 in Ripperda et al 2019)
	z = sqrt(lfac * lfac - 1.0);
	eps = lfac * tautilde / D - z * sqrt(Ssqr) / D + z * z / (1.0 + lfac);

    //Update inverted quantities
    for (i=0;i<3;i++){
		rho[0] = D / lfac;
		ug[0]= rho[0] * eps;
		E_guess[i]=Enew[i];
		vD_guess[i]=vD[i];
    }
	  
	return retval;
}

//gives jacobian and residuals
void res_3du_der(double D, double Dt_times_alpha, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]){
	double Enew[3],Enew_D[3], B_D[3], dEdu[3], Stilde_j[3], ExB[3], ncov[NDIM], ncon[NDIM], ucon[NDIM], tautilde, vU[3], decrossb[3], Stilde_uj[3];
	double lfac, z, esqr, bsqr, Ssqr, eps, p, enth, edotde, depsdu, denthdu, dpdu, S_ujdotdecrossb;
	int i, j, k;

	//lower B and raise vD
	for (i=0;i<3;i++){
		B_D[i] = 0.;
		vU[i] = 0.;
		for (j=0;j<3;j++){
			B_D[i] = B_D[i] + ggamma[i][j] * B[j];
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}
	
	//calculate Lorentz factor
	lfac = sqrt(1.0+vU[0]*vD[0]+vU[1]*vD[1]+vU[2]*vD[2]);
	
	//calculate new electric field
	getdEdu_resistive(Enew,E,vD,B_D,Dt_times_alpha,etares,ggammainv,sqrtgamma,lfac,dEdu);
	
	//lower Enew 
	for (i=0;i<3;i++){
		Enew_D[i] = 0.;
		for (j = 0; j < 3; j++) {
			Enew_D[i] = Enew_D[i] + ggamma[i][j] * Enew[j];
		}
	}
	
	//calculate bsqr and esqr
	bsqr = B[0] * B_D[0] + B[1] * B_D[1] + B[2] * B_D[2];
	esqr = Enew[0] * Enew_D[0] + Enew[1] * Enew_D[1] + Enew[2] * Enew_D[2];
	
	//calculate tautilde and stilde
	tautilde = tau - (bsqr + esqr) * 0.5;
	
	for (i = 0; i < 3; i++){
		ExB[i] == 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			ExB[i] = ExB[i] + sqrtgamma * lvc3u(i, j, k) * Enew[j] * B[k];
		}
	}
	for (i=0;i<3;i++) Stilde_j[i] = S_j[i] - ExB[i]; 
	
	//raise Stilde
	for (i=0;i<3;i++){
		Stilde_uj[i] = 0.;
		for (j=0;j<3;j++){
			Stilde_uj[i]= Stilde_uj[i] + ggammainv[i][j]*Stilde_j[j];
		}
	}
	
	Ssqr = Stilde_uj[0]*Stilde_j[0]+ Stilde_uj[1]*Stilde_j[1]+ Stilde_uj[2]*Stilde_j[2];
	
	//compute pressure
	z = sqrt(lfac*lfac - 1.0);
	eps = lfac * tautilde / D - z * sqrt(Ssqr) / D + z * z / (1.0 + lfac);
	p = (GAMMA - 1.0) * D / lfac * eps;
	enth = 1.0 + GAMMA / (GAMMA - 1.0) * p / D * lfac;
	
	//compute residuals
	res[0] = vD[0] - Stilde_j[0] / D / enth; 
    res[1] = vD[1] - Stilde_j[1] / D / enth;
    res[2] = vD[2] - Stilde_j[2] / D / enth;
   
	//compute Jacobian
	//1-direction
	for (i = 0; i < 3; i++){
		decrossb[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++){
			if((j==k) || (j==i) || (k==i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[0, j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[0,0]+Enew_D[1]*dEdu[0,1]+Enew_D[2]*dEdu[0,2]; 
    depsdu = vU[0]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[0]/(1.0+lfac) - pow(z/(1.0+lfac),2)*vU[0]/lfac;
    
    if (z != 0.0) depsdu = depsdu - vU[0]/z*sqrt(Ssqr)/D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/pow(lfac,3)*vU[0]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[0]/lfac)/D;
    
	Jac[0][0] = 1.0 + decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
    Jac[1][0] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
    Jac[2][0] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) *denthdu;
	
	//2-direction	
	for (i = 0; i < 3; i++){
		decrossb[i] = 0.;
		for (j = 0; j < 3; j++) for (k = 0; k < 3; k++){
			if((j==k) || (j==i) || (k==i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[1, j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[1,0]+Enew_D[1]*dEdu[1,1]+Enew_D[2]*dEdu[1,2]; 
    depsdu = vU[1]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[1]/(1.0+lfac) - pow(z/(1.0+lfac),2.)*vU[1]/lfac;
    
	if (z != 0.0) depsdu = depsdu - vU[1] / z * sqrt(Ssqr) / D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/pow(lfac,3)*vU[1]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[1]/lfac)/D;
    
    Jac[0][1] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) *denthdu;
    Jac[1][1] = 1.0 + decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
    Jac[2][1] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth)*denthdu;

	//3-direction
	for (i = 0; i < 3; i++){
		decrossb[i] == 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++){
			if((j==k) || (j==i) || (k==i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[2, j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[2,0]+Enew_D[1]*dEdu[2,1]+Enew_D[2]*dEdu[2,2]; 
    depsdu = vU[2]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[2]/(1.0+lfac) - pow(z/(1.0+lfac),2)*vU[2]/lfac;
    
    if (z != 0.0) depsdu = depsdu - vU[2]/z*sqrt(Ssqr)/D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/pow(lfac,3)*vU[2]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[2]/lfac)/D;
    
    Jac[0][2] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) *denthdu;
    Jac[1][2] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
	Jac[2][2] = 1.0 + decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;
}

//Recover E
void getE_resistive(double Enew[3], double E[3], double vD[3], double B_D[3], double Dt_times_alpha, double etares, double ggammainv[3][3], double sqrtgamma, double lfac){
	double vxbU[3],  vU[3], e0dotv;
	int i, j, k;
		
	//calculate gamma*v^i
	for (i = 0; i < 3; i++) {
		vU[i] = 0;
		for (j = 0; j < 3; j++) {
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}
	
	// ucov x B_D
	for (i=0; i < 3; i++){
		vxbU[i] = 0.;
		for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			vxbU[i] = vxbU[i] + (1.0 / sqrtgamma) * lvc3u(i, j, k) * vD[j] * B_D[k];
		}
	} 
	
	// ImEx: Eold_upper.ucov
	e0dotv = E[0] * vD[0] + E[1] * vD[1] + E[2] * vD[2];

	//eta<1 case
	for (i = 0; i < 3; i++) {
		Enew[i] = etares * E[i] / (etares + lfac * Dt_times_alpha) - Dt_times_alpha / (etares + lfac * Dt_times_alpha) * (vxbU[i] - etares * e0dotv / (etares * lfac + Dt_times_alpha) * vU[i]);
	}
}

//Recover E and DE/du
void getdEdu_resistive(double Enew[3], double E[3], double vD[3], double B_D[3], double Dt_times_alpha, double etares,double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]) {
	double vxbU[3], kxbU[3], ginvv[3], krond[3], vU[3];
	int i, j, k;
	double e0dotv,  denom1, denom2;

	//calculate gamma*v^i
	for (i = 0; i < 3; i++) {
		vU[i] = 0;
		for (j = 0; j < 3; j++) {
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}

	// ucov x B_D
	for (i = 0; i < 3; i++) {
		vxbU[i] = 0.;
		for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			vxbU[i] = vxbU[i] + (1.0 / sqrtgamma) * lvc3u(i, j, k) * vD[j] * B_D[k];
		}
	}

	// ImEx: Eold_upper.ucov
	e0dotv = E[0] * vD[0] + E[1] * vD[1] + E[2] * vD[2];

	//eta<1 case
	for (i = 0; i < 3; i++) Enew[i] = etares * E[i] / (etares + lfac * Dt_times_alpha) - Dt_times_alpha / (etares + lfac * Dt_times_alpha) * (vxbU[i] - etares * e0dotv / (etares * lfac + Dt_times_alpha) * vU[i]);

	denom1 = etares + lfac * Dt_times_alpha;
	denom2 = etares * lfac + Dt_times_alpha;

	// Derivative of u x B: dE/dv1
	krond[0] = 1.0;
	krond[1] = 0.0;
	krond[2] = 0.0;
	for (i = 0; i < 3; i++) {
		kxbU[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			kxbU[i] = kxbU[i] + (1.0 / sqrtgamma) * lvc3u(i, j, k) * krond[j] * B_D[k];
		}
	}

	//Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[0][i] = -E[i] * etares / pow(denom1, 2) * Dt_times_alpha * vU[0] / lfac - (-pow(Dt_times_alpha / denom1, 2) * vU[0] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) + Dt_times_alpha / denom1 * (kxbU[i] + etares * (-E[0] / denom2 * vU[i] + etares * e0dotv / pow(denom2, 2) * vU[0] / lfac * vU[i] - e0dotv / denom2 * ggammainv[0][i])));
	}

	//Derivative of u x B: dE/dv2
	krond[0] = 0.0;
	krond[1] = 1.0;
	krond[2] = 0.0;
	for (i = 0; i < 3; i++) {
		kxbU[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			kxbU[i] = kxbU[i] + (1.0 / sqrtgamma) * lvc3u(i, j, k) * krond[j] * B_D[k];
		}
	}

	// Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[1][i] = -E[i] * etares / pow(denom1, 2) * Dt_times_alpha * vU[1] / lfac - (-pow(Dt_times_alpha / denom1, 2) * vU[1] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) + Dt_times_alpha / denom1 * (kxbU[i] + etares * (-E[1] / denom2 * vU[i] + etares * e0dotv / pow(denom2, 2) * vU[1] / lfac * vU[i] - e0dotv / denom2 * ggammainv[1][i])));
	}

	// Derivative of u x B: dE/dv3
	krond[0] = 0.0;
	krond[1] = 0.0;
	krond[2] = 1.0;
	for (i = 0; i < 3; i++) {
		kxbU[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			kxbU[i] = kxbU[i] + (1.0 / sqrtgamma) * lvc3u(i, j, k) * krond[j] * B_D[k];
		}
	}

	// Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[2][i] = -E[i] * etares / pow(denom1, 2) * Dt_times_alpha * vU[2] / lfac - (-pow(Dt_times_alpha / denom1, 2) * vU[2] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) + Dt_times_alpha / denom1 * (kxbU[i] + etares * (-E[2] / denom2 * vU[i] + etares * e0dotv / pow(denom2, 2) * vU[2] / lfac * vU[i] - e0dotv / denom2 * ggammainv[2][i])));
	}
}