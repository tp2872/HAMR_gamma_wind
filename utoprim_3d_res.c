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


int Utoprim_3d_res(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim, double Dt, double eta){
	double D, S[3], tau, B[3], E[3], ncov[NDIM], ncon[NDIM], prim_tmp[NPR_HD], U_tmp[NPR];
	int i, j, ret=0;
	double alpha, gamma, gammasq,utsq, dtt, etares, u, p, enth;
	double E_guess[3], gamma_guess, ucov_guess[3], ExB[3], xi_guess, gammainv[3][3];

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
	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	for (i=0;i<3; i++){
		 for (j=0;j<3;j++){
		 	gammainv[i][j] = gcon[i+1][j+1]+ncon[i+1]*ncon[j+1]; //gamma^{ij}
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
	for (i = 0; i < 3; i++) B[i] = alpha * U[B1+i] / gdet;

	//Electric field to 3+1
	#pragma ivdep
	for (i = 1; i < 3; i++) E[i] = alpha * U[E1+i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	#pragma ivdep
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	//ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim);
	
	/////////
	
	//calculate Lorentz factor
	utsq = 0. ;
  	for(i=1;i<4;i++){
  		for(j=1;j<4;j++){
  			 utsq += gcov[i][j]*prim[UTCON1+i-1]*prim[UTCON1+j-1] ;
  		}
  	}

	if( (utsq < 0.) && (fabs(utsq) < 1.0e-13) ) { 
    	utsq = fabs(utsq);
  	}
  	if(utsq < 0. || utsq > UTSQ_TOO_BIG) {
    	retval = 2;
    	//return(retval) ;
    //  fprintf(stderr,"failure, utsq_too_big at %d %d\n", i,j);
  	}

  	gammasq = 1. + utsq ;
  	gamma  = sqrt(gammasq);
  	etares=eta;
  	dtt=Dt; 
  	sqrtgamma=gdet/alpha; //determinant for spatial part of metric
  	
  	//guesses for NR
  	gamma_guess=gamma;
  	u = U[UU] ;
  	p = pressure_rho0_u(U[RHO],u) ;
  	enth=(1.0+u+p)/U[RHO];
  	xi_guess=D*enth*gamma_guess;
  	
  	for (i=0;i<3;i++){
  		E_guess[i]=E[i];
  	}
  	
  	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				ExB[i] = ExB[i] + sqrtgamma*levicivita(i,j,k)*E_guess[j]*B[k];
			}
		}
	}
	for (i=0;i<3;i++){
  		ucov_guess[i]=(S[i] - ExB[i])*gamma_guess/xi_guess; //gamma*v_i
  	}
	
	//NR Step
	invert_3DU(D, Dt, etares, tau, S, gcov, gcon, gdet, ucov, B, E, ucov_guess, gammainv, tolerance, ret);
	//you get back gamma*v_i and E
	////////
	
	//Construct primitive variables
	#pragma ivdep
	for (i = 1; i < 3; i++){
		 prim_tmp[E1+i]=gdet*E[i]/alpha;
	}
	
	//

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		 #pragma ivdep
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}

//gives back E and gamma*v_i
double invert_3DU(double D, double Dt, double etares, double tau, double S[3], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double ucov[NDIM], double B[3], double E[3], double vD_guess[3], double gammainv[3][3], double tolerance, int ret){
	double vDi[3],vDprev[3], xk_3du[3], J_3du[3][3], f_3du[3], Enew[3], B_D[3];
	int i,j,k, nit, maxitnr, ii;
	double er, er1, normV, half;
	maxitr=100;
	
	for (i=0;i<3; i++) vDi[i] = vD_guess[i];

    //Newton-Raphson iteration
    er = 1.0;
    er1 = 1.0;
    nit = 0;
    half=0.5; //ask Bart what this is
      
    for (i=0;i<3;i++) vDprev[i] = vDi[i];
	ii=1
    do{ // Start of the Newton cycle
    	nit = nit + 1;
        if(nit>maxitnr/2){
        	// mix the last  value for convergence
        	vDi=half*(vDi+vDprev); //half??
          	// relax accuracy requirement
          	er1=10.0*er1;
          	// following avoids decrease of accuracy requirement 
          	// *every* iteration step beyond maxitnr/2
          	nit = nit - maxitnr/10;
          	}
        
        //Compute residual and derivatives
        for (i=0;i<3;i++) xk_3du[i] = vDi[i];   
        
        //double res_3du_der(double D, double Dt, double etares, double tau, 
        //double S_j[3], double ucov[NDIM], double gcov[NDIM], double gcon[NDIM], 
        //double gdet, double B[3], double E[3], double Jac[3][3], double res[3])
        //(xk_3du,f_3du,3,J_3du)
        //(xk, res, n, J)
        res_3du_der(D, Dt, etares, tau, S, xk_3du, ucov, gcov, gcon, gdet, B, E, J_3du, f_3du); 


		//find inverse of Jacobian
        //Compute increments
        
        //////
        /*Call function for LU decomposition of J_3du*/
        /*Solve J*dx=F*/
        /////
        

        //Update u
        for (i=0;i<3;i++) vDprev[i] = vDi[i];
        
        //update ucov
        for (i=0;i<3;i++) vDi[i] = vDi[i] - f_3du[i];
        
        //check convergence of ucov to exit loop
  		er = 0.0;
  		normV=0.0;
  		for(i=0;i<3;i++){
  			for(j=0;j<3;j++){
  				er+=(gammainv[i][j]*f_3du[i]*f_3du[j]);
  				normV+=(gammainv[i][j]*vDi[i]*vDi[j]);
  			}
  		}
  		
        if ((er < tolerance) || (er/(normV+1.e-16) <= er1*tolerance)){
        	ret=0;
        	break; //solution found!!
        }
        ret=1;
        ii++;
      }while(ii<maxitnr); // End of the Newton cycle

      //Recompute electric field
      for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			B_D[i]    = B_D[i] + ggamma[i,j]*B[j];
		}
	}
      getE_resistive(Enew,E,vDi,B_D,Dt,etares,alpha,gammainv,sqrtgamma,mylfac,dEdu);
	
      /*call lowerraise(ggamma,enew,eD)
      esqr = dot_product(enew,eD)
      tautilde = tau - (bsqr + esqr)/2.d0
      {^C& ExB(^C) = get_exb(enew(1:^NC), b(1:^NC), sqrtgamma, ^C) \}
      stilde(1:^NC) = s(1:^NC) - ExB(1:^NC)
      call lowerraise(gammainv, stilde, sU)
      ssqr = dot_product(stilde,sU)

      //Recompute lfac, xi
      call lowerraise(gammainv, vDi, vU)
      lfaci = dsqrt(1.d0 + dot_product(vDi,vU))
      zi = dsqrt(lfaci**2 - 1.d0)
      eps = lfaci*tautilde/d - zi*dsqrt(ssqr)/d + zi**2/(1.d0+lfaci)
      p = (eqpar(gamma_)-1.d0) * d/lfaci * eps
      h = 1.d0 + eqpar(gamma_)/(eqpar(gamma_)-1.d0) * p/d*lfaci
      xii = d*lfaci*h*/
      
      //update E and ucov
      for (i=0;i<3;i++){
      
      	E[i]=Enew[i];
      	vD_guess[i]=vDi[i];
      }

}


//gives jacobian and residuals
double res_3du_der(double D, double Dt, double etares, double tau, double S_j[3], double myvD, double ucov[NDIM], double gcov[NDIM], double gcon[NDIM], double gdet, double B[3], double E[3], double Jac[3][3], double res[3]){
	
	double Enew[3],Enew_D[3], B_D[3], dEdu[3], Stilde_j[3], ExB[3], ncov[NDIM], ncon[NDIM], ucon[NDIM], ggamma[3][3], gammainv[3][3], tautilde, myvU[3];
	double alpha, mylfac, sqrtgamma, myz, esqr, bsqr, Ssqr, eps, p, enth, edotde, depsdu, denthdu, dpdu, S_ujdotdecrossb;
	alpha = 1.0 / sqrt(-gcon[0][0]);
	
	sqrtgamma=gdet/alpha; //determinant for spatial part of metric
	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	//raise_g(ucov, gcon, ucon);
	
	for (i=0;i<3; i++){
		 for (j=0;j<3;j++){
		 	ggamma[i][j]   = gcov[i+1][j+1]+ncov[i+1]*ncov[j+1]; //gamma_{ij}
		 	gammainv[i][j] = gcon[i+1][j+1]+ncon[i+1]*ncon[j+1]; //gamma^{ij}
		 }
	}
	//lower B and raise myvD
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			B_D[i]    = B_D[i] + ggamma[i,j]*B[j];
			myvU[i]   = myvU[i] + gammainv[i,j]*myvD[j];
		}
	}
	
	//calculate Lorentz factor
	//gamma_calc(pv,&geom,&mylfac);
	//mylfac = alpha*ucon[0];
	//or
	mylfac = pow(1.0+myvU[0]*myvD[0]+myvU[1]*myvD[1]+myvU[2]*myvD[2],0.5);
	
	
	//calculate new electric field
	
	//E_output,E_initial=e0, new ucov, B_i, Dt=timestep, resistivity, 
	//alpha, gamma^{ij}, determinant, lorentz factor, derivative as output  
      
	getE_resistive(Enew,E,myvD,B_D,Dt,etares,alpha,gammainv,sqrtgamma,mylfac,dEdu);
	
	//lower Enew 
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			Enew_D[i] = Enew_D[i] + ggamma[i,j]*Enew[j];
		}
	}
	
	//calculate bsqr and esqr
	bsqr= B[0]*B_D[0]+B[1]*B_D[1]+B[2]*B_D[2];
	esqr= Enew[0]*Enew_D[0]+Enew[1]*Enew_D[1]+Enew[2]*Enew_D[2];
	
	//calculate tautilde and stilde
	tautilde = tau - (bsqr + esqr)/2.0;
	
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				ExB[i] = ExB[i] + sqrtgamma*levicivita(i,j,k)*Enew[j]*B[k];
			}
		}
	}
	for (i=0;i<3;i++) Stilde_j[i] = S_j[i] - ExB[i]; 
	
	
	//raise Stilde
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			S_uj[i]= S_uj[i] + gammainv[i,j]*Stilde_j[j];
		}
	}
	
	Ssqr = S_uj[0]*Stilde_j[0]+S_uj[1]*Stilde_j[1]+S_uj[2]*Stilde_j[2];
	
	//compute pressure
	myz = pow(mylfac*mylfac - 1.0, 0.5);
    eps = mylfac*tautilde/D - myz*pow(Ssqr,0.5)/D + myz*myz/(1.0 + mylfac)
	p = (GAMMA-1.0) * D/mylfac * eps
    enth = 1.0 + GAMMA/(GAMMA-1.0) * p / D *mylfac
	
	
	//compute residuals
	
	res[0] = myvD[0] - Stilde_j[0] / D / enth; 
    res[1] = myvD[1] - Stilde_j[1] / D / enth;
    res[2] = myvD[2] - Stilde_j[2] / D / enth;

    
	//compute Jacobian
	
	//1-direction
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				decrossb[i] = decrossb[i] + sqrtgamma*levicivita(i,j,k)*dEdu[0,j]*B[k]
			}
		}
	}
	edotde = Enew_D[0]*dEdu[0,0]+Enew_D[1]*dEdu[0,1]+Enew_D[2]*dEdu[0,2]; 
    depsdu = myvU[0]/mylfac*tautilde/D - mylfac*edotde/D + 2.0*myvU[0]/(1.0+mylfac) - pow(myz/(1.0+mylfac),2)*myvU[0]/mylfac;
    
    if (myz != 0.0) depsdu = depsdu - myvU[0]/myz*pow(Ssqr,0.5)/D;
    
    S_ujdotdecrossb = S_uj[0]*decrossb[0]+S_uj[1]*decrossb[1]+S_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (myz/D)*S_ujdotdecrossb/pow(Ssqr, 0.5);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/mylfac - eps/pow(mylfac,3)*myvU[0]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*mylfac + p*myvU[0]/mylfac)/D;
    
    Jac[0,0] = 1.0 + decrossb[0]/D/enth + Stilde_j[0]/D/(enth*enth)*denthdu;
    Jac[1,0] = decrossb[1]/D/enth + Stilde_j[1]/D/(enth*enth)*denthdu;
    Jac[2,0] = decrossb[2]/D/enth + Stilde_j[2]/D/(enth*enth)*denthdu;
	
	//2-direction
	
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				decrossb[i] = decrossb[i] + sqrtgamma*levicivita(i,j,k)*dEdu[1,j]*B[k]
			}
		}
	}
	edotde = Enew_D[0]*dEdu[1,0]+Enew_D[1]*dEdu[1,1]+Enew_D[2]*dEdu[1,2]; 
    depsdu = myvU[1]/mylfac*tautilde/D - mylfac*edotde/D + 2.0*myvU[1]/(1.0+mylfac) - pow(myz/(1.0+mylfac),2)*myvU[1]/mylfac;
    
    if (myz != 0.0) depsdu = depsdu - myvU[1]/myz*pow(Ssqr,0.5)/D;
    
    S_ujdotdecrossb = S_uj[0]*decrossb[0]+S_uj[1]*decrossb[1]+S_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (myz/D)*S_ujdotdecrossb/pow(Ssqr, 0.5);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/mylfac - eps/pow(mylfac,3)*myvU[1]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*mylfac + p*myvU[1]/mylfac)/D;
    
    Jac[0,1] = decrossb[0]/D/enth + Stilde_j[0]/D/(enth*enth)*denthdu;
    Jac[1,1] = 1.0 + decrossb[1]/D/enth + Stilde_j[1]/D/(enth*enth)*denthdu;
    Jac[2,1] = decrossb[2]/D/enth + Stilde_j[2]/D/(enth*enth)*denthdu;

	
	//3-direction
	
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				decrossb[i] = decrossb[i] + sqrtgamma*levicivita(i,j,k)*dEdu[2,j]*B[k]
			}
		}
	}
	edotde = Enew_D[0]*dEdu[2,0]+Enew_D[1]*dEdu[2,1]+Enew_D[2]*dEdu[2,2]; 
    depsdu = myvU[2]/mylfac*tautilde/D - mylfac*edotde/D + 2.0*myvU[2]/(1.0+mylfac) - pow(myz/(1.0+mylfac),2)*myvU[2]/mylfac;
    
    if (myz != 0.0) depsdu = depsdu - myvU[2]/myz*pow(Ssqr,0.5)/D;
    
    S_ujdotdecrossb = S_uj[0]*decrossb[0]+S_uj[1]*decrossb[1]+S_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (myz/D)*S_ujdotdecrossb/pow(Ssqr, 0.5);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/mylfac - eps/pow(mylfac,3)*myvU[2]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*mylfac + p*myvU[2]/mylfac)/D;
    
    Jac[0,2] = decrossb[0]/D/h + Stilde_j[0]/D/(h*h)*denthdu;
    Jac[1,2] = decrossb[1]/D/h + Stilde_j[1]/D/(h*h)*denthdu;
    Jac[2,2] = 1.0 + decrossb[2]/D/h + Stilde_j[2]/D/(h*h)*denthdu;
	
}


//myvD->gamma*v_i
//getE_resistive(Enew,E,ucov, B_D,Dt,etares,alpha,gammainv,sqrtgamma,mylfac, dEdu);
double getE_resistive(double Enew[3], double E[3], double myvD[3], double B_D[3], double Dt, double etares, double alpha, double gammainv[3][3], double sqrtgamma, double lfac, double dEdu[3]){
	double vxbU[3], kcbU[3], enew[3], ginvv[3], krond[3], myvU[3];
	int i, j, k;
	double e0dotv, sigma, denom1, denom2;
	
	// ImEx: Eold_upper.ucov
	e0dotv=E[0]*myvD[0]+E[1]*myvD[1]+E[2]*myvD[2];
	
	//calculate gamma*v^i
	for (i=0;i<3;i++){
		for (j=0;j<3;j++){
			myvU[i]    = myvU[i] + gammainv[i,j]*myvD[j];
		}
	}
	
	// ucov x B_D
	for (i=0; i < 3; i++){
		for (j=0; j<3; j++){
			for (k=0; k<3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				vxbU[i] = vxbU[i] + (1.0/sqrtgamma)*levicivita(i,j,k)*myvD[j]*B_D[k];
			}
		}
	} 
	
	//eta<1 case
	sigma = Dt * alpha;
	for (i=0;i<3;i++) enew[i] = etares*E[i]/(etares + lfac*sigma) - sigma/(etares+lfac*sigma) * (vxbU[i] - etares*e0dotv/(etares*lfac+sigma)*myvU[i])l
	denom1 = etares + lfac*sigma;
    denom2 = etares*lfac + sigma;
    
    // dE/dv1
    // Build gamma^1i
    for  (i=0;i<3;i++) ginvv[i] = gammainv[0,i]
    // Derivative of k x B
    krond[0] = 1.0
    krond[1] = 0.0
    krond[2] = 0.0
    for (i=0; i < 3; i++){
		for (j=0; j<3; j++){
			for (k=0; k<3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				kxbU[i] = kxbU[i] + (1.0/sqrtgamma)*levicivita(i,j,k)*krond[j]*B_D[k];
			}
		}
	} 
    // Build derivative
    for (i=0;i<3;i++){
    
    	dEdu[0,i] = -E[i] * etares / pow(denom1,2) * sigma * myvU[0]/lfac 
                -(-pow(sigma/denom1,2)*myvU[0]/lfac * (vxbU[i] - etares*e0dotv/denom2*myvU[i]) 
                  +sigma/denom1 * (kxbU[i] 
                   +etares * (-E[0]/denom2*myvU[i] + etares*e0dotv/pow(denom2,2)*myvU[0]/lfac*myvU[i]
                    - e0dotv/denom2*ginvv[i])));
    
    
    }
    
    // dE/dv2
    // Build gamma^2i
    for  (i=0;i<3;i++) ginvv[i] = gammainv[1,i]
    // Derivative of u x B
    krond[0] = 0.0
    krond[1] = 1.0
    krond[2] = 0.0
    for (i=0; i < 3; i++){
		for (j=0; j<3; j++){
			for (k=0; k<3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				kxbU[i] = kxbU[i] + (1.0/sqrtgamma)*levicivita(i,j,k)*krond[j]*B_D[k];
			}
		}
	} 
    // Build derivative
    for (i=0;i<3;i++){
    
    	dEdu[1,i] = -E[i] * etares / pow(denom1,2) * sigma * myvU[1]/lfac 
                -(-pow(sigma/denom1,2)*myvU[1]/lfac * (vxbU[i] - etares*e0dotv/denom2*myvU[i]) 
                  +sigma/denom1 * (kxbU[i] 
                   +etares * (-E[1]/denom2*myvU[i] + etares*e0dotv/pow(denom2,2)*myvU[1]/lfac*myvU[i]
                    - e0dotv/denom2*ginvv[i])));
    
    
    }
    
    // dE/dv3
    // Build gamma^3i
    for  (i=0;i<3;i++) ginvv[i] = gammainv[2,i]
    // Derivative of u x B
    krond[0] = 0.0
    krond[1] = 0.0
    krond[2] = 1.0
    for (i=0; i < 3; i++){
		for (j=0; j<3; j++){
			for (k=0; k<3; k++){
				if((j==k) || (j==i) || (k==i)) continue;
				kxbU[i] = kxbU[i] + (1.0/sqrtgamma)*levicivita(i,j,k)*krond[j]*B_D[k];
			}
		}
	} 
    // Build derivative
    for (i=0;i<3;i++){
    
    	dEdu[2,i] = -E[i] * etares / pow(denom1,2) * sigma * myvU[2]/lfac 
                -(-pow(sigma/denom1,2)*myvU[2]/lfac * (vxbU[i] - etares*e0dotv/denom2*myvU[i]) 
                  +sigma/denom1 * (kxbU[i] 
                   +etares * (-E[2]/denom2*myvU[i] + etares*e0dotv/pow(denom2,2)*myvU[2]/lfac*myvU[i]
                    - e0dotv/denom2*ginvv[i])));
    
    
    }
    
}



static int Utoprim_new_body(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet,  FTYPE prim[NPR]){

  FTYPE x_2d[NEWT_DIM];
  FTYPE QdotB,Bcon[NDIM],Bcov[NDIM],Econ[NDIM],Ecov[NDIM],Qcov[NDIM],Qcon[NDIM],ncov[NDIM],ncon[NDIM],Qsq,Qtcon[NDIM];
  FTYPE Sj[NDIM],tau
  FTYPE rho0,u,p,w,gammasq,gamma,gtmp,W_last,W,utsq,vsq,tmpdiff ;
    FTYPE alpha, ucovt, utsqp1, aco, bco, cco, pevar, agame, the;


  int i,j, n, retval, i_increase ;
    double dummy;
    



  n = NEWT_DIM ;

  // Assume ok initially:
  retval = 0;

  for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] ;

  // Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
  Bcon[0] = 0. ;
  for(i=1;i<4;i++) Bcon[i] = U[BCON1+i-1] ;

  lower_g(Bcon,gcov,Bcov) ;
  for(i = ECON1; i <= ECON3; i++) prim[i] = U[i] ;

  Econ[0] = 0. ;
  for(i=1;i<4;i++) Econ[i] = U[ECON1+i-1] ;

  lower_g(Econ,gcov,Ecov) ;



  Bsq = 0. ;
  for(i=1;i<4;i++) Bsq += Bcon[i]*Bcov[i] ;
  Esq = 0. ;
  for(i=1;i<4;i++) Esq += Econ[i]*Ecov[i] ;
  
  //QdotB = 0. ;
  //for(i=0;i<4;i++) QdotB += Qcov[i]*Bcon[i] ;
  //QdotBsq = QdotB*QdotB ;

  ncov_calc(gcon,ncov) ;
  raise_g(ncov,gcon,ncon);

  //Qdotn = Qcon[0]*ncov[0] ;

  //Qsq = 0. ;
  //for(i=0;i<4;i++) Qsq += Qcov[i]*Qcon[i] ;

  //Qtsq = Qsq + Qdotn*Qdotn ;

  D = U[RHO] ;

  /* calculate W from last timestep and use for guess */
  utsq = 0. ;
  for(i=1;i<4;i++)
    for(j=1;j<4;j++) utsq += gcov[i][j]*prim[UTCON1+i-1]*prim[UTCON1+j-1] ;


  if( (utsq < 0.) && (fabs(utsq) < 1.0e-13) ) { 
    utsq = fabs(utsq);
  }
  if(utsq < 0. || utsq > UTSQ_TOO_BIG) {
    retval = 2;
    return(retval) ;
    //  fprintf(stderr,"failure, utsq_too_big at %d %d\n", i,j);
  }

  gammasq = 1. + utsq ;
  gamma  = sqrt(gammasq);
	
  // Always calculate rho from D and gamma so that using D in EOS remains consistent
  //   i.e. you don't get positive values for dP/d(vsq) . 
  rho0 = D / gamma ;
  u = prim[UU] ;
  p = pressure_rho0_u(rho0,u) ;
  w = rho0 + u + p ;

  W_last = w*gammasq ;
  
  xi=rho0*enth*gammasq ;
  tau =xi - p - D + Bsq/2.0 + Esq^2/2.0 ;



  // Make sure that W is large enough so that v^2 < 1 : 
  i_increase = 0;
  while( (( W_last*W_last*W_last * ( W_last + 2.*Bsq ) 
	    - QdotBsq*(2.*W_last + Bsq) ) <= W_last*W_last*(Qtsq-Bsq*Bsq))
	 && (i_increase < 10) ) {
    W_last *= 10.;
    i_increase++;
  }
  
  // Calculate W and vsq: 
  
  // iterate on ucon only but also calculate E after each iteration
  x_2d[0] =  fabs( W_last );
  x_2d[1] = x1_of_x0( W_last ) ;
  // NR step
  retval = general_newton_raphson( x_2d, n, func_vsq ) ;  

  //new value for ucon
  W = x_2d[0];
  vsq = x_2d[1];
	
  /* Problem with solver, so return denoting error before doing anything further */
  if( (retval != 0) || (W == FAIL_VAL) ) {
    //  fprintf(stderr,"failure, in solver at %d %d\n", i,j);
    retval = retval*100+1;
    return(retval);

  }
  else{
    if(W <= 0. || W > W_TOO_BIG) {
      //  fprintf(stderr,"failure, W_too_big at %d %d\n", i,j);
      retval = 3;
      return(retval) ;
    }
  }

  // Calculate v^2:
  if( vsq >= 1. ) {
    retval = 4;
    return(retval) ;
  }

  // Recover the primitive variables from the scalars and conserved variables:
  gtmp = sqrt(1. - vsq);
  gamma = 1./gtmp ;
  rho0 = D * gtmp;

  w = W * (1. - vsq) ;
  p = pressure_rho0_w(rho0,w) ;
  u = w - (rho0 + p) ;
  

  // User may want to handle this case differently, e.g. do NOT return upon 
  // a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
  if( (rho0 <= 0.) || (u <= 0.) ) { 
    retval = 5;
    return(retval) ;
  }

    prim[RHO] = rho0 ;
    prim[UU] = u ;
  
  for(i=1;i<4;i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
  for(i=1;i<4;i++) prim[UTCON1+i-1] = gamma/(W+Bsq) * ( Qtcon[i] + QdotB*Bcon[i]/W ) ;
    

	
  /* set field components */
  for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] ;


  /* done! */
  return(retval) ;

}


/************************************************************

  general_newton_raphson(): 

    -- performs Newton-Rapshon method on an arbitrary system.

    -- inspired in part by Num. Rec.'s routine newt();

*****************************************************************/
static int general_newton_raphson( FTYPE x[], int n, void (*funcd) (FTYPE [], FTYPE [], FTYPE [], FTYPE [][NEWT_DIM], FTYPE *, FTYPE *, int) ){
  FTYPE f, df, dx[NEWT_DIM], x_old[NEWT_DIM];
  FTYPE resid[NEWT_DIM], jac[NEWT_DIM][NEWT_DIM];
  FTYPE errx, x_orig[NEWT_DIM];
  int    n_iter, id, jd, i_extra, doing_extra;
  FTYPE dW,dvsq,vsq_old,vsq,W,W_old;

  int   keep_iterating;


  // Initialize various parameters and variables:
  errx = 1. ; 
  df = f = 1.;
  i_extra = doing_extra = 0;
  for( id = 0; id < n ; id++)  x_old[id] = x_orig[id] = x[id] ;

  vsq_old = vsq = W = W_old = 0.;
  n_iter = 0;

  /* Start the Newton-Raphson iterations : */
  keep_iterating = 1;
  while( keep_iterating ) { 

    (*funcd) (x, dx, resid, jac, &f, &df, n);  /* returns with new dx, f, df */
      

    /* Save old values before calculating the new: */
    errx = 0.;
    for( id = 0; id < n ; id++) {
      x_old[id] = x[id] ;
    }

    /* Make the newton step: */
    for( id = 0; id < n ; id++) {
      x[id] += dx[id]  ;
    }
    
    // 
	// UPDATE E using getE_resistive and get dEdu for Jacobian
	// 

    /****************************************/
    /* Calculate the convergence criterion */
    /****************************************/
    errx  = (x[0]==0.) ?  fabs(dx[0]) : fabs(dx[0]/x[0]);
    //if 2D method, make sure both W and vsq converge
    //this is important in non-relativistic flows where
    //vsq could be << 1
    if( n > 1 ) {
      errx  += (x[1]==0.) ?  fabs(dx[1]) : fabs(dx[1]/x[1]);
    }


    /****************************************/
    /* Make sure that the new x[] is physical : */
    /****************************************/
    validate_x( x, x_old ) ;
    


    /*****************************************************************************/
    /* If we've reached the tolerance level, then just do a few extra iterations */
    /*  before stopping                                                          */
    /*****************************************************************************/
    
    if( (fabs(errx) <= NEWT_TOL) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0) ) {
      doing_extra = 1;
    }

    if( doing_extra == 1 ) i_extra++ ;

    if( ((fabs(errx) <= NEWT_TOL)&&(doing_extra == 0)) 
	|| (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER-1)) ) {
      keep_iterating = 0;
    }
	n_iter++;

  }   // END of while(keep_iterating)


    /*  Check for bad untrapped divergences : */
  if( (finite(f)==0) ||  (finite(df)==0) ) {
    //   fprintf(stderr,"failure, untrapped divergences, %g %g %g %g \n", f, df, x, dx);
    return(2);
  }


  if( fabs(errx) > MIN_NEWT_TOL){
    return(1);
    //  fprintf(stderr,"failure, tolerance not reached \n");
  } 
  if( (fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > NEWT_TOL) ){
    return(0);
  }
  if( fabs(errx) <= NEWT_TOL ){
    return(0);
  }

  return(0);

}





