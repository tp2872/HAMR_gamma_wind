
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
int invert_3DU(double D, double sigma, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
void res_3du_der(double D, double sigma, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);
void getE_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac);
void getdEdu_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]);
int invert_3DU_entropy(double D, double sigma, double etares, double entropy, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
void res_3du_der_entropy(double D, double sigma, double etares, double entropy, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);

int Utoprim_3d_res(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim, double Dt){
	double D, tau, S[3], B_guess[3], E_guess[3], ncov[NDIM], ncon[NDIM], U_tmp[NPR], rho , ug;
	int i, j, k, retval = 1;
	double alpha,sqrtgamma;
	double vD_guess[3], ggamma[3][3], ggammainv[3][3];

	//Return if rho*gamma is negative
	if (U[0] <= 0.) return(-100);

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
	tau = ncov[0] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3]) / gdet - D;
	
	//Momentum to 3+1
	S[0] = -ncov[0] * (U[U1] + ncov[1] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3])) / gdet;
	S[1] = -ncov[0] * (U[U2] + ncov[2] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3])) / gdet;
	S[2] = -ncov[0] * (U[U3] + ncov[3] * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3])) / gdet;

	//Magnetic field to 3+1
	#pragma ivdep
	B_guess[0] = alpha * U[B1] / gdet;
	B_guess[1] = alpha * U[B2] / gdet;
	B_guess[2] = alpha * U[B3] / gdet;

	//Electric field to 3+1
	E_guess[0] = alpha * U[E1] / gdet;
	E_guess[1] = alpha * U[E2] / gdet;
	E_guess[2] = alpha * U[E3] / gdet;
	  	
	//Guess of relative 4-velocity: gamma*v_i-->vD_guess (eq. 57)
	for (i = 0; i < 3; i++) {
		vD_guess[i] = 0.0;
		for (j = 0; j < 3; j++) {
			vD_guess[i] += ggamma[i][j] * prim[U1+j];
		}
	}

	//NR Step, you get back gamma*v_i and E
	retval=invert_3DU(D, Dt*alpha, ETA, tau, S, ggamma, ggammainv, sqrtgamma, &rho, &ug, B_guess, E_guess, vD_guess, tolerance);

	//Backup entropy inversion
	#if(DOKTOT)
	if(retval!=0){
		#if(FULL_ENTROPY)
		double kappa = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
		#else
		double kappa = U[KTOT] / U[RHO];
		#endif
		//retval = invert_3DU_entropy(D, Dt * alpha, ETA, kappa, S, ggamma, ggammainv, sqrtgamma, &rho, &ug, B_guess, E_guess, vD_guess, tolerance);
	}
	#endif

	//Transform new primitive variables back if there was no problem
	if (retval == 0) {
		prim[RHO] = rho;
		prim[UU] = ug;
		for (i = 0; i < 3; i++) {
			prim[U1 + i] = 0.0;
			for (j = 0; j < 3; j++) {
				prim[U1 + i] += ggammainv[i][j] * vD_guess[j];
			}
		}
		prim[E1] = E_guess[0] / alpha;
		prim[E2] = E_guess[1] / alpha;
		prim[E3] = E_guess[2] / alpha;
	}
	else {
		double B_D[3];
		for (i = 0; i < 3; i++) {
			B_D[i] = 0.;
			for (j = 0; j < 3; j++) {
				B_D[i] = B_D[i] + ggamma[i][j] * B_guess[j];
			}
		}
		vD_guess[0] = 0.;
		vD_guess[1] = 0.;
		vD_guess[2] = 0.;

		getE_resistive(E_guess, E_guess, vD_guess, vD_guess, B_D, Dt * alpha, ETA, ggammainv, sqrtgamma, 1.0);
		prim[E1] = E_guess[0] / alpha;
		prim[E2] = E_guess[1] / alpha;
		prim[E3] = E_guess[2] / alpha;
	}

	//Update B fields regardless to preserve Div.B==0 regardless if inversion is succesful
	#pragma ivdep
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	return(retval);
}

//gives back E and gamma*v_i
int invert_3DU(double D, double sigma, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double *rho, double *ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance){
	double vD[3], vU[3], vDprev[3], xk_3du[3], J_3du[3][3], J_3du_inv[3][3], f_3du[3], Enew[3], B_D[3], dvd[3], lfac, vD_small[3];
	double Enew_D[3], Stilde_j[3], ExB[3], Stilde_uj[3], Ssqr, bsqr, esqr, tautilde, z, eps;
	int i,j,k, nit, ii;
	double er, er_small=100000000.0, er1,normV, half;
	int maxitr = 100;
	int retval = 1;
	int retval_matrix;
	int tag=0;

	int i1, j1;
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
        if(nit>maxitr/2){
        	// mix the last  value for convergence
			for (i = 0; i < 3; i++) vD[i] = 0.5 * (vD[i] + vDprev[i]);
          	
			// relax accuracy requirement
			er1 = 10.0 * er1;
          	
			// following avoids decrease of accuracy requirement every iteration step beyond maxitnr/2
         	nit = nit - maxitr/10;
		}
        
        //Compute residual and derivatives
		for (i = 0; i < 3; i++) xk_3du[i] = vD[i];
        
		//Calculate jacobian and residuals
        res_3du_der(D, sigma, etares, tau, S, xk_3du, ggamma, ggammainv, sqrtgamma, B_guess, E_guess, J_3du, f_3du); 

        //Store previous ucov_tilde
        for (i=0;i<3;i++) vDprev[i] = vD[i];
        
		//Find inverse of Jacobian 
		retval_matrix=invert_matrix_3D(J_3du, J_3du_inv);

		//Print error if jacobian is singular
		if (retval_matrix == 1) {
			for (i1 = 0; i1 < 3; i1++)for (j1 = 0; j1 < 3; j1++) fprintf(stderr, "Jac(%d, %d): %f \n", i1, j1, J_3du[i][j]);
			break;
		}

        //Update ucov_tilde
		for (i = 0; i < 3; i++) {
			dvd[i] = (J_3du_inv[i][0] * f_3du[0] + J_3du_inv[i][1] * f_3du[1] + J_3du_inv[i][2] * f_3du[2]);
			vD[i] = vD[i] - dvd[i];
		}

        //check convergence of ucov to exit loop
  		er = 0.0;
  		normV=0.0;
  		for(i=0;i<3;i++)for(j=0;j<3;j++){
			er += (ggammainv[i][j] * dvd[i] * dvd[j]);
			normV += (ggammainv[i][j] * vD[i] * vD[j]);
  		}

		if ((er < tolerance) || (er / (normV + 1.e-16) <= er1 * tolerance)) {
			retval = 0;
			break; //solution found!!
		}

		ii++;
	} while (ii < maxitr); // End of the Newton cycle
	
	//Recompute electric field
	for (i=0;i<3;i++){
		B_D[i] = 0.;
		vU[i] = 0.;
		for (j=0;j<3;j++){
			B_D[i] = B_D[i] + ggamma[i][j] * B_guess[j];
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}

	if (retval != 0 || ii==maxitr) {
		//fprintf(stderr, "N: %d, retval: %d error: %f \n", ii, retval, log10(fabs(er)));
		//fprintf(stderr, "Inversion failure! (err: %f normV: %f ug: %f lfac: %f, tag: %d \n", er, normV, ug[0], lfac, tag);
		retval = 1;
		return retval;
	}
	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//Exit if Lorent factor smaller than 1
	if (lfac < 1.0) {
		//fprintf(stderr, "Lfac failure! \n");
		retval = 3;
		return retval;
	}

	//Get E and ucov_tilde
	getE_resistive(Enew, E_guess, vU, vD, B_D, sigma, etares, ggammainv, sqrtgamma, lfac);

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
	rho[0] = D / lfac;
	ug[0] = rho[0] * eps;

	//Exit if density or internal energy drops below 0
	if (rho[0] < 0.) {
		fprintf(stderr,"Density dropped below 0 in resistive inversion \n");
		retval = 2;
		return retval;
	}
	if (ug[0] < 0.) {
		//fprintf(stderr, "Internal energy dropped below 0 in resistive inversion: (err: %f normV: %f ug: %f lfac: %f, tag: %d \n", er, normV, ug[0], lfac, tag);
		retval = 3;
		return retval;
	}

	//Set velocit and updated (implicit) electric field
    for (i=0;i<3;i++){
		E_guess[i]=Enew[i];
		vD_guess[i]=vD[i];
    }
	  
	return retval;
}

//gives jacobian and residuals
void res_3du_der(double D, double sigma, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]){
	double Enew[3],Enew_D[3], B_D[3], dEdu[3][3], Stilde_j[3], ExB[3], tautilde, vU[3], decrossb[3], Stilde_uj[3];
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
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);
	
	//calculate new electric field
	getdEdu_resistive(Enew, E, vU, vD, B_D, sigma, etares, ggammainv, sqrtgamma, lfac, dEdu);

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
		ExB[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			ExB[i] =  ExB[i] + sqrtgamma * lvc3u(i, j, k) * Enew[j] * B[k];
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
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[0][j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[0][0]+Enew_D[1]*dEdu[0][1]+Enew_D[2]*dEdu[0][2]; 
    depsdu = vU[0]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[0]/(1.0+lfac) - (z*z/((1.0+lfac)* (1.0 + lfac)))*vU[0]/lfac;
    
    if (z != 0.0) depsdu = depsdu - vU[0]/z*sqrt(Ssqr)/D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/(lfac*lfac*lfac)*vU[0]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[0]/lfac)/D;
    
	Jac[0][0] = 1.0 + decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
    Jac[1][0] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
    Jac[2][0] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) *denthdu;
	
	//2-direction	
	for (i = 0; i < 3; i++){
		decrossb[i] = 0.;
		for (j = 0; j < 3; j++) for (k = 0; k < 3; k++){
			if((j==k) || (j==i) || (k==i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[1][j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[1][0]+Enew_D[1]*dEdu[1][1]+Enew_D[2]*dEdu[1][2]; 
    depsdu = vU[1]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[1]/(1.0+lfac) - (z * z / ((1.0 + lfac) * (1.0 + lfac))) *vU[1]/lfac;
    
	if (z != 0.0) depsdu = depsdu - vU[1] / z * sqrt(Ssqr) / D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/(lfac*lfac*lfac)*vU[1]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[1]/lfac)/D;
    
    Jac[0][1] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) *denthdu;
    Jac[1][1] = 1.0 + decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
    Jac[2][1] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth)*denthdu;

	//3-direction
	for (i = 0; i < 3; i++){
		decrossb[i] == 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++){
			if((j==k) || (j==i) || (k==i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[2][j] * B[k];
		}
	}
	edotde = Enew_D[0]*dEdu[2][0]+Enew_D[1]*dEdu[2][1]+Enew_D[2]*dEdu[2][2]; 
    depsdu = vU[2]/lfac*tautilde/D - lfac*edotde/D + 2.0*vU[2]/(1.0+lfac) - (z * z / ((1.0 + lfac) * (1.0 + lfac))) *vU[2]/lfac;
    
    if (z != 0.0) depsdu = depsdu - vU[2]/z*sqrt(Ssqr)/D;
    
    S_ujdotdecrossb = Stilde_uj[0]*decrossb[0]+Stilde_uj[1]*decrossb[1]+Stilde_uj[2]*decrossb[2];
    if (Ssqr != 0.0) depsdu = depsdu + (z/D)*S_ujdotdecrossb/sqrt(Ssqr);
    
    dpdu = (GAMMA-1.0) * D * (depsdu/lfac - eps/(lfac*lfac*lfac)*vU[2]);
    denthdu = GAMMA/(GAMMA-1.0) * (dpdu*lfac + p*vU[2]/lfac)/D;
    
    Jac[0][2] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) *denthdu;
    Jac[1][2] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) *denthdu;
	Jac[2][2] = 1.0 + decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

}

//Recover E
void getE_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac){
	double vxbU[3], e0dotv;
	int i, j, k;
		
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
		Enew[i] = etares * E[i] / (etares + lfac * sigma) - sigma / (etares + lfac * sigma) * (vxbU[i] - etares * e0dotv / (etares * lfac + sigma) * vU[i]);
	}
}

//Recover E and DE/du
void getdEdu_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares,double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]) {
	double vxbU[3], kxbU[3],  krond[3];
	int i, j, k;
	double e0dotv,  denom1, denom2;

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
	for (i = 0; i < 3; i++) {	
		Enew[i] = etares * E[i] / (etares + lfac * sigma) - sigma / (etares + lfac * sigma) * (vxbU[i] - etares * e0dotv / (etares * lfac + sigma) * vU[i]);
	}
	denom1 = etares + lfac * sigma;
	denom2 = etares * lfac + sigma;

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
		dEdu[0][i] = -E[i] * etares / (denom1*denom1) * sigma * vU[0] / lfac 
			- (-sigma * sigma / (denom1 * denom1) * vU[0] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) 
				+ sigma / denom1 * (kxbU[i] 
					+ etares * (-E[0] / denom2 * vU[i] + etares * e0dotv / (denom2*denom2) * vU[0] / lfac * vU[i] - e0dotv / denom2 * ggammainv[0][i])));
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
		dEdu[1][i] = -E[i] * etares / (denom1*denom1) * sigma * vU[1] / lfac 
			- (-(sigma * sigma / (denom1 * denom1)) * vU[1] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) 
				+ sigma / denom1 * (kxbU[i] 
					+ etares * (-E[1] / denom2 * vU[i] + etares * e0dotv / (denom2*denom2) * vU[1] / lfac * vU[i] - e0dotv / denom2 * ggammainv[1][i])));
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
		dEdu[2][i] = -E[i] * etares / (denom1*denom1) * sigma * vU[2] / lfac 
			- (-(sigma * sigma / (denom1 * denom1)) * vU[2] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i]) 
				+ sigma / denom1 * (kxbU[i] 
					+ etares * (-E[2] / denom2 * vU[i] + etares * e0dotv / (denom2*denom2) * vU[2] / lfac * vU[i] - e0dotv / denom2 * ggammainv[2][i])));
	}
}

//gives back E and gamma*v_i
int invert_3DU_entropy(double D, double sigma, double etares, double kappa, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance) {
	double vD[3], vU[3], vDprev[3], xk_3du[3], J_3du[3][3], J_3du_inv[3][3], f_3du[3], Enew[3], B_D[3], dvd[3], lfac, vD_small[3];
	int i, j, k, nit, ii;
	double er, er_small = 100000000.0, er1, normV, half;
	int maxitr = 100;
	int retval = 1;
	int retval_matrix;
	int tag = 0;

	int i1, j1;
	for (i = 0; i < 3; i++) vD[i] = vD_guess[i];

	//Newton-Raphson iteration
	er = 1.0;
	er1 = 1.0;
	nit = 0;

	for (i = 0; i < 3; i++) vDprev[i] = vD[i];
	ii = 1;

	// Start of the Newton RAphson loop
	do {
		nit = nit + 1;
		if (nit > maxitr / 2) {
			// mix the last  value for convergence
			for (i = 0; i < 3; i++) vD[i] = 0.5 * (vD[i] + vDprev[i]);

			// relax accuracy requirement
			er1 = 10.0 * er1;

			// following avoids decrease of accuracy requirement every iteration step beyond maxitnr/2
			nit = nit - maxitr / 10;
		}

		//Compute residual and derivatives
		for (i = 0; i < 3; i++) xk_3du[i] = vD[i];

		//Calculate jacobian and residuals
		res_3du_der_entropy(D, sigma, etares, kappa, S, xk_3du, ggamma, ggammainv, sqrtgamma, B_guess, E_guess, J_3du, f_3du);

		//Store previous ucov_tilde
		for (i = 0; i < 3; i++) vDprev[i] = vD[i];

		//Find inverse of Jacobian 
		retval_matrix = invert_matrix_3D(J_3du, J_3du_inv);

		//Print error if jacobian is singular
		if (retval_matrix == 1) {
			for (i1 = 0; i1 < 3; i1++)for (j1 = 0; j1 < 3; j1++) fprintf(stderr, "Jac(%d, %d): %f \n", i1, j1, J_3du[i][j]);
			break;
		}

		//Update ucov_tilde
		for (i = 0; i < 3; i++) {
			dvd[i] = (J_3du_inv[i][0] * f_3du[0] + J_3du_inv[i][1] * f_3du[1] + J_3du_inv[i][2] * f_3du[2]);
			vD[i] = vD[i] - dvd[i];
		}

		//check convergence of ucov to exit loop
		er = 0.0;
		normV = 0.0;
		for (i = 0; i < 3; i++)for (j = 0; j < 3; j++) {
			er += (ggammainv[i][j] * dvd[i] * dvd[j]);
			normV += (ggammainv[i][j] * vD[i] * vD[j]);
		}

		if ((er < tolerance) || (er / (normV + 1.e-16) <= er1 * tolerance)) {
			retval = 0;
			break; //solution found!!
		}

		ii++;
	} while (ii < maxitr); // End of the Newton cycle

	//Recompute electric field
	for (i = 0; i < 3; i++) {
		B_D[i] = 0.;
		vU[i] = 0.;
		for (j = 0; j < 3; j++) {
			B_D[i] = B_D[i] + ggamma[i][j] * B_guess[j];
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}

	if (retval != 0 || ii == maxitr) {
		//fprintf(stderr, "N: %d, retval: %d error: %f \n", ii, retval, log10(fabs(er)));
		//fprintf(stderr, "Inversion failure! (err: %f normV: %f ug: %f lfac: %f, tag: %d \n", er, normV, ug[0], lfac, tag);
		retval = 1;
		return retval;
	}
	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//Exit if Lorent factor smaller than 1
	if (lfac < 1.0) {
		//fprintf(stderr, "Lfac failure! \n");
		retval = 3;
		return retval;
	}

	//Get E and ucov_tilde
	getE_resistive(Enew, E_guess, vU, vD, B_D, sigma, etares, ggammainv, sqrtgamma, lfac);

	//Update inverted quantities
	rho[0] = D / lfac;
	ug[0] = (kappa * pow(rho[0], GAMMA)) / (GAMMA - 1.0);

	//Exit if density or internal energy drops below 0
	if (rho[0] < 0.) {
		//fprintf(stderr, "Density dropped below 0 in resistive inversion \n");
		retval = 2;
		return retval;
	}
	if (ug[0] < 0.) {
		//fprintf(stderr, "Internal energy dropped below 0 in resistive inversion: (err: %f normV: %f ug: %f lfac: %f, tag: %d \n", er, normV, ug[0], lfac, tag);
		retval = 3;
		return retval;
	}

	//Set velocit and updated (implicit) electric field
	for (i = 0; i < 3; i++) {
		E_guess[i] = Enew[i];
		vD_guess[i] = vD[i];
	}

	return retval;
}

//gives jacobian and residuals
void res_3du_der_entropy(double D, double sigma, double etares, double kappa, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]) {
	double Enew[3], Enew_D[3], B_D[3], dEdu[3][3], Stilde_j[3], ExB[3], vU[3], decrossb[3];
	double lfac, p, enth, denthdu, dpdu;
	int i, j, k;

	//lower B and raise vD
	for (i = 0; i < 3; i++) {
		B_D[i] = 0.;
		vU[i] = 0.;
		for (j = 0; j < 3; j++) {
			B_D[i] = B_D[i] + ggamma[i][j] * B[j];
			vU[i] = vU[i] + ggammainv[i][j] * vD[j];
		}
	}

	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//calculate new electric field
	getdEdu_resistive(Enew, E, vU, vD, B_D, sigma, etares, ggammainv, sqrtgamma, lfac, dEdu);

	//lower Enew 
	for (i = 0; i < 3; i++) {
		Enew_D[i] = 0.;
		for (j = 0; j < 3; j++) {
			Enew_D[i] = Enew_D[i] + ggamma[i][j] * Enew[j];
		}
	}

	for (i = 0; i < 3; i++) {
		ExB[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			ExB[i] = ExB[i] + sqrtgamma * lvc3u(i, j, k) * Enew[j] * B[k];
		}
	}
	for (i = 0; i < 3; i++) Stilde_j[i] = S_j[i] - ExB[i];

	//compute z, pressure and enthalpy
	p = kappa * pow(D / lfac, GAMMA);
	enth = 1.0 + GAMMA / (GAMMA - 1.0) * p / D * lfac;

	//compute residuals
	res[0] = vD[0] - Stilde_j[0] / D / enth;
	res[1] = vD[1] - Stilde_j[1] / D / enth;
	res[2] = vD[2] - Stilde_j[2] / D / enth;

	//compute Jacobian
	//1-direction
	for (i = 0; i < 3; i++) {
		decrossb[i] = 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[0][j] * B[k];
		}
	}
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[0];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[0] / lfac) / D;

	Jac[0][0] = 1.0 + decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][0] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][0] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//2-direction	
	for (i = 0; i < 3; i++) {
		decrossb[i] = 0.;
		for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[1][j] * B[k];
		}
	}
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[1];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[1] / lfac) / D;

	Jac[0][1] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][1] = 1.0 + decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][1] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//3-direction
	for (i = 0; i < 3; i++) {
		decrossb[i] == 0.;
		for (j = 0; j < 3; j++)for (k = 0; k < 3; k++) {
			if ((j == k) || (j == i) || (k == i)) continue;
			decrossb[i] = decrossb[i] + sqrtgamma * lvc3u(i, j, k) * dEdu[2][j] * B[k];
		}
	}
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[2];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[2] / lfac) / D;

	Jac[0][2] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][2] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][2] = 1.0 + decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;
}