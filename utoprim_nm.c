#include "u2p_util.h"
#include "include.h"
#include "decs.h"

/* these variables need to be shared between the functions
Utoprim_1D, residual, and utsq */
extern double Bsq, QdotBsq, Qtsq, Qdotn, D, S[2], fel;
#pragma omp threadprivate(Bsq, QdotBsq, Qtsq, Qdotn, D, S, fel)

static int Utoprim_NM_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_HD], double S2[NPR_2T], double tolerance, int lim
#if (DO_YE)
	, double ye
#endif
);

//Newman inversion routine serving as backup for utoprim2d
int Utoprim_NM(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM],double gdet, double prim[NPR], double tolerance, int lim
	#if(TWO_T)
	, double fel_input
	#endif
){
	double U_tmp[NPR_U], prim_tmp[NPR_HD], S2[NPR_2T];
	int i, ret;
	double alpha;

	if (U[0] <= 0.) {
		return(-100);
	}

	//First update the primitive B-fields
	#pragma ivdep
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha
	U_tmp[RHO]= alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma ivdep
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	#pragma ivdep
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	#pragma ivdep
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];
	

	//Calculate entropy variable for 2T fluids to recover EOS gamma
	#if(TWO_T)
	S2[0] = U[ENTRE] / U[RHO];
	S2[1] = U[ENTRI] / U[RHO];
	fel = fel_input;
	#endif
	if (U[ENTRE] == 0.0 || U[ENTRI] == 0) fprintf(stderr, "U-error \n");

	ret = Utoprim_NM_calc(U_tmp, gcov, gcon, gdet, prim_tmp, S2, tolerance, lim
		#if (DO_YE)
		, prim[YE]
		#endif
	);

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		#pragma ivdep
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}

		#if(TWO_T)
		if (prim[ENTRE] == 0.0 || prim[ENTRI] == 0) fprintf(stderr, "0-error\n");
		prim[ENTRE] = S2[0];
		prim[ENTRI] = S2[1];
		#endif
	}

	#if(DOKTOT)
	prim[KTOT] = U[KTOT] / U[RHO];
	#endif
    
	return(ret);
}

static int Utoprim_NM_calc(double U[NPR_U], double gcov[NDIM][NDIM],double gcon[NDIM][NDIM], double gdet, double prim[NPR_HD], double S2[2], double tolerance, int lim
	#if (DO_YE)
	, double ye
	#endif
)
{
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, w,  gamma, gamma_eos, vsq, errx=10000.;
	int i;

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma ivdep
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower_g(Bcon, gcov, Bcov);
	#pragma ivdep
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise_g(Qcov, gcon, Qcon);

	Bsq = 0.;
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);
	Qdotn = Qcon[0] * ncov[0];

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn*Qdotn;

	//Start inversion scheme AKA Newman et al
	double a, d, z, phi, R, Wsq, p_array[MAX_NEWT_ITER], epsilon, p_old, p_new;
	int iter = 0;
	int iter_tot = 0;
	int set_variables = 0;
    
    #if DOHELM
    double xdens, xpres, xener, xenth;
    double p_temp[UU + 1];
    // Helmholtz EOS
    xdens = prim[RHO];
    // -- to get min. pressure for a given density, set T = T_min = 1e4 K
    eos_mode_rhotemp_pres_min (xdens, 
		#if (DO_YE)
		ye,
		#else
		1.0, 
		#endif
		&xpres);
    p_array[0] = xpres;
    #else
    // Ideal gas EOS
	#if(TWO_T)
	gamma_eos = calc_gamma_gas_conserved(S2, prim[RHO]);
	#else
	gamma_eos = GAMMA;
	#endif
    p_array[0] = (gamma_eos - 1.)*prim[UU];
    #endif
	
    p_new = p_array[0];
	d = 0.5*(Qtsq*Bsq - QdotBsq);
	//if (d < 1e-20) return(1); // DANAT : edited for d very small (< 1e-30)
	do{
		set_variables = 0;
		a = -Qdotn + p_new + 0.5*Bsq;
		phi = acos(1. / a*sqrt((27.*d) / (4.*a)));
		epsilon = a / 3. - 2. / 3.*a*cos(2. / 3.*phi + 2. / 3.*M_PI);
        if (d < 1e-20) {
            epsilon = a; // Danat: in case d = 0, epsilon = a
        }
		z = epsilon - Bsq;

		vsq = (Qtsq*z*z + QdotBsq*(Bsq + 2. * z)) / (z*z*pow(Bsq + z, 2.));
        
        // DANAT addition
        if (fabs(vsq) < 1e-15) vsq = 0.0;
        if (fabs(vsq) >= 1.0) return(1); // it gives rho0 = NaN, therefore HelmEOS fails
        
		Wsq = 1. / (1. - vsq);
		w = z * (1. - vsq);
		gamma = 1. / sqrt(1. - vsq);
		rho0 = U[RHO] / gamma; 

        #if (DOHELM)
        // Helmholtz EOS
        p_temp[RHO] = rho0;
        p_temp[UU] = w - rho0;
        eos_mode_rhow_pres_u (p_temp, &xpres, &u); // DI_helmT
        #else
        // Ideal gas EOS
			#if(TWO_T)
			gamma_eos = calc_gamma_gas_w(S2, rho0, w, fel);
			if (isnan(gamma_eos))gamma_eos = GAMMA;
			#else
			gamma_eos = GAMMA;
			#endif
		u = (w - rho0) / gamma_eos;
        #endif

		iter++;
		iter_tot++;
        
        #if DOHELM
        // Helmholtz EOS
        p_array[iter] = xpres;
		#else
        // Ideal gas EOS
		p_array[iter] = (gamma_eos - 1.)*u;
        #endif
        
        p_old = p_array[iter - 1];
		p_new = p_array[iter];
		if (iter >= 2) {
			R = (p_array[iter] - p_array[iter - 1]) / (p_array[iter - 1] - p_array[iter - 2] + 1e-20); // Danat: what if p_array[iter-1] = p_array[iter-2]? Added 1e-20 in the denominator

			if (R<1. && R>0.) {
				set_variables = 1;
				p_new = p_array[iter - 1] + (p_array[iter] - p_array[iter - 1]) / (1. - R);
				p_old = p_array[iter];
				iter = 0.;
				p_array[iter] = p_new;
			}
		}
		errx = fabs(p_new - p_old) / fabs(p_new + p_old);
	} while (errx > tolerance && iter_tot < MAX_NEWT_ITER);

	//Return in different ways depending on tolerance and minimum tolerance
	if (fabs(errx) > MIN_NEWT_TOL) return(1);

	if (set_variables == 1){
		a = -Qdotn + p_new + 0.5*Bsq;
		phi = acos(1. / a*sqrt((27.*d) / (4.*a)));
		epsilon = a / 3. - 2. / 3.*a*cos(2. / 3.*phi + 2. / 3.*M_PI);
        if (d < 1e-20) {
            epsilon = a; // Danat: in case d = 0, epsilon = a
        }
		z = epsilon - Bsq;

		vsq = (Qtsq*z*z + QdotBsq*(Bsq + 2. * z)) / (z*z*pow(Bsq + z, 2.));
        // DANAT addition
        if (fabs(vsq) < 1e-15) vsq = 0.0;
		Wsq = 1. / (1. - vsq);
		w = z / Wsq;
		if (vsq >= 1.0 || vsq<0. || z <= 0. || z > W_TOO_BIG || !isfinite(vsq) || !isfinite(z)) {
			return(4);
		}
		gamma = sqrt(Wsq);
		rho0 = U[RHO] / gamma; 
        
        #if (DOHELM)
        // Helmholtz EOS
        p_temp[RHO] = rho0;
        p_temp[UU] = w - rho0;
        eos_mode_rhow_pres_u (p_temp, &p_new, &u);
        #else
		#if(TWO_T)
		gamma_eos = set_S_w(S2, rho0, w, fel);
		#else
		gamma_eos = GAMMA;
		#endif
        // Ideal gas EOS
        u = (w - rho0) / gamma_eos;
        p_new = (gamma_eos - 1.)*u;
        #endif
	}

	//If density or internal energy is negative return error code
	if ((rho0 < 0.0)) return(5);
	if ((p_new < 0.0) && (lim == BASIC)) return(6);

	prim[RHO] = rho0;
	prim[UU] = u;

	//Set 4-velocities
	#pragma ivdep
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (z + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / z);
	}

	/* done! */
    
	return(0);
}
