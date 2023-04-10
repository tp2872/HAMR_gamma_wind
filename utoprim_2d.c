
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
#include "include.h"
#include "decs.h"

/* these variables need to be shared between the functions
Utoprim_1D, residual, and utsq */
double Bsq, QdotBsq, Qtsq, Qdotn, D, S[2], fel;
#pragma omp threadprivate(Bsq, QdotBsq, Qtsq, Qdotn, D, S, fel)

// Declarations:
static double vsq_calc(double W);
static int Utoprim_new_body(double U[], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[], double tolerance, int lim
	#if (DO_YE)
	, double ye
	#endif
);
static int Utoprim_NM_calc(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_HD], double S2[NPR_2T], double tolerance, int lim
	#if (DO_YE)
	, double ye
	#endif
);
static int general_newton_raphson(double x[], void(*funcd) (double[], double[], double[], double[][NEWT_DIM_2], double *, double *
	#if (DOHELM_TEMPERATURE)
    , double*
	#endif
), 
	double tolerance
	#if (DOHELM_TEMPERATURE)
	, double* temp_prev
	#endif
);
static void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double *f, double *df
	#if (DOHELM_TEMPERATURE)
    , double* temp_prev
	#endif
);

static double x1_of_x0(double x0);
static double pressure_W_vsq(double W, double vsq);
static double dpdW_calc_vsq(double W, double vsq);
static double dpdvsq_calc(double W, double vsq);

/**********************************************************************/
/******************************************************************

Utoprim_2d():

-- Driver for new prim. var. solver.  The driver just translates
between the two sets of definitions for U and P.  The user may
wish to alter the translation as they see fit.


/  rho u^t           \
U =  |  T^t_\mu + rho u^t |  sqrt(-det(g_{\mu\nu}))
\   B^i              /

/    rho        \
P = |    uu         |
| \tilde{u}^i   |
\   B^i         /


Arguments:
U[NPR]    = conserved variables (current values on input/output);
gcov[NDIM][NDIM] = covariant form of the metric ;
gcon[NDIM][NDIM] = contravariant form of the metric ;
gdet             = sqrt( - determinant of the metric) ;
prim[NPR] = primitive variables (guess on input, calculated values on
output if there are no problems);

-- NOTE: for those using this routine for special relativistic MHD and are
unfamiliar with metrics, merely set
gcov = gcon = diag(-1,1,1,1)  and gdet = 1.  ;

******************************************************************/

int Utoprim_2d(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim
	#if(TWO_T)
	, double fel_input
	#endif
){
	double U_tmp[NPR_U],  prim_tmp[NPR_HD];
	int i, j, ret;
	double alpha;

	if (U[0] <= 0.) {
		return(-100);
	}

	//First update the primitive B-fields
	 #pragma ivdep
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0][0]);

	//Transform the CONSERVED variables into the new system
	U_tmp[RHO] = alpha * U[RHO] / gdet;
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
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	fel = fel_input;
	#endif

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
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
		//Set entropy variables
		#if(TWO_T)
		prim[ENTRE] = S[0];
		prim[ENTRI] = S[1];
		#endif
	}

	return(ret);
}


/**********************************************************************/
/**********************************************************************************

Utoprim_new_body():

-- Attempt an inversion from U to prim using the initial guess prim.

-- This is the main routine that calculates auxiliary quantities for the
Newton-Raphson routine.

-- assumes that
/  rho gamma        \
U = |  alpha T^t_\mu  |
\  alpha B^i        /



/    rho        \
prim = |    uu         |
| \tilde{u}^i   |
\  alpha B^i   /


return:  (i*100 + j)  where
i = 0 ->  Newton-Raphson solver either was not called (yet or not used)
or returned successfully;
1 ->  Newton-Raphson solver did not converge to a solution with the
given tolerances;
2 ->  Newton-Raphson procedure encountered a numerical divergence
(occurrence of "nan" or "+/-inf" ;

j = 0 -> success
1 -> failure: some sort of failure in Newton-Raphson;
2 -> failure: utsq<0 w/ initial p[] guess;
3 -> failure: W<0 or W>W_TOO_BIG
4 -> failure: v^2 > 1
5 -> failure: rho,uu <= 0 ;

**********************************************************************************/

static int Utoprim_new_body(double U[NPR_U], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_HD], double tolerance, int lim
	#if (DO_YE)
	, double ye
	#endif
)
{
	double x_2d[NEWT_DIM_2];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq, tmpdiff;
	int i, j, n, retval, i_increase;
	n = NEWT_DIM_2;

	//Assume ok initially:
	retval = 0;
	
	//Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma ivdep
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];
	lower_g(Bcon, gcov, Bcov);

	#pragma ivdep
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise_g(Qcov, gcon, Qcon);

	Bsq = 0.;
	/*#pragma ivdepreduction(+:Bsq)*/
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	//#pragma ivdepreduction(+:QdotB)
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);

	Qdotn = Qcon[0] * ncov[0];

	Qsq = 0.;
	//#pragma ivdepreduction(+:Qsq)
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	//calculate W from last timestep and use for guess
	utsq = 0.;
	for (i = 1; i < 4; i++) {
		for (j = 1; j < 4; j++) utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];
	}

	if ((utsq < 0.) && (fabs(utsq) < 1.0e-13)) {
		utsq = fabs(utsq);
	}
	if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
		retval = 2;
		return(retval);
	}

	gammasq = 1. + utsq;
	gamma = sqrt(gammasq);

	//Always calculate rho from D and gamma so that using D in EOS remains consistent; i.e. you don't get positive values for dP/d(vsq) . 
	rho0 = D / gamma;
	u = prim[UU];

    #if DOHELM
    // Helmholtz EOS
    prim[RHO] = rho0;
    eos_mode_rhou_pres (prim, &p);
    #else
	#if(TWO_T)
	double gamma_eos;
	gamma_eos = calc_gamma_gas_conserved(S, rho0);
	p = (gamma_eos - 1.0) * u;
	#else
    // Ideal gas EOS
    p = pressure_rho0_u(rho0, u);
	#endif
    #endif
    
	w = rho0 + u + p;
	W_last = w*gammasq;

	//Make sure that W is large enough so that v^2 < 1 : 
	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq)- QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq)) && (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	//Calculate W and vsq: 
	x_2d[0] = fabs(W_last);
	x_2d[1] = x1_of_x0(W_last);
	retval = general_newton_raphson(x_2d, func_vsq, tolerance
		#if(DOHELM_TEMPERATURE)
        , &prim[UU]
		#endif
    );
	W = x_2d[0];
	vsq = x_2d[1];

	//Problem with solver, so return denoting error before doing anything further
	if ((retval != 0) || (W == FAIL_VAL)) {
		retval = retval * 100 + 1;
		return(retval);
	}
	else{
		if (W <= 0. || W > W_TOO_BIG) {
			retval = 3;
			return(retval);
		}
	}

	// Calculate v^2:
	if (vsq >= 1.) {
		retval = 4;
		return(retval);
	}

	// Recover the primitive variables from the scalars and conserved variables:
	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;
	rho0 = D * gtmp;

	w = W * (1. - vsq);
    
    #if (DOHELM)
    // Helmholtz EOS
    prim[RHO] = rho0;
    prim[UU] = w - rho0;
    eos_mode_rhow_pres_u (prim, &p, &u); // DI_helmT
    #else
    // Ideal gas EOS
	#if(TWO_T)
	gamma_eos = set_S_w(S, rho0, w, fel);
	u = (w - rho0) / gamma_eos;
	p = (gamma_eos - 1.0) * u;
	#else
    p = pressure_rho0_w(rho0, w); // DANAT: change this for Helmholtz EOS! say, find p and u as f(rho0, w)
    u = w - (rho0 + p);
	#endif
    #endif
    
	// User may want to handle this case differently, e.g. do NOT return upon 
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.) ) {
		retval = 5;
		return(retval);
	}
	if ((u <= 0.) && (lim == BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#pragma ivdep
	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	#pragma ivdep
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);

	/* done! */
	return(retval);
}


/**********************************************************************/
/****************************************************************************
vsq_calc():

-- evaluate v^2 (spatial, normalized velocity) from
W = \gamma^2 w

****************************************************************************/
static double vsq_calc(double W){
	double Wsq, Xsq;

	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);

	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
}


/********************************************************************

x1_of_x0():

-- calculates v^2 from W  with some physical bounds checking;
-- asumes x0 is already physical
-- makes v^2 physical  if not;

*********************************************************************/

static double x1_of_x0(double x0){
	double x1, vsq;

	vsq = fabs(vsq_calc(x0)); // guaranteed to be positive 

	return((vsq > 1.) ? (1.0 - 1.e-15) : vsq);
}

/********************************************************************

validate_x():

-- makes sure that x[0,1] have physical values, based upon
their definitions:

*********************************************************************/

static void validate_x(double x[2], double x0[2]){
	/* Always take the absolute value of x[0] and check to see if it's too big:  */
	x[0] = fabs(x[0]);
	x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];


	x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
	x[1] = (x[1] > 1.) ? (1. - 1.e-15) : x[1];  /* if it's too big   */

	return;
}

/************************************************************

general_newton_raphson():

-- performs Newton-Rapshon method on an arbitrary system.

-- inspired in part by Num. Rec.'s routine newt();

*****************************************************************/
static int general_newton_raphson(double x[], void(*funcd) (double[], double[], double[], double[][NEWT_DIM_2], double *, double *
#if (DOHELM_TEMPERATURE)
    , double*
#endif
), double tolerance
#if (DOHELM_TEMPERATURE)
    , double* temp_prev
#endif
)
{
	double f, df, dx[NEWT_DIM_2], x_old[NEWT_DIM_2];
	double resid[NEWT_DIM_2], jac[NEWT_DIM_2][NEWT_DIM_2];
	double errx, x_orig[NEWT_DIM_2];
	int    n_iter, id,  i_extra, doing_extra;
	double  W, W_old;
	int  keep_iterating;

	//Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	
	#pragma ivdep
	for (id = 0; id < NEWT_DIM_2; id++)  x_old[id] = x_orig[id] = x[id];

	W = W_old = 0.;
	n_iter = 0;

	//Start the Newton-Raphson iterations
	keep_iterating = 1;
	while (keep_iterating) {
		//returns with new dx, f, df
		(*funcd) (x, dx, resid, jac, &f, &df
#if (DOHELM_TEMPERATURE)
                  , temp_prev
#endif

                  );

		//Save old values before calculating the new
		errx = 0.;
		#pragma ivdep
		for (id = 0; id < NEWT_DIM_2; id++) x_old[id] = x[id];
		#pragma ivdep
		for (id = 0; id < NEWT_DIM_2; id++) x[id] += dx[id];

		/****************************************/
		/* Make sure that the new x[] is physical : */
		/****************************************/
		validate_x(x, x_old);

		/****************************************/
		/* Calculate the convergence criterion */
		/****************************************/
		errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		/*****************************************************************************/
		/* If we've reached the tolerance level, then just do a few extra iterations */
		/*  before stopping                                                          */
		/*****************************************************************************/

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 1;

		if (doing_extra == 1) i_extra++;

		if (((fabs(errx) <= tolerance) && (doing_extra == 0))|| (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	} 

	//Check for bad untrapped divergences
	if ((isfinite(f) == 0) || (isfinite(df) == 0)) {
		return(2);
	}

	// Return in different ways depending on whether a solution was found:
	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance) return(0);

	return(0);
}



/**********************************************************************/
/*********************************************************************************
func_vsq():

-- calculates the residuals, and Newton step for general_newton_raphson();
-- for this method, x=W,vsq here;

Arguments:
x   = current value of independent var's (on input & output);
dx   = Newton-Raphson step (on output);
resid = residuals based on x (on output);
jac  = Jacobian matrix based on x (on output);
f    =  resid.resid/2  (on output)
df    = -2*f;  (on output)
n    = dimension of x[];
*********************************************************************************/

static void func_vsq(double x[], double dx[], double resid[], double jac[][NEWT_DIM_2], double *f, double *df
	#if (DOHELM_TEMPERATURE)
    , double* temp_prev
	#endif
)
{
	double  W, vsq, Wsq, p_tmp, dPdvsq, dPdW, temp, detJ, tmp2, tmp3;
	double t11, t16, t18, t2, t21, t23, t24, t25, t3, t35, t36, t4, t40, t9;

	//Set initial values
	W = x[0];
	vsq = x[1];
	Wsq = W*W;

    #if DOHELM
    // Helmholtz EOS
    double w = W * (1.0 - vsq);
    double rho = D * sqrt(1.0 - vsq);
    double gamma_sq = 1.0/(1.0 - vsq);
    double gamma = sqrt(gamma_sq);
    double dpdrho, dpde_d;
    double prim[UU + 1];
    prim[RHO] = rho;
    prim[UU] = w - rho;
    eos_mode_rhow_pres_dpdrho_dpde_d (prim, &p_tmp, &dpdrho, &dpde_d);
	#if (inversion_w_edits)
	// Danat: edit (DIMARK)
	double dudp = rho / dpde_d;
	dPdW = 1.0 / (1.0 + dudp) * (1.0 - vsq);
	dPdvsq = (-x[0] + 0.5 * D / sqrt((1.0 - vsq)) * (1. - dpdrho * dudp)) / (1. + dudp);
	#else 
    double dpdeps_o_rho = dpde_d / rho;
    double dpdvsq_1 = -0.5*D*gamma*dpdrho;
    double dpdvsq_2 = -0.5*(W + p_tmp*gamma_sq)/rho;
    dPdW = ( dpdeps_o_rho / (1.0 + dpdeps_o_rho) ) / gamma_sq;
    dPdvsq = (dpdvsq_1 + dpde_d * dpdvsq_2)/(1.0 + dpdeps_o_rho);
	#endif
	#elif(TWO_T)
	double gtmp, gamma_eos1, gamma_eos2, w, rho, dgamma, factor, dvsq, dW;

	//Temporary variables
	gtmp = 1. - vsq;
	w = W * gtmp;
	rho= D * sqrt(gtmp);
	gamma_eos1 = calc_gamma_gas_w(S, rho, w, fel);
	factor = (gamma_eos1 - 1.) / gamma_eos1;
	p_tmp = factor * (W * gtmp - D * sqrt(gtmp));

	//Offset sizes
	dW = 1.e-8 * rho;
	dvsq = MY_MIN(1.e-8, 1.0 - (vsq + 1.e-8));

	//Calculate dPdW
	gamma_eos2 = calc_gamma_gas_w(S, rho, (W + dW) * gtmp, fel);
	dgamma = (gamma_eos2 - gamma_eos1) / dW;
	dPdW = factor * gtmp + (W * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;

	//Calculate dPdvsq
	gamma_eos2 = calc_gamma_gas_w(S, D * sqrt(1.0 - (vsq + dvsq)), W * (1.0 - (vsq + dvsq)), fel);
	dgamma = (gamma_eos2 - gamma_eos1) / dvsq;
	dPdvsq = factor * (0.5 * D / sqrt(gtmp) - W) + (W * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;
    #else
    // Ideal gas EOS
	p_tmp = pressure_W_vsq(W, vsq);
	dPdW = dpdW_calc_vsq(W, vsq);
	dPdvsq = dpdvsq_calc(W, vsq);
    #endif

	// These expressions were calculated using Mathematica, but made into efficient 
	// code using Maple.  Since we know the analytic form of the equations, we can 
	// explicitly calculate the Newton-Raphson step: 

	t2 = -0.5*Bsq + dPdvsq;
	t3 = Bsq + W;
	t4 = t3*t3;
	t9 = 1 / Wsq;
	t11 = Qtsq - vsq*t4 + QdotBsq*(Bsq + 2.0*W)*t9;
	t16 = QdotBsq*t9;
	t18 = -Qdotn - 0.5*Bsq*(1.0 + vsq) + 0.5*t16 - W + p_tmp;
	t21 = 1 / t3;
	t23 = 1 / W;
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = t25*t3 + (Bsq - 2.0*dPdvsq)*(QdotBsq + vsq*Wsq*W)*t9*t23;
	t36 = 1 / t35;
	dx[0] = -(t2*t11 + t4*t18)*t21*t36;
	t40 = (vsq + t24)*t3;
	dx[1] = -(-t25*t11 - 2.0*t40*t18)*t21*t36;
	detJ = t3*t35;
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;

	*df = -resid[0] * resid[0] - resid[1] * resid[1];

	*f = -0.5 * (*df);
}


/**********************************************************************
**********************************************************************

The following routines specify the equation of state.  All routines
above here should be indpendent of EOS.  If the user wishes
to use another equation of state, the below functions must be replaced
by equivalent routines based upon the new EOS.

**********************************************************************
**********************************************************************/

/**********************************************************************/
/**********************************************************************
pressure_W_vsq():

-- Gamma-law equation of state;
-- pressure as a function of W, vsq, and D:
**********************************************************************/
static double pressure_W_vsq(double W, double vsq){
	double gtmp;
	gtmp = 1. - vsq;

	return((GAMMA - 1.) * (W * gtmp - D * sqrt(gtmp)) / GAMMA);
}


/**********************************************************************/
/**********************************************************************
dpdW_calc_vsq():

-- partial derivative of pressure with respect to W;
**********************************************************************/
static double dpdW_calc_vsq(double W, double vsq){
	return((GAMMA - 1.) * (1. - vsq) / GAMMA);
}

/**********************************************************************/
/**********************************************************************
dpdvsq_calc():

-- partial derivative of pressure with respect to vsq
**********************************************************************/
static double dpdvsq_calc(double W, double vsq){
	return((GAMMA - 1.) * (0.5 * D / sqrt(1. - vsq) - W) / GAMMA);
}


/******************************************************************************
END   OF   UTOPRIM_2D.C
******************************************************************************/


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
