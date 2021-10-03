/*************************************************************************************

utoprim_1dvsq2fix1.c: 
---------------

  -- uses eq. (27) of Noble  et al. or the "momentum equation" and ignores
        the energy equation (29) in order to use the additional EOS;
  

    Uses the 1D^*_{v^2} method: 
       -- solves for one independent variable (v^2, or vsq) via a 1D 
          Newton-Raphson method 
       -- like the 1D_{v^2} method, except can be used (in principle)
          with a general equation of state. The main difference is how
          it calculates the intermediary value of W: by performing a 
          nested set of Newton-Rapshon iterations to get the best W value
          for the current v^2 value.  

  -- Currently returns with an error state (>0) if a negative rest-mass
      density or internal energy density is calculated.  You may want 
      to change this aspect of the code so that it still calculates the 
      velocity and so that you can floor the densities.  If you want to 
      change this aspect of the code please comment out the "return(retval)"
      statement after "retval = 5;" statement in Utoprim_new_body();

******************************************************************************/

#include "u2p_util.h"

#define LTRACE 0

/* these variables need to be shared between the functions
   Utoprim_1D, residual, and utsq */
double Bsq2,QdotBsq2,Qtsq2,Qdotn2,D_2, K_atm2 ;
#pragma omp threadprivate(Bsq2,QdotBsq2,Qtsq2,Qdotn2,D_2, K_atm2)

// Declarations: 
static double vsq_calc(double W);
double calc_gamma_gas_conserved(double* S, double rho);
static double W_of_vsq(double vsq, double *p, double *rho, double *u
    #if(TWO_T)
    , double* S
    , double fel
    #endif
);
static double u_of_p(double p);
static double pressure_of_rho(double rho0);
static double dWdvsq_calc(double vsq, double rho, double p);
static int Utoprim_new_body(double U[], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet,  double prim[], double tolerance, int lim
    #if(TWO_T)
    , double* S
    , double fel
    #endif
);
static void func_1d_gnr(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df
    #if(TWO_T)
    , double* S
    , double fel
    #endif
);
static int general_newton_raphson( double x[],  void (*funcd) (double [], double [], double [], double [][NEWT_DIM_1], double *, double *
    #if(TWO_T)
    , double*
    , double
    #endif
    ), double tolerance
    #if(TWO_T)
    , double* S
    , double fel
    #endif
);
void set_S_kappa(double rho, double K_atm, double* S, double fel);

/**********************************************************************/
/******************************************************************

  Utoprim_1dvsq2fix1():
  
  -- Driver for new prim. var. solver.  The driver just translates
     between the two sets of definitions for U and P.  The user may 
     wish to alter the translation as they see fit.  

     It assumes that on input/output:

              /  rho u^t           \
         U =  |  T^t_t   + rho u^t |  sqrt(-det(g_{\mu\nu}))  
              |  T^t_\mu           |  
              \   B^i              /


             /    rho        \
	 P = |    uu         |
             | \tilde{u}^i   |
             \   B^i         /

     ala HARM. 

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

   -- different from Utoprim_1dvsq2() in that it also assumes an EOS of 
            P = K rho^\Gamma  or   P = K rho    
        depending on the value of G_ATM flag in u2p_defs.h


******************************************************************/

int Utoprim_1dvsq2fix1(double U[NPR_U], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR_U], double tolerance, int lim
    #if(TWO_T)
    , double fel
    #endif
)
{
    double U_tmp[NPR_U], prim_tmp[NPR_U];
    int i, j, ret; 
    double alpha;
    #if(TWO_T)
    double S[2];
    #endif

    if( U[0] <= 0. ) { 
    return(-100);
    }

    //First update the primitive B-fields
    for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet ;

    //Set the geometry variables
    alpha = 1.0/sqrt(-gcon[0][0]);
  
    //Transform the CONSERVED variables into the new system
    U_tmp[RHO] = alpha * U[RHO] / gdet;
    U_tmp[UU]  = alpha * (U[UU] - U[RHO])/gdet ;
    for( i = UTCON1; i <= UTCON3; i++ ) U_tmp[i] = alpha * U[i] / gdet;
    for( i = BCON1; i <= BCON3; i++ ) U_tmp[i] = alpha * U[i] / gdet;

    //Transform the PRIMITIVE variables into the new system
    for( i = 0; i < BCON1; i++ ) {
        prim_tmp[i] = prim[i];
    }
    for( i = BCON1; i <= BCON3; i++ ) {
     prim_tmp[i] = alpha*prim[i];
    }

    #if(DOKTOT)
        #if(FULL_ENTROPY)
        K_atm2 = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
        #else
        K_atm2 = U[KTOT] / U[RHO];
        #endif
    #endif

    //Set electron and ion entropies
    #if(TWO_T)
    S[0] = U[ENTRE] / U[RHO];
    S[1] = U[ENTRI] / U[RHO];
    #endif

    ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
        #if(TWO_T)
        , S
        , fel
        #endif
    );

    //Transform new primitive variables back if there was no problem 
    if( ret == 0 ) {
        for(i = 0; i < BCON1; i++) {
            prim[i] = prim_tmp[i];
        }

        //Set entropy variables
        #if(TWO_T)
        prim[ENTRE] = S[0];
        prim[ENTRI] = S[1];
        #endif
    }

    return( ret ) ;
}


/**********************************************************************/
/**********************************************************************************

  Utoprim_new_body():

     -- Attempt an inversion from U to prim using the initial guess prim.

     -- This is the main routine that calculates auxiliary quantities for the 
        Newton-Raphson routine. 

  -- assumes that 
             /  rho gamma        \
         U = |  alpha T^t_\mu    |
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
    #if(TWO_T)
    , double* S
    , double fel
    #endif
) {
    double x_1d[1];
    double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
    double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq, tmpdiff;
    int i, j, retval = 0, retval2, i_increase;

    //Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
    Bcon[0] = 0.;
    for (i = 1; i < 4; i++) Bcon[i] = U[BCON1 + i - 1];
    lower_g(Bcon, gcov, Bcov);

    for (i = 0; i < 4; i++) Qcov[i] = U[QCOV0 + i];
    raise_g(Qcov, gcon, Qcon);

    Bsq2 = 0.;
    for (i = 1; i < 4; i++) Bsq2 += Bcon[i] * Bcov[i];

    QdotB = 0.;
    for (i = 0; i < 4; i++) QdotB += Qcov[i] * Bcon[i];
    QdotBsq2 = QdotB * QdotB;

    ncov_calc(gcon, ncov);
    raise_g(ncov, gcon, ncon);

    Qdotn2 = Qcon[0] * ncov[0];

    Qsq = 0.;
    for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];

    Qtsq2 = Qsq + Qdotn2 * Qdotn2;

    D_2 = U[RHO];

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

    gammasq = (1. + utsq);
    gamma = sqrt(gammasq);

    //Always calculate rho from D and gamma so that using D in EOS remains consistent; i.e. you don't get positive values for dP/d(vsq) . 
    rho0 = D_2 / gamma;

    #if(TWO_T)
    double gamma_g = calc_gamma_gas_conserved(S, prim[RHO]);
    u = prim[UU];
    p = (gamma_g - 1.) * u;
    #else
    // 2. Gamma EOS
    u = prim[UU];
    p = (GAMMA - 1.) * u;
    #endif

    // DANAT: add EOS p as function of rho0 and u
    w = rho0 + u + p;
    W_last = w * gammasq;

    //Initialize independent variables for Newton-Raphson:
    x_1d[0] = 1. - 1. / gammasq;

    //Find vsq via Newton-Raphson:
    retval = general_newton_raphson(x_1d, func_1d_gnr, tolerance
        #if(TWO_T)
        , S
        , fel
        #endif   
    );

    //Problem with solver, so return denoting error before doing anything further/
    if (retval != 0) {
        retval = retval * 100 + 1;
        return(retval);
    }

    //Calculate v^2 :
    vsq = x_1d[0];
    if ((vsq >= 1.) || (vsq < 0.)) {
        retval = 4;
        return(retval);
    }

    //Find W from this vsq:
    W = W_of_vsq(vsq, &p, &rho0, &u
        #if(TWO_T)
        , S
        , fel
        #endif
    );

    //Recover the primitive variables from the scalars and conserved variables:
    gtmp = sqrt(1. - vsq);
    gamma = 1. / gtmp;

    w = W * (1. - vsq);

    //Return for negative density or internal energy
    if ((rho0 <= 0.)) {
        retval = 5;
        return(retval);
    }

    if ((u <= 0.) && (lim == BASIC)){
        retval = 6;
        return(retval);
    }

    //Set primitive density and internal energy
    prim[RHO] = rho0;
    prim[UU] = u;
    #if(TWO_T)
    set_S_kappa(rho0, K_atm2, S, fel);
    #endif

    //Set relative 4-velocities
    for (i = 1; i < 4; i++) {
        Qtcon[i] = Qcon[i] + ncon[i] * Qdotn2;
        prim[UTCON1 + i - 1] = gamma / (W + Bsq2) * (Qtcon[i] + QdotB * Bcon[i] / W);
    }

    return(retval) ;
}
  
/**********************************************************************/
/********************************************************************

  validate_x(): 
           
    -- makes sure that x[0] is physical, based upon its definition;
    -- "corrects" it if it is not physical;

*********************************************************************/

static void validate_x(double x[1], double x0[1] ) 
{ 
  double small = 1.e-10;

  x[0] = (x[0] >= 1.0)    ?  ( 0.5*(x0[0] + 1.) )    : x[0];
  x[0] = (x[0] <  -small) ?  ( 0.5*x0[0] )           : x[0];
  x[0] = fabs(x[0]);

  return;
}


/**********************************************************************/
/************************************************************

  general_newton_raphson(): 

    -- performs Newton-Rapshon method on an arbitrary system.

    -- inspired in part by Num. Rec.'s routine newt();

    Arguements: 

       -- x[]   = set of independent variables to solve for;
       -- n     = number of independent variables and residuals;
       -- funcd = name of function that calculates residuals, etc.;

*****************************************************************/
static int general_newton_raphson( double x[], void (*funcd) (double [], double [], double [],  double [][NEWT_DIM_1], double *, double *
    #if(TWO_T)
    , double*
    , double 
    #endif
    ), double tolerance
    #if(TWO_T)
    , double* S
    , double fel
    #endif
){
      double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
      double errx, x_orig[NEWT_DIM_1];
      int    n_iter=0, i_extra, doing_extra;
      double W,W_old, rho,p,u;

      int   keep_iterating;

      //Initialize various parameters and variables:
      errx = 1. ; 
      df =  f = 1.;
      i_extra = doing_extra = 0;

      x_old[0] = x_orig[0] = x[0];

      W = W_old = 0.;

      //Start the Newton-Raphson iterations
      keep_iterating = 1;
      while( keep_iterating ) { 
            (*funcd) (x, dx, resid, jac, &f, &df
                #if(TWO_T)
                , S
                , fel
                #endif
           );  //returns with new dx, f, df

            //Save old values before calculating the new
            errx = 0.;
            x_old[0] = x[0];

            //Make the Newton step
            x[0] += dx[0];

            /****************************************/
            /* Make sure that the new x[] is physical : */
            /****************************************/
            validate_x(x, x_old );

            /****************************************/
            /* Calculate the convergence criterion */
            /****************************************/
            W_old = W;
            W = W_of_vsq(x[0], &p, &rho, &u
                #if(TWO_T)
                , S
                , fel
                #endif
            );

            errx = (W == 0.) ? fabs(W - W_old) : fabs((W - W_old) / W);
            errx += (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

            /*****************************************************************************/
            /* If we've reached the tolerance level, then just do a few extra iterations */
            /*   before stopping                                                         */
            /*****************************************************************************/
    
            if( (fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0) ) doing_extra = 1;

            if (doing_extra == 1) i_extra++;

            // See if we've done the extra iterations, or have done too many iterations:
            if(((fabs(errx) <= tolerance)&&(doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER-1))){
              keep_iterating = 0;
            }

            n_iter++;
      } 

      //Check for bad untrapped divergences
      if((isfinite(f)==0) || (isfinite(df)==0)) return(2);
      
      //Return in different ways depending on tolerance and minimum tolerance
      if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
      if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
      if (fabs(errx) <= tolerance) return(0);

      return(0);
}

/********************************************************************************/
/********************************************************************** 
   func_1d_gnr(): 

        -- calculates the residuals, and Newton step for general_newton_raphson();
        -- for this method, x=vsq here;

     Arguments:
          x   = current value of independent var's (on input & output);
         dx   = Newton-Raphson step (on output);
        resid = residuals based on x (on output);
         jac  = Jacobian matrix based on x (on output);
         f    =  resid.resid/2  (on output)
        df    = -2*f;  (on output)
         n    = dimension of x[];
 *********************************************************************************/

static void func_1d_gnr(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df
    #if(TWO_T)
    , double* S
    , double fel
    #endif
){
  double vsq,W,W0,Wsq,W3,dWdvsq , dpdrho, fact_tmp, rho, p, u  ;
  int retval, iters; 

  vsq = x[0];

  // Calculate best value for W given current guess for vsq: 
  W = W_of_vsq(vsq, &p, &rho, &u
        #if(TWO_T)
      , S
      , fel
        #endif
  );
  Wsq = W*W;
  W3 = W*Wsq;

  // Doing this assuming  P = (G-1) u :
  dWdvsq = dWdvsq_calc(vsq, rho, p);

  fact_tmp = (Bsq2 + W) ;

  resid[0] = Qtsq2  -  vsq * fact_tmp * fact_tmp  +  QdotBsq2 * ( Bsq2 + 2.*W ) / Wsq ; 
  jac[0][0] =  -fact_tmp * ( fact_tmp +   2. * dWdvsq * ( vsq + QdotBsq2/W3 ) ) ; 

  dx[0] = -resid[0]/jac[0][0];

  *f = 0.5*resid[0]*resid[0];
  *df = -2. * (*f);
}

/********************************************************************** 
 ********************************************************************** 
   
 The following routines specify the equation of state.  All routines 
  above here should be indpendent of EOS.  If the user wishes 
  to use another equation of state, the below functions must be replaced 
  by equivalent routines based upon the new EOS. 

 **********************************************************************
**********************************************************************/


/* 
pressure as a function of rho0 and w = rho0 + u + p 
this is used by primtoU and Utoprim_1D
*/
static double pressure_of_rho(double rho0){
    return(K_atm2 * pow(rho0, GAMMA));
}

/* 
internal energy density as a function of the pressure
*/
static double u_of_p(double p){
    return(p / (GAMMA - 1.));
}

/* 
W as a function of v^2
*/
static double W_of_vsq(double vsq, double *p, double *rho, double *u
    #if(TWO_T)
    , double* S
    , double fel
    #endif
){
    double gtmp;
    gtmp = (1. - vsq);
    rho[0] = D_2 * sqrt(gtmp);
    // 2. Gamma EOS
    #if(TWO_T)
        //Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
        double gamg, game, gami, pe, pi, T_e, T_i, T_g;

        #if(CONSTANTGAMMA)
        game = GAMMA;
        gami = GAMMA;
            #if(FULL_ENTROPY)
            T_e = fabs((game - 1.0) * exp(S[0] * pow(rho[0], game - 1.0)));
            T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho[0], gami - 1.0)));
            #else
            T_e = fabs(S[0] * pow(rho[0], game - 1.0));
            T_i = fabs(S[1] * pow(rho[0], gami - 1.0));
            #endif
        #elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
        game = GAMMAE;
        gami = GAMMA;
            #if(FULL_ENTROPY)
            T_e = fabs((game - 1.0) * exp(S[0] * pow(rho[0], game - 1.0)));
            T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho[0], gami - 1.0)));
            #else
            T_e = fabs(S[0] * pow(rho[0], game - 1.0));
            T_i = fabs(S[1] * pow(rho[0], gami - 1.0));
            #endif
        #elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
            #if(FULL_ENTROPY_VARGAMMA)
            T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
            T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * exp(S[1])), 2. / 3.)) - 1.0) / MU_I);
            #else
            T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
            T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho[0] * S[1]), 2. / 3.)) - 1.0) / MU_I);
            #endif
        #endif

        //Calculate gas pressures
		pe = T_e * rho[0];
		pi = T_i * rho[0];

		//Update internal energy of electrons
		p[0] = (pe + pi);

		//Limit temperature ratios
		if (pe > (1.0 - 0.5 * FLOOR_ENTROPY) * p[0]) pe = (1.0 - FLOOR_ENTROPY) * p[0];
		if (pe < 0.5 * FLOOR_ENTROPY * p[0]) pe = FLOOR_ENTROPY * p[0];
		pi = p[0] - pe;

		//Set temperature
		T_e = pe / rho[0];
		T_i = pi / rho[0];

        //Calculate the internal energy
        #if(CONSTANTGAMMA)
        u[0] = p[0] / (GAMMA - 1.0);
        #elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
        game = GAMMAE;
        gami = GAMMA;
        gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + T_i / T_e)) / ((T_i / T_e) * (game - 1.0) + 1.0 * (gami - 1.0));
        u[0] = p[0] / (gamg - 1.0);
        #elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        game = (10.0 + 20.0 * T_e * MU_E * MASS_RATIO) / (6.0 + 15.0 * T_e * MU_E * MASS_RATIO);
        gami = (10.0 + 20.0 * T_i * MU_I) / (6.0 + 15.0 * T_i * MU_I);
        gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + T_i / T_e)) / ((T_i / T_e) * (game - 1.0) + 1.0 * (gami - 1.0));
        u[0] = p[0] / (gamg - 1.0);
        #endif
    #else
    p[0] = K_atm2 * pow(rho[0], GAMMA);
    u[0] = p[0] / (GAMMA - 1.);
    #endif
    return((rho[0] + u[0] + p[0]) / gtmp);
}

/* 
dW/dvsq as a function of v^2, rho, p
*/
static double dWdvsq_calc(double vsq, double rho, double p){
    return((GAMMA * (2. - GAMMA) * p + (GAMMA - 1.) * rho) / (2. * (GAMMA - 1.) * (1. - vsq) * (1. - vsq)));
}

void set_S_kappa(double rho, double K_atm, double* S, double fel) {
    //Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
    double game, gami, p, pe, pi, T_e, T_i;

    #if(CONSTANTGAMMA)
    game = GAMMA;
    gami = GAMMA;
        #if(FULL_ENTROPY)
        T_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)));
        T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)));
        #else
        T_e = fabs(S[0] * pow(rho, game - 1.0));
        T_i = fabs(S[1] * pow(rho, gami - 1.0));
        #endif
    #elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
    game = GAMMAE;
    gami = GAMMA;
        #if(FULL_ENTROPY)
        T_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)));
        T_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)));
        #else
        T_e = fabs(S[0] * pow(rho, game - 1.0));
        T_i = fabs(S[1] * pow(rho, gami - 1.0));
        #endif
    #elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
        T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[0])), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
        T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * exp(S[1])), 2. / 3.)) - 1.0) / MU_I);
        #else
        T_e = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO));
        T_i = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * S[1]), 2. / 3.)) - 1.0) / MU_I);
        #endif
    #endif

    //Calculate gas pressures
    pe = T_e * rho;
    pi = T_i * rho;

    //Calculate ug from kappa
    #if(CONSTANTGAMMA)
    p = K_atm * pow(rho, GAMMA);
    #elif(FIXEDGAMMA || VARGAMMA)   //  // variable gamma: Sadowski+17 & Chael+19  
    p = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(rho * K_atm, 2. / 3.)) - 1.0) / MU_G) * rho;
    #endif

    //Update internal energy of electrons
    double factor = p / (pe + pi);
    pe *= factor;
    pi *= factor;

    if (pe > 0.99 * p) pe = 0.99 * p;
    if (pe < 0.01 * p) pe = 0.01 * p;
    pi = p - pe;

    //Set temperature
    T_e = pe / rho;
    T_i = pi / rho;

    //Calculate the internal energy
    #if(CONSTANTGAMMA || FIXEDGAMMA)
        #if(FULL_ENTROPY)
        S[0] = 1.0 / (game - 1.0) * log(pe * pow(rho, -game));
        S[1] = 1.0 / (gami - 1.0) * log(pi * pow(rho, -gami));
        #else
        S[0] = pe * pow(rho, -game);
        S[1] = pi * pow(rho, -gami);
        #endif
    #elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
        #if(FULL_ENTROPY_VARGAMMA)
        S[0] = log(pow(T_e * (MU_E * MASS_RATIO), 1.5) * pow(T_e * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
        S[1] = log(pow(T_i * MU_I, 1.5) * pow(T_i * MU_I + 0.4, 1.5) / rho);
        #else
        S[0] = pow(T_e * (MU_E * MASS_RATIO), 1.5) * pow(T_e * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
        S[1] = pow(T_i * MU_I, 1.5) * pow(T_i * MU_I + 0.4, 1.5) / rho;
        #endif
    #endif
}


/****************************************************************************** 
             END   OF   UTOPRIM_1DVSQ2FIX.C
 ******************************************************************************/
