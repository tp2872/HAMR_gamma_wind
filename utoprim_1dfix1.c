
/*************************************************************************************/
/*************************************************************************************/
/*************************************************************************************

utoprim_1dfix1.c: 
---------------

  -- uses eq. (27) of Noble  et al. or the "momentum equation" and ignores
        the energy equation (29) in order to use the additional EOS;
  
    Uses the 1D_W method: 
       -- solves for one independent variable (W) via a 1D
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

#define LTRACE 0

/* these variables need to be shared between the functions
   Utoprim_1D, residual, and utsq */
double Bsq3,QdotBsq3,Qtsq3,Qdotn3,D_3, K_atm3 ;
double W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
#pragma omp threadprivate(W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old,Bsq3,QdotBsq3,Qtsq3,Qdotn3,D_3, K_atm3)

// Declarations: 
static double vsq_calc(double W);
static double u_of_p(double p);
static double pressure_of_rho(double rho0);
static int Utoprim_new_body(double U[], double gcov[NDIM][NDIM],  double gcon[NDIM][NDIM], double gdet,  double prim[], double tolerance, int lim);
static void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df);
static void func_1d_orig2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df);
static int general_newton_raphson( double x[],  void (*funcd) (double [], double [], double [], double [][NEWT_DIM_1], double *, double *), double tolerance);
static void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df);
static int gnr2( double x[], void (*funcd) (double [], double [], double [], double [][NEWT_DIM_1], double *, double *));

/**********************************************************************/
/******************************************************************

  Utoprim_1dfix1():

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

   -- different from Utoprim_2d() in that it also assumes an EOS of 
            P = K rho^\Gamma  or   P = K rho    
        depending on the value of G_ATM flag in u2p_defs.h

******************************************************************/

int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double tolerance, int lim
    #if(TWO_T)
    , double fel
    #endif
)
{
    double U_tmp[NPR_U], prim_tmp[NPR_HD];
    int i, j, ret;
    double alpha;

    if (U[0] <= 0.) {
        return(-100);
    }

    //First update the primitive B-fields
    for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

    //Set the geometry variables
    alpha = 1.0 / sqrt(-gcon[0][0]);

    //Transform the CONSERVED variables into the new system
    U_tmp[RHO] = alpha * U[RHO] / gdet;
    U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
    for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
    for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

    //Transform the PRIMITIVE variables into the new system
    for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

    #if(FULL_ENTROPY)
    K_atm3 = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
    #else
    K_atm3 = U[KTOT] / U[RHO];
    #endif

    ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim);

    /* Transform new primitive variables back if there was no problem : */
    if (ret == 0) {
        for (i = 0; i < BCON1; i++) {
            prim[i] = prim_tmp[i];
        }
    }
#if(DOFLR)
    prim[FLR] = U[FLR] / U[RHO];
#endif
#if (NEUTRON_STAR)
    prim[FLRFRAC] = U[FLRFRAC] / U[RHO];
#endif
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

static int Utoprim_new_body(double U[NPR_U], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet,  double prim[NPR_HD], double tolerance, int lim){
    double x_1d[1];
    double QdotB,Bcon[NDIM],Bcov[NDIM],Qcov[NDIM],Qcon[NDIM],ncov[NDIM],ncon[NDIM],Qsq,Qtcon[NDIM];
    double rho0,u,p,w,gammasq,gamma,gtmp,W_last,W,utsq,vsq,tmpdiff ;
    int i,j, retval=0, i_increase ;

    //Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
    Bcon[0] = 0. ;
    for(i=1;i<4;i++) Bcon[i] = U[BCON1+i-1] ;
    lower_g(Bcon,gcov,Bcov) ;
    for(i=0;i<4;i++) Qcov[i] = U[QCOV0+i] ;
    raise_g(Qcov,gcon,Qcon) ;
    Bsq3 = 0. ;
    for(i=1;i<4;i++) Bsq3 += Bcon[i]*Bcov[i] ;
    QdotB = 0. ;
    for(i=0;i<4;i++) QdotB += Qcov[i]*Bcon[i] ;
    QdotBsq3 = QdotB*QdotB ;
    ncov_calc(gcon,ncov) ;
    raise_g(ncov,gcon,ncon);
    Qdotn3 = Qcon[0]*ncov[0] ;
    Qsq = 0. ;
    for(i=0;i<4;i++) Qsq += Qcov[i]*Qcon[i] ;
    Qtsq3 = Qsq + Qdotn3*Qdotn3 ;

    D_3 = U[RHO] ;

    //calculate W from last timestep and use for guess
    utsq = 0. ;
    for (i = 1; i < 4; i++) {
        for (j = 1; j < 4; j++) utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];
    }

    //Assume machine precision issue
    if( (utsq < 0.) && (fabs(utsq) < 1.0e-13)) utsq = fabs(utsq);

    if(utsq < 0. || utsq > UTSQ_TOO_BIG) {
        retval = 2;
        return(retval) ;
    }

    gammasq = 1. + utsq ;
    gamma  = sqrt(gammasq);
	
    // Always calculate rho from D and gamma so that using D in EOS remains consistent; i.e. you don't get positive values for dP/d(vsq) . 
    rho0 = D_3 / gamma ;
    p = pressure_of_rho(rho0);
    u = u_of_p(p);
    w = rho0 + u + p ;
    
    W_last = w*gammasq ;

    //Make sure that W is large enough so that v^2 < 1 : 
    i_increase = 0;
    while(((W_last*W_last*W_last * (W_last + 2.*Bsq3) - QdotBsq3*(2.*W_last + Bsq3) ) <= W_last*W_last*(Qtsq3-Bsq3*Bsq3)) && (i_increase < 10) ) {
        W_last *= 10.;
        i_increase++;
    }
  
    W_for_gnr2 = W_for_gnr2_old = W_last;
    rho_for_gnr2 = rho_for_gnr2_old = rho0;

    //Calculate W: 
    x_1d[0] = W_last;
  
    #if(USE_ISENTROPIC)   
    retval = general_newton_raphson(x_1d, func_1d_orig1, tolerance);
    #else
    retval = general_newton_raphson(x_1d, func_1d_orig2, tolerance);
    #endif

    W = x_1d[0];

    //Problem with solver, so return denoting error before doing anything further
    if( (retval != 0) || (W == FAIL_VAL) ) {
        retval = retval*100+1;
        return(retval);
    }
    else{
        if(W <= 0. || W > W_TOO_BIG) {
            retval = 3;
            return(retval) ;
        }
    }

    // Calculate v^2 : 
    vsq = vsq_calc(W) ;
    if(vsq >= 1.) {
        retval = 4;
        return(retval) ;
    }

    // Recover the primitive variables from the scalars and conserved variables:
    gtmp = sqrt(1. - vsq);
    gamma = 1./gtmp ;
    rho0 = D_3 * gtmp;
    w = W * (1. - vsq) ;
    p = pressure_of_rho(rho0);
    u = u_of_p(p);

    // User may want to handle this case differently, e.g. do NOT return upon 
    // a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
    if((rho0 <= 0.)) { 
        retval = 5;
        return(retval) ;
    }

    if ((u <= 0.) && (lim == BASIC)) {
        retval = 6;
        return(retval);
    }

    prim[RHO] = rho0 ;
    prim[UU] = u ;

    for (i = 1; i < 4; i++) {
        Qtcon[i] = Qcon[i] + ncon[i] * Qdotn3;
        prim[UTCON1 + i - 1] = gamma / (W + Bsq3) * (Qtcon[i] + QdotB * Bcon[i] / W);
    }
	
    /* done! */
    return(retval) ;
}


/**********************************************************************/
/****************************************************************************
   vsq_calc(): 
    
      -- evaluate v^2 (spatial, normalized velocity) from 
            W = \gamma^2 w 

****************************************************************************/
static double vsq_calc(double W)
{
	double Wsq,Xsq;
	
	Wsq = W*W ;
	Xsq = (Bsq3 + W) * (Bsq3 + W);

	return(( Wsq * Qtsq3  + QdotBsq3 * (Bsq3 + 2.*W)) / (Wsq*Xsq));
}

/**********************************************************************/
/****************************************************************************
   dvsq_dW(): 
    
      -- evaluate the partial derivative of v^2 w.r.t. W

****************************************************************************/
static double dvsq_dW(double W)
{
	double W3,X3,X;
	
	X = Bsq3 + W;
	W3 = W*W*W ;
	X3 = X*X*X;

	return( -2.*( Qtsq3/X3  +  QdotBsq3 * (3*W*X + Bsq3*Bsq3) / ( W3 * X3 )  )  );
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
static int general_newton_raphson( double x[], void (*funcd) (double [], double [], double [], double [][NEWT_DIM_1], double *, double *), double tolerance){
    double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
    double errx, x_orig[NEWT_DIM_1];
    int    n_iter, id, jd, i_extra, doing_extra;
    double dW,dvsq,vsq_old,vsq,W,W_old;
    int   keep_iterating, i_increase;

    //Initialize various parameters and variables:
    errx = 1. ; 
    df = f = 1.;
    i_extra = doing_extra = 0;
    x_old[0] = x_orig[0] = x[0] ;
    vsq_old = vsq = W = W_old = 0.;
    n_iter = 0;

    //Start the Newton-Raphson iterations
    keep_iterating = 1;
    while( keep_iterating ) { 
        (*funcd) (x, dx, resid, jac, &f, &df);  /* returns with new dx, f, df */

        //Save old values before calculating the new: */
        errx = 0.;
        x_old[0] = x[0] ;

        //don't use line search : */
        x[0] += dx[0]  ;

        //METHOD specific:
        i_increase = 0;
        while(((x[0]*x[0]*x[0] * (x[0] + 2.*Bsq3) - QdotBsq3*(2.*x[0] + Bsq3)) <= x[0]*x[0]*(Qtsq3-Bsq3*Bsq3)) && (i_increase < 10)){
            x[0] -= (1.*i_increase) * dx[0] / 10. ;
            i_increase++;
        }

        /****************************************/
      /* Make sure that the new x[] is physical : */
      /****************************************/
        x[0] = fabs(x[0]);

        /****************************************/
        /* Calculate the convergence criterion */
        /****************************************/
        errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);
      
        /*****************************************************************************/
        /* If we've reached the tolerance level, then just do a few extra iterations */
        /*  before stopping                                                          */
        /*****************************************************************************/
    
        if( (fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0) ) {
            doing_extra = 1;
        }

        if(doing_extra == 1) i_extra++ ;

        if(((fabs(errx) <= tolerance) &&(doing_extra == 0)) ||(i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER-1))) {
            keep_iterating = 0;
        }

        n_iter++;
    }

    //Check for bad untrapped divergences
    if((isfinite(f)==0) || (isfinite(df)==0) || (isfinite(x[0])==0)) {
        return(2);
    }

    //Depending on convergence return flag
    if(fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
    if((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
    if(fabs(errx) <= tolerance) return(0);

    return(0);
}


/***********************************************************/
/********************************************************************** 

  gnr2()

    -- used to calculate rho from W

*****************************************************************/
static int gnr2( double x[], void (*funcd) (double [], double [], double [], double [][NEWT_DIM_1],double *,double *))
{
    double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
    double errx, x_orig[NEWT_DIM_1];
    int n_iter, i_extra, doing_extra, keep_iterating;
    double dW,dvsq,vsq_old,vsq,W,W_old;

    //Initialize various parameters and variables:
    errx = 1. ; 
    df = f = 1.;
    i_extra = doing_extra = 0;
    x_old[0] = x_orig[0] = x[0] ;

    n_iter = 0;

    //Start the Newton-Raphson iterations
    keep_iterating = 1;
    while( keep_iterating ) { 
        (*funcd) (x, dx, resid, jac, &f, &df);  //returns with new dx, f, df

        //Save old values before calculating the new
        x_old[0] = x[0] ;

        //Make the newton step
        x[0] += dx[0] ;

        //Make sure that the new x[] is physical
        x[0] = fabs(x[0]);

        //Calculate the convergence criterion
        errx+= (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

        //If we've reached the tolerance level, then just do a few extra iterations
        if( (fabs(errx) <= NEWT_TOL2) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0) ) {
            doing_extra = 0;
        }

        if(doing_extra == 1) i_extra++ ;

        // See if we've done the extra iterations, or have done too many iterations:
        if( ((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER-1))) {
            keep_iterating = 0;
        }

        n_iter++;
    }  

    /*  Check for bad untrapped divergences : */
    if( (isfinite(f)==0) || (isfinite(df)==0) || (isfinite(x[0])==0)  ) {
        return(2);
    }

    // Return in different ways depending on whether a solution was found:
    if (fabs(errx) > MIN_NEWT_TOL2) return(1);
    if ((fabs(errx) <= MIN_NEWT_TOL2) && (fabs(errx) > NEWT_TOL2)) return(0);
    if (fabs(errx) <= NEWT_TOL2) return(0);

    return(0);
}



/**********************************************************************/
/*********************************************************************************
   func_1d_orig[1,2](): 

        -- calculates the residuals, and Newton step for general_newton_raphson();
        -- for this method, x=W here;

     Arguments:
          x   = current value of independent var's (on input & output);
         dx   = Newton-Raphson step (on output);
        resid = residuals based on x (on output);
         jac  = Jacobian matrix based on x (on output);
         f    =  resid.resid/2  (on output)
        df    = -2*f;  (on output)
         n    = dimension of x[];
 *********************************************************************************/
//isentropic version:   eq.  (27)
static void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df){
    int retval, ntries;
    double  Dc, t1, t10,  t2 ,  t21,  t23,  t26,  t29,  t3 ,  t30;
    double  t32,  t33,  t34,  t38,  t5 ,  t51,  t67, t8, W, x_rho[1], rho, rho_g ;
    double pressure;

    W  = x[0];
    W_for_gnr2_old = W_for_gnr2;
    W_for_gnr2 = W;

    // get rho from NR:
    rho_g = x_rho[0] = rho_for_gnr2;
  
    ntries = 0;
    while ((retval = gnr2(x_rho, func_gnr2_rho)) && ( ntries++ < 10 )) {
        rho_g *= 10.;
        x_rho[0] = rho_g;
    }

    rho_for_gnr2_old = rho_for_gnr2; 
    rho = rho_for_gnr2 = x_rho[0];

    Dc = D_3; //D
    t1 = Dc*Dc; //D*D
    t2 = QdotBsq3*t1; //QdotBsq*D*D
    t3 = t2*Bsq3; //QdotBsq*D*D*Bsq
    t5 = Bsq3*Bsq3; //Bsq*Bsq
    t8 = t1*Bsq3; //D*D*Bsq
    t10 = t1*W; //D*D*W
    t21 = W*W; //W*W
    t23 = rho*rho; //rho*rho
    t26 = 1/t1; //1/(D*D)
    resid[0] = (t3+(2.0*t2+((Qtsq3-t5)*t1+(-2.0*t8-t10)*W)*W)*W+(t5+(2.0*Bsq3+W)*W)*t21*t23)*t26/t21; //eq27
    t29 = t1*t1; //D*D*D*D
    t30 = QdotBsq3*t29;
    t32 = GAMMA*K_atm3; //GAMMA*kappa
    t33 = pow(rho,1.0*GAMMA);
    t34 = t32*t33; //GAMMA*kappa*rho^gamma
    t38 = t23 * t33;
    t51 = GAMMA*t1*K_atm3*t33;
    t67 = t21*W; //W*W*W
   
    jac[0][0] = -2.0*(t30*Bsq3*t34+(t30*t34 +((-t38*Bsq3*t32+Bsq3*GAMMA*t1*K_atm3*t33)*t1+(-t38*GAMMA*K_atm3+t51)*t1*W)*t21)*W +((-t3+(-t2+(-t8-t10)*t21)*W)*W+(-t5-Bsq3*W)*t67*t23)*t23)*t26/(t51-W*t23)/t67;
    dx[0] = -resid[0]/jac[0][0];
    *f = 0.5*resid[0]*resid[0];
    *df = -2. * (*f);

    return;

}

//isothermal version:  eq. (27)
static void func_1d_orig2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df){
  double Dc ,   t1 ,   t10,   t2 ,   t21,   t23,   t26,   t3 ,   t5 ,   t8, W, rho ;
  double pressure;

  W  = x[0];
  Dc = D_3;
  t1 = Dc*Dc;
  t2 = QdotBsq3*t1;
  t3 = t2*Bsq3;
  t5 = Bsq3*Bsq3;
  t8 = t1*Bsq3;
  t10 = t1*W;
  t21 = W*W;
  rho = t1 * ( 1. + GAMMA*K_atm3/(GAMMA-1.) ) / W;
  t23 = rho*rho;
  t26 = 1/t1;
  resid[0] = (t3+(2.0*t2+((Qtsq3-t5)*t1+(-2.0*t8-t10)*W)*W)*W+(t5+(2.0*Bsq3+W)*W)*t21*t23)*t26/t21;
  jac[0][0] = -2.0*(t3+(t2+(t8+t10)*t21)*W+(t5+Bsq3*W)*t21*t23)*t26/t21/W;

  dx[0] = -resid[0]/jac[0][0];

  *f = 0.5*resid[0]*resid[0];
  *df = -2. * (*f);

  return;
}

/**********************************************************************/
/*********************************************************************************
   func_gnr2_rho():

        -- residual/jacobian routine to calculate rho from W via the polytrope:

        W  =  ( 1 + GAMMA * K_atm * rho^(GAMMA-1)/(GAMMA-1) ) D^2 / rho

     Arguments:
          x   = current value of independent var's (on input & output);
         dx   = Newton-Raphson step (on output);
        resid = residuals based on x (on output);
         jac  = Jacobian matrix based on x (on output);
         f    =  resid.resid/2  (on output)
        df    = -2*f;  (on output)
         n    = dimension of x[];
 *********************************************************************************/
// for the isentropic version:   eq.  (27)
static void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df){
    double A, B, C, rho, W, B0;

    rho = x[0];
    A = D_3*D_3;
    B0 = A * GAMMA * K_atm3 ;
    B  =  B0 / (GAMMA - 1.);
    W = W_for_gnr2;
    C = pow( rho, GAMMA - 1. );


    resid[0] = rho*W - A - B*C ;	
    jac[0][0] = W  -  B0 * C / rho;

    dx[0] = -resid[0]/jac[0][0];
    *f = 0.5*resid[0]*resid[0];
    *df = -2. * (*f);

    return;
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
    return(K_atm3 * pow(rho0, GAMMA));
}

/* 
internal energy density as a function of the pressure
*/
static double u_of_p(double p){ 
    return(p / (GAMMA - 1.));
}


/****************************************************************************** 
             END   OF   UTOPRIM_1D.C
 ******************************************************************************/

