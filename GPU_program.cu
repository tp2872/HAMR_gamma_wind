extern "C" {
#include "decsCUDA.h"
}
#include <cuda.h>
/*Whether or not to use the 3D version of the code*/
#define ThreeD (1)

/*Wheter to set floors in lab frame*/
#define ZAMO_FLOOR (1)

/*Set the pixel value of the frozen boundary*/
#define ibound 0

/*Whether or not to allow inflow for fluxes (see fix_flux())*/
#define INFLOW 0

/*Wheter or not to use the full dispersion relation*/
#define FULL_DISP (0)

/* whether or not to use Font's  adiabatic/isothermal prim. var. inversion method: */
#define DO_FONT_FIX (1)

/*Set grid parameters*/
#define BRAVO (1.0)
#define QUEBEC (2.0)
#define CHARLIE (0.55)
#define DELTA (3.0)

/*Set workgroup size*/
#define LOCAL_WORK_SIZE 64

/* use K(s)=K(r)=const. (G_ATM = GAMMA) of time or  T = T(r) = const. of time (G_ATM = 1.) */
#define USE_ISENTROPIC 1

/*Set the grid size for 2D and 3D*/
#define NG (2)
#define N1G (2)
#define N2G (2)
#define N3G ((ThreeD>0)?(NG):(0))
#define D1 (1)
#define D2 (1)
#define D3 ((ThreeD>0)?(1):(0))

/*Set the number of pixels for the polefix*/
#define POLEFIX 1*(1-D3)

/* your choice of floating-point data type */
#define FTYPE double
#define FTYPE2 double

/*Define variable quantities*/
#define NPG 5
#define NPR 8
#define NDIM 4

/** FIXUP PARAMETERS, magnitudes of rho and u, respectively, in the floor : **/
#define RHOMIN	(0.5*1.e-4)
#define UUMIN	(0.5*1.e-6)
#define RHOMINLIMIT (1.e-20)
#define UUMINLIMIT  (1.e-20)
#define POWRHO (2.50)
#define FLOORFACTOR (1.0)
#define BSQORHOMAX (9.*FLOORFACTOR)
#define BSQOUMAX (250.*FLOORFACTOR)
#define UORHOMAX (50.*FLOORFACTOR)

/* Max. value of gamma, the lorentz factor */
#define GAMMAMAX (50.)

#define AMD 0

/*Define misc. MACROs*/
#if AMD
#define dot(a,b) (mad(a[0],b[0],mad( a[1],b[1],mad( a[2],b[2] , a[3]*b[3]))))
#else
#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3])
#endif
#define delta(i,j) (((i) == (j)) ? 1. : 0.)
#define MY_MIN(fval1,fval2) ( ((fval1) < (fval2)) ? (fval1) : (fval2))
#define MY_MAX(fval1,fval2) ( ((fval1) > (fval2)) ? (fval1) : (fval2))

/* mnemonics for primitive vars; conserved vars */
#define RHO	(0)	
#define UU	(1)
#define U1	(2)
#define U2	(3)
#define U3	(4)
#define B1	(5)
#define B2	(6)
#define B3	(7)

/* mnemonics for dimensional indices */
#define TT	(0)     
#define RR	(1)
#define TH	(2)
#define PH	(3)

#define SMALL	(1.e-20)

/* use local lax-friedrichs or HLL flux:  these are relative weights on each numerical flux */
#define HLLF  (1.0)
#define LAXF  (0.0)

/* loop over all Space dimensions; first rank loop */
#define SLOOPA for(j=1;j<NDIM;j++)
/* loop over all Space dimensions; second rank loop */
#define SLOOP  for(j=1;j<NDIM;j++) for(k=1;k<NDIM;k++)
/* loop over Primitive variables */
#define PLOOP  for(k=0;k<NPR;k++)
/* loop over all Dimensions; second rank loop */
#define DLOOP  for(j=0;j<NDIM;j++) for(k=0;k<NDIM;k++)
/* loop over all Dimensions; first rank loop */
#define DLOOPA for(j=0;j<NDIM;j++)

/*Whether to move polar axis to a bit larger theta*/
#define COORDSINGFIX 1

/*Theta value where singularity is displaced to*/
#define SINGSMALL (1.E-20)

/* mnemonics for centering of grid functions */
#define FACE1	(0)	
#define FACE2	(1)
#define FACE3	(4)
#define CORN	(2)
#define CENT	(3)

/*Define Utoprim quantities*/
#define MAX_NEWT_ITER 30     /* Max. # of Newton-Raphson iterations for find_root_2D(); */
#define NEWT_TOL   1.0e-10    /* Min. of tolerance allowed for Newton-Raphson iterations */
#define MIN_NEWT_TOL  1.0e-10    /* Max. of tolerance allowed for Newton-Raphson iterations */
#define EXTRA_NEWT_ITER 2
#define NEWT_TOL2     1.0e-15      /* TOL of new 1D^*_{v^2} gnr2 method */
#define MIN_NEWT_TOL2 1.0e-10  /* TOL of new 1D^*_{v^2} gnr2 method */
#define W_TOO_BIG	1.e20	/* \gamma^2 (\rho_0 + u + p) is assumed to always be smaller than this.  Thisis used to detect solver failures */
#define UTSQ_TOO_BIG	1.e20    /* \tilde{u}^2 is assumed to be smaller than this.  Used to detect solver failures */
#define FAIL_VAL  1.e30    /* Generic value to which we set variables when a problem arises */
#define NUMEPSILON (2.2204460492503131e-16)

/*Set dimensions for Utoprim routines*/
#define NEWT_DIM 2
#define NEWT_DIM2 1

/* some mnemonics */
/* for primitive variables */
#define UTCON1 	2
#define UTCON2 	3
#define UTCON3 	4
#define BCON1	5
#define BCON2	6
#define BCON3	7

/* for conserved variables */
#define QCOV0	1
#define QCOV1	2
#define QCOV2	3
#define QCOV3	4

/* Adiabatic index used for the state equation */
#define GAMMA	(5./3.)  
#define G_ISOTHERMAL (1.)

#if( USE_ISENTROPIC ) 
#define G_ATM GAMMA
#else
#define G_ATM G_ISOTHERMAL
#endif

/*Declerations of functions for Utoprim*/
__device__ FTYPE vsq_calc(FTYPE W, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq);
__device__ int Utoprim_new_body(FTYPE U[], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[]);
__device__ int general_newton_raphson(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D);
__device__ void func_vsq(FTYPE[], FTYPE[], FTYPE[], FTYPE[][NEWT_DIM], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D);
__device__ FTYPE x1_of_x0(FTYPE x0, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq);
__device__ FTYPE W_of_vsq2(FTYPE vsq, FTYPE *p, FTYPE *rho, FTYPE *u, FTYPE D, FTYPE K_atm);
__device__ FTYPE u_of_p2(FTYPE p);
__device__ FTYPE pressure_of_rho2(FTYPE rho0,  FTYPE K_atm);
__device__ FTYPE dWdvsq_calc2(FTYPE vsq, FTYPE rho, FTYPE p );
__device__ int Utoprim_new_body2(FTYPE U[], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet,  FTYPE prim[],FTYPE K_atm);
__device__ void func_1d_gnr2(FTYPE x[], FTYPE dx[], FTYPE resid[], FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm);
__device__ void validate_x2(FTYPE x[1], FTYPE x0[1] ) ;
__device__ int general_newton_raphson2( FTYPE x[], int n,  FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm );
__device__ int Utoprim_1dvsq2fix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM],FTYPE gdet, FTYPE prim[NPR], FTYPE K );
__device__ void func_gnr2_rho(FTYPE x[], FTYPE dx[], FTYPE resid[], FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2);
__device__ void raise_g(FTYPE vcov[], FTYPE gcon[][NDIM], FTYPE vcon[]);
__device__ void lower_g(FTYPE vcon[], FTYPE gcov[][NDIM], FTYPE vcov[]);
__device__ void ncov_calc(FTYPE gcon[][NDIM],FTYPE ncov[]) ;
__device__ void bcon_calc_g(FTYPE prim[],FTYPE ucon[],FTYPE ucov[],FTYPE ncov[],FTYPE bcon[]); 
__device__ FTYPE pressure_rho0_u(FTYPE rho0, FTYPE u);
__device__ FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w);

__device__ void raise_g(FTYPE vcov[NDIM], FTYPE gcon[NDIM][NDIM], FTYPE vcon[NDIM])
{
  int i,j;
  #pragma unroll 8	
  for(i=0;i<NDIM;i++) {
    vcon[i] = 0. ;
	#pragma unroll 8	
    for(j=0;j<NDIM;j++) 
      vcon[i] += gcon[i][j]*vcov[j] ;
  }
  return ;
}

__device__ void lower_g(FTYPE vcon[NDIM], FTYPE gcov[NDIM][NDIM], FTYPE vcov[NDIM])
{
  int i,j;
  #pragma unroll 8	
  for(i=0;i<NDIM;i++) {
    vcov[i] = 0. ;
	#pragma unroll 8	
    for(j=0;j<NDIM;j++) 
      vcov[i] += gcov[i][j]*vcon[j] ;
  }
  return ;
}

__device__ void ncov_calc(FTYPE gcon[NDIM][NDIM],FTYPE ncov[NDIM]) 
{
  FTYPE lapse ;
  int i;

  lapse = sqrt(-1./gcon[0][0]) ;
  ncov[0] = -lapse ;
  #pragma unroll 8	
  for( i = 1; i < NDIM; i++) { 
    ncov[i] = 0. ;
  }
  return ;
}

__device__ void bcon_calc_g(FTYPE prim[NPR],FTYPE ucon[NDIM],FTYPE ucov[NDIM],FTYPE ncov[NDIM],FTYPE bcon[NDIM]) 
{
  FTYPE Bcon[NDIM] ;
  FTYPE u_dot_B ;
  FTYPE gamma ;
  int i ;

  // Bcon = \mathcal{B}^\mu  of the paper:
  Bcon[0] = 0. ;
  #pragma unroll 8	
  for(i=1;i<NDIM;i++) Bcon[i] = -ncov[0] * prim[BCON1+i-1] ;

  u_dot_B = 0. ;
  #pragma unroll 8	
  for(i=0;i<NDIM;i++) u_dot_B += ucov[i]*Bcon[i] ;

  gamma = -ucon[0]*ncov[0] ;
  #pragma unroll 8	
  for(i=0;i<NDIM;i++) bcon[i] = (Bcon[i] + ucon[i]*u_dot_B)/gamma ;
}

__device__ int gamma_calc_g(FTYPE *pr, FTYPE gcov[NDIM][NDIM], FTYPE *gamma)
{
  FTYPE utsq ;

  utsq =    gcov[1][1]*pr[UTCON1]*pr[UTCON1]
    + gcov[2][2]*pr[UTCON2]*pr[UTCON2]
    + gcov[3][3]*pr[UTCON3]*pr[UTCON3]
    + 2.*(  gcov[1][2]*pr[UTCON1]*pr[UTCON2]
	    + gcov[1][3]*pr[UTCON1]*pr[UTCON3]
	    + gcov[2][3]*pr[UTCON2]*pr[UTCON3]) ;

  if(utsq<0.0){
    if(fabs(utsq)>1.E-10){ // then assume not just machine precision
      return (1);
    }
    else utsq=fabs(utsq); // set floor
  }

  *gamma = sqrt(1. + utsq) ;
  return(0) ;
}


__device__ FTYPE pressure_rho0_u(FTYPE rho0, FTYPE u)
{
  return((GAMMA - 1.)*u) ;
}

__device__ FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w)
{
  return((GAMMA-1.)*(w - rho0)/GAMMA) ;
}

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(FTYPE x[], FTYPE dx[], FTYPE resid[], 
			 FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2)
{
  FTYPE A, B, C, rho, W, B0;
  
  A = D*D;
  B0 = A * GAMMA * K_atm ;
  B  =  B0 / (GAMMA - 1.);
  rho = x[0];
  W = W_for_gnr2;
  C = pow( rho, GAMMA - 1. );
  resid[0] = rho*W - A - B*C ;	
  jac[0][0] = W  -  B0 * C / rho;
  dx[0] = -resid[0]/jac[0][0];
  *f = 0.5*resid[0]*resid[0];
  *df = -2. * (*f);
  return;
}

__device__ int Utoprim_1dvsq2fix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K )
{
  FTYPE U_tmp[NPR], prim_tmp[NPR];
  int i, ret; 
  FTYPE alpha;

  if( U[0] <= 0. ) { 
    return(-100);
  }

  /* First update the primitive B-fields */
  #pragma unroll //3
  for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet ;

  /* Set the geometry variables: */
  alpha = 1.0/sqrt(-gcon[0][0]);
  
  /* Transform the CONSERVED variables into the new system */
  U_tmp[RHO] = alpha * U[RHO] / gdet;
  U_tmp[UU]  = alpha * (U[UU] - U[RHO])  / gdet ;
  #pragma unroll //3
  for( i = UTCON1; i <= UTCON3; i++ ) {
    U_tmp[i] = alpha * U[i] / gdet ;
  }
  #pragma unroll //3
  for( i = BCON1; i <= BCON3; i++ ) {
    U_tmp[i] = alpha * U[i] / gdet ;
  }

  /* Transform the PRIMITIVE variables into the new system */
  #pragma unroll //BCON1
  for( i = 0; i < BCON1; i++ ) {
    prim_tmp[i] = prim[i];
  }
  #pragma unroll //3
  for( i = BCON1; i <= BCON3; i++ ) {
    prim_tmp[i] = alpha*prim[i];
  }

  ret = Utoprim_new_body2(U_tmp, gcov, gcon, gdet, prim_tmp, K);

  /* Transform new primitive variables back if there was no problem : */ 
  if( ret == 0 ) {
	 #pragma unroll //BCON1
    for( i = 0; i < BCON1; i++ ) {
      prim[i] = prim_tmp[i];
    }
  }
  return( ret ) ;
}

__device__ int Utoprim_new_body2(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], 
			    FTYPE gcon[NDIM][NDIM], FTYPE gdet,  FTYPE prim[NPR], FTYPE K_atm)
{
  FTYPE x_1d[1];
  FTYPE QdotB,Bcon[NDIM],Bcov[NDIM],Qcov[NDIM],Qcon[NDIM],ncov[NDIM],ncon[NDIM],Qsq,Qtcon[NDIM];
  FTYPE rho0,u,p,gammasq,gamma,gtmp, W,utsq,vsq;
  int    i,j, retval;
  FTYPE Bsq, QdotBsq, Qtsq, Qdotn, D;

  // Assume ok initially:
  retval = 0 ;
  #pragma unroll //3
  for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] ;

  // Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
  Bcon[0] = 0. ;
  #pragma unroll //3
  for(i=1;i<4;i++) Bcon[i] = U[BCON1+i-1] ;

  lower_g(Bcon,gcov,Bcov) ;
  #pragma unroll //4
  for(i=0;i<4;i++) Qcov[i] = U[QCOV0+i] ;
  raise_g(Qcov,gcon,Qcon) ;

  Bsq = 0. ;
  #pragma unroll //3
  for(i=1;i<4;i++) Bsq += Bcon[i]*Bcov[i] ;

  QdotB = 0. ;
  #pragma unroll //4
  for(i=0;i<4;i++) QdotB += Qcov[i]*Bcon[i] ;
  QdotBsq = QdotB*QdotB ;

  ncov_calc(gcon,ncov) ;
  raise_g(ncov,gcon,ncon);

  Qdotn = Qcon[0]*ncov[0] ;

  Qsq = 0. ;
  #pragma unroll //4
  for(i=0;i<4;i++) Qsq += Qcov[i]*Qcon[i] ;

  Qtsq = Qsq + Qdotn*Qdotn ;

  D = U[RHO] ;

  /* calculate W from last timestep and use  for guess */
  utsq = 0. ;
  #pragma unroll //3
  for(i=1;i<4;i++)
	#pragma unroll //3
    for(j=1;j<4;j++) utsq += gcov[i][j]*prim[UTCON1+i-1]*prim[UTCON1+j-1] ;

  if( (utsq < 0.) && (fabs(utsq) < 1.0e-13) ) { 
    utsq = fabs(utsq);
  }
  if(utsq < 0. || utsq > UTSQ_TOO_BIG) {
    retval = 2;
    return(retval) ;
  }

  gammasq = 1. + utsq ;
  gamma  = sqrt(gammasq);
	
  // Always calculate rho from D and gamma so that using D in EOS remains consistent
  //   i.e. you don't get positive values for dP/d(vsq) . 
  rho0 = D / gamma ;
  u = prim[UU] ;
  p = pressure_rho0_u(rho0,u) ;

  // Initialize independent variables for Newton-Raphson:
  x_1d[0] = 1. - 1. / gammasq ; 

  // Find vsq via Newton-Raphson:
  retval = general_newton_raphson2( x_1d, 1, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm) ; 

  /* Problem with solver, so return denoting error before doing anything further */
  if( retval != 0 ) { 
    retval = retval*100+1;
    return(retval);
  }

  // Calculate v^2 :
  vsq = x_1d[0];
  if( (vsq >= 1.) || (vsq < 0.) ) {
    retval = 4;
    return(retval) ;
  }

  // Find W from this vsq:
  W = W_of_vsq2(vsq, &p, &rho0, &u, D, K_atm);

  // Recover the primitive variables from the scalars and conserved variables:
  gtmp = sqrt(1. - vsq);
  gamma = 1./gtmp ;

  // User may want to handle this case differently, e.g. do NOT return upon 
  // a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
  if( (rho0 <= 0.) || (u <= 0.) ) { 
    retval = 5;
    return(retval) ;
  }

  prim[RHO] = rho0 ;
  prim[UU] = u ;

  #pragma unroll //3
  for(i=1;i<4;i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
  #pragma unroll //3
  for(i=1;i<4;i++) prim[UTCON1+i-1] = gamma/(W+Bsq) * ( Qtcon[i] + QdotB*Bcon[i]/W ) ;
	
  /* set field components */
  #pragma unroll //3
  for(i = BCON1; i <= BCON3; i++) prim[i] = U[i] ;

  /* done! */
  return(retval) ;
}


__device__ int general_newton_raphson2( FTYPE x[], int n, 
				   FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm )
{
  FTYPE f, df, dx[NEWT_DIM2], x_old[NEWT_DIM2], resid[NEWT_DIM2], 
    jac[NEWT_DIM2][NEWT_DIM2];
  FTYPE errx;
  int    n_iter, id, i_extra, doing_extra;
  FTYPE W,W_old, rho,p,u;

  int   keep_iterating;

  // Initialize various parameters and variables:
  errx = 1. ; 
  df =  f = 1.;
  i_extra = doing_extra = 0;

  for( id = 0; id < n ; id++)  x_old[id] =  x[id] ;

  W = W_old = 0.;

  n_iter = 0;

  /* Start the Newton-Raphson iterations : */
  keep_iterating = 1;
  while( keep_iterating ) { 

	func_1d_gnr2(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm);/* returns with new dx, f, df */
	//func_vsq(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D);
    /* Save old values before calculating the new: */
    errx = 0.;
    for( id = 0; id < n ; id++) {
      x_old[id] = x[id] ;
    }

    for( id = 0; id < n ; id++) {
      x[id] += dx[id]  ;
    }

    /****************************************/
    /* Make sure that the new x[] is physical : */
    /****************************************/
    // METHOD specific
    validate_x2( x, x_old );

    /****************************************/
    /* Calculate the convergence criterion */
    /****************************************/

    /* For the new criterion, always look at error in "W" : */
    // METHOD specific
    W_old = W;
    W = W_of_vsq2( x[0], &p, &rho, &u, D, K_atm);
    errx  = (W==0.) ?  fabs(W-W_old) : fabs((W-W_old)/W);
    errx += (x[0]==0.) ?  fabs(x[0]-x_old[0]) : fabs((x[0]-x_old[0])/x[0]);
    
    if( (fabs(errx) <= NEWT_TOL) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0) ) {
      doing_extra = 1;
    }

    if( doing_extra == 1 ) i_extra++ ;

    // See if we've done the extra iterations, or have done too many iterations:
    if( ((fabs(errx) <= NEWT_TOL)&&(doing_extra == 0)) 
	|| (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER-1)) ) {
      keep_iterating = 0;
    }

    n_iter++;
  }   // END of while(keep_iterating)

  /*  Check for bad untrapped divergences : */
  if( (isfinite(f)==0) || (isfinite(df)==0) ) {
    return(2);
  }

  // Return in different ways depending on whether a solution was found:
  if( fabs(errx) > MIN_NEWT_TOL){

    return(1);
  }
  if( (fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > NEWT_TOL) ){
    //fprintf(stderr," totalcount = %d   1   %d  %26.20e \n",n_iter,i_extra,errx); fflush(stderr);
    return(0);
  }
  if( fabs(errx) <= NEWT_TOL ){
    //fprintf(stderr," totalcount = %d   2   %d  %26.20e \n",n_iter,i_extra,errx); fflush(stderr); 
    return(0);
  }
  return(0);
}

__device__ void validate_x2(FTYPE x[1], FTYPE x0[1] ) 
{
  FTYPE small = 1.e-10;
  x[0] = (x[0] >= 1.0)    ?  ( 0.5*(x0[0] + 1.) )    : x[0];
  x[0] = (x[0] <  -small) ?  ( 0.5*x0[0] )           : x[0];
  x[0] = fabs(x[0]);
  return;
}

__device__ void func_1d_gnr2(FTYPE x[], FTYPE dx[], FTYPE resid[], 
			FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n,FTYPE Bsq,FTYPE Qtsq, FTYPE QdotBsq,FTYPE Qdotn,FTYPE D, FTYPE K_atm)
{
  FTYPE vsq,W,Wsq,W3,dWdvsq , fact_tmp, rho, p, u  ;
  vsq = x[0];

  // Calculate best value for W given current guess for vsq: 
  W = W_of_vsq2(vsq, &p, &rho, &u ,D, K_atm);
  Wsq = W*W;
  W3 = W*Wsq;

  // Doing this assuming  P = (G-1) u :

  dWdvsq = dWdvsq_calc2(vsq, rho, p);

  fact_tmp = (Bsq + W) ;

  resid[0] = Qtsq  -  vsq * fact_tmp * fact_tmp  +  QdotBsq * ( Bsq + 2.*W ) / Wsq ; 
  jac[0][0] =  -fact_tmp * ( fact_tmp +   2. * dWdvsq * ( vsq + QdotBsq/W3 ) ) ; 

  dx[0] = -resid[0]/jac[0][0];

  *f = 0.5*resid[0]*resid[0];
  *df = -2. * (*f);
}

/* 
pressure as a function of rho0 and w = rho0 + u + p 
this is used by primtoU and Utoprim_1D
*/
__device__ FTYPE pressure_of_rho2(FTYPE rho0,  FTYPE K_atm)
{ 
  return( K_atm * pow( rho0, G_ATM )  );
}

/* 
internal energy density as a function of the pressure
*/
__device__ FTYPE u_of_p2(FTYPE p)
{
  return( p / (GAMMA - 1.) ) ;
}

/* 
W as a function of v^2
*/
__device__ FTYPE W_of_vsq2(FTYPE vsq, FTYPE *p, FTYPE *rho, FTYPE *u, FTYPE D, FTYPE K_atm)
{
  FTYPE gtmp;
  gtmp = (1. - vsq);
  *rho = D * sqrt(gtmp);
  *p = pressure_of_rho2(*rho, K_atm);
  *u = u_of_p2(*p);
  return( (*rho + *u + *p ) / gtmp  );
}

/* 
dW/dvsq as a function of v^2, rho, p
*/
__device__ FTYPE dWdvsq_calc2(FTYPE vsq, FTYPE rho, FTYPE p)
{
  return(  ( GAMMA*(2.-G_ATM)*p   + (GAMMA-1.)*rho ) / ( 2.*(GAMMA-1.)*(1.-vsq)*(1.-vsq) )   ) ;
}


__device__ int Utoprim_2d(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM],
	FTYPE gdet, FTYPE prim[NPR])
{
	FTYPE U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	FTYPE alpha;

	if (U[0] <= 0.) {
		return(-100);
	}

	/* First update the primitive B-fields */
	#pragma unroll //3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	/* Transform the CONSERVED variables into the new system */
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma unroll //3
	for (i = UTCON1; i <= UTCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}
	#pragma unroll //3
	for (i = BCON1; i <= BCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	#pragma unroll //BCON1
	for (i = 0; i < BCON1; i++) {
		prim_tmp[i] = prim[i];
	}
	#pragma unroll //3
	for (i = BCON1; i <= BCON3; i++) {
		prim_tmp[i] = alpha*prim[i];
	}

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		#pragma unroll //BCON1
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}
	return(ret);
}

__device__ int Utoprim_new_body(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM],
	FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR])
{
	FTYPE x_2d[NEWT_DIM];
	FTYPE QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	FTYPE rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, j, n, retval, i_increase;
	FTYPE Bsq, QdotBsq, Qtsq, Qdotn, D;

	n = NEWT_DIM;

	// Assume ok initially:
	retval = 0;
	#pragma unroll //3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll //3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower_g(Bcon, gcov, Bcov);
	#pragma unroll //4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise_g(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll //3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll //4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);

	Qdotn = Qcon[0] * ncov[0];

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	#if AMD
	Qtsq =  mad(Qdotn,Qdotn,Qsq);
	#else
	Qtsq = Qsq + Qdotn*Qdotn;
	#endif
	D = U[RHO];

	/* calculate W from last timestep and use for guess */
	utsq = 0.;
	#pragma unroll //3
	for (i = 1; i<4; i++)
		#pragma unroll //4
		for (j = 1; j<4; j++) utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];
		utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];

	if ((utsq < 0.) && (fabs(utsq) < 1.0e-13)) {
		utsq = fabs(utsq);
	}
	if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
		retval = 2;
		return(retval);
	}

	gammasq = 1. + utsq;
	gamma = sqrt(gammasq);

	// Always calculate rho from D and gamma so that using D in EOS remains consistent
	//   i.e. you don't get positive values for dP/d(vsq) . 
	rho0 = D / gamma;
	u = prim[UU];
	p = pressure_rho0_u(rho0, u);
	w = rho0 + u + p;

	W_last = w*gammasq;

	// Make sure that W is large enough so that v^2 < 1 : 
	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq)
		- QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq))
		&& (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	// Calculate W and vsq: 
	x_2d[0] = fabs(W_last);
	x_2d[1] = x1_of_x0(W_last, Bsq, Qtsq, QdotBsq);
	retval = general_newton_raphson(x_2d, n, Bsq, Qtsq, QdotBsq, Qdotn, D);

	W = x_2d[0];
	vsq = x_2d[1];

	/* Problem with solver, so return denoting error before doing anything further */
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
	p = pressure_rho0_w(rho0, w);
	u = w - (rho0 + p);

	// User may want to handle this case differently, e.g. do NOT return upon 
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.) || (u <= 0.)) {
		retval = 5;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#if AMD
	#pragma unroll //4
	for (i = 1; i<4; i++)  Qtcon[i] =  mad(ncon[i] , Qdotn,Qcon[i]);
	#pragma unroll //4
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (mad(QdotB,Bcon[i] / W,Qtcon[i]));
	#else
	#pragma unroll //4
	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	#pragma unroll //4
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);
	#endif
	/* set field components */
	#pragma unroll //3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	/* done! */
	return(retval);
}

__device__ FTYPE vsq_calc(FTYPE W, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq)
{
	FTYPE Wsq, Xsq;
	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);
	#if AMD
	return((mad(Wsq , Qtsq , QdotBsq * (Bsq + 2.*W))) / (Wsq*Xsq));
	#else
	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
	#endif
}

__device__ FTYPE x1_of_x0(FTYPE x0, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq)
{
	FTYPE vsq;
	FTYPE dv = 1.e-15;
	vsq = fabs(vsq_calc(x0, Bsq, Qtsq, QdotBsq)); // guaranteed to be positive 
	return((vsq > 1.) ? (1.0 - dv) : vsq);
}

__device__ void validate_x(FTYPE x[2], FTYPE x0[2])
{
	FTYPE dv = 1.e-15;

	/* Always take the absolute value of x[0] and check to see if it's too big:  */
	x[0] = fabs(x[0]);
	x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];

	x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
	x[1] = (x[1] > 1.) ? (1. - dv) : x[1];  /* if it's too big   */
	return;
}

__device__ int general_newton_raphson(FTYPE x[], int n,
	 FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D)
{
	FTYPE f, df, dx[NEWT_DIM], x_old[NEWT_DIM];
	FTYPE resid[NEWT_DIM], jac[NEWT_DIM][NEWT_DIM];
	FTYPE errx;
	int    n_iter, id, i_extra, doing_extra;

	int   keep_iterating;

	// Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	for (id = 0; id < n; id++)  x_old[id] = x[id];

	n_iter = 0;

	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;
	while (keep_iterating) {
		func_vsq(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D);  /* returns with new dx, f, df */

		/* Save old values before calculating the new: */
		errx = 0.;
		for (id = 0; id < n; id++) {
			x_old[id] = x[id];
		}

		/* Make the newton step: */
		for (id = 0; id < n; id++) {
			x[id] += dx[id];
		}
		errx = (x[0] == 0.) ? fabs(dx[0]) : fabs(dx[0] / x[0]);

		validate_x(x, x_old);

		if ((fabs(errx) <= NEWT_TOL) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) {
			doing_extra = 1;
		}

		if (doing_extra == 1) i_extra++;

		if (((fabs(errx) <= NEWT_TOL) && (doing_extra == 0))
			|| (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;

	}   // END of while(keep_iterating)

	/*  Check for bad untrapped divergences : */
	if ((isfinite(f) == 0) || (isfinite(df) == 0)) {
		return(2);
	}

	if (fabs(errx) > MIN_NEWT_TOL){
		return(1);
	}
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > NEWT_TOL)){
		return(0);
	}
	if (fabs(errx) <= NEWT_TOL){
		return(0);
	}
	return(0);
}

__device__ void func_vsq(FTYPE x[], FTYPE dx[], FTYPE resid[],
	FTYPE jac[][NEWT_DIM], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D)
{
	FTYPE  W, vsq, Wsq, p_tmp, dPdvsq, dPdW, gtmp;
	FTYPE t11;
	FTYPE t16;
	FTYPE t18;
	FTYPE t2;
	FTYPE t21;
	FTYPE t23;
	FTYPE t24;
	FTYPE t25;
	FTYPE t3;
	FTYPE t35;
	FTYPE t36;
	FTYPE t4;
	FTYPE t40;
	FTYPE t9;

	W = x[0];
	vsq = x[1];

	Wsq = W*W;
	gtmp = 1. - vsq;
	
	#if AMD
	p_tmp = (GAMMA - 1.) * (mad(W , gtmp, - D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - vsq) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (mad(0.5 , D / sqrt(1. - vsq), - W)) / GAMMA;
	#else
	p_tmp = (GAMMA - 1.) * (W * gtmp - D * sqrt(gtmp)) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - vsq) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (0.5 * D / sqrt(1. - vsq) - W) / GAMMA;
	#endif
	// These expressions were calculated using Mathematica, but made into efficient 
	// code using Maple.  Since we know the analytic form of the equations, we can 
	// explicitly calculate the Newton-Raphson step: 

	#if AMD
	t2 = mad(-0.5,Bsq , dPdvsq);
	t3 = Bsq + W;
	t4 = t3*t3;
	t9 = 1 / Wsq;
	t11 =  mad(QdotBsq,(Bsq + 2.0*W)*t9,mad(- vsq,t4,Qtsq)) ;
	t16 = QdotBsq*t9;
	t18 = -mad(0.5,Bsq*(1.0 + vsq),Qdotn) + mad(0.5,t16, - W + p_tmp);
	t21 = 1 / t3;
	t23 = 1 / W;
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = mad(t25,t3 , (mad(- 2.0,dPdvsq,Bsq))*(mad(vsq,Wsq*W,QdotBsq))*t9*t23);
	t36 = 1 / t35;
	dx[0] = -(mad(t2,t11 , t4*t18))*t21*t36;
	t40 = (vsq + t24)*t3;
	dx[1] = -(-mad(t25,t11,  2.0*t40*t18))*t21*t36;
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;
	*df = mad(-resid[0] , resid[0], - resid[1] * resid[1]);
	#else
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
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;
	*df = -resid[0] * resid[0] - resid[1] * resid[1];
	#endif
	*f = -0.5 * (*df);
}

/*Declare structs for 'other functions'*/
struct of_geom {
	FTYPE2 gcon[NDIM][NDIM] ;
	FTYPE2 gcov[NDIM][NDIM] ;
	FTYPE2 g ;
} ;

struct of_state {
	FTYPE2 ucon[NDIM] ;
	FTYPE2 ucov[NDIM] ;
	FTYPE2 bcon[NDIM] ;
	FTYPE2 bcov[NDIM] ;
} ;

/*Declare other functions*/
__device__ void get_state(FTYPE2 *pr, struct of_geom *geom, struct of_state *q);
__device__ void ucon_calc(FTYPE2 *pr, struct of_geom *geom, FTYPE2 *ucon);
__device__ void bcon_calc(FTYPE2 *pr, FTYPE2 *ucon, FTYPE2 *ucov, FTYPE2 *bcon) ;
__device__ int gamma_calc(FTYPE2 *pr, struct of_geom *geom, FTYPE2 *gamma);
__device__ void get_geometry(int N1, int N2, int ii, int jj, int kk, struct of_geom *geom, const  FTYPE2* __restrict__  gcov_GPU, const  FTYPE2* __restrict__  gcon_GPU, const  FTYPE2* __restrict__  gdet_GPU);
__device__ FTYPE2 slope_lim(FTYPE2 y1,FTYPE2 y2,FTYPE2 y3, int lim) ;
__device__ void raise(FTYPE2 *ucov, struct of_geom *geom, FTYPE2 *ucon);
__device__ void lower(FTYPE2 *ucon, struct of_geom *geom, FTYPE2 *ucov);
__device__ void primtoflux(FTYPE2 *pr, struct of_state *q, int dir, struct of_geom *geom, FTYPE2 *flux, FTYPE2 gam); 
__device__ void primtoU(FTYPE2 *pr, struct of_state *q, struct of_geom *geom, FTYPE2 *U, FTYPE2 gam);
__device__ void vchar(FTYPE2 *pr, struct of_state *q, struct of_geom *geom, int js, FTYPE2 *vmax, FTYPE2 *vmin, FTYPE2 gam);
__device__ void mhd_calc(FTYPE2 *pr, int dir, struct of_state *q, FTYPE2 *mhd, FTYPE2 gam); 
__device__ void bl_coord(FTYPE2 *X, FTYPE2 *r, FTYPE2 *th, FTYPE2 hslope, FTYPE2 R0);
__device__ void coord(int i, int j, int z, int loc, FTYPE2 *X, FTYPE2 fractheta, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, int N1_MPI, int N2_MPI, int N3_MPI);
__device__ void source(int N1, int N2, FTYPE2 *ph, struct of_geom *geom, int icurr, int jcurr, FTYPE2 *dU, FTYPE2 Dt, FTYPE2 gam, const  FTYPE2* __restrict__  conn);
__device__ void inflow_check(int N1,  int N2, FTYPE2* prim ,int ii, int jj, int type, const  FTYPE2* __restrict__  gcov1, const  FTYPE2* __restrict__  gcon2, const  FTYPE2* __restrict__  gdet3) ;
__device__ FTYPE2 bsq_calc(FTYPE2 *pr, struct of_geom *geom);
__device__ double NewtonRaphson(double start, size_t max_count, int dir, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq);

__device__ void primtoU(FTYPE2 *pr, struct of_state *q, struct of_geom *geom, FTYPE2 *U, FTYPE2 gam)
{
	primtoflux(pr,q,0,geom, U ,gam) ;
	return ;
}

/* add in source terms to equations of motion */
__device__ void source(int N1, int N2, FTYPE2 *ph, struct of_geom *geom, int icurr, int jcurr, FTYPE2 *dU, FTYPE2 Dt, FTYPE2 gam, const  FTYPE2* __restrict__  conn_GPU)
{
	FTYPE2 mhd[NDIM][NDIM] ;
	int k;
	struct of_state q ;
	double conn;
	int global_id=icurr*(N2+2*N2G)+jcurr;
	int fix_mem2=LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, &q, mhd[0], gam) ;
	mhd_calc(ph, 1, &q, mhd[1], gam) ;
	mhd_calc(ph, 2, &q, mhd[2], gam) ;
	mhd_calc(ph, 3, &q, mhd[3], gam) ;

	/* contract mhd stress tensor with connection */
	#pragma unroll 8	
	PLOOP dU[k] = 0. ;
	#pragma unroll //NDIM	
	for(k=0; k<NDIM; k++){
		dU[UU] += mhd[0][k]*conn_GPU[0*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 0*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U1] += mhd[1][k]*conn_GPU[1*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 1*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U2] += mhd[2][k]*conn_GPU[2*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 2*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U3] += mhd[3][k]*conn_GPU[3*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 3*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		conn=conn_GPU[0*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 1*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id];
		dU[UU] += mhd[1][k]*conn;
		dU[U1] += mhd[0][k]*conn;
		conn=conn_GPU[0*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 2*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[UU] += mhd[2][k]*conn;
		dU[U2] += mhd[0][k]*conn;
		conn=conn_GPU[0*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 3*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[UU] += mhd[3][k]*conn;
		dU[U3] += mhd[0][k]*conn;
		conn=conn_GPU[1*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 2*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U1] += mhd[2][k]*conn;
		dU[U2] += mhd[1][k]*conn;
		conn=conn_GPU[1*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 3*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U1] += mhd[3][k]*conn;
		dU[U3] += mhd[1][k]*conn ;
		conn=conn_GPU[2*NDIM*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 3*NDIM*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + k*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + global_id] ;
		dU[U2] += mhd[3][k]*conn;
		dU[U3] += mhd[2][k]*conn;
	}	

	//misc_source(ph, ii, jj, geom, &q, dU, Dt) ;
	#pragma unroll 8	
	PLOOP dU[k] *= geom->g ;
	/* done! */
}

__device__ void coord(int i, int j, int z, int loc, FTYPE2 *X, FTYPE2 fractheta, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, int N1_MPI, int N2_MPI, int N3_MPI)
{
        FTYPE2 startx[4], dx[4];
	    startx[1] = log(Rin - R0) ;
        startx[2] = 0.5*(1.-fractheta) ;
		startx[3] = 0.;
        dx[1] = log((Rout - R0)/(Rin - R0))/N1_MPI ;
        dx[2] = fractheta/N2_MPI ;
		dx[3] = 2.0*M_PI/N3_MPI ;
	    #if AMD
		if(loc == FACE1) {
            X[1] = mad(i,dx[1],startx[1])  ;
            X[2] = mad(j+0.5,dx[2],startx[2]) ;
			X[3] = mad(z+0.5,dx[3],startx[3]);
        }
        else if(loc == FACE2) {
            X[1] = mad(i+0.5,dx[1],startx[1])  ;
            X[2] = mad(j,dx[2],startx[2]) ;
			X[3] = mad(z+0.5,dx[3],startx[3]);
        }
		else if (loc == FACE3) {
            X[1] = mad(i+0.5,dx[1],startx[1])  ;
            X[2] = mad(j+0.5,dx[2],startx[2]) ;
			X[3] = mad(z,dx[3],startx[3]);
		}
        else if(loc == CENT) {
            X[1] = mad(i+0.5,dx[1],startx[1])  ;
            X[2] = mad(j+0.5,dx[2],startx[2]) ;
			X[3] = mad(z+0.5,dx[3],startx[3]);
        }
        else {
            X[1] = mad(i,dx[1],startx[1])  ;
            X[2] = mad(j,dx[2],startx[2]) ;
			X[3] = mad(z,dx[3],startx[3]);
        }
		#else
			    if(loc == FACE1) {
            X[1] = startx[1] + i*dx[1] ;
            X[2] = startx[2] + (j + 0.5)*dx[2] ;
			X[3] = startx[3] + (z + 0.5)*dx[3];
        }
        else if(loc == FACE2) {
            X[1] = startx[1] + (i + 0.5)*dx[1] ;
            X[2] = startx[2] + j*dx[2] ;
			X[3] = startx[3] + (z + 0.5)*dx[3];
        }
		else if (loc == FACE3) {
			X[1] = startx[1] + (i + 0.5)*dx[1];
			X[2] = startx[2] + (j + 0.5)*dx[2];
			X[3] = startx[3] + z*dx[3];
		}
        else if(loc == CENT) {
            X[1] = startx[1] + (i + 0.5)*dx[1] ;
            X[2] = startx[2] + (j + 0.5)*dx[2] ;
			X[3] = startx[3] + (z + 0.5)*dx[3];
        }
        else {
            X[1] = startx[1] + i*dx[1] ;
            X[2] = startx[2] + j*dx[2] ;
			X[3] = startx[3] + z*dx[3];
        }
		#endif
        return ;
}

__device__ void bl_coord(FTYPE2 *X, FTYPE2 *r, FTYPE2 *th, FTYPE2 hslope, FTYPE2 R0)
{
	*r = exp(X[1]) + R0 ;
	
	double A1 = 1. / (1. + pow(CHARLIE*log(*r) / log(10.), DELTA));
	double A2 = (0.5 - BRAVO*0.5) / pow(0.5, QUEBEC);
	double Xc = sqrt(pow(X[2], 2.));
	double sign=1.;
	
	//*th = M_PI*X[2] + 0.5*sin(M_PI + 2.*M_PI*X[2]) *(1.-1./ (1. + pow(CHARLIE*log(*r) / log(10.), DELTA)));
	if (X[2] < 0.0){
		sign = -1.;
	}
	if (X[2] > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X[2] < 0.5){
		*th =sign*(A1* M_PI*Xc + M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))*(1. - A1) + 0.50*(1. - A1)*sin(M_PI + 2.*M_PI*(BRAVO*Xc + A2*pow(Xc, QUEBEC))));
	}
	else{
		*th = M_PI - sign*(A1* M_PI*(1.-Xc) + M_PI*(BRAVO*(1.-Xc) + A2*pow(1.-Xc, QUEBEC))*(1. - A1) + 0.50*(1. - A1)*sin(M_PI + 2.*M_PI*(BRAVO*(1.-Xc) + A2*pow(1.-Xc, QUEBEC))));
	}

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

__device__ void primtoflux(FTYPE2 *pr, struct of_state *q, int dir, struct of_geom *geom, FTYPE2 *flux, FTYPE2 gam) 
{
	int k ;
	FTYPE2 mhd[NDIM] ;

	/* particle number flux */
	flux[RHO] = pr[RHO]*q->ucon[dir] ;

	mhd_calc(pr, dir, q, mhd, gam) ;

	/* MHD stress-energy tensor w/ first index up, 
	 * second index down. */
	flux[UU] = mhd[0] + flux[RHO] ;
	flux[U1] = mhd[1] ;
	flux[U2] = mhd[2] ;
	flux[U3] = mhd[3] ;

	/* dual of Maxwell tensor */
	#if AMD
	flux[B1]  = mad(q->bcon[1],q->ucon[dir], - q->bcon[dir]*q->ucon[1]) ;
	flux[B2]  = mad(q->bcon[2],q->ucon[dir], - q->bcon[dir]*q->ucon[2]) ;
	flux[B3]  = mad(q->bcon[3],q->ucon[dir], - q->bcon[dir]*q->ucon[3]);
	#else
	flux[B1]  = q->bcon[1]*q->ucon[dir] - q->bcon[dir]*q->ucon[1] ;
	flux[B2]  = q->bcon[2]*q->ucon[dir] - q->bcon[dir]*q->ucon[2] ;
	flux[B3]  = q->bcon[3]*q->ucon[dir] - q->bcon[dir]*q->ucon[3] ;
	#endif
	#pragma unroll 8
	PLOOP flux[k] *= geom->g ;
}

__device__ void vchar(FTYPE2 *pr, struct of_state *q, struct of_geom *geom, int js, FTYPE2 *vmax, FTYPE2 *vmin, FTYPE2 gam)
{
	FTYPE2 discr,vp,vm,bsq,EE,EF,va2,cs2,cms2,rho,u ;
	FTYPE2 Acov[NDIM],Bcov[NDIM],Acon[NDIM],Bcon[NDIM] ;
	FTYPE2 Asq,Bsq,Au,Bu,AB,Au2,Bu2,AuBu,A,B,C ;
	int j ;

	#pragma unroll //NDIM
	DLOOPA Acov[j] = 0. ;
	Acov[js] = 1. ;
	raise(Acov,geom,Acon) ;
	
	#pragma unroll //NDIM
	DLOOPA Bcov[j] = 0. ;
	Bcov[TT] = 1. ;
	raise(Bcov,geom,Bcon) ;

	/* find fast magnetosonic speed */
	bsq = dot(q->bcon,q->bcov) ;
	rho = pr[RHO] ;
	u = pr[UU] ;
	#if AMD
	EF = mad(gam,u,rho);
	#else
	EF = rho + gam*u ;
	#endif
	EE = bsq + EF ;
	va2 = bsq/EE ;
	cs2 = gam*(gam - 1.)*u/EF ;

//	if(cs2 < 0.) cs2 = SMALL ;
//	if(cs2 > 1.) cs2 = 1. ;
//	if(va2 < 0.) va2 = SMALL ;
//	if(va2 > 1.) va2 = 1. ;

	cms2 = cs2 + va2 - cs2*va2 ;	/* and there it is... */

	//cms2 *= 1.1 ;

	/* check on it! */
	if(cms2 < 0.) {
		//fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL ;
	}
	if(cms2 > 1.) {
		//fail(FAIL_COEFF_SUP) ;
		cms2 = 1. ;
	}

	/* now require that speed of wave measured by observer 
	   q->ucon is cms2 */
	Asq = dot(Acon,Acov) ;
	Bsq = dot(Bcon,Bcov) ;
	Au =  dot(Acov,q->ucon) ;
	Bu =  dot(Bcov,q->ucon) ;
	AB =  dot(Acon,Bcov) ;
	Au2 = Au*Au ;
	Bu2 = Bu*Bu ;
	AuBu = Au*Bu ;
	#if AMD
	A =     mad(-(Bsq + Bu2),cms2, Bu2)  ;
	B = 2.* mad(- (AB + AuBu),cms2,AuBu)   ;
	C =      mad(- (Asq + Au2),cms2,Au2)   ;
	discr = mad(B,B, - 4.*A*C) ;
	#else
	A =      Bu2  - (Bsq + Bu2)*cms2 ;
	B = 2.*( AuBu - (AB + AuBu)*cms2 ) ;
	C =      Au2  - (Asq + Au2)*cms2 ;
	discr = B*B - 4.*A*C ;
	#endif
	if((discr<0.0)&&(discr>-1.e-10)) discr=0.0;
	else if(discr < -1.e-10) {
		/*fprintf(stderr,"\n\t %g %g %g %g %g\n",A,B,C,discr,cms2) ;
		fprintf(stderr,"\n\t q->ucon: %g %g %g %g\n",q->ucon[0],q->ucon[1],
				q->ucon[2],q->ucon[3]) ;
		fprintf(stderr,"\n\t q->bcon: %g %g %g %g\n",q->bcon[0],q->bcon[1],
				q->bcon[2],q->bcon[3]) ;
		fprintf(stderr,"\n\t Acon: %g %g %g %g\n",Acon[0],Acon[1],
				Acon[2],Acon[3]) ;
		fprintf(stderr,"\n\t Bcon: %g %g %g %g\n",Bcon[0],Bcon[1],
				Bcon[2],Bcon[3]) ;
		fail(FAIL_VCHAR_DISCR) ;*/
		discr = 0. ;
	}

	discr = sqrt(discr) ;
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);
	
	#if( FULL_DISP ) 
	double vp2, vm2;
	vp2 = NewtonRaphson(vp , 5, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vm2 = NewtonRaphson(vm, 5, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vp= vp2;
	vm = vm2;
	#endif

	if(vp > vm) {
		*vmax = vp ;
		*vmin = vm ;
	}
	else {
		*vmax = vm ;
		*vmin = vp ;
	}

	return ;
}

__device__ double NewtonRaphson(double start, size_t max_count, int dir, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq)
{
	size_t count = 0;
	double dx = start/1000000.0;
	double x = start;
	double diff, derivative;
	do{
		diff = Drel(dir, x, ucon, ucov, bcon, geom, E, vasq, csq);
		derivative = (Drel(dir, x + dx, ucon, ucov, bcon, geom, E, vasq, csq) - diff) / dx;
		count++;
		x = x - diff / (derivative);
	} while (Drel(dir, x*0.99999, ucon, ucov, bcon, geom, E, vasq, csq)*Drel(dir, x*1.00001, ucon, ucov, bcon, geom, E, vasq, csq)>0.0 && (count < max_count));
	if (count >= max_count){
		x = start;
	}
	return x;
}

__device__ double Drel(int dir, double v, double *ucon, double *ucov, double *bcon, struct of_geom *geom, double E, double vasq, double csq){
	double kcov[NDIM], kcon[NDIM], Kcov[NDIM], Kcon[NDIM];
	double om, omsq, ksq, kvasq, cfsq, result;
	int i;
	kcov[0] = -v; kcov[1] = 0.0; kcov[2] = 0.0; kcov[3] = 0.0;
	if (dir == 1){
		kcov[1] = 1.0;
	}
	else if (dir == 2){
		kcov[2] = 1.0;
	}
	raise(kcov, geom, kcon);
	om = dot(ucon, kcov);
	omsq = pow(om, 2.0);
	#pragma unroll //NDIM
	for (i = 0; i < NDIM; i++){
		Kcov[i] = kcov[i] + ucov[i] * om;
		Kcon[i] = kcon[i] + ucon[i] * om;
	}
	ksq = dot(Kcov, Kcon);
	kvasq = pow(dot(kcov, bcon), 2.0) / E;
	cfsq = vasq + csq*(1.0 - vasq);
	result = 0.5*(cfsq*ksq + csq*kvasq + sqrt(pow(cfsq*ksq + csq*kvasq, 2.0) - 4.0*ksq*csq*kvasq)) - omsq;
	return result;
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc(FTYPE2 *pr, int dir, struct of_state *q, FTYPE2 *mhd, FTYPE2 gam) 
{
	int j ;
	FTYPE2 r,u,P,w,bsq,eta,ptot ;

        r = pr[RHO] ;
        u = pr[UU] ;
        P = (gam - 1.)*u ;
        w = P + r + u ;
	bsq = dot(q->bcon,q->bcov) ;
	eta = w + bsq ;
	#if AMD
	ptot = mad(0.5, bsq,P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* single row of mhd stress tensor, 
	 * first index up, second index down */
	#if AMD
	#pragma unroll //NDIM
	DLOOPA mhd[j] = mad(eta,q->ucon[dir]*q->ucov[j],mad(ptot,delta(dir,j),- q->bcon[dir]*q->bcov[j])) ;
	#else
	DLOOPA mhd[j] = eta*q->ucon[dir]*q->ucov[j]+ ptot*delta(dir,j) - q->bcon[dir]*q->bcov[j] ;
	#endif
}

__device__ void get_state(FTYPE2 *pr, struct of_geom *geom, struct of_state *q)
{
	/* get ucon */
	ucon_calc(pr, geom, q->ucon) ;
	lower(q->ucon, geom, q->ucov) ;
	bcon_calc(pr, q->ucon, q->ucov, q->bcon) ;
	lower(q->bcon, geom, q->bcov) ;

	return ;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ void raise(FTYPE2 *ucov, struct of_geom *geom, FTYPE2 *ucon)
{
	#if AMD
	ucon[0] = mad(geom->gcon[0][0],ucov[0], mad(
		 geom->gcon[0][1],ucov[1],mad( 
		 geom->gcon[0][2],ucov[2] 
		, geom->gcon[0][3]*ucov[3]))) ;
	ucon[1] = mad(geom->gcon[1][0],ucov[0], mad(
		 geom->gcon[1][1],ucov[1],mad( 
		 geom->gcon[1][2],ucov[2] 
		, geom->gcon[1][3]*ucov[3])));
	ucon[2] = mad(geom->gcon[2][0],ucov[0], mad(
		 geom->gcon[2][1],ucov[1],mad( 
		 geom->gcon[2][2],ucov[2] 
		, geom->gcon[2][3]*ucov[3]))) ;
	ucon[3] = mad(geom->gcon[3][0],ucov[0], mad(
		 geom->gcon[3][1],ucov[1],mad( 
		 geom->gcon[3][2],ucov[2] 
		, geom->gcon[3][3]*ucov[3]))) ;
	#else
		ucon[0] = geom->gcon[0][0]*ucov[0] 
		+ geom->gcon[0][1]*ucov[1] 
		+ geom->gcon[0][2]*ucov[2] 
		+ geom->gcon[0][3]*ucov[3] ;
	ucon[1] = geom->gcon[1][0]*ucov[0] 
		+ geom->gcon[1][1]*ucov[1] 
		+ geom->gcon[1][2]*ucov[2] 
		+ geom->gcon[1][3]*ucov[3] ;
	ucon[2] = geom->gcon[2][0]*ucov[0] 
		+ geom->gcon[2][1]*ucov[1] 
		+ geom->gcon[2][2]*ucov[2] 
		+ geom->gcon[2][3]*ucov[3] ;
	ucon[3] = geom->gcon[3][0]*ucov[0] 
		+ geom->gcon[3][1]*ucov[1] 
		+ geom->gcon[3][2]*ucov[2] 
		+ geom->gcon[3][3]*ucov[3] ;
	#endif
        return ;
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower(FTYPE2 *ucon, struct of_geom *geom, FTYPE2 *ucov)
{
	#if AMD
	ucov[0] = mad(geom->gcov[0][0],ucon[0],mad( 
		 geom->gcov[0][1],ucon[1],mad( 
		 geom->gcov[0][2],ucon[2], 
		 geom->gcov[0][3]*ucon[3]))) ;
	ucov[1] = mad(geom->gcov[1][0],ucon[0],mad( 
		 geom->gcov[1][1],ucon[1],mad( 
		 geom->gcov[1][2],ucon[2], 
		 geom->gcov[1][3]*ucon[3]))) ;
	ucov[2] = mad(geom->gcov[2][0],ucon[0],mad( 
		 geom->gcov[2][1],ucon[1],mad( 
		 geom->gcov[2][2],ucon[2], 
		 geom->gcov[2][3]*ucon[3]))) ;
	ucov[3] = mad(geom->gcov[3][0],ucon[0],mad( 
		 geom->gcov[3][1],ucon[1],mad( 
		 geom->gcov[3][2],ucon[2], 
		 geom->gcov[3][3]*ucon[3]))) ;
        return ;
	#else
		ucov[0] = geom->gcov[0][0]*ucon[0] 
		+ geom->gcov[0][1]*ucon[1] 
		+ geom->gcov[0][2]*ucon[2] 
		+ geom->gcov[0][3]*ucon[3] ;
	ucov[1] = geom->gcov[1][0]*ucon[0] 
		+ geom->gcov[1][1]*ucon[1] 
		+ geom->gcov[1][2]*ucon[2] 
		+ geom->gcov[1][3]*ucon[3] ;
	ucov[2] = geom->gcov[2][0]*ucon[0] 
		+ geom->gcov[2][1]*ucon[1] 
		+ geom->gcov[2][2]*ucon[2] 
		+ geom->gcov[2][3]*ucon[3] ;
	ucov[3] = geom->gcov[3][0]*ucon[0] 
		+ geom->gcov[3][1]*ucon[1] 
		+ geom->gcov[3][2]*ucon[2] 
		+ geom->gcov[3][3]*ucon[3] ;
	#endif
}

/* find contravariant four-velocity */
__device__ void ucon_calc(FTYPE2 *pr, struct of_geom *geom, FTYPE2 *ucon)
{
	FTYPE2 alpha,gamma ;
	FTYPE2 beta[NDIM] ;
	int j ;

	alpha = 1./sqrt(-geom->gcon[TT][TT]) ;
	#pragma unroll //4
	SLOOPA beta[j] = geom->gcon[TT][j]*alpha*alpha ;

	if( gamma_calc(pr,geom,&gamma) ) { 
	 // fflush(stderr);
	  //fprintf(stderr,"\nucon_calc(): gamma failure \n");
	 // fflush(stderr);
	 // fail(FAIL_GAMMA);
	}

	ucon[TT] = gamma/alpha ;
	#if AMD
	#pragma unroll //4
	SLOOPA ucon[j] = mad(- gamma,beta[j]/alpha,pr[U1+j-1])  ;
	#else
	#pragma unroll //4
	SLOOPA ucon[j] = pr[U1+j-1] - gamma*beta[j]/alpha ;
	#endif

	return ;
}

__device__ void bcon_calc(FTYPE2 *pr, FTYPE2 *ucon, FTYPE2 *ucov, FTYPE2 *bcon) 
{
	int j ;

	#if AMD
	bcon[TT] = mad(pr[B1],ucov[1],mad(pr[B2],ucov[2], pr[B3]*ucov[3])) ;
	#pragma unroll //3
	for(j=1;j<4;j++)
		bcon[j] = ( mad(bcon[TT],ucon[j],pr[B1-1+j]))/ucon[TT] ;
	#else
		bcon[TT] = pr[B1]*ucov[1] + pr[B2]*ucov[2] + pr[B3]*ucov[3] ;
	#pragma unroll //3
	for(j=1;j<4;j++)
		bcon[j] = (pr[B1-1+j] + bcon[TT]*ucon[j])/ucon[TT] ;
	#endif
	return ;
}

__device__ int gamma_calc(FTYPE2 *pr, struct of_geom *geom, FTYPE2 *gamma)
{
        FTYPE2 qsq ;
		#if AMD
        qsq =     mad(geom->gcov[1][1],pr[U1]*pr[U1],mad(
                  geom->gcov[2][2],pr[U2]*pr[U2],
                  geom->gcov[3][3]*pr[U3]*pr[U3]))
            + 2.*mad(geom->gcov[1][2],pr[U1]*pr[U2],mad(
                 geom->gcov[1][3],pr[U1]*pr[U3],
                 geom->gcov[2][3]*pr[U2]*pr[U3])) ;
		#else
		      qsq =     geom->gcov[1][1]*pr[U1]*pr[U1]
                + geom->gcov[2][2]*pr[U2]*pr[U2]
                + geom->gcov[3][3]*pr[U3]*pr[U3]
            + 2.*(geom->gcov[1][2]*pr[U1]*pr[U2]
                + geom->gcov[1][3]*pr[U1]*pr[U3]
                + geom->gcov[2][3]*pr[U2]*pr[U3]) ;
		#endif

        if( qsq < 0. ){
          if( fabs(qsq) > 1.E-10 ){ // then assume not just machine precision
            //fprintf(stderr,"gamma_calc():  failed: i,j,qsq = %d %d %28.18e \n", icurr,jcurr,qsq);
           // fprintf(stderr,"v[1-3] = %28.18e %28.18e %28.18e  \n",pr[U1],pr[U2],pr[U3]);
	    *gamma = 1.;
	    return (1);
	  }
          else qsq=1.E-10; // set floor
        }

        *gamma = sqrt(1. + qsq) ;

        return(0) ;
}

/* load local geometry into structure geom */
__device__ void get_geometry(int N1, int N2, int ii, int jj, int kk, struct of_geom *geom, const  FTYPE2* __restrict__  gcov_GPU, const  FTYPE2* __restrict__  gcon_GPU,  const  FTYPE2* __restrict__  gdet_GPU)
{
	int fix_mem2=LOCAL_WORK_SIZE-((N2+2*N2G)*(N1+2*N1G))%LOCAL_WORK_SIZE;
	geom->gcon[0][0] = gcon_GPU[kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[0][0] = gcov_GPU[kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[0][1] = gcon_GPU[1*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[0][1] = gcov_GPU[1*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[0][2] = gcon_GPU[2*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[0][2] = 0.0;
	geom->gcon[0][3] = 0.0;
	geom->gcov[0][3] = gcov_GPU[3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[1][0] = geom->gcon[0][1];
	geom->gcov[1][0] = geom->gcov[0][1];
	geom->gcon[1][1] = gcon_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+1*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[1][1] = gcov_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+1*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[1][2] = gcon_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+2*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[1][2] = gcov_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+2*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[1][3] = gcon_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[1][3] = gcov_GPU[1*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[2][0] = geom->gcon[0][2];
	geom->gcov[2][0] = geom->gcov[0][2];
	geom->gcon[2][1] = geom->gcon[1][2]; 
	geom->gcov[2][1] = geom->gcov[1][2];
	geom->gcon[2][2] = gcon_GPU[2*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+2*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[2][2] = gcov_GPU[2*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+2*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcon[2][3] = gcon_GPU[2*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[2][3] = 0.0;
	geom->gcon[3][0] = geom->gcon[0][3];
	geom->gcov[3][0] = geom->gcov[0][3];
	geom->gcon[3][1] = geom->gcon[1][3];
	geom->gcov[3][1] = geom->gcov[1][3];
	geom->gcon[3][2] = geom->gcon[2][3];
	geom->gcov[3][2] = geom->gcov[2][3];	 
	geom->gcon[3][3] = gcon_GPU[3*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->gcov[3][3] = gcov_GPU[3*NDIM*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2)+3*NPG*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj];
	geom->g = gdet_GPU[kk*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + ii*(N2+2*N2G) + jj] ;
}

__device__ void inflow_check(int N1, int N2, FTYPE2 *pr, int ii, int jj, int type, const  FTYPE2* __restrict__  gcov, const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet)
{
    struct of_geom geom ;
    FTYPE2 ucon[NDIM] ;
    int j,k ;
    FTYPE2 alpha,beta1,gamma,vsq ;
    get_geometry(N1, N2, ii,jj,CENT,&geom, gcov, gcon, gdet) ;
    ucon_calc(pr, &geom, ucon) ;

    if( ((ucon[1] > 0.) && (type==0)) || ((ucon[1] < 0.) && (type==1)) ) { 
            // find gamma and remove it from primitives 
	if( gamma_calc(pr,&geom,&gamma) ) { 
		// fflush(stderr);
		// fprintf(stderr,"\ninflow_check(): gamma failure \n");
		// fflush(stderr);
		// fail(FAIL_GAMMA);
	}
	pr[U1] /= gamma ;
	pr[U2] /= gamma ;
	pr[U3] /= gamma ;
	alpha = 1./sqrt(-geom.gcon[0][0]) ;
	beta1 = geom.gcon[0][1]*alpha*alpha ;

	// reset radial velocity so radial 4-velocity is zero 
	pr[U1] = beta1/alpha ;

	// now find new gamma and put it back in 
	vsq = 0. ;
	#pragma unroll //NDIM
	SLOOP vsq += geom.gcov[j][k]*pr[U1+j-1]*pr[U1+k-1] ;
	if( fabs(vsq) < 1.e-13 )  vsq = 1.e-13;
	if( vsq >= 1. ) { 
	    vsq = 1. - 1./(GAMMAMAX*GAMMAMAX) ;
	}
	gamma = 1./sqrt(1. - vsq) ;
	pr[U1] *= gamma ;
	pr[U2] *= gamma ;
	pr[U3] *= gamma ;
	}
}

__device__ FTYPE2 slope_lim(FTYPE2 y1,FTYPE2 y2,FTYPE2 y3, int lim) 
{
 FTYPE2 Dqm,Dqp,Dqc,s ;
/* woodward, or monotonized central, slope limiter */
	if(lim == 0) {
		Dqm = 2.*(y2 - y1) ;
		Dqp = 2.*(y3 - y2) ;
		Dqc = 0.5*(y3 - y1) ;
		s = Dqm*Dqp ;
		if(s<= 0.) return 0. ;
		else {
			if(fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
				return(Dqm) ;
			else if(fabs(Dqp) < fabs(Dqc))
				return(Dqp) ;
			else
				return(Dqc) ;
		}
	}
	/* van leer slope limiter */
	else if(lim == 1) {
		Dqm = (y2 - y1) ;
		Dqp = (y3 - y2) ;
		s = Dqm*Dqp ;
		if(s <= 0.) return 0. ;
		else
			return(2.*s/(Dqm+Dqp)) ;
	}

	/* minmod slope limiter (crude but robust) */
	else if(lim == 2) {
		Dqm = (y2 - y1) ;
		Dqp = (y3 - y2) ;
		s = Dqm*Dqp ;
		if(s <= 0.) return 0. ;
		else if(fabs(Dqm) < fabs(Dqp)) return Dqm ;
		else return Dqp ;
	}
		return(0.) ;
}

/* returns b^2 (i.e., twice magnetic pressure) */
__device__ FTYPE2 bsq_calc(FTYPE2 *pr, struct of_geom *geom)
{
	struct of_state q ;
	get_state(pr,geom,&q) ;
	return( dot(q.bcon,q.bcov) ) ;
}

__global__ void fluxcalcprep(int N1, int N2, int N3,  FTYPE2* F,  FTYPE2* dq, const FTYPE2* __restrict__ p, int dir, int lim)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int k;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 temp1[NPR], temp2[NPR], temp3[NPR];
	
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;}
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0;}
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1;}
	if(icurr>=N1G-D1 && jcurr>=N2G-D2 && zcurr>=N3G-D3 && icurr<N1+N1G+D1 && jcurr<N2+N2G+D2 && zcurr<N3+N3G+D3){
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			temp1[k]=p[k*(ksize)+global_id-idel*isize -jdel*(N3+2*N3G)-zdel];
			temp2[k]=p[k*(ksize)+global_id];
			temp3[k]=p[k*(ksize)+global_id+idel*isize +jdel*(N3+2*N3G)+zdel];
			dq[k*(ksize)+global_id]=slope_lim(temp1[k],temp2[k], temp3[k], lim);
		}
	}
 }

  __global__ void fluxcalc2D1(int N1, int N2, int N3,  FTYPE2* F, const FTYPE2* __restrict__ dq, const FTYPE2* __restrict__ p,
  const  FTYPE2* __restrict__  gcov, const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, FTYPE2* storage1, 
  FTYPE2* storage2, FTYPE2* storage3, FTYPE2* storage4, int dir)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int idel, jdel, zdel, k;
	int face;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 cmax_l,cmin_l;
	FTYPE2 p_l[NPR];
	FTYPE2 temp1[NPR], temp2[NPR];
	struct of_geom geom;
	struct of_state state_l ;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }

	/*Watch out that the 'temp' variables in the loop above are properly initialized!*/
	if(icurr>=(2-jdel-zdel)*D1 && jcurr>=(2-idel-zdel)*D2 && zcurr>=(2-idel-jdel)*D3 && icurr<N1+N1G+D1 && jcurr<N2+N2G+D2 && zcurr<N3+N3G+D3){
		get_geometry(N1, N2, icurr, jcurr, face, &geom, gcov, gcon, gdet) ;
						//if(icurr==52 && jcurr==52){
			//printf("testGPU: %f %d \n",gcov[face*((N2+2*N2G)*(N1+2*N1G)+LOCAL_WORK_SIZE-((N2+2*N2G)*(N1+2*N1G))%LOCAL_WORK_SIZE) + global_id], face);
		//}
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			#if AMD
			p_l[k] = mad(0.5,dq[k*(ksize)+global_id-idel*isize -jdel*(N3+2*N3G)-zdel],p[k*(ksize)+global_id-idel*isize -jdel*(N3+2*N3G)-zdel]) ;
			#else
			p_l[k] = p[k*(ksize)+global_id-idel*isize -jdel*(N3+2*N3G)-zdel] + 0.5*dq[k*(ksize)+global_id-idel*isize -jdel*(N3+2*N3G)-zdel];
			#endif
		}
		get_state(p_l,&geom,&state_l) ;
		primtoflux(p_l,&state_l,dir,&geom,temp1, GAMMA) ;
		primtoflux(p_l,&state_l,TT, &geom,temp2, GAMMA) ;
		vchar(p_l,&state_l,&geom,dir,&cmax_l,&cmin_l, GAMMA) ;
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			storage1[k*(ksize)+global_id]=temp1[k];
			storage2[k*(ksize)+global_id]=temp2[k];
		}
		storage3[global_id]=cmin_l;
		storage4[global_id]=cmax_l;
	}
 }

  __global__ void fluxcalc2D2(int N1, int N2, int N3,  FTYPE2* F,  const FTYPE2* __restrict__ dq, const FTYPE2* __restrict__ p, FTYPE2* dtij,
  const  FTYPE2* __restrict__  gcov, const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, const FTYPE2* __restrict__ storage1, const FTYPE2* __restrict__ storage2,
   const FTYPE2* __restrict__ storage3, const FTYPE2* __restrict__ storage4,FTYPE2 cour, FTYPE2 Rout, FTYPE2 Rin, FTYPE2 R0, FTYPE2 fractheta, int dir, int N1_MPI, int N2_MPI, int N3_MPI)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int local_id=threadIdx.x;
	int group_id=blockIdx.x;
	int local_size=blockDim.x;
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int idel, jdel, zdel,i, k;
	int face;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	double factor;
	FTYPE2 cmax_r,cmin_r,cmax,cmin;
	FTYPE2 ctop ;
	FTYPE2 p_r[NPR];
	FTYPE2 temp3[NPR], temp4[NPR];
	__shared__ FTYPE2 local_dtij[LOCAL_WORK_SIZE];
	struct of_geom geom;
	struct of_state state_r ;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor=cour*(log((Rout - R0)/(Rin - R0))/(double)N1_MPI);}
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2;factor=cour*(fractheta/(double)N2_MPI); }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; factor=cour*(2.0*M_PI/(double)N3_MPI);}
	
	local_dtij[local_id]=1.e9;

	/*Watch out that the 'temp' variables in the loop above are properly initialized!*/
	if(icurr>=(2-jdel-zdel)*D1 && jcurr>=(2-idel-zdel)*D2 && zcurr>=(2-idel-jdel)*D3 && icurr<N1+N1G+D1 && jcurr<N2+N2G+D2 && zcurr<N3+N3G+D3){
       	get_geometry(N1, N2, icurr, jcurr, face, &geom, gcov, gcon, gdet) ;

		#pragma unroll 8
		for(k=0; k< NPR; k++){
		#if AMD
			p_r[k] =  mad(- 0.5,dq[k*(ksize)+global_id],p[k*(ksize)+global_id]);
		#else
			p_r[k] = p[k*(ksize)+global_id] - 0.5*dq[k*(ksize)+global_id];
		#endif
		}

		get_state(p_r,&geom,&state_r) ;
		primtoflux(p_r,&state_r,dir,&geom,temp3, GAMMA) ;
		primtoflux(p_r,&state_r,TT, &geom,temp4, GAMMA) ;
		vchar(p_r,&state_r,&geom,dir,&cmax_r,&cmin_r, GAMMA) ;

		cmax = fabs(MY_MAX(MY_MAX(0., storage4[global_id]),  cmax_r)) ;
		cmin = fabs(MY_MAX(MY_MAX(0.,-storage3[global_id]), -cmin_r)) ;
		ctop = MY_MAX(cmax,cmin) ;
		
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			F[k*(ksize)+global_id]=HLLF*((cmax*storage1[k*(ksize)+global_id] + 
			cmin*temp3[k] - cmax*cmin*(temp4[k] - storage2[k*(ksize)+global_id]))/(cmax + cmin + SMALL)) 
			+LAXF*(0.5*(storage1[k*(ksize)+global_id] 
			+ temp3[k] - ctop*(temp4[k] - storage2[k*(ksize)+global_id])));
		}

        //evaluate restriction on timestep
        cmax = MY_MAX(cmax,cmin) ;
		local_dtij[local_id] = factor/cmax ;
	}

	__syncthreads();
	for(i=local_size/2; i>1; i=i/2){
		if(local_id<i){
			local_dtij[local_id]=MY_MIN(local_dtij[local_id], local_dtij[local_id+i]);
		}
		__syncthreads();
	}
	if(local_id==0){
		dtij[group_id]=MY_MIN(local_dtij[0], local_dtij[1]);
	}
	
 }

 __global__ void fix_flux(int N1, int N2,int N3, int N1_MPI, int N1_MPI_offset, int N2_MPI, int N2_MPI_offset,  FTYPE2* F1,  FTYPE2* F2,  FTYPE2* F3)
 {
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int icurr, jcurr, zcurr;
	int k;
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	if(global_id<(N1+2*N1G)*(N3+2*N3G)){
		zcurr=global_id%(N3+2*N3G);
		icurr=(global_id-zcurr)/(N3+2*N3G);
		if(icurr>=N1G-D1 && zcurr>=N3G-D3 && icurr<N1+N1G+D1 && zcurr<N3+N3G+D3) {
			if(N2_MPI_offset == 0){
				F1[B2*(ksize)+icurr*isize + 1*(N3+2*N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize +2*(N3+2*N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + 1*(N3+2*N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize +2*(N3+2*N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 8	
				PLOOP F2[k*(ksize)+icurr*isize+ 2*(N3+2*N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll 8	
				for(k=0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + N2G*(N3+2*N3G) + zcurr] = 0.0;
				}
			}
			if(N2_MPI_offset+N2 == N2_MPI){
				F1[B2*(ksize)+icurr*isize + (N2+2)*(N3+2*N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize +(N2+1)*(N3+2*N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (N2+2)*(N3+2*N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize +(N2+1)*(N3+2*N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 8	
				PLOOP F2[k*(ksize)+icurr*isize +(N2+2)*(N3+2*N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll 8	
				for(k=0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + (N2+N2G)*(N3+2*N3G) + zcurr] = 0.0;
				}
			}
			
		}
	}
	#if INFLOW==0
	else{
		global_id=global_id-(N1+2*N1G)*(N3+2*N3G);
		zcurr=global_id%(N3+2*N3G);
		jcurr=(global_id-zcurr)/(N3+2*N3G);
		if(jcurr>=N2G-D2 && zcurr>=N3G-D3 && jcurr<N2+N2G+D2 && zcurr<N3+N3G+D3) {
			if(N1_MPI_offset == 0){
				if (F1[RHO*(ksize)+2*isize + jcurr*(N3+2*N3G) + zcurr] > 0.) F1[RHO*(ksize)+2*isize + jcurr*(N3+2*N3G) + zcurr] = 0.;
			}
			if(N1_MPI_offset+N1 == N1_MPI){
				if (F1[RHO*(ksize)+(N1+2)*isize + jcurr*(N3+2*N3G) + zcurr] < 0.) F1[RHO*(ksize)+(N1+2)*isize + jcurr*(N3+2*N3G) + zcurr] = 0.;
			}
		}

	}
	#endif
}

__global__ void flux_ct1(int N1, int N2,int N3, const FTYPE2* __restrict__ F1, const FTYPE2* __restrict__ F2, const FTYPE2* __restrict__ F3,  FTYPE2* emf) 
 {
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	if(icurr>=N1G && jcurr>=N2G && zcurr>=N3G && icurr<N1+N1G+D1 && jcurr<N2+N2G+D2  && zcurr<N3+N3G+D3){
		#if (N2G>0 && N3G>0)
		emf[1*(ksize)+global_id] = 0.25*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id-1] - 
		F3[B2*(ksize)+global_id] - F3[B2*(ksize)+global_id- (N3+2*N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		emf[2*(ksize)+global_id] = 0.25*(F3[B1*(ksize)+global_id] + F3[B1*(ksize)+global_id-isize] - 
		F1[B3*(ksize)+global_id] - F1[B3*(ksize)+global_id-1]);
		#endif
		#if (N1G>0 && N2G>0)
		emf[3*(ksize)+global_id] = 0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id-(N3+2*N3G)] - 
		F2[B1*(ksize)+global_id] - F2[B1*(ksize)+global_id-isize]);
		#else
		emf[3*(ksize)+global_id] =0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id-(N3+2*N3G)]);
		#endif
	}
}

__global__ void flux_ct2(int N1, int N2, int N3,  FTYPE2* F1,  FTYPE2* F2, FTYPE2* F3,  const FTYPE2* __restrict__ emf) 
 {
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	double emf1=emf[1*(ksize)+global_id];
	double emf2=emf[2*(ksize)+global_id];
	double emf3=emf[3*(ksize)+global_id];
	if(icurr>=N1G && jcurr>=N2G && zcurr>=N3G && icurr<N1+N1G+D1 && jcurr<N2+N2G && zcurr<N3+N3G){
		#if (N1G>0)
		F1[B1*(ksize)+global_id]=0.0;
		#endif
		#if (N1G>0 && N2G>0)
		F1[B2*(ksize)+global_id]=0.5*(emf3+emf[3*(ksize)+global_id+(N3+2*N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		F1[B3*(ksize)+global_id]=-0.5*(emf2 + emf[2*(ksize)+global_id+1]);
		#endif
	}
	if(icurr>=N1G && jcurr>=N2G && zcurr>=N3G && icurr<N1+N1G && jcurr<N2+N2G+D2 && zcurr<N3+N3G){
		#if (N1G>0 && N2G>0)		
		F2[B1*(ksize)+global_id] = -0.5*(emf3 + emf[3*(ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F2[B3*(ksize)+global_id] = 0.5*(emf1 + emf[1*(ksize)+global_id+1]);
		#endif
		#if(N2G!=1)
		F2[B2*(ksize)+global_id]=0.0;
		#endif
	}
	if(icurr>=N1G && jcurr>=N2G && zcurr>=N3G && icurr<N1+N1G && jcurr<N2+N2G && zcurr<N3+N3G+D3){
		#if (N1G>0 && N3G>0)
		F3[B1*(ksize)+global_id] = 0.5*(emf2 + emf[2*(ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F3[B2*(ksize)+global_id] = -0.5*(emf1 + emf[1*(ksize)+global_id +(N3+2*N3G)]);
		#endif
		#if(N3G>0)
		F3[B3*(ksize)+global_id] = 0.;
		#endif
	}
 } 
 
__global__ void diag_flux(int N1, int N2,  FTYPE2* F1,  FTYPE2* diagflux, FTYPE2 fractheta, int n1_MPI)
 {

 }

  __global__ void Utoprim(int N1, int N2, int N3,  FTYPE2* F1,  FTYPE2* F2, FTYPE2* F3, const FTYPE2* __restrict__ pi_i,  FTYPE2* pb_i,  FTYPE2* pf_i, const  FTYPE2* __restrict__  gcov, 
 const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet,const  FTYPE2* __restrict__  conn, FTYPE2 Dt, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 fractheta, FTYPE2 gam, FTYPE2* U_i, int N1_MPI,
 int N2_MPI, int N3_MPI)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int k;
	struct of_geom geom;
	struct of_state q ;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 U[NPR], pi[NPR];
	FTYPE2 dx[4];
	dx[1] = log((Rout - R0)/(Rin - R0))/(double)N1_MPI ;
    dx[2] = fractheta/(double)N2_MPI ;
	dx[3]=2.0*M_PI/(double)N3_MPI;
	if(icurr>=N1G && icurr<N1+N1G && jcurr>=N2G && jcurr<N2+N2G && zcurr>=N3G && zcurr<N3+N3G){
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			pi[k]=pi_i[k*(ksize)+global_id];
		}
		get_geometry(N1, N2, icurr, jcurr, CENT, &geom, gcov, gcon, gdet) ;
		get_state(pi,&geom,&q) ;
		primtoU(pi,&q,&geom,U, gam) ;
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			#if( N1G > 0 )
			U[k] -= Dt*(F1[k*(ksize)+global_id+isize]- F1[k*(ksize)+global_id])/dx[1];
			#endif
			#if( N2G > 0 )
			U[k] -=Dt* (F2[k*(ksize)+global_id+(N3+2*N3G)] - F2[k*(ksize)+global_id])/dx[2];
			#endif
			#if( N3G > 0 )
			U[k] -= Dt*(F3[k*(ksize)+global_id+1] - F3[k*(ksize)+global_id]) / (dx[3]);
			#endif
		}
		
		#pragma unroll 8	
		for(k=0; k< NPR; k++){
			U_i[k*(ksize)+global_id]=U[k];
		}
	} 
 }
__global__ void Utoprim1(int N1, int N2, int N3,  FTYPE2* F1,  FTYPE2* F2, FTYPE2* F3,  FTYPE2* pi_i, const FTYPE2* __restrict__ pb_i,  FTYPE2* pf_i, const  FTYPE2* __restrict__  gcov,  const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, 
const  FTYPE2* __restrict__  conn, FTYPE2 Dt, FTYPE2* U_i)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int k;
	struct of_geom geom;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 dU[NPR], pb[NPR];
	if(icurr>=N1G && icurr<N1+N1G && jcurr>=N2G && jcurr<N2+N2G && zcurr>=N3G && zcurr<N3+N3G){
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			pb[k]=pb_i[k*(ksize)+global_id];
		}
		get_geometry(N1, N2, icurr, jcurr, CENT, &geom, gcov, gcon, gdet) ;
		source(N1, N2, pb,&geom,icurr, jcurr,dU,Dt, GAMMA, conn) ;
		
		#pragma unroll 8	
		for(k=0; k< NPR; k++){
			U_i[k*(ksize)+global_id]+=Dt*dU[k];
		}
	} 
 }

__global__ void Utoprim2(int N1, int N2, int N3,  FTYPE2* F1,  FTYPE2* F2, FTYPE2* F3, const FTYPE2* __restrict__ pi_i,  FTYPE2* pb_i,  FTYPE2* pf_i, const  FTYPE2* __restrict__  gcov,  const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, 
const  FTYPE2* __restrict__  conn, FTYPE2 Dt, FTYPE2* U_i,  int *pflag,  int *failimage)
 {
 	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int k, flag;
	struct of_geom geom;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 U[NPR], pi[NPR];

	if(icurr>=N1G && icurr<N1+N1G && jcurr>=N2G && jcurr<N2+N2G && zcurr>=N3G && zcurr<N3+N3G){
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			pi[k]=pi_i[k*(ksize)+global_id];
			U[k]=U_i[k*(ksize)+global_id];
		}

		get_geometry(N1, N2, icurr, jcurr, CENT, &geom, gcov, gcon, gdet) ;	
		flag=Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pi);
		pflag[global_id]=flag;
		if (flag){
			failimage[global_id]++;
			#if(!DO_FONT_FIX )
			pflag[0]=100;
			#endif
		}
		
		#pragma unroll 8	
		for(k=0; k<NPR; k++){
			pf_i[k*(ksize)+global_id]=pi[k];
		}
	} 
 }

__global__ void fixup(int N1, int N2, int N3,  FTYPE2* pv_i,  int* pflag,  int* failimage,
 const  FTYPE2* __restrict__  gcov, const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 fractheta, FTYPE2 hslope, FTYPE2 gam,  FTYPE2* U_i
 ,int N1_MPI, int N1_MPI_offset, int N2_MPI, int N2_MPI_offset, int N3_MPI, int N3_MPI_offset)
{
  	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=(global_id%(isize))%(N3+2*N3G);
	int jcurr=((global_id-zcurr)%(isize))/(N3+2*N3G);
	int icurr=(global_id - (jcurr*(N3+2*N3G)+zcurr)) / (isize);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int k, flag, dofloor=0;
	FTYPE2 r,th,X[NDIM],uuscal,rhoscal, rhoflr, uuflr;
	FTYPE2 f,gamma,bsq ;
	FTYPE2 pv[NPR], pv_prefloor[NPR],U_ent, dpv[NPR], U_prefloor[NPR], dU[NPR], U[NPR];
	struct of_geom geom ;
	struct of_state q;
	int ksize=isize*(N1+2*N1G)+fix_mem1;

	if(icurr>=N1G && jcurr>=N2G && zcurr>=N3G && icurr<N1+N1G && jcurr<N2+N2G && zcurr<N3+N3G){
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			pv[k]=pv_i[k*(ksize)+global_id]; 
		}
		
		//compute the square of fluid frame magnetic field (twice magnetic pressure)
		get_geometry(N1, N2, icurr, jcurr, CENT, &geom, gcov, gcon, gdet) ;
		#if( DO_FONT_FIX ) 
		if (pflag[global_id]) {
			#pragma unroll 8
			for(k=0; k< NPR; k++){
				U[k]=U_i[k*(ksize)+global_id];
			}
			get_state(pv, &geom, &q);
			U_ent = (geom.g*pv[0] * (gam - 1.)*pv[1] / pow(pv[0], gam)) * (q.ucon[0]);
			pflag[global_id]=Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pv,  U_ent);
			if (pflag[global_id]) {
				failimage[1*(ksize) + global_id]++ ;
				//pflag[global_id]=Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pv, Katm[icurr]);
				if (pflag[global_id]){
					pflag[0]=100;
					failimage[2*(ksize) + global_id]++ ;
				}
			}
		}
		#endif

		coord(icurr+N1_MPI_offset-2, jcurr+N2_MPI_offset-2,0.0, CENT, X, fractheta, Rin, R0, Rout, N1_MPI,N2_MPI, N3_MPI);
		bl_coord(X, &r, &th, hslope, R0);

		rhoscal = pow(r,-POWRHO) ;
		uuscal = pow(rhoscal,gam);

		rhoflr = RHOMIN*rhoscal;
		uuflr  = UUMIN*uuscal;
	
		bsq = bsq_calc(pv,&geom) ;

		//tie floors to the local values of magnetic field and internal energy density
		if( rhoflr < bsq / BSQORHOMAX ) rhoflr = bsq / (BSQORHOMAX);
		if( uuflr < bsq / BSQOUMAX ) uuflr = bsq / (BSQOUMAX);
		if( rhoflr < pv[UU] / UORHOMAX ) rhoflr = pv[UU] / (UORHOMAX);

		if( rhoflr < RHOMINLIMIT ) rhoflr = RHOMINLIMIT;
		if( uuflr  < UUMINLIMIT  ) uuflr  = UUMINLIMIT;
		
		//floor on density and internal energy density (momentum *not* conserved) 
		#pragma unroll 8
		PLOOP pv_prefloor[k] = pv[k];
		if (pv[RHO] < rhoflr ){
			pv[RHO] = rhoflr;
			dofloor = 1;
		}
		if (pv[UU] < uuflr){
			pv[UU] = uuflr;
			dofloor = 1;
		}

		#if( ZAMO_FLOOR )
		if (dofloor==1) {
			#pragma unroll 8
			PLOOP dpv[k] = pv[k] - pv_prefloor[k];

			//compute the conserved quantity associated with floor addition
			get_state(dpv, &geom, &q);
			primtoU(dpv, &q, &geom, dU, gam);

			//compute the prefloor conserved quantity
			get_state(pv_prefloor, &geom, &q);
			primtoU(pv_prefloor, &q, &geom, U_prefloor, gam);

			//add U_added to the current conserved quantity
			#pragma unroll 8
			PLOOP U[k] = U_prefloor[k]+ dU[k];

			flag = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pv);
			if (flag){
				failimage[global_id]++;
				pflag[global_id]=flag;
				#if( DO_FONT_FIX ) 
				U_ent = (geom.g*pv[0] * (gam - 1.)*pv[1] / pow(pv[0], gam)) * (q.ucon[0]);
				pflag[global_id]=Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pv, U_ent);
				if (pflag[global_id]) {
					failimage[1*(ksize) + global_id]++ ;
					//pflag[global_id]=Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pv, Katm[icurr]);
					//pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pv, U_ent);
					if (pflag[global_id]){
						pflag[0]=100;
						failimage[2*(ksize) + global_id]++ ;
					}
				}
				#else
				pflag[0]=100;
				#endif	
			}
		}
		#endif

		// limit gamma wrt normal observer 
		if( gamma_calc(pv,&geom,&gamma) ) { 
			// Treat gamma failure here as "fixable" for fixup_utoprim() 
			pflag[global_id] = -333;
			pflag[0]=100;
			failimage[3*(ksize) + global_id]++ ;
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
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			pv_i[k*(ksize)+global_id]=pv[k];
		}
	}
}


__global__ void fixuputoprim(int N1, int N2, int N3,  FTYPE2* pv,  int* pflag,  int* failimage, const  FTYPE2* __restrict__  gcov,
const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 fractheta, FTYPE2 hslope, FTYPE2 gam)
{

}

__global__ void boundprim1(int N1, int N2, int N3,  FTYPE2* pv,  int* pflag, const  FTYPE2* __restrict__  gcov, 
const  FTYPE2* __restrict__  gcon, const  FTYPE2* __restrict__  gdet, int N1_MPI, int N1_MPI_offset, int N2_MPI, int N2_MPI_offset, int N3_MPI, int N3_MPI_offset)
{
  	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int k;
	int zcurr=global_id%(N3+2*N3G);
	int jcurr=(global_id-zcurr)/(N3+2*N3G);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	FTYPE2 prim1[NPR], prim2[NPR],prim3[NPR], prim4[NPR],prim5[NPR], prim6[NPR]; 
	int prim7, prim8;
	int fix_mem2=LOCAL_WORK_SIZE-((N2+2*N2G)*(N1+2*N1G))%LOCAL_WORK_SIZE;

	// inner r boundary condition: u, gdet extrapolation 
	if(jcurr>=N2G && jcurr<N2+N2G && zcurr>=0 && zcurr<N3+2*N3G && N1_MPI_offset == 0){
		/*if(freeze!=0){
			#pragma unroll 8
			for(k=0; k< NPR; k++){
				pv[k*(ksize)+(ibound)*isize+global_id]=pbound[k*(isize+fix_mem1)+global_id];
				pv[k*(ksize)+(ibound+1)*isize+global_id]=pbound[NPR*(isize+fix_mem1)+k*((N3+2*N3G)*(N2+2*N2G)+fix_mem1)+global_id];
			}
			pflag[(ibound)*isize+global_id]=0;
			pflag[(ibound+1)*isize+global_id]=0;
		}*/
		
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			prim5[k]=pv[k*(ksize)+2*isize+global_id];
		}
		prim7=pflag[0*isize+global_id] ;
		
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			prim1[k] = prim5[k] ;
			prim2[k] = prim5[k] ;
		}
		pflag[1*isize+global_id] = prim7  ;
		//pflag[0*isize+global_id] = prim7  ;
			
		/*Make sure there is no inflow at inner boundary*/
		inflow_check(N1, N2, prim1,0,jcurr,0, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim2,0,jcurr,0, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim1,1,jcurr,0, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim2,1,jcurr,0, gcov, gcon, gdet) ;

		/*Write primitives back to global memory*/
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			pv[k*(ksize)+1*isize+global_id]=prim1[k]*gdet[3*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 2*(N2+2*N2G)+ jcurr]/gdet[3*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 1*(N2+2*N2G) + jcurr];
			pv[k*(ksize)+global_id]                        =prim2[k]*gdet[3*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + 2*(N2+2*N2G)+ jcurr]/gdet[3*((N2+2*N2G)*(N1+2*N1G)+fix_mem2) + jcurr];
		}
		global_id=-10;
		jcurr=-10;
		zcurr=-10;
	}

	if(global_id>=isize){
		global_id=global_id-isize;
		zcurr=global_id%(N3+2*N3G);
		jcurr=(global_id-zcurr)/(N3+2*N3G);
	}
	// outer r BC: outflow 
	if(jcurr>=N2G && jcurr<N2+N2G && zcurr>=0 && zcurr<N3+2*N3G && N1_MPI_offset + N1 == N1_MPI){
		#pragma unroll 8
		for(k=0; k< NPR; k++){
			prim6[k]=pv[k*(ksize)+(N1+1)*isize+global_id];
		}
		
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			prim3[k] = prim6[k] ;
			prim4[k] = prim6[k] ;
		}
		prim8=pflag[(N1+1)*isize+global_id] ;
		pflag[(N1+2)*isize+global_id] = prim8 ;
		//pflag[(N1+3)*isize+global_id] = prim8 ;
	
		/*Make sure there is no inflow at outer boundary*/
		inflow_check(N1, N2, prim3,N1+2,jcurr,1, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim4,N1+2,jcurr,1, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim3,N1+3,jcurr,1, gcov, gcon, gdet) ;
		inflow_check(N1, N2, prim4,N1+3,jcurr,1, gcov, gcon, gdet) ;
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			pv[k*(ksize)+(N1+2)*isize+global_id]=prim3[k];
			pv[k*(ksize)+(N1+3)*isize+global_id]=prim4[k];
		}
	}
}

__global__ void boundprim2(int N1, int N2, int N3, FTYPE2* pv, int* pflag, int N1_MPI, int N1_MPI_offset, int N2_MPI, 
int N2_MPI_offset, int N3_MPI, int N3_MPI_offset)
{
	int j, jref, k;
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int zcurr=global_id%(N3+2*N3G);
	int icurr=(global_id-zcurr)/(N3+2*N3G);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	jref=POLEFIX;
	// polar BCs
	if(icurr>=0 && icurr<N1+2*N1G && zcurr>=0 && zcurr<N3+2*N3G && N2_MPI_offset == 0) { 
		for(j=0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = (j+0.5)/(jref+0.5) * pv[3*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];

			//everything else copy (both poles)
			pv[0*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[0*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			pv[1*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[1*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			pv[2*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[2*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			pv[4*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[4*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			#if (N3G>0)
			pv[5*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[5*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			pv[6*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[6*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
			#endif
			pv[7*(ksize)+isize*icurr+(j+2)*(N3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(jref+2)*(N3+2*N3G)+zcurr];
		}
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr+1*(N3+2*N3G)+zcurr] = pv[k*(ksize)+isize*icurr+2*(N3+2*N3G)+zcurr] ;
			pv[k*(ksize)+isize*icurr+zcurr] = pv[k*(ksize)+isize*icurr+3*(N3+2*N3G)+zcurr] ;
		}
		pflag[isize*icurr+1*(N3+2*N3G)+zcurr] = pflag[isize*icurr+2*(N3+2*N3G)+zcurr] ;
		//pflag[isize*icurr+zcurr] = pflag[isize*icurr+3*(N3+2*N3G)+zcurr] ;
		
		// make sure b and u are antisymmetric at the poles 
		for(j=0;j<2;j++) {
			pv[3*(ksize)+isize*icurr+j*(N3+2*N3G)+zcurr] *= -1. ;
			pv[6*(ksize)+isize*icurr+j*(N3+2*N3G)+zcurr] *= -1. ;
		}
		global_id=-10;
		icurr=-10;
		zcurr=-10;
	}

	if(global_id>=(N1+2*N1G)*(N3+2*N3G)){
		global_id=global_id-(N1+2*N1G)*(N3+2*N3G);
		zcurr=global_id%(N3+2*N3G);
		icurr=(global_id-zcurr)/(N3+2*N3G);
	}

	if(icurr>=0 && icurr<N1+2*N1G && zcurr>=0 && zcurr<N3+2*N3G && N2_MPI_offset + N2 == N2_MPI) {
		for(j=0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = (j+0.5)/(jref+0.5) * pv[3*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];

			//everything else copy (both poles)
			pv[0*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[0*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			pv[1*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[1*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			pv[2*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[2*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			pv[4*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[4*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			#if (N3G>0)
			pv[5*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[5*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			pv[6*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[6*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];
			#endif
			pv[7*(ksize)+isize*icurr+(N2-1-j+2)*(N3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(N2-1-jref+2)*(N3+2*N3G)+zcurr];		
		}
		#pragma unroll 8
		for(k=0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr+(N2+2)*(N3+2*N3G)+zcurr] = pv[k*(ksize)+isize*icurr+(N2+1)*(N3+2*N3G)+zcurr] ;
			pv[k*(ksize)+isize*icurr+(N2+3)*(N3+2*N3G)+zcurr] = pv[k*(ksize)+isize*icurr+(N2)*(N3+2*N3G)+zcurr] ;
		}
		pflag[isize*icurr+(N3+2*N3G)*(N2+2)+zcurr] = pflag[isize*icurr+(N2+1)*(N3+2*N3G)+zcurr] ;
		//pflag[isize*icurr+(N3+2*N3G)*(N2+3)+zcurr] = pflag[isize*icurr+(N2)*(N3+2*N3G)+zcurr] ;

		// make sure b and u are antisymmetric at the poles 
		for(j=N2+N2G;j<N2+2*N2G;j++) {
			pv[3*(ksize)+isize*icurr+j*(N3+2*N3G)+zcurr] *= -1. ;
			pv[6*(ksize)+isize*icurr+j*(N3+2*N3G)+zcurr] *= -1. ;
		}
	}
}

__global__ void boundprim3(int N1, int N2, int N3, FTYPE2* pv, int* pflag, int N1_MPI, int N1_MPI_offset, int N2_MPI, 
int N2_MPI_offset, int N3_MPI, int N3_MPI_offset)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize=(N3+2*N3G)*(N2+2*N2G);
	int jcurr=global_id%(N2+2*N2G);
	int icurr=(global_id-jcurr)/(N2+2*N2G);
	int fix_mem1=LOCAL_WORK_SIZE-(isize*(N1+2*N1G))%LOCAL_WORK_SIZE;
	int k, z;
	int ksize=isize*(N1+2*N1G)+fix_mem1;
	//Inner phi-boundary
	if (icurr>=0 && jcurr>=0 && icurr<N1+2*N1G && jcurr<N2+2*N2G && N3_MPI_offset == 0){
		for (z = 0; z < N3G; z++){
			#pragma unroll 8
			for(k=0; k<NPR; k++){
				pv[k*(ksize)+icurr*isize+jcurr*(N3+2*N3G)+z] = pv[k*(ksize)+icurr*isize+jcurr*(N3+2*N3G)+(N3+z)];
			}
		}
		pflag[icurr*isize+jcurr*(N3+2*N3G)+D3] = pflag[icurr*isize+jcurr*(N3+2*N3G)+(N3+D3)];
		global_id=-10;
		jcurr=-10;
		icurr=-10;
	}

	if(global_id>=(N1+2*N1G)*(N2+2*N2G)){
		global_id=global_id-(N1+2*N1G)*(N2+2*N2G);
		jcurr=global_id%(N2+2*N2G);
		icurr=(global_id-jcurr)/(N2+2*N2G);
	}

	//Outer phi-boundary
	if (icurr>=0 && jcurr>=0 && icurr<N1+2*N1G && jcurr<N2+2*N2G && N3_MPI_offset + N3 == N3_MPI){
		for (z = N3G; z < 2*N3G; z++){
			#pragma unroll 8
			for(k=0; k<NPR; k++){
				pv[k*(ksize)+icurr*isize+jcurr*(N3+2*N3G)+(N3+z)] = pv[k*(ksize)+icurr*isize+jcurr*(N3+2*N3G)+z];
			}
		}
		pflag[icurr*isize+jcurr*(N3+2*N3G)+(N3+N3G)] = pflag[icurr*isize+jcurr*(N3+2*N3G)+N3G];
	}
}
extern "C" {


	void GPU_init(void)
	{
		
		/*N1 and N2 values to be exported to GPU memory*/
		N1_tot = N1_MPI;
		N2_tot = N2_MPI;
		N3_tot = N3_MPI;
		int freeze_GPU;
	
		/*Set the global work size and make sure that it is a multiple of the group size. The Nvidia OpenCL framework crashes otherwise!*/
		fix_mem = LOCAL_WORK_SIZE - ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G)) % LOCAL_WORK_SIZE;
		fix_mem2 = LOCAL_WORK_SIZE - ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G)) % LOCAL_WORK_SIZE;
		global_work_offset[0] = ibound*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G); //alert
		global_work_size[0] = (LOCAL_WORK_SIZE - ((N1_MPI + 2 * N1G)*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) - global_work_offset[0]) % LOCAL_WORK_SIZE) + ((N1_MPI + 2 * N1G)*(N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) - global_work_offset[0]);
		global_work_size_bound[0] = (LOCAL_WORK_SIZE - (2 * (N2_tot + 4) + 2 * (N1_tot + 4)) % LOCAL_WORK_SIZE) + 2 * (N2_tot + 4) + 2 * (N1_tot + 4);
		global_work_size_special[0] = (LOCAL_WORK_SIZE - ((N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) + (N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) + (N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G);
		global_work_size_special1[0] = (LOCAL_WORK_SIZE - ((N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) * 2) % LOCAL_WORK_SIZE) + (N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G) * 2;
		global_work_size_special2[0] = (LOCAL_WORK_SIZE - ((N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) * 2 ) % LOCAL_WORK_SIZE) + ((N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G) * 2);
		global_work_size_special3[0] = (LOCAL_WORK_SIZE - ((N1_MPI + 2 * N1G)*(N2_MPI + 2 * N2G) * 2) % LOCAL_WORK_SIZE) + ((N1_MPI + 2 * N1G)*(N2_MPI + 2 * N2G) * 2);
		local_work_size[0] = LOCAL_WORK_SIZE;
		nr_workgroups = (int)ceil((double)global_work_size[0] / (double)LOCAL_WORK_SIZE);

		/*Allocate memory to 1D arrays*/
		p_1 = (FTYPE2(*))malloc(NPR*((N3_MPI + 2*N3G)*(N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem)*sizeof(FTYPE2));
		#if(GPU_DEBUG)
		ph_1 = (FTYPE2(*))malloc(NPR*((N3_MPI + 2*N3G)*(N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem)*sizeof(FTYPE2));
		#endif
		pbound_1 = (FTYPE2(*))malloc(N_POINTS*NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + fix_mem)*sizeof(FTYPE2));
		gcov_GPU = (FTYPE2(*))malloc(((N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2));
		gcon_GPU = (FTYPE2(*))malloc(((N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2));
		conn_GPU = (FTYPE2(*))malloc(((N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem2)*NDIM*NDIM*NDIM*sizeof(FTYPE2));
		gdet_GPU = (FTYPE2(*))malloc(((N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem2)*NPG*sizeof(FTYPE2));
		pflag_GPU = (int(*))malloc(((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(int));
		failimage_GPU = (int(*))malloc(((N3_MPI + 2*N3G)*(N2_MPI + 2*N2G)*(N1_MPI + 2*N1G) + fix_mem) * NFAIL * sizeof(int));
		Katm_GPU = (FTYPE2(*))malloc((N1_MPI + 2*N1G)*sizeof(FTYPE2));
		dtij_GPU = (FTYPE2(*))malloc(nr_workgroups*sizeof(FTYPE2));

		if (boundfreeze1 == 1 || boundfreeze2==1){
			freeze_GPU = N_POINTS;
		}
		else{
			freeze_GPU = 0; //alert
		}

		/*Set cache config*/
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

		/*Allocate memory to buffers on GPU*/
		cudaMalloc(&BufferF1_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&BufferF2_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&BufferF3_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferdq_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferstorage1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferstorage2, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferstorage3, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferstorage4, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferp_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Bufferph_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2));
		//cudaMalloc(&Bufferpbound_1, N_POINTS*NPR*((N2_MPI + 2 * N2G) + fix_mem)*sizeof(FTYPE2));
		cudaMalloc(&Buffergcov, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2));
		cudaMalloc(&Buffergcon, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2));
		cudaMalloc(&Bufferconn, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NDIM*NDIM*NDIM*sizeof(FTYPE2));
		cudaMalloc(&Buffergdet, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*sizeof(FTYPE2));
		cudaMalloc(&Bufferdtij, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) / LOCAL_WORK_SIZE*sizeof(FTYPE2));
		cudaMalloc(&Bufferpflag, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(int));
		cudaMalloc(&Bufferfailimage, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) * NFAIL * sizeof(int));
		cudaMalloc(&Bufferdiagflux, 3 * MY_MAX((N1_MPI + 2 * N1G)*(N3_MPI + 2 * N3G), (N2_MPI + 2 * N2G)*(N3_MPI + 2 * N3G))*sizeof(FTYPE2));
		cudaMalloc(&BufferKatm, (N1_MPI + 2 * N1G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundrec1_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N1_MPI + 2 * N1G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundrec2_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundrec3_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N1_MPI + 2 * N1G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundrec4_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundsend1_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N1_MPI + 2 * N1G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundsend2_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundsend3_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N1_MPI + 2 * N1G)*sizeof(FTYPE2));
		cudaMalloc(&Bufferboundsend4_MPI, 2 * NPR*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*sizeof(FTYPE2));
		cudaMalloc(&Buffercornrec1_MPI, 4 * (N3_MPI + 2 * N3G)*NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornrec2_MPI, 4 * (N3_MPI + 2 * N3G)* NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornrec3_MPI, 4 * (N3_MPI + 2 * N3G)*NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornrec4_MPI, 4 * (N3_MPI + 2 * N3G)* NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornsend1_MPI, 4 * (N3_MPI + 2 * N3G)*NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornsend2_MPI, 4 * (N3_MPI + 2 * N3G)* NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornsend3_MPI, 4 * (N3_MPI + 2 * N3G)* NPR*sizeof(FTYPE2));
		cudaMalloc(&Buffercornsend4_MPI, 4 * (N3_MPI + 2 * N3G)*NPR*sizeof(FTYPE2));
	

	}
	double check = 1.0;
	void GPU_write(void)
	{
		int i, j, z, k,l, pg ,d1, d2;
		
	ZSLOOP(N1_MPI_offset-N1G, N1_MPI_offset + N1_MPI - 1 + N1G, N2_MPI_offset - N2G, N2_MPI_offset + N2_MPI - 1 + N2G){
		for (pg = 0; pg < NPG; pg++){
			gdet_GPU[pg*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + (i - N1_MPI_offset + N1G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)] = gdet[index2(i, j)][pg];
			for (d1 = 0; d1 < NDIM; d1++){
				for (d2 = 0; d2 < NDIM; d2++){
					gcov_GPU[d1*NDIM*NPG*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + d2*NPG*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + pg*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + (i - N1_MPI_offset + N1G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)] = gcov[index2(i, j)][pg][d1][d2];
					gcon_GPU[d1*NDIM*NPG*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + d2*NPG*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + pg*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + (i - N1_MPI_offset + N1G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)] = gcon[index2(i, j)][pg][d1][d2];
				}
			}
		}
		for (pg = 0; pg < NDIM; pg++){
			for (d1 = 0; d1 < NDIM; d1++){
				for (d2 = 0; d2 < NDIM; d2++){
					conn_GPU[d1*NDIM*NDIM*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + d2*NDIM*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + pg*((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2) + (i - N1_MPI_offset + N1G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)] = conn[index2(i, j)][pg][d1][d2];
				}
			}
		}
	}
		
		for (i = N1_MPI_offset; i < N1_MPI_offset+N1_MPI; i++){
			//Katm_GPU[i - N1_MPI_offset + N1G] = Katm[i];
		}
		ZSLOOP3D(N1_MPI_offset - N1G, N1_MPI_offset + N1_MPI - 1 + N1G, N2_MPI_offset - N2G, N2_MPI_offset + N2_MPI - 1 + N2G, N3_MPI_offset - N3G, N3_MPI_offset + N3_MPI - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)] = p[index(i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)] = ph[index(i, j, z)][k];
				#endif
				#if(ibound)
				if (i == N1_MPI_offset){
					for (l = 0; l < N_POINTS; l++){
						pbound_1[l*NPR*((N3_MPI + 2*N3G)*(N2_MPI + 2*N2G) + fix_mem) + k*((N2_MPI + 2*N2G) + fix_mem) + (j - N2_MPI_offset + N2G)] = pbound[j*(N3_MPI+2*N3G)+z][k][l];
					}	
				}
				#endif
			}
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)] = failimage[index(i, j, z)][k];
			}
			pflag_GPU[(i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)] = 1;
		}
		/*Initialize memory items that have to be passed on to the GPU*/
		cudaMemcpy(Bufferp_1, p_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#if (GPU_DEBUG)
		cudaMemcpy(Bufferph_1, ph_1, NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#endif
		cudaMemcpy(Bufferpflag, pflag_GPU, ((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(Bufferpbound_1, pbound_1, N_POINTS*NPR*((N2_MPI + 2 * N2G) + fix_mem)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		cudaMemcpy(Buffergcov, gcov_GPU, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		cudaMemcpy(Buffergcon, gcon_GPU, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		cudaMemcpy(Bufferconn, conn_GPU, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NDIM*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		cudaMemcpy(Buffergdet, gdet_GPU, ((N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem2)*NPG*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		cudaMemcpy(BufferKatm, Katm_GPU, (N1_MPI + 2 * N1G)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
		
		free(pflag_GPU);
		pflag_GPU = (int(*))malloc(1*sizeof(int));
	}

	double fluxcalc_GPU(int dir, int flag)
	{
		double ndt;
		int y;
		ndt = 1.e9;
		GPU_fluxcalcprep(dir, flag);
		GPU_fluxcalc2D(dir, flag);
		cudaMemcpy(dtij_GPU, Bufferdtij, (int)nr_workgroups*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
				
		//printf("dq: %f \n", dtij_GPU[10]);
		for (y = 0; y <nr_workgroups-1; y++){
			if (dtij_GPU[y] < ndt && dtij_GPU[y] <1.e9){
				ndt = dtij_GPU[y];
			}
		}
		//printf("ndt: %f \n", ndt);
		return(ndt) ;
	}

	void GPU_fluxcalcprep(int dir, int flag)
	{	
		/*Set arguments of kernel*/
		if (dir == 1){
			if (flag == 1){
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferph_1, dir, lim);
			}
			else{
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferp_1, dir, lim);
			}
		}
		else if (dir == 2){
			if (flag == 1){
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferph_1, dir, lim);
			}
			else{
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferp_1, dir, lim);
			}
		}
		else{
			if (flag == 1){
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferph_1, dir, lim);
			}
			else{
				fluxcalcprep <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferp_1, dir, lim);
			}
		}


	}

	void GPU_fluxcalc2D(int dir, int flag)
	{  
		/*Set arguments of kernel*/
		if (dir == 1){
			if (flag == 1){
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferph_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0,fractheta, dir, N1_tot, N2_tot, N3_tot);
			}
			else{
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferp_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, Bufferdq_1, Bufferp_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0,fractheta, dir, N1_tot, N2_tot, N3_tot);
			}
		}
		else if (dir == 2){
			if (flag == 1){
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferph_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0,fractheta, dir, N1_tot, N2_tot, N3_tot);
			}
			else{
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferp_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF2_1, Bufferdq_1, Bufferp_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0, fractheta,dir, N1_tot, N2_tot, N3_tot);
			}
		}
		else{
			if (flag == 1){
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferph_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0,fractheta, dir, N1_tot, N2_tot, N3_tot);
			}
			else{
				fluxcalc2D1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferp_1, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4, dir);
				fluxcalc2D2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF3_1, Bufferdq_1, Bufferp_1,Bufferdtij, Buffergcov, Buffergcon, Buffergdet, Bufferstorage1,
					Bufferstorage2, Bufferstorage3, Bufferstorage4,cour, Rout,Rin, R0, fractheta, dir, N1_tot, N2_tot, N3_tot);
			}
		}
	}

	void GPU_fix_flux(void)
	{
		fix_flux <<< global_work_size_special[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI,N1_tot,N1_MPI_offset,
			N2_tot, N2_MPI_offset, BufferF1_1, BufferF2_1, BufferF3_1);
	}

	void GPU_flux_ct1(void)
	{
		flux_ct1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferdq_1);
	}

	void GPU_flux_ct2(void)
	{
		flux_ct2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferdq_1);
	}

	void GPU_diag_flux(void)
	{

	}

	void GPU_Utoprim(int flag, double Dt)
	{

		if (flag == 1){
			Utoprim <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferph_1, Bufferp_1,Buffergcov,Buffergcon,Buffergdet,Bufferconn,Dt,Rin,R0,Rout,fractheta,gam,Bufferdq_1, N1_tot, N2_tot, N3_tot);
			Utoprim1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferph_1, Bufferp_1, Buffergcov, Buffergcon, Buffergdet, Bufferconn, Dt, Bufferdq_1);
			Utoprim2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferph_1, Bufferp_1, Buffergcov, Buffergcon, Buffergdet, Bufferconn, Dt, Bufferdq_1, Bufferpflag, Bufferfailimage);
		}
		else{
			Utoprim <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferp_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferconn, Dt, Rin, R0, Rout, fractheta, gam, Bufferdq_1,N1_tot, N2_tot, N3_tot);
			Utoprim1 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferp_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferconn, Dt, Bufferdq_1);
			Utoprim2 <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, BufferF1_1, BufferF2_1, BufferF3_1, Bufferp_1,
				Bufferp_1, Bufferph_1, Buffergcov, Buffergcon, Buffergdet, Bufferconn, Dt,  Bufferdq_1, Bufferpflag, Bufferfailimage);
		}
	}

	void GPU_fixuputoprim(int flag)
	{

	}

	void GPU_fixup(int flag)
	{
		if (flag == 0){
			fixup <<< nr_workgroups, local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferph_1, Bufferpflag, Bufferfailimage,
				Buffergcov, Buffergcon, Buffergdet,  Rin, R0, Rout, fractheta, hslope, gam, Bufferdq_1, N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
		else{
			fixup << < nr_workgroups, local_work_size[0] >> > (N1_MPI, N2_MPI, N3_MPI, Bufferp_1, Bufferpflag, Bufferfailimage,
				Buffergcov, Buffergcon, Buffergdet, Rin, R0, Rout, fractheta, hslope, gam, Bufferdq_1, N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
	}

	void GPU_boundprim(int flag, int MPI)
	{
	#if(N1G>0)
		if (n1_MPI == n_columns - 1 || n1_MPI == 0) GPU_boundprim1(flag);
	#endif
	#if(N2G>0)
		if (n2_MPI == n_rows - 1 || n2_MPI == 0)GPU_boundprim2(flag);
	#endif
	#if(N3G>0)
		if (n3_MPI == n_stacks - 1 || n3_MPI == 0)GPU_boundprim3(flag);
	#endif
		if ((n_rows > 1 || n_columns > 1) && (MPI == 1)){
			GPU_boundsend(flag);
			GPU_boundrec(flag);
		}
	}

	void GPU_boundprim1(int flag)
	{
		if (flag == 0){
			boundprim1 <<< global_work_size_special1[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferph_1, Bufferpflag,
				Buffergcov,Buffergcon,Buffergdet, N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
		else{
			boundprim1 <<< global_work_size_special1[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferp_1, Bufferpflag,
				Buffergcov, Buffergcon, Buffergdet, N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
	}

	void GPU_boundprim2(int flag)
	{
		if (flag == 0){
			boundprim2 <<< global_work_size_special2[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferph_1, Bufferpflag,
				 N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
		else{
			boundprim2 <<< global_work_size_special2[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferp_1, Bufferpflag,
				N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
	}

	void GPU_boundprim3(int flag)
	{
		if (flag == 0){
			boundprim3 <<< global_work_size_special3[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferph_1, Bufferpflag,
				N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
		else{
			boundprim3 <<< global_work_size_special3[0] / local_work_size[0], local_work_size[0] >>> (N1_MPI, N2_MPI, N3_MPI, Bufferp_1, Bufferpflag,
				N1_tot, N1_MPI_offset, N2_tot, N2_MPI_offset, N3_tot, N3_MPI_offset);
		}
	}

	void GPU_boundsend(int flag){

	}

	void GPU_boundrec(int flag){

	}

	void GPU_read(void)
	{
		int i, j, z, k;

		cudaMemcpy(p_1, Bufferp_1, (int)(NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
		#if(GPU_DEBUG)
		cudaMemcpy(ph_1, Bufferph_1, (int)(NPR*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
		#endif
		cudaMemcpy(failimage_GPU, Bufferfailimage, (int)((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) * NFAIL * sizeof(int), cudaMemcpyDeviceToHost);
	
		ZSLOOP3D(N1_MPI_offset - N1G, N1_MPI_offset + N1_MPI - 1 + N1G, N2_MPI_offset - N2G, N2_MPI_offset + N2_MPI - 1 + N2G, N3_MPI_offset - N3G, N3_MPI_offset + N3_MPI - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[index(i, j, z)][k] = p_1[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)];
				#if(GPU_DEBUG)
				ph[index(i, j, z)][k] = ph_1[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)];
				#endif
			}
			for (k = 0; k<NFAIL; k++){
				failimage[index(i, j, z)][k] = failimage_GPU[k*((N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G)*(N1_MPI + 2 * N1G) + fix_mem) + (i - N1_MPI_offset + N1G)*(N3_MPI + 2 * N3G)*(N2_MPI + 2 * N2G) + (j - N2_MPI_offset + N2G)*(N3_MPI + 2 * N3G) + (z - N3_MPI_offset + N3G)];
			}
		}
	}
}