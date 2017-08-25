extern "C" {
#include "decsCUDA.h"
}
#include <cuda.h>
/*Whether or not to use the 3D version of the code*/
#define ThreeD (1)

/*Wheter to cool the disk to predifined thickness H_OVER_R*/
#define COOL_DISK (0)
#define H_OVER_R (0.03)

/*Wheter to set floors in lab frame*/
#define ZAMO_FLOOR (1)

/*Set the pixel value of the frozen boundary*/
#define ibound 0

/*Whether or not to fix the Carbuncle problem*/
#define H_CORRECTION 0

/*Whether or not to allow inflow for fluxes (see fix_flux())*/
#define INFLOW 0

/*Set block size in each dimension*/
#define BS_1 40
#define BS_2 50
#define BS_3 12

#define STAGGERED (1)

/*Wheter or not to use a non symmetric metric for tilted disk*/
#define NONSYMMETRIC (0)

/*Wheter or not to use the full dispersion relation*/
#define FULL_DISP (0)

/* whether or not to use Font's  adiabatic/isothermal prim. var. inversion method: */
#define DO_FONT_FIX (1)

/*Whether or not to use the PPM slope limiter*/
#define PPM (0)

/*Whether or not to use the general relativistic van Leer slope limiter*/
#define LEER (0)

/*Enable fma operations for improved performance on AMD GPUs*/
//#define OPENCLBUILDOPTIONS "-cl-mad-enable"
#define AMD 0

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
#define POLEFIX 2

/* your choice of floating-point data type */
#define FTYPE double
#define FTYPE2 double

#define DOKTOT 1
#define KTOT 8

/*Define variable quantities*/
#define NPG 5
#define NPR 9
#define NDIM 4

/** FIXUP PARAMETERS, magnitudes of rho and u, respectively, in the floor : **/
#define RHOMIN	(1.0*1.e-6)
#define UUMIN	(1.0*1.e-7)
#define RHOMINLIMIT (1.e-20)
#define UUMINLIMIT  (1.e-20)
#define POWRHO (2.0)
#define FLOORFACTOR (1.0)
#define BSQORHOMAX (50.*FLOORFACTOR)
#define BSQOUMAX (750.*FLOORFACTOR)
#define UORHOMAX (150.*FLOORFACTOR)

/* Max. value of gamma, the lorentz factor */
#define GAMMAMAX (80.)

/*Define misc. MACROs*/
#if AMD
#define dot(a,b) (fma(a[0],b[0],fma( a[1],b[1],fma( a[2],b[2] , a[3]*b[3]))))
#else
#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3])
#endif
#define delta(i,j) (((i) == (j)) ? 1. : 0.)
#define MY_MIN(fval1,fval2) ( ((fval1) < (fval2)) ? (fval1) : (fval2))
#define MY_MAX(fval1,fval2) ( ((fval1) > (fval2)) ? (fval1) : (fval2))
#define MY_SIGN(fval) ( ((fval) <0.) ? -1. : 1. )

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
#define HLLF  (1.00)
#define LAXF  (0.00)

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
__device__ FTYPE pressure_of_rho2(FTYPE rho0, FTYPE K_atm);
__device__ FTYPE dWdvsq_calc2(FTYPE vsq, FTYPE rho, FTYPE p);
__device__ int Utoprim_new_body2(FTYPE U[], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[], FTYPE K_atm);
__device__ void func_1d_gnr2(FTYPE x[], FTYPE dx[], FTYPE resid[], FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm);
__device__ void validate_x2(FTYPE x[1], FTYPE x0[1]);
__device__ int general_newton_raphson2(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm);
__device__ int Utoprim_1dvsq2fix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K);
__device__ void func_gnr2_rho(FTYPE x[], FTYPE dx[], FTYPE resid[], FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2);
__device__ raise_g(FTYPE vcov[], FTYPE gcon[][NDIM], FTYPE ucon[]);
__device__ lower_g(FTYPE vcon[], FTYPE gcov[][NDIM], FTYPE ucov[]);
__device__ ncov_calc(FTYPE gcon[][NDIM], FTYPE ncov[]);
__device__ bcon_calc_g(FTYPE prim[], FTYPE ucon[], FTYPE ucov[], FTYPE ncov[], FTYPE bcon[]);
FTYPE pressure_rho0_u(FTYPE rho0, FTYPE u);
FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w);
__device__ int Utoprim_1dfix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K);
__device__ int Utoprim_new_body3(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K_atm);
__device__ FTYPE pressure_of_rho3(FTYPE rho0, FTYPE K_atm);
__device__ FTYPE vsq_calc3(FTYPE W, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm);
__device__ int general_newton_raphson3(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2, FTYPE rho_for_gnr2, FTYPE W_for_gnr2_old, FTYPE rho_for_gnr2_old);
__device__ void func_1d_orig1(FTYPE x[], FTYPE dx[], FTYPE resid[],
	FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2, FTYPE rho_for_gnr2, FTYPE W_for_gnr2_old, FTYPE rho_for_gnr2_old);
__device__ int gnr2(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2);

__device__ int Utoprim_1dfix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K)
{
	FTYPE U_tmp[NPR], prim_tmp[NPR];
	int i, j, ret;
	FTYPE alpha, K_atm;

	if (U[0] <= 0.) {
		return(-100);
	}
	K_atm = K;

	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	alpha = 1.0 / sqrt(-gcon[0][0]);

	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	for (i = UTCON1; i <= UTCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}
	for (i = BCON1; i <= BCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}

	for (i = 0; i < BCON1; i++) {
		prim_tmp[i] = prim[i];
	}
	for (i = BCON1; i <= BCON3; i++) {
		prim_tmp[i] = alpha*prim[i];
	}

	ret = Utoprim_new_body3(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm);
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}
	return(ret);
}


__device__ int Utoprim_new_body3(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM],
	FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K_atm)
{

	FTYPE x_1d[1];
	FTYPE QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	FTYPE rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq, tmpdiff;
	int i, j, retval, i_increase;
	FTYPE W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
	FTYPE Bsq, QdotBsq, Qtsq, Qdotn, D;
	retval = 0;

	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	Bcon[0] = 0.;
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower_g(Bcon, gcov, Bcov);

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

	D = U[RHO];

	utsq = 0.;
	for (i = 1; i<4; i++)
		for (j = 1; j<4; j++) utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];


	if ((utsq < 0.) && (fabs(utsq) < 1.0e-13)) {
		utsq = fabs(utsq);
	}
	if (utsq < 0. || utsq > UTSQ_TOO_BIG) {
		retval = 2;
		return(retval);
	}

	gammasq = 1. + utsq;
	gamma = sqrt(gammasq);

	rho0 = D / gamma;
	p = pressure_of_rho3(rho0, K_atm);
	u = u_of_p2(p);
	w = rho0 + u + p;

	W_last = w*gammasq;

	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq)
		- QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq))
		&& (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	W_for_gnr2 = W_for_gnr2_old = W_last;
	rho_for_gnr2 = rho_for_gnr2_old = rho0;

	x_1d[0] = W_last;
	retval = general_newton_raphson3(x_1d, 1, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old);

	W = x_1d[0];

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

	vsq = vsq_calc3(W, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm);
	if (vsq >= 1.) {
		retval = 4;
		return(retval);
	}

	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;
	rho0 = D * gtmp;

	w = W * (1. - vsq);

	p = pressure_of_rho3(rho0, K_atm);
	u = u_of_p2(p);

	if ((rho0 <= 0.) || (u <= 0.)) {
		retval = 5;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;


	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];
	return(retval);
}

__device__ FTYPE pressure_of_rho3(FTYPE rho0, FTYPE K_atm)
{
	return(K_atm * pow(rho0, G_ATM));
}

__device__ FTYPE vsq_calc3(FTYPE W, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm)
{
	FTYPE Wsq, Xsq;
	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);
	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
}

__device__ int general_newton_raphson3(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2, FTYPE rho_for_gnr2, FTYPE W_for_gnr2_old, FTYPE rho_for_gnr2_old)
{
	FTYPE f, df, dx[NEWT_DIM2], x_old[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	FTYPE errx, x_orig[NEWT_DIM2];
	int    n_iter, id, jd, i_extra, doing_extra;
	FTYPE dW, dvsq, vsq_old, vsq, W, W_old;
	int   keep_iterating, i_increase;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	for (id = 0; id < n; id++)  x_old[id] = x_orig[id] = x[id];

	vsq_old = vsq = W = W_old = 0.;

	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		#if( USE_ISENTROPIC )   
		func_1d_orig1(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old);  /* returns with new dx, f, df */
		#endif

		errx = 0.;
		for (id = 0; id < n; id++) {
			x_old[id] = x[id];
		}

		for (id = 0; id < n; id++) {
			x[id] += dx[id];
		}

		i_increase = 0;
		while (((x[0] * x[0] * x[0] * (x[0] + 2.*Bsq) -
			QdotBsq*(2.*x[0] + Bsq)) <= x[0] * x[0] * (Qtsq - Bsq*Bsq))
			&& (i_increase < 10)) {
			x[0] -= (1.*i_increase) * dx[0] / 10.;
			i_increase++;
		}
		errx = (x[0] == 0.) ? fabs(dx[0]) : fabs(dx[0] / x[0]);
		x[0] = fabs(x[0]);

		if ((fabs(errx) <= NEWT_TOL) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) {
			doing_extra = 1;
		}

		if (doing_extra == 1) i_extra++;

		if (((fabs(errx) <= NEWT_TOL) && (doing_extra == 0)) ||
			(i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		n_iter++;
	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
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

__device__ int gnr2(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2)
{
	FTYPE f, df, dx[NEWT_DIM2], x_old[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	FTYPE errx, x_orig[NEWT_DIM2];
	int    n_iter, id, jd, i_extra, doing_extra;
	FTYPE dW, dvsq, vsq_old, vsq, W, W_old;
	int   keep_iterating;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	for (id = 0; id < n; id++)  x_old[id] = x_orig[id] = x[id];
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		func_gnr2_rho(x, dx, resid, jac, &f, &df, n, D, K_atm, W_for_gnr2);  /* returns with new dx, f, df */

		errx = 0.;
		for (id = 0; id < n; id++) {
			x_old[id] = x[id];
		}

		/* Make the newton step: */
		for (id = 0; id < n; id++) {
			x[id] += dx[id];
		}

		/* Calculate the convergence criterion */
		for (id = 0; id < n; id++) {
			errx += (x[id] == 0.) ? fabs(dx[id]) : fabs(dx[id] / x[id]);
		}
		errx /= 1.*n;

		x[0] = fabs(x[0]);

		if ((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) {
			doing_extra = 0;
		}

		if (doing_extra == 1) i_extra++;

		if (((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0)) ||
			(i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;

	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
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

//isentropic version:   eq.  (27)
__device__ void func_1d_orig1(FTYPE x[], FTYPE dx[], FTYPE resid[],
	FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2, FTYPE rho_for_gnr2, FTYPE W_for_gnr2_old, FTYPE rho_for_gnr2_old)
{
	int retval, ntries;
	FTYPE  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	FTYPE  t32, t33, t34, t38, t5, t51, t67, t8, W, x_rho[1], rho, rho_g;

	W = x[0];
	W_for_gnr2_old = W_for_gnr2;
	W_for_gnr2 = W;

	// get rho from NR:
	rho_g = x_rho[0] = rho_for_gnr2;

	ntries = 0;
	while ((retval = gnr2(x_rho, 1, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2)) && (ntries++ < 10)) {
		rho_g *= 10.;
		x_rho[0] = rho_g;
	}

	rho_for_gnr2_old = rho_for_gnr2;
	rho = rho_for_gnr2 = x_rho[0];

	Dc = D;
	t1 = Dc*Dc;
	t2 = QdotBsq*t1;
	t3 = t2*Bsq;
	t5 = Bsq*Bsq;
	t8 = t1*Bsq;
	t10 = t1*W;
	t21 = W*W;
	t23 = rho*rho;
	t26 = 1 / t1;
	resid[0] = (t3 + (2.0*t2 + ((Qtsq - t5)*t1
		+ (-2.0*t8 - t10)*W)*W)*W + (t5 + (2.0*Bsq + W)*W)*t21*t23)*t26 / t21;
	t29 = t1*t1;
	t30 = QdotBsq*t29;
	t32 = GAMMA*K_atm;
	t33 = pow(rho, 1.0*GAMMA);
	t34 = t32*t33;
	t38 = t23 * t33;
	t51 = GAMMA*t1*K_atm*t33;
	t67 = t21*W;
	jac[0][0] = -2.0*(t30*Bsq*t34 + (t30*t34
		+ ((-t38*Bsq*t32 + Bsq*GAMMA*t1*K_atm*t33)*t1
		+ (-t38*GAMMA*K_atm + t51)*t1*W)*t21)*W
		+ ((-t3 + (-t2 + (-t8 - t10)*t21)*W)*W + (-t5 - Bsq*W)*t67*t23)*t23)*t26 / (t51 - W*t23) / t67;

	dx[0] = -resid[0] / jac[0][0];

	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);

	return;
}


__device__ raise_g(FTYPE ucov[NDIM], FTYPE gcon[NDIM][NDIM], FTYPE ucon[NDIM])
{
	#if AMD
	ucon[0] = fma(gcon[0][0], ucov[0], fma(
		gcon[0][1], ucov[1], fma(
		gcon[0][2], ucov[2]
		, gcon[0][3] * ucov[3])));
	ucon[1] = fma(gcon[1][0], ucov[0], fma(
		gcon[1][1], ucov[1], fma(
		gcon[1][2], ucov[2]
		, gcon[1][3] * ucov[3])));
	ucon[2] = fma(gcon[2][0], ucov[0], fma(
		gcon[2][1], ucov[1], fma(
		gcon[2][2], ucov[2]
		, gcon[2][3] * ucov[3])));
	ucon[3] = fma(gcon[3][0], ucov[0], fma(
		gcon[3][1], ucov[1], fma(
		gcon[3][2], ucov[2]
		, gcon[3][3] * ucov[3])));
	#else
	ucon[0] = gcon[0][0] * ucov[0]
		+ gcon[0][1] * ucov[1]
		+ gcon[0][2] * ucov[2]
		+ gcon[0][3] * ucov[3];
	ucon[1] = gcon[1][0] * ucov[0]
		+ gcon[1][1] * ucov[1]
		+ gcon[1][2] * ucov[2]
		+ gcon[1][3] * ucov[3];
	ucon[2] = gcon[2][0] * ucov[0]
		+ gcon[2][1] * ucov[1]
		+ gcon[2][2] * ucov[2]
		+ gcon[2][3] * ucov[3];
	ucon[3] = gcon[3][0] * ucov[0]
		+ gcon[3][1] * ucov[1]
		+ gcon[3][2] * ucov[2]
		+ gcon[3][3] * ucov[3];
#endif
	return;
}

__device__ lower_g(FTYPE ucon[NDIM], FTYPE gcov[NDIM][NDIM], FTYPE ucov[NDIM])
{
	#if AMD
	ucov[0] = fma(gcov[0][0], ucon[0], fma(
		gcov[0][1], ucon[1], fma(
		gcov[0][2], ucon[2],
		gcov[0][3] * ucon[3])));
	ucov[1] = fma(gcov[1][0], ucon[0], fma(
		gcov[1][1], ucon[1], fma(
		gcov[1][2], ucon[2],
		gcov[1][3] * ucon[3])));
	ucov[2] = fma(gcov[2][0], ucon[0], fma(
		gcov[2][1], ucon[1], fma(
		gcov[2][2], ucon[2],
		gcov[2][3] * ucon[3])));
	ucov[3] = fma(gcov[3][0], ucon[0], fma(
		gcov[3][1], ucon[1], fma(
		gcov[3][2], ucon[2],
		gcov[3][3] * ucon[3])));
	return;
	#else
	ucov[0] = gcov[0][0] * ucon[0]
		+ gcov[0][1] * ucon[1]
		+ gcov[0][2] * ucon[2]
		+ gcov[0][3] * ucon[3];
	ucov[1] = gcov[1][0] * ucon[0]
		+ gcov[1][1] * ucon[1]
		+ gcov[1][2] * ucon[2]
		+ gcov[1][3] * ucon[3];
	ucov[2] = gcov[2][0] * ucon[0]
		+ gcov[2][1] * ucon[1]
		+ gcov[2][2] * ucon[2]
		+ gcov[2][3] * ucon[3];
	ucov[3] = gcov[3][0] * ucon[0]
		+ gcov[3][1] * ucon[1]
		+ gcov[3][2] * ucon[2]
		+ gcov[3][3] * ucon[3];
	#endif
	return;
}

__device__ ncov_calc(FTYPE gcon[NDIM][NDIM], FTYPE ncov[NDIM])
{
	FTYPE lapse;
	int i;

	lapse = sqrt(-1. / gcon[0][0]);
	ncov[0] = -lapse;
#pragma unroll NPR	
	for (i = 1; i < NDIM; i++) {
		ncov[i] = 0.;
	}
	return;
}

__device__ bcon_calc_g(FTYPE prim[NPR], FTYPE ucon[NDIM], FTYPE ucov[NDIM], FTYPE ncov[NDIM], FTYPE bcon[NDIM])
{
	FTYPE Bcon[NDIM];
	FTYPE u_dot_B;
	FTYPE gamma;
	int i;

	// Bcon = \mathcal{B}^\mu  of the paper:
	Bcon[0] = 0.;
	#pragma unroll NPR	
	for (i = 1; i<NDIM; i++) Bcon[i] = -ncov[0] * prim[BCON1 + i - 1];

	u_dot_B = 0.;
	#pragma unroll NPR	
	for (i = 0; i<NDIM; i++) u_dot_B += ucov[i] * Bcon[i];

	gamma = -ucon[0] * ncov[0];
	#pragma unroll NPR	
	for (i = 0; i<NDIM; i++) bcon[i] = (Bcon[i] + ucon[i] * u_dot_B) / gamma;
}

__device__ int gamma_calc_g(FTYPE *pr, FTYPE gcov[NDIM][NDIM], FTYPE *gamma)
{
	FTYPE utsq;

	utsq = gcov[1][1] * pr[UTCON1] * pr[UTCON1]
		+ gcov[2][2] * pr[UTCON2] * pr[UTCON2]
		+ gcov[3][3] * pr[UTCON3] * pr[UTCON3]
		+ 2.*(gcov[1][2] * pr[UTCON1] * pr[UTCON2]
		+ gcov[1][3] * pr[UTCON1] * pr[UTCON3]
		+ gcov[2][3] * pr[UTCON2] * pr[UTCON3]);

	if (utsq<0.0){
		if (fabs(utsq)>1.E-10){ // then assume not just machine precision
			return (1);
		}
		else utsq = fabs(utsq); // set floor
	}

	*gamma = sqrt(1. + utsq);
	return(0);
}


__device__ FTYPE pressure_rho0_u(FTYPE rho0, FTYPE u)
{
	return((GAMMA - 1.)*u);
}

__device__ FTYPE pressure_rho0_w(FTYPE rho0, FTYPE w)
{
	return((GAMMA - 1.)*(w - rho0) / GAMMA);
}

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(FTYPE x[], FTYPE dx[], FTYPE resid[],
	FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE D, FTYPE K_atm, FTYPE W_for_gnr2)
{
	FTYPE A, B, C, rho, W, B0;

	A = D*D;
	B0 = A * GAMMA * K_atm;
	B = B0 / (GAMMA - 1.);
	rho = x[0];
	W = W_for_gnr2;
	C = pow(rho, GAMMA - 1.);
	resid[0] = rho*W - A - B*C;
	jac[0][0] = W - B0 * C / rho;
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
	return;
}

__device__ int Utoprim_1dvsq2fix1(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K)
{
	FTYPE U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	FTYPE alpha;

	if (U[0] <= 0.) {
		return(-100);
	}

	/* First update the primitive B-fields */
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	/* Transform the CONSERVED variables into the new system */
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma unroll 3
	for (i = UTCON1; i <= UTCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	#pragma unroll BCON1
	for (i = 0; i < BCON1; i++) {
		prim_tmp[i] = prim[i];
	}
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) {
		prim_tmp[i] = alpha*prim[i];
	}

	ret = Utoprim_new_body2(U_tmp, gcov, gcon, gdet, prim_tmp, K);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
	#pragma unroll BCON1
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}
	return(ret);
}

__device__ int Utoprim_new_body2(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM],
	FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR], FTYPE K_atm)
{
	FTYPE x_1d[1];
	FTYPE QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	FTYPE rho0, u, p, gammasq, gamma, gtmp, W, utsq, vsq;
	int    i, j, retval;
	FTYPE Bsq, QdotBsq, Qtsq, Qdotn, D;

	// Assume ok initially:
	retval = 0;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower_g(Bcon, gcov, Bcov);
	#pragma unroll 4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise_g(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);

	Qdotn = Qcon[0] * ncov[0];

	Qsq = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	/* calculate W from last timestep and use  for guess */
	utsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++)
	#pragma unroll 3
		for (j = 1; j<4; j++) utsq += gcov[i][j] * prim[UTCON1 + i - 1] * prim[UTCON1 + j - 1];

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

	// Initialize independent variables for Newton-Raphson:
	x_1d[0] = 1. - 1. / gammasq;

	// Find vsq via Newton-Raphson:
	retval = general_newton_raphson2(x_1d, 1, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm);

	/* Problem with solver, so return denoting error before doing anything further */
	if (retval != 0) {
		retval = retval * 100 + 1;
		return(retval);
	}

	// Calculate v^2 :
	vsq = x_1d[0];
	if ((vsq >= 1.) || (vsq < 0.)) {
		retval = 4;
		return(retval);
	}

	// Find W from this vsq:
	W = W_of_vsq2(vsq, &p, &rho0, &u, D, K_atm);

	// Recover the primitive variables from the scalars and conserved variables:
	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;

	// User may want to handle this case differently, e.g. do NOT return upon 
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.) || (u <= 0.)) {
		retval = 5;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#pragma unroll 3
	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	#pragma unroll 3
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);

	/* set field components */
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	/* done! */
	return(retval);
}

__device__ int general_newton_raphson2(FTYPE x[], int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm)
{
	FTYPE f, df, dx[NEWT_DIM2], x_old[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	FTYPE errx;
	int    n_iter, id, i_extra, doing_extra;
	FTYPE W, W_old, rho, p, u;

	int   keep_iterating;

	// Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;

	for (id = 0; id < n; id++)  x_old[id] = x[id];

	W = W_old = 0.;

	n_iter = 0;

	/* Start the Newton-Raphson iterations : */
	keep_iterating = 1;
	while (keep_iterating) {

		func_1d_gnr2(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm);/* returns with new dx, f, df */

		errx = 0.;
		for (id = 0; id < n; id++) {
			x_old[id] = x[id];
		}

		for (id = 0; id < n; id++) {
			x[id] += dx[id];
		}

		validate_x2(x, x_old);

		W_old = W;
		W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm);
		errx = (W == 0.) ? fabs(W - W_old) : fabs((W - W_old) / W);
		errx += (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= NEWT_TOL) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) {
			doing_extra = 1;
		}

		if (doing_extra == 1) i_extra++;

		// See if we've done the extra iterations, or have done too many iterations:
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

	// Return in different ways depending on whether a solution was found:
	if (fabs(errx) > MIN_NEWT_TOL){

		return(1);
	}
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > NEWT_TOL)){
		//fprintf(stderr," totalcount = %d   1   %d  %26.20e \n",n_iter,i_extra,errx); fflush(stderr);
		return(0);
	}
	if (fabs(errx) <= NEWT_TOL){
		//fprintf(stderr," totalcount = %d   2   %d  %26.20e \n",n_iter,i_extra,errx); fflush(stderr); 
		return(0);
	}
	return(0);
}

__device__ void validate_x2(FTYPE x[1], FTYPE x0[1])
{
	FTYPE small = 1.e-10;
	x[0] = (x[0] >= 1.0) ? (0.5*(x0[0] + 1.)) : x[0];
	x[0] = (x[0] <  -small) ? (0.5*x0[0]) : x[0];
	x[0] = fabs(x[0]);
	return;
}

__device__ void func_1d_gnr2(FTYPE x[], FTYPE dx[], FTYPE resid[], FTYPE jac[][NEWT_DIM2], FTYPE *f, FTYPE *df, int n, FTYPE Bsq, FTYPE Qtsq, FTYPE QdotBsq, FTYPE Qdotn, FTYPE D, FTYPE K_atm)
{
	FTYPE vsq, W, Wsq, W3, dWdvsq, fact_tmp, rho, p, u;
	vsq = x[0];

	// Calculate best value for W given current guess for vsq: 
	W = W_of_vsq2(vsq, &p, &rho, &u, D, K_atm);
	Wsq = W*W;
	W3 = W*Wsq;

	// Doing this assuming  P = (G-1) u :

	dWdvsq = dWdvsq_calc2(vsq, rho, p);

	fact_tmp = (Bsq + W);

	resid[0] = Qtsq - vsq * fact_tmp * fact_tmp + QdotBsq * (Bsq + 2.*W) / Wsq;
	jac[0][0] = -fact_tmp * (fact_tmp + 2. * dWdvsq * (vsq + QdotBsq / W3));

	dx[0] = -resid[0] / jac[0][0];

	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
}

__device__ FTYPE pressure_of_rho2(FTYPE rho0, FTYPE K_atm)
{
	return(K_atm * pow(rho0, G_ATM));
}

__device__ FTYPE u_of_p2(FTYPE p)
{
	return(p / (GAMMA - 1.));
}

__device__ FTYPE W_of_vsq2(FTYPE vsq, FTYPE *p, FTYPE *rho, FTYPE *u, FTYPE D, FTYPE K_atm)
{
	FTYPE gtmp;
	gtmp = (1. - vsq);
	*rho = D * sqrt(gtmp);
	*p = pressure_of_rho2(*rho, K_atm);
	*u = u_of_p2(*p);
	return((*rho + *u + *p) / gtmp);
}

__device__ FTYPE dWdvsq_calc2(FTYPE vsq, FTYPE rho, FTYPE p)
{
	return((GAMMA*(2. - G_ATM)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
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
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0][0]);

	/* Transform the CONSERVED variables into the new system */
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma unroll 3
	for (i = UTCON1; i <= UTCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	#pragma unroll BCON1
	for (i = 0; i < BCON1; i++) {
		prim_tmp[i] = prim[i];
	}
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) {
		prim_tmp[i] = alpha*prim[i];
	}

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
	#pragma unroll BCON1
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	#if(DOKTOT )
	prim[KTOT] = U[KTOT] / U[RHO];
	#endif

	return(ret);
}

__device__ int Utoprim_new_body(FTYPE U[NPR], FTYPE gcov[NDIM][NDIM], FTYPE gcon[NDIM][NDIM], FTYPE gdet, FTYPE prim[NPR])
{
	FTYPE x_2d[NEWT_DIM];
	FTYPE QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	FTYPE rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, j, n, retval, i_increase;
	FTYPE Bsq, QdotBsq, Qtsq, Qdotn, D;

	n = NEWT_DIM;

	// Assume ok initially:
	retval = 0;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower_g(Bcon, gcov, Bcov);
	#pragma unroll 4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise_g(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov_calc(gcon, ncov);
	raise_g(ncov, gcon, ncon);

	Qdotn = Qcon[0] * ncov[0];

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	#if AMD
	Qtsq = fma(Qdotn, Qdotn, Qsq);
	#else
	Qtsq = Qsq + Qdotn*Qdotn;
	#endif
	D = U[RHO];

	/* calculate W from last timestep and use for guess */
	utsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++)
	#pragma unroll 4
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
	#pragma unroll 4
	for (i = 1; i<4; i++)  Qtcon[i] = fma(ncon[i], Qdotn, Qcon[i]);
	#pragma unroll 4
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (fma(QdotB, Bcon[i] / W, Qtcon[i]));
	#else
	#pragma unroll 4
	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	#pragma unroll 4
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);
	#endif
	/* set field components */
	#pragma unroll 3
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
	return((fma(Wsq, Qtsq, QdotBsq * (Bsq + 2.*W))) / (Wsq*Xsq));
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

	p_tmp = (GAMMA - 1.) * (fma(W, gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - vsq) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - vsq), -W)) / GAMMA;

	// These expressions were calculated using Mathematica, but fmae into efficient 
	// code using Maple.  Since we know the analytic form of the equations, we can 
	// explicitly calculate the Newton-Raphson step: 

	#if AMD
	t2 = fma(-0.5, Bsq, dPdvsq);
	t3 = Bsq + W;
	t4 = t3*t3;
	t9 = 1 / Wsq;
	t11 = fma(QdotBsq, (Bsq + 2.0*W)*t9, fma(-vsq, t4, Qtsq));
	t16 = QdotBsq*t9;
	t18 = -fma(0.5, Bsq*(1.0 + vsq), Qdotn) + fma(0.5, t16, -W + p_tmp);
	t21 = 1 / t3;
	t23 = 1 / W;
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = fma(t25, t3, (fma(-2.0, dPdvsq, Bsq))*(fma(vsq, Wsq*W, QdotBsq))*t9*t23);
	t36 = 1 / t35;
	dx[0] = -(fma(t2, t11, t4*t18))*t21*t36;
	t40 = (vsq + t24)*t3;
	dx[1] = -(-fma(t25, t11, 2.0*t40*t18))*t21*t36;
	jac[0][0] = -2.0*t40;
	jac[0][1] = -t4;
	jac[1][0] = t25;
	jac[1][1] = t2;
	resid[0] = t11;
	resid[1] = t18;
	*df = fma(-resid[0], resid[0], -resid[1] * resid[1]);
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
	FTYPE2 gcon[NDIM][NDIM];
	FTYPE2 gcov[NDIM][NDIM];
	FTYPE2 g;
};

struct of_state {
	FTYPE2 ucon[NDIM];
	FTYPE2 ucov[NDIM];
	FTYPE2 bcon[NDIM];
	FTYPE2 bcov[NDIM];
};

/*Declare other functions*/
__device__ get_state(FTYPE2 *  pr, struct of_geom *  geom, struct of_state *  q);
__device__ ucon_calc(FTYPE2 *  pr, struct of_geom *  geom, FTYPE2 *  ucon);
__device__ bcon_calc(FTYPE2 *  pr, FTYPE2 *  ucon, FTYPE2 *  ucov, FTYPE2 *  bcon);
__device__ int gamma_calc(FTYPE2 *  pr, struct of_geom *  geom, FTYPE2 *  gamma);
__device__ get_geometry(int N1, int N2, int ii, int jj, int zz, int kk, struct of_geom *  geom, const  FTYPE2* __restrict__ gcov_GPU, const  FTYPE2* __restrict__ gcon_GPU, const  FTYPE2* __restrict__ gdet_GPU);
FTYPE2 slope_lim(FTYPE2 y1, FTYPE2 y2, FTYPE2 y3, int lim);
__device__ raise(FTYPE2 *  ucov, struct of_geom *  geom, FTYPE2 *  ucon);
__device__ lower(FTYPE2 *  ucon, struct of_geom *  geom, FTYPE2 *  ucov);
__device__ primtoflux(FTYPE2 *  pr, struct of_state *  q, int dir, struct of_geom *  geom, FTYPE2 *  flux, FTYPE2 gam);
__device__ primtoU(FTYPE2 *  pr, struct of_state *  q, struct of_geom *  geom, FTYPE2 *U, FTYPE2 gam);
__device__ vchar(FTYPE2 *  pr, struct of_state *  q, struct of_geom *  geom, int js, FTYPE2 *  vmax, FTYPE2 *  vmin, FTYPE2 gam);
__device__ mhd_calc(FTYPE2 *  pr, int dir, struct of_state *  q, FTYPE2 *  mhd, FTYPE2 gam);
__device__ source(int N1, int N2, FTYPE2 *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, FTYPE2 *dU, FTYPE2 Dt, FTYPE2 gam, const  FTYPE2* __restrict__ Imageconn,
struct of_state *  q, double a, double r);
__device__ misc_source(FTYPE2 *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, FTYPE2 *  dU,
	double a, double gam, double r, double Dt);
__device__ inflow_check(int N1, int N2, FTYPE2 *  prim, int ii, int jj, int zz, int type, const  FTYPE2* __restrict__ gcov1, const  FTYPE2* __restrict__ gcon2, const  FTYPE2* __restrict__ gdet3);
__device__ FTYPE2 bsq_calc(FTYPE2 *  pr, struct of_geom *  geom);
__device__ double NewtonRaphson(double start, size_t max_count, int dir, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq);
__device__ double readImageDouble(int4 a);
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);

/* find relative 4-velocity from 4-velocity (both in code coords) */
__device__ ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon)
{
	double alpha, beta[NDIM], gamma;
	int j;

	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	alpha = 1. / sqrt(-geom->gcon[TT][TT]);
	SLOOPA beta[j] = geom->gcon[TT][j] * alpha*alpha;
	gamma = alpha*ucon[TT];


	utcon[0] = 0;
	SLOOPA utcon[j] = ucon[j] + gamma*beta[j] / alpha;
}

__device__ ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
{
	double AA, BB, CC, DD, one_over_alpha_sq;
	//compute the Lorentz factor based on contravariant 3-velocity
	AA = geom->gcov[TT][TT];
	BB = 2.*(geom->gcov[TT][1] * vcon[1] +
		geom->gcov[TT][2] * vcon[2] +
		geom->gcov[TT][3] * vcon[3]);
	CC = geom->gcov[1][1] * vcon[1] * vcon[1] +
		geom->gcov[2][2] * vcon[2] * vcon[2] +
		geom->gcov[3][3] * vcon[3] * vcon[3] +
		2.*(geom->gcov[1][2] * vcon[1] * vcon[2] +
		geom->gcov[1][3] * vcon[1] * vcon[3] +
		geom->gcov[2][3] * vcon[2] * vcon[3]);

	DD = -1. / (AA + BB + CC);

	one_over_alpha_sq = -geom->gcon[TT][TT];

	if (DD<one_over_alpha_sq) {
		DD = one_over_alpha_sq;
	}

	*ut = sqrt(DD);

}

__device__ primtoU(FTYPE2 *pr, struct of_state *q, struct of_geom *geom, FTYPE2 *U, FTYPE2 gam)
{
	primtoflux(pr, q, 0, geom, U, gam);
	return;
}

/* add in source terms to equations of motion */
__device__ source(int N1, int N2, FTYPE2 *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, FTYPE2 *  dU, FTYPE2 Dt, FTYPE2 gam,
	const  FTYPE2* __restrict__ Imageconn, struct of_state *  q, double a, double r)
{
	FTYPE2 mhd[NDIM][NDIM];
	int fix_mem2 = LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int k;
	//struct of_state q ;
	double conn;
	//get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, q, mhd[0], gam);
	mhd_calc(ph, 1, q, mhd[1], gam);
	mhd_calc(ph, 2, q, mhd[2], gam);
	mhd_calc(ph, 3, q, mhd[3], gam);

	/* contract mhd stress tensor with connection */
	#pragma unroll NPR	
	PLOOP dU[k] = 0.;
	
	#pragma unroll NDIM	
	for (k = 0; k<NDIM; k++){
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

	//Add cooling term if needed
	#if (COOL_DISK)
	misc_source(ph, icurr, jcurr, geom, q, dU, a, gam, r, Dt);
	#endif

	dU[UU] *= geom->g;
	dU[U1] *= geom->g;
	dU[U2] *= geom->g;
	dU[U3] *= geom->g;
	/* done! */
}

__device__ misc_source(FTYPE2 *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, FTYPE2 *  dU,
	double a, double gam, double r, double Dt){
	double epsilon = ph[UU] / ph[RHO];
	double om_kepler = 1. / (pow(r, 3. / 2.) + a);
	double T_target = M_PI / 2.*pow(H_OVER_R*r*om_kepler, 2.);
	double Y = (gam - 1.)*epsilon / T_target;
	double lambda = om_kepler*ph[UU] * sqrt(Y - 1. + fabs(Y - 1.));
	double int_energy = q->ucov[0] * q->ucon[0] * ph[UU];
	double bsq = bsq_calc(ph, geom);
	if (bsq / ph[RHO]<1. || r<10.){
		if (fabs(q->ucov[0] * lambda)*Dt<0.1*fabs(int_energy)){
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
		}
		else{
			lambda *= (0.1*fabs(int_energy)) / (fabs(q->ucov[0] * lambda)*Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
		}
	}
}

__device__ primtoflux(FTYPE2 *  pr, struct of_state *  q, int dir, struct of_geom *  geom, FTYPE2 *  flux, FTYPE2 gam)
{
	int k;
	FTYPE2 mhd[NDIM];

	/* particle number flux */
	flux[RHO] = pr[RHO] * q->ucon[dir];

	mhd_calc(pr, dir, q, mhd, gam);

	/* MHD stress-energy tensor w/ first index up,
	* second index down. */
	flux[UU] = mhd[0] + flux[RHO];
	flux[U1] = mhd[1];
	flux[U2] = mhd[2];
	flux[U3] = mhd[3];

	/* dual of Maxwell tensor */
	#if AMD
	flux[B1] = fma(q->bcon[1], q->ucon[dir], -q->bcon[dir] * q->ucon[1]);
	flux[B2] = fma(q->bcon[2], q->ucon[dir], -q->bcon[dir] * q->ucon[2]);
	flux[B3] = fma(q->bcon[3], q->ucon[dir], -q->bcon[dir] * q->ucon[3]);
	#else
	flux[B1] = q->bcon[1] * q->ucon[dir] - q->bcon[dir] * q->ucon[1];
	flux[B2] = q->bcon[2] * q->ucon[dir] - q->bcon[dir] * q->ucon[2];
	flux[B3] = q->bcon[3] * q->ucon[dir] - q->bcon[dir] * q->ucon[3];
	#endif

	#if(DOKTOT )
	flux[KTOT] = flux[RHO] * pr[KTOT];
	#endif

	#pragma unroll NPR
	PLOOP flux[k] *= geom->g;
}

__device__ vchar(FTYPE2 *  pr, struct of_state *  q, struct of_geom *  geom, int js, FTYPE2 *  vmax, FTYPE2 *  vmin, FTYPE2 gam)
{
	FTYPE2 discr, vp, vm, bsq, EE, EF, va2, cs2, cms2, rho, u;
	FTYPE2 Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	FTYPE2 Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	#pragma unroll NDIM
	DLOOPA Acov[j] = 0.;
	Acov[js] = 1.;
	raise(Acov, geom, Acon);

	#pragma unroll NDIM
	DLOOPA Bcov[j] = 0.;
	Bcov[TT] = 1.;
	raise(Bcov, geom, Bcon);

	/* find fast magnetosonic speed */
	bsq = dot(q->bcon, q->bcov);
	rho = pr[RHO];
	u = pr[UU];
	#if AMD
	EF = fma(gam, u, rho);
	#else
	EF = rho + gam*u;
	#endif
	EE = bsq + EF;
	cs2 = gam*(gam - 1.)*u / EF;
	va2 = bsq / EE;


	//	if(cs2 < 0.) cs2 = SMALL ;
	//	if(cs2 > 1.) cs2 = 1. ;
	//	if(va2 < 0.) va2 = SMALL ;
	//	if(va2 > 1.) va2 = 1. ;

	cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

	//cms2 *= 1.1 ;

	/* check on it! */
	if (cms2 < 0.) {
		//fail(FAIL_COEFF_NEG) ;
		cms2 = SMALL;
	}
	if (cms2 > 1.) {
		//fail(FAIL_COEFF_SUP) ;
		cms2 = 1.;
	}

	/* now require that speed of wave measured by observer
	q->ucon is cms2 */
	Asq = dot(Acon, Acov);
	Bsq = dot(Bcon, Bcov);
	Au = q->ucon[js];
	Bu = q->ucon[TT];
	AB = dot(Acon, Bcov);
	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;
	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2.* fma(-(AB + AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4.*A*C);
	#else
	A = Bu2 - (Bsq + Bu2)*cms2;
	B = 2.*(AuBu - (AB + AuBu)*cms2);
	C = Au2 - (Asq + Au2)*cms2;
	discr = B*B - 4.*A*C;
	#endif
	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
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
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	#if( FULL_DISP ) 
	double vp2, vm2;
	vp2 = NewtonRaphson(vp, 1, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vm2 = NewtonRaphson(vm, 1, js, q->ucon, q->ucov, q->bcon, geom, EE, va2, cs2);
	vp = vp2;
	vm = vm2;
	#endif

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
	}
	else {
		*vmax = vm;
		*vmin = vp;
	}

	return;
}

__device__ double NewtonRaphson(double start, size_t max_count, int dir, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq)
{
	size_t count = 0;
	double dx = start / 100.0;
	double x = start;
	double diff, derivative;
	do{
		diff = Drel(dir, x, ucon, ucov, bcon, geom, E, vasq, csq);
		derivative = (Drel(dir, x + dx, ucon, ucov, bcon, geom, E, vasq, csq) - diff) / (dx + SMALL);
		count++;
		x = x - diff / (derivative + SMALL);
	} while (Drel(dir, x*0.99999, ucon, ucov, bcon, geom, E, vasq, csq)*Drel(dir, x*1.00001, ucon, ucov, bcon, geom, E, vasq, csq)>0.0 && (count < max_count));
	if (count >= max_count){
		x = start;
	}
	return x;
}

__device__ double Drel(int dir, double v, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq){
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
	else if (dir == 3){
		kcov[3] = 1.0;
	}
	raise(kcov, geom, kcon);
	om = dot(ucon, kcov);
	omsq = pow(om, 2.0);
	#pragma unroll NDIM
	for (i = 0; i < NDIM; i++){
		Kcov[i] = kcov[i] + ucov[i] * om;
		Kcon[i] = kcon[i] + ucon[i] * om;
	}
	ksq = dot(Kcov, Kcon);
	kvasq = pow(dot(kcov, bcon), 2.0) / (E + SMALL);
	cfsq = vasq + csq*(1.0 - vasq);
	result = 0.5*(cfsq*ksq + csq*kvasq + sqrt(pow(cfsq*ksq + csq*kvasq, 2.0) - 4.0*ksq*csq*kvasq)) - omsq;
	return result;
}

/* MHD stress tensor, with first index up, second index down */
__device__ mhd_calc(FTYPE2 *pr, int dir, struct of_state *q, FTYPE2 *mhd, FTYPE2 gam)
{
	int j;
	FTYPE2 r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];
	P = (gam - 1.)*u;
	w = P + r + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	#if AMD
	ptot = fma(0.5, bsq, P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* single row of mhd stress tensor,
	* first index up, second index down */
	#if AMD
	#pragma unroll NDIM
	DLOOPA mhd[j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
	#else
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
	#endif
}

__device__ get_state(FTYPE2 *  pr, struct of_geom *  geom, struct of_state *  q)
{
	/* get ucon */
	ucon_calc(pr, geom, q->ucon);
	lower(q->ucon, geom, q->ucov);
	bcon_calc(pr, q->ucon, q->ucov, q->bcon);
	lower(q->bcon, geom, q->bcov);

	return;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ raise(FTYPE2 *  ucov, struct of_geom *  geom, FTYPE2 *  ucon)
{
	#if AMD
	ucon[0] = fma(geom->gcon[0][0], ucov[0], fma(
		geom->gcon[0][1], ucov[1], fma(
		geom->gcon[0][2], ucov[2]
		, geom->gcon[0][3] * ucov[3])));
	ucon[1] = fma(geom->gcon[1][0], ucov[0], fma(
		geom->gcon[1][1], ucov[1], fma(
		geom->gcon[1][2], ucov[2]
		, geom->gcon[1][3] * ucov[3])));
	ucon[2] = fma(geom->gcon[2][0], ucov[0], fma(
		geom->gcon[2][1], ucov[1], fma(
		geom->gcon[2][2], ucov[2]
		, geom->gcon[2][3] * ucov[3])));
	ucon[3] = fma(geom->gcon[3][0], ucov[0], fma(
		geom->gcon[3][1], ucov[1], fma(
		geom->gcon[3][2], ucov[2]
		, geom->gcon[3][3] * ucov[3])));
	#else
	ucon[0] = geom->gcon[0][0] * ucov[0]
		+ geom->gcon[0][1] * ucov[1]
		+ geom->gcon[0][2] * ucov[2]
		+ geom->gcon[0][3] * ucov[3];
	ucon[1] = geom->gcon[1][0] * ucov[0]
		+ geom->gcon[1][1] * ucov[1]
		+ geom->gcon[1][2] * ucov[2]
		+ geom->gcon[1][3] * ucov[3];
	ucon[2] = geom->gcon[2][0] * ucov[0]
		+ geom->gcon[2][1] * ucov[1]
		+ geom->gcon[2][2] * ucov[2]
		+ geom->gcon[2][3] * ucov[3];
	ucon[3] = geom->gcon[3][0] * ucov[0]
		+ geom->gcon[3][1] * ucov[1]
		+ geom->gcon[3][2] * ucov[2]
		+ geom->gcon[3][3] * ucov[3];
	#endif
	return;
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ lower(FTYPE2 *  ucon, struct of_geom *  geom, FTYPE2 *  ucov)
{
	#if AMD
	ucov[0] = fma(geom->gcov[0][0], ucon[0], fma(
		geom->gcov[0][1], ucon[1], fma(
		geom->gcov[0][2], ucon[2],
		geom->gcov[0][3] * ucon[3])));
	ucov[1] = fma(geom->gcov[1][0], ucon[0], fma(
		geom->gcov[1][1], ucon[1], fma(
		geom->gcov[1][2], ucon[2],
		geom->gcov[1][3] * ucon[3])));
	ucov[2] = fma(geom->gcov[2][0], ucon[0], fma(
		geom->gcov[2][1], ucon[1], fma(
		geom->gcov[2][2], ucon[2],
		geom->gcov[2][3] * ucon[3])));
	ucov[3] = fma(geom->gcov[3][0], ucon[0], fma(
		geom->gcov[3][1], ucon[1], fma(
		geom->gcov[3][2], ucon[2],
		geom->gcov[3][3] * ucon[3])));
	return;
	#else
	ucov[0] = geom->gcov[0][0] * ucon[0]
		+ geom->gcov[0][1] * ucon[1]
		+ geom->gcov[0][2] * ucon[2]
		+ geom->gcov[0][3] * ucon[3];
	ucov[1] = geom->gcov[1][0] * ucon[0]
		+ geom->gcov[1][1] * ucon[1]
		+ geom->gcov[1][2] * ucon[2]
		+ geom->gcov[1][3] * ucon[3];
	ucov[2] = geom->gcov[2][0] * ucon[0]
		+ geom->gcov[2][1] * ucon[1]
		+ geom->gcov[2][2] * ucon[2]
		+ geom->gcov[2][3] * ucon[3];
	ucov[3] = geom->gcov[3][0] * ucon[0]
		+ geom->gcov[3][1] * ucon[1]
		+ geom->gcov[3][2] * ucon[2]
		+ geom->gcov[3][3] * ucon[3];
	#endif
}

/* find contravariant four-velocity */
__device__ ucon_calc(FTYPE2 *  pr, struct of_geom *  geom, FTYPE2 *  ucon)
{
	FTYPE2 alpha, gamma;
	FTYPE2 beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[TT][TT]);
	#pragma unroll 4
	SLOOPA beta[j] = geom->gcon[TT][j] * alpha*alpha;

	if (gamma_calc(pr, geom, &gamma)) {
		// fflush(stderr);
		//fprintf(stderr,"\nucon_calc(): gamma failure \n");
		// fflush(stderr);
		// fail(FAIL_GAMMA);
	}

	ucon[TT] = gamma / alpha;
	#if AMD
	#pragma unroll 4
	SLOOPA ucon[j] = fma(-gamma, beta[j] / alpha, pr[U1 + j - 1]);
	#else
	#pragma unroll 4
	SLOOPA ucon[j] = pr[U1 + j - 1] - gamma*beta[j] / alpha;
	#endif

	return;
}

__device__ bcon_calc(FTYPE2 *  pr, FTYPE2 *  ucon, FTYPE2 *  ucov, FTYPE2 *  bcon)
{
	int j;

	#if AMD
	bcon[TT] = fma(pr[B1], ucov[1], fma(pr[B2], ucov[2], pr[B3] * ucov[3]));
	#pragma unroll 3
	for (j = 1; j<4; j++)
		bcon[j] = (fma(bcon[TT], ucon[j], pr[B1 - 1 + j])) / ucon[TT];
	#else
	bcon[TT] = pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3];
	#pragma unroll 3
	for (j = 1; j<4; j++)
		bcon[j] = (pr[B1 - 1 + j] + bcon[TT] * ucon[j]) / ucon[TT];
	#endif
	return;
}

__device__ int gamma_calc(FTYPE2 *  pr, struct of_geom *  geom, FTYPE2 *  gamma)
{
	FTYPE2 qsq;
	#if AMD
	qsq = fma(geom->gcov[1][1], pr[U1] * pr[U1], fma(
		geom->gcov[2][2], pr[U2] * pr[U2],
		geom->gcov[3][3] * pr[U3] * pr[U3]))
		+ 2.*fma(geom->gcov[1][2], pr[U1] * pr[U2], fma(
		geom->gcov[1][3], pr[U1] * pr[U3],
		geom->gcov[2][3] * pr[U2] * pr[U3]));
	#else
	qsq = geom->gcov[1][1] * pr[U1] * pr[U1]
		+ geom->gcov[2][2] * pr[U2] * pr[U2]
		+ geom->gcov[3][3] * pr[U3] * pr[U3]
		+ 2.*(geom->gcov[1][2] * pr[U1] * pr[U2]
		+ geom->gcov[1][3] * pr[U1] * pr[U3]
		+ geom->gcov[2][3] * pr[U2] * pr[U3]);
	#endif

	if (qsq < 0.){
		if (fabs(qsq) > 1.E-10){ // then assume not just machine precision
			//fprintf(stderr,"gamma_calc():  failed: i,j,qsq = %d %d %28.18e \n", icurr,jcurr,qsq);
			// fprintf(stderr,"v[1-3] = %28.18e %28.18e %28.18e  \n",pr[U1],pr[U2],pr[U3]);
			*gamma = 1.;
			return (1);
		}
		else qsq = 1.E-10; // set floor
	}

	*gamma = sqrt(1. + qsq);

	return(0);
}

/* load local geometry into structure geom */
__device__ get_geometry(int N1, int N2, int ii, int jj, int zz, int kk, struct of_geom *  geom, const  FTYPE2* __restrict__ gcov_GPU, const  FTYPE2* __restrict__ gcon_GPU, const  FTYPE2* __restrict__ gdet_GPU)
{
	int fix_mem2 = LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	geom->gcon[0][0] = gcon_GPU[kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[0][0] = gcov_GPU[kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[0][1] = gcon_GPU[1 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[0][1] = gcov_GPU[1 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[0][2] = gcon_GPU[2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[0][2] = gcov_GPU[2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[0][3] = gcon_GPU[3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[0][3] = gcov_GPU[3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[1][0] = geom->gcon[0][1];
	geom->gcov[1][0] = geom->gcov[0][1];
	geom->gcon[1][1] = gcon_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 1 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[1][1] = gcov_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 1 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[1][2] = gcon_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[1][2] = gcov_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[1][3] = gcon_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[1][3] = gcov_GPU[1 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[2][0] = geom->gcon[0][2];
	geom->gcov[2][0] = geom->gcov[0][2];
	geom->gcon[2][1] = geom->gcon[1][2];
	geom->gcov[2][1] = geom->gcov[1][2];
	geom->gcon[2][2] = gcon_GPU[2 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[2][2] = gcov_GPU[2 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 2 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcon[2][3] = gcon_GPU[2 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
	geom->gcov[2][3] = gcov_GPU[2 * NDIM*NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + 3 * NPG*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + kk*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];
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

__device__ inflow_check(int N1, int N2, FTYPE2 *  pr, int ii, int jj, int zz, int type, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet)
{
	struct of_geom geom;
	FTYPE2 ucon[NDIM];
	int j, k;
	FTYPE2 alpha, beta1, gamma, vsq;
	get_geometry(N1, N2, ii, jj, zz, CENT, &geom, gcov, gcon, gdet);
	ucon_calc(pr, &geom, ucon);

	if (((ucon[1] > 0.) && (type == 0)) || ((ucon[1] < 0.) && (type == 1))) {
		// find gamma and remove it from primitives 
		if (gamma_calc(pr, &geom, &gamma)) {
			// fflush(stderr);
			// fprintf(stderr,"\ninflow_check(): gamma failure \n");
			// fflush(stderr);
			// fail(FAIL_GAMMA);
		}
		pr[U1] /= gamma;
		pr[U2] /= gamma;
		pr[U3] /= gamma;
		alpha = 1. / sqrt(-geom.gcon[0][0]);
		beta1 = geom.gcon[0][1] * alpha*alpha;

		// reset radial velocity so radial 4-velocity is zero 
		pr[U1] = beta1 / alpha;

		// now find new gamma and put it back in 
		vsq = 0.;
		
		#pragma unroll NDIM
		SLOOP vsq += geom.gcov[j][k] * pr[U1 + j - 1] * pr[U1 + k - 1];
		
		if (fabs(vsq) < 1.e-13)  vsq = 1.e-13;
		if (vsq >= 1.) {
			vsq = 1. - 1. / (GAMMAMAX*GAMMAMAX);
		}
		gamma = 1. / sqrt(1. - vsq);
		pr[U1] *= gamma;
		pr[U2] *= gamma;
		pr[U3] *= gamma;
	}
}

__device__ FTYPE2 slope_lim(FTYPE2 y1, FTYPE2 y2, FTYPE2 y3, int dir)
{
	FTYPE2 Dqm, Dqp, Dqc, s;
	/* woodward, or monotonized central, slope limiter */
	Dqm = (1.5)*(y2 - y1);
	Dqp = (1.5)*(y3 - y2);
	Dqc = 0.5*(y3 - y1);
	s = Dqm*Dqp;
	if (s <= 0.) return 0.;
	else {
		if (fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
			return(Dqm);
		else if (fabs(Dqp) < fabs(Dqc))
			return(Dqp);
		else
			return(Dqc);
	}
}

__device__ para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout)
{
	int i;
	double y[5], dq[5];
	double Dqm, Dqc, Dqp, aDqm, aDqp, aDqc, s, l, r, qa, qd, qe;

	y[0] = x1;
	y[1] = x2;
	y[2] = x3;
	y[3] = x4;
	y[4] = x5;

	/*CW1.7 */
	for (i = 1; i<4; i++) {
		Dqm = 2. *(y[i] - y[i - 1]);
		Dqp = 2. *(y[i + 1] - y[i]);
		Dqc = 0.5 *(y[i + 1] - y[i - 1]);
		aDqm = fabs(Dqm);
		aDqp = fabs(Dqp);
		aDqc = fabs(Dqc);
		s = Dqm*Dqp;
		Dqm = MY_MIN(aDqm, aDqp);
		if (aDqc< Dqm){
			if (Dqc>0.) dq[i] = (aDqc)*(double)(s>0.);
			else dq[i] = (-aDqc)*(double)(s>0.);
		}
		else{
			if (Dqc>0.) dq[i] = (Dqm)*(double)(s>0.);
			else dq[i] = (-Dqm)*(double)(s>0.);
		}

	}

	// CW1.6
	l = 0.5*(y[2] + y[1]) - (dq[2] - dq[1]) / 6.0;
	r = 0.5*(y[3] + y[2]) - (dq[3] - dq[2]) / 6.0;

	qa = (r - y[2])*(y[2] - l);
	qd = (r - l);
	qe = 6.0*(y[2] - 0.5*(l + r));

	if (qa <= 0.) {
		l = y[2];
		r = y[2];
	}

	if (qd*(qd - qe)<0.0) l = 3.0*y[2] - 2.0*r;
	else if (qd*(qd + qe)<0.0) r = 3.0*y[2] - 2.0*l;

	lout[0] = l;   //a_L,j
	rout[0] = r;
}

/* returns b^2 (i.e., twice magnetic pressure) */
__device__ FTYPE2 bsq_calc(FTYPE2 *  pr, struct of_geom *  geom)
{
	struct of_state q;
	get_state(pr, geom, &q);
	return(dot(q.bcon, q.bcov));
}


__global__ void fluxcalcprep(int N1, int N2, int N3, __global FTYPE2 *   F, __global FTYPE2 *  dq, __global FTYPE2 *  p, int dir, int lim, FTYPE2 hslope, FTYPE2 fractheta, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, int number, __global FTYPE2 *  V)
{
	int global_id = get_global_id(0);
	int isize = (N3 + 2 * D3)*(N2 + 2 * D2);
	int zcurr = (global_id % (isize)) % (N3 + 2 * D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + 2 * D3);
	int icurr = (global_id - (jcurr*(N3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3;
	jcurr += N2G - 1;
	icurr += N1G - 1;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int jsize = N3 + 2 * N3G;
	int k = 0;
	if (global_id<(N1 + 2 * D1) * (N2 + 2 * D2) * (N3 + 2 * D3)) k = 1;
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	FTYPE2 x1, x2, x3, x4, x5, temp[1], result[1];

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }
	if (k == 1){
		#if(PPM)
		if (number == 1){
			#pragma unroll NPR	
			for (k = 0; k<NPR; k++){
				x1 = p[k*(ksize)+global_id - 3 * zdel - 3 * (N3 + 2 * N3G)*jdel - 3 * isize*idel];
				x2 = p[k*(ksize)+global_id - 2 * zdel - 2 * (N3 + 2 * N3G)*jdel - 2 * isize*idel];
				x3 = p[k*(ksize)+global_id - 1 * zdel - 1 * (N3 + 2 * N3G)*jdel - 1 * isize*idel];
				x4 = p[k*(ksize)+global_id];
				x5 = p[k*(ksize)+global_id + 1 * zdel + 1 * (N3 + 2 * N3G)*jdel + 1 * isize*idel];
				para(x1, x2, x3, x4, x5, temp, result);
				dq[k*(ksize)+global_id] = result[0];
			}
		}
		else{
			#pragma unroll NPR	
			for (k = 0; k<NPR; k++){
				x1 = p[k*(ksize)+global_id - 2 * zdel - 2 * (N3 + 2 * N3G)*jdel - 2 * isize*idel];
				x2 = p[k*(ksize)+global_id - 1 * zdel - 1 * (N3 + 2 * N3G)*jdel - 1 * isize*idel];
				x3 = p[k*(ksize)+global_id];
				x4 = p[k*(ksize)+global_id + 1 * zdel + 1 * (N3 + 2 * N3G)*jdel + 1 * isize*idel];
				x5 = p[k*(ksize)+global_id + 2 * zdel + 2 * (N3 + 2 * N3G)*jdel + 2 * isize*idel];
				para(x1, x2, x3, x4, x5, result, temp);
				dq[k*(ksize)+global_id] = result[0];
			}
		}
		#elif(LEER)
		if (number == 1){
			double d_XL = V[(dir - 1)*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)];
			double CFL = (V[(3 + (dir - 1))*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]) / (d_XL);
			double CBL = (V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id - 2 * (dir == 1)*isize - 2 * (dir == 2)*jsize - 2 * (dir == 3)]) /
				(V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - V[(dir - 1)*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]);
			for (k = 0; k<NPR; k++){
				double d_C = (p[k*ksize + global_id] - p[k*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]) / (V[(3 + (dir - 1))*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]);
				double d_L = (p[k*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - p[k*ksize + global_id - 2 * (dir == 1)*isize - 2 * (dir == 2)*jsize - 2 * (dir == 3)]) /
					(V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id - 2 * (dir == 1)*isize - 2 * (dir == 2)*jsize - 2 * (dir == 3)]);
				if (d_L*d_C <= 0.){
					dq[k*(ksize)+global_id] = p[k*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)];
				}
				else{
					dq[k*(ksize)+global_id] = p[k*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] + (d_XL*d_L*d_C*(CFL*d_L + CBL*d_C)) / (d_L*d_L + (CFL + CBL - 2.)*d_C*d_L + d_C*d_C);
				}
			}
		}
		else{
			double d_XL = V[(dir - 1)*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)];
			double d_XR = V[(3 + (dir - 1))*ksize + global_id] - V[(dir - 1)*ksize + global_id];
			double CFL = (V[(3 + (dir - 1))*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]) / (d_XL);
			double CBL = (V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id - 2 * (dir == 1)*isize - 2 * (dir == 2)*jsize - 2 * (dir == 3)]) /
				(V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)] - V[(dir - 1)*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]);
			double CFR = (V[(3 + (dir - 1))*ksize + global_id + (dir == 1)*isize + (dir == 2)*jsize + (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id])
				/ (V[(dir - 1)*ksize + global_id + (dir == 1)*isize + (dir == 2)*jsize + (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id]);
			double CBR = (V[(3 + (dir - 1))*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]) / (d_XR);
			for (k = 0; k<NPR; k++){
				double d_C = (p[k*ksize + global_id] - p[k*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]) / (V[(3 + (dir - 1))*ksize + global_id] - V[(3 + (dir - 1))*ksize + global_id - (dir == 1)*isize - (dir == 2)*jsize - (dir == 3)]);
				double d_R = (p[k*ksize + global_id + (dir == 1)*isize + (dir == 2)*jsize + (dir == 3)] - p[k*ksize + global_id]) / (V[(3 + (dir - 1))*ksize + global_id + (dir == 1)*isize + (dir == 2)*jsize + (dir == 3)] - V[(3 + (dir - 1))*ksize + global_id]);
				if (d_R*d_C <= 0.){
					dq[k*(ksize)+global_id] = p[k*ksize + global_id];
				}
				else{
					dq[k*(ksize)+global_id] = p[k*ksize + global_id] - (d_XR*d_R*d_C*(CFL*d_R + CBL*d_C)) / (d_R*d_R + (CFL + CBL - 2.)*d_C*d_R + d_C*d_C);
				}
			}

		}
		//if(p[0*(ksize)+global_id]/dq[0*(ksize)+global_id]>10.) printf("test %d %d %d %d: %f %f %f \n ",number ,icurr, jcurr, zcurr, log(p[0*(ksize)+global_id]),log(p[0*(ksize)+global_id-(dir==1)*isize-(dir==2)*jsize-(dir==3)]),log(dq[0*(ksize)+global_id]));
		#else
		#pragma unroll NPR	
		for (k = 0; k<NPR; k++){
			dq[k*(ksize)+global_id] = slope_lim(p[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel], p[k*(ksize)+global_id], p[k*(ksize)+global_id + idel*isize + jdel*(N3 + 2 * N3G) + zdel], 0);
		}
		#endif
	}
}


__global__ void fluxcalc2D2(int N1, int N2, int N3, __global FTYPE2 *  F, __global FTYPE2 *  dq, __global FTYPE2 *  pv, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, int lim, int dir,
	FTYPE2 gam, FTYPE2 hslope, FTYPE2 fractheta, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 cour, __global FTYPE2*  dtij, __local FTYPE2* local_dtij, int N1_MPI, int N1_MPI_offset, int POLE_1, int POLE_2,
	int N3_MPI, int N3_MPI_offset, __global FTYPE2* storage1, __global FTYPE2* storage2, __global FTYPE2* storage3, __global FTYPE2* storage4, double dx_1, double dx_2, double dx_3, __global FTYPE2 *  eta_avg, __global FTYPE2 *  ps)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int group_id = get_group_id(0);
	int isize = (N3 + 2 * D3 - (dir == 3))*(N2 + 2 * D2 - (dir == 2));
	int zcurr = (global_id % (isize)) % (N3 + 2 * D3 - (dir == 3));
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + 2 * D3 - (dir == 3));
	int icurr = (global_id - (jcurr*(N3 + 2 * D3 - (dir == 3)) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3 + (dir == 3);
	jcurr += (N2G - 1) + (dir == 2);
	icurr += (N1G - 1) + (dir == 1);
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int k = 0;
	if (global_id<(N1 + 2 * D1 - (dir == 1)) * (N2 + 2 * D2 - (dir == 2)) * (N3 + 2 * D3 - (dir == 3))) k = 1;
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel, i;
	int face;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	double factor;
	FTYPE2 cmax_r, cmin_r, cmax, cmin;
	FTYPE2 ctop;
	FTYPE2 temp3[NPR], temp4[NPR];
	FTYPE2 cmax_l, cmin_l;
	FTYPE2 p[NPR];
	FTYPE2 temp1[NPR], temp2[NPR];
	struct of_geom geom;
	struct of_state state;
	local_dtij[local_id] = 1.e9;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx_1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx_2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx_3; }

	if (k == 1){
		get_geometry(N1, N2, icurr, jcurr, zcurr, face, &geom, gcov, gcon, gdet);

		#if(PPM || LEER)
		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			p[k] = dq[k*(ksize)+global_id];
		}
		#else
		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			#if AMD
			p[k] = fma(0.5, dq[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel], pv[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel]);
			#else
			p[k] = pv[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel] + 0.5*dq[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel];
			#endif
		}
		#endif
		#if(STAGGERED)
		for (k = 0; k< NPR; k++){
			if ((dir == 1 && k == B1) || (dir == 2 && k == B2) || (dir == 3 && k == B3)){
				p[k] = ps[(k - B1)*(ksize)+global_id];
			}

			if (dir == 2 && k == B1 && ((jcurr == N2 + N2G && POLE_2 == 1) || (jcurr == N2G && POLE_1 == 1))){
				#if AMD
				p[k] = 0.;
				#else
				p[k] = 0.;
				#endif
			}
		}
		#endif

		get_state(p, &geom, &state);
		primtoflux(p, &state, dir, &geom, temp1, gam);
		primtoflux(p, &state, TT, &geom, temp2, gam);
		vchar(p, &state, &geom, dir, &cmax_l, &cmin_l, gam);

			#if(PPM || LEER)
			#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			p[k] = dq[k*(ksize)+global_id];
		}
		#else
		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			#if AMD
			p[k] = fma(-0.5, dq[k*(ksize)+global_id], pv[k*(ksize)+global_id]);
			#else
			p[k] = pv[k*(ksize)+global_id] - 0.5*dq[k*(ksize)+global_id];
			#endif
		}
		#endif
		#if(STAGGERED)
		for (k = 0; k< NPR; k++){
			if ((dir == 1 && k == B1) || (dir == 2 && k == B2) || (dir == 3 && k == B3)){
				p[k] = ps[(k - B1)*(ksize)+global_id];
			}
			if (dir == 2 && k == B1 && ((jcurr == N2 + N2G && POLE_2 == 1) || (jcurr == N2G && POLE_1 == 1))){
				#if AMD
				p[k] = 0.;
				#else
				p[k] = 0.;
				#endif
			}
		}
		#endif
		get_state(p, &geom, &state);
		primtoflux(p, &state, dir, &geom, temp3, gam);
		primtoflux(p, &state, TT, &geom, temp4, gam);
		vchar(p, &state, &geom, dir, &cmax_r, &cmin_r, gam);

		cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
		cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
		ctop = MY_MAX(cmax, cmin);
		#pragma unroll NPR	
		for (k = 0; k<NPR; k++){
			F[k*(ksize)+global_id] = HLLF*((cmax*temp1[k] +
				cmin*temp3[k] - cmax*cmin*(temp4[k] - temp2[k])) / (cmax + cmin + SMALL))
				+ LAXF*(0.5*(temp1[k]
				+ temp3[k] - ctop*(temp4[k] - temp2[k])));
		}

		cmax = MY_MAX(cmax, cmin);
		local_dtij[local_id] = factor / cmax;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (i = get_local_size(0) / 2; i>1; i = i / 2){
		if (local_id<i){
			local_dtij[local_id] = MY_MIN(local_dtij[local_id], local_dtij[local_id + i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_id == 0){
		dtij[group_id] = MY_MIN(local_dtij[0], local_dtij[1]);
	}
}

__global__ void fix_flux(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4)
{
	int global_id = get_global_id(0);
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int icurr, jcurr, zcurr;
	int k;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	if (global_id<(N1 + 2 * N1G)*(N3 + 2 * N3G)){
		zcurr = global_id % (N3 + 2 * N3G);
		icurr = (global_id - zcurr) / (N3 + 2 * N3G);
		if (icurr >= N1G - D1 && zcurr >= N3G - D3 && icurr<N1 + N1G + D1 && zcurr<N3 + N3G + D3) {
			if (NBR_1 < 0){
				F1[B2*(ksize)+icurr*isize + (N2G - 1)*(N3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + N2G*(N3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (N2G - 1)*(N3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + N2G*(N3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll NPR	
				PLOOP F2[k*(ksize)+icurr*isize + N2G*(N3 + 2 * N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll NPR	
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + N2G*(N3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
			if (NBR_3 < 0){
				F1[B2*(ksize)+icurr*isize + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + (N2 + N2G - 1)*(N3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + (N2 + N2G - 1)*(N3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll NPR	
				PLOOP F2[k*(ksize)+icurr*isize + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll NPR	
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
		}
	}
	#if INFLOW==0
	else{
		global_id = global_id - (N1 + 2 * N1G)*(N3 + 2 * N3G);
		zcurr = global_id % (N3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (N3 + 2 * N3G);
		if (jcurr >= N2G - D2 && zcurr >= N3G - D3 && jcurr<N2 + N2G + D2 && zcurr<N3 + N3G + D3) {
			if (NBR_4<0){
				if (F1[RHO*(ksize)+N1G*isize + jcurr*(N3 + 2 * N3G) + zcurr] > 0.) F1[RHO*(ksize)+N1G*isize + jcurr*(N3 + 2 * N3G) + zcurr] = 0.;
			}
			if (NBR_2<0){
				if (F1[RHO*(ksize)+(N1 + N1G)*isize + jcurr*(N3 + 2 * N3G) + zcurr] < 0.) F1[RHO*(ksize)+(N1 + N1G)*isize + jcurr*(N3 + 2 * N3G) + zcurr] = 0.;
			}
		}

	}
	#endif
}

__global__ void consttransport1(int N1, int N2, int N3, __global FTYPE2 *  pb_i, __global FTYPE2 *  E_cent, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet)
{
	int global_id = get_global_id(0);
	int isize = (N3 + N3G)*(N2 + N2G);
	int zcurr = (global_id % (isize)) % (N3 + N3G);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + N3G);
	int icurr = (global_id - (jcurr*(N3 + N3G) + zcurr)) / (isize);
	zcurr += (N3G - D3);
	jcurr += (N2G - D2);
	icurr += (N1G - D1);
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	int jsize = N3 + 2 * N3G;
	double pb[NPR];
	struct of_geom geom;
	struct of_state q;
	int k;

	if (icurr >= D1 && jcurr >= D2  && zcurr >= D3  && icurr<N1 + N1G + D1 && jcurr<N2 + N2G + D2  && zcurr<N3 + N3G + D3){
		for (k = 0; k<NPR; k++){
			pb[k] = pb_i[k*(ksize)+global_id];
		}
		get_geometry(N1, N2, icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		get_state(pb, &geom, &q);
		#if(N3G>0)
		E_cent[1 * ksize + global_id] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
		E_cent[2 * ksize + global_id] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
		#endif
		E_cent[3 * ksize + global_id] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);
	}
}

__global__ void consttransport2(int N1, int N2, int N3, __global FTYPE2 *  emf, __global FTYPE2 *  E_cent, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3,
	__global FTYPE2 *  pb_i, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, int POLE_1, int POLE_2)
{
	int global_id = get_global_id(0);
	int isize = (N3 + D3)*(N2 + D2);
	int zcurr = (global_id % (isize)) % (N3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + D3);
	int icurr = (global_id - (jcurr*(N3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	int jsize = N3 + 2 * N3G;
	double v[NDIM], pb[NPR];
	int k;
	struct of_geom geom;

	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G + D1 && jcurr<N2 + N2G + D2  && zcurr<N3 + N3G + D3){
		for (k = 0; k<NPR; k++){
			pb[k] = pb_i[k*(ksize)+global_id];
		}
		get_geometry(N1, N2, icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		ucon_calc(pb, &geom, v);

		double dE_LEFT_13_1 = E_cent[1 * (ksize)+global_id] - F3[B2*(ksize)+global_id];
		double dE_LEFT_13_2 = E_cent[1 * (ksize)+global_id - jsize*D2] - F3[B2*(ksize)+global_id - jsize*D2];
		double dE_RIGHT_13_1 = F3[B2*(ksize)+global_id + D3 - D3] - E_cent[1 * (ksize)+global_id - D3];
		double dE_RIGHT_13_2 = F3[B2*(ksize)+global_id + D3 - jsize*D2 - D3] - E_cent[1 * (ksize)+global_id - jsize*D2 - D3];
		double dE_LEFT_12_1 = E_cent[1 * (ksize)+global_id] + F2[B3*(ksize)+global_id];
		double dE_LEFT_12_2 = E_cent[1 * (ksize)+global_id - D3] + F2[B3*(ksize)+global_id - D3];
		double dE_RIGHT_12_1 = -F2[B3*(ksize)+global_id + D2*jsize - D2*jsize] - E_cent[1 * (ksize)+global_id - D2*jsize];
		double dE_RIGHT_12_2 = -F2[B3*(ksize)+global_id + D2*jsize - D2*jsize - D3] - E_cent[1 * (ksize)+global_id - D2*jsize - D3];
		double dE_LEFT_21_1 = E_cent[2 * (ksize)+global_id] - F1[B3*(ksize)+global_id];
		double dE_LEFT_21_2 = E_cent[2 * (ksize)+global_id - D3] - F1[B3*(ksize)+global_id - D3];
		double dE_RIGHT_21_1 = F1[B3*(ksize)+global_id + D1*isize - D1*isize] - E_cent[2 * (ksize)+global_id - D1*isize];
		double dE_RIGHT_21_2 = F1[B3*(ksize)+global_id + D1*isize - D1*isize - D3] - E_cent[2 * (ksize)+global_id - D1*isize - D3];
		double dE_LEFT_23_1 = E_cent[2 * (ksize)+global_id] + F3[B1*(ksize)+global_id];
		double dE_LEFT_23_2 = E_cent[2 * (ksize)+global_id - D1*isize] + F3[B1*(ksize)+global_id - D1*isize];
		double dE_RIGHT_23_1 = -F3[B1*(ksize)+global_id + D3 - D3] - E_cent[2 * (ksize)+global_id - D3];
		double dE_RIGHT_23_2 = -F3[B1*(ksize)+global_id + D3 - isize*D1 - D3] - E_cent[2 * (ksize)+global_id - isize*D1 - D3];
		double dE_LEFT_31_1 = E_cent[3 * (ksize)+global_id] + F1[B2*(ksize)+global_id];
		double dE_LEFT_31_2 = E_cent[3 * (ksize)+global_id - D2*jsize] + F1[B2*(ksize)+global_id - D2*jsize];
		double dE_RIGHT_31_1 = -F1[B2*(ksize)+global_id + D1*isize - D1*isize] - E_cent[3 * (ksize)+global_id - D1*isize];
		double dE_RIGHT_31_2 = -F1[B2*(ksize)+global_id + D1*isize - D1*isize - D2*jsize] - E_cent[3 * (ksize)+global_id - D1*isize - D2*jsize];
		double dE_LEFT_32_1 = E_cent[3 * (ksize)+global_id] - F2[B1*(ksize)+global_id];
		double dE_LEFT_32_2 = E_cent[3 * (ksize)+global_id - D1*isize] - F2[B1*(ksize)+global_id - D1*isize];
		double dE_RIGHT_32_1 = F2[B1*(ksize)+global_id + D2*jsize - D2*jsize] - E_cent[3 * (ksize)+global_id - D2*jsize];
		double dE_RIGHT_32_2 = F2[B1*(ksize)+global_id + D2*jsize - D1*isize - D2*jsize] - E_cent[3 * (ksize)+global_id - D1*isize - D2*jsize];

		emf[1 * (ksize)+global_id] = 0.25*((-F2[B3*(ksize)+global_id] - (dE_LEFT_13_1* (double)(v[2] <= 0.0) + dE_LEFT_13_2* (double)(v[2]>0.0)))
			+ (-F2[B3*(ksize)+global_id - D3] + (dE_RIGHT_13_1* (double)(v[2] <= 0.0) + dE_RIGHT_13_2* (double)(v[2]>0.0))) +
			+(F3[B2*(ksize)+global_id] - (dE_LEFT_12_1* (double)(v[3] <= 0.0) + dE_LEFT_12_2* (double)(v[3]>0.0)))
			+ (F3[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_12_1* (double)(v[3] <= 0.0) + dE_RIGHT_12_2* (double)(v[3]>0.0))));
		emf[2 * (ksize)+global_id] = 0.25*((-F3[B1*(ksize)+global_id] - (dE_LEFT_21_1* (double)(v[3] <= 0.0) + dE_LEFT_21_2* (double)(v[3]>0.0)))
			+ (-F3[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_21_1* (double)(v[3] <= 0.0) + dE_RIGHT_21_2* (double)(v[3]>0.0)))
			+ (F1[B3*(ksize)+global_id] - (dE_LEFT_23_1* (double)(v[1] <= 0.0) + dE_LEFT_23_2* (double)(v[1]>0.0)))
			+ (F1[B3*(ksize)+global_id - D3] + (dE_RIGHT_23_1* (double)(v[1] <= 0.0) + dE_RIGHT_23_2* (double)(v[1]>0.0))));
		emf[3 * (ksize)+global_id] = 0.25*((F2[B1*(ksize)+global_id] - (dE_LEFT_31_1* (double)(v[2] <= 0.0) + dE_LEFT_31_2* (double)(v[2]>0.0)))
			+ (F2[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_31_1* (double)(v[2] <= 0.0) + dE_RIGHT_31_2* (double)(v[2]>0.0)))
			+ (-F1[B2*(ksize)+global_id] - (dE_LEFT_32_1* (double)(v[1] <= 0.0) + dE_LEFT_32_2* (double)(v[1]>0.0)))
			+ (-F1[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_32_1* (double)(v[1] <= 0.0) + dE_RIGHT_32_2* (double)(v[1]>0.0))));

		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == N2 + N2G)){
			emf[3 * (ksize)+global_id] = 0.5*(F2[B1*(ksize)+global_id] + F2[B1*(ksize)+global_id - isize]);
			emf[1 * (ksize)+global_id] = -0.5*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id - D3]);
		}
	}
}

__global__ void consttransport3(int N1, int N2, int N3, double dx_1, double dx_2, double dx_3, const  FTYPE2* __restrict__ gdet_GPU, __global FTYPE2 *  psi, __global FTYPE2 *  psf,
	__global FTYPE2 *  E_corn, double Dt, int POLE_1, int POLE_2)
{
	int global_id = get_global_id(0);
	int isize = (N3 + D3)*(N2 + D2);
	int zcurr = (global_id % (isize)) % (N3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + D3);
	int icurr = (global_id - (jcurr*(N3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;

	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G + D1 && jcurr<N2 + N2G  && zcurr<N3 + N3G){
		psf[global_id] = psi[global_id] - Dt / dx_2*(E_corn[3 * ksize + global_id + (N3 + 2 * N3G)] - E_corn[3 * ksize + global_id]) / gdet_GPU[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];;
		#if(N3G>0)
		psf[global_id] += Dt / dx_3*(E_corn[2 * ksize + global_id + D3] - E_corn[2 * ksize + global_id]) / gdet_GPU[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G + D2  && zcurr<N3 + N3G){
		psf[1 * ksize + global_id] = psi[1 * ksize + global_id] + Dt / dx_1*(E_corn[3 * ksize + global_id + isize] - E_corn[3 * ksize + global_id]) / gdet_GPU[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];;
		#if(N3G>0)
		psf[1 * ksize + global_id] += -Dt / dx_3*(E_corn[1 * ksize + global_id + D3] - E_corn[1 * ksize + global_id]) / gdet_GPU[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G && zcurr<N3 + N3G + D3){
		#if(N3G>0)
		psf[2 * ksize + global_id] = psi[2 * ksize + global_id] - Dt / dx_1*(E_corn[2 * ksize + global_id + isize] - E_corn[2 * ksize + global_id]) / gdet_GPU[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj];;
		psf[2 * ksize + global_id] += Dt / dx_2*(E_corn[1 * ksize + global_id + (N3 + 2 * N3G)] - E_corn[1 * ksize + global_id]) / gdet_GPU[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + ii*(N2 + 2 * N2G) + jj]; 
		#endif
	}
}

__global__ void flux_ct1(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2 *  emf)
{
	int global_id = get_global_id(0);
	int isize = (N3 + D3)*(N2 + D2);
	int zcurr = (global_id % (isize)) % (N3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + D3);
	int icurr = (global_id - (jcurr*(N3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G + D1 && jcurr<N2 + N2G + D2  && zcurr<N3 + N3G + D3){
		#if (N2G>0 && N3G>0)
		emf[1 * (ksize)+global_id] = -0.25*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id - 1] -
			F3[B2*(ksize)+global_id] - F3[B2*(ksize)+global_id - (N3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		emf[2 * (ksize)+global_id] = -0.25*(F3[B1*(ksize)+global_id] + F3[B1*(ksize)+global_id - isize] -
			F1[B3*(ksize)+global_id] - F1[B3*(ksize)+global_id - 1]);
		#endif
		#if (N1G>0 && N2G>0)
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (N3 + 2 * N3G)] -
			F2[B1*(ksize)+global_id] - F2[B1*(ksize)+global_id - isize]);
		#else
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (N3 + 2 * N3G)]);
		#endif
	}
}

__global__ void flux_ct2(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2 *  emf)
{
	int global_id = get_global_id(0);
	int isize = (N3 + D3)*(N2 + D2);
	int zcurr = (global_id % (isize)) % (N3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + D3);
	int icurr = (global_id - (jcurr*(N3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	double emf1 = -emf[1 * (ksize)+global_id];
	double emf2 = -emf[2 * (ksize)+global_id];
	double emf3 = -emf[3 * (ksize)+global_id];
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G + D1 && jcurr<N2 + N2G && zcurr<N3 + N3G){
		#if (N1G>0)
		F1[B1*(ksize)+global_id] = 0.0;
		#endif
		#if (N1G>0 && N2G>0)
		F1[B2*(ksize)+global_id] = 0.5*(emf3 - emf[3 * (ksize)+global_id + (N3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		F1[B3*(ksize)+global_id] = -0.5*(emf2 - emf[2 * (ksize)+global_id + 1]);
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G + D2 && zcurr<N3 + N3G){
		#if (N1G>0 && N2G>0)		
		F2[B1*(ksize)+global_id] = -0.5*(emf3 - emf[3 * (ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F2[B3*(ksize)+global_id] = 0.5*(emf1 - emf[1 * (ksize)+global_id + 1]);
		#endif
		#if(N2G>0)
		F2[B2*(ksize)+global_id] = 0.0;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G && zcurr<N3 + N3G + D3){
		#if (N1G>0 && N3G>0)
		F3[B1*(ksize)+global_id] = 0.5*(emf2 - emf[2 * (ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F3[B2*(ksize)+global_id] = -0.5*(emf1 - emf[1 * (ksize)+global_id + (N3 + 2 * N3G)]);
		#endif
		#if(N3G>0)
		F3[B3*(ksize)+global_id] = 0.;
		#endif
	}
}



__global__ void fixup(int N1, int N2, int N3, __global FTYPE2* pi_i, __global FTYPE2* pb_i, __global FTYPE2* pf_i, __global FTYPE2 *  psf,
	__global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2* radius, __global int* pflag, __global int* failimage,
	const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, const  FTYPE2* __restrict__ conn, __global FTYPE2* Katm, FTYPE2 gam, FTYPE2 dx_1, FTYPE2 dx_2, FTYPE2 dx_3, FTYPE2 a, FTYPE2 Dt,
	int full_step)
{
	int global_id = get_global_id(0);
	int isize = N3*N2;
	int zcurr = (global_id % (isize)) % N3;
	int jcurr = ((global_id - zcurr) % (isize)) / (N3);
	int icurr = (global_id - (jcurr*N3 + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int k = 0;
	if (global_id<N1*N2*N3) k = 1;
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;

	struct of_geom geom;
	struct of_state q;
	int flag = 0, dofloor = 0, m;
	FTYPE2 r, X, uuscal, rhoscal, rhoflr, uuflr;
	FTYPE2 f, gamma, bsq;
	FTYPE2 pf[NPR], pf_prefloor[NPR], U_ent, dpf[NPR], U_prefloor[NPR], dU[NPR], U[NPR];
	double trans, betapar, betasq, betasqmax, one_over_ucondr_, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut;
	double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM], Xtrans;

	if (k == 1){
		get_geometry(N1, N2, icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		if (full_step == 0){
			for (k = 0; k<NPR; k++){
				pf[k] = pi_i[k*(ksize)+global_id];
			}
			get_state(pf, &geom, &q);
			primtoU(pf, &q, &geom, U, gam);
			#pragma unroll NPR	
			for (k = 0; k<NPR; k++){
				pi_i[k*(ksize)+global_id] = U[k];
			}
		}
		else{
			#pragma unroll NPR	
			for (k = 0; k<NPR; k++){
				U[k] = pi_i[k*(ksize)+global_id];
			}
			#pragma unroll NPR	
			for (k = 0; k<NPR; k++){
				pf[k] = pb_i[k*(ksize)+global_id];
			}
			if (full_step == 1){
				get_state(pf, &geom, &q);
			}
		}

		#pragma unroll NPR	
		for (k = 0; k<NPR; k++){
			#if( N1G > 0 )
			U[k] -= Dt*(F1[k*(ksize)+global_id + isize] - F1[k*(ksize)+global_id]) / dx_1;
			#endif
			#if( N2G > 0 )
			U[k] -= Dt*(F2[k*(ksize)+global_id + (N3 + 2 * N3G)] - F2[k*(ksize)+global_id]) / dx_2;
			#endif
			#if( N3G > 0 )
			U[k] -= Dt*(F3[k*(ksize)+global_id + 1] - F3[k*(ksize)+global_id]) / dx_3;
			#endif
		}

		source(N1, N2, pf, &geom, icurr, jcurr, zcurr, dU, Dt, gam, conn, &q, a, radius[icurr]);

		#pragma unroll NPR	
		for (k = 0; k< NPR; k++){
			U[k] += Dt*(dU[k]);
		}

		#if(STAGGERED)
		U[B1] = (psf[0 * ksize + global_id] * gdet[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[0 * ksize + global_id + isize] * gdet_GPU[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + (icurr+D1)*(N2 + 2 * N2G) + jcurr]) / 2.0;
		U[B2] = (psf[1 * ksize + global_id] * gdet[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[1 * ksize + global_id + (N3 + 2 * N3G)] * gdet_GPU[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + (jcurr+D2)]) / 2.0;
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id] * gdet[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[2 * ksize + global_id + D3] * gdet_GPU[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif

		pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf);
		if (pflag[global_id]){
			failimage[global_id]++;
		}

		//compute the square of fluid frame magnetic field (twice magnetic pressure)
		#if( DO_FONT_FIX ) 
		if (pflag[global_id]) {
			#if DOKTOT
			pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
			#endif
			if (pflag[global_id]) {
				failimage[1 * (ksize)+global_id]++;
				pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
				if (pflag[global_id]){
					pflag[0] = global_id;
					failimage[2 * (ksize)+global_id]++;
				}
			}
		}
		#endif

		r = radius[icurr];
		rhoscal = pow(r, -POWRHO);
		uuscal = pow(rhoscal, gam);

		rhoflr = RHOMIN*rhoscal;
		uuflr = UUMIN*uuscal;

		ucon_calc(pf, &geom, q.ucon);
		lower(q.ucon, &geom, q.ucov);
		bcon_calc(pf, q.ucon, q.ucov, q.bcon);
		lower(q.bcon, &geom, q.bcov);
		bsq = dot(q.bcon, q.bcov);

		//tie floors to the local values of magnetic field and internal energy density
		if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);
		if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
		if (rhoflr < pf[UU] / UORHOMAX) rhoflr = pf[UU] / (UORHOMAX);

		if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
		if (uuflr  < UUMINLIMIT) uuflr = UUMINLIMIT;

		//floor on density and internal energy density (momentum *not* conserved) 
		#pragma unroll NPR
		PLOOP pf_prefloor[k] = pf[k];
		if (pf[RHO] <rhoflr){
			pf[RHO] = rhoflr;
			dofloor = 1;
		}
		if (pf[UU] < uuflr){
			pf[UU] = uuflr;
			dofloor = 1;
		}

		#if( ZAMO_FLOOR )
		if (dofloor && (trans = 10.*bsq / MY_MIN(pf[RHO], pf[UU]) - 1.) > 0.) {
			//ucon_calc(pf_prefloor, &geom, q.ucon) ;
			//lower(q.ucon, &geom, q.ucov) ;
			if (trans > 1.) {
				trans = 1.;
			}

			betapar = -q.bcon[0] / ((bsq + SMALL)*q.ucon[0]);
			betasq = betapar*betapar*bsq;
			betasqmax = 1. - 1. / (GAMMAMAX*GAMMAMAX);
			if (betasq > betasqmax) {
				betasq = betasqmax;
			}
			gamma = 1. / sqrt(1 - betasq);
			#pragma unroll NDIM
			for (m = 0; m < NDIM; m++) {
				ucondr[m] = gamma*(q.ucon[m] + betapar*q.bcon[m]);
			}

			Bcon[0] = 0.;

			#pragma unroll 3
			for (m = 1; m < NDIM; m++) {
				Bcon[m] = pf[B1 - 1 + m];
			}

			lower(Bcon, &geom, Bcov);
			udotB = dot(q.ucon, Bcov);
			Bsq = dot(Bcon, Bcov);
			B = sqrt(Bsq);

			//enthalpy before the floors
			wold = pf_prefloor[RHO] + pf_prefloor[UU] * gam;

			//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
			QdotB = udotB*wold*q.ucon[0];

			//enthalpy after the floors
			wnew = pf[RHO] + pf[UU] * gam;
			//wnew = wold;

			x = 2.*QdotB / (B*wnew*ucondr[0] + SMALL);

			//new parallel velocity
			vpar = x / (ucondr[0] * (1. + sqrt(1. + x*x)));

			one_over_ucondr_t = 1. / ucondr[0];

			//new contravariant 3-velocity, v^i
			vcon[0] = 1.;

			#pragma unroll 3
			for (m = 1; m < NDIM; m++) {
				//parallel (to B) plus perpendicular (to B) velocities
				vcon[m] = vpar*Bcon[m] / (B + SMALL) + ucondr[m] * one_over_ucondr_t;
			}

			//compute u^t corresponding to the new v^i
			ut_calc_3vel(vcon, &geom, &ut);

			#pragma unroll NDIM
			for (m = 0; m < NDIM; m++) {
				ucon[m] = ut*vcon[m];
			}
			ucon_to_utcon(ucon, &geom, utcon);

			//now convert 3-vel to relative 4-velocity and put it into pv[U1..U3]
			//\tilde u^i = u^t(v^i-g^{ti}/g^{tt})
			#pragma unroll 3
			for (m = 1; m < NDIM; m++) {
				pf[m + UU] = utcon[m] * trans + pf_prefloor[m + UU] * (1. - trans);
			}
		}
		#else
		if (dofloor == 1) {
			#pragma unroll NPR
			PLOOP dpf[k] = pf[k] - pf_prefloor[k];

			//compute the conserved quantity associated with floor addition
			get_state(dpf, &geom, &q);
			primtoU(dpf, &q, &geom, dU, gam);

			//compute the prefloor conserved quantity
			get_state(pf_prefloor, &geom, &q);
			primtoU(pf_prefloor, &q, &geom, U_prefloor, gam);

			//add U_added to the current conserved quantity
			#pragma unroll NPR
			PLOOP U[k] = U_prefloor[k] + dU[k];

			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf);
			if (pflag[global_id]){
				failimage[global_id]++;
				#if( DO_FONT_FIX ) 
				U_ent = (geom.g*pf[0] * (gam - 1.)*pf[1] / pow(pf[0], gam)) * (q.ucon[0]);
				pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
				if (pflag[global_id]) {
					failimage[1 * (ksize)+global_id]++;
					pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, Katm[icurr]);
					if (pflag[global_id]){
						pflag[0] = 100;
						failimage[2 * (ksize)+global_id]++;
					}
				}
				#else
				pflag[0] = 100;
				#endif	
			}
		}
		#endif

		// limit gamma wrt normal observer 
		if (gamma_calc(pf, &geom, &gamma)) {
			// Treat gamma failure here as "fixable" for fixup_utoprim() 
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}
		else {
			if (gamma > GAMMAMAX) {
				f = sqrt(
					(GAMMAMAX*GAMMAMAX - 1.) /
					(gamma*gamma - 1.)
					);
				pf[U1] *= f;
				pf[U2] *= f;
				pf[U3] *= f;
			}
		}
		#if DOKTOT
		pf_i[KTOT*(ksize)+global_id] = (gam - 1.)*pf[UU] * pow(pf[RHO], -gam);
		#endif
		#pragma unroll NPR	
		for (k = 0; k< NPR - DOKTOT; k++){
			pf_i[k*(ksize)+global_id] = pf[k];
		}
	}
}

/* 26 */
#define AVG2_1(pr,icurr,jcurr,zcurr, N1, N2, N3,k) (0.5*(pr[k*(ksize)+(icurr)*isize+(jcurr+1)*(N3+2*N3G) + zcurr]+pr[k*(ksize)+(icurr)*isize+(jcurr-1)*(N3+2*N3G)+ zcurr]))

/* 48 */
#define AVG2_2(pr,icurr,jcurr,zcurr, N1, N2, N3,k) (0.5*(pr[k*(ksize)+(icurr-1)*isize+(jcurr)*(N3+2*N3G) + zcurr]+pr[k*(ksize)+(icurr+1)*isize+(jcurr)*(N3+2*N3G)+ zcurr]))

/* 910 */
#define AVG2_3(pr,icurr,jcurr,zcurr, N1, N2, N3,k) (0.5*(pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G) + (zcurr-1)]+pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G)+ (zcurr+1)]))

/* 2468  */
#define AVG4_1(pr,icurr,jcurr,zcurr, N1, N2, N3,k) (0.25*(pr[k*(ksize)+(icurr)*isize+(jcurr+1)*(N3+2*N3G) + zcurr]+pr[k*(ksize)+(icurr)*isize+(jcurr-1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr-1)*isize+(jcurr)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr+1)*isize+(jcurr)*(N3+2*N3G)+ zcurr]))

/* 1357  */
#define AVG4_2(pr,icurr,jcurr,zcurr, N1, N2, N3, k) (0.25*(pr[k*(ksize)+(icurr+1)*isize+(jcurr+1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr+1)*isize+(jcurr-1)+ zcurr]*(N3+2*N3G)+pr[k*(ksize)+(icurr-1)*isize+(jcurr+1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr-1)*isize+(jcurr-1)*(N3+2*N3G)+ zcurr]))

/* 2468910  */
#define AVG6_1(pr,icurr,jcurr,zcurr, N1, N2, N3,k) (1.0/6.0*(pr[k*(ksize)+(icurr)*isize+(jcurr+1)*(N3+2*N3G) + zcurr]+pr[k*(ksize)+(icurr)*isize+(jcurr-1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr-1)*isize+(jcurr)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr+1)*isize+(jcurr)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G) + (zcurr+1)]+pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G) + (zcurr-1)]))

/* 1357910  */
#define AVG6_2(pr,icurr,jcurr,zcurr, N1, N2, N3, k) (1.0/6.0*(pr[k*(ksize)+(icurr+1)*isize+(jcurr+1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr+1)*isize+(jcurr-1)+ zcurr]*(N3+2*N3G)+pr[k*(ksize)+(icurr-1)*isize+(jcurr+1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr-1)*isize+(jcurr-1)*(N3+2*N3G)+ zcurr]+pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G) + (zcurr+1)]+pr[k*(ksize)+(icurr)*isize+(jcurr)*(N3+2*N3G) + (zcurr-1)]))

__global__ void fixuputoprim(int N1, int N2, int N3, __global FTYPE2 *  pv, __global int *  pflag, __global int *  failimage, const  FTYPE2* __restrict__ gcov,
	const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 fractheta, FTYPE2 hslope, FTYPE2 gam)
{
	int global_id = get_global_id(0);
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int zcurr = (global_id % (isize)) % (N3 + 2 * N3G);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + 2 * N3G);
	int icurr = (global_id - (jcurr*(N3 + 2 * N3G) + zcurr)) / (isize);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int k;
	int pf[11];
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	/* Fix the interior points first */
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G && zcurr<N3 + N3G) {
		if (pflag[global_id] != 0) {
			pf[1] = !pflag[(icurr - 1)*isize + (jcurr + 1)*(N3 + 2 * N3G) + zcurr];   pf[2] = !pflag[(icurr)*isize + (jcurr + 1)*(N3 + 2 * N3G) + zcurr];  pf[3] = !pflag[(icurr + 1)*isize + (jcurr + 1)*(N3 + 2 * N3G) + zcurr];
			pf[8] = !pflag[(icurr - 1)*isize + (jcurr)*(N3 + 2 * N3G) + zcurr];                           pf[4] = !pflag[(icurr + 1)*isize + (jcurr)*(N3 + 2 * N3G) + zcurr];
			pf[7] = !pflag[(icurr - 1)*isize + (jcurr - 1)*(N3 + 2 * N3G) + zcurr];   pf[6] = !pflag[(icurr)*isize + (jcurr - 1)*(N3 + 2 * N3G) + zcurr];  pf[5] = !pflag[(icurr + 1)*isize + (jcurr - 1)*(N3 + 2 * N3G) + zcurr];
			#if(N3G>0)
			pf[9] = !pflag[(icurr)*isize + (jcurr)*(N3 + 2 * N3G) + (zcurr + 1)];	pf[10] = !pflag[(icurr)*isize + (jcurr)*(N3 + 2 * N3G) + (zcurr - 1)];
			#else	
			pf[9] = 0;	pf[10] = 0;
			#endif																 
			/* Now the pf's  are true if they represent good points */
			failimage[4 * (ksize)+global_id]++;
			/* if nothing better to do, then leave densities and B-field unchanged, set v^i = 0 */
			pv[0 * (ksize)+global_id] = 0.5*(AVG4_1(pv, icurr, jcurr, zcurr, N1, N2, N3, 0) + AVG4_2(pv, icurr, jcurr, zcurr, N1, N2, N3, 0));
			pv[1 * (ksize)+global_id] = 0.5*(AVG4_1(pv, icurr, jcurr, zcurr, N1, N2, N3, 1) + AVG4_2(pv, icurr, jcurr, zcurr, N1, N2, N3, 1));
			pv[2 * (ksize)+global_id] = pv[3 * (ksize)+global_id] = pv[4 * (ksize)+global_id] = 0.;
			pflag[global_id] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
		}
	}
}

__global__ void boundprim1(int N1, int N2, int N3, __global FTYPE2 *   pv, __global int *  pflag, __global int *  failimage, const  FTYPE2* __restrict__ gcov,
	const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, __global FTYPE2* pbound, FTYPE2 Rin, FTYPE2 R0, FTYPE2 Rout, FTYPE2 fractheta,
	FTYPE2 hslope, FTYPE2 gam, int freeze, int NBR_2, int NBR_4, __global FTYPE2 *  ps)
{
	int global_id = get_global_id(0);
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int k;
	int zcurr = global_id % (N3 + 2 * N3G);
	int jcurr = (global_id - zcurr) / (N3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	FTYPE2 prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];
	int prim7, prim8;

	// inner r boundary condition: u, gdet extrapolation 
	if (jcurr >= 0 && jcurr<N2 + 2 * N2G && zcurr >= 0 && zcurr<N3 + 2 * N3G && NBR_4 == -1){
		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			prim5[k] = pv[k*(ksize)+N1G*isize + global_id];
		}
		prim7 = pflag[0 * isize + global_id];

		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			prim1[k] = prim5[k];
			prim2[k] = prim5[k];
			#if(N1G==3)
			prim3[k] = prim5[k];
			#endif
		}

		/*Make sure there is no inflow at inner boundary*/
		inflow_check(N1, N2, prim1, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		inflow_check(N1, N2, prim2, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(N1, N2, prim3, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		#endif
		inflow_check(N1, N2, prim1, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
		inflow_check(N1, N2, prim2, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(N1, N2, prim3, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
		#endif
		/*Write primitives back to global memory*/
		#pragma unroll NPR
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+global_id] = prim2[k];
			pv[k*(ksize)+1 * isize + global_id] = prim1[k];
			#if(N1G==3)
			pv[k*(ksize)+2 * isize + global_id] = prim3[k];
			#endif
		}

		#if(STAGGERED)
		ps[1 * (ksize)+0 * isize + global_id] = ps[1 * (ksize)+N1G*isize + global_id];
		ps[1 * (ksize)+1 * isize + global_id] = ps[1 * (ksize)+N1G*isize + global_id];
		ps[2 * (ksize)+0 * isize + global_id] = ps[2 * (ksize)+N1G*isize + global_id];
		ps[2 * (ksize)+1 * isize + global_id] = ps[2 * (ksize)+N1G*isize + global_id];
		#if(N1G==3)
		ps[1 * (ksize)+2 * isize + global_id] = ps[1 * (ksize)+N1G*isize + global_id];
		ps[2 * (ksize)+2 * isize + global_id] = ps[2 * (ksize)+N1G*isize + global_id];
		#endif
		#endif

		global_id = -10;
		jcurr = -10;
		zcurr = -10;
	}

	if (global_id<isize){
		global_id = -10;
		jcurr = -10;
		zcurr = -10;
	}
	else if (global_id >= isize){
		global_id = global_id - isize;
		zcurr = global_id % (N3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (N3 + 2 * N3G);
	}

	// outer r BC: outflow 
	if (jcurr >= 0 && jcurr<N2 + 2 * N2G && zcurr >= 0 && zcurr<N3 + 2 * N3G && NBR_2 == -1){
		#pragma unroll NPR
		for (k = 0; k< NPR; k++){
			prim6[k] = pv[k*(ksize)+(N1 + N1G - 1)*isize + global_id];
		}

		#pragma unroll NPR
		for (k = 0; k<NPR; k++){
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}
		prim8 = pflag[(N1 + N1G - 1)*isize + global_id];

		/*Make sure there is no inflow at outer boundary*/
		inflow_check(N1, N2, prim3, N1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		inflow_check(N1, N2, prim4, N1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(N1, N2, prim5, N1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		#endif
		inflow_check(N1, N2, prim3, N1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		inflow_check(N1, N2, prim4, N1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(N1, N2, prim5, N1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		#endif

		#pragma unroll NPR
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+(N1 + N1G)*isize + global_id] = prim3[k];
			pv[k*(ksize)+(N1 + N1G + 1)*isize + global_id] = prim4[k];
		#if(N1G==3)
			pv[k*(ksize)+(N1 + N1G + 2)*isize + global_id] = prim5[k];
		#endif
		}
		#if(STAGGERED)
		ps[1 * (ksize)+(N1 + N1G)*isize + global_id] = ps[1 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		ps[1 * (ksize)+(N1 + N1G + 1)*isize + global_id] = ps[1 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(N1 + N1G)*isize + global_id] = ps[2 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(N1 + N1G + 1)*isize + global_id] = ps[2 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		#if(N1G==3)
		ps[1 * (ksize)+(N1 + N1G + 2)*isize + global_id] = ps[1 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(N1 + N1G + 2)*isize + global_id] = ps[2 * (ksize)+(N1 + N1G - 1)*isize + global_id];
		#endif
		#endif
	}
}

__global__ void boundprim2(int N1, int N2, int N3, __global int *  pflag, __global FTYPE2 *  pv, int NBR_1, int NBR_3, const  FTYPE2* __restrict__ gdet, int AMR_POLE, __global FTYPE2 *  ps)
{
	int j, jref, k;
	int global_id = get_global_id(0);
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int zcurr = global_id % (N3 + 2 * N3G);
	int icurr = (global_id - zcurr) / (N3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	jref = POLEFIX;

	// polar BCs
	if (icurr >= 0 && icurr<N1 + 2 * N1G && zcurr >= 0 && zcurr<N3 + 2 * N3G && NBR_1 == -1) {
		for (j = 0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3 * (ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[3 * (ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];

			//everything else copy (both poles)
			pv[0 * (ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[0 * (ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[1 * (ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[1 * (ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[2 * (ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[2 * (ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[4 * (ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[4 * (ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];
			#if (N2G==0)
			//pv[7*(ksize)+isize*icurr+(j+N2G)*(N3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(jref+N2G)*(N3+2*N3G)+zcurr];
			#endif
			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (jref + N2G)*(N3 + 2 * N3G) + zcurr];
			#endif
		}
		#pragma unroll NPR
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr + (N2G - 1)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G)*(N3 + 2 * N3G) + zcurr];
			pv[k*(ksize)+isize*icurr + (N2G - 2)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G + 1)*(N3 + 2 * N3G) + zcurr];
			#if(N2G==3)
			pv[k*(ksize)+isize*icurr + (N2G - 3)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G + 2)*(N3 + 2 * N3G) + zcurr];
			#endif
		}

		// make sure b and u are antisymmetric at the poles 
		for (j = 0; j<N2G; j++) {
			pv[3 * (ksize)+isize*icurr + j*(N3 + 2 * N3G) + zcurr] *= -1.;
			pv[6 * (ksize)+isize*icurr + j*(N3 + 2 * N3G) + zcurr] *= -1.;
		}

		#if(STAGGERED)
		ps[0 * (ksize)+isize*icurr + (N2G - 1)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G)*(N3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+isize*icurr + (N2G - 2)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G + 1)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 1)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 2)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G + 1)*(N3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+isize*icurr + (N2G - 3)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G + 2)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 3)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G + 2)*(N3 + 2 * N3G) + zcurr];
		#endif
		#endif
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}

	if (global_id<(N1 + 2 * N1G)*(N3 + 2 * N3G)){
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}
	else if (global_id >= (N1 + 2 * N1G)*(N3 + 2 * N3G)){
		global_id = global_id - (N1 + 2 * N1G)*(N3 + 2 * N3G);
		zcurr = global_id % (N3 + 2 * N3G);
		icurr = (global_id - zcurr) / (N3 + 2 * N3G);
	}

	if (icurr >= 0 && icurr<N1 + 2 * N1G && zcurr >= 0 && zcurr<N3 + 2 * N3G && NBR_3 == -1) {
		for (j = 0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3 * (ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[3 * (ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];

			//everything else copy (both poles)
			pv[0 * (ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[0 * (ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[1 * (ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[1 * (ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[2 * (ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[2 * (ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];
			pv[4 * (ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[4 * (ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];
			#if (N2G==0)
			//pv[7*(ksize)+isize*icurr+(N2-1-j+N2G)*(N3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(N2-1-jref+N2G)*(N3+2*N3G)+zcurr];		
			#endif
			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (N2 - 1 - j + N2G)*(N3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (N2 - 1 - jref + N2G)*(N3 + 2 * N3G) + zcurr];
			#endif
		}
		#pragma unroll NPR
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2 + N2G - 1)*(N3 + 2 * N3G) + zcurr];
			pv[k*(ksize)+isize*icurr + (N2 + N2G + 1)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2 + N2G - 2)*(N3 + 2 * N3G) + zcurr];
			#if(N2G==3)
			pv[k*(ksize)+isize*icurr + (N2 + N2G + 2)*(N3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2 + N2G - 3)*(N3 + 2 * N3G) + zcurr];
			#endif
		}

		// make sure b and u are antisymmetric at the poles 
		for (j = N2 + N2G; j<N2 + 2 * N2G; j++) {
			pv[3 * (ksize)+isize*icurr + j*(N3 + 2 * N3G) + zcurr] *= -1.;
			pv[6 * (ksize)+isize*icurr + j*(N3 + 2 * N3G) + zcurr] *= -1.;
		}

		#if(STAGGERED)
		ps[0 * (ksize)+isize*icurr + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2 + N2G - 1)*(N3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+isize*icurr + (N2 + N2G + 1)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2 + N2G - 2)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2 + N2G - 1)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2 + N2G + 1)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2 + N2G - 2)*(N3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+isize*icurr + (N2 + N2G + 2)*(N3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2 + N2G - 3)*(N3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2 + N2G + 2)*(N3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2 + N2G - 3)*(N3 + 2 * N3G) + zcurr];
		#endif
		#endif
	}
}

/*The minimum timeinterval at which refinement takes place, TREF can't go below it*/
#define AMR_SWITCHTIMELEVEL 32
#define NT 0

__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (i = i1; i <i2; i++){
			send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
	}
	#if(STAGGERED)
	for (i = i1; i <i2; i++){
		send[(NPR + 0)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+(i + N1G + (i1>N1G))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G), FACE1, 0)));
		send[(NPR + 1)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
		send[(NPR + 2)*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
	}
	#endif
}

__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (j = j1; j <j2; j++){
			send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
	}
	#if(STAGGERED)
	for (j = j1; j <j2; j++){
		send[(NPR + 0)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
		send[(NPR + 1)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[1 * (ksize)+icurr*isize + (j + N2G + (j1>N2G))*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G), icurr, FACE2, 0)));
		send[(NPR + 2)*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
	}
	#endif
}

__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (z = z1; z <z2; z++){
			send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
		}
	}
	#if(STAGGERED)
	for (z = z1; z <z2; z++){
		send[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
		send[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
		send[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G))] * readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
	}
	#endif
}

__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;

	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (i = i1; i <i2; i += 1 + REF_1){
			send[k*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.125*(
				pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
				pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] +
				pv[k*(ksize)+(i + REF_1 + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + REF_1 + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
		}
	}
	#if(STAGGERED)
	for (i = i1; i <i2; i += 1 + REF_1){
		send[(NPR + 0)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
		ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
		ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))) +
		ps[0 * (ksize)+(i + N1G + (i1>N1G)*(1 + REF_1))*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1>N1G)*(1 + REF_1))*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + (i1>N1G)*(1 + REF_1), FACE1, 0))));

		send[(NPR + 1)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0))) +
		ps[1 * (ksize)+(i + N1G)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0))) +
		ps[1 * (ksize)+(i + N1G + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE2, 0))) +
		ps[1 * (ksize)+(i + N1G + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE2, 0))));

		send[(NPR + 2)*jsize2*zsize2*(i2 - i1) / (1 + REF_1) + (i - i1) / (1 + REF_1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0))) +
		ps[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G, FACE3, 0))) +
		ps[2 * (ksize)+(i + N1G + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + REF_1, FACE3, 0))) +
		ps[2 * (ksize)+(i + N1G + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + REF_1)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G + REF_1, FACE3, 0))));
	}
	#endif
}

__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (j = j1; j <j2; j += 1 + REF_2){
			send[k*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.125*(
				pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
				pv[k*(ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] +
				pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
		}
	}
	#if(STAGGERED)
	for (j = j1; j <j2; j += 1 + REF_2){
		send[(NPR + 0)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j+N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE1, 0))));

		send[(NPR + 1)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr, FACE2, 0))) +
		ps[1 * (ksize)+(icurr)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr, FACE2, 0))) +
		ps[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr + REF_1, FACE2, 0))) +
		ps[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G + (j1>N2G)*(1 + REF_2))*(BS_3 + 2 * N3G) + (zcurr + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1>N2G)*(1 + REF_2))];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1>N2G)*(1 + REF_2), icurr + REF_1, FACE2, 0))));

		send[(NPR + 2)*isize2*zsize2*(j2 - j1) / (1 + REF_2) + (j - j1) / (1 + REF_2)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] =
			0.25*(ps[2 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0))) +
		ps[2 * (ksize)+(icurr)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr, FACE3, 0))) +
		ps[2 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr + REF_1, FACE3, 0))) +
		ps[2 * (ksize)+(icurr + REF_1)*isize + (j + N2G + REF_2)*(BS_3 + 2 * N3G) + (zcurr)] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (j + N2G +REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + REF_2, icurr + REF_1, FACE3, 0))));
	}
	#endif
}

__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (k = 0; k < NPR; k++){
		#pragma unroll NG
		for (z = z1; z <z2; z += 1 + REF_3){
			send[k*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = 0.125*(
				pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + REF_3] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] +
				pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G + REF_3] +
				pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G + REF_3]);
		}
	}
	#if(STAGGERED)
	for (z = z1; z <z2; z += 1 + REF_3){
		send[(NPR + 0)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
			0.25*(ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr + REF_2)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE1, 0))) +
		ps[0 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr + REF_2)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE1, 0))));

		send[(NPR + 1)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
			0.25*(ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) +
		ps[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) +
		ps[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE2, 0))) +
		ps[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + REF_3)] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + REF_1)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE2, 0))));

		send[(NPR + 2)*isize2*jsize2*(z2 - z1) / (1 + REF_3) + (z - z1) / (1 + REF_3)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] =
			0.25*(ps[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0))) +
		ps[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE3, 0))) +
		ps[2 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE3, 0))) +
		ps[2 * (ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + (z + N3G + (z1>N3G)*(1 + REF_3))] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr + REF_1, FACE3, 0))));
	}
	#endif
}

__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	//When at timestep where n_rec does not evolve
	if (nstep != -1 && timelevel_rec>timelevel){
		for (k = 0; k < NPR; k++){
			for (i = i1; i <i2; i++){
				p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
			}
		}
		for (i = i1; i <i2; i++){
			/*ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			if((jcurr<N2G || jcurr>=BS_2+N2G || zcurr<N3G || zcurr>=BS_3+N3G)&& update_staggered==1){
			ps[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			}
			else psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			*/
		}
	}

	//When at same timestep where n_rec evolves
	if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec>timelevel){
		for (k = 0; k < NPR; k++){
			for (i = i1; i <i2; i++){
				p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
			}
			for (i = i1; i <i2; i++){
				tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] - p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
			}
		}
		for (i = i1; i <i2; i++){
			/*ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			if((jcurr<N2G || jcurr>=BS_2+N2G || zcurr<N3G || zcurr>=BS_3+N3G)&& update_staggered==1){
			ps[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			psh[0*(ksize)+(i+N1G+(i1<0))*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			}
			else psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*jsize2*zsize2*(i2-i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
			*/
		}
		for (i = i1; i <i2; i++){
			tempreceive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
				= (receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1<0))*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0))) 
				- ps[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
			tempreceive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
				= (receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)))
				- ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
			tempreceive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
				= (receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)))
				- ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
		}
	}

	//Reset gradient to 0 at eg refinement steps
	if (nstep == -1){
		for (k = 0; k < NPR + 3; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				tempreceive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] = 0.0;
			}
		}
	}

	//Reset primitve variables
	if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
			}
		}
		#if(STAGGERED)
		for (i = i1; i <i2; i++){
			if ((jcurr<N2G || jcurr >= BS_2 + N2G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
				ps[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1<0))*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0)));
				psh[0 * (ksize)+(i + N1G + (i1<0))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G + (i1<0))*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G + (i1<0), FACE1, 0)));
			}
			else psh[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE1, 0)));
			ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE3((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));

			psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*jsize2*zsize2*(i2 - i1) + (i - i1 + i_offset * N1G / (1 + REF_1))*jsize2*zsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
		}
		#endif
	}
}

__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int reverse, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;
	if (reverse == 0){
		//When at timestep where n_rec does not evolve
		if (nstep != -1 && timelevel_rec>timelevel){
			for (k = 0; k < NPR; k++){
				for (j = j1; j <j2; j++){
					p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
			}
			for (j = j1; j <j2; j++){
				/*ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+=0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				if((icurr<N1G || icurr>=BS_1+N1G || zcurr<N3G || zcurr>=BS_3+N3G) && update_staggered==1 ){
				ps[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				}
				else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				*/
			}
		}

		//When at same timestep where n_rec evolves
		if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec>timelevel){
			for (k = 0; k < NPR; k++){
				for (j = j1; j <j2; j++){
					p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
				for (j = j1; j <j2; j++){
					tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
						= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] - p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				}
			}
			for (j = j1; j <j2; j++){
				/*ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(0+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				if((icurr<N1G || icurr>=BS_1+N1G || zcurr<N3G || zcurr>=BS_3+N3G) && update_staggered==1 ){
				ps[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[1*(ksize)+icurr*isize+(j+N2G+(j1<0))*(BS_3+2*N3G)+zcurr] -= 0.5*dt*(double)timelevel_rec*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				}
				else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] += 0.5*dt*(double)timelevel*tempreceive[(1+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]-=0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*zsize2*(j2-j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))];
				*/
			}
			for (j = j1; j <j2; j++){
				tempreceive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0))) 
					- ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				tempreceive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G +(j1<0))]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0))) 
					- ps[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
				tempreceive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))]
					= (receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0))) 
					- ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]) / (0.5*dt*timelevel_rec);
			}
		}

		//Reset gradient to 0 at eg refinement steps
		if (nstep == -1){
			for (k = 0; k < NPR + 3; k++){
				#pragma unroll NG
				for (j = j1; j <j2; j++){
					tempreceive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] = 0.0;
				}
			}
		}

		//Reset primitve variables
		if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
			for (k = 0; k < NPR; k++){
				#pragma unroll NG
				for (j = j1; j <j2; j++){
					p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
					ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				}
			}
			#if(STAGGERED)
			for (j = j1; j <j2; j++){
				ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
				if ((icurr<N1G || icurr >= BS_1 + N1G || zcurr<N3G || zcurr >= BS_3 + N3G) && update_staggered == 1){
					ps[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1<0))]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0)));
					psh[1 * (ksize)+icurr*isize + (j + N2G + (j1<0))*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
						gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G + (j1<0))]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G + (j1<0), icurr, FACE2, 0)));
				}
				else psh[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(1 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0)));
				ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
				psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));

				psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j - j1 + j_offset * N2G / (1 + REF_2))*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			}
			#endif
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (j = j1; j <j2; j++){
				if (k == 3 || k == 4 || k == 6 || k == 7) factor = -1.;
				else factor = 1.;
				p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
				ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = factor*receive[k*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))];
			}
		}
		#if(STAGGERED)
		for (j = j1; j <j2; j++){
			ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			if (update_staggered == 1){
				//ps[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
				//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
				//psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
				//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
			}
			//else psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = -receive[(1+NPR)*isize2*zsize2*(j2-j1) + (j2-j-1)*isize2*zsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*zsize2+(zcurr-z1-N3G + z_offset * N3G / (1 + REF_3))]/
			//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j+N2G,icurr,FACE2,0)));
			ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(0 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -receive[(2 + NPR)*isize2*zsize2*(j2 - j1) + (j2 - j - 1)*isize2*zsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*zsize2 + (zcurr - z1 - N3G + z_offset * N3G / (1 + REF_3))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j + N2G)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
		}
		#endif
	}
}

__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	//When at timestep where n_rec does not evolve
	if (nstep != -1 && timelevel_rec>timelevel){
		for (k = 0; k < NPR; k++){
			for (z = z1; z <z2; z++){
				p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += 0.5*dt*(double)timelevel*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
			}
		}
		for (z = z1; z <z2; z++){
			/*ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]+=0.5*dt*(double)timelevel*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			if((icurr<N1G || icurr>=BS_1+N1G || jcurr<N2G || jcurr>=BS_2+N2G) && update_staggered==1 ){
			ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			}
			else psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] += 0.5*dt*(double)timelevel*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			*/
		}
	}

	//When at same timestep where n_rec evolves
	if (nstep%timelevel_rec == timelevel_rec - 1 && timelevel_rec>timelevel){
		for (k = 0; k < NPR; k++){
			for (z = z1; z <z2; z++){
				p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
			}
			for (z = z1; z <z2; z++){
				tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
					= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] - p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
			}
		}
		for (z = z1; z <z2; z++){
			/*ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+0)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)]-=0.5*dt*(double)timelevel_rec*tempreceive[(NPR+1)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			if((icurr<N1G || icurr>=BS_1+N1G || jcurr<N2G || jcurr>=BS_2+N2G) && update_staggered==1 ){
			ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G+(z1<0))] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			}
			else psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] -= 0.5*dt*(double)timelevel_rec*tempreceive[(2+NPR)*isize2*jsize2*(z2-z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr-i1-N1G + i_offset * N1G / (1 + REF_1))*jsize2+(jcurr-j1-N2G + j_offset * N2G / (1 + REF_2))];
			*/
		}
		for (z = z1; z <z2; z++){
			tempreceive[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
				= (receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0))) 
				- ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
			tempreceive[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
				= (receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0))) 
				- ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]) / (0.5*dt*timelevel_rec);
			tempreceive[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))]
				= (receive[(NPR + 2)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0))) 
				- ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))]) / (0.5*dt*timelevel_rec);
		}
	}

	//Reset gradient to 0 at eg refinement steps
	if (nstep == -1){
		for (k = 0; k < NPR + 3; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				tempreceive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] = 0.0;
			}
		}
	}

	//Reset primitve variables
	if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
				ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))];
			}
		}
		#if(STAGGERED)
		for (z = z1; z <z2; z++){
			ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			if ((icurr<N1G || icurr >= BS_1 + N1G || jcurr<N2G || jcurr >= BS_2 + N2G) && update_staggered == 1){
				ps[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
				psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G + (z1<0))] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
					gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
			}
			else psh[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(2 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
			psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(0 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(1 + NPR)*isize2*jsize2*(z2 - z1) + (z - z1 + z_offset * N3G / (1 + REF_3))*isize2*jsize2 + (icurr - i1 - N1G + i_offset * N1G / (1 + REF_1))*jsize2 + (jcurr - j1 - N2G + j_offset * N2G / (1 + REF_2))] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
		}
		#endif
	}
}
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int i, ii, ij, iz, is, js, zs, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;

	#pragma unroll NG
	for (i = i1; i <i2; i++){
		if (i1 < 0 && REF_1 == 1) ii = REF_1;
		else if (REF_1 == 1) ii = 0;
		else ii = i - i1;
		ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + REF_2)) / (1 + REF_2) + REF_2;
		iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + REF_3)) / (1 + REF_3) + REF_3;

		is = ((i == i1) ? (-1) : (1));
		js = (((jcurr - j1 - N2G) % (1 + REF_2) == 0) ? (-1) : (1));
		zs = (((zcurr - z1 - N3G) % (1 + REF_3) == 0) ? (-1) : (1));
		for (k = 0; k < NPR; k++){
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];
				temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];
			}

			if (nstep == -1 && timelevel_rec>timelevel)temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz];

			if (nstep != -1 && timelevel_rec>timelevel){
				receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] = temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] +
					(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] - temp2receive[(k)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz]);
			}
		}
		for (k = 0; k < NPR; k++){
			avg = 0.5*(prim[k*(ksize)+(N1G + (1 - ii)*(BS_1 - 1))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + prim[k*(ksize)+(ii + N1G + (1 - ii)*(BS_1 - 2))*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
			if (ii == 0){
				dq1 = slope_lim(avg, receive[k*jsize2*zsize2*(i2 - i1) + 0 * jsize2*zsize2 + ij*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + REF_1*jsize2*zsize2 + ij*zsize2 + iz]);
			}
			else{
				dq1 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + 0 * jsize2*zsize2 + ij*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + REF_1*jsize2*zsize2 + ij*zsize2 + iz], avg);
			}
			//if(jcurr-N2G-j1<(j2-j1)/2)dq2=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij+REF_2)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//else dq2=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij-REF_2)*zsize2+iz];
			//if(zcurr-N3G-z1<(z2-z1)/2)dq3=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz+REF_3]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
			//else dq3=receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz]-receive[k*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + (ij)*zsize2+iz-REF_3];
			dq2 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij - REF_2)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]);
			dq3 = slope_lim(receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + (iz - REF_3)], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + iz], receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij)*zsize2 + (iz + REF_3)]);
			//dq2=dq3=0.;
			p[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			ph[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[k*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
		}

		#if(STAGGERED)
		if (js == 1){
			ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]) /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i+N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + (ij + REF_2)*zsize2 + iz]) /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
		}
		else{
			ps[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
			psh[1 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 1)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE2, 0)));
		}
		if (zs == 1){
			ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + (iz + REF_3)]) /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
			psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] + receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + (iz + REF_3)]) /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
		}
		else{
			ps[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
			psh[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*jsize2*zsize2*(i2 - i1) + ii*jsize2*zsize2 + ij*zsize2 + iz] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE3, 0)));
		}


		//ps[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
		//ps[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
		//ps[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
		//psh[0*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
		//psh[1*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
		//psh[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*jsize2*zsize2*(i2-i1) + ii*jsize2*zsize2 + ij*zsize2+iz];
	#endif
	}
}

__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int j, ii, ij, iz, is, js, zs, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;

	#pragma unroll NG
	for (j = j1; j <j2; j++){
		if (j1 < 0 && REF_2 == 1) ij = REF_2;
		else if (REF_2 == 1) ij = 0;
		else ij = j - j1;

		ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + REF_1)) / (1 + REF_1) + REF_1;
		iz = (zcurr - z1 - N3G - (zcurr - z1 - N3G) % (1 + REF_3)) / (1 + REF_3) + REF_3;

		is = (((icurr - i1 - N1G) % (1 + REF_1) == 0) ? (-1) : (1));
		js = ((j == j1) ? (-1) : (1));
		zs = (((zcurr - z1 - N3G) % (1 + REF_3) == 0) ? (-1) : (1));

		for (k = 0; k < NPR; k++){
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];
				temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];
			}

			if (nstep == -1 && timelevel_rec>timelevel)temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz];

			if (nstep != -1 && timelevel_rec>timelevel){
				receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] = temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] +
					(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] - temp2receive[(k)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz]);
			}
		}
		for (k = 0; k < NPR; k++){
			avg = 0.5*(prim[k*(ksize)+icurr*isize + (N2G + (1 - ij)*(BS_2 - 1))*(BS_3 + 2 * N3G) + zcurr] + prim[k*(ksize)+icurr*isize + (ij + N2G + (1 - ij)*(BS_2 - 2))*(BS_3 + 2 * N3G) + zcurr]);
			if (ij == 0){
				dq2 = slope_lim(avg, receive[k*isize2*zsize2*(j2 - j1) + 0 * isize2*zsize2 + ii*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + REF_2*isize2*zsize2 + ii*zsize2 + iz]);
			}
			else{
				dq2 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + 0 * isize2*zsize2 + ii*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + REF_2*isize2*zsize2 + ii*zsize2 + iz], avg);
			}
			//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii+REF_1)*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//else dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii-REF_1)*zsize2+iz];
			//if(zcurr-N3G-z1<(z2-z1)/2)dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz+REF_3)]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//else dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz-REF_3)];
			//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii+REF_1)*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			//else dq1=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + (ii-REF_1)*zsize2+iz];
			//dq3=receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+(iz+REF_3)]-receive[k*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
			dq1 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii - REF_1)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]);
			dq3 = slope_lim(receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + (iz - REF_3)], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + iz], receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii)*zsize2 + (iz + REF_3)]);
			//dq1=dq3=0.;
			p[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			ph[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[k*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
		}

		#if(STAGGERED)
		if (is == 1){
			ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]) /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + (ii + REF_1)*zsize2 + iz]) /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
		}
		else{
			ps[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
			psh[0 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 0)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE1, 0)));
		}
		if (zs == 1){
			ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + (iz + REF_3)]) /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.5*(receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] + receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + (iz + REF_3)]) /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
		}
		else{
			ps[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
			psh[2 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(NPR + 2)*isize2*zsize2*(j2 - j1) + ij*isize2*zsize2 + ii*zsize2 + iz] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + j + N2G]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE3, 0)));
		}

		//ps[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
		//ps[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
		//ps[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
		//psh[0*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+0)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
		//psh[1*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+1)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
		//psh[2*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] = receive[(NPR+2)*isize2*zsize2*(j2-j1) + ij*isize2*zsize2 + ii*zsize2+iz];
	#endif
	}
}

__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec)
{
	int z, ii, ij, iz, is, js, zs, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg, dq1, dq2, dq3;

	#pragma unroll NG
	for (z = z1; z <z2; z++){
		if (z1 < 0 && REF_3 == 1) iz = REF_3;
		else if (REF_3 == 1) iz = 0;
		else iz = z - z1;
		ii = (icurr - i1 - N1G - (icurr - i1 - N1G) % (1 + REF_1)) / (1 + REF_1) + REF_1;
		ij = (jcurr - j1 - N2G - (jcurr - j1 - N2G) % (1 + REF_2)) / (1 + REF_2) + REF_2;
		is = (((icurr - i1 - N1G) % (1 + REF_1) == 0) ? (-1) : (1));
		js = (((jcurr - j1 - N2G) % (1 + REF_2) == 0) ? (-1) : (1));
		zs = ((z == z1) ? (-1) : (1));

		for (k = 0; k < NPR; k++){
			if (nstep%timelevel_rec == timelevel_rec - 1 || nstep == -1){
				temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];
				temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];
			}

			if (nstep == -1 && timelevel_rec>timelevel)temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij];

			if (nstep != -1 && timelevel_rec>timelevel){
				receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] = temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] +
					(double)((nstep + 1) % timelevel_rec) / ((double)timelevel_rec)*(temp1receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] - temp2receive[(k)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij]);
			}
		}
		for (k = 0; k < NPR; k++){
			avg = 0.5*(prim[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (N3G + (1 - iz)*(BS_3 - 1))] + prim[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (iz + N3G + (1 - iz)*(BS_3 - 1))]);
			if (iz == 0){
				dq3 = slope_lim(avg, receive[k*isize2*jsize2*(z2 - z1) + 0 * isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + REF_3*isize2*jsize2 + ii*jsize2 + ij]);
			}
			else{
				dq3 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + 0 * isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + REF_3*isize2*jsize2 + ii*jsize2 + ij], avg);
			}
			//if(icurr-N1G-i1<(i2-i1)/2)dq1=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + (ii+REF_1)*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//else dq1=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + (ii-REF_1)*jsize2+ij];
			//if(jcurr-N2G-j1<(j2-j1)/2)dq2=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+(ij+REF_2)]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
			//else dq2=receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij]-receive[k*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+(ij-REF_2)];
			dq1 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii - REF_1)*jsize2 + ij]);
			dq2 = slope_lim(receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii)*jsize2 + (ij + REF_2)], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij], receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii)*jsize2 + (ij - REF_2)]);
			//dq1=dq2=0.;
			p[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
			ph[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[k*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + 0.25*(double)(is)*dq1 + 0.25*(double)(js)*dq2 + 0.25*(double)(zs)*dq3;
		}

		#if(STAGGERED)
		if (is == 1){
			ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij]) /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + (ii + REF_1)*jsize2 + ij]) /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
		}
		else{
			ps[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
			psh[0 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 0)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
				gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE1, 0)));
		}
		if (js == 1){
			ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + (ij + REF_2)]) /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = 0.5*(receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] + receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + (ij + REF_2)]) /
				gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
		}
		else{
			ps[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
			psh[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(NPR + 1)*isize2*jsize2*(z2 - z1) + iz*isize2*jsize2 + ii*jsize2 + ij] /
				gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]; //readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE2, 0)));
		}
		//ps[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+0)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
		//psh[0*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+0)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
		//ps[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+1)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
		//psh[1*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+1)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
		//ps[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+2)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
		//psh[2*(ksize)+icurr*isize+jcurr*(BS_3+2*N3G)+(z+N3G)] = receive[(NPR+2)*isize2*jsize2*(z2-z1) + iz*isize2*jsize2 + ii*jsize2+ij];
#endif
	}
}

__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
}

__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (j = j1; j <j2; j++){
				send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (j = j1; j <j2; j++){
				send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
	}
}

__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
		}
	}
}

__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (k = 0; k<NPR; k++){
				for (i = i1; i <i2; i++){
					temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
		}
		if (calc_corr == 2){
			for (k = 0; k<NPR; k++){
				for (i = i1; i <i2; i++){
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
		if (calc_corr == 3){
			for (k = 0; k<NPR; k++){
				for (i = i1; i <i2; i++){
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
		if (calc_corr == 5){
			for (k = 0; k<NPR; k++){
				for (i = i1; i <i2; i++){
					temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						+= receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						= factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						+= factor*pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}

		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
		}
		else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]
						+= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]
						-= temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		else if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (k = 0; k<NPR; k++){
				for (i = i1; i < i2; i++){
					temp1[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]
						+= (receive[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
		}
	}
}

__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
		}
		if (calc_corr == 2){
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
		if (calc_corr == 3){
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				}
			}
		}
		if (calc_corr == 5){
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						+= receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
				}
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						= factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						+= factor*pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
				}
			}
		}

		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
		}
		else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						+= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					pv[k*(ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
						-= temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (k = 0; k<NPR; k++){
				for (j = j1; j <j2; j++){
					temp1[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]
						+= (receive[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
				}
			}
		}
	}
}

__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
			}
		}
		if (calc_corr == 2){
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
		}
		if (calc_corr == 3){
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
				}
			}
		}
		if (calc_corr == 5){
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						+= receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
				}
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						= factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						+= factor*pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
				}
			}
		}

		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
			}
		}
		else if (calc_corr == 2 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]
						+= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		else if (calc_corr == 3 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					pv[k*(ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)]
						-= temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor; //times dt_old/dt_new to add in future code
				}
			}
		}
		else if (calc_corr == 5 && nstep2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
			for (k = 0; k<NPR; k++){
				for (z = z1; z <z2; z++){
					temp1[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]
						+= (receive[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
				}
			}
		}
	}
}

__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*factor*(
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (i = i1; i <i2; i++){
				send[k*jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += 0.25*factor*(
					pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] +
					pv[k*(ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
}

__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (j = j1; j <j2; j++){
				send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*factor*(
					pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
					+ pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (j = j1; j <j2; j++){
				send[k*isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += 0.25*factor*(
					pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[k*(ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] + pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]
					+ pv[k*(ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
			}
		}
	}
}

__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = 0.25*factor*(
					pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
					+ pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
			}
		}
	}
	else{
		for (k = 0; k < NPR; k++){
			#pragma unroll NG
			for (z = z1; z <z2; z++){
				send[k*isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += 0.25*factor*(
					pv[k*(ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] + pv[k*(ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G]
					+ pv[k*(ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
			}
		}
	}
}

__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1)) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1))) / ((z2 - z1)) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		//k=2
		for (i = i1; i <i2; i++){
			send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		//k=3
		for (i = i1; i <i2; i++){
			send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
	}
	else{
		//k=2
		for (i = i1; i <i2; i++){
			send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		//k=3
		for (i = i1; i <i2; i++){
			send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
	}
}

__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1)) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1))) / ((z2 - z1)) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		//k=1;
		for (j = j1; j <j2; j++){
			send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
		//k=3;
		for (j = j1; j <j2; j++){
			send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
	}
	else{
		//k=1;
		for (j = j1; j <j2; j++){
			send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
		//k=3;
		for (j = j1; j <j2; j++){
			send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
		}
	}
}

__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1)) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1))) / ((j2 - j1)) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (first_timestep == 1){
		//k=1;
		for (z = z1; z <z2; z++){
			send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
		}
		//k=2;
		for (z = z1; z <z2; z++){
			send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
		}
	}
	else{
		//k=1;
		for (z = z1; z <z2; z++){
			send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
		}
		//k=2;
		for (z = z1; z <z2; z++){
			send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G];
		}
	}
}

__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg;

	if (mode == 0){
		if (first_timestep == 1){
			//k=2
			for (i = i1; i <i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
				//pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
				//pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
			}
			//k=3
			for (i = i1; i <i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				//pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=0.5*(pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(i+N1G)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+zcurr]);
				//pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]+pv[3*(ksize)+(i+N1G)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]);
			}
		}
		else{
			//k=2
			for (i = i1; i <i2; i++){
				send[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
				//pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
				//pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
			}
			//k=3
			for (i = i1; i <i2; i++){
				send[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				//pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=0.5*(pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(i+N1G)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+zcurr]);
				//pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]+pv[3*(ksize)+(i+N1G)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]);
			}
		}
	}
	else{
		/*//k=2
		for (i = i1; i <i2; i++){
		avg=0.5*(pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]);
		pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
		pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=avg;
		avg=0.5*(pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]+pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[2*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		pv[2*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		}
		//k=3
		for (i = i1; i <i2; i++){
		avg=0.5*(pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
		pv[3*(ksize)+(i+N1G)*isize+jcurr*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		avg=0.5*(pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=avg;
		pv[3*(ksize)+(i+N1G)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		}*/
	}
}

__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg;

	if (mode == 0){
		if (first_timestep == 1){
			//k=1;
			for (j = j1; j <j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
				//pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
				//pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
			}
			//k=3;
			for (j = j1; j <j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				//pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=0.5*(pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr+2*REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]);
				//pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]+pv[3*(ksize)+(icurr+2*REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]);
			}
		}
		else{
			//k=1;
			for (j = j1; j <j2; j++){
				send[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr]);
				//pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
				//pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+2*REF_3)]);
			}
			//k=3;
			for (j = j1; j <j2; j++){
				send[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] += factor*0.5*(
					pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3]);
				//pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=0.5*(pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr+2*REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]);
				//pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=0.5*(pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr+2*REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]);
			}
		}
	}
	else{
		//k=1;
		/*for (j = j1; j <j2; j++){
		avg=0.5*(pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]);
		pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=avg;
		pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=avg;
		avg=0.5*(pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]+pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[1*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		pv[1*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		}
		//k=3;
		for (j = j1; j <j2; j++){
		avg=0.5*(pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=avg;
		pv[3*(ksize)+(icurr)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		avg=0.5*(pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]);
		pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+zcurr]=avg;
		pv[3*(ksize)+(icurr+REF_1)*isize+(j+N2G)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
		}*/
	}
}

__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg;

	if (mode == 0){
		if (first_timestep == 1){
			//k=1;
			for (z = z1; z <z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
				//pv[1*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[1*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+z+N3G]);
				//pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+z+N3G]);
			}
			//k=2;
			for (z = z1; z <z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
				//pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[2*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr+2*REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]);
				//pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr+2*REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]);
			}
		}
		else{
			//k=1;
			for (z = z1; z <z2; z++){
				send[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += factor*0.5*(
					pv[1 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[1 * (ksize)+(icurr + REF_1)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G]);
				//pv[1*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[1*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+z+N3G]);
				//pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+2*REF_2)*(BS_3+2*N3G)+z+N3G]);
			}
			//k=2;
			for (z = z1; z <z2; z++){
				send[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] += factor*0.5*(
					pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G]);
				//pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[2*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr+2*REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]);
				//pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=0.5*(pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr+2*REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]);
			}
		}
	}
	else{
		//k=1
		/*for (z = z1; z <z2; z++){
		avg=0.5*(pv[1*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]);
		pv[1*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=avg;
		pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=avg;
		avg=0.5*(pv[1*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]+pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]);
		pv[1*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=avg;
		pv[1*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=avg;
		}
		//k=2
		for (z = z1; z <z2; z++){
		avg=0.5*(pv[2*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]);
		pv[2*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=avg;
		pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=avg;
		avg=0.5*(pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]+pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]);
		pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr)*(BS_3+2*N3G)+z+N3G]=avg;
		pv[2*(ksize)+(icurr+REF_1)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+z+N3G]=avg;
		}*/
	}
}

__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2)
{
	int i, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int z22 = z2 + D3;
	int zcurr = global_id % (z22 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z22 - z1)) / (z22 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
		}
		else if (calc_corr == 2){
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 3){
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G)prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
			}
		}
		else if (calc_corr == 5){
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = (receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = (receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
			}
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) prim[3 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (i = i1; i < i2; i++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += (receive[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
			}
			for (i = i1; i < i2; i++){
				if (jcurr >= j1 + N2G + d1*D2 && jcurr<j2 + N2G + d2*D2 && zcurr >= z1 + N3G && zcurr<z2 + N3G) temp1[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] += (receive[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * jsize2*zsize2*(i2 - i1) + (i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)]);
			}
		}
	}
}


__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2)
{
	int j, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int z22 = z2 + D3;
	int zcurr = global_id % (z22 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z22 - z1)) / (z22 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
		}
		else if (calc_corr == 2){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 3){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= (temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= (temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]) / factor;
			}
		}
		else if (calc_corr == 4){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G && zcurr<z2 + N3G && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
				if (zcurr >= z1 + N3G && zcurr<z2 + N3G && icurr >= i1 + N1G && icurr<i2 + N1G) prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.;//0.5*(prim[3*(ksize)+icurr*isize+(j+N2G)*(BS_3+2*N3G)+zcurr] - receive[1*isize2*zsize2*(j2-j1) + (j - j1)*isize2*zsize2 + (icurr-i1-N1G)*zsize2+(zcurr-z1-N3G)] / factor);
			}
		}
		else if (calc_corr == 5){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * factor;
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = factor*prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += factor*prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
			}
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = (receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = (receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
			}
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] += temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  prim[3 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] -= temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
			}
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (j = j1; j <j2; j++){
				if (zcurr >= z1 + N3G + e1*D3 && zcurr<z2 + N3G + e2*D3 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += (receive[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[0 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
			}
			for (j = j1; j <j2; j++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && zcurr >= z1 + N3G && zcurr<z2 + N3G)  temp1[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] += (receive[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] - temp2[1 * isize2*zsize2*(j2 - j1) + (j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)]);
			}
		}
	}
}

__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2)
{
	int z, k;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int j22 = j2 + 1;
	int jcurr = global_id % (j22 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j22 - j1)) / (j22 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
			}
		}
		else if (calc_corr == 2){
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
		}
		else if (calc_corr == 3){
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= (temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= (temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]) / factor;
			}
		}
		else if (calc_corr == 5){
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] * factor;
			}
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = factor*prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += factor*prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
			}
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = (receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = (receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
			}
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] += temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) prim[1 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) prim[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] -= temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
			}
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
			for (z = z1; z <z2; z++){
				if (jcurr >= j1 + N2G + e1*D2 && jcurr<j2 + N2G + e2*D2 && icurr >= i1 + N1G && icurr<i2 + N1G) temp1[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += (receive[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[0 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
			}
			for (z = z1; z <z2; z++){
				if (icurr >= i1 + N1G + d1*D1 && icurr<i2 + N1G + d2*D1 && jcurr >= j1 + N2G && jcurr<j2 + N2G) temp1[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] += (receive[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] - temp2[1 * isize2*jsize2*(z2 - z1) + (z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)]);
			}
		}
	}
}

__global__ void packsendE1corn(int i1, int i2, int j, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = global_id + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	else send[global_id] += factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
}

__global__ void packsendE2corn(int i, int j1, int j2, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = global_id + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	else send[global_id] += factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
}

__global__ void packsendE3corn(int i, int j, int z1, int z2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	else send[global_id] += factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
}

__global__ void packsendE1corncourse(int i1, int i2, int j, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = global_id*(1 + REF_1) + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	else send[global_id] += 0.5*factor*(pv[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[1 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr]);
	//double avg=0.5*(pv[1*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[1*(ksize)+(icurr+REF_1)*isize+jcurr*(BS_3+2*N3G)+zcurr]);
	//pv[1*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
	//pv[1*(ksize)+(icurr+REF_1)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
}

__global__ void packsendE2corncourse(int i, int j1, int j2, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = global_id*(1 + REF_2) + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
	else send[global_id] += 0.5*factor*(pv[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr]);
	//double avg=0.5*(pv[2*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]);
	//pv[2*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
	//pv[2*(ksize)+(icurr)*isize+(jcurr+REF_2)*(BS_3+2*N3G)+zcurr]=avg;
}

__global__ void packsendE3corncourse(int i, int j, int z1, int z2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id*(1 + REF_3) + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (first_timestep == 1) send[global_id] = 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)]);
	else send[global_id] += 0.5*factor*(pv[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] + pv[3 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + (zcurr + REF_3)]);
	//double avg=0.5*(pv[3*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]+pv[3*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+(zcurr+REF_3)]);
	//pv[3*(ksize)+(icurr)*isize+jcurr*(BS_3+2*N3G)+zcurr]=avg;
	//pv[3*(ksize)+(icurr)*isize+(jcurr)*(BS_3+2*N3G)+(zcurr+REF_3)]=avg;
}

__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor){
	int global_id = get_global_id(0);
	int icurr = global_id + i1 + N1G;
	int jcurr = j + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			temp1[global_id] = receive[global_id] - prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
		else if (calc_corr == 2){
			prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
		}
		else if (calc_corr == 3){
			prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
		}
		else if (calc_corr == 5){
			temp1[global_id] += receive[global_id] - prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			temp2[global_id] = factor*prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			temp2[global_id] += factor*prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			temp1[global_id] = (receive[global_id] - temp2[global_id]);
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			prim[1 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
			temp1[global_id] += (receive[global_id] - temp2[global_id]);
		}
	}
}

__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = global_id + j1 + N2G;
	int zcurr = z + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			temp1[global_id] = receive[global_id] - prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
		else if (calc_corr == 2){
			prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
		}
		else if (calc_corr == 3){
			prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
		}
		else if (calc_corr == 5){
			temp1[global_id] += receive[global_id] - prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			temp2[global_id] = factor*prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			temp2[global_id] += factor*prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			temp1[global_id] = (receive[global_id] - temp2[global_id]);
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT* timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			prim[2 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
			temp1[global_id] += (receive[global_id] - temp2[global_id]);
		}
	}
}

__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor){
	int global_id = get_global_id(0);
	int icurr = i + N1G;
	int jcurr = j + N2G;
	int zcurr = global_id + z1 + N3G;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (timelevel_rec <= timelevel){
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){
			temp1[global_id] = receive[global_id] - prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
		else if (calc_corr == 2){
			prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor;
		}
		else if (calc_corr == 3){
			prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= (temp1[global_id]) / factor;
		}
		else if (calc_corr == 5){
			temp1[global_id] += receive[global_id] - prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * factor;
		}
	}
	else{
		if ((calc_corr == 1 || calc_corr == 5) && nstep_2 % (2 * timelevel_rec) == 2 * timelevel - 1){ //store used flux in present timestep to calculate later correction
			temp2[global_id] = factor*prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		else if ((calc_corr == 1 || calc_corr == 5)){
			temp2[global_id] += factor*prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
		}
		if (calc_corr == 1 && nstep % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //receive 'correct'flux from more refined AMR block and insert correction wrt flux from previous step 'temp2' into 'temp1'
			temp1[global_id] = (receive[global_id] - temp2[global_id]);
		}
		else if (calc_corr == 2 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //add correction to fluxes before applying fluxes to conserved quantities
			prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] += temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 3 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1 && (nstep_2 % (2 * AMR_SWITCHTIMELEVEL) != 2 * NT * timelevel_rec - 1)){ //remove corrections to fluxes after applyting fluxes to conserved quantities
			prim[3 * (ksize)+(icurr)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] -= temp1[global_id] / factor; //times dt_old/dt_new to add in future code
		}
		else if (calc_corr == 5 && nstep_2 % (2 * timelevel_rec) == 2 * timelevel_rec - 1){ //add correction to fluxes before applying fluxes to conserved quantities
			temp1[global_id] += (receive[global_id] - temp2[global_id]);
		}
	}
}

__global__ void packsendB1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send)
{
	int i;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (i = i1; i <i2; i++){
		send[(i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] = pv[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr];
	}
}

__global__ void packsendB2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send)
{
	int j;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (j = j1; j <j2; j++){
		send[(j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] = pv[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr];
	}
}

__global__ void packsendB3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send)
{
	int z;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (z = z1; z <z2; z++){
		send[(z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] = pv[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)];
	}
}

__global__ void packsendBaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int i;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int jcurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_2) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (i = i1; i <i2; i++){
		send[(i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G) / (1 + REF_2)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*(
			pv[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE1, 0))) +
		pv[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr + REF_3] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + jcurr];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE1, 0))) +
		pv[0 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G, FACE1, 0))) +
		pv[0 * (ksize)+(i + N1G)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + zcurr + REF_3] * gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i + N1G)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, i + N1G, FACE1, 0))));
	}
}

__global__ void packsendBaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int j;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % ((z2 - z1) / (1 + REF_3))*(1 + REF_3) + z1 + N3G;
	int icurr = (global_id - global_id % ((z2 - z1) / (1 + REF_3))) / ((z2 - z1) / (1 + REF_3))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (j = j1; j <j2; j++){
		send[(j - j1)*isize2*zsize2 + (icurr - i1 - N1G) / (1 + REF_1)*zsize2 + (zcurr - z1 - N3G) / (1 + REF_3)] = 0.25*(
			pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (j+N2G)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0))) +
		pv[1 * (ksize)+(icurr)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (j + N2G)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0))) +
		pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr + REF_1, FACE2, 0))) +
		pv[1 * (ksize)+(icurr + REF_1)*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr + REF_3] * gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (j + N2G)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr + REF_1, FACE2, 0))));
	}
}

__global__ void packsendBaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU)
{
	int z;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % ((j2 - j1) / (1 + REF_2))*(1 + REF_2) + j1 + N2G;
	int icurr = (global_id - global_id % ((j2 - j1) / (1 + REF_2))) / ((j2 - j1) / (1 + REF_2))*(1 + REF_1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	for (z = z1; z <z2; z++){
		send[(z - z1)*isize2*jsize2 + (icurr - i1 - N1G) / (1 + REF_1)*jsize2 + (jcurr - j1 - N2G) / (1 + REF_2)] = 0.25*(
			pv[2 * (ksize)+(icurr)*isize + (jcurr)*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0))) +
		pv[2 * (ksize)+(icurr)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr, FACE3, 0))) +
		pv[2 * (ksize)+(icurr + REF_1)*isize + jcurr*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr + REF_1, FACE3, 0))) +
		pv[2 * (ksize)+(icurr + REF_1)*isize + (jcurr + REF_2)*(BS_3 + 2 * N3G) + z + N3G] * gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+REF_1)*(BS_2 + 2 * N2G) + (jcurr+REF_2)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr + REF_2, icurr + REF_1, FACE3, 0))));
	}
}

__global__ void unpackreceiveB1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, int div, __global FTYPE2 *  pv, __global FTYPE2 *  receive, const  FTYPE2* __restrict__ gdet_GPU)
{
	int i;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int jcurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + j1 + N2G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;
	for (i = i1; i <i2; i++){
		if (div == 1) factor = gdet_GPU[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (i+N1G)*(BS_2 + 2 * N2G) + (jcurr)];// readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, i + N1G, FACE1, 0)));
		pv[0 * (ksize)+(i + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = receive[(i - i1)*jsize2*zsize2 + (jcurr - j1 - N2G)*zsize2 + (zcurr - z1 - N3G)] / factor;
	}
}

__global__ void unpackreceiveB2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, int div, __global FTYPE2 *  pv, __global FTYPE2 *  receive, const  FTYPE2* __restrict__ gdet_GPU, int neg)
{
	int j;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (z2 - z1) + z1 + N3G;
	int icurr = (global_id - global_id % (z2 - z1)) / (z2 - z1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;

	for (j = j1; j <j2; j++){
		if (div == 1) factor = gdet_GPU[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (j+N2G)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0)));
		if (neg == 1) factor = -1.;
		if (div == 1 && neg == 1) factor = -readImageDouble(read_imagei(gdet_GPU, sample, (int4)(j + N2G, icurr, FACE2, 0)));
		pv[1 * (ksize)+icurr*isize + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = receive[(j - j1)*isize2*zsize2 + (icurr - i1 - N1G)*zsize2 + (zcurr - z1 - N3G)] / factor;
	}
}

__global__ void unpackreceiveB3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, int div, __global FTYPE2 *  pv, __global FTYPE2 *  receive, const  FTYPE2* __restrict__ gdet_GPU)
{
	int z;
	int global_id = get_global_id(0);
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int jcurr = global_id % (j2 - j1) + j1 + N2G;
	int icurr = (global_id - global_id % (j2 - j1)) / (j2 - j1) + i1 + N1G;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor = 1.;

	for (z = z1; z <z2; z++){
		if (div == 1) factor = gdet_GPU[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr)*(BS_2 + 2 * N2G) + (jcurr)];//readImageDouble(read_imagei(gdet_GPU, sample, (int4)(jcurr, icurr, FACE3, 0)));
		pv[2 * (ksize)+icurr*isize + jcurr*(BS_3 + 2 * N3G) + (z + N3G)] = receive[(z - z1)*isize2*jsize2 + (icurr - i1 - N1G)*jsize2 + (jcurr - j1 - N2G)] / factor;
	}
}

void GPU_step_ch()
{
#if (1)
	double ndt, inmsg;
	int i, j, z, k, n, uu;

	if (rank == 0){
		//fprintf(stderr, "h");
	}
	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
	}
	for (uu = 0; uu < 2 * AMR_MAXTIMELEVEL; uu++){
		set_prestep();
		ndt = advance_GPU();   /* time step primitive variables to the half step */
		for (n = 0; n < n_active; n++){
			if (prestep_full[n_ord[n]] == 1)  GPU_fixup(1, n_ord[n]);
			else if (prestep_half[n_ord[n]] == 1) GPU_fixup(0, n_ord[n]);
		}
		GPU_boundprim(0);    /* Set boundary conditions for primitive variables, flag bad ghost zones */

		nstep++;
		#if(PRESTEP)
		for (n = 0; n < n_active; n++){
			if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == 0 && (block[n_ord[n]][AMR_PRESTEP] != 0))block[n_ord[n]][AMR_PRESTEP] = 0;
			else if (block[n_ord[n]][AMR_PRESTEP] == 1)block[n_ord[n]][AMR_PRESTEP] = 2;
		}
		#endif
	}

	/* Repeat and rinse for the full time (aka corrector) step:  */
	if (rank == 0){
		//fprintf(stderr, "f");
	}

	/* Determine next time increment based on current characteristic speeds: */
	if (dt < 1.e-9) {
		fprintf(stderr, "timestep too small\n");
		exit(11);
	}

	/* increment time */
	t += (double)(AMR_MAXTIMELEVEL)*dt;

	/* set next timestep */
	if (ndt > SAFE*dt) ndt = SAFE*dt;
	dt = ndt;

	/*Calculate smallest timestep for all MPI threads*/

		#if (MPI_enable)
	MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi_cartcomm);
		#endif

	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0) set_timelevel();

	#if(TIMESTEP_JET)
	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 0)set_timelevel_jet();
	#endif

	if (t + dt > tf) dt = tf - t;  /* but don't step beyond end of run */
	#endif
}

void set_prestep(void){
	int n;

#if(PRESTEP)
	int timelevel_min = AMR_MAXTIMELEVEL;
	int blocks_per_timestep = 0;
	int blocks_this_timestep = 0;

	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	//Find the minimum timelevel on this node
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_TIMELEVEL] < timelevel_min) timelevel_min = block[n_ord[n]][AMR_TIMELEVEL];
	}

	//Calculate the number of blocks you want to evolve simultaneously
	blocks_per_timestep = (count_node[0] - count_node[0] % (AMR_MAXTIMELEVEL / timelevel_min)) / (AMR_MAXTIMELEVEL / timelevel_min);
	for (n = 0; n < n_active; n++){
		if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep++;
	}

	//If you don't have sufficient blocks this timestep preevolve some blocks if available
	if ((nstep % timelevel_min) == timelevel_min - 1){
		for (n = 0; n < n_active; n++){
			if (blocks_this_timestep < blocks_per_timestep && nstep % (block[n_ord[n]][AMR_TIMELEVEL]) != block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0 && (block[n_ord[n]][AMR_POLE] == 0)){
				block[n_ord[n]][AMR_PRESTEP] = 1;
				block[n_ord[n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) - (block[n_ord[n]][AMR_TIMELEVEL] - 1));
				blocks_this_timestep++;
			}
		}
	}

	#if(N_GPU>1)
	for (gpu = 0; gpu < N_GPU; gpu++){
		timelevel_min = AMR_MAXTIMELEVEL;
		blocks_per_timestep = 0;
		blocks_this_timestep = 0;
		//Find the minimum timelevel on this gpu
		for (n = 0; n < n_active_gpu[gpu]; n++){
			if (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] < timelevel_min) timelevel_min = block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL];
		}

		//Calculate the number of blocks you want to evolve simultaneously
		blocks_per_timestep = (count_gpu[gpu] - count_gpu[gpu] % (AMR_MAXTIMELEVEL / timelevel_min)) / (AMR_MAXTIMELEVEL / timelevel_min);
		for (n = 0; n < n_active_gpu[gpu]; n++){
			if (nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) == block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1 || block[n_ord[n]][AMR_PRESTEP] == 1 || block[n_ord[n]][AMR_PRESTEP] == 0)blocks_this_timestep++;
		}

		//If you don't have sufficient blocks this timestep preevolve some blocks if available
		if ((nstep % timelevel_min) == timelevel_min - 1){
			for (n = 0; n < n_active_gpu[gpu]; n++){
				if (blocks_this_timestep < blocks_per_timestep && nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) != block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1
					&& (block[n_ord_gpu[gpu][n]][AMR_PRESTEP] == 0) && (block[n_ord_gpu[gpu][n]][AMR_POLE] == 0)){
					block[n_ord_gpu[gpu][n]][AMR_PRESTEP] = 1;
					block[n_ord_gpu[gpu][n]][AMR_NSTEP] = nstep - (nstep % (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL]) - (block[n_ord_gpu[gpu][n]][AMR_TIMELEVEL] - 1));
					blocks_this_timestep++;
				}
			}
		}
	}
	#endif
	//If at end of switchtimelevel do not pre-evolve
	for (n = 0; n < n_active; n++){
		if (block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) >= 2 * AMR_SWITCHTIMELEVEL - 2 * AMR_MAXTIMELEVEL){
			block[n_ord[n]][AMR_PRESTEP] = 0;
			block[n_ord[n]][AMR_NSTEP] = nstep;
		}
	}
	#else
	for (n = 0; n < n_active; n++){
		block[n_ord[n]][AMR_PRESTEP] = 0;
		block[n_ord[n]][AMR_NSTEP] = nstep;
	}
	#endif
}

double advance_GPU(void)
{
	int i, n;
	gpu = 1;


	if (nstep % (2 * AMR_MAXTIMELEVEL) == 0){
		ndt1 = ndt2 = ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[n_ord[n]][0] = bdt[n_ord[n]][1] = bdt[n_ord[n]][2] = bdt[n_ord[n]][3] = 1e9;
		}
	}
	for (n = 0; n < n_active; n++){
		prestep_half[n_ord[n]] = (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) < block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
		prestep_full[n_ord[n]] = (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 0)
			|| (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) < 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) >  block[n_ord[n]][AMR_TIMELEVEL] - 1 && block[n_ord[n]][AMR_PRESTEP] == 1);
	}

	#if(N1G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(1, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(1, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][1] = fluxcalc_GPU(n_ord[n]);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt1 = 1e9;
		for (n = 0; n < n_active; n++){
			if (block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1]);
			}
			else{
				ndt1 = MY_MIN(ndt1, bdt[n_ord[n]][1] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}

	#else
	ndt1 = 1e9;
	#endif
	#if(N2G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(2, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(2, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][2] = fluxcalc_GPU(n_ord[n]);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt2 = 1e9;
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2]);
			}
			else{
				ndt2 = MY_MIN(ndt2, bdt[n_ord[n]][2] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}

	#else
	ndt2 = 1e9;
	#endif
	#if(N3G>0)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_fluxcalc2D(3, 1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_fluxcalc2D(3, 0, n_ord[n]);
	}

	read_time_GPU();
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1) bdt[n_ord[n]][3] = fluxcalc_GPU(n_ord[n]);
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt3 = 1e9;
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3]);
			}
			else{
				ndt3 = MY_MIN(ndt3, bdt[n_ord[n]][3] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}
	#else
	ndt3 = 1e9;
	#endif

	gpu = 1;
	rc = 0;

	//MPI communication
	/*for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
			if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
			break;
		}
	}*/

	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		flux_send1(F1, BufferF1_1, n_ord[n]);
		flux_send2(F2, BufferF2_1, n_ord[n]);
		#if(N3G>0)
		flux_send3(F3, BufferF3_1, n_ord[n]);
		#endif
	}

	#if(PRESTEP)
	//For last timestep synchronize electric fields immediately
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		flux_rec1(F1, BufferF1_1, n_ord[n], 5);
		flux_rec2(F2, BufferF2_1, n_ord[n], 5);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 5);
		#endif
	}

	//For first timestep do not synchronize electrice fields 
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		flux_rec1(F1, BufferF1_1, n_ord[n], 2);
		flux_rec2(F2, BufferF2_1, n_ord[n], 2);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 2);
		#endif
	}
	#else
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		flux_rec1(F1, BufferF1_1, n_ord[n], 1);
		flux_rec2(F2, BufferF2_1, n_ord[n], 1);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 1);
		#endif
	}

	//For first timestep do not synchronize electrice fields
	for (n = 0; n < n_active; n++)if ((nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)){ //
		flux_rec1(F1, BufferF1_1, n_ord[n], 2);
		flux_rec2(F2, BufferF2_1, n_ord[n], 2);
		#if(N3G>0)
		flux_rec3(F3, BufferF3_1, n_ord[n], 2);
		#endif
	}
	#endif
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomF \n");

	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++) if (prestep_full[n_ord[n]] == 1 || prestep_half[n_ord[n]] == 1) GPU_fix_flux(n_ord[n]);
	#endif
	#if(STAGGERED)
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport1(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport1(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport2(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport2(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}
	#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
	//MPI_Barrier(mpi_cartcomm);
	rc = 0;
	GPU_consttransport_bound();
	//MPI communication
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE \n");
	#endif
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_consttransport3(1, dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_consttransport3(0, 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL], n_ord[n]);
	}

	#else
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct1(n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_flux_ct2(n_ord[n]);
	#endif
	double timestep;
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) timestep = dt*(double)block[n_ord[n]][AMR_TIMELEVEL];
		else if (prestep_half[n_ord[n]] == 1)timestep = 0.5*dt*(double)block[n_ord[n]][AMR_TIMELEVEL];

		clSetKernelArg(kernel_fixup[n_ord[n]], 23, sizeof(FTYPE2), &timestep);

		if (prestep_full[n_ord[n]] == 1)clSetKernelArg(kernel_fixup[n_ord[n]], 6, sizeof(cl_mem), (void *)&Bufferps_1[n_ord[n]]);
		else if (prestep_half[n_ord[n]] == 1) clSetKernelArg(kernel_fixup[n_ord[n]], 6, sizeof(cl_mem), (void *)&Bufferpsh_1[n_ord[n]]);
	}

	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1) GPU_Utoprim(1, n_ord[n]);
		else if (prestep_half[n_ord[n]] == 1) GPU_Utoprim(0, n_ord[n]);
	}

	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1){
		ndt = 1e9;
		for (n = 0; n < n_active; n++){
			bdt[n_ord[n]][0] = 1. / (1. / bdt[n_ord[n]][1] + 1. / bdt[n_ord[n]][2] + 1. / bdt[n_ord[n]][3]);
			if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
				ndt = MY_MIN(ndt, bdt[n_ord[n]][0]);
			}
			else{
				ndt = MY_MIN(ndt, bdt[n_ord[n]][0] / ((double)block[n_ord[n]][AMR_TIMELEVEL]));
			}
		}
	}

	//ndt = defcon * 1. / (1. / ndt1 + 1. / ndt2 + 1. / ndt3);
	return defcon * ndt;
	return 0.;
}

/**/
void read_time_GPU(void){
	int n;
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1){
			clEnqueueReadBuffer(commandQueueGPU[n_ord[n]], Bufferdtij[n_ord[n]], CL_FALSE, (int)0 * sizeof(FTYPE2), nr_workgroups[n_ord[n]] * sizeof(FTYPE2), dtij_GPU[n_ord[n]], 0, NULL, NULL);
			clFlush(commandQueueGPU[n_ord[n]]);
		}
	}
}

double fluxcalc_GPU(int n)
{
	double ndt;
	int y;
	ndt = 1.e9;
	//status=clEnqueueReadBuffer(commandQueueGPU[n], Bufferdtij[n], CL_TRUE, (int)0 * sizeof(FTYPE2), nr_workgroups[n] * sizeof(FTYPE2), dtij_GPU[n], 0, NULL, NULL);
	clFinish(commandQueueGPU[n]);

	for (y = 0; y < nr_workgroups[n]; y++){
		if (dtij_GPU[n][y] < ndt && dtij_GPU[n][y] < 1.e9){
			ndt = dtij_GPU[n][y];
		}
	}

	return(ndt);
	return 0.;
}

void GPU_init(void)
{
	int c;

	/*Getting platforms and choose an available one*/
	platform = NULL;	//the chosen platform
	cl_platform_id* platforms;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS){
		fprintf(stderr, "\nError: Getting platforms! \n");
	}

	if (rank == 0) printf("numplatforms : %d \n", numPlatforms);

	/*For clarity, choose the first available platform. We use the AMD OpenCL SDK*/
	if (numPlatforms > 0){
		platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
		clGetPlatformIDs(numPlatforms, platforms, &status);
		platform = platforms[0];
	}

	/*Query the platform and choose the first GPU device if has one. Otherwise use the CPU as device*/
	numDevices = 0;
#if(!Intel_CL)
	for (c = 0; c < numPlatforms; c++){
		status = clGetDeviceIDs(platforms[c], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if (numDevices >= 1) break;
	}
#else
	for (c = 0; c < numPlatforms; c++){
		status = clGetDeviceIDs(&(platforms[c]), CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		if (numDevices >= 1) break;
	}
#endif

	if (numDevices == 0){ //no GPU available.
		fprintf(stderr, "No GPU device available.");
		fprintf(stderr, "Choosing CPU as default device.\n");
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else{
		status = clGetDeviceIDs(platforms[c], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platforms[c], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}

	/*Check that number of devices per node is put in correctly*/
	if (numDevices != N_GPU && rank == 0){
		printf("Number of devices per node incorrect! %d %d \n", numDevices, status);
	}

	/*Create context*/
	context[0] = clCreateContext(NULL, numDevices, &devices[0], NULL, NULL, &status);
	if (status != 0) printf("Error context: %d \n", status);

	/*Get size of kernel source*/
	programHandle = fopen("GPU_program1.cl", "r");
	fseek(programHandle, 0, SEEK_END);
	programSize = ftell(programHandle);
	rewind(programHandle);

	/*Read kernel source into buffer*/
	programBuffer = (char*)malloc(programSize + 1);
	programBuffer[programSize] = '\0';
	fread(programBuffer, sizeof(char), programSize, programHandle);
	fclose(programHandle);
	while (programBuffer[programSize - 1] != '}'){
		programSize--;
	}

	GPU_program[0] = clCreateProgramWithSource(context[0], 1, (const char**)&programBuffer, &programSize, NULL);
	//free(programBuffer);

	/*Get size of kernel source*/
	programHandle = fopen("GPU_program2.cl", "r");
	fseek(programHandle, 0, SEEK_END);
	programSize = ftell(programHandle);
	rewind(programHandle);

	/*Read kernel source into buffer*/
	programBuffer = (char*)malloc(programSize + 1);
	programBuffer[programSize] = '\0';
	fread(programBuffer, sizeof(char), programSize, programHandle);
	fclose(programHandle);
	while (programBuffer[programSize - 1] != '}'){
		programSize--;
	}
	GPU_program[1] = clCreateProgramWithSource(context[0], 1, (const char**)&programBuffer, &programSize, NULL);
	//free(programBuffer);

	/*Build program*/
	status = clBuildProgram(GPU_program[0], numDevices, &devices[0], NULL, NULL, NULL);
	if (status != 0) printf("Error program 1: %d \n", status);

	status = clBuildProgram(GPU_program[1], numDevices, &devices[0], NULL, NULL, NULL);
	if (status != 0) printf("Error program 2: %d \n", status);
	int i, j;
	for (i = 0; i < numDevices; i++){
		for (j = 0; j < NQ; j++){
			commandQueue[i*NQ + j] = clCreateCommandQueue(context[0], devices[i], 0, &status);
		}
	}
}

void set_arrays_GPU(int n){
#if (1)
	int i, j, z;
	cl_int freeze_GPU;
	cl_int N1_tot, N2_tot, N3_tot;

	/*N1 and N2 values to be exported to GPU memory*/
	int offset = 0;

	/*Set the global work size and make sure that it is a multiple of the group size. The Nvidia OpenCL framework crashes otherwise!*/
	fix_mem[n] = LOCAL_WORK_SIZE - ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)) % LOCAL_WORK_SIZE;
	fix_mem2[n] = LOCAL_WORK_SIZE - ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)) % LOCAL_WORK_SIZE;
	global_work_offset[n][0] = ibound*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G); //alert
	global_work_size_special[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) + (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) +
		(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G);
	global_work_size_special1[n][0] = (LOCAL_WORK_SIZE - ((N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) * NG;
	global_work_size_special2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) * NG;
	global_work_size_special3[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG;

	local_work_size[0] = LOCAL_WORK_SIZE;
	global_work_size1[n][0] = (LOCAL_WORK_SIZE - (N1_GPU[n] * N2_GPU[n] * N3_GPU[n]) % LOCAL_WORK_SIZE) + (N1_GPU[n] * N2_GPU[n] * N3_GPU[n]); //utoprim
	global_work_size2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3); //fluxcalc
	global_work_size3[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3); //flux_ct
	nr_workgroups[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE) + 1;

	/*Allocate memory to 1D arrays*/
	p_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	dq_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2)); //array to store temporary data
	#if(STAGGERED)
	ps_1[n] = (FTYPE2(*))calloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	psh_1[n] = (FTYPE2(*))calloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	#endif
	//#if(GPU_DEBUG)
	ph_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	//#endif
	dU_GPU[n] = (FTYPE2(*))calloc(NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)), sizeof(FTYPE2));
	pbound_1[n] = (FTYPE2(*))calloc(N_POINTS*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n]), sizeof(FTYPE2));
	#if(!NONSYMMETRIC)
	gcov_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	gcon_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	conn_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM, sizeof(FTYPE2));
	gdet_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG, sizeof(FTYPE2));
	#else
	gcov_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	gcon_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	conn_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NDIM*NDIM*NDIM, sizeof(FTYPE2));
	gdet_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG, sizeof(FTYPE2));
	#endif
	pflag_GPU[n] = (cl_int(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(cl_int));
	failimage_GPU[n] = (cl_int(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL, sizeof(cl_int));
	Katm_GPU[n] = (FTYPE2(*))calloc((N1_GPU[n] + 2 * N1G), sizeof(FTYPE2));

	N1_tot = N1;
	N2_tot = N2;
	N3_tot = N3;

	if (boundfreeze1 == 1 || boundfreeze2 == 1){
		freeze_GPU = N_POINTS;
	}
	else{
		freeze_GPU = 0; //alert
	}

	/*Creating command queue associate with the context*/
	commandQueueGPU[n] = commandQueue[0]; //clCreateCommandQueue(context[0], devices[0], 0, &status);

	/*Allocate memory to buffers on GPU*/
	cudaMalloc(BufferF1_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(BufferF2_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(BufferF3_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferdq_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(BufferE_1[n], NDIM*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#if(LEER)
	cudaMalloc(BufferV[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#endif
	cudaMalloc(Bufferradius[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(Bufferstorage1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferstorage2[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferstorage3[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferstorage4[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferp_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#if(STAGGERED)
	cudaMalloc(Bufferps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(Bufferpsh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#endif
	cudaMalloc(Bufferpbound_1[n], N_POINTS*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n])*sizeof(FTYPE2);
	cudaMalloc(Bufferdtij[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) / LOCAL_WORK_SIZE*sizeof(FTYPE2));
	cudaMalloc(Bufferpflag[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(cl_int));
	cudaMalloc(Bufferfailimage[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(cl_int));
	cudaMalloc(Bufferdiagflux[n], 3 * MY_MAX((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G))*sizeof(FTYPE2));
	cudaMalloc(BufferKatm[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(BufferdU[n], NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G))*sizeof(FTYPE2));

	cudaHostAlloc(&dtij_GPU[n], nr_workgroups[n] * sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&Bufferdtij[n], dtij_GPU[n], 0);

	cudaHostAlloc(&send1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&send2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&send3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&send4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif

	#if(N3G>0)
	cudaHostAlloc(&send5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send5_1[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send5_3[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send5_5[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send5_7[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&send6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&send6_2[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_4[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_6[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_8[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&receive1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive2_1[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive3_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive4_5[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&receive5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive5_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive6_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&tempreceive1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&tempreceive5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive5_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive5_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive5_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive5_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempreceive6_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive6_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive6_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive6_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&send1_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&send5_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive1_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&receive1_flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive1_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&send1_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&send5_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive1_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&send1_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send2_E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send3_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send4_E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&send5_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send6_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive1_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	cudaHostAlloc(&receive1_E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_E1[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_E1[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&receive1_3E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive1_3E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_4E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_7E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive1_8E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_1E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_2E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_3E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive2_4E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_1E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_2E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_5E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive3_6E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_5E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_6E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_7E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive4_8E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive5_1E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_3E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_5E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive5_7E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_2E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_4E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_6E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive6_8E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#endif
	#if(N3G>0)
	cudaHostAlloc(&send_E1_corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E1_corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E1_corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E1_corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E2_corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E2_corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E2_corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E2_corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&send_E3_corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E3_corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E3_corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&send_E3_corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive_E1_corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive_E3_corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostAlloc(&receive_E1_corn9_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn9_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn10_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn10_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn11_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn11_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn12_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn12_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn5_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn5_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn6_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn6_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn7_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn7_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn8_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn8_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive_E3_corn1_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn1_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn2_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn2_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn3_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn3_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn4_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn4_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&tempreceive_E1_corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive_E3_corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostAlloc(&tempreceive_E1_corn9_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn9_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn10_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn10_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn11_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn11_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn12_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E1_corn12_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn5_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn5_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn6_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn6_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn7_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn7_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn8_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E2_corn8_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&tempreceive_E3_corn1_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn1_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn2_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn2_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn3_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn3_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn4_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&tempreceive_E3_corn4_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#if(N3G>0)
	cudaHostAlloc(&receive_E1_corn9_12[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn9_22[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn10_12[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn10_22[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn11_12[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn11_22[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn12_12[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E1_corn12_22[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn5_12[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn5_22[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn6_12[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn6_22[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn7_12[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn7_22[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn8_12[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E2_corn8_22[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif
	cudaHostAlloc(&receive_E3_corn1_12[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn1_22[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn2_12[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn2_22[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn3_12[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn3_22[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn4_12[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	cudaHostAlloc(&receive_E3_corn4_22[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2), cudaHostAllocMapped);
	#endif

	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend1_3[n], send1_3[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_4[n], send1_4[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_7[n], send1_7[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_8[n], send1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend2[n], send2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend2_1[n], send2_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend2_2[n], send2_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend2_3[n], send2_3[n],0);
	cudaHostGetDevicePointer(&Buffersend2_4[n], send2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend3[n], send3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend3_1[n], send3_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_2[n], send3_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_5[n], send3_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_6[n], send3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend4[n], send4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend4_5[n], send4_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_6[n], send4_6[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_7[n], send4_7[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_8[n], send4_8[n], 0);
	#endif

	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5[n], send5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend5_1[n], send5_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_3[n], send5_3[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_5[n], send5_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_7[n], send5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend6[n], send6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend6_2[n], send6_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_4[n], send6_4[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_6[n], send6_6[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_8[n], send6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1[n], receive1[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3[n], receive1_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4[n], receive1_4[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7[n], receive1_7[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8[n], receive1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec2[n], receive2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec2_1[n], receive2_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2[n], receive2_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3[n], receive2_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4[n], receive2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec3[n], receive3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec3_1[n], receive3_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2[n], receive3_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5[n], receive3_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6[n], receive3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec4[n], receive4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec4_5[n], receive4_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6[n], receive4_6[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7[n], receive4_7[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8[n], receive4_8[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5[n], receive5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec5_1[n], receive5_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3[n], receive5_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5[n], receive5_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7[n], receive5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec6[n], receive6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec6_2[n], receive6_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4[n], receive6_4[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6[n], receive6_6[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8[n], receive6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&tempBufferrec1[n], tempreceive1[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec1_3[n], tempreceive1_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_4[n], tempreceive1_4[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_7[n], tempreceive1_7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_8[n], tempreceive1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec2[n], tempreceive2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec2_1[n], tempreceive2_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_2[n], tempreceive2_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_3[n], tempreceive2_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_4[n], tempreceive2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec3[n], tempreceive3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec3_1[n], tempreceive3_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_2[n], tempreceive3_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_5[n], tempreceive3_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_6[n], tempreceive3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec4[n], tempreceive4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec4_5[n], tempreceive4_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_6[n], tempreceive4_6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_7[n], tempreceive4_7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_8[n], tempreceive4_8[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrec5[n], tempreceive5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec5_1[n], tempreceive5_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_3[n], tempreceive5_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_5[n], tempreceive5_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_7[n], tempreceive5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec6[n], tempreceive6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec6_2[n], tempreceive6_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_4[n], tempreceive6_4[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_6[n], tempreceive6_6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_8[n], tempreceive6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1flux[n], send1_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend2flux[n], send2_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend3flux[n], send3_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend4flux[n], send4_flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5flux[n], send5_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend6flux[n], send6_flux[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1flux[n], receive1_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2flux[n], receive2_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3flux[n], receive3_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4flux[n], receive4_flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5flux[n], receive5_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6flux[n], receive6_flux[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3flux[n], receive1_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux[n], receive1_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux[n], receive1_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux[n], receive1_8flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux[n], receive2_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux[n], receive2_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux[n], receive2_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux[n], receive2_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux[n], receive3_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux[n], receive3_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux[n], receive3_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux[n], receive3_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux[n], receive4_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux[n], receive4_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux[n], receive4_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux[n], receive4_8flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux[n], receive5_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux[n], receive5_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux[n], receive5_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux[n], receive5_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux[n], receive6_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux[n], receive6_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux[n], receive6_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux[n], receive6_8flux[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1flux1[n], receive1_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2flux1[n], receive2_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3flux1[n], receive3_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4flux1[n], receive4_flux1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5flux1[n], receive5_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6flux1[n], receive6_flux1[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3flux1[n], receive1_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux1[n], receive1_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux1[n], receive1_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux1[n], receive1_8flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux1[n], receive2_1flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux1[n], receive2_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux1[n], receive2_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux1[n], receive2_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux1[n], receive3_1flux1[n],  0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux1[n], receive3_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux1[n], receive3_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux1[n], receive3_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux1[n], receive4_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux1[n], receive4_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux1[n], receive4_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux1[n], receive4_8flux1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux1[n], receive5_1flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux1[n], receive5_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux1[n], receive5_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux1[n], receive5_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux1[n], receive6_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux1[n], receive6_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux1[n], receive6_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux1[n], receive6_8flux1[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1_3flux2[n], receive1_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux2[n], receive1_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux2[n], receive1_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux2[n], receive1_8flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux2[n], receive2_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux2[n], receive2_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux2[n], receive2_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux2[n], receive2_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux2[n], receive3_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux2[n], receive3_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux2[n], receive3_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux2[n], receive3_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux2[n], receive4_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux2[n], receive4_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux2[n], receive4_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux2[n], receive4_8flux2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux2[n], receive5_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux2[n], receive5_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux2[n], receive5_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux2[n], receive5_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux2[n], receive6_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux2[n], receive6_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux2[n], receive6_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux2[n], receive6_8flux2[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1fine[n], send1_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend2fine[n], send2_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend3fine[n], send3_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend4fine[n], send4_fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5fine[n], send5_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend6fine[n], send6_fine[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1fine[n], receive1_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2fine[n], receive2_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3fine[n], receive3_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4fine[n], receive4_fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5fine[n], receive5_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6fine[n], receive6_fine[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3fine[n], receive1_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4fine[n], receive1_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7fine[n], receive1_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8fine[n], receive1_8fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1fine[n], receive2_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2fine[n], receive2_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3fine[n], receive2_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4fine[n], receive2_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1fine[n], receive3_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2fine[n], receive3_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5fine[n], receive3_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6fine[n], receive3_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5fine[n], receive4_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6fine[n], receive4_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7fine[n], receive4_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8fine[n], receive4_8fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1fine[n], receive5_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3fine[n], receive5_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5fine[n], receive5_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7fine[n], receive5_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2fine[n], receive6_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4fine[n], receive6_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6fine[n], receive6_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8fine[n], receive6_8fine[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1E[n], send1_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend2E[n], send2_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend3E[n], send3_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend4E[n], send4_E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5E[n], send5_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend6E[n], send6_E[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1E[n], receive1_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2E[n], receive2_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3E[n], receive3_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4E[n], receive4_E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5E[n], receive5_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6E[n], receive6_E[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3E[n], receive1_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E[n], receive1_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E[n], receive1_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E[n], receive1_8E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E[n], receive2_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E[n], receive2_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E[n], receive2_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E[n], receive2_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E[n], receive3_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E[n], receive3_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E[n], receive3_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E[n], receive3_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E[n], receive4_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E[n], receive4_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E[n], receive4_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E[n], receive4_8E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E[n], receive5_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E[n], receive5_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E[n], receive5_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E[n], receive5_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E[n], receive6_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E[n], receive6_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E[n], receive6_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E[n], receive6_8E[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1E1[n], receive1_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2E1[n], receive2_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3E1[n], receive3_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4E1[n], receive4_E1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5E1[n], receive5_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6E1[n], receive6_E1[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3E1[n], receive1_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E1[n], receive1_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E1[n], receive1_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E1[n], receive1_8E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E1[n], receive2_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E1[n], receive2_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E1[n], receive2_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E1[n], receive2_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E1[n], receive3_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E1[n], receive3_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E1[n], receive3_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E1[n], receive3_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E1[n], receive4_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E1[n], receive4_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E1[n], receive4_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E1[n], receive4_8E1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E1[n], receive5_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E1[n], receive5_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E1[n], receive5_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E1[n], receive5_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E1[n], receive6_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E1[n], receive6_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E1[n], receive6_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E1[n], receive6_8E1[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1_3E2[n], receive1_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E2[n], receive1_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E2[n], receive1_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E2[n], receive1_8E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E2[n], receive2_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E2[n], receive2_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E2[n], receive2_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E2[n], receive2_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E2[n], receive3_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E2[n], receive3_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E2[n], receive3_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E2[n], receive3_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E2[n], receive4_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E2[n], receive4_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E2[n], receive4_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E2[n], receive4_8E2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E2[n], receive5_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E2[n], receive5_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E2[n], receive5_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E2[n], receive5_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E2[n], receive6_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E2[n], receive6_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E2[n], receive6_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E2[n], receive6_8E2[n], 0);
	#endif
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&BuffersendE1corn9[n], send_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn10[n], send_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn11[n], send_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn12[n], send_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn5[n], send_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn6[n], send_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn7[n], send_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn8[n], send_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&BuffersendE3corn1[n], send_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn2[n], send_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn3[n], send_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn4[n], send_E3_corn4[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9[n], receive_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10[n], receive_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11[n], receive_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12[n], receive_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5[n], receive_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6[n], receive_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7[n], receive_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8[n], receive_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1[n], receive_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2[n], receive_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3[n], receive_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4[n], receive_E3_corn4[n], 0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9_3[n], receive_E1_corn9_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn9_7[n], receive_E1_corn9_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_1[n], receive_E1_corn10_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_5[n], receive_E1_corn10_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_4[n], receive_E1_corn11_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_6[n], receive_E1_corn11_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_2[n], receive_E1_corn12_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_8[n], receive_E1_corn12_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_2[n], receive_E2_corn5_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_4[n], receive_E2_corn5_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_1[n], receive_E2_corn6_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_3[n], receive_E2_corn6_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_5[n], receive_E2_corn7_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_7[n], receive_E2_corn7_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_6[n], receive_E2_corn8_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_8[n], receive_E2_corn8_2[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1_3[n], receive_E3_corn1_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn1_4[n], receive_E3_corn1_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_1[n], receive_E3_corn2_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_2[n], receive_E3_corn2_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_5[n], receive_E3_corn3_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_6[n], receive_E3_corn3_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_7[n], receive_E3_corn4_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_8[n], receive_E3_corn4_2[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrecE1corn9[n], tempreceive_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10[n], tempreceive_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11[n], tempreceive_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12[n], tempreceive_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5[n], tempreceive_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6[n], tempreceive_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7[n], tempreceive_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8[n], tempreceive_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrecE3corn1[n], tempreceive_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2[n], tempreceive_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3[n], tempreceive_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4[n], tempreceive_E3_corn4[n], 0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrecE1corn9_3[n], tempreceive_E1_corn9_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn9_7[n], tempreceive_E1_corn9_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10_1[n], tempreceive_E1_corn10_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10_5[n], tempreceive_E1_corn10_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11_2[n], tempreceive_E1_corn11_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11_6[n], tempreceive-E1_corn11_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12_4[n], tempreceive_E1_corn12_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12_8[n], tempreceive_E1_corn12_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5_2[n], tempreceive_E2_corn5_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5_4[n], tempreceive_E2_corn5_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6_1[n], tempreceive_E2_corn6_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6_3[n], tempreceive_E2_corn6_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7_5[n], tempreceive_E2_corn7_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7_7[n], tempreceive_E2_corn7_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8_6[n], tempreceive_E2_corn8_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8_8[n], tempreceive_E2_corn8_2[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrecE3corn1_3[n], tempreceive_E3_corn1_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn1_4[n], tempreceive_E3_corn1_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2_1[n], tempreceive_E3_corn2_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2_2[n], tempreceive_E3_corn2_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3_5[n], tempreceive_E3_corn3_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3_6[n], tempreceive_E3_corn3_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4_7[n], tempreceive_E3_corn4_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4_8[n], tempreceive_E3_corn4_2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9_32[n], receive_E1_corn9_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn9_72[n], receive_E1_corn9_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_12[n], receive_E1_corn10_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_52[n], receive_E1_corn10_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_22[n], receive_E1_corn11_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_62[n], receive_E1_corn11_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_42[n], receive_E1_corn12_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_82[n], receive_E1_corn12_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_22[n], receive_E2_corn5_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_42[n], receive_E2_corn5_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_12[n], receive_E2_corn6_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_32[n], receive_E2_corn6_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_52[n], receive_E2_corn7_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_72[n], receive_E2_corn7_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_62[n], receive_E2_corn8_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_82[n], receive_E2_corn8_22[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1_32[n], receive_E3_corn1_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn1_42[n], receive_E3_corn1_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_12[n], receive_E3_corn2_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_22[n], receive_E3_corn2_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_52[n], receive_E3_corn3_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_62[n], receive_E3_corn3_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_72[n], receive_E3_corn4_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_82[n], receive_E3_corn4_22[n], 0);
	#endif

	#endif

	/*Set arguments of kernel*/
	int POLE_1, POLE_2;
	if (block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3)) POLE_1 = 1;
	else POLE_1 = 0;
	if (block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3)) POLE_2 = 1;
	else POLE_2 = 0;

	int pg, d1, d2, k;
	#pragma omp parallel private(i, j, z, k, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (pg = 0; pg < NPG; pg++){
				#if(!NONSYMMETRIC)
				gdet_GPU[n][pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gdet[n][index2(n, i, j, z)][pg];
				#else
				#endif
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						gcov_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[n][index2(n, i, j, z)][pg][d1][d2];
						gcon_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[n][index2(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}

			for (pg = 0; pg < NDIM; pg++){
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						conn_GPU[n][d1*NDIM*NDIM*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + d2*NDIM*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2) 
							+ (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset + N2G)] = conn[n][index2(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}
			#if(LEER)
			for (k = 0; k < 6; k++){
				dq_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = V[n][index(n, i, j, z)][k];
			}
			#endif
		}
	}
	
	#if(LEER)
	status =cudaMemcpy(BufferV[n], dq_1[n], 6 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);;
	#endif
		
	#if(!NONSYMMETRIC)
	status += cudaMemcpy(Buffergdet[n], gdet_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2)*NPG*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Buffergcov[n], gcov_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Buffergcon[n], gcov_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2)*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferconn[n], conn_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2)*NDIM*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#else
	#endif

	if (status != 0) printf("Error in setting kernel arguments 2: %d \n", status);
	free(gcov_GPU[n]);
	free(gcon_GPU[n]);
	free(conn_GPU[n]);
	free(gdet_GPU[n]);
	#endif
}
double check = 1.0;

void GPU_write(int n)
{
	int i, j, z, k, l, pg, d1, d2;
	double radius_GPU[(N1 + 2 * N1G)], r, th, phi, X[NDIM];
	for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		coord(n, i, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		radius_GPU[(i - N1_GPU_offset[n] + N1G)] = r;
		Katm_GPU[n][i - N1_GPU_offset[n] + N1G] = Katm[n][i - N1_GPU_offset[n] + N1G];
	}

	#pragma omp parallel private(i, j, z, k, l, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = p[n][index(n, i, j, z)][k];
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index(n, i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index(n, i, j, z)][k];
				#endif
				#if(ibound)
				if (i == N1_GPU_offset[n]){
					for (l = 0; l < N_POINTS; l++){
						pbound_1[n][l*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n]) + k*((N2_GPU[n] + 2 * N2G) + fix_mem[n]) + (j - N2_GPU_offset[n] + N2G)] = pbound[j*(N3_GPU[n] + 2 * N3G) + z][k][l];
					}
				}
				#endif
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[n][index(n, i, j, z)][k];
				psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = psh[n][index(n, i, j, z)][k];

			}
			#endif
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
			}
			pflag_GPU[n][(i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
		}
	}

	status = 0;
	#if (ELLIPTICAL2)
	for (i = N1_GPU_offset[n] - N1G; i<N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		for (j = N2_GPU_offset[n] - N2G; j<N2_GPU_offset[n] + N2_GPU[n] + N2G; j++){
			for (k = 0; k < NPR; k++){
				dU_GPU[n][k*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = dU_s[n][index2(n, i, j, 0)][k];
			}
		}
	}
	status = cudaMemcpy(BufferdU[n], dU_GPU[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#endif
	
	/*Initialize memory items that have to be passed on to the GPU*/
	status = cudaMemcpy(Bufferp_1[n], p_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferph_1[n], ph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#if(STAGGERED)
	status += cudaMemcpy(Bufferps_1[n], ps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferpsh_1[n], psh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#endif
	status += cudaMemcpy(Bufferpflag[n], pflag_GPU[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(cl_int), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferfailimage[n], failimage_GPU[n], NFAIL*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(cl_int), cudaMemcpyHostToDevice);
	status += cudaMemcpy(BufferKatm[n], Katm_GPU[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferradius[n], radius_GPU, (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaMemcpyHostToDevice);

	if (status != 0) printf("Error in GPU_write: %d \n", status);
}

void GPU_hcor(int n){
	#if (1)
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_hcor1[n], 1, global_work_offset[n], global_work_size[n], local_work_size, 0, NULL, NULL);
	clFlush(commandQueueGPU[n]);

	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_hcor2[n], 1, global_work_offset[n], global_work_size2[n], local_work_size, 0, NULL, NULL);
	clFlush(commandQueueGPU[n]);

	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_hcor3[n], 1, global_work_offset[n], global_work_size2[n], local_work_size, 0, NULL, NULL);
	clFlush(commandQueueGPU[n]);
	#endif
}


void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	#if (1)
	/*Set arguments of kernel*/
	if (dir == 1){
		clSetKernelArg(kernel_fluxcalcprep[n], 3, sizeof(cl_mem), (void *)&BufferF1_1[n]);
	}
	else if (dir == 2){
		clSetKernelArg(kernel_fluxcalcprep[n], 3, sizeof(cl_mem), (void *)&BufferF2_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalcprep[n], 3, sizeof(cl_mem), (void *)&BufferF3_1[n]);
	}
	if (flag == 1){
		clSetKernelArg(kernel_fluxcalcprep[n], 5, sizeof(cl_mem), (void *)&Bufferph_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalcprep[n], 5, sizeof(cl_mem), (void *)&Bufferp_1[n]);
	}
	clSetKernelArg(kernel_fluxcalcprep[n], 6, sizeof(cl_int), &dir);
	clSetKernelArg(kernel_fluxcalcprep[n], 13, sizeof(cl_int), &ppm_solver);

	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fluxcalcprep[n], 1, global_work_offset[n], global_work_size2[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error Fluxcalcprep %d\n", status);

	global_work_size2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1 - (dir == 1)) * (N2_GPU[n] + 2 * D2 - (dir == 2)) * (N3_GPU[n] + 2 * D3 - (dir == 3))) % LOCAL_WORK_SIZE)
		+ (N1_GPU[n] + 2 * D1 - (dir == 1)) * (N2_GPU[n] + 2 * D2 - (dir == 2)) * (N3_GPU[n] + 2 * D3 - (dir == 3)); //fluxcalc
	nr_workgroups[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE) - 1;
	clFlush(commandQueueGPU[n]);
	#endif
}


void GPU_fluxcalc2D(int dir, int flag, int n)
{
	#if (1)
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);

	/*Set arguments of kernel*/
	if (dir == 1){
		clSetKernelArg(kernel_fluxcalc2D1[n], 3, sizeof(cl_mem), (void *)&BufferF1_1[n]);
	}
	else if (dir == 2){
		clSetKernelArg(kernel_fluxcalc2D1[n], 3, sizeof(cl_mem), (void *)&BufferF2_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalc2D1[n], 3, sizeof(cl_mem), (void *)&BufferF3_1[n]);
	}
	if (flag == 1){
		clSetKernelArg(kernel_fluxcalc2D1[n], 5, sizeof(cl_mem), (void *)&Bufferph_1[n]);
		clSetKernelArg(kernel_fluxcalc2D1[n], 29, sizeof(cl_mem), (void *)&Bufferpsh_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalc2D1[n], 5, sizeof(cl_mem), (void *)&Bufferp_1[n]);
		clSetKernelArg(kernel_fluxcalc2D1[n], 29, sizeof(cl_mem), (void *)&Bufferps_1[n]);
	}
	clSetKernelArg(kernel_fluxcalc2D1[n], 10, sizeof(cl_int), &dir);

	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fluxcalc2D1[n], 1, global_work_offset[n], global_work_size2[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error Fluxcalc2D1 %d\n", status);
	clFlush(commandQueueGPU[n]);

	/*Calculate reconstructed right state*/
	#if(PPM || LEER)
	global_work_size2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3)) % LOCAL_WORK_SIZE)
		+ (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3); //fluxcalc
	GPU_fluxcalcprep(dir, flag, 2, n);
	#endif

	/*Set arguments of kernel*/
	if (dir == 1){
		clSetKernelArg(kernel_fluxcalc2D2[n], 3, sizeof(cl_mem), (void *)&BufferF1_1[n]);
	}
	else if (dir == 2){
		clSetKernelArg(kernel_fluxcalc2D2[n], 3, sizeof(cl_mem), (void *)&BufferF2_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalc2D2[n], 3, sizeof(cl_mem), (void *)&BufferF3_1[n]);
	}
	if (flag == 1){
		clSetKernelArg(kernel_fluxcalc2D2[n], 5, sizeof(cl_mem), (void *)&Bufferph_1[n]);
		clSetKernelArg(kernel_fluxcalc2D2[n], 34, sizeof(cl_mem), (void *)&Bufferpsh_1[n]);
	}
	else{
		clSetKernelArg(kernel_fluxcalc2D2[n], 5, sizeof(cl_mem), (void *)&Bufferp_1[n]);
		clSetKernelArg(kernel_fluxcalc2D2[n], 34, sizeof(cl_mem), (void *)&Bufferps_1[n]);
	}
	clSetKernelArg(kernel_fluxcalc2D2[n], 10, sizeof(cl_int), &dir);

	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fluxcalc2D2[n], 1, global_work_offset[n], global_work_size2[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error Fluxcalc2D2 %d\n", status);
	clFlush(commandQueueGPU[n]);
	global_work_size2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3)) % LOCAL_WORK_SIZE)
		+ (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3); //fluxcalc
	#endif
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fix_flux[n], 1, 0, global_work_size_special[n], local_work_size, 0, NULL, NULL);
	if (status != 0)printf("Error fixflux %d \n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_consttransport_bound(void){
	int n;

	gpu = 1;
		#if(PRESTEP)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		#if(!TIMESTEP_JET)
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		E_send1(E_corn, BufferE_1, n_ord[n]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_send3(E_corn, BufferE_1, n_ord[n]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		#endif
	}

	//For last timestep synchronize electric fields immediately
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		#if(!TIMESTEP_JET)
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		E_rec1(E_corn, BufferE_1, n_ord[n], 5);
		E_rec2(E_corn, BufferE_1, n_ord[n], 5);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_rec3(E_corn, BufferE_1, n_ord[n], 5);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		#endif
	}

	//For first timestep do not synchronize electrice fields 
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		#if(!TIMESTEP_JET)
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
		#endif
	}
	#else
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_send_corn(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	#endif
	
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send1(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec1(E_corn, BufferE_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send2(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec2(E_corn, BufferE_1, n_ord[n], 2);

	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_send3(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E_rec3(E_corn, BufferE_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_send_corn(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_send_corn(E_corn, BufferE_1, n_ord[n]);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1)E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	#endif
	#endif
	#endif
	#if(TRANS_BOUND)
	E_average();
	#endif
}

void GPU_consttransport1(int flag, double Dt, int n){
	size_t global_work_size_local[1];

	/*Run kernel*/
	if (flag == 1){
		clSetKernelArg(kernel_consttransport1[n], 3, sizeof(cl_mem), (void *)&Bufferph_1[n]);
	}
	else{
		clSetKernelArg(kernel_consttransport1[n], 3, sizeof(cl_mem), (void *)&Bufferp_1[n]);
	}
	global_work_size_local[0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G);
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_consttransport1[n], 1, global_work_offset[n], global_work_size_local, local_work_size, 0, NULL, NULL);
	if (status != 0)printf("Error consttransport1 %d \n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_consttransport2(int flag, double Dt, int n){
	size_t global_work_size_local[1];

	/*Run kernel*/
	if (flag == 1){
		clSetKernelArg(kernel_consttransport2[n], 8, sizeof(cl_mem), (void *)&Bufferph_1[n]);
	}
	else{
		clSetKernelArg(kernel_consttransport2[n], 8, sizeof(cl_mem), (void *)&Bufferp_1[n]);
	}
	global_work_size_local[0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3);
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_consttransport2[n], 1, global_work_offset[n], global_work_size_local, local_work_size, 0, NULL, NULL);
	if (status != 0)printf("Error constransport2 %d \n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_consttransport3(int flag, double Dt, int n){
	size_t global_work_size_local[1];

	/*Run kernel*/
	clSetKernelArg(kernel_consttransport3[n], 10, sizeof(FTYPE2), &Dt);
	if (flag == 1){
		clSetKernelArg(kernel_consttransport3[n], 7, sizeof(cl_mem), (void *)&Bufferps_1[n]);
		clSetKernelArg(kernel_consttransport3[n], 8, sizeof(cl_mem), (void *)&Bufferps_1[n]);
	}
	else{
		clSetKernelArg(kernel_consttransport3[n], 7, sizeof(cl_mem), (void *)&Bufferps_1[n]);
		clSetKernelArg(kernel_consttransport3[n], 8, sizeof(cl_mem), (void *)&Bufferpsh_1[n]);
	}
	global_work_size_local[0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3);
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_consttransport3[n], 1, global_work_offset[n], global_work_size_local, local_work_size, 0, NULL, NULL);
	if (status != 0)printf("Error constransport3 %d \n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_flux_ct1(int n)
{
	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_flux_ct1[n], 1, global_work_offset[n], global_work_size3[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error fluxct1 %d\n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_flux_ct2(int n)
{
	/*Run kernel*/
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_flux_ct2[n], 1, global_work_offset[n], global_work_size3[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error fluxct2 %d\n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_Utoprim(int flag, int n)
{

}

void GPU_fixuputoprim(int flag, int n)
{
	size_t global_work_size_local[1];
	if (flag == 0){
		clSetKernelArg(kernel_fixuputoprim[n], 3, sizeof(cl_mem), (void *)&Bufferph_1[n]);
	}
	else{
		clSetKernelArg(kernel_fixuputoprim[n], 3, sizeof(cl_mem), (void *)&Bufferp_1[n]);
	}
	global_work_size_local[0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G);
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fixuputoprim[n], 1, global_work_offset[n], global_work_size_local, local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error fixuputoprim %d\n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_fixup(int flag, int n)
{
	if (flag == 0){
		clSetKernelArg(kernel_fixup[n], 4, sizeof(cl_mem), (void *)&Bufferp_1[n]);
		clSetKernelArg(kernel_fixup[n], 5, sizeof(cl_mem), (void *)&Bufferph_1[n]);
		clSetKernelArg(kernel_fixup[n], 24, sizeof(cl_int), &flag);

	}
	else{
		clSetKernelArg(kernel_fixup[n], 4, sizeof(cl_mem), (void *)&Bufferph_1[n]);
		clSetKernelArg(kernel_fixup[n], 5, sizeof(cl_mem), (void *)&Bufferp_1[n]);
		clSetKernelArg(kernel_fixup[n], 24, sizeof(cl_int), &flag);
	}
	status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_fixup[n], 1, global_work_offset[n], global_work_size1[n], local_work_size, 0, NULL, NULL);
	if (status != 0) printf("Error fixup %d\n", status);
	clFlush(commandQueueGPU[n]);
}

void GPU_boundprim(int bound_force)
{
	int i, n;
	int temp = nstep;
	gpu = 1;

	if (bound_force == 1) nstep = -1;
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim1(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim2(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	//For last timestep do not receive synchronized electrice fields 
	#if(PRESTEP)
	rc = 0;
	//MPI communication
	for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
			if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
			break;
		}
	}

	if (rank == 0){
		begin2 = clock();
	}
	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			flux_rec1(F1, BufferF1_1, n_ord[n], 3);
			flux_rec2(F2, BufferF2_1, n_ord[n], 3);
			flux_rec3(F3, BufferF3_1, n_ord[n], 3);

			#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
			#if(!TIMESTEP_JET)
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
			E_rec1(E_corn, BufferE_1, n_ord[n], 3);
			E_rec2(E_corn, BufferE_1, n_ord[n], 3);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 3);
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
			#endif
			#endif
		}

		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);

			#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
			#if(!TIMESTEP_JET)
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			E_rec1(E_corn, BufferE_1, n_ord[n], 1);
			E_rec2(E_corn, BufferE_1, n_ord[n], 1);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 1);
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			#endif
			#endif
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE/BoundcomF \n");
	#endif


	rc = 0;
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, Bufferp_1, 0, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, Bufferph_1, 0, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, Bufferp_1, 0, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, Bufferph_1, 0, n_ord[n]);
	}

	if (N3 > 1){
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
		}

		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, Bufferp_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, Bufferph_1, 0, n_ord[n]);
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");
	//MPI communication
	for (n = 0; n < n_active; n++){
		//clFinish(commandQueueGPU[n_ord[n]]);
	}
	for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
			if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
			break;
		}
	}

	if (rank == 0){
		end2 = clock();
		time_spent3 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	}
	receive_tag = 0;

	#if (STAGGERED && COPY_BFIELD)
	rc = 0;
	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){ //watch out does this for both half and full timestep while only needed for full timestep
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send3(ps, Bufferps_1, n_ord[n]);
		}
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_rec1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_rec2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) B_rec3(ps, Bufferps_1, n_ord[n]);
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB \n");
	#endif
	nstep = temp;
	
}

void GPU_boundprim1(int flag, int n)
{
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			clSetKernelArg(kernel_boundprim1[n], 3, sizeof(cl_mem), (void *)&Bufferph_1[n]);
			clSetKernelArg(kernel_boundprim1[n], 19, sizeof(cl_mem), (void *)&Bufferpsh_1[n]);
		}
		else{
			clSetKernelArg(kernel_boundprim1[n], 3, sizeof(cl_mem), (void *)&Bufferp_1[n]);
			clSetKernelArg(kernel_boundprim1[n], 19, sizeof(cl_mem), (void *)&Bufferps_1[n]);
		}
		status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_boundprim1[n], 1, 0, global_work_size_special1[n], local_work_size, 0, NULL, NULL);
		if (status != 0) printf("Error boundprim1 %d\n", status);
		clFlush(commandQueueGPU[n]);
	}
}

void GPU_boundprim2(int flag, int n)
{
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			clSetKernelArg(kernel_boundprim2[n], 4, sizeof(cl_mem), (void *)&Bufferph_1[n]);
			clSetKernelArg(kernel_boundprim2[n], 9, sizeof(cl_mem), (void *)&Bufferpsh_1[n]);
		}
		else{
			clSetKernelArg(kernel_boundprim2[n], 4, sizeof(cl_mem), (void *)&Bufferp_1[n]);
			clSetKernelArg(kernel_boundprim2[n], 9, sizeof(cl_mem), (void *)&Bufferps_1[n]);
		}
		status = clEnqueueNDRangeKernel(commandQueueGPU[n], kernel_boundprim2[n], 1, 0, global_work_size_special2[n], local_work_size, 0, NULL, NULL);
		if (status != 0) printf("Error boundprim2 %d\n", status);
		clFlush(commandQueueGPU[n]);
	}
}

void GPU_read(int n)
{
	int i, j, z, k, l, pg, d1, d2;
	status = 0;
	status += cudaMemcpy(p_1[n], Bufferp_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	status += cudaMemcpy(ph_1[n], Bufferph_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	#if(STAGGERED)
	status += cudaMemcpy(ps_1[n], Bufferps_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	status += cudaMemcpy(psh_1[n], Bufferpsh_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	#endif
	status += cudaMemcpy(failimage_GPU[n], Bufferfailimage[n], (int)((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(cl_int), cudaMemcpyDeviceToHost);

	#pragma omp parallel private(i, j, z, k, l, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[n][index(n, i, j, z)][k] = p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				ph[n][index(n, i, j, z)][k] = ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];

				#if(GPU_DEBUG)
				ph[n][index(n, i, j, z)][k] = ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			#endif
			}
			for (k = 0; k < NFAIL; k++){
				failimage[n][index(n, i, j, z)][k] = failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps[n][index(n, i, j, z)][k] = ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				psh[n][index(n, i, j, z)][k] = psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
		#endif
		}
	}
	if (status != 0)printf("Error in GPU_read: %d \n", status);
}


void GPU_finish(int n)
{
	//Attention
	//clFinish(commandQueueGPU[n]);

	free(p_1[n]);
	free(dq_1[n]);
	#if(STAGGERED)
	free(ps_1[n]);
	free(psh_1[n]);
	#endif
	free(ph_1[n]);
	free(dU_GPU[n]);
	free(pbound_1[n]);
	free(pflag_GPU[n]);
	free(failimage_GPU[n]);
	free(Katm_GPU[n]);
	
	status += cudaFree(Bufferdtij[n]);
	status += cudaFree(BufferF1_1[n]);
	status += cudaFree(BufferF2_1[n]);
	status += cudaFree(BufferF3_1[n]);
	status += cudaFree(Bufferdq_1[n]);
	status += cudaFree(BufferE_1[n]);
	#if(LEER)
	status += cudaFree(BufferV[n]);
	#endif
	status += cudaFree(Bufferradius[n]);
	status += cudaFree(Bufferstorage1[n]);
	status += cudaFree(Bufferstorage2[n]);
	status += cudaFree(Bufferstorage3[n]);
	status += cudaFree(Bufferstorage4[n]);
	//status += cudaFree(Buffereta_avg[n]);
	status += cudaFree(Bufferp_1[n]);
	status += cudaFree(Bufferph_1[n]);
	#if(STAGGERED)
	status += cudaFree(Bufferps_1[n]);
	status += cudaFree(Bufferpsh_1[n]);
	#endif
	status += cudaFree(Bufferpbound_1[n]);
	status += cudaFree(Bufferpflag[n]);
	status += cudaFree(Bufferfailimage[n]);
	status += cudaFree(Bufferdiagflux[n]);
	status += cudaFree(BufferKatm[n]);
	status += cudaFree(BufferdU[n]);
	status += cudaFree(Buffersend1[n]);
	status += cudaFree(Buffersend1_3[n]);
	status += cudaFree(Buffersend1_4[n]);
	status += cudaFree(Buffersend1_7[n]);
	status += cudaFree(Buffersend1_8[n]);
	status += cudaFree(Buffersend2[n]);
	status += cudaFree(Buffersend2_1[n]);
	status += cudaFree(Buffersend2_2[n]);
	status += cudaFree(Buffersend2_3[n]);
	status += cudaFree(Buffersend2_4[n]);
	status += cudaFree(Buffersend3[n]);
	status += cudaFree(Buffersend3_1[n]);
	status += cudaFree(Buffersend3_2[n]);
	status += cudaFree(Buffersend3_5[n]);
	status += cudaFree(Buffersend3_6[n]);
	status += cudaFree(Buffersend4[n]);
	status += cudaFree(Buffersend4_5[n]);
	status += cudaFree(Buffersend4_6[n]);
	status += cudaFree(Buffersend4_7[n]);
	status += cudaFree(Buffersend4_8[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5[n]);
	status += cudaFree(Buffersend5_1[n]);
	status += cudaFree(Buffersend5_3[n]);
	status += cudaFree(Buffersend5_5[n]);
	status += cudaFree(Buffersend5_7[n]);
	status += cudaFree(Buffersend6[n]);
	status += cudaFree(Buffersend6_2[n]);
	status += cudaFree(Buffersend6_4[n]);
	status += cudaFree(Buffersend6_6[n]);
	status += cudaFree(Buffersend6_8[n]);
	#endif
	status += cudaFree(Bufferrec1[n]);
	status += cudaFree(Bufferrec1_3[n]);
	status += cudaFree(Bufferrec1_4[n]);
	status += cudaFree(Bufferrec1_7[n]);
	status += cudaFree(Bufferrec1_8[n]);
	status += cudaFree(Bufferrec2[n]);
	status += cudaFree(Bufferrec2_1[n]);
	status += cudaFree(Bufferrec2_2[n]);
	status += cudaFree(Bufferrec2_3[n]);
	status += cudaFree(Bufferrec2_4[n]);
	status += cudaFree(Bufferrec3[n]);
	status += cudaFree(Bufferrec3_1[n]);
	status += cudaFree(Bufferrec3_2[n]);
	status += cudaFree(Bufferrec3_5[n]);
	status += cudaFree(Bufferrec3_6[n]);
	status += cudaFree(Bufferrec4[n]);
	status += cudaFree(Bufferrec4_5[n]);
	status += cudaFree(Bufferrec4_6[n]);
	status += cudaFree(Bufferrec4_7[n]);
	status += cudaFree(Bufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5[n]);
	status += cudaFree(Bufferrec5_1[n]);
	status += cudaFree(Bufferrec5_3[n]);
	status += cudaFree(Bufferrec5_5[n]);
	status += cudaFree(Bufferrec5_7[n]);
	status += cudaFree(Bufferrec6[n]);
	status += cudaFree(Bufferrec6_2[n]);
	status += cudaFree(Bufferrec6_4[n]);
	status += cudaFree(Bufferrec6_6[n]);
	status += cudaFree(Bufferrec6_8[n]);
	#endif
	status += cudaFree(tempBufferrec1[n]);
	status += cudaFree(tempBufferrec1_3[n]);
	status += cudaFree(tempBufferrec1_4[n]);
	status += cudaFree(tempBufferrec1_7[n]);
	status += cudaFree(tempBufferrec1_8[n]);
	status += cudaFree(tempBufferrec2[n]);
	status += cudaFree(tempBufferrec2_1[n]);
	status += cudaFree(tempBufferrec2_2[n]);
	status += cudaFree(tempBufferrec2_3[n]);
	status += cudaFree(tempBufferrec2_4[n]);
	status += cudaFree(tempBufferrec3[n]);
	status += cudaFree(tempBufferrec3_1[n]);
	status += cudaFree(tempBufferrec3_2[n]);
	status += cudaFree(tempBufferrec3_5[n]);
	status += cudaFree(tempBufferrec3_6[n]);
	status += cudaFree(tempBufferrec4[n]);
	status += cudaFree(tempBufferrec4_5[n]);
	status += cudaFree(tempBufferrec4_6[n]);
	status += cudaFree(tempBufferrec4_7[n]);
	status += cudaFree(tempBufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrec5[n]);
	status += cudaFree(tempBufferrec5_1[n]);
	status += cudaFree(tempBufferrec5_3[n]);
	status += cudaFree(tempBufferrec5_5[n]);
	status += cudaFree(tempBufferrec5_7[n]);
	status += cudaFree(tempBufferrec6[n]);
	status += cudaFree(tempBufferrec6_2[n]);
	status += cudaFree(tempBufferrec6_4[n]);
	status += cudaFree(tempBufferrec6_6[n]);
	status += cudaFree(tempBufferrec6_8[n]);
	#endif
	status += cudaFree(Buffersend1flux[n]);
	status += cudaFree(Buffersend2flux[n]);
	status += cudaFree(Buffersend3flux[n]);
	status += cudaFree(Buffersend4flux[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5flux[n]);
	status += cudaFree(Buffersend6flux[n]);
	#endif
	status += cudaFree(Bufferrec1flux[n]);
	status += cudaFree(Bufferrec2flux[n]);
	status += cudaFree(Bufferrec3flux[n]);
	status += cudaFree(Bufferrec4flux[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5flux[n]);
	status += cudaFree(Bufferrec6flux[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux[n]);
	status += cudaFree(Bufferrec1_4flux[n]);
	status += cudaFree(Bufferrec1_7flux[n]);
	status += cudaFree(Bufferrec1_8flux[n]);
	status += cudaFree(Bufferrec2_1flux[n]);
	status += cudaFree(Bufferrec2_2flux[n]);
	status += cudaFree(Bufferrec2_3flux[n]);
	status += cudaFree(Bufferrec2_4flux[n]);
	status += cudaFree(Bufferrec3_1flux[n]);
	status += cudaFree(Bufferrec3_2flux[n]);
	status += cudaFree(Bufferrec3_5flux[n]);
	status += cudaFree(Bufferrec3_6flux[n]);
	status += cudaFree(Bufferrec4_5flux[n]);
	status += cudaFree(Bufferrec4_6flux[n]);
	status += cudaFree(Bufferrec4_7flux[n]);
	status += cudaFree(Bufferrec4_8flux[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux[n]);
	status += cudaFree(Bufferrec5_3flux[n]);
	status += cudaFree(Bufferrec5_5flux[n]);
	status += cudaFree(Bufferrec5_7flux[n]);
	status += cudaFree(Bufferrec6_2flux[n]);
	status += cudaFree(Bufferrec6_4flux[n]);
	status += cudaFree(Bufferrec6_6flux[n]);
	status += cudaFree(Bufferrec6_8flux[n]);
	#endif
	status += cudaFree(Bufferrec1flux1[n]);
	status += cudaFree(Bufferrec2flux1[n]);
	status += cudaFree(Bufferrec3flux1[n]);
	status += cudaFree(Bufferrec4flux1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5flux1[n]);
	status += cudaFree(Bufferrec6flux1[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux1[n]);
	status += cudaFree(Bufferrec1_4flux1[n]);
	status += cudaFree(Bufferrec1_7flux1[n]);
	status += cudaFree(Bufferrec1_8flux1[n]);
	status += cudaFree(Bufferrec2_1flux1[n]);
	status += cudaFree(Bufferrec2_2flux1[n]);
	status += cudaFree(Bufferrec2_3flux1[n]);
	status += cudaFree(Bufferrec2_4flux1[n]);
	status += cudaFree(Bufferrec3_1flux1[n]);
	status += cudaFree(Bufferrec3_2flux1[n]);
	status += cudaFree(Bufferrec3_5flux1[n]);
	status += cudaFree(Bufferrec3_6flux1[n]);
	status += cudaFree(Bufferrec4_5flux1[n]);
	status += cudaFree(Bufferrec4_6flux1[n]);
	status += cudaFree(Bufferrec4_7flux1[n]);
	status += cudaFree(Bufferrec4_8flux1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux1[n]);
	status += cudaFree(Bufferrec5_3flux1[n]);
	status += cudaFree(Bufferrec5_5flux1[n]);
	status += cudaFree(Bufferrec5_7flux1[n]);
	status += cudaFree(Bufferrec6_2flux1[n]);
	status += cudaFree(Bufferrec6_4flux1[n]);
	status += cudaFree(Bufferrec6_6flux1[n]);
	status += cudaFree(Bufferrec6_8flux1[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux2[n]);
	status += cudaFree(Bufferrec1_4flux2[n]);
	status += cudaFree(Bufferrec1_7flux2[n]);
	status += cudaFree(Bufferrec1_8flux2[n]);
	status += cudaFree(Bufferrec2_1flux2[n]);
	status += cudaFree(Bufferrec2_2flux2[n]);
	status += cudaFree(Bufferrec2_3flux2[n]);
	status += cudaFree(Bufferrec2_4flux2[n]);
	status += cudaFree(Bufferrec3_1flux2[n]);
	status += cudaFree(Bufferrec3_2flux2[n]);
	status += cudaFree(Bufferrec3_5flux2[n]);
	status += cudaFree(Bufferrec3_6flux2[n]);
	status += cudaFree(Bufferrec4_5flux2[n]);
	status += cudaFree(Bufferrec4_6flux2[n]);
	status += cudaFree(Bufferrec4_7flux2[n]);
	status += cudaFree(Bufferrec4_8flux2[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux2[n]);
	status += cudaFree(Bufferrec5_3flux2[n]);
	status += cudaFree(Bufferrec5_5flux2[n]);
	status += cudaFree(Bufferrec5_7flux2[n]);
	status += cudaFree(Bufferrec6_2flux2[n]);
	status += cudaFree(Bufferrec6_4flux2[n]);
	status += cudaFree(Bufferrec6_6flux2[n]);
	status += cudaFree(Bufferrec6_8flux2[n]);
	#endif
	status += cudaFree(Buffersend1fine[n]);
	status += cudaFree(Buffersend2fine[n]);
	status += cudaFree(Buffersend3fine[n]);
	status += cudaFree(Buffersend4fine[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5fine[n]);
	status += cudaFree(Buffersend6fine[n]);
	#endif
	status += cudaFree(Bufferrec1fine[n]);
	status += cudaFree(Bufferrec2fine[n]);
	status += cudaFree(Bufferrec3fine[n]);
	status += cudaFree(Bufferrec4fine[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5fine[n]);
	status += cudaFree(Bufferrec6fine[n]);
	#endif
	status += cudaFree(Bufferrec1_3fine[n]);
	status += cudaFree(Bufferrec1_4fine[n]);
	status += cudaFree(Bufferrec1_7fine[n]);
	status += cudaFree(Bufferrec1_8fine[n]);
	status += cudaFree(Bufferrec2_1fine[n]);
	status += cudaFree(Bufferrec2_2fine[n]);
	status += cudaFree(Bufferrec2_3fine[n]);
	status += cudaFree(Bufferrec2_4fine[n]);
	status += cudaFree(Bufferrec3_1fine[n]);
	status += cudaFree(Bufferrec3_2fine[n]);
	status += cudaFree(Bufferrec3_5fine[n]);
	status += cudaFree(Bufferrec3_6fine[n]);
	status += cudaFree(Bufferrec4_5fine[n]);
	status += cudaFree(Bufferrec4_6fine[n]);
	status += cudaFree(Bufferrec4_7fine[n]);
	status += cudaFree(Bufferrec4_8fine[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1fine[n]);
	status += cudaFree(Bufferrec5_3fine[n]);
	status += cudaFree(Bufferrec5_5fine[n]);
	status += cudaFree(Bufferrec5_7fine[n]);
	status += cudaFree(Bufferrec6_2fine[n]);
	status += cudaFree(Bufferrec6_4fine[n]);
	status += cudaFree(Bufferrec6_6fine[n]);
	status += cudaFree(Bufferrec6_8fine[n]);
	#endif
	status += cudaFree(Buffersend1E[n]);
	status += cudaFree(Buffersend2E[n]);
	status += cudaFree(Buffersend3E[n]);
	status += cudaFree(Buffersend4E[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5E[n]);
	status += cudaFree(Buffersend6E[n]);
	#endif
	status += cudaFree(Bufferrec1E[n]);
	status += cudaFree(Bufferrec2E[n]);
	status += cudaFree(Bufferrec3E[n]);
	status += cudaFree(Bufferrec4E[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5E[n]);
	status += cudaFree(Bufferrec6E[n]);
	#endif
	status += cudaFree(Bufferrec1_3E[n]);
	status += cudaFree(Bufferrec1_4E[n]);
	status += cudaFree(Bufferrec1_7E[n]);
	status += cudaFree(Bufferrec1_8E[n]);
	status += cudaFree(Bufferrec2_1E[n]);
	status += cudaFree(Bufferrec2_2E[n]);
	status += cudaFree(Bufferrec2_3E[n]);
	status += cudaFree(Bufferrec2_4E[n]);
	status += cudaFree(Bufferrec3_1E[n]);
	status += cudaFree(Bufferrec3_2E[n]);
	status += cudaFree(Bufferrec3_5E[n]);
	status += cudaFree(Bufferrec3_6E[n]);
	status += cudaFree(Bufferrec4_5E[n]);
	status += cudaFree(Bufferrec4_6E[n]);
	status += cudaFree(Bufferrec4_7E[n]);
	status += cudaFree(Bufferrec4_8E[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E[n]);
	status += cudaFree(Bufferrec5_3E[n]);
	status += cudaFree(Bufferrec5_5E[n]);
	status += cudaFree(Bufferrec5_7E[n]);
	status += cudaFree(Bufferrec6_2E[n]);
	status += cudaFree(Bufferrec6_4E[n]);
	status += cudaFree(Bufferrec6_6E[n]);
	status += cudaFree(Bufferrec6_8E[n]);
	#endif
	status += cudaFree(Bufferrec1E1[n]);
	status += cudaFree(Bufferrec2E1[n]);
	status += cudaFree(Bufferrec3E1[n]);
	status += cudaFree(Bufferrec4E1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5E1[n]);
	status += cudaFree(Bufferrec6E1[n]);
	#endif
	status += cudaFree(Bufferrec1_3E1[n]);
	status += cudaFree(Bufferrec1_4E1[n]);
	status += cudaFree(Bufferrec1_7E1[n]);
	status += cudaFree(Bufferrec1_8E1[n]);
	status += cudaFree(Bufferrec2_1E1[n]);
	status += cudaFree(Bufferrec2_2E1[n]);
	status += cudaFree(Bufferrec2_3E1[n]);
	status += cudaFree(Bufferrec2_4E1[n]);
	status += cudaFree(Bufferrec3_1E1[n]);
	status += cudaFree(Bufferrec3_2E1[n]);
	status += cudaFree(Bufferrec3_5E1[n]);
	status += cudaFree(Bufferrec3_6E1[n]);
	status += cudaFree(Bufferrec4_5E1[n]);
	status += cudaFree(Bufferrec4_6E1[n]);
	status += cudaFree(Bufferrec4_7E1[n]);
	status += cudaFree(Bufferrec4_8E1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E1[n]);
	status += cudaFree(Bufferrec5_3E1[n]);
	status += cudaFree(Bufferrec5_5E1[n]);
	status += cudaFree(Bufferrec5_7E1[n]);
	status += cudaFree(Bufferrec6_2E1[n]);
	status += cudaFree(Bufferrec6_4E1[n]);
	status += cudaFree(Bufferrec6_6E1[n]);
	status += cudaFree(Bufferrec6_8E1[n]);
	#endif
	status += cudaFree(Bufferrec1_3E2[n]);
	status += cudaFree(Bufferrec1_4E2[n]);
	status += cudaFree(Bufferrec1_7E2[n]);
	status += cudaFree(Bufferrec1_8E2[n]);
	status += cudaFree(Bufferrec2_1E2[n]);
	status += cudaFree(Bufferrec2_2E2[n]);
	status += cudaFree(Bufferrec2_3E2[n]);
	status += cudaFree(Bufferrec2_4E2[n]);
	status += cudaFree(Bufferrec3_1E2[n]);
	status += cudaFree(Bufferrec3_2E2[n]);
	status += cudaFree(Bufferrec3_5E2[n]);
	status += cudaFree(Bufferrec3_6E2[n]);
	status += cudaFree(Bufferrec4_5E2[n]);
	status += cudaFree(Bufferrec4_6E2[n]);
	status += cudaFree(Bufferrec4_7E2[n]);
	status += cudaFree(Bufferrec4_8E2[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E2[n]);
	status += cudaFree(Bufferrec5_3E2[n]);
	status += cudaFree(Bufferrec5_5E2[n]);
	status += cudaFree(Bufferrec5_7E2[n]);
	status += cudaFree(Bufferrec6_2E2[n]);
	status += cudaFree(Bufferrec6_4E2[n]);
	status += cudaFree(Bufferrec6_6E2[n]);
	status += cudaFree(Bufferrec6_8E2[n]);
	#endif
	#if(N3G>0)
	status += cudaFree(BuffersendE1corn9[n]);
	status += cudaFree(BuffersendE1corn10[n]);
	status += cudaFree(BuffersendE1corn11[n]);
	status += cudaFree(BuffersendE1corn12[n]);
	status += cudaFree(BuffersendE2corn5[n]);
	status += cudaFree(BuffersendE2corn6[n]);
	status += cudaFree(BuffersendE2corn7[n]);
	status += cudaFree(BuffersendE2corn8[n]);
	#endif
	status += cudaFree(BuffersendE3corn1[n]);
	status += cudaFree(BuffersendE3corn2[n]);
	status += cudaFree(BuffersendE3corn3[n]);
	status += cudaFree(BuffersendE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9[n]);
	status += cudaFree(BufferrecE1corn10[n]);
	status += cudaFree(BufferrecE1corn11[n]);
	status += cudaFree(BufferrecE1corn12[n]);
	status += cudaFree(BufferrecE2corn5[n]);
	status += cudaFree(BufferrecE2corn6[n]);
	status += cudaFree(BufferrecE2corn7[n]);
	status += cudaFree(BufferrecE2corn8[n]);
	#endif
	status += cudaFree(BufferrecE3corn1[n]);
	status += cudaFree(BufferrecE3corn2[n]);
	status += cudaFree(BufferrecE3corn3[n]);
	status += cudaFree(BufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9_3[n]);
	status += cudaFree(BufferrecE1corn9_7[n]);
	status += cudaFree(BufferrecE1corn10_1[n]);
	status += cudaFree(BufferrecE1corn10_5[n]);
	status += cudaFree(BufferrecE1corn11_2[n]);
	status += cudaFree(BufferrecE1corn11_6[n]);
	status += cudaFree(BufferrecE1corn12_4[n]);
	status += cudaFree(BufferrecE1corn12_8[n]);
	status += cudaFree(BufferrecE2corn5_2[n]);
	status += cudaFree(BufferrecE2corn5_4[n]);
	status += cudaFree(BufferrecE2corn6_1[n]);
	status += cudaFree(BufferrecE2corn6_3[n]);
	status += cudaFree(BufferrecE2corn7_5[n]);
	status += cudaFree(BufferrecE2corn7_7[n]);
	status += cudaFree(BufferrecE2corn8_6[n]);
	status += cudaFree(BufferrecE2corn8_8[n]);
	#endif
	status += cudaFree(BufferrecE3corn1_3[n]);
	status += cudaFree(BufferrecE3corn1_4[n]);
	status += cudaFree(BufferrecE3corn2_1[n]);
	status += cudaFree(BufferrecE3corn2_2[n]);
	status += cudaFree(BufferrecE3corn3_5[n]);
	status += cudaFree(BufferrecE3corn3_6[n]);
	status += cudaFree(BufferrecE3corn4_7[n]);
	status += cudaFree(BufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrecE1corn9[n]);
	status += cudaFree(tempBufferrecE1corn10[n]);
	status += cudaFree(tempBufferrecE1corn11[n]);
	status += cudaFree(tempBufferrecE1corn12[n]);
	status += cudaFree(tempBufferrecE2corn5[n]);
	status += cudaFree(tempBufferrecE2corn6[n]);
	status += cudaFree(tempBufferrecE2corn7[n]);
	status += cudaFree(tempBufferrecE2corn8[n]);
	#endif
	status += cudaFree(tempBufferrecE3corn1[n]);
	status += cudaFree(tempBufferrecE3corn2[n]);
	status += cudaFree(tempBufferrecE3corn3[n]);
	status += cudaFree(tempBufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrecE1corn9_3[n]);
	status += cudaFree(tempBufferrecE1corn9_7[n]);
	status += cudaFree(tempBufferrecE1corn10_1[n]);
	status += cudaFree(tempBufferrecE1corn10_5[n]);
	status += cudaFree(tempBufferrecE1corn11_2[n]);
	status += cudaFree(tempBufferrecE1corn11_6[n]);
	status += cudaFree(tempBufferrecE1corn12_4[n]);
	status += cudaFree(tempBufferrecE1corn12_8[n]);
	status += cudaFree(tempBufferrecE2corn5_2[n]);
	status += cudaFree(tempBufferrecE2corn5_4[n]);
	status += cudaFree(tempBufferrecE2corn6_1[n]);
	status += cudaFree(tempBufferrecE2corn6_3[n]);
	status += cudaFree(tempBufferrecE2corn7_5[n]);
	status += cudaFree(tempBufferrecE2corn7_7[n]);
	status += cudaFree(tempBufferrecE2corn8_6[n]);
	status += cudaFree(tempBufferrecE2corn8_8[n]);
	#endif
	status += cudaFree(tempBufferrecE3corn1_3[n]);
	status += cudaFree(tempBufferrecE3corn1_4[n]);
	status += cudaFree(tempBufferrecE3corn2_1[n]);
	status += cudaFree(tempBufferrecE3corn2_2[n]);
	status += cudaFree(tempBufferrecE3corn3_5[n]);
	status += cudaFree(tempBufferrecE3corn3_6[n]);
	status += cudaFree(tempBufferrecE3corn4_7[n]);
	status += cudaFree(tempBufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9_32[n]);
	status += cudaFree(BufferrecE1corn9_72[n]);
	status += cudaFree(BufferrecE1corn10_12[n]);
	status += cudaFree(BufferrecE1corn10_52[n]);
	status += cudaFree(BufferrecE1corn11_22[n]);
	status += cudaFree(BufferrecE1corn11_62[n]);
	status += cudaFree(BufferrecE1corn12_42[n]);
	status += cudaFree(BufferrecE1corn12_82[n]);
	status += cudaFree(BufferrecE2corn5_22[n]);
	status += cudaFree(BufferrecE2corn5_42[n]);
	status += cudaFree(BufferrecE2corn6_12[n]);
	status += cudaFree(BufferrecE2corn6_32[n]);
	status += cudaFree(BufferrecE2corn7_52[n]);
	status += cudaFree(BufferrecE2corn7_72[n]);
	status += cudaFree(BufferrecE2corn8_62[n]);
	status += cudaFree(BufferrecE2corn8_82[n]);
	#endif
	status += cudaFree(BufferrecE3corn1_32[n]);
	status += cudaFree(BufferrecE3corn1_42[n]);
	status += cudaFree(BufferrecE3corn2_12[n]);
	status += cudaFree(BufferrecE3corn2_22[n]);
	status += cudaFree(BufferrecE3corn3_52[n]);
	status += cudaFree(BufferrecE3corn3_62[n]);
	status += cudaFree(BufferrecE3corn4_72[n]);
	status += cudaFree(BufferrecE3corn4_82[n]);

	status += cudaFree(Imagegcov[n]);
	status += cudaFree(Imagegcon[n]);
	status += cudaFree(Imageconn[n]);
	status += cudaFree(Imagegdet[n]);
	
	//attention
	/*
	#if H_CORRECTION
	status += clReleaseKernel(kernel_hcor1[n]);
	status += clReleaseKernel(kernel_hcor2[n]);
	status += clReleaseKernel(kernel_hcor3[n]);
	#endif
	status += clReleaseKernel(kernel_fluxcalcprep[n]);
	status += clReleaseKernel(kernel_fluxcalc2D1[n]);
	status += clReleaseKernel(kernel_fluxcalc2D2[n]);
	status += clReleaseKernel(kernel_flux_ct1[n]);
	status += clReleaseKernel(kernel_flux_ct2[n]);
	status += clReleaseKernel(kernel_consttransport1[n]);
	status += clReleaseKernel(kernel_consttransport2[n]);
	status += clReleaseKernel(kernel_consttransport3[n]);
	status += clReleaseKernel(kernel_fix_flux[n]);
	status += clReleaseKernel(kernel_Utoprim[n]);
	status += clReleaseKernel(kernel_Utoprim2[n]);
	status += clReleaseKernel(kernel_Utoprim1[n]);
	status += clReleaseKernel(kernel_fixup[n]);
	status += clReleaseKernel(kernel_fixuputoprim[n]);
	status += clReleaseKernel(kernel_boundprim1[n]);
	status += clReleaseKernel(kernel_boundprim2[n]);
	status += clReleaseKernel(kernel_packsend1[n]);
	status += clReleaseKernel(kernel_packsend2[n]);
	status += clReleaseKernel(kernel_packsend3[n]);
	status += clReleaseKernel(kernel_packsend1flux[n]);
	status += clReleaseKernel(kernel_packsend2flux[n]);
	status += clReleaseKernel(kernel_packsend3flux[n]);
	status += clReleaseKernel(kernel_unpackreceive1[n]);
	status += clReleaseKernel(kernel_unpackreceive2[n]);
	status += clReleaseKernel(kernel_unpackreceive3[n]);
	status += clReleaseKernel(kernel_unpackreceive1flux[n]);
	status += clReleaseKernel(kernel_unpackreceive2flux[n]);
	status += clReleaseKernel(kernel_unpackreceive3flux[n]);
	status += clReleaseKernel(kernel_packsendaverage1[n]);
	status += clReleaseKernel(kernel_packsendaverage2[n]);
	status += clReleaseKernel(kernel_packsendaverage3[n]);
	status += clReleaseKernel(kernel_unpackreceivecoarse1[n]);
	status += clReleaseKernel(kernel_unpackreceivecoarse2[n]);
	status += clReleaseKernel(kernel_unpackreceivecoarse3[n]);
	status += clReleaseKernel(kernel_packsendfluxaverage1[n]);
	status += clReleaseKernel(kernel_packsendfluxaverage2[n]);
	status += clReleaseKernel(kernel_packsendfluxaverage3[n]);
	status += clReleaseKernel(kernel_packsend1E[n]);
	status += clReleaseKernel(kernel_packsend2E[n]);
	status += clReleaseKernel(kernel_packsend3E[n]);
	status += clReleaseKernel(kernel_unpackreceive1E[n]);
	status += clReleaseKernel(kernel_unpackreceive2E[n]);
	status += clReleaseKernel(kernel_unpackreceive3E[n]);
	status += clReleaseKernel(kernel_packsendEaverage1[n]);
	status += clReleaseKernel(kernel_packsendEaverage2[n]);
	status += clReleaseKernel(kernel_packsendEaverage3[n]);
	status += clReleaseKernel(kernel_packsendE1corn[n]);
	status += clReleaseKernel(kernel_packsendE2corn[n]);
	status += clReleaseKernel(kernel_packsendE3corn[n]);
	status += clReleaseKernel(kernel_packsendE1corncourse[n]);
	status += clReleaseKernel(kernel_packsendE2corncourse[n]);
	status += clReleaseKernel(kernel_packsendE3corncourse[n]);
	status += clReleaseKernel(kernel_unpackreceiveE1corn[n]);
	status += clReleaseKernel(kernel_unpackreceiveE2corn[n]);
	status += clReleaseKernel(kernel_unpackreceiveE3corn[n]);

	status += clReleaseKernel(kernel_packsendB1[n]);
	status += clReleaseKernel(kernel_packsendB2[n]);
	status += clReleaseKernel(kernel_packsendB3[n]);
	status += clReleaseKernel(kernel_packsendBaverage1[n]);
	status += clReleaseKernel(kernel_packsendBaverage2[n]);
	status += clReleaseKernel(kernel_packsendBaverage3[n]);
	status += clReleaseKernel(kernel_unpackreceiveB1[n]);
	status += clReleaseKernel(kernel_unpackreceiveB2[n]);
	status += clReleaseKernel(kernel_unpackreceiveB3[n]);
	*/
	if (status != 0) printf("Error in GPU_finish_1: %d \n", status);
}
