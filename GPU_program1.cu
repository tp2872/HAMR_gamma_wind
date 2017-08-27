
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
#define double double
#define double double

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
__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq);
__device__ int Utoprim_new_body(double U[], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[]);
__device__ int general_newton_raphson(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D);
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D);
__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq);
__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm);
__device__ double u_of_p2(double p);
__device__ double pressure_of_rho2(double rho0, double K_atm);
__device__ double dWdvsq_calc2(double vsq, double rho, double p);
__device__ int Utoprim_new_body2(double U[], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[], double K_atm);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ void validate_x2(double x[1], double x0[1]);
__device__ int general_newton_raphson2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ int Utoprim_1dvsq2fix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K);
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM2], double *f, double *df, int n, double D, double K_atm, double W_for_gnr2);
__device__ void raise_g(double vcov[], double gcon[][NDIM], double ucon[]);
__device__ void lower_g(double vcon[], double gcov[][NDIM], double ucov[]);
__device__ void ncov_calc(double gcon[][NDIM], double ncov[]);
__device__ void bcon_calc_g(double prim[], double ucon[], double ucov[], double ncov[], double bcon[]);
__device__ double pressure_rho0_u(double rho0, double u);
__device__ double pressure_rho0_w(double rho0, double w);
__device__ int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K);
__device__ int Utoprim_new_body3(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K_atm);
__device__ double pressure_of_rho3(double rho0, double K_atm);
__device__ double vsq_calc3(double W, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ int general_newton_raphson3(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old);
__device__ void func_1d_orig1(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old);
__device__ int gnr2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2);

__device__ int Utoprim_1dfix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K)
{
	double U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	double alpha, K_atm;

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


__device__ int Utoprim_new_body3(double U[NPR], double gcov[NDIM][NDIM],
	double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K_atm)
{

	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, j, retval, i_increase;
	double W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;
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

__device__ double pressure_of_rho3(double rho0, double K_atm)
{
	return(K_atm * pow(rho0, G_ATM));
}

__device__ double vsq_calc3(double W, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm)
{
	double Wsq, Xsq;
	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);
	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
}

__device__ int general_newton_raphson3(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old)
{
	double f, df, dx[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	double errx;
	int    n_iter, id,  i_extra, doing_extra;
	int   keep_iterating, i_increase;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;


	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		#if( USE_ISENTROPIC )   
		func_1d_orig1(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old);  /* returns with new dx, f, df */
		#endif

		errx = 0.;

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

__device__ int gnr2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2)
{
	double f, df, dx[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	double errx;
	int    n_iter, id, i_extra, doing_extra;
	int   keep_iterating;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		func_gnr2_rho(x, dx, resid, jac, &f, &df, n, D, K_atm, W_for_gnr2);  /* returns with new dx, f, df */

		errx = 0.;

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
__device__ void func_1d_orig1(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old)
{
	int ntries;
	double  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	double  t32, t33, t34, t38, t5, t51, t67, t8, W, x_rho[1], rho, rho_g;

	W = x[0];
	W_for_gnr2_old = W_for_gnr2;
	W_for_gnr2 = W;

	// get rho from NR:
	rho_g = x_rho[0] = rho_for_gnr2;

	ntries = 0;
	while ((gnr2(x_rho, 1, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2)) && (ntries++ < 10)) {
		rho_g *= 10.;
		x_rho[0] = rho_g;
	}

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


__device__ void raise_g(double ucov[NDIM], double gcon[NDIM][NDIM], double ucon[NDIM])
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

__device__ void lower_g(double ucon[NDIM], double gcov[NDIM][NDIM], double ucov[NDIM])
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

__device__ void ncov_calc(double gcon[NDIM][NDIM], double ncov[NDIM])
{
	double lapse;
	int i;

	lapse = sqrt(-1. / gcon[0][0]);
	ncov[0] = -lapse;
#pragma unroll 9	
	for (i = 1; i < NDIM; i++) {
		ncov[i] = 0.;
	}
	return;
}

__device__ void bcon_calc_g(double prim[NPR], double ucon[NDIM], double ucov[NDIM], double ncov[NDIM], double bcon[NDIM])
{
	double Bcon[NDIM];
	double u_dot_B;
	double gamma;
	int i;

	// Bcon = \mathcal{B}^\mu  of the paper:
	Bcon[0] = 0.;
	#pragma unroll 9	
	for (i = 1; i<NDIM; i++) Bcon[i] = -ncov[0] * prim[BCON1 + i - 1];

	u_dot_B = 0.;
	#pragma unroll 9	
	for (i = 0; i<NDIM; i++) u_dot_B += ucov[i] * Bcon[i];

	gamma = -ucon[0] * ncov[0];
	#pragma unroll 9	
	for (i = 0; i<NDIM; i++) bcon[i] = (Bcon[i] + ucon[i] * u_dot_B) / gamma;
}

__device__ int gamma_calc_g(double *pr, double gcov[NDIM][NDIM], double *gamma)
{
	double utsq;

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


__device__ double pressure_rho0_u(double rho0, double u)
{
	return((GAMMA - 1.)*u);
}

__device__ double pressure_rho0_w(double rho0, double w)
{
	return((GAMMA - 1.)*(w - rho0) / GAMMA);
}

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM2], double *f, double *df, int n, double D, double K_atm, double W_for_gnr2)
{
	double A, B, C, rho, W, B0;

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

__device__ int Utoprim_1dvsq2fix1(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K)
{
	double U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	double alpha;

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
	#pragma unroll 5
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
	#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}
	return(ret);
}

__device__ int Utoprim_new_body2(double U[NPR], double gcov[NDIM][NDIM],
	double gcon[NDIM][NDIM], double gdet, double prim[NPR], double K_atm)
{
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, gammasq, gamma, gtmp, W, utsq, vsq;
	int    i, j, retval;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

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

__device__ int general_newton_raphson2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm)
{
	double f, df, dx[NEWT_DIM2], x_old[NEWT_DIM2], resid[NEWT_DIM2],
		jac[NEWT_DIM2][NEWT_DIM2];
	double errx;
	int    n_iter, id, i_extra, doing_extra;
	double W, W_old, rho, p, u;

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

__device__ void validate_x2(double x[1], double x0[1])
{
	double small = 1.e-10;
	x[0] = (x[0] >= 1.0) ? (0.5*(x0[0] + 1.)) : x[0];
	x[0] = (x[0] <  -small) ? (0.5*x0[0]) : x[0];
	x[0] = fabs(x[0]);
	return;
}

__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm)
{
	double vsq, W, Wsq, W3, dWdvsq, fact_tmp, rho, p, u;
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

__device__ double pressure_of_rho2(double rho0, double K_atm)
{
	return(K_atm * pow(rho0, G_ATM));
}

__device__ double u_of_p2(double p)
{
	return(p / (GAMMA - 1.));
}

__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm)
{
	double gtmp;
	gtmp = (1. - vsq);
	*rho = D * sqrt(gtmp);
	*p = pressure_of_rho2(*rho, K_atm);
	*u = u_of_p2(*p);
	return((*rho + *u + *p) / gtmp);
}

__device__ double dWdvsq_calc2(double vsq, double rho, double p)
{
	return((GAMMA*(2. - G_ATM)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
}


__device__ int Utoprim_2d(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM],
	double gdet, double prim[NPR])
{
	double U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	double alpha;

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
	#pragma unroll 5
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
	#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	#if(DOKTOT )
	prim[KTOT] = U[KTOT] / U[RHO];
	#endif

	return(ret);
}

__device__ int Utoprim_new_body(double U[NPR], double gcov[NDIM][NDIM], double gcon[NDIM][NDIM], double gdet, double prim[NPR])
{
	double x_2d[NEWT_DIM];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov[NDIM], ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, j, n, retval, i_increase;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

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

__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq)
{
	double Wsq, Xsq;
	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);
	#if AMD
	return((fma(Wsq, Qtsq, QdotBsq * (Bsq + 2.*W))) / (Wsq*Xsq));
	#else
	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
	#endif
}

__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq)
{
	double vsq;
	double dv = 1.e-15;
	vsq = fabs(vsq_calc(x0, Bsq, Qtsq, QdotBsq)); // guaranteed to be positive 
	return((vsq > 1.) ? (1.0 - dv) : vsq);
}

__device__ void validate_x(double x[2], double x0[2])
{
	double dv = 1.e-15;

	/* Always take the absolute value of x[0] and check to see if it's too big:  */
	x[0] = fabs(x[0]);
	x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];

	x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
	x[1] = (x[1] > 1.) ? (1. - dv) : x[1];  /* if it's too big   */
	return;
}

__device__ int general_newton_raphson(double x[], int n,
	double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D)
{
	double f, df, dx[NEWT_DIM], x_old[NEWT_DIM];
	double resid[NEWT_DIM], jac[NEWT_DIM][NEWT_DIM];
	double errx;
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

__device__ void func_vsq(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D)
{
	double  W, vsq, Wsq, p_tmp, dPdvsq, dPdW, gtmp;
	double t11;
	double t16;
	double t18;
	double t2;
	double t21;
	double t23;
	double t24;
	double t25;
	double t3;
	double t35;
	double t36;
	double t4;
	double t40;
	double t9;

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
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

struct of_state {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
};

/*Declare other functions*/
__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q);
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon);
__device__ void bcon_calc(double *  pr, double *  ucon, double *  ucov, double *  bcon);
__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma);
__device__ void get_geometry(int N1, int N2, int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU);
__device__ double slope_lim(double y1, double y2, double y3, int lim);
__device__ void raise(double *  ucov, struct of_geom *  geom, double *  ucon);
__device__ void lower(double *  ucon, struct of_geom *  geom, double *  ucov);
__device__ void  primtoflux(double *  pr, struct of_state *  q, int dir, struct of_geom *  geom, double *  flux, double gam);
__device__ void primtoU(double *  pr, struct of_state *  q, struct of_geom *  geom, double *U, double gam);
__device__ void vchar(double *  pr, struct of_state *  q, struct of_geom *  geom, int js, double *  vmax, double *  vmin, double gam);
__device__ void mhd_calc(double *  pr, int dir, struct of_state *  q, double *  mhd, double gam);
__device__ void source(int N1, int N2, double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *dU, double Dt, double gam, const  double* __restrict__ Imageconn,
struct of_state *  q, double a, double r, int global_id);
__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,
	double a, double gam, double r, double Dt);
__device__ void inflow_check(int N1, int N2, double *  prim, int ii, int jj, int zz, int type, const  double* __restrict__ gcov1, const  double* __restrict__ gcon2, const  double* __restrict__ gdet3);
__device__ double bsq_calc(double *  pr, struct of_geom *  geom);
__device__ double NewtonRaphson(double start, size_t max_count, int dir, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  ucov, double *  bcon, struct of_geom *  geom, double E, double vasq, double csq);
__device__ double readImageDouble(int4 a);
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);

/* find relative 4-velocity from 4-velocity (both in code coords) */
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon)
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

__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
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

__device__ void primtoU(double *pr, struct of_state *q, struct of_geom *geom, double *U, double gam)
{
	primtoflux(pr, q, 0, geom, U, gam);
	return;
}

/* add in source terms to equations of motion */
__device__ void source(int N1, int N2, double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *  dU, double Dt, double gam,
	const  double* __restrict__ conn_GPU, struct of_state *  q, double a, double r, int test)
{
	double mhd[NDIM][NDIM];
	int fix_mem2 = LOCAL_WORK_SIZE - ((N2 + 2 * N2G)*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int k;
	//struct of_state q ;
	double conn;
	int global_id = icurr*(N2 + 2 * N2G) + jcurr;
	//get_state(ph, geom, &q) ;
	mhd_calc(ph, 0, q, mhd[0], gam);
	mhd_calc(ph, 1, q, mhd[1], gam);
	mhd_calc(ph, 2, q, mhd[2], gam);
	mhd_calc(ph, 3, q, mhd[3], gam);

	/* contract mhd stress tensor with connection */
	#pragma unroll 9	
	PLOOP dU[k] = 0.;
	
	#pragma unroll 4	
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

__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,
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

__device__ void primtoflux(double *  pr, struct of_state *  q, int dir, struct of_geom *  geom, double *  flux, double gam)
{
	int k;
	double mhd[NDIM];

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

	#pragma unroll 9
	PLOOP flux[k] *= geom->g;
}

__device__ void vchar(double *  pr, struct of_state *  q, struct of_geom *  geom, int js, double *  vmax, double *  vmin, double gam)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2, rho, u;
	double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	int j;

	#pragma unroll 4
	DLOOPA Acov[j] = 0.;
	Acov[js] = 1.;
	raise(Acov, geom, Acon);

	#pragma unroll 4
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
	#pragma unroll 4
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
__device__ void mhd_calc(double *pr, int dir, struct of_state *q, double *mhd, double gam)
{
	int j;
	double r, u, P, w, bsq, eta, ptot;

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
	#pragma unroll 4
	DLOOPA mhd[j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
	#else
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
	#endif
}

__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q)
{
	/* get ucon */
	ucon_calc(pr, geom, q->ucon);
	lower(q->ucon, geom, q->ucov);
	bcon_calc(pr, q->ucon, q->ucov, q->bcon);
	lower(q->bcon, geom, q->bcov);

	return;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ void raise(double *  ucov, struct of_geom *  geom, double *  ucon)
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
__device__ void lower(double *  ucon, struct of_geom *  geom, double *  ucov)
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
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon)
{
	double alpha, gamma;
	double beta[NDIM];
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

__device__ void bcon_calc(double *  pr, double *  ucon, double *  ucov, double *  bcon)
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

__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma)
{
	double qsq;
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
__device__ void get_geometry(int N1, int N2, int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU)
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

__device__ void inflow_check(int N1, int N2, double *  pr, int ii, int jj, int zz, int type, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	struct of_geom geom;
	double ucon[NDIM];
	int j, k;
	double alpha, beta1, gamma, vsq;
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
		
		#pragma unroll 4
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

__device__  double slope_lim(double y1, double y2, double y3, int dir)
{
	double Dqm, Dqp, Dqc, s;
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

__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout)
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
__device__ double bsq_calc(double *  pr, struct of_geom *  geom)
{
	struct of_state q;
	get_state(pr, geom, &q);
	return(dot(q.bcon, q.bcov));
}


__global__ void fluxcalcprep(int N1, int N2, int N3, double *   F, double *  dq, double *  p, int dir, int lim, int number, double *  V)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (N3 + 2 * D3)*(N2 + 2 * D2);
	int zcurr = (global_id % (isize)) % (N3 + 2 * D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + 2 * D3);
	int icurr = (global_id - (jcurr*(N3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3;
	jcurr += N2G - 1;
	icurr += N1G - 1;
	isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int k = 0;
	if (global_id<(N1 + 2 * D1) * (N2 + 2 * D2) * (N3 + 2 * D3)) k = 1;
	global_id = isize*icurr + (N3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }
	if (k == 1){
		#if(PPM)
		int jsize = N3 + 2 * N3G;
		double x1, x2, x3, x4, x5, temp[1], result[1];
		if (number == 1){
			#pragma unroll 9	
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
			#pragma unroll 9	
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
		#else
		#pragma unroll 9	
		for (k = 0; k<NPR; k++){
			dq[k*(ksize)+global_id] = slope_lim(p[k*(ksize)+global_id - idel*isize - jdel*(N3 + 2 * N3G) - zdel], p[k*(ksize)+global_id], p[k*(ksize)+global_id + idel*isize + jdel*(N3 + 2 * N3G) + zdel], 0);
		}
		#endif
	}
}
#include <stdio.h>

__global__ void fluxcalc2D2(int N1, int N2, int N3, double *  F, double *  dq, double *  pv, double *  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir,
	double gam, double cour, double*  dtij, int POLE_1, int POLE_2, double* storage1, double* storage2, double* storage3, double* storage4, double dx_1, double dx_2, double dx_3)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;
	int group_id = blockIdx.x;
	int local_size = blockDim.x;
	__shared__ double local_dtij[LOCAL_WORK_SIZE];

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
	double cmax_r, cmin_r, cmax, cmin;
	double ctop;
	double temp3[NPR], temp4[NPR];
	double cmax_l, cmin_l;
	double p[NPR];
	double temp1[NPR], temp2[NPR];
	struct of_geom geom;
	struct of_state state;
	local_dtij[local_id] = 1.e9;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx_1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx_2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx_3; }
	if (k == 1){
		get_geometry(N1, N2, icurr, jcurr, zcurr, face, &geom, gcov, gcon, gdet);

		#if(PPM || LEER)
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			p[k] = dq[k*(ksize)+global_id];
		}
		#else
		#pragma unroll 9
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
			#pragma unroll 9
		for (k = 0; k< NPR; k++){
			p[k] = dq[k*(ksize)+global_id];
		}
		#else
		#pragma unroll 9
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
		#pragma unroll 9	
		for (k = 0; k<NPR; k++){
			F[k*(ksize)+global_id] = HLLF*((cmax*temp1[k] +
				cmin*temp3[k] - cmax*cmin*(temp4[k] - temp2[k])) / (cmax + cmin + SMALL))
				+ LAXF*(0.5*(temp1[k]
				+ temp3[k] - ctop*(temp4[k] - temp2[k])));
		}
		if (icurr == 50 && jcurr == 50 && zcurr == 50)printf("rhogpu: %f %f %f \n", dq[global_id], cmax_l, p[0]);

		cmax = MY_MAX(cmax, cmin);
		local_dtij[local_id] = factor / cmax;
	}
	__syncthreads();
	for (i = local_size / 2; i>1; i = i / 2){
		if (local_id<i){
			local_dtij[local_id] = MY_MIN(local_dtij[local_id], local_dtij[local_id + i]);
		}
		__syncthreads();
	}
	if (local_id == 0){
		dtij[group_id] = MY_MIN(local_dtij[0], local_dtij[1]);
	}
}

__global__ void fix_flux(int N1, int N2, int N3, double *  F1, double *  F2, double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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
				#pragma unroll 9	
				PLOOP F2[k*(ksize)+icurr*isize + N2G*(N3 + 2 * N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll 9	
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
				#pragma unroll 9	
				PLOOP F2[k*(ksize)+icurr*isize + (N2 + N2G)*(N3 + 2 * N3G) + zcurr] = 0.;
				#endif	
				#pragma unroll 9	
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

__global__ void consttransport1(int N1, int N2, int N3, double *  pb_i, double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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

__global__ void consttransport2(int N1, int N2, int N3, double *  emf, double *  E_cent, double *  F1, double *  F2, double *  F3,
	double *  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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

__global__ void consttransport3(int N1, int N2, int N3, double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	double *  E_corn, double Dt)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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
		psf[global_id] = psi[global_id] - Dt / dx_2*(E_corn[3 * ksize + global_id + (N3 + 2 * N3G)] - E_corn[3 * ksize + global_id]) / gdet_GPU[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];;
		#if(N3G>0)
		psf[global_id] += Dt / dx_3*(E_corn[2 * ksize + global_id + D3] - E_corn[2 * ksize + global_id]) / gdet_GPU[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G + D2  && zcurr<N3 + N3G){
		psf[1 * ksize + global_id] = psi[1 * ksize + global_id] + Dt / dx_1*(E_corn[3 * ksize + global_id + isize] - E_corn[3 * ksize + global_id]) / gdet_GPU[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];;
		#if(N3G>0)
		psf[1 * ksize + global_id] += -Dt / dx_3*(E_corn[1 * ksize + global_id + D3] - E_corn[1 * ksize + global_id]) / gdet_GPU[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];;
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G && zcurr<N3 + N3G + D3){
		#if(N3G>0)
		psf[2 * ksize + global_id] = psi[2 * ksize + global_id] - Dt / dx_1*(E_corn[2 * ksize + global_id + isize] - E_corn[2 * ksize + global_id]) / gdet_GPU[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];;
		psf[2 * ksize + global_id] += Dt / dx_2*(E_corn[1 * ksize + global_id + (N3 + 2 * N3G)] - E_corn[1 * ksize + global_id]) / gdet_GPU[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr];
		#endif
	}
}

__global__ void flux_ct1(int N1, int N2, int N3, double *  F1, double *  F2, double *  F3, double *  emf)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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

__global__ void flux_ct2(int N1, int N2, int N3, double *  F1, double *  F2, double *  F3, double *  emf)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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

__global__ void fixup(int N1, int N2, int N3, double* pi_i, double* pb_i, double* pf_i, double *  psf,
	double *  F1, double *  F2, double *  F3, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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
	int  dofloor = 0, m;
	double r, uuscal, rhoscal, rhoflr, uuflr;
	double f, gamma, bsq;
	double pf[NPR], pf_prefloor[NPR], dU[NPR], U[NPR];
	double trans, betapar, betasq, betasqmax, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut;
	double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];

	if (k == 1){
		get_geometry(N1, N2, icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		if (full_step == 0){
			for (k = 0; k<NPR; k++){
				pf[k] = pi_i[k*(ksize)+global_id];
			}
			get_state(pf, &geom, &q);
			primtoU(pf, &q, &geom, U, gam);
			#pragma unroll 9	
			for (k = 0; k<NPR; k++){
				pi_i[k*(ksize)+global_id] = U[k];
			}
		}
		else{
			#pragma unroll 9	
			for (k = 0; k<NPR; k++){
				U[k] = pi_i[k*(ksize)+global_id];
			}
			#pragma unroll 9	
			for (k = 0; k<NPR; k++){
				pf[k] = pb_i[k*(ksize)+global_id];
			}
			if (full_step == 1){
				get_state(pf, &geom, &q);
			}
		}

		#pragma unroll 9	
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

		source(N1, N2, pf, &geom, icurr, jcurr, zcurr, dU, Dt, gam, conn, &q, a, radius[icurr], global_id);

		#pragma unroll 9	
		for (k = 0; k< NPR; k++){
			U[k] += Dt*(dU[k]);
		}

		#if(STAGGERED)
		U[B1] = (psf[0 * ksize + global_id] * gdet[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[0 * ksize + global_id + isize] * gdet[FACE1*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + (icurr+D1)*(N2 + 2 * N2G) + jcurr]) / 2.0;
		U[B2] = (psf[1 * ksize + global_id] * gdet[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[1 * ksize + global_id + (N3 + 2 * N3G)] * gdet[FACE2*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + (jcurr+D2)]) / 2.0;
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id] * gdet[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr] + psf[2 * ksize + global_id + D3] * gdet[FACE3*((N2 + 2 * N2G)*(N1 + 2 * N1G) + fix_mem2) + icurr*(N2 + 2 * N2G) + jcurr]) / 2.0;
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
		#pragma unroll 9
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
			#pragma unroll 4
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

			#pragma unroll 4
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
			double dpf[NPR], U_prefloor[NPR],Xtransone_over_ucondr;
			#pragma unroll 9
			PLOOP dpf[k] = pf[k] - pf_prefloor[k];

			//compute the conserved quantity associated with floor addition
			get_state(dpf, &geom, &q);
			primtoU(dpf, &q, &geom, dU, gam);

			//compute the prefloor conserved quantity
			get_state(pf_prefloor, &geom, &q);
			primtoU(pf_prefloor, &q, &geom, U_prefloor, gam);

			//add U_added to the current conserved quantity
			#pragma unroll 9
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
		#pragma unroll 9	
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

__global__ void fixuputoprim(int N1, int N2, int N3, double *  pv, int *  pflag, int *  failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int zcurr = (global_id % (isize)) % (N3 + 2 * N3G);
	int jcurr = ((global_id - zcurr) % (isize)) / (N3 + 2 * N3G);
	int icurr = (global_id - (jcurr*(N3 + 2 * N3G) + zcurr)) / (isize);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	/* Fix the interior points first */
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<N1 + N1G && jcurr<N2 + N2G && zcurr<N3 + N3G) {
		if (pflag[global_id] != 0) {
			/* if nothing better to do, then leave densities and B-field unchanged, set v^i = 0 */
			pv[0 * (ksize)+global_id] = 0.5*(AVG4_1(pv, icurr, jcurr, zcurr, N1, N2, N3, 0) + AVG4_2(pv, icurr, jcurr, zcurr, N1, N2, N3, 0));
			pv[1 * (ksize)+global_id] = 0.5*(AVG4_1(pv, icurr, jcurr, zcurr, N1, N2, N3, 1) + AVG4_2(pv, icurr, jcurr, zcurr, N1, N2, N3, 1));
			pv[2 * (ksize)+global_id] = pv[3 * (ksize)+global_id] = pv[4 * (ksize)+global_id] = 0.;
			pflag[global_id] = 0;                /* The cell has been fixed so we can use it for interpolation elsewhere */
		}
	}
}

__global__ void boundprim1(int N1, int N2, int N3, double *   pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double *  ps)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (N3 + 2 * N3G)*(N2 + 2 * N2G);
	int k;
	int zcurr = global_id % (N3 + 2 * N3G);
	int jcurr = (global_id - zcurr) / (N3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(N1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(N1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];

	// inner r boundary condition: u, gdet extrapolation 
	if (jcurr >= 0 && jcurr<N2 + 2 * N2G && zcurr >= 0 && zcurr<N3 + 2 * N3G && NBR_4 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim5[k] = pv[k*(ksize)+N1G*isize + global_id];
		}

		#pragma unroll 9
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
		#pragma unroll 9
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
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim6[k] = pv[k*(ksize)+(N1 + N1G - 1)*isize + global_id];
		}

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}

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

		#pragma unroll 9
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

__global__ void boundprim2(int N1, int N2, int N3, double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
{
	int j, jref, k;
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
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
		#pragma unroll 9
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
		#pragma unroll 9
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