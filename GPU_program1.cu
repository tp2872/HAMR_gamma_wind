#include "config.h"

/*Declerations of functions for Utoprim*/
__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq);
__device__ int Utoprim_new_body(double U[], double gcov[10], double gcon[10], double gdet, double prim[]
                                #if (DOHELM)
                                , const  double* __restrict__ gpu_eos_table
                                #endif
                                );
__device__ int general_newton_raphson(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
                                        #if (DOHELM)
                                        , const  double* __restrict__ gpu_eos_table
                                        #endif
                                    );
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                         );
__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq);
__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm);
__device__ double dWdvsq_calc2(double vsq, double rho, double p);
__device__ int Utoprim_new_body2(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double K_atm);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ void validate_x2(double x[1], double x0[1]);
__device__ int general_newton_raphson2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ int Utoprim_1dvsq2fix1(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K);
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, int n, double D, double K_atm, double W_for_gnr2);
__device__ int Utoprim_1dfix1(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K);
__device__ int Utoprim_new_body3(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K_atm);
__device__ double vsq_calc3(double W, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ int general_newton_raphson3(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old);
__device__ void func_1d_orig1(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM_1], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old);
__device__ int gnr2(double x[], int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2);
__device__ int Utoprim_NM_calc(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR]
                                #if (DOHELM)
                                , const  double* __restrict__ gpu_eos_table
                                #endif
                               );
__device__ int Utoprim_NM(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR]
                            #if (DOHELM)
                            , const  double* __restrict__ gpu_eos_table
                            #endif
                            );

/*Matrix Inversion*/
__device__ int invert_matrix(double Am[][NDIM], double Aminv[][NDIM]);
__device__ int LU_decompose(double A[][NDIM], int permute[]);
__device__ void LU_substitution(double A[][NDIM], double B[], int permute[]);

/*Declare other functions*/
__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q);
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon);
__device__ void bcon_calc(double *  pr, double *  ucon, double *  ucov, double *  bcon);
__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma);
__device__ void get_geometry(int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU);
__device__ void get_trans(int ii, int jj, int zz, int kk, struct of_trans * trans, const  double* __restrict__ Mud_GPU, const  double* __restrict__ Mud_inv_GPU);
__device__ double slope_lim(double y1, double y2, double y3, int lim);
__device__ void raise(double ucov[NDIM], double gcon[10], double ucon[NDIM]);
__device__ void lower(double ucon[NDIM], double gcov[10], double ucov[NDIM]);
__device__ void primtoflux(double *  pr, struct of_state *  q, int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin, double gam
                            #if (DOHELM)
                            , const  double* __restrict__ gpu_eos_table
                            #endif
                           );
__device__ void primtoU(double *  pr, struct of_state *  q, struct of_geom *  geom, double *U, double gam
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                        );
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *dU, double Dt, double gam, const  double* __restrict__ conn,struct of_state *  q, double a, double r
						#if (DOHELM)
						, const  double* __restrict__ gpu_eos_table
						#endif
);
__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,	double a, double gam, double r, double Dt);
__device__ void inflow_check(double *  prim, int ii, int jj, int zz, int type, const  double* __restrict__ gcov1, const  double* __restrict__ gcoBS_2, const  double* __restrict__ gdet3);
__device__ double bsq_calc(double *  pr, struct of_geom *  geom);
__device__ double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double readImageDouble(int4 a);
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd
                            #if (DOHELM)
                            , const  double* __restrict__ gpu_eos_table
                            #endif
                         );



/*************************************************************************/
/*************************************************************************

invert_matrix():

Uses LU decomposition and back substitution to invert a matrix
A[][] and assigns the inverse to Ainv[][].  This routine does not
destroy the original matrix A[][].

Returns (1) if a singular matrix is found,  (0) otherwise.

*************************************************************************/


__device__ int invert_matrix(double Am[][NDIM], double Aminv[][NDIM])
{
	int i, j;
	int n = NDIM;
	int permute[NDIM];
	double dxm[NDIM], Amtmp[NDIM][NDIM];

	for (i = 0; i < NDIM*NDIM; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose(Amtmp, permute) != 0) {
		//fprintf(stderr, "invert_matrix(): singular matrix encountered! \n");
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

/*************************************************************************/
/*************************************************************************

LU_decompose():

Performs a LU decomposition of the matrix A using Crout's method
with partial implicit pivoting.  The exact LU decomposition of the
matrix can be reconstructed from the resultant row-permuted form via
the integer array permute[]

The algorithm closely follows ludcmp.c of "Numerical Recipes
in C" by Press et al. 1992.

This will be used to solve the linear system  A.x = B

Returns (1) if a singular matrix is found,  (0) otherwise.

*************************************************************************/



__device__ int LU_decompose(double A[][NDIM], int permute[])
{
	double row_norm[NDIM];

	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */

	double  absmax, maxtemp, mintemp;

	int i, j, k, max_row;
	int n = NDIM;


	max_row = 0;

	/* Find the maximum elements per row so that we can pretend later
	we have unit-normalized each equation: */

	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);

			if (maxtemp > absmax) {
				absmax = maxtemp;
			}
		}

		/* Make sure that there is at least one non-zero element in this row: */
		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}


	/* The following the calculates the matrix composed of the sum
	of the lower (L) tridagonal matrix and the upper (U) tridagonal
	matrix that, when multiplied, form the original maxtrix.
	This is what we call the LU decomposition of the maxtrix.
	It does this by a recursive procedure, starting from the
	upper-left, proceding down the column, and then to the next
	column to the right.  The decomposition can be done in place
	since element {i,j} require only those elements with {<=i,<=j}
	which have already been computed.
	See pg. 43-46 of "Num. Rec." for a more thorough description.
	*/

	/* For each of the columns, starting from the left ... */
	for (j = 0; j < n; j++) {

		/* For each of the rows starting from the top.... */

		/* Calculate the Upper part of the matrix:  i < j :   */
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		/* Calculate the Lower part of the matrix:  i <= j :   */

		for (i = j; i < n; i++) {

			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			/* Find the maximum element in the column given the implicit
			unit-normalization (represented by row_norm[i]) of each row:
			*/
			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}

		}

		/* Swap the row with the largest element (of column j) with row_j.  absmax
		This is the partial pivoting procedure that ensures we don't divide
		by 0 (or a small number) when we solve the linear system.
		Also, since the procedure starts from left-right/top-bottom,
		the pivot values are chosen from a pool involving all the elements
		of column_j  in rows beneath row_j.  This ensures that
		a row  is not permuted twice, which would mess things up.
		*/
		if (max_row != j) {

			/* Don't swap if it will send a 0 to the last diagonal position.
			Note that the last column cannot pivot with any other row,
			so this is the last chance to ensure that the last two
			columns have non-zero diagonal elements.
			*/

			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}

				/* Don't forget to swap the normalization factors, too...
				but we don't need the jth element any longer since we
				only look at rows beneath j from here on out.
				*/
				row_norm[max_row] = row_norm[j];
			}
		}

		/* Set the permutation record s.t. the j^th element equals the
		index of the row swapped with the j^th row.  Note that since
		this is being done in successive columns, the permutation
		vector records the successive permutations and therefore
		index of permute[] also indexes the chronology of the
		permutations.  E.g. permute[2] = {2,1} is an identity
		permutation, which cannot happen here though.
		*/

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}


		/* Normalize the columns of the Lower tridiagonal part by their respective
		diagonal element.  This is not done in the Upper part because the
		Lower part's diagonal elements were set to 1, which can be done w/o
		any loss of generality.
		*/
		if (j != (n - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < n; i++) {
				A[i][j] *= maxtemp;
			}
		}

	}

	return(0);

	/* End of LU_decompose() */

}


/************************************************************************
/************************************************************************

LU_substitution():

Performs the forward (w/ the Lower) and backward (w/ the Upper)
substitutions using the LU-decomposed matrix A[][] of the original
matrix A' of the linear equation:  A'.x = B.  Upon entry, A[][]
is the LU matrix, B[] is the source vector, and permute[] is the
array containing order of permutations taken to the rows of the LU
matrix.  See LU_decompose() for further details.

Upon exit, B[] contains the solution x[], A[][] is left unchanged.

************************************************************************/


__device__ void LU_substitution(double A[][NDIM], double B[], int permute[])
{
	int i, j;
	int n = NDIM;
	double tmpvar, tmpvar2;


	/* Perform the forward substitution using the LU matrix.
	*/
	for (i = 0; i < n; i++) {

		/* Before doing the substitution, we must first permute the
		B vector to match the permutation of the LU matrix.
		Since only the rows above the currrent one matter for
		this row, we can permute one at a time.
		*/
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}


	/* Perform the backward substitution using the LU matrix.
	*/
	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}

	/* End of LU_substitution() */

}

__device__ void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double *F);
__device__ void vchar_FT(double *pr, double ucon[NDIM], double bcon[NDIM], int dir, double *vmax, double *vmin);
__device__ void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *vmax, double *vmin
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                      );
__device__ void primtoflux_FT(double *pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR]);
__device__ void calc_HLLC(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ void calc_HLLD(int dir, double cmin_roe, double cmax_roe, double int_velocity, double l_ucon[NDIM], double r_ucon[NDIM], double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ double calc_HLLD_pres(int dir, int *fail_HLLC, int *fail_HLLD, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double *eta_l, double *eta_r, double *w_al, double *w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
__device__ void calc_HLLD_state(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double ptot, double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double eta_l, double eta_r, double w_al, double w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
__device__ void check_HLLD_par(int dir, int * fail_HLLD, double cmin_roe, double cmax_roe, double ptot, double w_al, double w_ar, double eta_l, double eta_r, double vcon_cl[NDIM], double vcon_cr[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double B_c[NDIM]);
__device__ double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double vcon_cl[NDIM], double vcon_cr[NDIM], double *eta_l, double *eta_r, double  *w_al, double *w_ar);

// EOS functions
#if (DOHELM)
__device__ void interp_eostable(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele);
__device__ void eos_helm (const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double abar, double zbar, double *pres, double *ener, double *dpresdt, double *denerdt, double *dpresdd, double *cs2);
__device__ void eos_mode_rhou_pres (const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres);
__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres, double *cs2);
__device__ void eos_mode_rhow_pres_dpdrho_dpde_d (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *dpdrho, double *dpde_d);
__device__ void eos_mode_rhow_pres_u (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *u);
__device__ void eos_mode_rhotemp_pres_min (const  double* __restrict__ gpu_eos_table, double den, double *pres);
__device__ void eos_mode_rhopres_u (const  double* __restrict__ gpu_eos_table, double den, double p_goal, double *u);
#endif

__device__ int Utoprim_NM(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR]
                            #if (DOHELM)
                            , const  double* __restrict__ gpu_eos_table
                            #endif
                          )
{

	double U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	double alpha;


	if (U[0] <= 0.) {
		return(-100);
	}

	/* First update the primitive B-fields */
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	/* Set the geometry variables: */
	alpha = 1.0 / sqrt(-gcon[0]);

	/* Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	U_tmp[RHO] = alpha * U[RHO] / gdet; //W=ucon[0]*alpha
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	for (i = UTCON1; i <= UTCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}
	for (i = BCON1; i <= BCON3; i++) {
		U_tmp[i] = alpha * U[i] / gdet;
	}

	/* Transform the PRIMITIVE variables into the new system */
	for (i = 0; i < BCON1; i++) {
		prim_tmp[i] = prim[i];
	}
	for (i = BCON1; i <= BCON3; i++) {
		prim_tmp[i] = alpha*prim[i];
	}

	ret = Utoprim_NM_calc(U_tmp, gcov, gcon, gdet, prim_tmp
                            #if (DOHELM)
                            , gpu_eos_table
                            #endif
                          );

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	prim[KTOT] = U[KTOT] / U[RHO];

	return(ret);

}

__device__ int Utoprim_NM_calc(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR]
                                #if (DOHELM)
                                , const  double* __restrict__ gpu_eos_table
                                #endif
                               )
{
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u,  w,  gamma,   vsq;
	double Bsq, QdotBsq, Qtsq, Qdotn;

	int i;

	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower(Bcon, gcov, Bcov);
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);

	Bsq = 0.;
	/*#pragma ivdepreduction(+:Bsq)*/
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	//#pragma ivdepreduction(+:QdotB)
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;

	Qsq = 0.;

	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn*Qdotn;

	//Start inversion scheme AKA Newman et al
	double a, d, z, phi, R, Wsq, p_array[3], epsilon, p_old, p_new;
	int iter = 0;
	int iter_tot = 0;
	int set_variables = 0;

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    // -- to get min. pressure for a given density, set T = T_min = 1e4 K
    double xpres;
    eos_mode_rhotemp_pres_min (gpu_eos_table, prim[RHO], &xpres);
    p_array[0] = xpres;
    #else
    // 2. Ideal gas EOS
	p_array[0] = (GAMMA - 1.)*prim[UU];
    #endif

	p_new = p_array[0];
	d = 0.5*(Qtsq*Bsq - QdotBsq);
	if (d < 0.0) return(1);

	do{
		set_variables = 0;
		p_old = p_array[iter%3];
		a = -Qdotn + p_new + 0.5*Bsq;
		if (a < pow(27.*d/4.,1./3.)) return 1;
		phi = acos(1. / a*sqrt((27.*d) / (4.*a)));
		epsilon = a / 3. - 2. / 3.*a*cos(2. / 3.*phi + 2. / 3.*M_PI);
		z = epsilon - Bsq;

		vsq = (Qtsq*z*z + QdotBsq*(Bsq + 2. * z)) / (z*z*pow(Bsq + z, 2.));

		// DANAT: add this - therefore rho0 is nan
		if (fabs(vsq) < 1e-15) vsq = 0.0;
		if (fabs(vsq) >= 1.0) return(1);

		Wsq = 1. / (1. - vsq);
		w = z * (1. - vsq);
		gamma = sqrt(Wsq);
		rho0 = U[RHO] / gamma; //Watch out you may need this for a more complicated EOS

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        double xpres;
        eos_mode_rhow_pres_u (gpu_eos_table, rho0, w, &xpres, &u);
        #else
        // 2. Ideal gas EOS
		u = (w - rho0) / GAMMA;
        #endif

		iter++;
		iter_tot++;

        #if (DOHELM)
        // 1. Helmholtz EOS
        p_array[iter % 3] = xpres;
        #else
        // 2. Ideal gas EOS
		p_array[iter % 3] = (GAMMA - 1.)*u;
        #endif

		p_new = p_array[iter % 3];
		if (iter >= 2) {
			R = (p_array[iter % 3] - p_array[(iter - 1) % 3]) / (p_array[(iter - 1) % 3] - p_array[(iter - 2) % 3]);

			if (R<1. && R>0.) {
				set_variables = 1;
				p_new = p_array[(iter - 1) % 3] + (p_array[iter % 3] - p_array[(iter - 1) % 3]) / (1. - R);
				iter = 0.;
				p_array[iter % 3] = p_new;
			}
		}
	} while (fabs(p_new - p_old) > 0.01*NEWT_TOL*(p_new + p_old) && iter_tot < MAX_NEWT_ITER);

	if (set_variables == 1){
		a = -Qdotn + p_new + 0.5*Bsq;
		phi = acos(1. / a*sqrt((27.*d) / (4.*a)));
		epsilon = a / 3. - 2. / 3.*a*cos(2. / 3.*phi + 2. / 3.*M_PI);
		z = epsilon - Bsq;

		vsq = (Qtsq*z*z + QdotBsq*(Bsq + 2. * z)) / (z*z*pow(Bsq + z, 2.));
		Wsq = 1. / (1. - vsq);
		w = z * (1. - vsq);
		gamma = sqrt(Wsq);
		rho0 = U[RHO] / gamma; //Watch out you may need this for a more complicated EOS

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        eos_mode_rhow_pres_u (gpu_eos_table, rho0, w, &p_new, &u);
        #else
        // 2. Ideal gas EOS
		u = (w - rho0) / GAMMA;
		p_new = (GAMMA - 1.)*u;
        #endif
	}
	if (iter_tot >= MAX_NEWT_ITER || p_new < 0.0 || rho0<0.0 || vsq>=1.0 || vsq<0. || z <= 0. || z > W_TOO_BIG ||gamma>GAMMAMAX || gamma<1.){
		return(1);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (z + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / z);

	/* set field components */
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	/* done! */
	return(0);
}


__device__ int Utoprim_1dfix1(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K)
{
	double U_tmp[NPR], prim_tmp[NPR];
	int i, ret;
	double alpha, K_atm;

	if (U[0] <= 0.) {
		return(-100);
	}
	K_atm = K;

	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	alpha = 1.0 / sqrt(-gcon[0]);

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

	#if(DOKTOT )
	prim[KTOT] = U[KTOT] / U[RHO];
	#endif

	return(ret);
}


__device__ int Utoprim_new_body3(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K_atm)
{

	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, retval, i_increase;
	double W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;
	retval = 0;

	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	Bcon[0] = 0.;
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower(Bcon, gcov, Bcov);

	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);


	Bsq = 0.;
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov=-sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
	utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
	utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
	utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
	utsq += 2*gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
	utsq += gcov[9] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 3 - 1]; //1,2


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
	p = K_atm * pow(rho0, G_ATM);
	u = p / (GAMMA - 1.);
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

	p = K_atm * pow(rho0, G_ATM);
	u = p / (GAMMA - 1.);

	if ((rho0 <= 0.) || (u <= 0.)) {
		retval = 5;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	for (i = 1; i<4; i++) Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];
	return(retval);
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
	double f, df, dx[NEWT_DIM_1], resid[NEWT_DIM_1],
		jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int    n_iter, id, i_extra, doing_extra;
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
	double f, df, dx[NEWT_DIM_1], resid[NEWT_DIM_1],
		jac[NEWT_DIM_1][NEWT_DIM_1];
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
	double jac[][NEWT_DIM_1], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old)
{
	int ntries;
	double  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	double  t32, t33, t34, t38, t5, t51, t67, t8, W, x_rho[1], rho, rho_g;

	W = x[0];
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

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(double x[], double dx[], double resid[],
	double jac[][NEWT_DIM_1], double *f, double *df, int n, double D, double K_atm, double W_for_gnr2)
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

__device__ int Utoprim_1dvsq2fix1(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double K)
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
	alpha = 1.0 / sqrt(-gcon[0]);

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

	#if(DOKTOT )
	prim[KTOT] = U[KTOT] / U[RHO];
	#endif

	return(ret);
}

__device__ int Utoprim_new_body2(double U[NPR], double gcov[10],
	double gcon[10], double gdet, double prim[NPR], double K_atm)
{
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, gammasq, gamma, gtmp, W, utsq, vsq;
	int    i, retval;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

	// Assume ok initially:
	retval = 0;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower(Bcon, gcov, Bcov);
	#pragma unroll 4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	Qsq = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	Qtsq = Qsq + Qdotn*Qdotn;

	D = U[RHO];

	/* calculate W from last timestep and use  for guess */
	utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
	utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
	utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
	utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
	utsq += 2 * gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
	utsq += gcov[9] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 3 - 1]; //1,2

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
	p = (GAMMA - 1.)*u;

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
	double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1],
		jac[NEWT_DIM_1][NEWT_DIM_1];
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

__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm)
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

__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm)
{
	double gtmp;
	gtmp = (1. - vsq);
	*rho = D * sqrt(gtmp);
	*p = K_atm * pow(*rho, G_ATM);
	*u = *p / (GAMMA - 1.);
	return((*rho + *u + *p) / gtmp);
}

__device__ double dWdvsq_calc2(double vsq, double rho, double p)
{
	return((GAMMA*(2. - G_ATM)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
}


__device__ int Utoprim_2d(double U[NPR], double gcov[10], double gcon[10],
	double gdet, double prim[NPR]
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    )
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
	alpha = 1.0 / sqrt(-gcon[0]);

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

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp
                            #if (DOHELM)
                            , gpu_eos_table
                            #endif
                           );

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
#include <stdio.h>

__device__ int Utoprim_new_body(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR]
                                #if (DOHELM)
                                , const  double* __restrict__ gpu_eos_table
                                #endif
                                )
{
	double x_2d[NEWT_DIM_2];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, n, retval, i_increase;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

	n = NEWT_DIM_2;

	// Assume ok initially:
	retval = 0;
	//#pragma unroll 3
	//for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

	// Calculate various scalars (Q.B, Q^2, etc)  from the conserved variables:
	Bcon[0] = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bcon[i] = U[BCON1 + i - 1];

	lower(Bcon, gcov, Bcov);
	#pragma unroll 4
	for (i = 0; i<4; i++) Qcov[i] = U[QCOV0 + i];
	raise(Qcov, gcon, Qcon);

	Bsq = 0.;
	#pragma unroll 3
	for (i = 1; i<4; i++) Bsq += Bcon[i] * Bcov[i];

	QdotB = 0.;
	#pragma unroll 4
	for (i = 0; i<4; i++) QdotB += Qcov[i] * Bcon[i];
	QdotBsq = QdotB*QdotB;

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov;

	Qsq = 0.;
	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];

	#if AMD
	Qtsq = fma(Qdotn, Qdotn, Qsq);
	#else
	Qtsq = Qsq + Qdotn*Qdotn;
	#endif
	D = U[RHO];

	/* calculate W from last timestep and use for guess */
	utsq = gcov[4] * prim[UTCON1 + 1 - 1] * prim[UTCON1 + 1 - 1]; //1,1
	utsq += 2.*gcov[5] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 1 - 1]; //1,2
	utsq += 2.*gcov[6] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 1 - 1]; //1,3
	utsq += gcov[7] * prim[UTCON1 + 2 - 1] * prim[UTCON1 + 2 - 1]; //2,2
	utsq += 2 * gcov[8] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 2 - 1]; //2,3
	utsq += gcov[9] * prim[UTCON1 + 3 - 1] * prim[UTCON1 + 3 - 1]; //3,3

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

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    eos_mode_rhou_pres (gpu_eos_table, rho0, u, &p);
    #else
    // 2. Ideal gas EOS
	p = (GAMMA - 1.)*u;
    #endif

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
	retval = general_newton_raphson(x_2d, n, Bsq, Qtsq, QdotBsq, Qdotn, D
                                    #if (DOHELM)
                                    , gpu_eos_table
                                    #endif
                                    );

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

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    eos_mode_rhow_pres_u (gpu_eos_table, rho0, w, &p, &u);
    #else
    // 2. Ideal gas EOS
	p = (GAMMA - 1.)*(w - rho0) / GAMMA;
	u = w - (rho0 + p);
    #endif

	// User may want to handle this case differently, e.g. do NOT return upon
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.) || (u <= 0.)) {
		retval = 5;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#if AMD
	#pragma unroll 3
	for (i = 1; i<4; i++)  Qtcon[i] = fma(ncon[i], Qdotn, Qcon[i]);
	#pragma unroll 3
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (fma(QdotB, Bcon[i] / W, Qtcon[i]));
	#else
	#pragma unroll 3
	for (i = 1; i<4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
	#pragma unroll 3
	for (i = 1; i<4; i++) prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB*Bcon[i] / W);
	#endif
	/* set field components */
	//#pragma unroll 3
	//for (i = BCON1; i <= BCON3; i++) prim[i] = U[i];

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
	double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    )
{
	double f, df, dx[NEWT_DIM_2], x_old[NEWT_DIM_2];
	double resid[NEWT_DIM_2], jac[NEWT_DIM_2][NEWT_DIM_2];
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
		func_vsq(x, dx, resid, jac, &f, &df, n, Bsq, Qtsq, QdotBsq, Qdotn, D
                #if (DOHELM)
                 , gpu_eos_table
                #endif
                );  /* returns with new dx, f, df */

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
	double jac[][NEWT_DIM_2], double *f, double *df, int n, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
)
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

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    double rho = D * sqrt(1. - vsq);
    double w = W * (1. - vsq);
    double dpdrho, dpde_d;

    eos_mode_rhow_pres_dpdrho_dpde_d (gpu_eos_table, rho, w, &p_tmp, &dpdrho, &dpde_d);

    double dpdeps_o_rho = dpde_d / rho;
    double dpdvsq_1 = -0.5 * D / sqrt(1. - vsq) * dpdrho;
    double dpdvsq_2 = -0.5 * (W + p_tmp / sqrt(1. - vsq)) / rho;
    dPdW = ( dpdeps_o_rho / (1.0 + dpdeps_o_rho) ) * (1. - vsq);
    dPdvsq = (dpdvsq_1 + dpde_d * dpdvsq_2)/(1.0 + dpdeps_o_rho);

    #else
    // 2. Ideal gas EOS
	p_tmp = (GAMMA - 1.) * (fma(W, gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - vsq) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - vsq), -W)) / GAMMA;
    #endif

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
	double gcov[10];
	double gcon[10];
	double g;
};

struct of_trans {
	double Mud[NDIM][NDIM];
	double Mud_inv[NDIM][NDIM];
};

struct of_state {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
};

/* find relative 4-velocity from 4-velocity (both in code coords) */
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon)
{
	double alpha, beta[NDIM], gamma;
	int j;

	/* now solve for v-- we can use the same u^t because
	* it didn't change under KS -> KS' */
	alpha = 1. / sqrt(-geom->gcon[0]);
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;
	gamma = alpha*ucon[0];

	utcon[0] = 0;
	SLOOPA utcon[j] = ucon[j] + gamma*beta[j] / alpha;
}

__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut)
{
	double AA, BB, CC, DD, one_over_alpha_sq;
	//compute the Lorentz factor based on contravariant 3-velocity
	AA = geom->gcov[0];
	BB = 2.*(geom->gcov[1] * vcon[1] +
		geom->gcov[2] * vcon[2] +
		geom->gcov[3] * vcon[3]);
	CC = geom->gcov[4] * vcon[1] * vcon[1] +
		geom->gcov[7] * vcon[2] * vcon[2] +
		geom->gcov[9] * vcon[3] * vcon[3] +
		2.*(geom->gcov[5] * vcon[1] * vcon[2] +
		geom->gcov[6] * vcon[1] * vcon[3] +
		geom->gcov[8] * vcon[2] * vcon[3]);

	DD = -1. / (AA + BB + CC);

	one_over_alpha_sq = -geom->gcon[0];

	if (DD<one_over_alpha_sq) {
		DD = one_over_alpha_sq;
	}

	*ut = sqrt(DD);
}

__device__ void primtoU(double *pr, struct of_state *q, struct of_geom *geom, double *U, double gam
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                        )
{
	double h, l;
	primtoflux(pr, q, 0, geom, U,&h,&l, gam
                #if (DOHELM)
                , gpu_eos_table
                #endif
               );
	return;
}

/* add in source terms to equations of motion */
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *  dU, double Dt, double gam, const  double* __restrict__ conn_GPU, struct of_state *  q, double a, double r
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
{
	double mhd[NDIM][NDIM];
	int k, j, dir;
	double conn, P, w, bsq, eta, ptot;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif

	#if (DOHELM)
	// Helmholtz EOS
	eos_mode_rhou_pres(gpu_eos_table, ph[RHO], ph[UU], &P);
	#else
	// Ideal gas EOS
	P = (gam - 1.)*ph[UU];
	#endif
	w = P + ph[RHO] + ph[UU];
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	#if AMD
	ptot = fma(0.5, bsq, P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* single row of mhd stress tensor,
	* first index up, second index down */
	for (dir = 0; dir < NDIM; dir++){
		#if AMD
		#pragma unroll 4
		DLOOPA mhd[dir][j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
		#else
		DLOOPA mhd[dir][j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
		#endif
	}

	/* contract mhd stress tensor with connection */
	#pragma unroll 9
	PLOOP dU[k] = 0.;

	#pragma unroll 4
	for (k = 0; k<NDIM; k++){
		#if(NSY)
		dU[UU] += mhd[0][k] * conn_GPU[0 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[1][k] * conn_GPU[4 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd[2][k] * conn_GPU[7 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd[3][k] * conn_GPU[9 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[1][k] * conn;
		dU[U1] += mhd[0][k] * conn;
		conn = conn_GPU[2 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[2][k] * conn;
		dU[U2] += mhd[0][k] * conn;
		conn = conn_GPU[3 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[3][k] * conn;
		dU[U3] += mhd[0][k] * conn;
		conn = conn_GPU[5 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[2][k] * conn;
		dU[U2] += mhd[1][k] * conn;
		conn = conn_GPU[6 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[3][k] * conn;
		dU[U3] += mhd[1][k] * conn;
		conn = conn_GPU[8 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd[3][k] * conn;
		dU[U3] += mhd[2][k] * conn;
		#else
		dU[UU] += mhd[0][k] * conn_GPU[0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[1][k] * conn_GPU[4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd[2][k] * conn_GPU[7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd[3][k] * conn_GPU[9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[1][k] * conn;
		dU[U1] += mhd[0][k] * conn;
		conn = conn_GPU[2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[2][k] * conn;
		dU[U2] += mhd[0][k] * conn;
		conn = conn_GPU[3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd[3][k] * conn;
		dU[U3] += mhd[0][k] * conn;
		conn = conn_GPU[5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[2][k] * conn;
		dU[U2] += mhd[1][k] * conn;
		conn = conn_GPU[6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd[3][k] * conn;
		dU[U3] += mhd[1][k] * conn;
		conn = conn_GPU[8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd[3][k] * conn;
		dU[U3] += mhd[2][k] * conn;
		#endif
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
	double bsq = dot(q->bcon,q->bcov);
	if (bsq / ph[RHO]<1. || r<10.){
		if (fabs(q->ucov[0] * lambda)*Dt<0.1*fabs(int_energy)){
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda;
		}
		else{
			lambda *= (0.1*fabs(int_energy)) / (fabs(q->ucov[0] * lambda)*Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda;
		}
	}
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                        )
{
	int j;
	double r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    eos_mode_rhou_pres (gpu_eos_table, r, u, &P);
    #else
    // 2. Ideal gas EOS
	P = (GAMMA - 1.)*u;
    #endif

	w = P + r + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
}


__device__ void primtoflux(double *  pr, struct of_state *  q, int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin, double gam
                            #if (DOHELM)
                            , const  double* __restrict__ gpu_eos_table
                            #endif
                           )
{
	int j, k;
	double mhd[NDIM];
	double P, w, bsq, eta, ptot;

	/*Calculate misc quantities*/

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    double cs2_helm;
    eos_mode_rhou_pres_cs2 (gpu_eos_table, pr[RHO], pr[UU], &P, &cs2_helm);
	w = pr[RHO] + pr[UU] + P;
    #else
    // 2. Ideal gas EOS
    P = (gam - 1.)*pr[UU];
    #if AMD
    w = fma(gam, pr[UU], pr[RHO]);
    #else
    w = pr[RHO] + gam*pr[UU];
    #endif
    #endif

	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	#if AMD
	ptot = fma(0.5, bsq, P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* particle number flux */
	flux[RHO] = pr[RHO] * q->ucon[dir];

	/* single row of mhd stress tensor,
	* first index up, second index down */
	#if AMD
	#pragma unroll 4
	DLOOPA mhd[j] = fma(eta, q->ucon[dir] * q->ucov[j], fma(ptot, delta(dir, j), -q->bcon[dir] * q->bcov[j]));
	#else
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
	#endif

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

	/*Calculate wavespeed*/
	if (dir != 0){
		double discr, vp, vm, va2, cs2, cms2;
		double Acon_0, Acon_js;
		double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
		if (dir == 1){
			Acon_0 = geom->gcon[1];
			Acon_js = geom->gcon[4];
		}
		else if (dir == 2){
			Acon_0 = geom->gcon[2];
			Acon_js = geom->gcon[7];
		}
		else if (dir == 3){
			Acon_0 = geom->gcon[3];
			Acon_js = geom->gcon[9];
		}

		/* find fast magnetosonic speed */

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        // cs2 was already calculated above
        cs2 = cs2_helm;
        #else
        // 2. Ideal gas EOS
        cs2 = gam*(gam - 1.)*pr[UU] / w;
        #endif

		va2 = bsq / eta;
		cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

		/* check on it! */
		if (cms2 < 0.) {
			//fail(FAIL_COEFF_NEG) ;
			cms2 = SMALL;
		}
		if (cms2 > 1.) {
			//fail(FAIL_COEFF_SUP) ;
			cms2 =1.;
		}

		/* now require that speed of wave measured by observer
		q->ucon is cms2 */
		Asq = Acon_js;
		Bsq = geom->gcon[0];// dot(Bcon, Bcov);
		Au = q->ucon[dir];
		Bu = q->ucon[0];
		AB = Acon_0;
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

		*vmax = MY_MAX(vp, vm);
		*vmin = MY_MIN(vp, vm);
	}
	return;
}


__device__ double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq)
{
	int count = 0;
	int keep_looping = 1;
	double dx;
	double x;
	double error_1, error_2, derror_dx;

	x = start;
	error_1 = Drel(dir, x, ucon, bcon, E, vasq, csq);

	while (keep_looping) {
		error_2 = Drel(dir, x + x*pow(10., -6.), ucon, bcon, E, vasq, csq);
		derror_dx = (error_2 - error_1) / (x*pow(10., -6.));
		dx = error_1 / (derror_dx);
		x = x - dx;
		error_1 = Drel(dir, x, ucon, bcon, E, vasq, csq);

		if ((count >= max_count) || fabs(dx / x) < fabs(x)*pow(10., -4.)) keep_looping = 0;
		count++;
	}

	if ((count >= max_count) || (fabs(x)>fabs(start))) {
		x = start;
	}
	return x;
}

__device__ double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq) {
	double kcov[NDIM], kcon[NDIM], Kcov[NDIM], Kcon[NDIM];
	double om, omsq, ksq, kvasq, cfsq, result;
	int i;
	kcov[0] = -v; kcov[1] = 0.0; kcov[2] = 0.0; kcov[3] = 0.0, kcon[0] = v;
	if (dir == 1) {
		kcov[1] = 1.0;
		kcon[1] = 1.;
	}
	else if (dir == 2) {
		kcov[2] = 1.0;
		kcon[2] = 1.;
	}
	else if (dir == 3) {
		kcov[3] = 1.0;
		kcon[3] = 1.;
	}
	om = dot(ucon, kcov);
	omsq = pow(om, 2.0);

	for (i = 0; i < NDIM; i++) {
		Kcon[i] = kcon[i] + ucon[i] * om;
		Kcov[i] = kcon[i] + ucon[i] * om;
	}
	Kcov[0] *= -1.;
	ksq = dot(Kcov, Kcon);
	kvasq = pow(dot(kcov, bcon), 2.0) / (E + SMALL);
	cfsq = vasq + csq*(1.0 - vasq);
	result = 0.5*(cfsq*ksq + csq*kvasq + sqrt(pow(cfsq*ksq + csq*kvasq, 2.0) - 4.0*ksq*csq*kvasq)) - omsq;
	return result;
}

__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q)
{
	/* get ucon */
	ucon_calc(pr, geom, q->ucon);
	lower(q->ucon, geom->gcov, q->ucov);
	bcon_calc(pr, q->ucon, q->ucov, q->bcon);
	lower(q->bcon, geom->gcov, q->bcov);

	return;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ void raise(double ucov[NDIM], double gcon[10], double ucon[NDIM])
{
	#if AMD
	ucon[0] = fma(gcon[0], ucov[0], fma(
		gcon[1], ucov[1], fma(
		gcon[2], ucov[2]
		, gcon[3] * ucov[3])));
	ucon[1] = fma(gcon[1], ucov[0], fma(
		gcon[4], ucov[1], fma(
		gcon[5], ucov[2]
		, gcon[6] * ucov[3])));
	ucon[2] = fma(gcon[2], ucov[0], fma(
		gcon[5], ucov[1], fma(
		gcon[7], ucov[2]
		, gcon[8] * ucov[3])));
	ucon[3] = fma(gcon[3], ucov[0], fma(
		gcon[6], ucov[1], fma(
		gcon[8], ucov[2]
		, gcon[9] * ucov[3])));
	#else
	ucon[0] = gcon[0] * ucov[0]
		+ gcon[1] * ucov[1]
		+ gcon[2] * ucov[2]
		+ gcon[3] * ucov[3];
	ucon[1] = gcon[1] * ucov[0]
		+ gcon[4] * ucov[1]
		+ gcon[5] * ucov[2]
		+ gcon[6] * ucov[3];
	ucon[2] = gcon[2] * ucov[0]
		+ gcon[5] * ucov[1]
		+ gcon[7] * ucov[2]
		+ gcon[8] * ucov[3];
	ucon[3] = gcon[3] * ucov[0]
		+ gcon[6] * ucov[1]
		+ gcon[8] * ucov[2]
		+ gcon[9] * ucov[3];
	#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower(double ucon[NDIM], double gcov[10], double ucov[NDIM])
{
#if AMD
	ucov[0] = fma(gcov[0], ucon[0], fma(
		gcov[1], ucon[1], fma(
		gcov[2], ucon[2],
		gcov[3] * ucon[3])));
	ucov[1] = fma(gcov[1], ucon[0], fma(
		gcov[4], ucon[1], fma(
		gcov[5], ucon[2],
		gcov[6] * ucon[3])));
	ucov[2] = fma(gcov[2], ucon[0], fma(
		gcov[5], ucon[1], fma(
		gcov[7], ucon[2],
		gcov[8] * ucon[3])));
	ucov[3] = fma(gcov[3], ucon[0], fma(
		gcov[6], ucon[1], fma(
		gcov[8], ucon[2],
		gcov[9] * ucon[3])));
	return;
#else
	ucov[0] = gcov[0]*ucon[0]
		+ gcov[1]*ucon[1]
		+ gcov[2]*ucon[2]
		+ gcov[3]*ucon[3] ;
	ucov[1] = gcov[1]*ucon[0]
		+ gcov[4]*ucon[1]
		+ gcov[5]*ucon[2]
		+ gcov[6]*ucon[3] ;
	ucov[2] = gcov[2]*ucon[0]
		+ gcov[5]*ucon[1]
		+ gcov[7]*ucon[2]
		+ gcov[8]*ucon[3] ;
	ucov[3] = gcov[3]*ucon[0]
		+ gcov[6]*ucon[1]
		+ gcov[8]*ucon[2]
		+ gcov[9]*ucon[3] ;
#endif
}

/* find contravariant four-velocity */
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon)
{
	double alpha, gamma;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0]);
	#pragma unroll 4
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;

	gamma_calc(pr, geom, &gamma);

	ucon[0] = gamma / alpha;
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
	bcon[0] = fma(pr[B1], ucov[1], fma(pr[B2], ucov[2], pr[B3] * ucov[3]));
	#pragma unroll 3
	for (j = 1; j<4; j++)
		bcon[j] = (fma(bcon[0], ucon[j], pr[B1 - 1 + j])) / ucon[0];
	#else
	bcon[0] = pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3];
	#pragma unroll 3
	for (j = 1; j<4; j++)
		bcon[j] = (pr[B1 - 1 + j] + bcon[0] * ucon[j]) / ucon[0];
	#endif
	return;
}

__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma)
{
	double qsq;
	#if AMD
	qsq = fma(geom->gcov[4], pr[U1] * pr[U1], fma(
		geom->gcov[7], pr[U2] * pr[U2],
		geom->gcov[9] * pr[U3] * pr[U3]))
		+ 2.*fma(geom->gcov[5], pr[U1] * pr[U2], fma(
		geom->gcov[6], pr[U1] * pr[U3],
		geom->gcov[8] * pr[U2] * pr[U3]));
	#else
	qsq = geom->gcov[4] * pr[U1] * pr[U1]
		+ geom->gcov[7] * pr[U2] * pr[U2]
		+ geom->gcov[9] * pr[U3] * pr[U3]
		+ 2.*(geom->gcov[5] * pr[U1] * pr[U2]
		+ geom->gcov[6] * pr[U1] * pr[U3]
		+ geom->gcov[8] * pr[U2] * pr[U3]);
	#endif

	if (qsq < 0.){
		if (fabs(qsq) > 1.E-10){ // then assume not just machine precision
			*gamma = 1.;
			return (1);
		}
		else qsq = 1.E-10; // set floor
	}

	*gamma = sqrt(1. + qsq);

	return(0);
}

/* load local geometry into structure geom */
__device__ void get_geometry(int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU)
{
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jj*(BS_3 + 2 * N3G) + zz;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_2 + 2 * N2G) + jj;
	#endif
	#if(NSY)
	geom->gcon[0] = gcon_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[0] = gcov_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[1] = gcon_GPU[1 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[1] = gcov_GPU[1 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[2] = gcon_GPU[2 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[2] = gcov_GPU[2 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[3] = gcon_GPU[3 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[3] = gcov_GPU[3 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[4] = gcon_GPU[4 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[4] = gcov_GPU[4 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[5] = gcon_GPU[5 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[5] = gcov_GPU[5 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[6] = gcon_GPU[6 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[6] = gcov_GPU[6 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[7] = gcon_GPU[7 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[7] = gcov_GPU[7 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[8] = gcon_GPU[8 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[8] = gcov_GPU[8 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[9] = gcon_GPU[9 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[9] = gcov_GPU[9 * NPG*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->g = gdet_GPU[kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	#else
	geom->gcon[0] = gcon_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[0] = gcov_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[1] = gcon_GPU[1 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[1] = gcov_GPU[1 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[2] = gcon_GPU[2 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[2] = gcov_GPU[2 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[3] = gcon_GPU[3 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[3] = gcov_GPU[3 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[4] = gcon_GPU[4 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[4] = gcov_GPU[4 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[5] = gcon_GPU[5 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[5] = gcov_GPU[5 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[6] = gcon_GPU[6 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[6] = gcov_GPU[6 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[7] = gcon_GPU[7 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[7] = gcov_GPU[7 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[8] = gcon_GPU[8 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[8] = gcov_GPU[8 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcon[9] = gcon_GPU[9 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->gcov[9] = gcov_GPU[9 * NPG * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	geom->g = gdet_GPU[kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	#endif
}

/* load local orthonormal tetrad transformation matrix for HLLC/HLLD solvers*/
__device__ void get_trans(int ii, int jj, int zz, int kk, struct of_trans *trans, const  double* __restrict__ Mud_GPU, const  double* __restrict__ Mud_inv_GPU)
{
	int i, j;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jj*(BS_3 + 2 * N3G) + zz;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = ii*(BS_2 + 2 * N2G) + jj;
	#endif

	#if(NSY)
	for (i = 0; i < NDIM; i++)for (j = 0; j < NDIM; j++) {
		trans->Mud[i][j] = Mud_GPU[(i*NDIM + j) * NSOLVER*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		trans->Mud_inv[i][j] = Mud_inv_GPU[(i*NDIM + j) * NSOLVER*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	}
	#else
	for (i = 0; i < NDIM; i++)for (j = 0; j < NDIM; j++) {
		trans->Mud[i][j] = Mud_GPU[(i*NDIM + j) * NSOLVER * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		trans->Mud_inv[i][j] = Mud_inv_GPU[(i*NDIM + j) * NSOLVER * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + kk*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
	}
	#endif
}

__device__ void inflow_check(double *  pr, int ii, int jj, int zz, int type, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	struct of_geom geom;
	double ucon[NDIM];
	double alpha, beta1, gamma, vsq;
	get_geometry(ii, jj, zz, CENT, &geom, gcov, gcon, gdet);
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
		alpha = 1. / sqrt(-geom.gcon[0]);
		beta1 = geom.gcon[1] * alpha*alpha;

		// reset radial velocity so radial 4-velocity is zero
		pr[U1] = beta1 / alpha;

		// now find new gamma and put it back in
		vsq = geom.gcov[4] * pr[UTCON1 + 1 - 1] * pr[UTCON1 + 1 - 1]; //1,1
		vsq += 2.*geom.gcov[5] * pr[UTCON1 + 2 - 1] * pr[UTCON1 + 1 - 1]; //1,2
		vsq += 2.*geom.gcov[6] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 1 - 1]; //1,3
		vsq += geom.gcov[7] * pr[UTCON1 + 2 - 1] * pr[UTCON1 + 2 - 1]; //2,2
		vsq += 2 * geom.gcov[8] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 2 - 1]; //2,3
		vsq += geom.gcov[9] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 3 - 1]; //1,2

		vsq = MY_MAX(1.e-13,vsq);
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
	Dqm = (2.0)*(y2 - y1); // Danat: changed the coefficient from 1.5 to 2.0 for comparison between CPU and GPU codes
	Dqp = (2.0)*(y3 - y2);
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

__device__ void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double *F) {
	double Sp;

	Sp = (x4 - x2) / (x5 - x1);
	F[0] = MY_MAX(0., MY_MIN(1., 10.*(Sp - 0.75)));
	if (fabs(x4 - x2) / MY_MIN(x4, x2) < 0.33) F[0] = 0;
}


/* returns b^2 (i.e., twice magnetic pressure) */
__device__ double bsq_calc(double *  pr, struct of_geom *  geom)
{
	struct of_state q;
	get_state(pr, geom, &q);
	return(dot(q.bcon, q.bcov));
}

__device__ double interp(double y1, double y2, double y3)
{
	double Dqm, Dqp, Dqc, s;
	/* woodward, or monotonized central, slope limiter */
	Dqm = (2.0)*(y2 - y1);
	Dqp = (2.0)*(y3 - y2);
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

//#include "decsCUDA.h"

__global__ void interpolate(double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + 2 * D3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3);
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3;
	jcurr += (N2G - 1)*D2;
	icurr += (N1G - 1)*D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel = 0, zoffset = 0, z2 = 0, z3 = 0, z4 = 0;
	double x2, x3, x4;
	double temp;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (zdel) {
		if (zcurr == N3G - D3) {
			z2 = -1 * zdel;
			z3 = 0;
			z4 = 1 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G) {
			z2 = -zoffset - 1 * zdel;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G + zdel*zsize) {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
		else if (zcurr == BS_3 + N3G) {
			z2 = -1 * zdel*zsize;
			z3 = 0;
			z4 = 1 * zdel;
		}
		else if (zcurr - zoffset == BS_3 + N3G - zdel*zsize) {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + zdel*zsize;
		}
		else {
			z2 = -zoffset - 1 * zdel*zsize;
			z3 = -zoffset;
			z4 = -zoffset + 1 * zdel*zsize;
		}
	}

	if (k == 1) {
		#pragma unroll 9
		for (k = 0; k<NPR; k++) {
			x2 = p[MY_MAX(k*(ksize)+global_id + z2 - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x3 = p[k*(ksize)+global_id + z3];
			x4 = p[MY_MIN(k*(ksize)+global_id + z4 + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			temp = 0.5*interp(x2, x3, x4);
			dq1[k*(ksize)+global_id] = x3 - temp;
			dq2[k*(ksize)+global_id] = x3 + temp;
		}
	}
}

__global__ void fluxcalcprep(const  double* __restrict__   F, double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int lim, int number, const  double* __restrict__  V, int POLE_1, int POLE_2)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3 + 2 * D3 )*(BS_2 + 2 * D2 );
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3 );
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3 );
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3 ) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3;
	jcurr += (N2G - 1)*D2;
	icurr += (N1G - 1)*D1;
	if (global_id<(BS_1 + 2 * D1 ) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3 )) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel = 0, zoffset = 0, z1 = -2, z2 = -1, z3 = 0, z4 = 1, z5 = 2;
	double x1, x2, x3, x4, x5, FF=0.;
	double temp, result;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;

	if (zdel){
		if (zcurr == N3G - D3) {
			z1 = - 2 * zdel;
			z2 = - 1 * zdel;
			z3 = 0;
			z4 = 1 * zdel*zsize;
			z5 = 2 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G) {
			z1 = - zoffset - 2 * zdel;
			z2 = - zoffset - 1 * zdel;
			z3 = - zoffset;
			z4 = - zoffset + 1 * zdel*zsize;
			z5 = - zoffset + 2 * zdel*zsize;
		}
		else if (zcurr - zoffset == N3G + zdel*zsize) {
			z1 = - zoffset - 1 * zdel*zsize - 1 * zdel;
			z2 = - zoffset - 1 * zdel*zsize;
			z3 = - zoffset;
			z4 = - zoffset + 1 * zdel*zsize;
			z5 = - zoffset + 2 * zdel*zsize;
		}
		else if (zcurr == BS_3 + N3G) {
			z1 = - 2 * zdel*zsize;
			z2 = - 1 * zdel*zsize;
			z3 = 0;
			z4 = 1 * zdel;
			z5 = 2 * zdel;
		}
		else if (zcurr - zoffset == BS_3 + N3G - zdel*zsize) {
			z1 = - zoffset - 2 * zdel*zsize;
			z2 = - zoffset - 1 * zdel*zsize;
			z3 = - zoffset;
			z4 = - zoffset + zdel*zsize;
			z5 = - zoffset + zdel*zsize + 1 * zdel;
		}
		else{
			z1 = - zoffset - 2 * zdel*zsize;
			z2 = - zoffset - 1 * zdel*zsize;
			z3 = - zoffset;
			z4 = - zoffset + 1 * zdel*zsize;
			z5 = - zoffset + 2 * zdel*zsize;
		}
	}
	#endif
	if (k == 1){
		#if(PPM)
			#if(PPM_FLATTENER)
			x1 = p[MY_MAX(RHO*(ksize)+global_id + z1*zdel - 2 * (BS_3 + 2 * N3G)*jdel - 2 * isize*idel, 0)];
			x2 = p[MY_MAX(RHO*(ksize)+global_id + z2*zdel - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x3 = p[RHO*(ksize)+global_id + z3*zdel];
			x4 = p[MY_MIN(RHO*(ksize)+global_id + z4*zdel + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			x5 = p[MY_MIN(RHO*(ksize)+global_id + z5*zdel + 2 * (BS_3 + 2 * N3G)*jdel + 2 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			calculate_flattener(x1, x2, x3, x4, x5, &FF);
			x2 = p[MY_MAX((UU + dir)*(ksize)+global_id + z2*zdel - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x4 = p[MY_MIN((UU + dir)*(ksize)+global_id + z4*zdel + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			if (x4 - x2 > 0.) FF = 0.;
			#endif
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			x1 = p[MY_MAX(k*(ksize)+global_id + z1*zdel - 2 * (BS_3 + 2 * N3G)*jdel - 2 * isize*idel, 0)];
			x2 = p[MY_MAX(k*(ksize)+global_id + z2*zdel - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x3 = p[k*(ksize)+global_id + z3*zdel];
			x4 = p[MY_MIN(k*(ksize)+global_id + z4*zdel + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			x5 = p[MY_MIN(k*(ksize)+global_id + z5*zdel + 2 * (BS_3 + 2 * N3G)*jdel + 2 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			para(x1, x2, x3, x4, x5, &result, &temp);
			dq1[k*(ksize)+global_id] = FF*x3 + (1. - FF)*result;
			dq2[k*(ksize)+global_id] = FF*x3 + (1. - FF)*temp;
		}
		#else
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			x2 = p[MY_MAX(k*(ksize)+global_id + z2*zdel - 1 * (BS_3 + 2 * N3G)*jdel - 1 * isize*idel, 0)];
			x3 = p[k*(ksize)+global_id + z3*zdel];
			x4 = p[MY_MIN(k*(ksize)+global_id + z4*zdel + 1 * (BS_3 + 2 * N3G)*jdel + 1 * isize*idel, NPR*((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + fix_mem1))];
			temp=0.5*slope_lim(x2, x3, x4, 0);
			dq1[k*(ksize)+global_id] = x3-temp;
			dq2[k*(ksize)+global_id] = x3+temp;
		}
		#endif
	}
}

__global__ void reconstruct_internal(double* p, double* ps, const  double* __restrict__ dq1, const  double* __restrict__ dq2, const  double* __restrict__ gdet_GPU, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr*(BS_3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += D2;
	icurr += D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;

	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel = 0, zoffset = 0, u;
	int zsize2 = 1, zlevel2 = 0, zoffset2 = 0;
	double temp[NPR];


	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if ((k == 1)){
		if (zoffset == 0){
			for (k = 0; k < NPR; k++) temp[k] = p[k*(ksize)+global_id - zoffset];
			for (u = 0; u < zsize; u++){
				for (k = 0; k < NPR; k++) p[k*ksize + global_id - zoffset + u] = temp[k] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*(dq2[k*(ksize)+global_id - zoffset] - dq1[k*(ksize)+global_id - zoffset]);
			}

			temp[0] = ps[0 * (ksize)+global_id - zoffset];
			temp[2] = ps[2 * (ksize)+global_id - zoffset];
			for (u = 0; u < zsize; u++){
				ps[0 * ksize + global_id - zoffset + u] = temp[0] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*0.5*(dq2[B1*(ksize)+global_id - zoffset] + dq2[B1*(ksize)+global_id - isize - zoffset] - dq1[B1*(ksize)+global_id - zoffset] - dq1[B1*(ksize)+global_id - isize - zoffset]);
				ps[2 * ksize + global_id - zoffset + u] = temp[2] + ((double)u) / ((double)zsize)*(ps[2 * (ksize)+global_id - zoffset + zsize] - ps[2 * (ksize)+global_id - zoffset]);
			}
		}

		#if(N_LEVELS_1D_INT>0 && D3>0)
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - D2 - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize = (int)(0.001+pow(2.0, (double)zlevel));
		zoffset = (zcurr - N3G) % zsize;
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel2 = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel2 = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr + D2 - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize2 = (int)(0.001 + pow(2.0, (double)zlevel2));
		zoffset2 = (zcurr - N3G) % zsize2;
		#endif
		if (zoffset2 == 0) {
			if ((POLE_1 == 1 && jcurr - N2G < BS_2 / 2) && (jcurr != N2G)) {
				temp[1] = ps[1 * (ksize)+global_id - zoffset2];
				for (u = 0; u < zsize2; u++) {
					ps[1 * ksize + global_id - zoffset2 + u] = temp[1] + (((double)u + 0.5) - 0.5*(double)zsize2) / ((double)zsize)*0.5*(dq2[B2*(ksize)+global_id - (BS_3 + 2 * N3G) - zoffset] - dq1[B2*(ksize)+global_id - (BS_3 + 2 * N3G) - zoffset]);
					ps[1 * ksize + global_id - zoffset2 + u] += (((double)u + 0.5) - 0.5*(double)zsize2) / ((double)zsize2)*0.5*(dq2[B2*(ksize)+global_id - zoffset2] - dq1[B2*(ksize)+global_id - zoffset2]);
				}
			}
		}
		if (zoffset == 0) {
			if ((POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) && (jcurr + D2 != BS_2 + N2G)){
				temp[1] = ps[1 * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset];
				for (u = 0; u < zsize; u++){
					ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] = temp[1] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*0.5*(dq2[B2*(ksize)+global_id - zoffset] - dq1[B2*(ksize)+global_id - zoffset]);
					ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] +=  (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize2)*0.5*(dq2[B2*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset2] - dq1[B2*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset2]);
				}
			}
		}
	}
}

__global__ void fluxcalc2D2(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir,
	double gam, double cour, double*  dtij, int POLE_1, int POLE_2, double dx_1, double dx_2, double dx_3, int calc_time
#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
#endif
    )
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;
	int group_id = blockIdx.x;
	int local_size = blockDim.x;
	__shared__ double local_dtij[LOCAL_WORK_SIZE];
	int k = 0;
	int isize, icurr, jcurr, zcurr;
	isize = (BS_3 + 2 * D3 - (dir == 3))*(BS_2 + 2 * D2 - (dir == 2));
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3 - (dir == 3));
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3 - (dir == 3));
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3 - (dir == 3)) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3 + (dir == 3);
	jcurr += (N2G - 1)*D2 + (dir == 2);
	icurr += (N1G - 1)*D1 + (dir == 1);
	if (global_id<(BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel, i;
	int face;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
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
	int zsize = 1, zlevel = 0, zoffset = 0;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx_1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx_2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx_3*((double)zsize); }

	if (k == 1){
		get_geometry(icurr, jcurr, zcurr, face, &geom, gcov, gcon, gdet);

		if (zoffset != 0 && dir == 3){
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				p[k] = 0.5*(pv[k*(ksize)+global_id] + pv[k*(ksize)+global_id - D3]);
			}
		}
		else{
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				p[k] = dq2[k*(ksize)+global_id - idel*isize - jdel*(BS_3 + 2 * N3G) - zdel];
			}
		}
		#if(STAGGERED)
		for (k = 0; k< NPR; k++){
			if ((dir == 1 && k == B1) || (dir == 2 && k == B2) || (dir == 3 && k == B3)){
				p[k] = ps[(k - B1)*(ksize)+global_id];
			}

			if (dir == 2 && k == B1 && ((jcurr == BS_2 + N2G && POLE_2 == 1) || (jcurr == N2G && POLE_1 == 1))){
				#if AMD
				p[k] = 0.;
				#else
				p[k] = 0.;
				#endif
			}
		}
		#endif

		get_state(p, &geom, &state);
		primtoflux(p, &state, dir, &geom, temp1, &cmax_l, &cmin_l, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                   );
		primtoflux(p, &state, 0, &geom, temp2, &cmax_l, &cmin_l, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                   );
		//vchar(p, &state, &geom, dir, &cmax_l, &cmin_l, gam);

		if (zoffset != 0 && dir == 3){
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				p[k] = 0.5*(pv[k*(ksize)+global_id] + pv[k*(ksize)+global_id - D3]);
			}
		}
		else{
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				p[k] = dq1[k*(ksize)+global_id];
			}
		}
		#if(STAGGERED)
		for (k = 0; k< NPR; k++){
			if ((dir == 1 && k == B1) || (dir == 2 && k == B2) || (dir == 3 && k == B3)){
				p[k] = ps[(k - B1)*(ksize)+global_id];
			}
			if (dir == 2 && k == B1 && ((jcurr == BS_2 + N2G && POLE_2 == 1) || (jcurr == N2G && POLE_1 == 1))){
				#if AMD
				p[k] = 0.;
				#else
				p[k] = 0.;
				#endif
			}
		}
		#endif
		get_state(p, &geom, &state);
		primtoflux(p, &state, dir, &geom, temp3, &cmax_r, &cmin_r, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                   );
		primtoflux(p, &state, 0, &geom, temp4, &cmax_r, &cmin_r, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                   );
		//vchar(p, &state, &geom, dir, &cmax_r, &cmin_r,  gam);

		cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
		cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
		ctop = MY_MAX(cmax, cmin);

		#pragma unroll 9
		for (k = 0; k<NPR_U; k++){
			#if(HLLF)
			F[k*(ksize)+global_id] = (cmax*temp1[k] + cmin*temp3[k] - cmax*cmin*(temp4[k] - temp2[k])) / (cmax + cmin + SMALL);
			#else
			F[k*(ksize)+global_id] =  LAXF*(0.5*(temp1[k] + temp3[k] - ctop*(temp4[k] - temp2[k])));
			#endif
		}


		local_dtij[local_id] = factor / ctop;
	}
	if (calc_time == 1){
		__syncthreads();
		for (i = local_size / 2; i > 1; i = i / 2){
			if (local_id < i){
				local_dtij[local_id] = MY_MIN(local_dtij[local_id], local_dtij[local_id + i]);
			}
			__syncthreads();
		}
		if (local_id == 0){
			dtij[group_id] = MY_MIN(local_dtij[0], local_dtij[1]);
		}
	}
}

__device__ void primtoflux_FT(double *pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR])
{
	int j, k;
	double  P, w, bsq, eta, ptot;

	/* particle number flux */
	flux[RHO] = pr[RHO] * ucon[dir];

	/* MHD stress tensor, with first index up, second index down */
	P = (GAMMA - 1.)*pr[UU];
	w = P + pr[RHO] + pr[UU];
	bsq = -bcon[0] * bcon[0] + bcon[1] * bcon[1] + bcon[2] * bcon[2] + bcon[3] * bcon[3];
	eta = w + bsq;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	flux[UU] = -eta*ucon[dir] * ucon[0] + ptot*delta(dir, 0) + bcon[dir] * bcon[0];
	for (j = 1; j < NDIM; j++)flux[UU + j] = eta*ucon[dir] * ucon[j] + ptot*delta(dir, j) - bcon[dir] * bcon[j];

	/* dual of Maxwell tensor */
	for (k = B1; k <= B3; k++) {
		flux[k] = bcon[k - 4] * ucon[dir] - bcon[dir] * ucon[k - 4];
	}
	#if(DOKTOT )
	flux[KTOT] = flux[RHO] * pr[KTOT];
	#endif
}

__device__ void vchar_FT(double * pr, double ucon[NDIM], double bcon[NDIM], int dir, double *vmax, double *vmin)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2;
	double Asq, Bsq, Au, Bu, Au2, Bu2, AuBu, A, B, C;
	int j;

	/* find fast magnetosonic speed */
	bsq = -bcon[0] * bcon[0] + bcon[1] * bcon[1] + bcon[2] * bcon[2] + bcon[3] * bcon[3];
	#if AMD
	EF = fma(gam, pr[UU], pr[RHO]);
	#else
	EF = pr[RHO] + GAMMA* pr[UU];
	#endif
	EE = bsq + EF;

	/* find fast magnetosonic speed */
	cs2 = GAMMA*(GAMMA - 1.)*pr[UU] / EF;
	va2 = bsq / EE;
	cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

	/* check on it! */
	if (cms2 < 0.) {
		cms2 = SMALL;
	}
	if (cms2 > 1.) {
		cms2 = 1.;
	}

	/* now require that speed of wave measured by observer q->ucon is cms2 */
	Asq = 1.;
	Bsq = -1.;
	Au = ucon[dir];
	Bu = ucon[0];
	Au2 = Au*Au;
	Bu2 = Bu*Bu;
	AuBu = Au*Bu;

	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2.* fma(-(AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4.*A*C);
	#else
	A = Bu2 - (Bsq + Bu2)*cms2;
	B = 2.*(AuBu - (AuBu)*cms2);
	C = Au2 - (Asq + Au2)*cms2;
	discr = B*B - 4.*A*C;
	#endif

	if ((discr<0.0) && (discr>-1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) {
		discr = 0.;
	}

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	#if( FULL_DISP )
	double vp2, vm2;
	vp2 = NewtonRaphson(vp, 5, dir, ucon, bcon, EE, va2, cs2);
	vm2 = NewtonRaphson(vm, 5, dir, ucon, bcon, EE, va2, cs2);
	if (fabs(vp2 - vm2) > pow(10., -4.)) {
		vp = vp2;
		vm = vm2;
	}
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

__device__ void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *vmax, double *vmin
                        #if (DOHELM)
                        , const  double* __restrict__ gpu_eos_table
                        #endif
                      )
{
	double discr, vp, vm, va2, cs2, cms2;
	double bsq, eta, w;
	double Acon_0, Acon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	if (dir == 1) {
		Acon_0 = geom->gcon[1];
		Acon_js = geom->gcon[4];
	}
	else if (dir == 2) {
		Acon_0 = geom->gcon[2];
		Acon_js = geom->gcon[7];
	}
	else if (dir == 3) {
		Acon_0 = geom->gcon[3];
		Acon_js = geom->gcon[9];
	}

	/* find fast magnetosonic speed */

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    double xpres;
    eos_mode_rhou_pres_cs2 (gpu_eos_table, pr[RHO], pr[UU], &xpres, &cs2);
    w = pr[RHO] + pr[UU] + xpres;
    #else
    // 2. Ideal gas EOS
	#if AMD
	w = fma(gam, pr[UU], pr[RHO]);
	#else
	w = pr[RHO] + GAMMA*pr[UU];
	#endif
    cs2 = GAMMA*(GAMMA - 1.)*pr[UU] / w;
    #endif

	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	va2 = bsq / eta;
	cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

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
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	AB = Acon_0;
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
	else if (discr < -1.e-10) discr = 0.;

	discr = sqrt(discr);
	vp = -(-B + discr) / (2.*A);
	vm = -(-B - discr) / (2.*A);

	*vmax = MY_MAX(vp, vm);
	*vmin = MY_MIN(vp, vm);

	return;
}

__global__ void fix_flux(double *  F1, double *  F2, double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int icurr, jcurr, zcurr;
	int k;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (global_id<(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		zcurr = global_id % (BS_3 + 2 * N3G);
		icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
		if (icurr >= N1G - D1 && zcurr >= N3G - D3 && icurr<BS_1 + N1G + D1 && zcurr<BS_3 + N3G + D3) {
			if (NBR_1 < 0){
				F1[B2*(ksize)+icurr*isize + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 9
				PLOOP F2[k*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr] = 0.;
				#endif
				#pragma unroll 9
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + N2G*(BS_3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
			if (NBR_3 < 0){
				F1[B2*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = -F1[B2*(ksize)+icurr*isize + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
				#if(N3G>0)
				F3[B2*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = -F3[B2*(ksize)+icurr*isize + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
				#endif
				#if INFLOW==0
				#pragma unroll 9
				PLOOP F2[k*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.;
				#endif
				#pragma unroll 9
				for (k = 0; k<NPR; k++){
					F2[k*(ksize)+icurr*isize + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = 0.0;
				}
			}
		}
	}
	#if INFLOW==0
	else{
		global_id = global_id - (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
		zcurr = global_id % (BS_3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
		if (jcurr >= N2G - D2 && zcurr >= N3G - D3 && jcurr<BS_2 + N2G + D2 && zcurr<BS_3 + N3G + D3) {
			if (NBR_4<0){
				if (F1[RHO*(ksize)+N1G*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] > 0.) F1[RHO*(ksize)+N1G*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.;
			}
			if (NBR_2<0){
				if (F1[RHO*(ksize)+(BS_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] < 0.) F1[RHO*(ksize)+(BS_1 + N1G)*isize + jcurr*(BS_3 + 2 * N3G) + zcurr] = 0.;
			}
		}

	}
	#endif
}

__global__ void consttransport1(const  double* __restrict__  pb_i, double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3 + 2 * D3)*(BS_2 + 2 * D2);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3);
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3) + zcurr)) / (isize);
	zcurr += (N3G - D3)*D3;
	jcurr += (N2G - D2)*D2;
	icurr += (N1G - D1)*D1;
	if (global_id<(BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double pb[NPR];
	struct of_geom geom;
	struct of_state q;

	if (k==1){
		for (k = 0; k<NPR; k++){
			pb[k] = pb_i[k*(ksize)+global_id];
		}
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		ucon_calc(pb, &geom, q.ucon);
		lower(q.ucon, geom.gcov, q.ucov);
		bcon_calc(pb, q.ucon, q.ucov, q.bcon);

		#if(N3G>0)
		E_cent[1 * ksize + global_id] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
		E_cent[2 * ksize + global_id] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
		#endif
		E_cent[3 * ksize + global_id] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);
	}
}

__global__ void consttransport2(double *  emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3 + D3)*(BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id<(BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int jsize = BS_3 + 2 * N3G;

	if (k==1){
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

		emf[1 * (ksize)+global_id] = 0.25*((-F2[B3*(ksize)+global_id] - (dE_LEFT_13_1* (double)(F2[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_13_2* (double)(F2[RHO*(ksize)+global_id]>0.0)))
			+ (-F2[B3*(ksize)+global_id - D3] + (dE_RIGHT_13_1* (double)(F2[RHO*(ksize)+global_id - D3] <= 0.0) + dE_RIGHT_13_2* (double)(F2[RHO*(ksize)+global_id - D3]>0.0))) +
			+(F3[B2*(ksize)+global_id] - (dE_LEFT_12_1* (double)(F3[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_12_2* (double)(F3[RHO*(ksize)+global_id]>0.0)))
			+ (F3[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_12_1* (double)(F3[RHO*(ksize)+global_id - D2*jsize] <= 0.0) + dE_RIGHT_12_2* (double)(F3[RHO*(ksize)+global_id - D2*jsize]>0.0))));
		emf[2 * (ksize)+global_id] = 0.25*((-F3[B1*(ksize)+global_id] - (dE_LEFT_21_1* (double)(F3[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_21_2* (double)(F3[RHO*(ksize)+global_id]>0.0)))
			+ (-F3[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_21_1* (double)(F3[RHO*(ksize)+global_id - D1*isize] <= 0.0) + dE_RIGHT_21_2* (double)(F3[RHO*(ksize)+global_id - D1*isize]>0.0)))
			+ (F1[B3*(ksize)+global_id] - (dE_LEFT_23_1* (double)(F1[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_23_2* (double)(F1[RHO*(ksize)+global_id]>0.0)))
			+ (F1[B3*(ksize)+global_id - D3] + (dE_RIGHT_23_1* (double)(F1[RHO*(ksize)+global_id - D3] <= 0.0) + dE_RIGHT_23_2* (double)(F1[RHO*(ksize)+global_id - D3]>0.0))));
		emf[3 * (ksize)+global_id] = 0.25*((F2[B1*(ksize)+global_id] - (dE_LEFT_31_1* (double)(F2[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_31_2* (double)(F2[RHO*(ksize)+global_id]>0.0)))
			+ (F2[B1*(ksize)+global_id - D1*isize] + (dE_RIGHT_31_1* (double)(F2[RHO*(ksize)+global_id - D1*isize] <= 0.0) + dE_RIGHT_31_2* (double)(F2[RHO*(ksize)+global_id - D1*isize]>0.0)))
			+ (-F1[B2*(ksize)+global_id] - (dE_LEFT_32_1* (double)(F1[RHO*(ksize)+global_id] <= 0.0) + dE_LEFT_32_2* (double)(F1[RHO*(ksize)+global_id]>0.0)))
			+ (-F1[B2*(ksize)+global_id - D2*jsize] + (dE_RIGHT_32_1* (double)(F1[RHO*(ksize)+global_id - D2*jsize] <= 0.0) + dE_RIGHT_32_2* (double)(F1[RHO*(ksize)+global_id - D2*jsize] >0.0))));

		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == BS_2 + N2G)){
			emf[3 * (ksize)+global_id] = 0.;
			emf[1 * (ksize)+global_id] = -0.5*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id - D3]);
		}
	}
}

__global__ void consttransport3(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0, i, imin[3], jmin[3], zmin[3], imax[3], jmax[3], zmax[3];
	isize = (BS_3 + D3)*(BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id<(BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	for (i = 0; i < 3; i++){
		imin[i] = N1G;
		jmin[i] = N2G;
		zmin[i] = N3G;
		imax[i] = BS_1 + N1G;
		jmax[i] = BS_2 + N2G;
		zmax[i] = BS_3 + N3G;
	}
	imax[0] += D1;
	jmax[1] += D2;
	zmax[2] += D3;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel=0, zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	#if(NSY)
	int index1 = FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	int index2 = FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif

	if (icurr >= imin[0] && jcurr >= jmin[0] && zcurr >= zmin[0] && icurr<imax[0] && jcurr<jmax[0]  && zcurr<zmax[0] && k==1){
		if (zoffset == 0){
			temp = 0.;
			for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[global_id - zoffset + u] * gdet_GPU[index1 - NSY*(zoffset - u)];
			for (u = 0; u < zsize; u++){
				temp += -Dt / ((double)zsize*dx_2)*(E_corn[3 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] - E_corn[3 * ksize + global_id - zoffset + u]) ;
			}
			#if(N3G>0)
			temp += Dt / ((double)zsize*dx_3)*(E_corn[2 * ksize + global_id - zoffset + D3*zsize] - E_corn[2 * ksize + global_id - zoffset]) ;
			#endif
			for (u = 0; u < zsize; u++)psf[global_id - zoffset + u] = temp / gdet_GPU[index1 - NSY*(zoffset - u)];
		}
	}

	if (icurr >= imin[2] && jcurr >= jmin[2] && zcurr >= zmin[2] && icurr<imax[2] && jcurr<jmax[2] && zcurr<zmax[2] && k == 1){
		#if(N3G>0)
		if (zoffset == 0){
			temp = psi[2 * ksize + global_id - zoffset] * gdet_GPU[index3 - NSY*(zoffset)];
			temp +=  - Dt / dx_1*(E_corn[2 * ksize + global_id + isize - zoffset] - E_corn[2 * ksize + global_id - zoffset]) ;
			temp += Dt / dx_2*(E_corn[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset] - E_corn[1 * ksize + global_id - zoffset]);
			if (zcurr == BS_3 + N3G) zsize = 1;
			for (u = 0; u < zsize; u++)psf[2 * ksize + global_id - zoffset + u] = temp / gdet_GPU[index3 - NSY*(zoffset-u)];
		}
		#endif
	}

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - D2 - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif
	if (icurr >= imin[1] && jcurr >= jmin[1] && zcurr >= zmin[1] && icurr<imax[1] && jcurr<jmax[1] && zcurr<zmax[1] && k == 1){
		if (zoffset == 0){
			temp = 0.;
			for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[1 * ksize + global_id - zoffset + u] * gdet_GPU[index2 - NSY*(zoffset - u)];
			for (u = 0; u < zsize; u++){
				temp += Dt / ((double)zsize*dx_1)*(E_corn[3 * ksize + global_id + isize - zoffset + u] - E_corn[3 * ksize + global_id - zoffset + u]) ;
			}
			#if(N3G>0)
			temp += -Dt / ((double)zsize*dx_3)*(E_corn[1 * ksize + global_id - zoffset + D3*zsize] - E_corn[1 * ksize + global_id - zoffset]);
			#endif
			for (u = 0; u < zsize; u++)psf[1 * ksize + global_id - zoffset + u] = temp / gdet_GPU[index2 - NSY*(zoffset - u)];
		}
	}
}

__global__ void consttransport3_post(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int icurr, jcurr, zcurr, k=0, isize;
	if (global_id < (BS_2 + D2)*(BS_3 + D3)){
		k = 1;
		global_id -= 0;
		icurr = 0;
		zcurr = global_id%(BS_3 + D3);
		jcurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= (BS_2 + D2)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3)){
		k = 2;
		global_id -= (BS_2 + D2)*(BS_3 + D3);
		icurr = BS_1 - 1;
		zcurr = global_id%(BS_3 + D3);
		jcurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3)){
		k = 3;
		global_id -= 2 * (BS_2 + D2)*(BS_3 + D3);
		jcurr = 0;
		zcurr = global_id%(BS_3 + D3);
		icurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3)){
		k = 4;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + (BS_1+ D1)*(BS_3 + D3));
		jcurr = BS_2 - 1;
		zcurr = global_id%(BS_3 + D3);
		icurr = (global_id - zcurr) / (BS_3 + D3);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2)){
		k = 5;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3));
		zcurr = 0;
		jcurr = global_id%(BS_2 + D2);
		icurr = (global_id - jcurr) / (BS_2 + D2);
	}
	else if (global_id >= 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2) && global_id < 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_2 + D2)){
		k = 6;
		global_id -= (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1+ D1)*(BS_3 + D3) + (BS_1+ D1)*(BS_2 + D2));
		zcurr = BS_3 - 1;
		jcurr = global_id%(BS_2 + D2);
		icurr = (global_id - jcurr) / (BS_2 + D2);
	}
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel=0, zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	#if(NSY)
	int index1 = FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr+(k==2))*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	int index2 = FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + (k == 4))*(BS_3 + 2 * N3G) + zcurr;
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + (zcurr + (k==6));
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + (k == 2))*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (jcurr + (k == 4));
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif

	if (k >= 1){
		if (icurr >= N1G + (k != 1) && jcurr >= N2G && zcurr >= N3G + (k == 3 || k == 4) && icurr < BS_1 + N1G + D1 - (k != 2) && jcurr < BS_2 + N2G  && zcurr < BS_3 + N3G - (k == 3 || k == 4)){
			if (zoffset == 0){
				temp = 0.;
				for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[global_id + (k == 2)*isize - zoffset + u] * gdet_GPU[index1 - NSY*(zoffset - u)];
				for (u = 0; u < zsize; u++){
					temp += -Dt / ((double)zsize*dx_2)*(E_corn[3 * ksize + global_id + (k == 2)*isize + (BS_3 + 2 * N3G) - zoffset + u] - E_corn[3 * ksize + global_id + (k == 2)*isize - zoffset + u]);
				}
				#if(N3G>0)
				temp += Dt / ((double)zsize*dx_3)*(E_corn[2 * ksize + global_id + (k == 2)*isize - zoffset + D3*zsize] - E_corn[2 * ksize + global_id + (k == 2)*isize - zoffset]);
				for (u = 0; u < zsize; u++)psf[global_id + (k == 2)*isize - zoffset + u] = temp / gdet_GPU[index1 - NSY*(zoffset - u)];
				#endif
			}
		}

		if (icurr >= N1G && jcurr >= N2G + (k == 1 || k == 2) && zcurr >= N3G + (k != 5) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G - (k == 1 || k == 2) && zcurr < BS_3 + N3G + D3 - (k != 6)){
			#if(N3G>0)
			if (zoffset == 0){
				temp = psi[2 * ksize + global_id - zoffset + (k == 6)] * gdet_GPU[index3 - NSY*(zoffset)];
				temp += -Dt / dx_1*(E_corn[2 * ksize + global_id + isize - zoffset + (k == 6)] - E_corn[2 * ksize + global_id - zoffset + (k == 6)]) ;
				temp += Dt / dx_2*(E_corn[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + (k == 6)] - E_corn[1 * ksize + global_id - zoffset + (k == 6)]);
				if (zcurr == BS_3 + N3G) zsize = 1;
				for (u = 0; u < zsize; u++)psf[2 * ksize + global_id - zoffset + u + (k == 6)] = temp / gdet_GPU[index3 - NSY*(zoffset-u)];
			}
			#endif
		}

		#if(N_LEVELS_1D_INT>0 && D3>0)
		if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
		if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (D2 + BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
		zsize = (int)(0.001+pow(2.0, (double)zlevel));
		zoffset = (zcurr - N3G) % zsize;
		#endif
		if (icurr >= N1G && jcurr >= N2G + (k != 3) && zcurr >= N3G + (k == 1 || k == 2) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G + D2 - (k != 4) && zcurr < BS_3 + N3G - (k == 1 || k == 2)){
			if (zoffset == 0){
				temp = 0.;
				for (u = 0; u < zsize; u++) temp += 1.0 / ((double)zsize)*psi[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u] * gdet_GPU[index2 - NSY*(zoffset - u)];
				for (u = 0; u < zsize; u++){
					temp += Dt / ((double)zsize*dx_1)*(E_corn[3 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) + isize - zoffset + u] - E_corn[3 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u]) ;
				}
				temp += -Dt / ((double)zsize*dx_3)*(E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + D3*zsize] - E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset]);
				for (u = 0; u < zsize; u++)psf[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + u] = temp / gdet_GPU[index2 - NSY*(zoffset - u)];
			}
		}
	}
}

__global__ void flux_ct1(const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double *  emf)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + D3)*(BS_2 + D2);
	int zcurr = (global_id % (isize)) % (BS_3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	int icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G + D1 && jcurr<BS_2 + N2G + D2  && zcurr<BS_3 + N3G + D3){
		#if (N2G>0 && N3G>0)
		emf[1 * (ksize)+global_id] = -0.25*(F2[B3*(ksize)+global_id] + F2[B3*(ksize)+global_id - 1] -
			F3[B2*(ksize)+global_id] - F3[B2*(ksize)+global_id - (BS_3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		emf[2 * (ksize)+global_id] = -0.25*(F3[B1*(ksize)+global_id] + F3[B1*(ksize)+global_id - isize] -
			F1[B3*(ksize)+global_id] - F1[B3*(ksize)+global_id - 1]);
		#endif
		#if (N1G>0 && N2G>0)
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (BS_3 + 2 * N3G)] -
			F2[B1*(ksize)+global_id] - F2[B1*(ksize)+global_id - isize]);
		#else
		emf[3 * (ksize)+global_id] = -0.25*(F1[B2*(ksize)+global_id] + F1[B2*(ksize)+global_id - (BS_3 + 2 * N3G)]);
		#endif
	}
}

__global__ void flux_ct2(double *  F1, double *  F2, double *  F3, const  double* __restrict__  emf)
{
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + D3)*(BS_2 + D2);
	int zcurr = (global_id % (isize)) % (BS_3 + D3);
	int jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	int icurr = (global_id - (jcurr*(BS_3 + D3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double emf1 = -emf[1 * (ksize)+global_id];
	double emf2 = -emf[2 * (ksize)+global_id];
	double emf3 = -emf[3 * (ksize)+global_id];
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G + D1 && jcurr<BS_2 + N2G && zcurr<BS_3 + N3G){
		#if (N1G>0)
		F1[B1*(ksize)+global_id] = 0.0;
		#endif
		#if (N1G>0 && N2G>0)
		F1[B2*(ksize)+global_id] = 0.5*(emf3 - emf[3 * (ksize)+global_id + (BS_3 + 2 * N3G)]);
		#endif
		#if (N1G>0 && N3G>0)
		F1[B3*(ksize)+global_id] = -0.5*(emf2 - emf[2 * (ksize)+global_id + 1]);
		#endif
	}
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G && jcurr<BS_2 + N2G + D2 && zcurr<BS_3 + N3G){
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
	if (icurr >= N1G && jcurr >= N2G && zcurr >= N3G && icurr<BS_1 + N1G && jcurr<BS_2 + N2G && zcurr<BS_3 + N3G + D3){
		#if (N1G>0 && N3G>0)
		F3[B1*(ksize)+global_id] = 0.5*(emf2 - emf[2 * (ksize)+global_id + isize]);
		#endif
		#if (N2G>0 && N3G>0)
		F3[B2*(ksize)+global_id] = -0.5*(emf1 - emf[1 * (ksize)+global_id + (BS_3 + 2 * N3G)]);
		#endif
		#if(N3G>0)
		F3[B3*(ksize)+global_id] = 0.;
		#endif
	}
}

__global__ void Utoprim0(const  double* __restrict__ pi_i, const  double* __restrict__ pb_i, double* pf_i, double *  psf,
	const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step)
{

}

__global__ void Utoprim1(double* pi_i, double* pb_i, double* pf_i, double *  psf,
	double *  F1, double *  F2, double *  F3, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step)
{

}

__global__ void Utoprim2(double* __restrict__ pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step)
{

}


//For P100/V100 GPUs replace Utoprim0, Utoprim1, Utoprim2, fixup by this kernel
#if (DOHELM)
__global__ void fixup(double* pi_i, double* pb_i, double* pf_i, double* storage2, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int POLE_1, int POLE_2,
    const double* __restrict__ gpu_eos_table)
#else
__global__ void fixup(double* pi_i, double* pb_i, double* pf_i, double* storage2, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int POLE_1, int POLE_2)
#endif
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize, icurr, jcurr, zcurr, k=0;
	isize = (BS_3)*(BS_2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr*(BS_3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	if (global_id < (BS_1)*(BS_2)*(BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	struct of_geom geom;
	struct of_state q;
	int  dofloor = 0, m;
	double r, uuscal, rhoscal, rhoflr, uuflr;
	double f, gamma, bsq;
	double pf[NPR], pf_prefloor[NPR], dU[NPR], U[NPR];
	double trans, betapar, betasq, betasqmax, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut;
	double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];
	int zsize = 1, zlevel = 0, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1){
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		if (full_step == 0){
			for (k = 0; k < NPR; k++){
				pf[k] = 0.0;
				for (u = 0; u < zsize; u++){
					pf[k] += (1.0/((double)zsize))*pi_i[k*(ksize)+global_id-zoffset+u];
				}
			}
			get_state(pf, &geom, &q);
			primtoU(pf, &q, &geom, U, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                    );
			#pragma unroll 9
			for (k = 0; k<NPR; k++){
				storage2[k*(ksize)+global_id] = U[k];
			}
		}
		else{
			#pragma unroll 9
			for (k = 0; k<NPR; k++){
				U[k] = storage2[k*(ksize)+global_id];
			}
			for (k = 0; k < NPR; k++){
				pf[k] = 0.0;
				for (u = 0; u < zsize; u++){
					pf[k] += (1.0 / ((double)zsize))*pb_i[k*(ksize)+global_id - zoffset + u];
				}
			}
			if (full_step == 1){
				get_state(pf, &geom, &q);
			}
		}
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			for (u = 0; u < zsize; u++){
				#if( N1G > 0 )
				U[k] -= Dt*(F1[k*(ksize)+global_id + isize - zoffset + u] - F1[k*(ksize)+global_id - zoffset + u]) / (dx_1*(double)zsize);
				#endif
				#if( N2G > 0 )
				U[k] -= Dt*(F2[k*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k*(ksize)+global_id - zoffset + u]) / (dx_2*(double)zsize);
				#endif
			}
			#if( N3G > 0 )
			U[k] -= Dt*(F3[k*(ksize)+global_id - zoffset + zsize] - F3[k*(ksize)+global_id - zoffset]) / (dx_3*(double)zsize);
			#endif
		}

		source(pf, &geom, icurr, jcurr, zcurr, dU, Dt, gam, conn, &q, a, radius[icurr]
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			U[k] += Dt*(dU[k]);
		}

		#if(NSY)
		U[B1] = 0.0;
		U[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++){
			U[B1] = (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			U[B2] = (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + D2)*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
		}
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
		#endif
		#endif
		#else
		#if(STAGGERED)
		U[B1] = 0.0;
		U[B2] = 0.0;
		for (u = 0; u < zsize; u++){
			U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_2 + 2 * N2G) + jcurr]) / (2.0*(double)zsize);
			U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0*(double)zsize);
		}
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif
		#endif

		#if(NEWMAN)
		pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                        #if (DOHELM)
                                        , gpu_eos_table
                                        #endif
                                        );
		if (pflag[global_id]){
			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                            #if (DOHELM)
                                            , gpu_eos_table
                                            #endif
                                            );
		}
		#else
		pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                        #if (DOHELM)
                                        , gpu_eos_table
                                        #endif
                                        );
		if (pflag[global_id]) {
			pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
										#if (DOHELM)
										, gpu_eos_table
										#endif
			);
		}
		#endif
		//compute the square of fluid frame magnetic field (twice magnetic pressure)
		#if( DO_FONT_FIX )
		if (pflag[global_id]) {
			failimage[global_id]++;
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

		// Danat: multiply by max. density; change to a device global variable later 
		#if (DOHELM)
		double rhomax_helm = 1e-7;
		rhoflr = RHOMIN * rhoscal * rhomax_helm;
		uuflr = UUMIN * uuscal * rhomax_helm;
		#else
		rhoflr = RHOMIN * rhoscal;
		uuflr = UUMIN * uuscal;
		#endif

		ucon_calc(pf, &geom, q.ucon);
		lower(q.ucon, geom.gcov, q.ucov);
		bcon_calc(pf, q.ucon, q.ucov, q.bcon);
		lower(q.bcon, geom.gcov, q.bcov);
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

		#if( DRIFT_FLOOR )
		if (dofloor && (trans = 10.*bsq / MY_MIN(pf[RHO], pf[UU]) - 1.) > 0.) {
			//ucon_calc(pf_prefloor, &geom, q.ucon) ;
			//lower(q.ucon, geom.gcov, q.ucov) ;
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

			lower(Bcon, geom.gcov, Bcov);
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
		#elif(ZAMO_FLOOR)
		if (dofloor == 1) {
			double dpf[NPR], U_prefloor[NPR],Xtransone_over_ucondr;
			#pragma unroll 9
			PLOOP dpf[k] = pf[k] - pf_prefloor[k];

			//compute the conserved quantity associated with floor addition
			get_state(dpf, &geom, &q);
			primtoU(dpf, &q, &geom, dU, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                    );

			//compute the prefloor conserved quantity
			get_state(pf_prefloor, &geom, &q);
			primtoU(pf_prefloor, &q, &geom, U_prefloor, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                    );

			//add U_added to the current conserved quantity
			#pragma unroll 9
			PLOOP U[k] = U_prefloor[k] + dU[k];

			#if(NEWMAN)
			pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                            #if (DOHELM)
                                            , gpu_eos_table
                                            #endif
                                            );
			if (pflag[global_id]){
				pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
			}
			#else
			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                            #if (DOHELM)
                                            , gpu_eos_table
                                            #endif
                                            );
			if (pflag[global_id]) {
				pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
			}
			#endif
			if (pflag[global_id]){
				failimage[global_id]++;
				#if( DO_FONT_FIX )
				U[KTOT] = (geom.g*pf[0] * (gam - 1.)*pf[1] / pow(pf[0], gam)) * (q.ucon[0]);
				pf[KTOT] = U[KTOT] / U[RHO];
				pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
				if (pflag[global_id]) {
					failimage[1 * (ksize)+global_id]++;
					pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
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


#if (DOHELM)
__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int POLE_1, int POLE_2, const double* __restrict__ gpu_eos_table)
#else
__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
    const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
    const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int POLE_1, int POLE_2)
#endif
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int ki = 0,k=0, ksize, isize, fix_mem1,fix_mem2, icurr,jcurr,zcurr;
	if (global_id < BS_2*BS_3){
		ki = 1;
		global_id -= 0;
		icurr = 0;
		zcurr = global_id%BS_3;
		jcurr = (global_id - zcurr) / BS_3;
		k = 1;
	}
	else if (global_id >= BS_2*BS_3 && global_id < 2 * BS_2*BS_3){
		ki = 2;
		global_id -= BS_2*BS_3;
		icurr = BS_1-1;
		zcurr = global_id%BS_3;
		jcurr = (global_id - zcurr) / BS_3;
		k = 1;
	}
	else if (global_id >= 2*BS_2*BS_3 && global_id < 2 * BS_2*BS_3+BS_1*BS_3){
		ki = 3;
		global_id -= 2*BS_2*BS_3;
		jcurr = 0;
		zcurr = global_id%BS_3;
		icurr = (global_id - zcurr) / BS_3;
		k = 1;
	}
	else if (global_id >= 2 * BS_2*BS_3 + BS_1*BS_3 && global_id < 2 * BS_2*BS_3 + 2*BS_1*BS_3){
		ki = 4;
		global_id -= (2 * BS_2*BS_3 + BS_1*BS_3);
		jcurr = BS_2-1;
		zcurr = global_id%BS_3;
		icurr = (global_id - zcurr) / BS_3;
		k = 1;
	}
	else if (global_id >= 2 * BS_2*BS_3 + 2 * BS_1*BS_3 && global_id < 2 * BS_2*BS_3 + 2 * BS_1*BS_3 + BS_1*BS_2){
		ki = 5;
		global_id -= 2 * BS_2*BS_3 + 2 * BS_1*BS_3;
		zcurr = 0;
		jcurr = global_id%BS_2;
		icurr = (global_id - jcurr) / BS_2;
		k = 1;
	}
	else if (global_id >= 2 * BS_2*BS_3 + 2 * BS_1*BS_3 + BS_1*BS_2 && global_id < 2 * BS_2*BS_3 + 2 * BS_1*BS_3 + 2 * BS_1*BS_2){
		ki = 6;
		global_id -= 2 * BS_2*BS_3 + 2 * BS_1*BS_3 + BS_1*BS_2;
		zcurr = BS_3 - 1;
		jcurr = global_id%BS_2;
		icurr = (global_id - jcurr) / BS_2;
		k = 1;
	}
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	fix_mem2 = fix_mem1;
	#else
	fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	struct of_geom geom;
	struct of_state q;
	int  dofloor = 0, m;
	double r, uuscal, rhoscal, rhoflr, uuflr;
	double f, gamma, bsq;
	double pf[NPR], pf_prefloor[NPR], U[NPR];
	double trans, betapar, betasq, betasqmax, udotB, Bsq, B, wold, wnew, QdotB, x, vpar, one_over_ucondr_t, ut;
	double ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], ucon[NDIM], vcon[NDIM], utcon[NDIM];
	int zsize = 1, zlevel = 0, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k > 0){
		if (icurr >= N1G  && jcurr >= N2G + (ki == 1 || ki == 2) && zcurr >= N3G + (ki == 1 || ki == 2) + (ki == 3 || ki == 4) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G - (ki == 1 || ki == 2) && zcurr < BS_3 + N3G - (ki == 1 || ki == 2) - (ki == 3 || ki == 4)){

			get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);

			for (k = 0; k < NPR; k++){
				pf[k] = 0.0;
				for (u = 0; u < zsize; u++){
					pf[k] += (1.0 / ((double)zsize))*pi_i[k*(ksize)+global_id - zoffset + u];
				}
			}
			get_state(pf, &geom, &q);
			primtoU(pf, &q, &geom, U, gam
                    #if (DOHELM)
                    , gpu_eos_table
                    #endif
                    );

			#pragma unroll 9
			for (k = 0; k<NPR; k++){
				for (u = 0; u < zsize; u++){
					#if( N1G > 0 )
					U[k] -= Dt*(F1[k*(ksize)+global_id + isize - zoffset + u] - F1[k*(ksize)+global_id - zoffset + u]) / (dx_1*(double)zsize);
					#endif
					#if( N2G > 0 )
					U[k] -= Dt*(F2[k*(ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k*(ksize)+global_id - zoffset + u]) / (dx_2*(double)zsize);
					#endif
				}
				#if( N3G > 0 )
				U[k] -= Dt*(F3[k*(ksize)+global_id - zoffset + zsize] - F3[k*(ksize)+global_id - zoffset]) / (dx_3*(double)zsize);
				#endif
			}

			#if(NSY)
			U[B1] = 0.0;
			U[B2] = 0.0;
			#if(STAGGERED)
			for (u = 0; u < zsize; u++){
				U[B1] = (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
				U[B2] = (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + D2)*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			}
			#if(N3G>0)
			U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
			#endif
			#endif
			#else
			#if(STAGGERED)
			U[B1] = 0.0;
			U[B2] = 0.0;
			for (u = 0; u < zsize; u++){
				U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_2 + 2 * N2G) + jcurr]) / (2.0*(double)zsize);
				U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0*(double)zsize);
			}
			#if(N3G>0)
			U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr]) / 2.0;
			#endif
			#endif
			#endif

			#if(NEWMAN)
			pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                            #if (DOHELM)
                                            , gpu_eos_table
                                            #endif
                                            );
			if (pflag[global_id]){
				pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
			}
			#else
			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                            #if (DOHELM)
                                            , gpu_eos_table
                                            #endif
                                            );
			if (pflag[global_id]) {
				pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
			}
			#endif

			//compute the square of fluid frame magnetic field (twice magnetic pressure)
			#if( DO_FONT_FIX )
			if (pflag[global_id]) {
				failimage[global_id]++;
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

			// Danat: multiply by max. density; change to a device global variable later 
			#if (DOHELM)
			double rhomax_helm = 1e-7;
			rhoflr = RHOMIN * rhoscal * rhomax_helm;
			uuflr = UUMIN * uuscal * rhomax_helm;
			#else
			rhoflr = RHOMIN*rhoscal;
			uuflr = UUMIN*uuscal;
			#endif

			ucon_calc(pf, &geom, q.ucon);
			lower(q.ucon, geom.gcov, q.ucov);
			bcon_calc(pf, q.ucon, q.ucov, q.bcon);
			lower(q.bcon, geom.gcov, q.bcov);
			bsq = dot(q.bcon, q.bcov);

			//tie floors to the local values of magnetic field and internal energy density
			if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);
			if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
			if (rhoflr < pf[UU] / UORHOMAX) rhoflr = pf[UU] / (UORHOMAX);

			if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
			if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;

			//floor on density and internal energy density (momentum *not* conserved)
			#pragma unroll 9
			PLOOP pf_prefloor[k] = pf[k];
			if (pf[RHO] < rhoflr){
				pf[RHO] = rhoflr;
				dofloor = 1;
			}
			if (pf[UU] < uuflr){
				pf[UU] = uuflr;
				dofloor = 1;
			}

			#if( DRIFT_FLOOR )
			if (dofloor && (trans = 10.*bsq / MY_MIN(pf[RHO], pf[UU]) - 1.) > 0.) {
				//ucon_calc(pf_prefloor, &geom, q.ucon) ;
				//lower(q.ucon, geom.gcov, q.ucov) ;
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

				lower(Bcon, geom.gcov, Bcov);
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
			#elif(ZAMO_FLOOR)
			if (dofloor == 1) {
				double dpf[NPR], U_prefloor[NPR], Xtransone_over_ucondr;
				#pragma unroll 9
				PLOOP dpf[k] = pf[k] - pf_prefloor[k];

				//compute the conserved quantity associated with floor addition
				get_state(dpf, &geom, &q);
				primtoU(dpf, &q, &geom, dU, gam
                        #if (DOHELM)
                        , gpu_eos_table
                        #endif
                        );

				//compute the prefloor conserved quantity
				get_state(pf_prefloor, &geom, &q);
				primtoU(pf_prefloor, &q, &geom, U_prefloor, gam
                        #if (DOHELM)
                        , gpu_eos_table
                        #endif
                        );

				//add U_added to the current conserved quantity
				#pragma unroll 9
				PLOOP U[k] = U_prefloor[k] + dU[k];

				#if(NEWMAN)
				pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
				if (pflag[global_id]){
					pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                                    #if (DOHELM)
                                                    , gpu_eos_table
                                                    #endif
                                                    );
				}
				#else
				pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf
                                                #if (DOHELM)
                                                , gpu_eos_table
                                                #endif
                                                );
				if (pflag[global_id]) {
					pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf
                                                    #if (DOHELM)
                                                    , gpu_eos_table
                                                    #endif
                                                    );
				}
				#endif
				if (pflag[global_id]){
					failimage[global_id]++;
					#if( DO_FONT_FIX )
					U[KTOT] = (geom.g*pf[0] * (gam - 1.)*pf[1] / pow(pf[0], gam)) * (q.ucon[0]);
					pf[KTOT] = U[KTOT] / U[RHO];
					pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
					if (pflag[global_id]) {
						failimage[1 * (ksize)+global_id]++;
						pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, pf[KTOT]);
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
			for (k = 0; k < NPR - DOKTOT; k++){
				pf_i[k*(ksize)+global_id] = pf[k];
			}
		}
	}
}

__global__ void cleanup_post(double* F1, double* F2, double* F3, double* E_corn)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3+2*N3G)*(BS_2+2*N2G);
	int zcurr = (global_id % (isize)) % (BS_3+2*N3G);
	int jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * N3G);
	int icurr = (global_id - (jcurr*(BS_3+2*N3G) + zcurr)) / (isize);
	int k = 0;
	if (global_id<(BS_1+2*N1G)*(BS_2+2*N2G)*(BS_3+2*N3G)) k = 1;
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	if (k == 1){
		for (k = 0; k < NPR; k++){
			F1[k*ksize + global_id] = 0.;
			F2[k*ksize + global_id] = 0.;
			F3[k*ksize + global_id] = 0.;
		}
		for (k = 0; k < NDIM; k++) E_corn[k*ksize + global_id] = 0.;
	}
}


__global__ void fixuputoprim(double *  pv, int *  pflag, int *  failimage)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3)*(BS_2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr*(BS_3) + zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	if (global_id < (BS_1)*(BS_2)*(BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR];
	int counter = 0;

	/* Fix the interior points first */
	if (k==1) {
		if (pflag[global_id] != 0) {
			for (k = 0; k < NPR; k++) avg[k] = 0.;
			if (icurr - 1 >= N1G){
				if (pflag[global_id - isize] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id - isize];
					avg[KTOT] += pv[KTOT * (ksize)+global_id - isize];
					counter++;
				}
			}
			if (icurr + 1 < BS_1 + N1G){
				if (pflag[global_id + isize] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id + isize];
					avg[KTOT] += pv[KTOT * (ksize)+global_id + isize];
					counter++;
				}
			}
			if (jcurr - 1 >= N2G){
				if (pflag[global_id - (BS_3 + 2 * N3G)] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id - (BS_3 + 2 * N3G)];
					avg[KTOT] += pv[KTOT * (ksize)+global_id - (BS_3 + 2 * N3G)];
					counter++;
				}
			}
			if (jcurr + 1 < BS_2 + N2G){
				if (pflag[global_id + (BS_3 + 2 * N3G)] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id + (BS_3 + 2 * N3G)];
					avg[KTOT] += pv[KTOT * (ksize)+global_id + (BS_3 + 2 * N3G)];
					counter++;
				}
			}
			if (zcurr - 1 >= N3G){
				if (pflag[global_id - D3] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id - D3];
					avg[KTOT] += pv[KTOT * (ksize)+global_id - D3];
					counter++;
				}
			}
			if (zcurr + 1 < BS_3 + N3G){
				if (pflag[global_id + D3] == 0){
					for (k = 0; k < B1; k++) avg[k] += pv[k * (ksize)+global_id + D3];
					avg[KTOT] += pv[KTOT * (ksize)+global_id + D3];
					counter++;
				}
			}
			for (k = 0; k < B1; k++) pv[k * (ksize)+global_id] = 1. / ((double)counter)*avg[k];
			pv[KTOT * (ksize)+global_id] = 1. / ((double)counter)*avg[KTOT];
		}
	}
}

__global__ void boundprim1(double *   pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double *  ps)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int k;
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];

	// inner r boundary condition: u, gdet extrapolation
	if (jcurr >= 0 && jcurr<BS_2 + 2 * N2G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_4 == -1){
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
		inflow_check(prim1, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		inflow_check(prim2, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(prim3, 0, jcurr, zcurr, 0, gcov, gcon, gdet);
		#endif
		inflow_check(prim1, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
		inflow_check(prim2, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(prim3, 1, jcurr, zcurr, 0, gcov, gcon, gdet);
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
		zcurr = global_id % (BS_3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	}

	// outer r BC: outflow
	if (jcurr >= 0 && jcurr<BS_2 + 2 * N2G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_2 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim6[k] = pv[k*(ksize)+(BS_1 + N1G - 1)*isize + global_id];
		}

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}

		/*Make sure there is no inflow at outer boundary*/
		inflow_check(prim3, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		inflow_check(prim4, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet);
		#endif
		inflow_check(prim3, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		inflow_check(prim4, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet);
		#endif

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+(BS_1 + N1G)*isize + global_id] = prim3[k];
			pv[k*(ksize)+(BS_1 + N1G + 1)*isize + global_id] = prim4[k];
		#if(N1G==3)
			pv[k*(ksize)+(BS_1 + N1G + 2)*isize + global_id] = prim5[k];
		#endif
		}
		#if(STAGGERED)
		ps[1 * (ksize)+(BS_1 + N1G)*isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		ps[1 * (ksize)+(BS_1 + N1G + 1)*isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G)*isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G + 1)*isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		#if(N1G==3)
		ps[1 * (ksize)+(BS_1 + N1G + 2)*isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G + 2)*isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1)*isize + global_id];
		#endif
		#endif
	}
}

__global__ void boundprim2(double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
{
	int j, jref, k;
	  int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	jref = POLEFIX;

	// polar BCs
	if (icurr >= 0 && icurr<BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_1 == -1) {
		for (j = 0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[3 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			//everything else copy (both poles)
			pv[0 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[0 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[1 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[1 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[2 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[4 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[4 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#if (N2G==0)
			//pv[7*(ksize)+isize*icurr+(j+N2G)*(BS_3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(jref+N2G)*(BS_3+2*N3G)+zcurr];
			#endif
			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif
		}
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[k*(ksize)+isize*icurr + (N2G - 2)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G + 1)*(BS_3 + 2 * N3G) + zcurr];
			#if(N2G==3)
			pv[k*(ksize)+isize*icurr + (N2G - 3)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (N2G + 2)*(BS_3 + 2 * N3G) + zcurr];
			#endif
		}

		// make sure b and u are antisymmetric at the poles
		for (j = 0; j<N2G; j++) {
			pv[3 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
			pv[6 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
		}

		#if(STAGGERED)
		ps[0 * (ksize)+isize*icurr + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G)*(BS_3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+isize*icurr + (N2G - 2)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G + 1)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 1)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 2)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G + 1)*(BS_3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+isize*icurr + (N2G - 3)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (N2G + 2)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (N2G - 3)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (N2G + 2)*(BS_3 + 2 * N3G) + zcurr];
		#endif
		#endif
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}

	if (global_id<(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}
	else if (global_id >= (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		global_id = global_id - (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
		zcurr = global_id % (BS_3 + 2 * N3G);
		icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	}

	if (icurr >= 0 && icurr<BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_3 == -1) {
		for (j = 0; j<jref; j++){
			//linear interpolation of transverse velocity (both poles)
			pv[3 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[3 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			//everything else copy (both poles)
			pv[0 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[0 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[1 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[1 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[2 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[2 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[4 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[4 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#if (N2G==0)
			//pv[7*(ksize)+isize*icurr+(BS_2-1-j+N2G)*(BS_3+2*N3G)+zcurr] = pv[7*(ksize)+isize*icurr+(BS_2-1-jref+N2G)*(BS_3+2*N3G)+zcurr];
			#endif
			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif
		}
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k*(ksize)+isize*icurr + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
			pv[k*(ksize)+isize*icurr + (BS_2 + N2G + 1)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (BS_2 + N2G - 2)*(BS_3 + 2 * N3G) + zcurr];
			#if(N2G==3)
			pv[k*(ksize)+isize*icurr + (BS_2 + N2G + 2)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (BS_2 + N2G - 3)*(BS_3 + 2 * N3G) + zcurr];
			#endif
		}

		// make sure b and u are antisymmetric at the poles
		for (j = BS_2 + N2G; j<BS_2 + 2 * N2G; j++) {
			pv[3 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
			pv[6 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
		}

		#if(STAGGERED)
		ps[0 * (ksize)+isize*icurr + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+isize*icurr + (BS_2 + N2G + 1)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (BS_2 + N2G - 2)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (BS_2 + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (BS_2 + N2G - 1)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (BS_2 + N2G + 1)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (BS_2 + N2G - 2)*(BS_3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+isize*icurr + (BS_2 + N2G + 2)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (BS_2 + N2G - 3)*(BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+isize*icurr + (BS_2 + N2G + 2)*(BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+isize*icurr + (BS_2 + N2G - 3)*(BS_3 + 2 * N3G) + zcurr];
		#endif
		#endif
	}
}

__global__ void boundprim_trans(double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
{
	int j, k;
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;

	// polar BCs
	if (icurr >= 0 && icurr<BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_1 == 1) {
		for (j = -N2G; j < 0; j++){
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				pv[k*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (-j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			}
			pv[U2*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[U3*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[B2*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[B3*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;

			#if(STAGGERED)
			ps[0 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (-j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			ps[2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -ps[2 * (ksize)+isize*icurr + (-j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			#endif
		}
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}

	if (global_id<(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}
	else if (global_id >= (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)){
		global_id = global_id - (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
		zcurr = global_id % (BS_3 + 2 * N3G);
		icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	}

	if (icurr >= 0 && icurr<BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_3 == 1) {
		for (j = BS_2; j < BS_2 + N2G; j++){
			#pragma unroll 9
			for (k = 0; k < NPR; k++){
				pv[k*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[k*(ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			}
			pv[U2*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[U3*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[B2*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[B3*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;

			#if(STAGGERED)
			ps[0 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			ps[2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -ps[2 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			#endif
		}
	}
}

#if(FRAME_TRANSFORM)
__global__ void fluxcalc2D_FT(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet,
	const  double* __restrict__ Mud, const  double* __restrict__ Mud_inv, int lim, int dir, double gam, double cour, double*  dtij, int POLE_1, int POLE_2, double dx_1, double dx_2, double dx_3, int calc_time)
{
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	int local_id = threadIdx.x;
	int group_id = blockIdx.x;
	int local_size = blockDim.x;
	__shared__ double local_dtij[LOCAL_WORK_SIZE];
	int k = 0;
	int isize, icurr, jcurr, zcurr;
	isize = (BS_3 + 2 * D3 - (dir == 3))*(BS_2 + 2 * D2 - (dir == 2));
	zcurr = (global_id % (isize)) % (BS_3 + 2 * D3 - (dir == 3));
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * D3 - (dir == 3));
	icurr = (global_id - (jcurr*(BS_3 + 2 * D3 - (dir == 3)) + zcurr)) / (isize);
	zcurr += (N3G - 1)*D3 + (dir == 3);
	jcurr += (N2G - 1)*D2 + (dir == 2);
	icurr += (N1G - 1)*D1 + (dir == 1);
	if (global_id<(BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) k = 1;
	isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	global_id = isize*icurr + (BS_3 + 2 * N3G)*jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int idel, jdel, zdel, i, i1, j1, j2;
	int face;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	int zsize = 1, zlevel = 0, zoffset = 0;
	double factor;
	double cmax_r, cmin_r, ctop, cmax_l, cmin_l, cmax[2], cmin[2], cmax_roe, cmin_roe;
	double p_l[NPR], p_r[NPR], F1[NPR], F_FT[2][NPR], F_HLL[2][NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR];
	double l_ucon[NDIM], r_ucon[NDIM], l_bcon[NDIM], r_bcon[NDIM], int_velocity;
	struct of_geom geom;
	struct of_state state_l, state_r;
	struct of_trans trans;
	local_dtij[local_id] = 1.e9;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001 + pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx_1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx_2; }
	else if (dir == 3) {idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx_3*((double)zsize);}

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, face, &geom, gcov, gcon, gdet);

		if (zoffset != 0 && dir == 3) {
			#pragma unroll 9
			for (k = 0; k < NPR; k++) {
				p_l[k] = 0.5*(pv[k*(ksize)+global_id] + pv[k*(ksize)+global_id - D3]);
				p_r[k] = p_l[k];
			}
		}
		else {
			#pragma unroll 9
			for (k = 0; k < NPR; k++) {
				p_l[k] = dq2[k*(ksize)+global_id - idel*isize - jdel*(BS_3 + 2 * N3G) - zdel];
				p_r[k] = dq1[k*(ksize)+global_id];
			}
		}
		#if(STAGGERED)
		for (k = B1; k <= B3; k++) {
			if ((dir == 1 && k == B1) || (dir == 2 && k == B2) || (dir == 3 && k == B3)) {
				p_l[k] = ps[(k - B1)*(ksize)+global_id];
				p_r[k] = p_l[k];
			}

			if (dir == 2 && k == B1 && ((jcurr == BS_2 + N2G && POLE_2 == 1) || (jcurr == N2G && POLE_1 == 1))) {
				p_l[k] = 0.;
				p_r[k] = 0.;
			}
		}
		#endif

		//Get interface velocity
		int_velocity = geom.gcon[dir] / (sqrt(geom.gcon[dir] * geom.gcon[dir] - geom.gcon[0] * geom.gcon[4 * (dir == 1) + 7 * (dir == 2) + 9 * (dir == 3)]));

		//First calculate HLL fluxes for F[B1], F[B2] and F[B3]
		get_state(p_l, &geom, &state_l);
		get_state(p_r, &geom, &state_r);

		vchar(p_l, &state_l, &geom, dir, &(cmax_l), &(cmin_l));
		vchar(p_r, &state_r, &geom, dir, &(cmax_r), &(cmin_r));

		cmax[1] = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
		cmin[1] = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
		ctop = MY_MAX(cmax[1], cmin[1]);

		//Transform 4 velocities and 4 magnetic fields to orthonormal frame
		get_trans(icurr, jcurr, zcurr, dir, &trans, Mud, Mud_inv);
		for (i1 = 0; i1 < NDIM; i1++) {
			l_ucon[i1] = 0.0;
			l_bcon[i1] = 0.0;
			r_ucon[i1] = 0.0;
			r_bcon[i1] = 0.0;
			for (j1 = 0; j1 < NDIM; j1++) {
				l_ucon[i1] += state_l.ucon[j1] * trans.Mud_inv[i1][j1];
				l_bcon[i1] += state_l.bcon[j1] * trans.Mud_inv[i1][j1];
				r_ucon[i1] += state_r.ucon[j1] * trans.Mud_inv[i1][j1];
				r_bcon[i1] += state_r.bcon[j1] * trans.Mud_inv[i1][j1];
			}
		}

		primtoflux_FT(p_l, l_ucon, l_bcon, dir, F_l);
		primtoflux_FT(p_r, r_ucon, r_bcon, dir, F_r);
		primtoflux_FT(p_l, l_ucon, l_bcon, 0, U_l);
		primtoflux_FT(p_r, r_ucon, r_bcon, 0, U_r);

		vchar_FT(p_l, l_ucon, l_bcon, dir, &(cmax_l), &(cmin_l));
		vchar_FT(p_r, r_ucon, r_bcon, dir, &(cmax_r), &(cmin_r));

		//Get wavespeed defined as maximum of left and right state
		cmax_roe = MY_MAX(cmax_r, cmax_l);
		cmin_roe = MY_MIN(cmin_r, cmin_l);

		//Get interface velocity
		int_velocity = geom.gcon[dir] / (sqrt(geom.gcon[dir] * geom.gcon[dir] - geom.gcon[0] * geom.gcon[4 * (dir == 1) + 7 * (dir == 2) + 9 * (dir == 3)]));

		//Get HLL fluxes and conserved states
		if (cmax_roe <= int_velocity) {
			for (k = 0; k < NPR; k++) F_FT[0][k] = U_r[k];
			for (k = 0; k < NPR; k++) F_FT[1][k] = F_r[k];
		}
		else if (cmin_roe >= int_velocity) {
			for (k = 0; k < NPR; k++) F_FT[0][k] = U_l[k];
			for (k = 0; k < NPR; k++) F_FT[1][k] = F_l[k];
		}
		else {
			for (k = 0; k < NPR; k++) F_HLL[0][k] = (F_l[k] - F_r[k] + cmax_roe*U_r[k] - cmin_roe*U_l[k]) / (cmax_roe - cmin_roe + SMALL);
			for (k = 0; k < NPR; k++) F_HLL[1][k] = ((cmax_roe * F_l[k] - cmin_roe * F_r[k] + cmax_roe * cmin_roe * (U_r[k] - U_l[k])) / (cmax_roe - cmin_roe + SMALL));

			int do_hydro = (fabs(F_HLL[0][dir + B1 -1] * F_HLL[0][dir + B1 - 1] * l_ucon[0] * r_ucon[0]) < pow(10., -14.)*fabs(F_HLL[0][UU]));

			if (do_hydro) {
				calc_HLLC_hydro(dir, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, F_FT, F_HLL, F_l, F_r, U_l, U_r);
			}
			else {
				#if(HLLD)
				calc_HLLD(dir, cmin_roe, cmax_roe, int_velocity, l_ucon, r_ucon, F_FT, F_HLL, F_l, F_r, U_l, U_r);
				#elif(HLLC)
				calc_HLLC(dir, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, F_FT, F_HLL, F_l, F_r, U_l, U_r);
				#else
				for (k = 0; k < NPR; k++) {
					F_FT[0][k] = F_HLL[0][k];
					F_FT[1][k] = F_HLL[1][k];
				}
				#endif
			}
		}

		//Transform stress energy tensor from orthonormal frame to coordinate basis
		for (j1 = 0; j1<NDIM; j1++) {
			F1[j1 + UU] = 0.;
			for (j2 = 0; j2<NDIM; j2++) {
				F1[j1 + UU] += F_FT[0][j2 + UU] * trans.Mud[dir][0] * trans.Mud_inv[j2][j1];
				F1[j1 + UU] += F_FT[1][j2 + UU] * trans.Mud[dir][dir] * trans.Mud_inv[j2][j1];
			}
		}

		//Transform (dual) Maxwell tensor from orthonormal frame to coordinate basis. First set 0 component div.B=0 based on values from solver (e.g. HLL)
		F_FT[0][U3] = 0.;
		F_FT[1][U3] = -F_FT[0][B1 - 1 + dir];
		for (j1 = 1; j1<NDIM; j1++) {
			F1[j1 + U3] = 0.;
			for (j2 = 0; j2<NDIM; j2++) {
				F1[j1 + U3] += F_FT[0][j2 + U3] * trans.Mud[dir][0] * trans.Mud[j1][j2];
				F1[j1 + U3] += F_FT[1][j2 + U3] * trans.Mud[dir][dir] * trans.Mud[j1][j2];
			}
		}

		//Transform mass and entropy flux from orthonormal frame to coordinate basis
		F1[RHO] = F_FT[0][RHO] * trans.Mud[dir][0];
		F1[RHO] += F_FT[1][RHO] * trans.Mud[dir][dir];
		F1[KTOT] = F_FT[0][KTOT] * trans.Mud[dir][0];
		F1[KTOT] += F_FT[1][KTOT] * trans.Mud[dir][dir];

		//Make conserved quantity consisten with H-AMR
		F1[UU] += F1[RHO];

		for (k = 0; k<NPR; k++) F[k*(ksize)+global_id] = geom.g*F1[k];

		local_dtij[local_id] = factor / ctop;
	}
	if (calc_time == 1) {
		__syncthreads();
		for (i = local_size / 2; i > 1; i = i / 2) {
			if (local_id < i) {
				local_dtij[local_id] = MY_MIN(local_dtij[local_id], local_dtij[local_id + i]);
			}
			__syncthreads();
		}
		if (local_id == 0) {
			dtij[group_id] = MY_MIN(local_dtij[0], local_dtij[1]);
		}
	}
}
#endif

__device__ void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double A, B, C, D, vcon, ptot;
	int k, fail_HLLC = 0;
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	//Calculate x-component 3-velocity
	A = -F_HLL[1][UU];
	B = -F_HLL[1][UGEN_1] + F_HLL[0][UU];
	C = F_HLL[0][UGEN_1];
	D = B*B - 4.*A*C;
	vcon = (-B - sqrt(D)) / (2.*A);

	//Calculate total pressure ptot=pgas+0.5*bsq
	ptot = -(-F_HLL[1][UU])*vcon + F_HLL[1][UGEN_1];
	if (!(fabs(ptot) > 0.) || (vcon < cmin_roe) || (vcon > cmax_roe) || !(fabs(vcon) > 0.)) fail_HLLC = 1;

	if (cmax_roe > int_velocity && vcon <= int_velocity && fail_HLLC == 0) {
		//Set Rankine-Hugoniot jump conditions
		F_FT[0][RHO] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[RHO];
		F_FT[0][UU] = (cmax_roe*U_r[UU] + U_r[UGEN_1] - ptot*vcon) / (cmax_roe - vcon + SMALL);
		F_FT[0][UGEN_1] = (-F_FT[0][UU] + ptot)*vcon;
		F_FT[0][UGEN_2] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[UGEN_2];
		F_FT[0][UGEN_3] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[UGEN_3];
		F_FT[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_FT[0][BGEN_2] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[BGEN_2];
		F_FT[0][BGEN_3] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[BGEN_3];
		F_FT[0][KTOT] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon + SMALL)*U_r[KTOT];

		//Calculate HLLC flux
		for (k = 0; k < NPR; k++) F_FT[1][k] = (F_r[k] + cmax_roe*(F_FT[0][k] - U_r[k]));
	}
	else if (cmin_roe < int_velocity && vcon >= int_velocity && fail_HLLC == 0) {
		//Set Rankine-Hugoniot jump conditions
		F_FT[0][RHO] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[RHO];
		F_FT[0][UU] = (cmin_roe*U_l[UU] + U_l[UGEN_1] - ptot*vcon) / (cmin_roe - vcon + SMALL);
		F_FT[0][UGEN_1] = (-F_FT[0][UU] + ptot)*vcon;
		F_FT[0][UGEN_2] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[UGEN_2];
		F_FT[0][UGEN_3] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[UGEN_3];
		F_FT[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_FT[0][BGEN_2] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[BGEN_2];
		F_FT[0][BGEN_3] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[BGEN_3];
		F_FT[0][KTOT] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon + SMALL)*U_l[KTOT];

		//Calculate HLLC flux
		for (k = 0; k < NPR; k++) F_FT[1][k] = (F_l[k] + cmin_roe*(F_FT[0][k] - U_l[k]));
	}
	else {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_HLL[0][k];
			F_FT[1][k] = F_HLL[1][k];
		}
	}
}

__device__ void calc_HLLC(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double A, B, C, D, vcon[NDIM], gammasq, ptot, v_dot_B;
	int k, fail_HLLC = 0;
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	//Calculate x-component 3-velocity
	A = -F_HLL[1][UU] - (F_HLL[0][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[1][BGEN_3]);
	B = -F_HLL[1][UGEN_1] + F_HLL[0][UU] + (F_HLL[0][BGEN_2] * F_HLL[0][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[0][BGEN_3]) + (F_HLL[1][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[1][BGEN_3] * F_HLL[1][BGEN_3]);
	C = F_HLL[0][UGEN_1] - (F_HLL[0][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[1][BGEN_3]);
	D = B*B - 4.*A*C;
	vcon[GEN_1] = (-B - sqrt(D)) / (2.*A);

	//Calculate other components 3-velocity
	vcon[GEN_2] = (F_HLL[0][BGEN_2] * vcon[GEN_1] - F_HLL[1][BGEN_2]) / F_HLL[0][BGEN_1];
	vcon[GEN_3] = (F_HLL[0][BGEN_3] * vcon[GEN_1] - F_HLL[1][BGEN_3]) / F_HLL[0][BGEN_1];

	//Calculate gamma factor
	gammasq = 1. / (1. - (vcon[1] * vcon[1] + vcon[2] * vcon[2] + vcon[3] * vcon[3]));

	//Calculate total pressure ptot=pgas+0.5*bsq
	v_dot_B = (vcon[1] * F_HLL[0][B1] + vcon[2] * F_HLL[0][B2] + vcon[3] * F_HLL[0][B3]);
	ptot = -(-F_HLL[1][UU] - F_HLL[0][BGEN_1] * (v_dot_B))*vcon[dir] + F_HLL[1][UGEN_1] + pow(F_HLL[0][BGEN_1], 2.0) / gammasq;

	if (!(fabs(ptot) > 0.) || (vcon[GEN_1] < cmin_roe) || (vcon[GEN_1] > cmax_roe) || !(fabs(vcon[GEN_1]) > 0.)) {
		fail_HLLC = 1;
	}

	if (cmax_roe > int_velocity && vcon[dir] <= int_velocity && fail_HLLC == 0) {
		//Set Rankine-Hugoniot jump conditions
		F_FT[0][RHO] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon[dir] + SMALL)*U_r[RHO];
		F_FT[0][UU] = (cmax_roe*U_r[UU] + U_r[UGEN_1] - ptot*vcon[dir] + v_dot_B*F_HLL[0][BGEN_1]) / (cmax_roe - vcon[dir] + SMALL);
		F_FT[0][UGEN_1] = (-F_FT[0][UU] + ptot)*vcon[dir] - v_dot_B*F_HLL[0][BGEN_1];
		F_FT[0][UGEN_2] = (-F_HLL[0][BGEN_1] * (F_HLL[0][BGEN_2] / (gammasq)+v_dot_B*vcon[GEN_2]) + cmax_roe*U_r[UGEN_2] - F_r[UGEN_2]) / (cmax_roe - vcon[dir] + SMALL);
		F_FT[0][UGEN_3] = (-F_HLL[0][BGEN_1] * (F_HLL[0][BGEN_3] / (gammasq)+v_dot_B*vcon[GEN_3]) + cmax_roe*U_r[UGEN_3] - F_r[UGEN_3]) / (cmax_roe - vcon[dir] + SMALL);
		F_FT[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_FT[0][BGEN_2] = F_HLL[0][BGEN_2];
		F_FT[0][BGEN_3] = F_HLL[0][BGEN_3];
		F_FT[0][KTOT] = (cmax_roe - r_ucon[dir] / r_ucon[0]) / (cmax_roe - vcon[dir] + SMALL)*U_r[KTOT];

		//Calculate HLLC flux
		for (k = 0; k < NPR; k++) F_FT[1][k] = (F_r[k] + cmax_roe*(F_FT[0][k] - U_r[k]));
	}
	else if (cmin_roe < int_velocity && vcon[dir] >= int_velocity && fail_HLLC == 0) {
		//Set Rankine-Hugoniot jump conditions
		F_FT[0][RHO] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon[dir] + SMALL)*U_l[RHO];
		F_FT[0][UU] = (cmin_roe*U_l[UU] + U_l[UGEN_1] - ptot*vcon[dir] + v_dot_B*F_HLL[0][BGEN_1]) / (cmin_roe - vcon[dir] + SMALL);
		F_FT[0][UGEN_1] = (-F_FT[0][UU] + ptot)*vcon[dir] - v_dot_B*F_HLL[0][BGEN_1];
		F_FT[0][UGEN_2] = (-F_HLL[0][BGEN_1] * (F_HLL[0][BGEN_2] / (gammasq)+v_dot_B*vcon[GEN_2]) + cmin_roe*U_l[UGEN_2] - F_l[UGEN_2]) / (cmin_roe - vcon[dir] + SMALL);
		F_FT[0][UGEN_3] = (-F_HLL[0][BGEN_1] * (F_HLL[0][BGEN_3] / (gammasq)+v_dot_B*vcon[GEN_3]) + cmin_roe*U_l[UGEN_3] - F_l[UGEN_3]) / (cmin_roe - vcon[dir] + SMALL);
		F_FT[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_FT[0][BGEN_2] = F_HLL[0][BGEN_2];
		F_FT[0][BGEN_3] = F_HLL[0][BGEN_3];
		F_FT[0][KTOT] = (cmin_roe - l_ucon[dir] / l_ucon[0]) / (cmin_roe - vcon[dir] + SMALL)*U_l[KTOT];

		//Calculate HLLC flux
		for (k = 0; k < NPR; k++) F_FT[1][k] = (F_l[k] + cmin_roe*(F_FT[0][k] - U_l[k]));
	}
	else {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_HLL[0][k];
			F_FT[1][k] = F_HLL[1][k];
		}
	}
}

__device__ void calc_HLLD(int dir, double cmin_roe, double cmax_roe, double int_velocity, double l_ucon[NDIM], double r_ucon[NDIM], double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double K_al[NDIM], B_al[NDIM], K_ar[NDIM], B_ar[NDIM], vcon_al[NDIM], vcon_ar[NDIM], eta_l, eta_r, w_al, w_ar, vcon_cl[NDIM], vcon_cr[NDIM], B_c[NDIM], R_l[NPR], R_r[NPR], ptot;
	int k, fail_HLLC = 0, fail_HLLD = 0;

	for (k = 0; k < NPR; k++) R_l[k] = (cmin_roe*U_l[k] - F_l[k]);
	for (k = 0; k < NPR; k++) R_r[k] = (cmax_roe*U_r[k] - F_r[k]);

	//Calculate necessary pressure using Newton Raphson solve
	ptot = calc_HLLD_pres(dir, &fail_HLLC, &fail_HLLD, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, K_al, B_al, K_ar, B_ar, vcon_al, vcon_ar, &eta_l, &eta_r, &w_al, &w_ar, vcon_cl, vcon_cr, F_FT, F_HLL, F_l, F_r, U_l, U_r, R_l, R_r, B_c);

	//Check generated parameters for consistency if previous line did not fail
	if (fail_HLLD == 0) check_HLLD_par(dir, &fail_HLLD, cmin_roe, cmax_roe, ptot, w_al, w_ar, eta_l, eta_r, vcon_cl, vcon_cr, vcon_al, vcon_ar, K_al, K_ar, B_c);

	if (fail_HLLD == 0) { //Calculate state using HLLD solver
		calc_HLLD_state(dir, l_ucon, r_ucon, ptot, int_velocity, cmin_roe, cmax_roe, K_al, B_al, K_ar, B_ar, vcon_al, vcon_ar, eta_l, eta_r, w_al, w_ar, vcon_cl, vcon_cr, F_FT, F_HLL, F_l, F_r, U_l, U_r, R_l, R_r, B_c);
	}
	else if (fail_HLLC == 0) { //Calculate using HLLC solver
		calc_HLLC(dir, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, F_FT, F_HLL, F_l, F_r, U_l, U_r);
	}
	else { //Calculate using HLL solver
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_HLL[0][k];
			F_FT[1][k] = F_HLL[1][k];
		}
	}
}

__device__ double calc_HLLD_pres(int dir, int *fail_HLLC, int *fail_HLLD, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double *eta_l, double *eta_r, double *w_al, double *w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]) {
	double A, B, C, D, gammasq, vcon[NDIM], ptot_HLLC, ptot, v_dot_B;
	int keep_iterating = 1;
	int n_iter = 0;
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	/*Provide estimate for ptot from HLLC solver*/
	//Calculate x-component 3-velocity
	A = -F_HLL[1][UU] - (F_HLL[0][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[1][BGEN_3]);
	B = -F_HLL[1][UGEN_1] + F_HLL[0][UU] + (F_HLL[0][BGEN_2] * F_HLL[0][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[0][BGEN_3]) + (F_HLL[1][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[1][BGEN_3] * F_HLL[1][BGEN_3]);
	C = F_HLL[0][UGEN_1] - (F_HLL[0][BGEN_2] * F_HLL[1][BGEN_2] + F_HLL[0][BGEN_3] * F_HLL[1][BGEN_3]);
	D = B*B - 4.*A*C;
	vcon[GEN_1] = (-B - sqrt(MY_MAX(0.,D))) / (2.*A);

	//Calculate other components 3-velocity
	vcon[GEN_2] = (F_HLL[0][BGEN_2] * vcon[GEN_1] - F_HLL[1][BGEN_2]) / F_HLL[0][BGEN_1];
	vcon[GEN_3] = (F_HLL[0][BGEN_3] * vcon[GEN_1] - F_HLL[1][BGEN_3]) / F_HLL[0][BGEN_1];

	//Calculate lorentz factor
	gammasq = 1. / (1. - (vcon[1] * vcon[1] + vcon[2] * vcon[2] + vcon[3] * vcon[3]));

	//If vcon unphysical fail HLLC solver. Still try to obtain HLLD solution
	if ((vcon[dir] < cmin_roe) || (vcon[dir] > cmax_roe) || !(fabs(vcon[dir]) > 0.)) fail_HLLC[0] = 1;

	//Calculate total pressure ESTIMATE based on HLLC solver value: ptot=pgas+0.5*bsq
	v_dot_B = (vcon[1] * F_HLL[0][B1] + vcon[2] * F_HLL[0][B2] + vcon[3] * F_HLL[0][B3]);
	ptot_HLLC = -(-F_HLL[1][UU] - F_HLL[0][BGEN_1] * (v_dot_B))*vcon[dir] + F_HLL[1][UGEN_1] + pow(F_HLL[0][BGEN_1], 2.0) / gammasq;
	ptot = ptot_HLLC;

	//If ptot invalid, tell the code not to use the HLLC solver and revert to hydro estimate for HLLD solver
	if (!(fabs(ptot) > 0.)) {
		fail_HLLC[0] = 1;
		A = 1.;
		B = (-F_HLL[0][UU] - F_HLL[1][UGEN_1]);
		C = -F_HLL[0][UGEN_1] * F_HLL[1][UU] + F_HLL[1][UGEN_1] * F_HLL[0][UU];
		D = B*B - 4.*A*C;
		ptot = (-B + sqrt(MY_MAX(0., D))) / (2.*A);
		if (!(fabs(ptot) > 0.)) {
			fail_HLLD[0] = 1;
			return -10.;
		}
	}

	//Newton Raphson loop to find pressure of intermediate states in HLLD solver
	double error_1, error_2;
	double ptot_old, de_dptot, de_dlptot, dlptot, lptot, d_ptot = 0.;
	error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

	while (keep_iterating) {
		//Calculate error and error/d_ptot
		error_2 = calc_error_HLLD(dir, 0, ptot + pow(10., -8.)*ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

		//Save old value of ptot
		ptot_old = ptot;

		//Make the newton step in log-space
		//de_dptot = (error_2 - error_1) / (pow(10., -8.)*ptot);
		//de_dlptot = de_dptot*ptot;
		//dlptot = error_1 / de_dlptot;
		//lptot = log(ptot_old) - dlptot;
		//ptot = exp(lptot);
		//d_ptot = ptot - ptot_old;

		de_dptot = (error_2 - error_1) / (pow(10., -8.)*ptot);
		d_ptot = error_1 / de_dptot;
		ptot = ptot - d_ptot;

		//Calculate updated value of ptot
		error_2 = error_1;
		error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

		if ((fabs(d_ptot) <= pow(10., -7.)*ptot) || n_iter > 10) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	//If Newton-Raphson solver did not converge, reset ptot to ptot_HLLC and tag fail_HLLD
	if (!(fabs(ptot) > 0.) || ((fabs(d_ptot) > pow(10., -6.)*ptot))) {
		ptot = ptot_HLLC;
		fail_HLLD[0] = 1;
	}

	return ptot;
}

__device__ void calc_HLLD_state(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double ptot, double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double eta_l, double eta_r, double w_al, double w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]) {
	double v_dot_B, F_al[2][NPR], F_ar[2][NPR], F_cl[2][NPR], F_cr[2][NPR];
	int k, GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	//Calculate state between outer waves and alfven waves according to equations 32-34
	if ((cmin_roe < int_velocity) && (int_velocity <= vcon_cl[dir])) {
		v_dot_B = vcon_al[1] * B_al[1] + vcon_al[2] * B_al[2] + vcon_al[3] * B_al[3];
		F_al[0][RHO] = R_l[RHO] / (cmin_roe - vcon_al[GEN_1]);
		F_al[0][UU] = (R_l[UU] - ptot*vcon_al[GEN_1] + v_dot_B*B_al[GEN_1]) / (cmin_roe - vcon_al[GEN_1] + SMALL);
		F_al[0][UGEN_1] = (-F_al[0][UU] + ptot)*vcon_al[GEN_1] - v_dot_B*B_al[GEN_1];
		F_al[0][UGEN_2] = (-F_al[0][UU] + ptot)*vcon_al[GEN_2] - v_dot_B*B_al[GEN_2];
		F_al[0][UGEN_3] = (-F_al[0][UU] + ptot)*vcon_al[GEN_3] - v_dot_B*B_al[GEN_3];
		F_al[0][BGEN_1] = F_HLL[0][BGEN_1]; //Check this logic
		F_al[0][BGEN_2] = B_al[GEN_2];
		F_al[0][BGEN_3] = B_al[GEN_3];
		F_al[0][KTOT] = R_l[KTOT] / (cmin_roe - vcon_al[GEN_1]);
	}

	if ((cmax_roe > int_velocity) && (vcon_cl[dir] <= int_velocity)) {
		v_dot_B = vcon_ar[1] * B_ar[1] + vcon_ar[2] * B_ar[2] + vcon_ar[3] * B_ar[3];
		F_ar[0][RHO] = R_r[RHO] / (cmax_roe - vcon_ar[GEN_1]);
		F_ar[0][UU] = (R_r[UU] - ptot*vcon_ar[GEN_1] + v_dot_B*B_ar[GEN_1]) / (cmax_roe - vcon_ar[GEN_1] + SMALL);
		F_ar[0][UGEN_1] = (-F_ar[0][UU] + ptot)*vcon_ar[GEN_1] - v_dot_B*B_ar[GEN_1];
		F_ar[0][UGEN_2] = (-F_ar[0][UU] + ptot)*vcon_ar[GEN_2] - v_dot_B*B_ar[GEN_2];
		F_ar[0][UGEN_3] = (-F_ar[0][UU] + ptot)*vcon_ar[GEN_3] - v_dot_B*B_ar[GEN_3];
		F_ar[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_ar[0][BGEN_2] = B_ar[GEN_2];
		F_ar[0][BGEN_3] = B_ar[GEN_3];
		F_ar[0][KTOT] = R_r[KTOT] / (cmax_roe - vcon_ar[GEN_1]);
	}

	if ((K_al[dir] < int_velocity) && (int_velocity <= vcon_cl[dir])) {
		v_dot_B = vcon_cl[1] * B_c[1] + vcon_cl[2] * B_c[2] + vcon_cl[3] * B_c[3];
		F_cl[0][RHO] = F_al[0][RHO] * (K_al[GEN_1] - vcon_al[GEN_1]) / (K_al[GEN_1] - vcon_cl[GEN_1]);
		F_cl[0][UU] = -(-K_al[GEN_1] * F_al[0][UU] - F_al[0][UGEN_1] + ptot*vcon_cl[GEN_1] - v_dot_B*F_al[0][BGEN_1]) / (K_al[GEN_1] - vcon_cl[GEN_1]);
		F_cl[0][UGEN_1] = (-F_al[0][UU] + ptot)*vcon_cl[GEN_1] - v_dot_B*B_c[GEN_1];
		F_cl[0][UGEN_2] = (-F_al[0][UU] + ptot)*vcon_cl[GEN_2] - v_dot_B*B_c[GEN_2];
		F_cl[0][UGEN_3] = (-F_al[0][UU] + ptot)*vcon_cl[GEN_3] - v_dot_B*B_c[GEN_3];
		F_cl[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_cl[0][BGEN_2] = B_c[GEN_2];
		F_cl[0][BGEN_3] = B_c[GEN_3];
		F_cl[0][KTOT] = F_al[0][KTOT] * (K_al[GEN_1] - vcon_al[GEN_1]) / (K_al[GEN_1] - vcon_cl[GEN_1]);
	}

	if ((vcon_cr[dir] <= int_velocity) && (int_velocity < K_ar[dir])) {
		v_dot_B = vcon_cr[1] * B_c[1] + vcon_cr[2] * B_c[2] + vcon_cr[3] * B_c[3];
		F_cr[0][RHO] = F_ar[0][RHO] * (K_ar[GEN_1] - vcon_ar[GEN_1]) / (K_ar[GEN_1] - vcon_cr[GEN_1]);
		F_cr[0][UU] = -(-K_ar[GEN_1] * F_ar[0][UU] - F_ar[0][UGEN_1] + ptot*vcon_cr[GEN_1] - v_dot_B*F_ar[0][BGEN_1]) / (K_ar[GEN_1] - vcon_cr[GEN_1]);
		F_cr[0][UGEN_1] = (-F_ar[0][UU] + ptot)*vcon_cr[GEN_1] - v_dot_B*B_c[GEN_1];
		F_cr[0][UGEN_2] = (-F_ar[0][UU] + ptot)*vcon_cr[GEN_2] - v_dot_B*B_c[GEN_2];
		F_cr[0][UGEN_3] = (-F_ar[0][UU] + ptot)*vcon_cr[GEN_3] - v_dot_B*B_c[GEN_3];
		F_cr[0][BGEN_1] = F_HLL[0][BGEN_1];
		F_cr[0][BGEN_2] = B_c[GEN_2];
		F_cr[0][BGEN_3] = B_c[GEN_3];
		F_cr[0][KTOT] = F_ar[0][KTOT] * (K_ar[GEN_1] - vcon_ar[GEN_1]) / (K_ar[GEN_1] - vcon_cr[GEN_1]);
	}

	if ((cmin_roe < int_velocity) && (int_velocity <= K_al[dir])) {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_al[0][k];
			F_FT[1][k] = F_l[k] + cmin_roe * (F_al[0][k] - U_l[k]);
		}
	}
	else if ((K_al[dir] < int_velocity) && (int_velocity <= vcon_cl[dir])) {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_cl[0][k];
			F_FT[1][k] = F_l[k] + cmin_roe * (F_al[0][k] - U_l[k]) + K_al[dir] * (F_cl[0][k] - F_al[0][k]);
		}
	}
	else if ((vcon_cr[dir] <= int_velocity) && (int_velocity < K_ar[dir])) {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_cr[0][k];
			F_FT[1][k] = F_r[k] + cmax_roe * (F_ar[0][k] - U_r[k]) + K_ar[dir] * (F_cr[0][k] - F_ar[0][k]);
		}
	}
	else if (((cmax_roe > int_velocity) && (int_velocity > K_ar[dir]))) {
		for (k = 0; k < NPR; k++) {
			F_FT[0][k] = F_ar[0][k];
			F_FT[1][k] = F_r[k] + cmax_roe * (F_ar[0][k] - U_r[k]);
		}
	}
}

//Calculates speed of contact mode and checks that all speeds and parameters of the solution are physical
__device__ void check_HLLD_par(int dir, int * fail_HLLD, double cmin_roe, double cmax_roe, double ptot, double w_al, double w_ar, double eta_l, double eta_r, double vcon_cl[NDIM], double vcon_cr[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double B_c[NDIM]) {
	double vsq;
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	vcon_cl[dir] = (vcon_cl[dir] + vcon_cr[dir])*0.5;
	vcon_cr[dir] = vcon_cl[dir];
	vcon_cl[GEN_2] = (vcon_cl[GEN_2] + vcon_cr[GEN_2])*0.5;
	vcon_cr[GEN_2] = vcon_cl[GEN_2];
	vcon_cl[GEN_3] = (vcon_cl[GEN_3] + vcon_cr[GEN_3])*0.5;
	vcon_cr[GEN_3] = vcon_cl[GEN_3];

	//Check that contact wave lies between inner and outer Alfven speed
	if (vcon_cl[dir] < K_al[dir] || vcon_cl[dir] < cmin_roe) fail_HLLD[0] = 1;
	if (vcon_cl[dir] > K_ar[dir] || vcon_cl[dir] > cmax_roe) fail_HLLD[0] = 1;

	//Check that contact wave is going slower than v=0.99c
	vsq = (vcon_cl[1] * vcon_cl[1] + vcon_cl[2] * vcon_cl[2] + vcon_cl[3] * vcon_cl[3]);
	if (!(vsq < 0.99)) fail_HLLD[0] = 1;

	//If wavefan inconsisten revert to HLLC
	if (fabs(w_al) <= fabs(ptot) || vcon_al[dir] <= cmin_roe || K_al[dir] <= cmin_roe || w_al <= 0.) fail_HLLD[0] = 1;
	if (fabs(w_ar) <= fabs(ptot) || vcon_ar[dir] >= cmax_roe || K_ar[dir] >= cmax_roe || w_ar <= 0.) fail_HLLD[0] = 1;

	//Check that v_al is going slower than v=0.99c
	vsq = (vcon_al[1] * vcon_al[1] + vcon_al[2] * vcon_al[2] + vcon_al[3] * vcon_al[3]);
	if (!(vsq < 0.99)) fail_HLLD[0] = 1;

	//Check that v_ar is going slower than v=0.99c
	vsq = (vcon_ar[1] * vcon_ar[1] + vcon_ar[2] * vcon_ar[2] + vcon_ar[3] * vcon_al[3]);
	if (!(vsq < 0.99)) fail_HLLD[0] = 1;

	//Check that left Alfven wave is going slower than v=0.99c
	vsq = (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]);
	if (!(vsq < 0.99)) fail_HLLD[0] = 1;

	//Check that left Alfven wave is going slower than v=0.99c
	vsq = (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]);
	if (!(vsq < 0.99)) fail_HLLD[0] = 1;
}

__device__ double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double vcon_cl[NDIM], double vcon_cr[NDIM], double *eta_l, double *eta_r, double  *w_al, double *w_ar) {
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;
	double A, C, G, X, Q, error = 0.;
	double delta_Kx, Y_l, Y_r, B_hat[NDIM];

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	//Calculate left wave speed in Riemann fan and w=rho+p+u+b^2
	A = R_l[UGEN_1] + cmin_roe*R_l[UU] + ptot * (1. - cmin_roe*cmin_roe);
	G = R_l[BGEN_2] * R_l[BGEN_2] + R_l[BGEN_3] + R_l[BGEN_3];
	C = R_l[UGEN_2] * R_l[BGEN_2] + R_l[UGEN_3] * R_l[BGEN_3];
	Q = -A - G + (BX * BX) * (1. - cmin_roe*cmin_roe);
	X = BX * (A*cmin_roe*BX + C) - (A + G)*(cmin_roe*ptot - R_l[UU]);
	vcon_al[GEN_1] = (BX * (A*BX + cmin_roe*C) - (A + G)*(ptot + R_l[UGEN_1])) ;
	vcon_al[GEN_2] = (Q*R_l[UGEN_2] + R_l[BGEN_2] * (C + BX * (cmin_roe*R_l[UGEN_1] + R_l[UU]))) ;
	vcon_al[GEN_3] = (Q*R_l[UGEN_3] + R_l[BGEN_3] * (C + BX * (cmin_roe*R_l[UGEN_1] + R_l[UU]))) ;
	w_al[0] = ptot + (-R_l[UU] * X - (vcon_al[GEN_1] * R_l[UGEN_1] + vcon_al[GEN_2] * R_l[UGEN_2] + vcon_al[GEN_3] * R_l[UGEN_3])) / (cmin_roe*X - vcon_al[GEN_1] + SMALL);
	vcon_al[GEN_1] = vcon_al[GEN_1] / X;
	vcon_al[GEN_2] = vcon_al[GEN_2] / X;
	vcon_al[GEN_3] = vcon_al[GEN_3] / X;

	//Calculate magnetic fields according to eq. 21
	B_al[GEN_1] = BX;
	B_al[GEN_2] = -(R_l[BGEN_2] * (cmin_roe*ptot - R_l[UU]) - BX*R_l[UGEN_2]) / A;
	B_al[GEN_3] = -(R_l[BGEN_3] * (cmin_roe*ptot - R_l[UU]) - BX*R_l[UGEN_3]) / A;

	//Calculate right wave speed in Riemann fan and w=rho+p+u+b^2
	A = R_r[UGEN_1] + cmax_roe*R_r[UU] + ptot * (1. - cmax_roe*cmax_roe);
	G = R_r[BGEN_2] * R_r[BGEN_2] + R_r[BGEN_3] + R_r[BGEN_3];
	C = R_r[UGEN_2] * R_r[BGEN_2] + R_r[UGEN_3] * R_r[BGEN_3];
	Q = -A - G + (BX * BX) * (1. - cmax_roe*cmax_roe);
	X = BX * (A*cmax_roe*BX + C) - (A + G)*(cmax_roe*ptot - R_r[UU]);
	vcon_ar[GEN_1] = (BX * (A*BX + cmax_roe*C) - (A + G)*(ptot + R_r[UGEN_1])) ;
	vcon_ar[GEN_2] = (Q*R_r[UGEN_2] + R_r[BGEN_2] * (C + BX * (cmax_roe*R_r[UGEN_1] + R_r[UU]))) ;
	vcon_ar[GEN_3] = (Q*R_r[UGEN_3] + R_r[BGEN_3] * (C + BX * (cmax_roe*R_r[UGEN_1] + R_r[UU]))) ;
	w_ar[0] = ptot + (-R_r[UU] * X - (vcon_ar[GEN_1] * R_r[UGEN_1] + vcon_ar[GEN_2] * R_r[UGEN_2] + vcon_ar[GEN_3] * R_r[UGEN_3])) / (cmax_roe*X - vcon_ar[GEN_1] + SMALL);
	vcon_ar[GEN_1] = vcon_ar[GEN_1] / X;
	vcon_ar[GEN_2] = vcon_ar[GEN_2] / X;
	vcon_ar[GEN_3] = vcon_ar[GEN_3] / X;

	//Calculate magnetic fields according to eq. 21
	B_ar[GEN_1] = BX;
	B_ar[GEN_2] = -(R_r[BGEN_2] * (cmax_roe*ptot - R_r[UU]) - BX*R_r[UGEN_2]) / A;
	B_ar[GEN_3] = -(R_r[BGEN_3] * (cmax_roe*ptot - R_r[UU]) - BX*R_r[UGEN_3]) / A;

	//B_al[GEN_1] = BX;
	//B_al[GEN_2] = (R_l[BGEN_2] - B_al[GEN_1] * vcon_al[GEN_2]) / (cmin_roe - vcon_al[GEN_1]);
	//B_al[GEN_3] = (R_l[BGEN_3] - B_al[GEN_1] * vcon_al[GEN_3]) / (cmin_roe - vcon_al[GEN_1]);

	//B_ar[GEN_1] = BX;
	//B_ar[GEN_2] = (R_r[BGEN_2] - B_ar[GEN_1] * vcon_ar[GEN_2]) / (cmax_roe - vcon_ar[GEN_1]);
	//B_ar[GEN_3] = (R_r[BGEN_3] - B_ar[GEN_1] * vcon_ar[GEN_3]) / (cmax_roe - vcon_ar[GEN_1]);

	//Calculate K-vector according to eq. 43
	eta_l[0] = -((double)(BX > 0.0) - (double)(BX <= 0.0))*sqrt(w_al[0]);
	K_al[GEN_1] = (R_l[UGEN_1] + ptot + R_l[BGEN_1] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);
	K_al[GEN_2] = (R_l[UGEN_2] + R_l[BGEN_2] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);
	K_al[GEN_3] = (R_l[UGEN_3] + R_l[BGEN_3] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);

	eta_r[0] = ((double)(BX>0.0) - (double)(BX <= 0.0))*sqrt(w_ar[0]);
	K_ar[GEN_1] = (R_r[UGEN_1] + ptot + R_r[BGEN_1] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);
	K_ar[GEN_2] = (R_r[UGEN_2] + R_r[BGEN_2] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);
	K_ar[GEN_3] = (R_r[UGEN_3] + R_r[BGEN_3] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);
	delta_Kx = (K_ar[dir] - K_al[dir]) + pow(10., -12.);

	//Calculate magnetic field between alfven waves and contact discontinuity according to eq. 45
	B_c[GEN_1] = BX*delta_Kx;
	B_c[GEN_2] = ((B_ar[GEN_2] * (K_ar[GEN_1] - vcon_ar[dir]) + B_ar[dir] * vcon_ar[GEN_2]) - (B_al[GEN_2] * (K_al[GEN_1] - vcon_al[dir]) + B_al[dir] * vcon_al[GEN_2]));
	B_c[GEN_3] = ((B_ar[GEN_3] * (K_ar[GEN_1] - vcon_ar[dir]) + B_ar[dir] * vcon_ar[GEN_3]) - (B_al[GEN_3] * (K_al[GEN_1] - vcon_al[dir]) + B_al[dir] * vcon_al[GEN_3]));

	//Calculate error for Newton step
	//B_hat[1] = delta_Kx*B_c[1];
	//B_hat[2] = delta_Kx*B_c[2];
	//B_hat[3] = delta_Kx*B_c[3];
	//Y_l = (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3])) / (eta_l[0] * delta_Kx - (K_al[1] * B_hat[1] + K_al[2] * B_hat[2] + K_al[3] * B_hat[3]));
	//Y_r = (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3])) / (eta_r[0] * delta_Kx - (K_ar[1] * B_hat[1] + K_ar[2] * B_hat[2] + K_ar[3] * B_hat[3]));
	//error = delta_Kx*(1. - B_ar[dir] * (Y_r - Y_l));

	//Calculate state around contact discontiuity according to equations 47, 50-52
	vcon_cl[GEN_1] = (K_al[GEN_1] - (B_c[GEN_1] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] * delta_Kx - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_2] = (K_al[GEN_2] - (B_c[GEN_2] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] * delta_Kx - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_3] = (K_al[GEN_3] - (B_c[GEN_3] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] * delta_Kx - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cr[GEN_1] = (K_ar[GEN_1] - (B_c[GEN_1] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] * delta_Kx - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_2] = (K_ar[GEN_2] - (B_c[GEN_2] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] * delta_Kx - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_3] = (K_ar[GEN_3] - (B_c[GEN_3] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] * delta_Kx - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));

	B_c[GEN_1] = BX;
	B_c[GEN_2] = B_c[GEN_2] / delta_Kx;
	B_c[GEN_3] = B_c[GEN_3] / delta_Kx;

	return (vcon_cr[GEN_1] - vcon_cl[GEN_1]);

	/*if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3; UGEN_2 = U1; UGEN_3 = U2;
		BGEN_1 = B3; BGEN_2 = B1; BGEN_3 = B2;
	}

	//Calculate left wave speed in Riemann fan and w=rho+p+u+b^2
	A = R_l[UGEN_1] + cmin_roe*R_l[UU] + ptot * (1. - cmin_roe*cmin_roe);
	G = R_l[BGEN_2] * R_l[BGEN_2] + R_l[BGEN_3] + R_l[BGEN_3];
	C = R_l[UGEN_2] * R_l[BGEN_2] + R_l[UGEN_3] * R_l[BGEN_3];
	Q = -A - G + (BX * BX) * (1. - cmin_roe*cmin_roe);
	X = BX * (A*cmin_roe*BX + C) - (A + G)*(cmin_roe*ptot - R_l[UU]);
	vcon_al[GEN_1] = (BX * (A*BX + cmin_roe*C) - (A + G)*(ptot + R_l[UGEN_1])) / (X);
	vcon_al[GEN_2] = (Q*R_l[UGEN_2] + R_l[BGEN_2] * (C + BX * (cmin_roe*R_l[UGEN_1] + R_l[UU]))) / (X);
	vcon_al[GEN_3] = (Q*R_l[UGEN_3] + R_l[BGEN_3] * (C + BX * (cmin_roe*R_l[UGEN_1] + R_l[UU]))) / (X);
	w_al[0] = ptot + (-R_l[UU] - (vcon_al[GEN_1] * R_l[UGEN_1] + vcon_al[GEN_2] * R_l[UGEN_2] + vcon_al[GEN_3] * R_l[UGEN_3])) / (cmin_roe - vcon_al[GEN_1] + SMALL);

	//Calculate right wave speed in Riemann fan and w=rho+p+u+b^2
	A = R_r[UGEN_1] + cmax_roe*R_r[UU] + ptot * (1. - cmax_roe*cmax_roe);
	G = R_r[BGEN_2] * R_r[BGEN_2] + R_r[BGEN_3] + R_r[BGEN_3];
	C = R_r[UGEN_2] * R_r[BGEN_2] + R_r[UGEN_3] * R_r[BGEN_3];
	Q = -A - G + (BX * BX) * (1. - cmax_roe*cmax_roe);
	X = BX * (A*cmax_roe*BX + C) - (A + G)*(cmax_roe*ptot - R_r[UU]);
	vcon_ar[GEN_1] = (BX * (A*BX + cmax_roe*C) - (A + G)*(ptot + R_r[UGEN_1])) / (X);
	vcon_ar[GEN_2] = (Q*R_r[UGEN_2] + R_r[BGEN_2] * (C + BX * (cmax_roe*R_r[UGEN_1] + R_r[UU]))) / (X);
	vcon_ar[GEN_3] = (Q*R_r[UGEN_3] + R_r[BGEN_3] * (C + BX * (cmax_roe*R_r[UGEN_1] + R_r[UU]))) / (X);
	w_ar[0] = ptot + (-R_r[UU] - (vcon_ar[GEN_1] * R_r[UGEN_1] + vcon_ar[GEN_2] * R_r[UGEN_2] + vcon_ar[GEN_3] * R_r[UGEN_3])) / (cmax_roe - vcon_ar[GEN_1] + SMALL);

	//Calculate magnetic fields according to eq. 21
	B_al[GEN_1] = BX;
	B_al[GEN_2] = (R_l[BGEN_2] - B_al[GEN_1] * vcon_al[GEN_2]) / (cmin_roe - vcon_al[GEN_1]);
	B_al[GEN_3] = (R_l[BGEN_3] - B_al[GEN_1] * vcon_al[GEN_3]) / (cmin_roe - vcon_al[GEN_1]);

	B_ar[GEN_1] = BX;
	B_ar[GEN_2] = (R_r[BGEN_2] - B_ar[GEN_1] * vcon_ar[GEN_2]) / (cmax_roe - vcon_ar[GEN_1]);
	B_ar[GEN_3] = (R_r[BGEN_3] - B_ar[GEN_1] * vcon_ar[GEN_3]) / (cmax_roe - vcon_ar[GEN_1]);

	//Calculate K-vector according to eq. 43
	eta_l[0] = -((double)(BX > 0.0) - (double)(BX <= 0.0))*sqrt(fabs(w_al[0]));
	K_al[GEN_1] = (R_l[UGEN_1] + ptot + R_l[BGEN_1] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);
	K_al[GEN_2] = (R_l[UGEN_2] + R_l[BGEN_2] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);
	K_al[GEN_3] = (R_l[UGEN_3] + R_l[BGEN_3] * eta_l[0]) / (cmin_roe * ptot - R_l[UU] + BX * eta_l[0]);

	eta_r[0] = ((double)(BX>0.0) - (double)(BX <= 0.0))*sqrt(fabs(w_ar[0]));
	K_ar[GEN_1] = (R_r[UGEN_1] + ptot + R_r[BGEN_1] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);
	K_ar[GEN_2] = (R_r[UGEN_2] + R_r[BGEN_2] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);
	K_ar[GEN_3] = (R_r[UGEN_3] + R_r[BGEN_3] * eta_r[0]) / (cmax_roe * ptot - R_r[UU] + BX * eta_r[0]);

	//Calculate magnetic field between alfven waves and contact discontinuity according to eq. 45
	B_c[GEN_1] = BX;
	B_c[GEN_2] = ((B_ar[GEN_2] * (K_ar[GEN_1] - vcon_ar[dir]) + B_ar[dir] * vcon_ar[GEN_2]) - (B_al[GEN_2] * (K_al[GEN_1] - vcon_al[dir]) + B_al[dir] * vcon_al[GEN_2])) / (K_ar[dir] - K_al[dir] + pow(10., -12.));
	B_c[GEN_3] = ((B_ar[GEN_3] * (K_ar[GEN_1] - vcon_ar[dir]) + B_ar[dir] * vcon_ar[GEN_3]) - (B_al[GEN_3] * (K_al[GEN_1] - vcon_al[dir]) + B_al[dir] * vcon_al[GEN_3])) / (K_ar[dir] - K_al[dir] + pow(10., -12.));

	//Calculate error for Newton step
	delta_Kx = (K_ar[dir] - K_al[dir]);
	B_hat[1] = delta_Kx*B_c[1];
	B_hat[2] = delta_Kx*B_c[2];
	B_hat[3] = delta_Kx*B_c[3];
	Y_l = (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3])) / (eta_l[0] * delta_Kx - (K_al[1] * B_hat[1] + K_al[2] * B_hat[2] + K_al[3] * B_hat[3]));
	Y_r = (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3])) / (eta_r[0] * delta_Kx - (K_ar[1] * B_hat[1] + K_ar[2] * B_hat[2] + K_ar[3] * B_hat[3]));
	error = delta_Kx*(1. - B_ar[dir] * (Y_r - Y_l));

	//Calculate state around contact discontiuity according to equations 47, 50-52
	vcon_cl[GEN_1] = (K_al[GEN_1] - (B_c[GEN_1] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_2] = (K_al[GEN_2] - (B_c[GEN_2] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_3] = (K_al[GEN_3] - (B_c[GEN_3] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l[0] - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cr[GEN_1] = (K_ar[GEN_1] - (B_c[GEN_1] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_2] = (K_ar[GEN_2] - (B_c[GEN_2] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_3] = (K_ar[GEN_3] - (B_c[GEN_3] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r[0] - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));

	vcon_cl[dir] = (vcon_cl[dir] + vcon_cr[dir])*0.5;
	vcon_cr[dir] = vcon_cl[dir];

	return error;*/
}


// EOS function calls
#if (DOHELM)
__device__ void interp_eostable(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele) {
    int iat, jat;
    double fi[36];
    double xt, xd, mxt, mxd;
    double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
    double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;

    //  hash locate this temperature and density
    jat = (int)((log10(btemp) - eos_tlo)*(double)(EOSJMAX - 1) / (11.0 - eos_tlo)) + 1;
    jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
    iat = (int)((log10(din) - eos_dlo)*(double)(EOSIMAX - 1) / (11.0 - eos_dlo)) + 1;
    iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

    //  access the table locations only once
    fi[0] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 0];
    fi[1] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 0];
    fi[2] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 0];
    fi[3] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 0];

    fi[4] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 2];
    fi[5] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 2];
    fi[6] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 2];
    fi[7] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 2];

    fi[8] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 4];
    fi[9] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 4];
    fi[10] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 4];
    fi[11] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 4];

    fi[12] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 1];
    fi[13] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 1];
    fi[14] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 1];
    fi[15] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 1];

    fi[16] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 3];
    fi[17] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 3];
    fi[18] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 3];
    fi[19] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 3];

    fi[20] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 5];
    fi[21] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 5];
    fi[22] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 5];
    fi[23] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 5];

    fi[24] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 6];
    fi[25] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 6];
    fi[26] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 6];
    fi[27] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 6];

    fi[28] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 7];
    fi[29] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 7];
    fi[30] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 7];
    fi[31] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 7];

    fi[32] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 8];
    fi[33] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 8];
    fi[34] = gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 8];
    fi[35] = gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 8];

    double tstp = (11.0 - eos_tlo)/(double)(EOSJMAX-1);
    double dstp  = (11.0 - eos_dlo)/(double)(EOSIMAX-1);
    double eos_t_jat = pow (10.0, (eos_tlo + jat * tstp));
    double eos_d_iat = pow (10.0, (eos_dlo + iat * dstp));
    double eos_dt_jat = pow (10.0, (eos_tlo + (jat + 1) * tstp)) - pow (10.0, (eos_tlo + jat * tstp));
    double eos_dd_iat = pow (10.0, (eos_dlo + (iat + 1) * dstp)) - pow (10.0, (eos_dlo + iat * dstp));

    //  various differences
    xt = MY_MAX((btemp - eos_t_jat) / eos_dt_jat, 0.0); // fix here
    xd = MY_MAX((din - eos_d_iat) / eos_dd_iat, 0.0); // fix here
    mxt = 1.0 - xt;
    mxd = 1.0 - xd;

    //  the density and temperature basis functions
    si0t = psi0(xt);
    si1t = psi1(xt)*eos_dt_jat; // fix here
    si2t = psi2(xt)*eos_dt_jat * eos_dt_jat; // fix here

    si0mt = psi0(mxt);
    si1mt = -psi1(mxt)*eos_dt_jat; // fix here
    si2mt = psi2(mxt)*eos_dt_jat * eos_dt_jat; // fix here

    si0d = psi0(xd);
    si1d = psi1(xd)*eos_dd_iat; // fix here
    si2d = psi2(xd)*eos_dd_iat * eos_dd_iat; // fix here

    si0md = psi0(mxd);
    si1md = -psi1(mxd)*eos_dd_iat; // fix here
    si2md = psi2(mxd)*eos_dd_iat * eos_dd_iat; // fix here

    // the free energy
    *free = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    // the first derivatives of the basis functions
    dsi0d = dpsi0(xd) / eos_dd_iat; // fix here
    dsi1d = dpsi1(xd);
    dsi2d = dpsi2(xd)*eos_dd_iat; // fix here

    dsi0md = -dpsi0(mxd) / eos_dd_iat; // fix here
    dsi1md = dpsi1(mxd);
    dsi2md = -dpsi2(mxd)*eos_dd_iat; // fix here

    // derivative with respect to density
    *df_d = h5(si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

    // the first derivatives of the basis functions
    dsi0t = dpsi0(xt) / eos_dt_jat; // fix here
    dsi1t = dpsi1(xt);
    dsi2t = dpsi2(xt)*eos_dt_jat; // fix here

    dsi0mt = -dpsi0(mxt) / eos_dt_jat; // fix here
    dsi1mt = dpsi1(mxt);
    dsi2mt = -dpsi2(mxt)*eos_dt_jat; // fix here

    // derivative with respect to temperature
    *df_t = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    // the second derivatives of the basis functions
    ddsi0t = ddpsi0(xt) / (eos_dt_jat * eos_dt_jat); // fix here
    ddsi1t = ddpsi1(xt) / eos_dt_jat; // fix here
    ddsi2t = ddpsi2(xt);
    ddsi0mt = ddpsi0(mxt) / (eos_dt_jat * eos_dt_jat); // fix here
    ddsi1mt = -ddpsi1(mxt) / eos_dt_jat; // fix here
    ddsi2mt = ddpsi2(mxt);

    // second derivative with respect to temperature
    *df_tt = h5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md, fi);

    //  second derivative with respect to temperature and density
    *df_dt = h5(dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, fi);

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    si0t = xpsi0(xt);
    si1t = xpsi1(xt)*eos_dt_jat; // fix here

    si0mt = xpsi0(mxt);
    si1mt = -xpsi1(mxt)*eos_dt_jat; // fix here

    si0d = xpsi0(xd);
    si1d = xpsi1(xd)*eos_dd_iat; // fix here

    si0md = xpsi0(mxd);
    si1md = -xpsi1(mxd)*eos_dd_iat; // fix here

    //  pressure derivative with density
    *dpepdd =   gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 9] * si0d * si0t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 9] * si0md * si0t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 9] * si0d * si0mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 9] * si0md * si0mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 11] * si0d * si1t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 11] * si0md * si1t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 11] * si0d * si1mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 11] * si0md * si1mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 10] * si1d * si0t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 10] * si1md * si0t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 10] * si1d * si0mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 10] * si1md * si0mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 12] * si1d * si1t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 12] * si1md * si1t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 12] * si1d * si1mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 12] * si1md * si1mt;

    // h3dpd(iat,jat,
    //      si0t,   si1t,   si0mt,   si1mt,
    //      si0d,   si1d,   si0md,   si1md,
    //      eos_dpdf, eos_dpdft, eos_dpdfd, eos_dpdfdt);

    *dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

    //  electron chemical potential etaele
    *etaele =   gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 13] * si0d * si0t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 13] * si0md * si0t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 13] * si0d * si0mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 13] * si0md * si0mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 15] * si0d * si1t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 15] * si0md * si1t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 15] * si0d * si1mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 15] * si0md * si1mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 14] * si1d * si0t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 14] * si1md * si0t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 14] * si1d * si0mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 14] * si1md * si0mt +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat) * 21 + 16] * si1d * si1t +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat) * 21 + 16] * si1md * si1t +
                gpu_eos_table[(iat) * EOSJMAX * 21 + (jat + 1) * 21 + 16] * si1d * si1mt +
                gpu_eos_table[(iat + 1) * EOSJMAX * 21 + (jat + 1) * 21 + 16] * si1md * si1mt;

    // h3e(iat,jat,
    //                si0t,   si1t,   si0mt,   si1mt,
    //                si0d,   si1d,   si0md,   si1md);

    //  electron + positron number densities
    /*xnefer = eos_xf[(iat)*EOSJMAX + jat] * si0d*si0t + eos_xf[(iat + 1)*EOSJMAX + jat] * si0md*si0t
     + eos_xf[(iat)*EOSJMAX + jat + 1] * si0d*si0mt + eos_xf[(iat + 1)*EOSJMAX + jat + 1] * si0md*si0mt
     + eos_xft[(iat)*EOSJMAX + jat] * si0d*si1t + eos_xft[(iat + 1)*EOSJMAX + jat] * si0md*si1t
     + eos_xft[(iat)*EOSJMAX + jat + 1] * si0d*si1mt + eos_xft[(iat + 1)*EOSJMAX + jat + 1] * si0md*si1mt
     + eos_xfd[(iat)*EOSJMAX + jat] * si1d*si0t + eos_xfd[(iat + 1)*EOSJMAX + jat] * si1md*si0t
     + eos_xfd[(iat)*EOSJMAX + jat + 1] * si1d*si0mt + eos_xfd[(iat + 1)*EOSJMAX + jat + 1] * si1md*si0mt
     + eos_xfdt[(iat)*EOSJMAX + jat] * si1d*si1t + eos_xfdt[(iat + 1)*EOSJMAX + jat] * si1md*si1t
     + eos_xfdt[(iat)*EOSJMAX + jat + 1] * si1d*si1mt + eos_xfdt[(iat + 1)*EOSJMAX + jat + 1] * si1md*si1mt;*/

    // h3x(iat,jat,
    //              si0t,   si1t,   si0mt,   si1mt,
    //              si0d,   si1d,   si0md,   si1md);
}


__device__ void eos_helm (const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double abar, double zbar, double *pres, double *ener, double *dpresdt, double *denerdt, double *dpresdd, double *cs2)
{
    // Local variables
    double ytot1, ye, local_coulombMult;
    double x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, deni, tempi, kt, prad, dpraddd, dpraddt, erad, deraddd, deraddt, srad, dsraddd, dsraddt;
    double xni, pion, dpiondd, dpiondt, eion, deiondd, deiondt;
    double sion, dsiondd, dsiondt;
    double pele, dpepdd, dpepdt, eele, deepdd, deepdt;
    double entr, dentrdt, dentrdd;
    double sele, dsepdd, dsepdt;
    double presi, gamc, kavoy;
    double etaele, xnefer, denerdd;

    // For the interpolations
    double free,df_d,df_t,df_dd,df_tt,df_dt;
    double zFunc, z0, z1, z2, z3, z4, z5, z6, din;

    // For the coulomb corrections
    double ktinv,dxnidd,dsdd,lami,inv_lami,lamidd,s0,s1,s2,s3,s4,plasg,plasg_inv,plasgdd,plasgdt,ecoul,decouldd,decouldt,pcoul,dpcouldd,dpcouldt,scoul,dscouldd,dscouldt;

    // Added by Calhoun for calculations for the Aprox13t network
    double deradda,dxnida,dpionda,deionda,dsepda,deepda,decoulda,dsda,dsdda,lamida,plasgda,denerda,deraddz,deiondz,deepdz,decouldz,dsepdz,plasgdz,denerdz;

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;

	if (den > 1.0e11 || den < 1.0e-11) 
		return;

	if (btemp > 1.0e11 || btemp < 1.0e4)
		return;

    // Useful relations
    ytot1 = 1.0 / abar;
    ye = ytot1 * zbar;
    kt = kerg * btemp;
    ktinv = 1.0 / kt;
    din = zbar * ytot1 * den;

#if (bAprox13t)
    dsda = 4.0 / 3.0*M_PI * dxnida; //Calhoun
    lamida   = z2 * dsda / s1;      //Calhoun
    plasgda  = z3 * lamida;         //Calhoun
    plasgdz  = 2.0 * plasg/zbar;    //Calhoun
#endif

    //Look up the desired quantities in the eos table
    interp_eostable(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, &etaele);

    // the desired electron-positron thermodynamic quantities
    x3 = din * din;
    pele = x3 * df_d;
    sele = -df_t * ye;
    eele = ye * free + btemp * sele;

    dxnidd = avo * ytot1;
    xni = dxnidd * den;
    pion = xni * kt;

    // uniform background corrections & only the needed parts for speed
    // plasg is the plasma coupling parameter
    // split up calculations below -- they all used to depend upon a redefined z
    s1 = 4.0 / 3.0 * M_PI * xni;
    lami = 1.0 / pow(s1, third);
    inv_lami = 1.0 / lami;
    plasg = zbar*zbar*esqu*ktinv*inv_lami;

    //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    deni = 1.0 / den;
    tempi = 1.0 / btemp;
    if (plasg >= 1.0) {
        x4 = pow(plasg, 0.25);
        z4 = eos_c1 / x4;
        ecoul = dxnidd * kt * (eos_a1*plasg + eos_b1*x4 + z4 + d1cc);
        pcoul = third * den * ecoul;
        kavoy = kergavo * ytot1;
        scoul = -kavoy*(3.0*eos_b1*x4 - 5.0*z4 + d1cc*(log(plasg) - 1.0) - e1cc);
    }
    //yakovlev & shalybkov 1989 equations 102, 103, 104
    else if (plasg < 1.0) {
        x5 = plasg * sqrt(plasg);
        y3 = pow(plasg, eos_b2);
        z5 = eos_c2 * x5 - third * eos_a2 * y3;
        pcoul = -pion * z5;
        ecoul = 3.0 * pcoul * deni;
        kavoy = kergavo * ytot1;
        scoul = -kavoy*(eos_c2*x5 - eos_a2*(eos_b2 - 1.0) / eos_b2*y3);
    }

    //  radiation section:
    prad = asoli3 * btemp * btemp * btemp * btemp;
    x1 = prad * deni;
    erad = 3.0 * x1;
    srad = (x1 + erad)*tempi;

    // Set the coulomb multiplier to a local value -- we might change it only within this call
    local_coulombMult = eos_coulombMult;

    // assume that NaN always compares as false in an inequality
    if (!(prad + pion + pele + pcoul*eos_coulombMult > 0.0)) {
        printf("[eos_helm] Negative total pressure. %e %e %e %e\n", prad + pion + pele, pcoul);
        printf("%s %e %e\n", " values: dens,temp: ",den,btemp);
        printf("%s %e %e\n", " values: abar,zbar: ",abar,zbar);
        printf("%s %e\n", " coulomb coupling parameter Gamma: ",plasg);

        if ( !(abar > 0.0) ) {
            printf("%s %e\n", "  However, abar is negative, abar=",abar);
            printf("%s\n", "      It is possible that the mesh is of low quality.");
            printf("%s\n", "[eos_helm] ERROR: abar is negative.");
        }

        if (prad + pion + pele > 0.0) {
            printf("%s %e %e\n", " nonpositive P caused by coulomb correction: Pnocoul,Pwithcoul: ", prad + pion + pele, prad + pion + pele + pcoul*eos_coulombMult);

            if (eos_coulombMult > 0.0) {
                printf("  set runtime parameter eos_coulombMult to zero if plasma Coulomb corrections not important\n");
            }
            if (eos_coulombAbort) {
                printf("[eos_helm] ERROR: coulomb correction causing negative total pressure.\n");
            }
            else {
                printf("Setting coulombMult to zero for this call, eos_coulombAbort=false\n");
                local_coulombMult = 0.0;
            }
        }
        else {
            printf("Prad %e\nPion %e\nPele %e\nPcoul %e\nPtot %e\ndf_d %e\n", prad, pion, pele, pcoul*eos_coulombMult, prad + pion + pele + pcoul*eos_coulombMult, df_d);
            printf("[eos_helm] ERROR: negative total pressure.\n");
        }
    }

#if (bAprox13t)
    dxnida = -xni*ytot1; //Calhoun
    dpionda = dxnida * kt; //Calhoun
    deionda = 1.5 * dpionda*deni;  //Calhoun
    deiondz = 0.0; //Calhoun
#endif

    //  sackur-tetrode equation for the ion entropy of
    //  a single ideal gas characterized by abar
    *pres = prad + pion + pele + pcoul * local_coulombMult;

    eion = 1.5 * pion * deni;

    *ener = erad + eion + eele + ecoul * local_coulombMult;

    sion = (pion*deni + eion)*tempi + kavoy*log(pow(abar, 2.5) * deni*avoinv *pow(sioncon * btemp, 1.5));

    entr = srad + sion + sele + scoul * local_coulombMult;

    if (calc_derivatives){
        //Calculate pressure derivatives
        dpraddt = 4.0 * prad * tempi;
        dpraddd = 0.0;

        dpiondd = avo * ytot1 * kt;
        dpiondt = xni * kerg;

        dpepdt = x3 * df_dt;

        //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        if (plasg >= 1.0) {
            plasg_inv = 1.0 / plasg;
            y1 = dxnidd * kt * (eos_a1 + 0.25*plasg_inv*(eos_b1*x4 - z4));
            dsdd = 4.0 / 3.0*M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            decouldd = y1 * plasgdd;
            decouldt = y1 * plasgdt + ecoul * tempi;
            dpcouldd = third * (ecoul + den * decouldd);
            dpcouldt = third * den  * decouldt;
        }
        //yakovlev & shalybkov 1989 equations 102, 103, 104
        else if (plasg < 1.0) {
            plasg_inv = 1.0 / plasg;
            s2 = (1.5*eos_c2*x5 - third*eos_a2*eos_b2*y3)*plasg_inv;
            dxnidd = avo * ytot1;
            dsdd = 4.0 / 3.0*M_PI * dxnidd;
            lamidd = -third * lami / s1 * dsdd;
            plasgdd = -plasg * inv_lami * lamidd;
            plasgdt = -plasg * ktinv * kerg;
            dpcouldd = -dpiondd*z5 - pion*s2*plasgdd;
            dpcouldt = -dpiondt*z5 - pion*s2*plasgdt;
            decouldd = 3.0*dpcouldd*deni - ecoul*deni;
            decouldt = 3.0*dpcouldt*deni;
        }
        *dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * local_coulombMult; //pressure derivative vs density
        *dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * local_coulombMult; //pressure derivative vs temperature

        //Calculate energy derivatives
        deiondd = (1.5 * dpiondd - eion)*deni;
        deiondt = 1.5 * xni * kerg *deni;

        deraddd = -erad*deni;
        deraddt = 4.0 * erad * tempi;
        dsepdt = -df_tt * ye;
        dsepdd = -df_dt * ye * ye;
        deepdt = btemp * dsepdt;
        deepdd = ye*ye*df_d + btemp*dsepdd;
#if (bAprox13t)
        //  Calhoun next two lines
        deradda = 0.0;
        deraddz = 0.0;
#endif
        denerdd = deraddd + deiondd + deepdd + decouldd * local_coulombMult; //energy derivative vs density
        *denerdt = deraddt + deiondt + deepdt + decouldt * local_coulombMult; //energy derivative vs temperature

        //Calculate entropy derivatives
        dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt*deni + deraddt - srad)*tempi;

        dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
        dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion) * tempi*tempi + 1.5 * kavoy * tempi;

        //yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        if (plasg >= 1.0) {
            y2 = -kavoy*plasg_inv*(0.75*eos_b1*x4 + 1.25*z4 + d1cc);
            dscouldd = y2 * plasgdd;
            dscouldt = y2 * plasgdt;
        }
        //yakovlev & shalybkov 1989 equations 102, 103, 104
        else if (plasg < 1.0) {
            s3 = -kavoy*plasg_inv*(1.5*eos_c2*x5 - eos_a2*(eos_b2 - 1.0)*y3);
            dscouldd = s3 * plasgdd;
            dscouldt = s3 * plasgdt;
        }
        dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * local_coulombMult;//entropy derivative vs density and density
        dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * local_coulombMult;//entropy derivative vs density and time

        // calculate relativistic soundspeeds
        double zz, zzi, chit, chid, cv, cp, x, gam1, gam2, gam3, nabad, z;
        zz = (*pres) * deni;
        zzi = den / (*pres);
        chit = btemp / (*pres) * (*dpresdt);
        chid = (*dpresdd) * zzi;
        cv = (*denerdt);
        x = zz * chit / (btemp * cv);
        gam3 = x + 1.0;
        gam1 = chit * x + chid;
        nabad = x / gam1;
        gam2 = 1.0 / (1.0 - nabad);
        cp = cv * gam1 / chid;
        z = 1.0 + ((*ener) + (c_light * c_light)) * zzi;
        *cs2 = gam1 / z; // already in the units of the code (c = 1)
    }

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;

    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;

    return;
}


__device__ void eos_mode_rhou_pres (const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    double ener_goal = u_goal / den;

    if (ener_goal <= 0.0)
        temp_ini_guess = 1.0e4;
    else
        temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
	if (temp_ini_guess > 1e11) temp_ini_guess = 1e11;

    double temp_new, temp_old;
    double ener_tmp, ener_old;
    double dpdt, dedt, dpdrho;
    double cs2;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &ener_tmp, &dpdt, &dedt, &dpdrho, &cs2);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
}

__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres, double *cs2) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_e = 1.0e-5;

    // initial guess : temperature
    double temp_ini_guess;
    double ener_goal = u_goal / den;

    if (ener_goal <= 0.0)
        temp_ini_guess = 1.0e4;
    else
        temp_ini_guess = pow(den * ener_goal * conv_pres_CODE2CGS / asol, 0.25);
	if (temp_ini_guess > 1e11) temp_ini_guess = 1e11;

    double temp_new, temp_old;
    double ener_tmp, ener_old;
    double dpdt, dedt, dpdrho;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < max_iterations; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &ener_tmp, &dpdt, &dedt, &dpdrho, cs2);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_e < tolerance_e) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

	return;
}

__device__ void eos_mode_rhow_pres_dpdrho_dpde_d (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *dpdrho, double *dpde_d) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    double h_goal = w_goal * deni;

    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
	if (temp_ini_guess > 1e11) temp_ini_guess = 1e11;

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpdt, dedt, dhdt;
    double h_tmp;
    double cs2;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xpres = *pres;
    double xener = 0.0;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &xener, &dpdt, &dedt, dpdrho, &cs2);

        h_tmp = xener + (*pres) * deni;
        dhdt = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdt * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, xener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *dpde_d = dpdt / dedt;
}

__device__ void eos_mode_rhow_pres_u (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *u) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_h = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    double h_goal = w_goal * deni;

    if (h_goal < 1.0) temp_ini_guess = 1.0e4;
    else temp_ini_guess = pow(den * (h_goal - 1.0) * conv_ener_CODE2CGS * conv_dens_CODE2CGS / asol, 0.25);
	if (temp_ini_guess > 1e11) temp_ini_guess = 1e11;

    double temp_new, temp_old;
    double ener_old, pres_old;
    double dpresdener_d, dhdtemp;
    double h_tmp;
    double cs2;
    double dpdrho, dpdt, dedt, dpde_d;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xener;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;

    for(i = 0; i < max_iterations; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &xener, &dpdt, &dedt, &dpdrho, &cs2);

        h_tmp = xener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

        // do not allow temp to change more than 10 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);

        //printf("num = %d, err in T = %e, err in h = %e, T = %e, h = %e, p = %e, e = %e\n", i, error, error_h, temp_new, h_tmp, *pres, *ener);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;
        if(error < tolerance && error_h < tolerance_h) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *u = xener * den;
}

__device__ void eos_mode_rhotemp_pres_min (const  double* __restrict__ gpu_eos_table, double den, double *pres) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = 1.0e4;
    double ener, dpdt, dedt, dpdrho;
    double cs2;
    eos_helm(gpu_eos_table, 1, temp, den, 1.0, 1.0, pres, &ener, &dpdt, &dedt, &dpdrho, &cs2);
}

__device__ void eos_mode_rhopres_u (const  double* __restrict__ gpu_eos_table, double den, double p_goal, double *u) {

    // Parameters of Newton-Raphson iterations
    int max_iterations = 50;
    double tolerance = 1.0e-5;
    double tolerance_p = 1.0e-5;

    double deni = 1.0 / den;

    // initial guess : temperature
    double temp_ini_guess;
    if (p_goal <= 0.0) temp_ini_guess = 1.0e4;
    temp_ini_guess = pow(p_goal * conv_pres_CODE2CGS * asoli3_inv, 0.25);
	if (temp_ini_guess > 1e11) temp_ini_guess = 1e11;

    double temp_new, temp_old;
    double p_tmp;
    double cs2;
    double xener;

    double error, error_p;
    int i;

    double dpdt, dedt, dpdrho;

    temp_old = temp_ini_guess;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance
    for(i = 0; i < max_iterations; i++){

        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, &p_tmp, &xener, &dpdt, &dedt, &dpdrho, &cs2);

        temp_new = temp_old - (p_tmp - p_goal) / dpdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 2.0) temp_new = 2.0 * temp_old;
        if (temp_old / temp_new > 2.0) temp_new = 0.5 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((p_tmp - p_goal) / p_goal);

        //printf("num = %d, T = %e, err in T = %e, err in p = %e, p = %e, dpdt = %e\n", i, temp_new, error, error_p, p_tmp, dpdt);
        if (temp_new < 1.0e4) temp_new = 1.0e4;
        if (temp_new > 1.0e11) temp_new = 1.0e11;

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < tolerance && error_p < tolerance_p) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

    *u = xener * den;
}
#endif
