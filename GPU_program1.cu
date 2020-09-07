#include "config.h"

/*Declerations of functions for Utoprim2D*/
__device__ int Utoprim_2d(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim);
__device__ int Utoprim_new_body(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double tolerance, int lim);
__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance);
__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq);
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D);
__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq);

/*Declerations of functions for Utoprim_1dvsq2fix1*/
__device__ int Utoprim_1dvsq2fix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim);
__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm);
__device__ double dWdvsq_calc2(double vsq, double rho, double p);
__device__ int Utoprim_new_body2(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double K_atm, double tolerance, int lim);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm);
__device__ void validate_x2(double x[1], double x0[1]);
__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance);
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double D, double K_atm, double W_for_gnr2);

/*Declerations of functions for Utoprim_1dfix1*/
__device__ int Utoprim_1dfix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim);
__device__ int Utoprim_new_body3(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim);
__device__ int general_newton_raphson3(double x[],  double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old, double tolerance);
__device__ void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old);
__device__ int gnr2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2);

/*Declerations of functions for Utoprim_NM*/
__device__ int Utoprim_NM_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim);
__device__ int Utoprim_NM(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim);

/*Declerations of functions related to (M1) radiation scheme*/
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim);
__device__ int Rtoprim_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim);
__device__ int implicit_rad_solve_PMHD(double pb[NPR], double U[NPR], struct of_geom geom, double dU[NPR], double Dt);
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int js, double* vmax, double* vmin, double dx);
__device__ double calc_kappa_abs(double* ph);
__device__ double calc_kappa_emmit(double* ph);
__device__ double calc_kappa_es(double* ph);
__device__ void mhd_calc_rad(double* pr, int dir, struct of_state_rad* q_rad, double* mhd_rad);
__device__ void ucon_calc_rad(double* pr, struct of_geom* geom, double* ucon_rad);
__device__ void get_state_rad(double* pr, struct of_geom* geom, struct of_state_rad* q_rad);

/*Declerations of functions related to (M1) radiation inversion scheme*/
__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max);
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max);
__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged);
__device__ int implicit_rad_solve_UMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged);
__device__ int implicit_rad_solve_PRAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged);
__device__ int implicit_rad_solve_URAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged);
__device__ int implicit_rad_solve_EMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged);

/*Matrix Inversion*/
__device__ int invert_matrix(double Am[][NDIM], double Aminv[][NDIM]);
__device__ int LU_decompose(double A[][NDIM], int permute[]);
__device__ void LU_substitution(double A[][NDIM], double B[], int permute[]);
__device__ void source_rad(double* ph, struct of_geom* geom, double* dU);
__device__ void calc_Gcon(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM]);
__device__ int gamma_calc_rad(double* pr, struct of_geom* geom, double* gamma_rad);

/*Declare other functions*/
__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q);
__device__ void ucon_calc(double *  pr, struct of_geom *  geom, double *  ucon);
__device__ void bcon_calc(double *  pr, double *  ucon, double *  ucov, double *  bcon);
__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma);
__device__ void vchar(double* pr, struct of_state* q, struct of_geom* geom, int dir, double* vmax, double* vmin);
__device__ void get_geometry(int ii, int jj, int zz, int kk, struct of_geom *  geom, const  double* __restrict__ gcov_GPU, const  double* __restrict__ gcon_GPU, const  double* __restrict__ gdet_GPU);
__device__ void get_trans(int ii, int jj, int zz, int kk, struct of_trans * trans, const  double* __restrict__ Mud_GPU, const  double* __restrict__ Mud_inv_GPU);
__device__ double slope_lim(double y1, double y2, double y3, int lim);
__device__ void raise(double ucov[NDIM], double gcon[10], double ucon[NDIM]);
__device__ void lower(double ucon[NDIM], double gcov[10], double ucov[NDIM]);
__device__ void primtoflux(double *  pr, struct of_state *  q,  int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin);
__device__ void primtoflux_rad(double* pr, struct of_state_rad* q_rad, int dir, struct of_geom* geom, double* flux);
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *dU, double Dt, const  double* __restrict__ conn,struct of_state *  q, double r);
__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,	 double r, double Dt);
__device__ void inflow_check(double *  prim, int ii, int jj, int zz, int type, const  double* __restrict__ gcov1, const  double* __restrict__ gcoBS_2, const  double* __restrict__ gdet3);
__device__ double bsq_calc(double *  pr, struct of_geom *  geom);
__device__ double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double readImageDouble(int4 a);
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd);
__device__ int fixup_cell(double pf[NDIM], double r, struct of_geom* geom, struct of_state* q);

/*Advanced Riemann solver related functions*/
__device__ void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double* F);
__device__ void vchar_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double* vmax, double* vmin);
__device__ void primtoflux_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR]);
__device__ void calc_HLLC(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ void calc_HLLD(int dir, double cmin_roe, double cmax_roe, double int_velocity, double l_ucon[NDIM], double r_ucon[NDIM], double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
__device__ double calc_HLLD_pres(int dir, int* fail_HLLC, int* fail_HLLD, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double* eta_l, double* eta_r, double* w_al, double* w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
__device__ void calc_HLLD_state(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double ptot, double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double eta_l, double eta_r, double w_al, double w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
__device__ void check_HLLD_par(int dir, int* fail_HLLD, double cmin_roe, double cmax_roe, double ptot, double w_al, double w_ar, double eta_l, double eta_r, double vcon_cl[NDIM], double vcon_cr[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double B_c[NDIM]);
__device__ double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double vcon_cl[NDIM], double vcon_cr[NDIM], double* eta_l, double* eta_r, double* w_al, double* w_ar);

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

struct of_state_rad {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
};

__device__ void implicit_rad_solve(double *pb, double *U_n, double *U_i, double *U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double *dU, double Dt, double cell_size, double y_max) {
	double error_t = 1.e-3;
	int k;
	double delta_Ur, U_ft[NPR], pb_i[NPR];

	//Initialize temporary variables
	PLOOP{
		dU[k] = 0.;
		U_ft[k] = U_i[k];
		pb_i[k] = pb[k];
	}

	//Set initial values and error before attempting implicit solver
	implicit_rad_solve_init(pb_i, U_n, U_i, U_ft, geom, dU, Dt, &error_t, cell_size, y_max);

	//If we've reached the tolerance level, exit immediately and update variables
	if (error_t < 1.e-12) {
		for (k = 0; k < NPR; k++) {
			pb[k] = pb_i[k];
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i[k]) / Dt;
		}
	}
	else {
		//Determine relative updae to radiation variables
		delta_Ur = (U_n[UU_RAD] - U_i[UU_RAD] + 1.e-150) / (fabs(U_i[UU_RAD]) + fabs(U_n[UU_RAD]));

		//Check if fluid is in extreme radiation subdominant regime
		if (((U_n[UU_RAD] / U_n[UU]) < 1.e-5) || ((pb[UU_RAD] / pb[UU]) < 1.e-5) || (fabs(delta_Ur) < 1.e-5)) {
			//If error is below set margin, accept solution, otherwise try PMHD
			//if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, y_max, 0, 0);

			//If error is below set margin, accept solution, otherwise try PRAD
			//if (error_t > 1.e-9)implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try UMHD
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is below set margin, accept solution, otherwise try PMHD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			/*
			//If error is below set margin, accept solution, otherwise try PRAD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try URAD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);
			*/

			//If error is still below set margin, accept solution, otherwise try UMHD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try URAD staged
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 1);

			//If error is still below set margin, accept solution, otherwise try URAD staged
			//if (error_t > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 1);

			//If error is below set margin, accept solution, otherwise try PRAD staged
			//if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 1);

			//If error is still below set margin, accept solution, otherwise try UMHD staged
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 1);

			//If error is still below set margin, accept solution, otherwise try EMHD staged
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 1);

			/*/
			//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 1);

			//If error is still below set margin, accept solution, otherwise try URAD with entropy staged
			//if (error_t > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 1);
			*/

			//If error is below set margin, accept solution, otherwise try PRAD with entropy staged
			//if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, y_max,1, 1);

			//If error is still below set margin, accept solution, otherwise try UMHD with entropy staged
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 1);

			//If error is still below set margin, accept solution, otherwise try EMHD with entropy staged
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 1);
		}
		else {
			//If error is below set margin, accept solution, otherwise try PMHD
			//if (error_t > 1.e-9)implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size, y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try UMHD
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 0, 0);

			//If error is below set margin, accept solution, otherwise try PRAD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			/*/
			//If error is still below set margin, accept solution, otherwise try URAD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try URAD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n, U_i, U_ft,pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);
			*/

			//If error is still below set margin, accept solution, otherwise try UMHD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD with entropy
			//if (error_t > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, &error_t, cell_size,y_max, 1, 0);
		}

		//As final resort attempt subcycling
		if (error_t > 1.e-7) {
			//subcycle_rad_solve(pb_i, U_n, U_i, U_ft, pflag, pflag_rad, geom, dU, Dt, cell_size);
		}
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
}

//Calculate initial error for source term and set initial guess values
__device__ void implicit_rad_solve_init(double *pb, double *U_n, double *U_i, double *U_f, struct of_geom* geom, double *dU, double Dt, double* error_t, double cell_size, double y_max) {
	double kappa_abs, kappa_es, tau = 0., norm;
	int k, pflag, pflag_rad;
	struct of_state q;
	struct of_state_rad q_rad;

	//Calculate optical depth
	kappa_abs = calc_kappa_abs(pb);
	kappa_es = calc_kappa_es(pb);
	tau = (kappa_abs + kappa_es) * cell_size;

	if (tau < 0.66) {
		//Set guess values for primitives after implicit step based on optical depth
		pflag = Utoprim_2d(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC);
		#if(DO_FONT_FIX)
		if (pflag) {
			pflag = Utoprim_1dvsq2fix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC);
			if (pflag) {
				pflag = Utoprim_1dfix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC);
			}
		}
		#endif	 

		//Even if MHD inversion fails, use updated value of radiation variable as gues
		pflag_rad = Rtoprim(U_f, geom->gcov, geom->gcon, geom->g, pb, y_max, BASIC);
	}

	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q);
	mhd_calc(pb, 0, &q, &U_f[UU]);
	for (k = UU; k <= U3; k++)U_f[k] *= geom->g;
	U_f[UU] += U_f[RHO];

	//Recompute entropy for consistency
	#if(FULL_ENTROPY)
	U_f[KTOT] = geom->g * pb[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb[UU] * pow(pb[RHO], -GAMMA));
	#else
	U_f[KTOT] = geom->g * pb[RHO] * q.ucon[0] * (GAMMA - 1.) * pb[UU] * pow(pb[RHO], -GAMMA);
	#endif

	//Recompute R_t^mu for consistency
	get_state_rad(pb, geom, &q_rad);
	mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
	for (k = UU_RAD; k <= U3_RAD; k++)U_f[k] *= geom->g;

	//Calculate source term for U_i
	source_rad(pb, geom, dU);

	//Calculate iterated error at start of iteration
	norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	error_t[0] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1] - U_i[U1] - Dt * dU[U1]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2] - U_i[U2] - Dt * dU[U2]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3] - U_i[U3] - Dt * dU[U3]) / norm);

	//Calculate total error at start of iteration
	norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
	error_t[0] += 0.25 * (fabs(U_f[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1_RAD] - U_i[U1_RAD] - Dt * dU[U1_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2_RAD] - U_i[U2_RAD] - Dt * dU[U2_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3_RAD] - U_i[U3_RAD] - Dt * dU[U3_RAD]) / norm);
}

__device__ int implicit_rad_solve_PMHD(double *pb, double *U_n, double *U_i, double *U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double *dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4][4], dEdpb_inv[4][4],  error_new[5], offset = 1.e-8;
	double T_GAS, norm, norm_S, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1,  n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;

	//Set error to previous value
	for (k = 0; k < 5; k++) error_new[k] = error_t[0];

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		U_old[k] = U_f[k];
		dU_old[k] = dU[k];
		U_new[k] = U_old[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		for (i = UU; i <= U3; i++) {
			n_iter_jacob = 0;

			do {
				PLOOP pb_new[k] = pb_old[k];
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb_old[i] + dpb;
				}
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					pb_new[i] = pb_old[i] + dpb;
				}

				get_state(pb_new, geom, &q);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				mhd_calc(pb_new, 0, &q, &U_new[UU]); // Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] = U_new[UU] + U_new[RHO];
				if (do_entropy == 1) {
					#if(FULL_ENTROPY)
					U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA)));
					#else
					U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
					#endif
				}

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Calculate source function and jacobian
				source_rad(pb_new, geom, dU_new);
				for (k = U1; k <= U3; k++) {
					E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdpb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb;
				}
				if (do_entropy == 1) {
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb[0][i - UU] = (E_new[0] - E_old[0]) / dpb;

				//Invert Jacobian
				flag = invert_matrix(dEdpb, dEdpb_inv);

				n_iter_jacob++;
			} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

			if (flag) {
				return 1;
			}
		}

		//Set primitive variables before Newton step
		PLOOP pb_new[k] = pb_old[k];

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.0;
			for (k = 0; k < 4; k++) {
				dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
				pb_new[k + UU] = pb_old[k + UU] + dpb;
			}
		}
		else {
			//Set damping factor for Newton Raphson method
			if (n_iter == 0 || n_iter == 4) D = 0.5;
			else if (n_iter == 8) D = 0.25;
			else D = 1.;

			if (n_iter / 4 == 0) { //momentum only step
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			if (n_iter / 4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
		}

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		mhd_calc(pb_new, 0, &q, &U_new[UU]);
		for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
		U_new[UU] = U_new[UU] + U_new[RHO];

		#if(FULL_ENTROPY)
		U_new[KTOT] = geom->g * pb_new[RHO] * q.ucon[0] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
		#else
		U_new[KTOT] = geom->g * (pb_new[RHO] * q.ucon[0] * (GAMMA - 1.) * pb_new[UU] * pow(pb_new[RHO], -GAMMA));
		#endif

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives using TYPE2 limiter
		flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

		//Get radiative source term
		source_rad(pb_new, geom, dU_new);

		//Calculate iterated error
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10;
		//else offset = 1.e-8;

		//Calculate total error
		if (do_entropy == 1) {
			T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
			#if(FULL_ENTROPY)
			norm_S = T_GAS * (fabs(U_i[KTOT]) + fabs(U_new[KTOT]) + fabs(Dt * dU_new[KTOT]));
			error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
			#else
			double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (pb_new[UU]) / pow(pb_new[RHO], GAMMA);
			norm_S = T_GAS * (fabs(U_i[RHO] / (GAMMA - 1.) * log(U_i[KTOT] / U_i[RHO])) + fabs(U_new[RHO] / (GAMMA - 1.) * log(U_new[KTOT] / U_new[RHO])) + fabs(Dt * dU_new[KTOT] / dK_dS));
			error_new[n_iter % 5] += 0.25 * T_GAS * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / dK_dS) / (norm);
			#endif
		}
		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5]) <= 1.e-12) || (n_iter >= 20)) {
			keep_iterating = 0;
		}

		//If error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < 1.0 * (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			keep_iterating = 0;
		}

		//If error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		if (keep_iterating) {
			PLOOP{
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}
		}

		//If error decreased compared to start value, update variables
		if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5]) < 1.e-4) {
			error_t[0] = error_new[n_iter % 5];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
		}

		n_iter++;
	}
	return(0);
}

__device__ int invert_matrix(double Am[][NDIM], double Aminv[][NDIM]){
	int i, j;
	int permute[NDIM];
	double dxm[NDIM], Amtmp[NDIM][NDIM];

	for (i = 0; i < NDIM*NDIM; i++) Amtmp[0][i] = Am[0][i];

	//Get the LU matrix:
	if (LU_decompose(Amtmp, permute) != 0) return(1);

	for (i = 0; i < NDIM; i++) {
		for (j = 0; j < NDIM; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		//Solve the linear system for the i^th column of the inverse matrix
		LU_substitution(Amtmp, dxm, permute);

		for (j = 0; j < NDIM; j++) Aminv[j][i] = dxm[j];
	}

	return(0);
}


__device__ int LU_decompose(double A[][NDIM], int permute[]){
	double row_norm[NDIM];
	double  absmax, maxtemp;
	int i, j, k, max_row;

	max_row = 0;

	//Find the maximum elements per row so that we can pretend late we have unit-normalized each equation

	for (i = 0; i < NDIM; i++) {
		absmax = 0.;

		for (j = 0; j < NDIM; j++) {
			maxtemp = fabs(A[i][j]);

			if (maxtemp > absmax) absmax = maxtemp;
		}

		//Make sure that there is at least one non-zero element in this row:
		if (absmax == 0.) return(1);

		row_norm[i] = 1. / absmax; //Set the row's normalization factor.
	}

	/* For each of the columns, starting from the left ... */
	for (j = 0; j < NDIM; j++) {

		/* For each of the rows starting from the top.... */

		/* Calculate the Upper part of the matrix:  i < j :   */
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) A[i][j] -= A[i][k] * A[k][j];
		}

		absmax = 0.0;

		/* Calculate the Lower part of the matrix:  i <= j :   */
		for (i = j; i < NDIM; i++) {
			for (k = 0; k < j; k++) A[i][j] -= A[i][k] * A[k][j];

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}

		}

		if (max_row != j) {
		
			if ((j == (NDIM - 2)) && (A[j][j + 1] == 0.)) max_row = j;
			else {
				for (k = 0; k < NDIM; k++) {
					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;
				}

				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) A[j][j] = 1.e-30;

		if (j != (NDIM - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < NDIM; i++) A[i][j] *= maxtemp;
		}
	}

	return(0);
	/* End of LU_decompose() */
}


__device__ void LU_substitution(double A[][NDIM], double B[], int permute[])
{
	int i, j;
	double tmpvar;

	/* Perform the forward substitution using the LU matrix.
	*/
	for (i = 0; i < NDIM; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	/* Perform the backward substitution using the LU matrix.
	*/
	for (i = (NDIM - 1); i >= 0; i--) {
		for (j = (i + 1); j < NDIM; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}

	/* End of LU_substitution() */
}

//Inversion from radiation conserved to primitive quantities
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim){
	double U_tmp[NPR_R], prim_tmp[NPR_R];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
	for (i = 0; i <= U3_RAD - UU_RAD; i++) U_tmp[i] = alpha * U[i + NPR_U] / gdet;

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i <= U3_RAD - UU_RAD; i++) prim_tmp[i] = prim[i + NPR_U]; //radiation prims

	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, y_max, lim);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i <= U3_RAD - UU_RAD; i++) {
		prim[i + NPR_U] = prim_tmp[i];
	}

	return(ret);
}

__device__ int Rtoprim_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim){
	double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f;
	int i, returnval = 0;

	for (i = 0; i < 4; i++) Qcov[i] = U[i];
	raise(Qcov, gcon, Qcon);

	ncov = -sqrt(-1. / gcon[0]);
	ncon[0] = gcon[0] * ncov;
	ncon[1] = gcon[1] * ncov;
	ncon[2] = gcon[2] * ncov;
	ncon[3] = gcon[3] * ncov;

	Qdotn = Qcon[0] * ncov; //-Erad in McKinney2013
	for (i = 1; i < 4; i++)  Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;  //Utilde in McKinney2013

	for (i = 0; i < 4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn * Qdotn; //Utilde^2 in McKinney2013

	y = Qtsq / (Qdotn * Qdotn + 1.e-150); //Definition from McKinney2013. Should only range [0,1].
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = pressure * 3.; // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	if (isnan(-Qdotn)) {
		prim[0] = 1.e-150;
		//if (y > 1. || isnan(y)) returnval = 1;
		y = 0.;
	}
	if (prim[0] < 0) {
		prim[0] = 0.5 * fabs(prim[0]);
		Qdotn *= -1.0;
	}
	if (y <= 0.) {
		for (i = 1; i < 4; i++) prim[i] = 0.0;
		y = 0.;
	}
	if (y >= 1.0 || isnan(y)) {
		prim[0] = 1.e-150;
		prim[1] = 0.;
		prim[2] = 0.;
		prim[3] = 0.;
		gammasq = 1.0;

		// Get Ebar and p_rad as usual
		pressure = -Qdotn / (4. * gammasq - 1.);
		prim[0] = pressure * 3.; // Erad = 3*p_rad

		// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
		for (i = 1; i < 4; i++) {
			if (!isnan(Qtcon[i])) {
				prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);
			}
		}
		y = 0.;
		returnval = 0;
	}
	if (y > y_max) {
		Uabs = 0.5 * (sqrt(Qtsq) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++)prim[i] = Qtcon[i] / Uabs;

		qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
			+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
		if (qsq < 0. && fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;

		if (lim == TYPE2) {
			Qdotn = -(1.e-150 + sqrt(Qtsq / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad		
			returnval = 0;
		}
	}
	return(returnval);
}

__device__ int Utoprim_NM(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha;

	//If mass flux negative, return immediately
	if (U[0] <= 0.) return(-100);

	//First update the primitive B-fields
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	U_tmp[RHO] = alpha * U[RHO] / gdet; //W=ucon[0]*alpha
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	ret = Utoprim_NM_calc(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim);

	//Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}

__device__ int Utoprim_NM_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u,  w,  gamma,   vsq, errx;
	double Bsq, QdotBsq, Qtsq, Qdotn;
	int i;

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
	
	Qsq = 0.;

	for (i = 0; i<4; i++) Qsq += Qcov[i] * Qcon[i];
	Qtsq = Qsq + Qdotn*Qdotn;
	
	//Start inversion scheme AKA Newman et al
	double a, d, z, phi, R, Wsq, p_array[3], epsilon, p_old, p_new;
	int iter = 0;
	int iter_tot = 0;
	int set_variables = 0;
	p_array[0] = (GAMMA - 1.)*prim[UU];
	p_new = p_array[0];
	d = 0.5*(Qtsq*Bsq - QdotBsq);
	if (d < 0.0) return(1);
	do {
		set_variables = 0;
		a = -Qdotn + p_new + 0.5 * Bsq;
		phi = acos(1. / a * sqrt((27. * d) / (4. * a)));
		epsilon = a / 3. - 2. / 3. * a * cos(2. / 3. * phi + 2. / 3. * M_PI);
		z = epsilon - Bsq;

		vsq = (Qtsq * z * z + QdotBsq * (Bsq + 2. * z)) / (z * z * pow(Bsq + z, 2.));
		Wsq = 1. / (1. - vsq);
		w = z * (1. - vsq);
		gamma = 1. / sqrt(1. - vsq);
		rho0 = U[RHO] / gamma;
		u = (w - rho0) / GAMMA; //Watch out you may need this for a more complicated EOS

		iter++;
		iter_tot++;
		p_array[iter] = (GAMMA - 1.) * u;
		p_old = p_array[iter - 1];
		p_new = p_array[iter];
		if (iter >= 2) {
			R = (p_array[iter] - p_array[iter - 1]) / (p_array[iter - 1] - p_array[iter - 2]);

			if (R < 1. && R>0.) {
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
	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);

	if (set_variables == 1) {
		a = -Qdotn + p_new + 0.5 * Bsq;
		phi = acos(1. / a * sqrt((27. * d) / (4. * a)));
		epsilon = a / 3. - 2. / 3. * a * cos(2. / 3. * phi + 2. / 3. * M_PI);
		z = epsilon - Bsq;

		vsq = (Qtsq * z * z + QdotBsq * (Bsq + 2. * z)) / (z * z * pow(Bsq + z, 2.));
		Wsq = 1. / (1. - vsq);
		w = z / Wsq;
		if (vsq >= 1.0 || vsq<0. || z <= 0. || z > W_TOO_BIG || !isfinite(vsq) || !isfinite(z)) {
			return(4);
		}
		gamma = sqrt(Wsq);

		rho0 = U[RHO] / gamma; //Watch out you may need this for a more complicated EOS
		u = (w - rho0) / GAMMA;
		p_new = (GAMMA - 1.) * u;
	}

	//If density or internal energy is negative return error code
	if ((rho0 < 0.0)) return(5);
	if ((p_new < 0.0) && (lim == BASIC)) return(6);

	prim[RHO] = rho0;
	prim[UU] = u;

	//Set 4-velocities
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (z + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / z);
	}

	/* done! */
	return(0);
}

__device__ int Utoprim_1dfix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha, K_atm;

	if (U[0] <= 0.) return(-100);

	//First update the primitive B-fields
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha */
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	#if(DOKTOT)
	#if(FULL_ENTROPY)
	K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	#else
	K_atm = U[KTOT] / U[RHO];
	#endif
	#endif

	ret = Utoprim_new_body3(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim);
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}

__device__ int Utoprim_new_body3(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim){
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, retval=0, i_increase;
	double W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

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
	p = K_atm * pow(rho0, GAMMA);
	u = p / (GAMMA - 1.);
	w = rho0 + u + p;

	W_last = w*gammasq;

	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq) - QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq)) && (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	W_for_gnr2 = W_for_gnr2_old = W_last;
	rho_for_gnr2 = rho_for_gnr2_old = rho0;

	x_1d[0] = W_last;
	retval = general_newton_raphson3(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old, tolerance);

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

	vsq = vsq_calc(W, Bsq, Qtsq, QdotBsq);
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

	if ((rho0 <= 0.)) {
		retval = 5;
		return(retval);
	}

	if ((u <= 0.) && (lim==BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}

	return(retval);
}

__device__ int general_newton_raphson3(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old, double tolerance){
	double f, df, x_old[NEWT_DIM_1], dx[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int n_iter=0,  i_extra=0, doing_extra=0;
	int keep_iterating, i_increase;

	errx = 1.;
	df = f = 1.;
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		#if(USE_ISENTROPIC)   
		func_1d_orig1(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old);  /* returns with new dx, f, df */
		#endif

		//Save old values before calculating the new
		x_old[0] = x[0];

		//Make Newton step
		x[0] += dx[0];

		i_increase = 0;
		while (((x[0] * x[0] * x[0] * (x[0] + 2.*Bsq) - QdotBsq*(2.*x[0] + Bsq)) <= x[0] * x[0] * (Qtsq - Bsq*Bsq)) && (i_increase < 10)) {
			x[0] -= (1.*i_increase) * dx[0] / 10.;
			i_increase++;
		}

		//Make sure value remains physical
		x[0] = fabs(x[0]);

		//Calculate the convergence criterion
		errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 1;
		if (doing_extra == 1) i_extra++;
		if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}
		n_iter++;
	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
		return(2);
	}

	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance) return(0);

	return(0);
}

__device__ int gnr2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2){
	double f, df, x_old[NEWT_DIM_1], dx[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int n_iter, i_extra, doing_extra;
	int keep_iterating;

	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	n_iter = 0;

	keep_iterating = 1;
	while (keep_iterating) {
		func_gnr2_rho(x, dx, resid, jac, &f, &df, D, K_atm, W_for_gnr2);  /* returns with new dx, f, df */
		
		//Save old values before calculating the new
		x_old[0] = x[0];

		//Make the newton step
		x[0] += dx[0];

		//Make sure x[0] is physical
		x[0] = fabs(x[0]);

		//Calculate the convergence criterion
		errx = (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 0;
		if (doing_extra == 1) i_extra++;
		if (((fabs(errx) <= NEWT_TOL2) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	if ((isfinite(f) == 0) || (isfinite(df) == 0) || (isfinite(x[0]) == 0)) {
		return(2);
	}

	if (fabs(errx) > MIN_NEWT_TOL2)return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL2) && (fabs(errx) > NEWT_TOL2))return(0);
	if (fabs(errx) <= NEWT_TOL2)return(0);

	return(0);
}

//isentropic version:   eq.  (27)
__device__ void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old){
	int ntries;
	double  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	double  t32, t33, t34, t38, t5, t51, t67, t8,  x_rho[1], rho, rho_g;

	//W = x[0];
	W_for_gnr2 = x[0];

	// get rho from NR:
	rho_g = x_rho[0] = rho_for_gnr2;

	ntries = 0;
	while ((gnr2(x_rho, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2)) && (ntries++ < 10)) {
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
	t10 = t1* x[0];
	t21 = x[0] * x[0];
	t23 = rho*rho;
	t26 = 1. / t1;
	resid[0] = (t3 + (2.0*t2 + ((Qtsq - t5)*t1 + (-2.0*t8 - t10)* x[0])* x[0])* x[0] + (t5 + (2.0*Bsq + x[0])* x[0])*t21*t23)*t26 / t21;
	t29 = t1*t1;
	t30 = QdotBsq*t29;
	t32 = GAMMA*K_atm;
	t33 = pow(rho, 1.0*GAMMA);
	t34 = t32*t33;
	t38 = t23 * t33;
	t51 = GAMMA*t1*K_atm*t33;
	t67 = t21* x[0];

	jac[0][0] = -2.0*(t30*Bsq*t34 + (t30*t34+ ((-t38*Bsq*t32 + Bsq*GAMMA*t1*K_atm*t33)*t1+ (-t38*GAMMA*K_atm + t51)*t1* x[0])*t21)* x[0] + ((-t3 + (-t2 + (-t8 - t10)*t21)* x[0])* x[0] + (-t5 - Bsq* x[0])*t67*t23)*t23)*t26 / (t51 - x[0] *t23) / t67;
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);

	return;
}

// for the isentropic version:   eq.  (27)
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double D, double K_atm, double W_for_gnr2){
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

__device__ int Utoprim_1dvsq2fix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha, K_atm;

	if (U[0] <= 0.) {
		return(-100);
	}

	//First update the primitive B-fields
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into the new system
	U_tmp[RHO] = alpha * U[RHO] / gdet;
	U_tmp[UU] = alpha * (U[UU] - U[RHO]) / gdet;
	#pragma unroll 3
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	//Transform the PRIMITIVE variables into the new system
	#pragma unroll 5
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	#if(DOKTOT)
	#if(FULL_ENTROPY)
	K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	#else
	K_atm = U[KTOT] / U[RHO];
	#endif
	#endif

	ret = Utoprim_new_body2(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim);

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}

__device__ int Utoprim_new_body2(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim){
	double x_1d[1];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, gammasq, gamma, gtmp, W, utsq, vsq;
	int    i, retval=0;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

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

	//Calculate W from last timestep and use  for guess */
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

	//Initialize independent variables for Newton-Raphson:
	x_1d[0] = 1. - 1. / gammasq;

	//Find vsq via Newton-Raphson:
	retval = general_newton_raphson2(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, tolerance);

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

	//Find W from this vsq:
	W = W_of_vsq2(vsq, &p, &rho0, &u, D, K_atm);

	//Recover the primitive variables from the scalars and conserved variables:
	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;

	// User may want to handle this case differently, e.g. do NOT return upon 
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.)) {
		retval = 5;
		return(retval);
	}

	if ((u <= 0.) && (lim==BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#pragma unroll 3
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}

	/* done! */
	return(retval);
}

__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance){
	double f, df, dx[NEWT_DIM_1], x_old[NEWT_DIM_1], resid[NEWT_DIM_1], jac[NEWT_DIM_1][NEWT_DIM_1];
	double errx;
	int    n_iter,  i_extra, doing_extra;
	double W, W_old, rho, p, u;
	int   keep_iterating;

	//Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;

	x_old[0] = x[0];

	W = W_old = 0.;

	n_iter = 0;

	//Start the Newton-Raphson iterations
	keep_iterating = 1;
	while (keep_iterating) {
		func_1d_gnr2(x, dx, resid, jac, &f, &df,Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm);/* returns with new dx, f, df */

		//Set old values
		errx = 0.;
		x_old[0] = x[0];

		//Make Newton step
		x[0] += dx[0];

		validate_x2(x, x_old);

		//Calculate W=w*gamma
		W_old = W;
		W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm);
		errx = (W == 0.) ? fabs(W - W_old) : fabs((W - W_old) / W);
		errx += (x[0] == 0.) ? fabs(x[0] - x_old[0]) : fabs((x[0] - x_old[0]) / x[0]);

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0)) doing_extra = 1;

		if (doing_extra == 1) i_extra++;

		//See if we've done the extra iterations, or have done too many iterations:
		if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	} 

	//Check for bad untrapped divergences
	if ((isfinite(f) == 0) || (isfinite(df) == 0))return(2);

	// Return in different ways depending on whether a solution was found:
	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance) return(0);

	return(0);
}

__device__ void validate_x2(double x[1], double x0[1]){
	double small = 1.e-10;
	x[0] = (x[0] >= 1.0) ? (0.5*(x0[0] + 1.)) : x[0];
	x[0] = (x[0] <  -small) ? (0.5*x0[0]) : x[0];
	x[0] = fabs(x[0]);
	return;
}

__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm){
	double W, Wsq, W3, dWdvsq, fact_tmp, rho, p, u;
	//vsq = x[0];

	// Calculate best value for W given current guess for vsq: 
	W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm);
	Wsq = W*W;
	W3 = W*Wsq;

	// Doing this assuming  P = (G-1) u :
	dWdvsq = dWdvsq_calc2(x[0], rho, p);
	fact_tmp = (Bsq + W);
	resid[0] = Qtsq - x[0] * fact_tmp * fact_tmp + QdotBsq * (Bsq + 2.*W) / Wsq;
	jac[0][0] = -fact_tmp * (fact_tmp + 2. * dWdvsq * (x[0] + QdotBsq / W3));
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
}

__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm){
	double gtmp;
	gtmp = (1. - vsq);
	*rho = D * sqrt(gtmp);
	*p = K_atm * pow(*rho, GAMMA);
	*u = *p / (GAMMA - 1.);
	return((*rho + *u + *p) / gtmp);
}

__device__ double dWdvsq_calc2(double vsq, double rho, double p){
	return((GAMMA*(2. - G_ATM)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
}

__device__ int Utoprim_2d(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
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
	for (i = UTCON1; i <= UTCON3; i++) U_tmp[i] = alpha * U[i] / gdet;
	#pragma unroll 3
	for (i = BCON1; i <= BCON3; i++) U_tmp[i] = alpha * U[i] / gdet;

	/* Transform the PRIMITIVE variables into the new system */
	#pragma unroll 5
	for (i = 0; i < BCON1; i++) prim_tmp[i] = prim[i];

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}
#include <stdio.h>

__device__ int Utoprim_new_body(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim){
	double x_2d[NEWT_DIM_2];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, retval=0, i_increase;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;

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
	p = (GAMMA - 1.)*u;
	w = rho0 + u + p;

	W_last = w*gammasq;

	// Make sure that W is large enough so that v^2 < 1 : 
	i_increase = 0;
	while (((W_last*W_last*W_last * (W_last + 2.*Bsq) - QdotBsq*(2.*W_last + Bsq)) <= W_last*W_last*(Qtsq - Bsq*Bsq))&& (i_increase < 10)) {
		W_last *= 10.;
		i_increase++;
	}

	// Calculate W and vsq: 
	x_2d[0] = fabs(W_last);
	x_2d[1] = x1_of_x0(W_last, Bsq, Qtsq, QdotBsq);
	retval = general_newton_raphson(x_2d, Bsq, Qtsq, QdotBsq, Qdotn, D, tolerance);

	W = x_2d[0];
	vsq = x_2d[1];

	//Problem with solver, so return denoting error before doing anything further */
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

	//Calculate v^2:
	if (vsq >= 1.) {
		retval = 4;
		return(retval);
	}

	//Recover the primitive variables from the scalars and conserved variables:
	gtmp = sqrt(1. - vsq);
	gamma = 1. / gtmp;
	rho0 = D * gtmp;

	w = W * (1. - vsq);
	p = (GAMMA - 1.)*(w - rho0) / GAMMA;
	u = w - (rho0 + p);

	// User may want to handle this case differently, e.g. do NOT return upon 
	// a negative rho/u, calculate v^i so that rho/u can be floored by other routine:
	if ((rho0 <= 0.)) {
		retval = 5;
		return(retval);
	}
	if ((u <= 0.) && (lim==BASIC)) {
		retval = 6;
		return(retval);
	}

	prim[RHO] = rho0;
	prim[UU] = u;

	#if AMD
	#pragma unroll 3
	for (i = 1; i < 4; i++) {
		Qtcon[i] = fma(ncon[i], Qdotn, Qcon[i]);
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (fma(QdotB, Bcon[i] / W, Qtcon[i]));
	}
	#else
	#pragma unroll 3
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}
	#endif

	/* done! */
	return(retval);
}

__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq){
	double Wsq, Xsq;
	Wsq = W*W;
	Xsq = (Bsq + W) * (Bsq + W);
	#if AMD
	return((fma(Wsq, Qtsq, QdotBsq * (Bsq + 2.*W))) / (Wsq*Xsq));
	#else
	return((Wsq * Qtsq + QdotBsq * (Bsq + 2.*W)) / (Wsq*Xsq));
	#endif
}

__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq){
	double vsq;

	vsq = fabs(vsq_calc(x0, Bsq, Qtsq, QdotBsq)); // guaranteed to be positive 
	return((vsq > 1.) ? (1.0 - 1.e-15) : vsq);
}

__device__ void validate_x(double x[2], double x0[2]){

	/* Always take the absolute value of x[0] and check to see if it's too big:  */
	x[0] = fabs(x[0]);
	x[0] = (x[0] > W_TOO_BIG) ? x0[0] : x[0];

	x[1] = (x[1] < 0.) ? 0. : x[1];  /* if it's too small */
	x[1] = (x[1] > 1.) ? (1. - 1.e-15) : x[1];  /* if it's too big   */
	return;
}

__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance){
	double f, df, dx[NEWT_DIM_2], x_old[NEWT_DIM_2];
	double resid[NEWT_DIM_2], jac[NEWT_DIM_2][NEWT_DIM_2];
	double errx;
	int n_iter, id, i_extra, doing_extra, keep_iterating;

	// Initialize various parameters and variables:
	errx = 1.;
	df = f = 1.;
	i_extra = doing_extra = 0;
	for (id = 0; id < NEWT_DIM_2; id++)  x_old[id] = x[id];

	n_iter = 0;

	//Start the Newton-Raphson iterations
	keep_iterating = 1;
	while (keep_iterating) {
		func_vsq(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D);  /* returns with new dx, f, df */

		//Save old values before calculating the new
		errx = 0.;
		for (id = 0; id < NEWT_DIM_2; id++) x_old[id] = x[id];

		//Make the newton step
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

		if ((fabs(errx) <= tolerance) && (doing_extra == 0) && (EXTRA_NEWT_ITER > 0))doing_extra = 1;
		if (doing_extra == 1) i_extra++;
		if (((fabs(errx) <= tolerance) && (doing_extra == 0)) || (i_extra > EXTRA_NEWT_ITER) || (n_iter >= (MAX_NEWT_ITER - 1))) {
			keep_iterating = 0;
		}

		n_iter++;
	} 

	//Check for bad untrapped divergences
	if ((isfinite(f) == 0) || (isfinite(df) == 0)) return(2);

	//Depending on error return OK or error
	if (fabs(errx) > MY_MIN(tolerance, MIN_NEWT_TOL)) return(1);
	if ((fabs(errx) <= MIN_NEWT_TOL) && (fabs(errx) > tolerance)) return(0);
	if (fabs(errx) <= tolerance)return(0);
	
	return(0);
}

__device__ void func_vsq(double x[], double dx[], double resid[], double jac[][NEWT_DIM_2], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D){
	double Wsq, p_tmp, dPdvsq, dPdW, gtmp;
	double t11, t16, t18, t2, t21, t23,t24, t25, t3, t35, t36, t4, t40, t9;

	//W = x[0];
	//vsq = x[1];

	Wsq = x[0] * x[0];
	gtmp = 1. - x[1];
	#if(AMD)
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#else
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#endif

	// These expressions were calculated using Mathematica, but fmae into efficient  code using Maple.  Since we know the analytic form of the equations, we can explicitly calculate the Newton-Raphson step: 
	#if(AMD)
	t2 = fma(-0.5, Bsq, dPdvsq);
	t3 = Bsq + x[0];
	t4 = t3*t3;
	t9 = 1. / Wsq;
	t11 = fma(QdotBsq, (Bsq + 2.0* x[0])*t9, fma(-x[1], t4, Qtsq));
	t16 = QdotBsq*t9;
	t18 = -fma(0.5, Bsq*(1.0 + x[1]), Qdotn) + fma(0.5, t16, -x[0] + p_tmp);
	t21 = 1. / t3;
	t23 = 1. / x[0];
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = fma(t25, t3, (fma(-2.0, dPdvsq, Bsq))*(fma(x[1], Wsq* x[0], QdotBsq))*t9*t23);
	t36 = 1. / t35;
	dx[0] = -(fma(t2, t11, t4*t18))*t21*t36;
	t40 = (x[1] + t24)*t3;
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
	t3 = Bsq + x[0];
	t4 = t3*t3;
	t9 = 1. / Wsq;
	t11 = Qtsq - x[1] *t4 + QdotBsq*(Bsq + 2.0* x[0])*t9;
	t16 = QdotBsq*t9;
	t18 = -Qdotn - 0.5*Bsq*(1.0 + x[1]) + 0.5*t16 - x[0] + p_tmp;
	t21 = 1. / t3;
	t23 = 1. / x[0];
	t24 = t16*t23;
	t25 = -1.0 + dPdW - t24;
	t35 = t25*t3 + (Bsq - 2.0*dPdvsq)*(QdotBsq + x[1] *Wsq* x[0])*t9*t23;
	t36 = 1. / t35;
	dx[0] = -(t2*t11 + t4*t18)*t21*t36;
	t40 = (x[1] + t24)*t3;
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

//Apply floors to a cell
__device__ int fixup_cell(double *pf, double r, struct of_geom* geom, struct of_state* q) {
	double rhoscal, uuscal, rhoflr, uuflr, bsq, wold, wnew, QdotB, trans, vpar, one_over_ucondr_t, x, f;
	double pf_prefloor[NPR_U], betapar, betasq, betasqmax, gamma, ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], vcon[NDIM], ucon[NDIM], utcon[NDIM], B, Bsq, udotB, ut;
	int dofloor=0, flag = 0, m, k;

	rhoscal = pow(r, -POWRHO);
	uuscal = pow(rhoscal, GAMMA);

	rhoflr = RHOMIN * rhoscal;
	uuflr = UUMIN * uuscal;

	get_state(pf, geom, q);
	bsq = dot(q->bcon, q->bcov);

	//tie floors to the local values of magnetic field and internal energy density
	if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	if (rhoflr < pf[UU] / UORHOMAX) rhoflr = pf[UU] / (UORHOMAX);

	if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
	if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;

	//floor on density and internal energy density (momentum *not* conserved) 
	#pragma unroll 9
	for (k = 0; k < NPR_U; k++)pf_prefloor[k] = pf[k];
	if (pf[RHO] < rhoflr) {
		pf[RHO] = rhoflr;
		dofloor = 1;
	}
	if (pf[UU] < uuflr) {
		pf[UU] = uuflr;
		dofloor = 1;
	}

	#if(DRIFT_FLOOR)
	trans = 10. * bsq / MY_MIN(pf[RHO], pf[UU]) - 1.;
	if (dofloor && (trans) > 0.) {
		if (trans > 1.) trans = 1.;
		betapar = -q->bcon[0] / ((bsq + SMALL) * q->ucon[0]);
		betasq = betapar * betapar * bsq;
		betasqmax = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		if (betasq > betasqmax) betasq = betasqmax;

		gamma = 1. / sqrt(1 - betasq);
		#pragma unroll 4
		for (m = 0; m < NDIM; m++) ucondr[m] = gamma * (q->ucon[m] + betapar * q->bcon[m]);

		Bcon[0] = 0.;
		#pragma unroll 3
		for (m = 1; m < NDIM; m++) Bcon[m] = pf[B1 - 1 + m];

		lower(Bcon, geom->gcov, Bcov);
		udotB = dot(q->ucon, Bcov);
		Bsq = dot(Bcon, Bcov);
		B = sqrt(Bsq);

		//enthalpy before the floors
		wold = pf_prefloor[RHO] + pf_prefloor[UU] * GAMMA;

		//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
		QdotB = udotB * wold * q->ucon[0];

		//enthalpy after the floors
		wnew = pf[RHO] + pf[UU] * GAMMA;

		x = 2. * QdotB / (B * wnew * ucondr[0] + SMALL);

		//new parallel velocity
		vpar = x / (ucondr[0] * (1. + sqrt(1. + x * x)));

		one_over_ucondr_t = 1. / ucondr[0];

		//new contravariant 3-velocity, v^i
		vcon[0] = 1.;

		#pragma unroll 3
		for (m = 1; m < NDIM; m++) {
			//parallel (to B) plus perpendicular (to B) velocities
			vcon[m] = vpar * Bcon[m] / (B + SMALL) + ucondr[m] * one_over_ucondr_t;
		}

		//compute u^t corresponding to the new v^i
		ut_calc_3vel(vcon, geom, &ut);

		#pragma unroll 4
		for (m = 0; m < NDIM; m++) ucon[m] = ut * vcon[m];
		ucon_to_utcon(ucon,geom, utcon);

		//now convert 3-vel to relative 4-velocity and put it into pv[U1..U3]
		//\tilde u^i = u^t(v^i-g^{ti}/g^{tt})
		#pragma unroll 3
		for (m = 1; m < NDIM; m++) {
			pf[m + UU] = utcon[m] * trans + pf_prefloor[m + UU] * (1. - trans);
		}
	}
	#elif(ZAMO_FLOOR)
	if (dofloor == 1) {
		double dpf[NPR_U], U_prefloor[NPR_U], Xtransone_over_ucondr;
		struct of_state_rad q_rad;
		#pragma unroll 9
		for (k = 0; k < NPR_U; k++) dpf[k] = pf[k] - pf_prefloor[k];

		//compute the conserved quantity associated with floor addition
		get_state(dpf, geom, q);
		primtoflux(dpf, q, 0, geom, dU, NULL, NULL);

		//compute the prefloor conserved quantity
		get_state(pf_prefloor, geom, q);
		primtoflux(pf_prefloor, q, 0, geom, U_prefloor, NULL, NULL);

		//add U_added to the current conserved quantity
		#pragma unroll 9
		for (k = 0; k < NPR_U; k++) U[k] = U_prefloor[k] + dU[k];

		#if(NEWMAN)
		flag = Utoprim_NM(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC);
		#else
		flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC);
		#endif
		if (pflag[global_id]) {
			#if( DO_FONT_FIX ) 
			flag = Utoprim_1dvsq2fix1(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC);
			if (pflag[global_id]) {
				flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC);
			}
			#endif	
		}
	}
	#endif

	// limit gamma wrt normal observer 
	if (gamma_calc(pf, geom, &gamma)) {
		flag = 4;
	}
	else {
		if (gamma > GAMMAMAX) {
			f = sqrt((GAMMAMAX * GAMMAMAX - 1.) / (gamma * gamma - 1.));
			pf[U1] *= f;
			pf[U2] *= f;
			pf[U3] *= f;
		}
	}

	return flag;
}


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
	BB = 2.*(geom->gcov[1] * vcon[1] +geom->gcov[2] * vcon[2] +geom->gcov[3] * vcon[3]);
	CC = geom->gcov[4] * vcon[1] * vcon[1] +geom->gcov[7] * vcon[2] * vcon[2] +geom->gcov[9] * vcon[3] * vcon[3] +2.*(geom->gcov[5] * vcon[1] * vcon[2] +geom->gcov[6] * vcon[1] * vcon[3] + geom->gcov[8] * vcon[2] * vcon[3]);
	DD = -1. / (AA + BB + CC);
	one_over_alpha_sq = -geom->gcon[0];
	if (DD<one_over_alpha_sq) DD = one_over_alpha_sq;

	*ut = sqrt(DD);
}

/* add in geometrical and cooling source terms to equations of motion */
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *  dU, double Dt, const  double* __restrict__ conn_GPU, struct of_state *  q, double r)
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

	P = (GAMMA - 1.)*ph[UU];
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
	misc_source(ph, icurr, jcurr, geom, q, dU, r, Dt);
	#endif

	dU[UU] *= geom->g;
	dU[U1] *= geom->g;
	dU[U2] *= geom->g;
	dU[U3] *= geom->g;

	//Add M1 radiation terms
	#if(RAD_M1)
	double mhd_rad[NDIM][NDIM];
	struct of_state_rad q_rad;
	get_state_rad(ph, geom, &q_rad);
	mhd_calc_rad(ph, 0, &q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, &q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, &q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, &q_rad, mhd_rad[3]);

	//contract radiation stress tensor with connection
	#pragma unroll 4	
	for (k = 0; k<NDIM; k++) {
		#if(NSY)
		dU[UU_RAD] += mhd_rad[0][k] * conn_GPU[0 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[1][k] * conn_GPU[4 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2_RAD] += mhd_rad[2][k] * conn_GPU[7 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3_RAD] += mhd_rad[3][k] * conn_GPU[9 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[1][k] * conn;
		dU[U1_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[2 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[2][k] * conn;
		dU[U2_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[3 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[5 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[2][k] * conn;
		dU[U2_RAD] += mhd_rad[1][k] * conn;
		conn = conn_GPU[6 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[1][k] * conn;
		conn = conn_GPU[8 * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[2][k] * conn;
		#else
		dU[UU_RAD] += mhd_rad[0][k] * conn_GPU[0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[1][k] * conn_GPU[4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2_RAD] += mhd_rad[2][k] * conn_GPU[7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3_RAD] += mhd_rad[3][k] * conn_GPU[9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[1][k] * conn;
		dU[U1_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[2][k] * conn;
		dU[U2_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[0][k] * conn;
		conn = conn_GPU[5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[2][k] * conn;
		dU[U2_RAD] += mhd_rad[1][k] * conn;
		conn = conn_GPU[6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[1][k] * conn;
		conn = conn_GPU[8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + k*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2_RAD] += mhd_rad[3][k] * conn;
		dU[U3_RAD] += mhd_rad[2][k] * conn;
		#endif
	}
	dU[UU_RAD] *= geom->g;
	dU[U1_RAD] *= geom->g;
	dU[U2_RAD] *= geom->g;
	dU[U3_RAD] *= geom->g;
	#endif

	/* done! */
}

__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,  double r, double Dt){
	double epsilon = ph[UU] / ph[RHO];
	double om_kepler = 1. / (pow(r, 3. / 2.) + BH_SPIN);
	double T_target = M_PI / 2.*pow(H_OVER_R*r*om_kepler, 2.);
	double Y = (GAMMA - 1.)*epsilon / T_target;
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
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd)
{
	int j;
	double r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];
	P = (GAMMA - 1.)*u;
	w = P + r + u;
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	ptot = P + 0.5*bsq;

	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd[j] = eta*q->ucon[dir] * q->ucov[j] + ptot*delta(dir, j) - q->bcon[dir] * q->bcov[j];
}

__device__ void mhd_calc_rad(double * pr, int dir, struct of_state_rad * q_rad, double * mhd_rad){
	int j;
	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd_rad[j] = 4. / 3. * pr[UU_RAD] * q_rad->ucon[dir] * q_rad->ucov[j] + 1. / 3. * pr[UU_RAD] * delta(dir, j);
}

__device__ void source_rad(double *  ph, struct of_geom *  geom, double * dU)
{
	#if(RAD_M1)
	double mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM], ucon[NDIM], ucov[NDIM], Tg;
	int k;
	struct of_state_rad q_rad;

	PLOOP dU[k] = 0.;

	//Add M1 radiation terms
	get_state_rad(ph, geom, &q_rad);
	mhd_calc_rad(ph, 0, &q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, &q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, &q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, &q_rad, mhd_rad[3]);

	//Add radiation 4-force
	ucon_calc(ph, geom, ucon);
	lower(ucon, geom->gcov, ucov);
	calc_Gcon(ph, Gcon, ucon, ucov, mhd_rad);
	lower(Gcon, geom->gcov, Gcov);

	dU[UU] = Gcov[0];
	dU[U1] = Gcov[1];
	dU[U2] = Gcov[2];
	dU[U3] = Gcov[3];

	dU[UU_RAD] = -Gcov[0];
	dU[U1_RAD] = -Gcov[1];
	dU[U2_RAD] = -Gcov[2];
	dU[U3_RAD] = -Gcov[3];

	#if(DOKTOT)
	Tg = (GAMMA - 1.) * (ph[UU]) / (ph[RHO]);
	#if(FULL_ENTROPY)
	dU[KTOT] = -1. / Tg * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
	#else
	double dK_dS = (GAMMA - 1.) * (GAMMA - 1.) * (ph[UU]) / pow(ph[RHO], GAMMA); //Multiply the next line with this to get evolution for K=P/rho^gamma instead of S=1/(gamma-1)*log(P/rho^gamma)
	dU[KTOT] = -dK_dS / Tg * (Gcov[0] * ucon[0] + Gcov[1] * ucon[1] + Gcov[2] * ucon[2] + Gcov[3] * ucon[3]);
	#endif
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}

//Calculate radiation 4-force
__device__ void calc_Gcon(double * ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM]) {
	#if(RAD_M1)
	int i;
	double lambda, Tg, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], arad;
	kappa_abs = calc_kappa_abs(ph);
	kappa_emmit = calc_kappa_emmit(ph);
	kappa_es = calc_kappa_es(ph);
	arad = ARAD / (ENERGY_DENSITY_SCALE / pow(MMW * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS, 4.));

	Tg = (GAMMA - 1.) * ph[UU] / ph[RHO];
	lambda = kappa_emmit * arad * pow(Tg, 4.); //in units of erg/s/cm^3
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs * R_dot_ucon[i] + lambda * ucon[i]) - kappa_es * (R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]) * ucon[i]);
	}
	#endif
}

__device__ void primtoflux_rad(double* pr, struct of_state_rad* q_rad, int dir, struct of_geom* geom, double* flux){
	int k;

	//Radiation energy tensor
	#if(RAD_M1)
	mhd_calc_rad(pr, dir, q_rad, &flux[UU_RAD]);
	#endif
	for (k = UU_RAD; k <= U3_RAD; k++) flux[k] *= geom->g;

	return;
}

__device__ void primtoflux(double *  pr, struct of_state *  q,  int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin)
{
	int j, k;
	double mhd[NDIM];
	double P, w, bsq, eta, ptot;

	/*Calculate misc quantities*/
	P = (GAMMA - 1.)*pr[UU];
	#if(AMD)
	w = fma(gam, pr[UU], pr[RHO]);
	#else
	w = pr[RHO] + GAMMA*pr[UU];
	#endif
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	#if(AMD)
	ptot = fma(0.5, bsq, P);
	#else
	ptot = P + 0.5*bsq;
	#endif

	/* particle number flux */
	flux[RHO] = pr[RHO] * q->ucon[dir];

	/* single row of mhd stress tensor,
	* first index up, second index down */
	#if(AMD)
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
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * P * pow(pr[RHO], -GAMMA);
	#endif
	#endif

	#pragma unroll 9
	for (k = 0; k < NPR_U;k++) flux[k] *= geom->g;

	//Calculate wavespeed
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

		//find fast magnetosonic speed
		cs2 = GAMMA*(GAMMA - 1.)*pr[UU] / w;
		va2 = bsq / eta;
		cms2 = cs2 + va2 - cs2*va2;	/* and there it is... */

		//check on it!
		if (cms2 < 0.) cms2 = SMALL;
		if (cms2 > 1.) cms2 =1.;

		//now require that speed of wave measured by observer q->ucon is cms2
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
	}
	return;
}

//Calculate radiative wave velocity
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int dir, double* vmax, double* vmin, double dx) {
	#if(RAD_M1)
	double discr, vp, vm, tau, kappa_tot, crad2, cmin_rad, cmax_rad, cmin_mhd, cmax_mhd;
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

	/* find radiation wave speed */
	crad2 = 1.0 / 3.0;

	/* check on it! */
	if (crad2 < 0.) crad2 = SMALL;
	if (crad2 > 1.) crad2 = 1.;

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q_rad->ucon[dir];
	Bu = q_rad->ucon[0];
	AB = Acon_0;
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10)discr = 0.;

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	if (vp > vm) {
		cmax_rad = vp;
		cmin_rad = vm;
	}
	else {
		cmax_rad = vm;
		cmin_rad = vp;
	}

	/* find radiation wave speed */
	kappa_tot = calc_kappa_abs(pr) + calc_kappa_es(pr);
	tau = kappa_tot * sqrt(geom->gcov[(dir == 1) * 4 + (dir == 2) * 7 + (dir == 3) * 9]) * dx;
	crad2 = MY_MIN(pow(4. / (3. * tau), 2.), 0.999999999);

	/* check on it! */
	if (crad2 < 0.) crad2 = SMALL;
	if (crad2 > 1.) crad2 = 1.;

	/* now require that speed of wave measured by observer q->ucon is crad2 */
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;

	A = Bu2 - (Bsq + Bu2) * crad2;
	B = 2. * (AuBu - (AB + AuBu) * crad2);
	C = Au2 - (Asq + Au2) * crad2;

	discr = B * B - 4. * A * C;
	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.; 

	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	if (vp > vm) {
		cmax_mhd = vp;
		cmin_mhd = vm;
	}
	else {
		cmax_mhd = vm;
		cmin_mhd = vp;
	}

	/*Set velocity as minimum of optically thin and optically thick limit*/
	*vmax = MY_MIN(cmax_mhd, cmax_rad);
	*vmin = MY_MAX(cmin_mhd, cmin_rad);

	return;
	#endif
}

//Calculate total absorption opacity
__device__ double calc_kappa_abs(double* ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	double Tg = fabs(MMW * MH_CGS * (GAMMA - 1.) * (ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * ph[RHO] * MASS_DENSITY_SCALE));
	double Tr = fabs(pow(ph[UU_RAD] * ENERGY_DENSITY_SCALE * ARAD, 0.25));
	
	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO]) * pow(Tg, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * (Z_AB / 0.02) * Ye * pow(Tg, -1.7) * pow(Tr, -3.);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * pow(Tg, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Tg));
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * pow(Tg, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Tg)) * (1. + 4.4 * pow(10., -10.) * Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	kappa_abs = 1.7 * pow(10., -25.) * pow(Tg, -7. / 2.) * pow(MH_CGS, -2.);

	return(kappa_abs * (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS);
}

//Calculate total emmission opacity
__device__ double calc_kappa_emmit(double* ph) {
	double kappa_abs, kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff;
	double Ye = (1. + X_AB) / 2.;
	double Tg = fabs(MMW * MH_CGS * (GAMMA - 1.) * (ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * ph[RHO] * MASS_DENSITY_SCALE));

	kappa_m = 0.1 * Z_AB;
	kappa_h = 1.1 * pow(10., -25.) * sqrt(Z_AB * ph[RHO]) * pow(Tg, 7.7);
	kappa_chianti = 4.0 * pow(10., 34.) * (Z_AB / 0.02) * Ye * pow(Tg, -4.7);
	kappa_bf = 3.0 * pow(10., 25.) * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * pow(Tg, -3.5) * log(1. + 1.6);
	kappa_ff = 4.0 * pow(10., 22.) * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * pow(Tg, -3.5) * log(1. + 1.6) * (1. + 4.4 * pow(10., -10.) * Tg);
	kappa_abs = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff));
	kappa_abs = 1.7 * pow(10., -25.) * pow(Tg, -7. / 2.) * pow(MH_CGS, -2.);

	return(kappa_abs * (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS);
}

//Calculate total (electron) scattering opacity
__device__ double calc_kappa_es(double* ph) {
	double kappa_es;
	double Tg = MMW * MH_CGS * (GAMMA - 1.) * (ph[UU] * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * ph[RHO] * MASS_DENSITY_SCALE);
	kappa_es = 0.2 * (1 + X_AB) / (1. + pow(Tg / (4.5 * pow(10., 8.)), 0.86));
	kappa_es = 0.2 * (1 + X_AB);
	return(kappa_es * (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS);
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

/* find ucon, ucov, bcon, bcov from radiation primitive variables */
__device__ void get_state_rad(double * pr, struct of_geom * geom, struct of_state_rad * q_rad)
{
	/* get radiation ucon */
	ucon_calc_rad(pr, geom, q_rad->ucon);
	lower(q_rad->ucon, geom->gcov, q_rad->ucov);

	return;
}

/* Raises a covariant rank-1 tensor to a contravariant one */
__device__ void raise(double ucov[NDIM], double gcon[10], double ucon[NDIM])
{
	#if AMD
	ucon[0] = fma(gcon[0], ucov[0], fma(gcon[1], ucov[1], fma(gcon[2], ucov[2], gcon[3] * ucov[3])));
	ucon[1] = fma(gcon[1], ucov[0], fma(gcon[4], ucov[1], fma(gcon[5], ucov[2], gcon[6] * ucov[3])));
	ucon[2] = fma(gcon[2], ucov[0], fma(gcon[5], ucov[1], fma(gcon[7], ucov[2], gcon[8] * ucov[3])));
	ucon[3] = fma(gcon[3], ucov[0], fma(gcon[6], ucov[1], fma(gcon[8], ucov[2], gcon[9] * ucov[3])));
	#else
	ucon[0] = gcon[0] * ucov[0]+ gcon[1] * ucov[1]+ gcon[2] * ucov[2]+ gcon[3] * ucov[3];
	ucon[1] = gcon[1] * ucov[0]+ gcon[4] * ucov[1]+ gcon[5] * ucov[2]+ gcon[6] * ucov[3];
	ucon[2] = gcon[2] * ucov[0]+ gcon[5] * ucov[1]+ gcon[7] * ucov[2]+ gcon[8] * ucov[3];
	ucon[3] = gcon[3] * ucov[0]+ gcon[6] * ucov[1]+ gcon[8] * ucov[2]+ gcon[9] * ucov[3];
	#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower(double ucon[NDIM], double gcov[10], double ucov[NDIM])
{
	#if AMD
	ucov[0] = fma(gcov[0], ucon[0], fma(gcov[1], ucon[1], fma(gcov[2], ucon[2],gcov[3] * ucon[3])));
	ucov[1] = fma(gcov[1], ucon[0], fma(gcov[4], ucon[1], fma(gcov[5], ucon[2],gcov[6] * ucon[3])));
	ucov[2] = fma(gcov[2], ucon[0], fma(gcov[5], ucon[1], fma(gcov[7], ucon[2],gcov[8] * ucon[3])));
	ucov[3] = fma(gcov[3], ucon[0], fma(gcov[6], ucon[1], fma(gcov[8], ucon[2],gcov[9] * ucon[3])));
	return;
	#else
	ucov[0] = gcov[0]*ucon[0] + gcov[1]*ucon[1] + gcov[2]*ucon[2] + gcov[3]*ucon[3] ;
	ucov[1] = gcov[1]*ucon[0] + gcov[4]*ucon[1] + gcov[5]*ucon[2] + gcov[6]*ucon[3] ;
	ucov[2] = gcov[2]*ucon[0] + gcov[5]*ucon[1] + gcov[7]*ucon[2] + gcov[8]*ucon[3] ;
	ucov[3] = gcov[3]*ucon[0] + gcov[6]*ucon[1] + gcov[8]*ucon[2] + gcov[9]*ucon[3] ;
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

/* find contravariant radiation four-velocity */
__device__ void ucon_calc_rad(double * pr, struct of_geom * geom, double *ucon_rad)
{
	double alpha, gamma;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0]);
	#pragma unroll 4
	SLOOPA beta[j] = geom->gcon[j] * alpha*alpha;

	gamma_calc_rad(pr, geom, &gamma);

	ucon_rad[0] = gamma / alpha;
	#if AMD
	#pragma unroll 4
	SLOOPA ucon_rad[j] = fma(-gamma, beta[j] / alpha, pr[U1_RAD + j - 1]);
	#else
	#pragma unroll 4
	SLOOPA ucon_rad[j] = pr[U1_RAD + j - 1] - gamma*beta[j] / alpha;
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
	for (j = 1; j<4; j++) bcon[j] = (pr[B1 - 1 + j] + bcon[0] * ucon[j]) / ucon[0];
	#endif
	return;
}

__device__ int gamma_calc(double *  pr, struct of_geom *  geom, double *  gamma)
{
	double qsq;
	#if AMD
	qsq = fma(geom->gcov[4], pr[U1] * pr[U1], fma(geom->gcov[7], pr[U2] * pr[U2],geom->gcov[9] * pr[U3] * pr[U3]))
		+ 2.*fma(geom->gcov[5], pr[U1] * pr[U2], fma(geom->gcov[6], pr[U1] * pr[U3],geom->gcov[8] * pr[U2] * pr[U3]));
	#else
	qsq = geom->gcov[4] * pr[U1] * pr[U1]+ geom->gcov[7] * pr[U2] * pr[U2]+ geom->gcov[9] * pr[U3] * pr[U3]
		+ 2.*(geom->gcov[5] * pr[U1] * pr[U2]+ geom->gcov[6] * pr[U1] * pr[U3] + geom->gcov[8] * pr[U2] * pr[U3]);
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

__device__ int gamma_calc_rad(double *  pr, struct of_geom *  geom, double *  gamma_rad)
{
	double qsq_rad;
	#if AMD
	qsq_rad = fma(geom->gcov[4], pr[U1_RAD] * pr[U1_RAD], fma(geom->gcov[7], pr[U2_RAD] * pr[U2_RAD],geom->gcov[9] * pr[U3_RAD] * pr[U3_RAD]))
		+ 2.*fma(geom->gcov[5], pr[U1_RAD] * pr[U2_RAD], fma(geom->gcov[6], pr[U1_RAD] * pr[U3_RAD],geom->gcov[8] * pr[U2_RAD] * pr[U3_RAD]));
	#else
	qsq_rad = geom->gcov[4] * pr[U1_RAD] * pr[U1_RAD]+ geom->gcov[7] * pr[U2_RAD] * pr[U2_RAD]+ geom->gcov[9] * pr[U3_RAD] * pr[U3_RAD]
		+ 2.*(geom->gcov[5] * pr[U1_RAD] * pr[U2_RAD]+ geom->gcov[6] * pr[U1_RAD] * pr[U3_RAD]+ geom->gcov[8] * pr[U2_RAD] * pr[U3_RAD]);
	#endif

	if (qsq_rad < 0.) {
		if (fabs(qsq_rad) > 1.E-10) { // then assume not just machine precision
			*gamma_rad = 1.;
			return (1);
		}
		else qsq_rad = 1.E-10; // set floor
	}

	*gamma_rad = sqrt(1. + qsq_rad);

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
		gamma_calc(pr, &geom, &gamma);
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
		vsq += geom.gcov[9] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 3 - 1]; //3,3
		
		vsq = MY_MAX(1.e-13,vsq);
		if (vsq >= 1.) {
			vsq = 1. - 1. / (GAMMAMAX*GAMMAMAX);
		}
		gamma = 1. / sqrt(1. - vsq);
		pr[U1] *= gamma;
		pr[U2] *= gamma;
		pr[U3] *= gamma;
	}

	#if(RAD_M1)
	double ucon_rad[NDIM], gamma_rad, vsq_rad;
	ucon_calc_rad(pr, &geom, ucon_rad);
	if (((ucon_rad[1] > 0.) && (type == 0)) || ((ucon_rad[1] < 0.) && (type == 1))) {
		/* find gamma and remove it from primitives */
		gamma_calc_rad(pr, &geom, &gamma_rad);
		pr[U1_RAD] /= gamma_rad;
		pr[U2_RAD] /= gamma_rad;
		pr[U3_RAD] /= gamma_rad;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[U1_RAD] = beta1 / alpha;

		// now find new gamma and put it back in 		
		vsq_rad = geom.gcov[4] * pr[U1_RAD + 1 - 1] * pr[U1_RAD + 1 - 1]; //1,1
		vsq_rad += 2. * geom.gcov[5] * pr[U1_RAD + 2 - 1] * pr[U1_RAD + 1 - 1]; //1,2
		vsq_rad += 2. * geom.gcov[6] * pr[U1_RAD + 3 - 1] * pr[U1_RAD + 1 - 1]; //1,3
		vsq_rad += geom.gcov[7] * pr[U1_RAD + 2 - 1] * pr[U1_RAD + 2 - 1]; //2,2
		vsq_rad += 2 * geom.gcov[8] * pr[U1_RAD + 3 - 1] * pr[U1_RAD + 2 - 1]; //2,3
		vsq_rad += geom.gcov[9] * pr[U1_RAD + 3 - 1] * pr[U1_RAD + 3 - 1]; //3,3

		vsq_rad = MY_MAX(1.e-13, vsq_rad);
		if (vsq_rad >= 1.) {
			vsq_rad = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		}
		gamma_rad = 1. / sqrt(1. - vsq_rad);
		pr[U1_RAD] *= gamma_rad;
		pr[U2_RAD] *= gamma_rad;
		pr[U3_RAD] *= gamma_rad;

		/* done */
	}
	#endif
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
	int zsize = 1, zoffset = 0, z2 = 0, z3 = 0, z4 = 0;
	double x2, x3, x4;
	double temp;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int  zlevel = 0;
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
	int  z1 = -2, z2 = -1, z3 = 0, z4 = 1, z5 = 2;
	double x1, x2, x3, x4, x5, FF=0.;
	double temp, result;
	if (dir == 1) { idel = 1; jdel = 0; zdel = 0; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; }

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zsize = 1, zlevel = 0, zoffset = 0;
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
	int zsize = 1, zoffset = 0, u;
	int zsize2 = 1, zoffset2 = 0;
	double temp[NPR];

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int  zlevel = 0, zlevel2 = 0;
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
	double cour, double*  dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag)
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
	int idel, jdel, zdel, i, face;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double factor;
	double cmax_r, cmin_r, cmax, cmin, cmax_l, cmin_l, ctop, ctop_rad;
	double temp1[NPR], temp2[NPR], temp3[NPR], temp4[NPR], p[NPR];
	struct of_geom geom;
	struct of_state state;
	#if(RAD_M1)
	double cmax_r_rad, cmin_r_rad, cmax_l_rad, cmin_l_rad, cmax_rad, cmin_rad;
	struct of_state_rad state_rad;
	#endif
	local_dtij[local_id] = 1.e9;
	int zsize = 1, zoffset = 0;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx*((double)zsize); }

	if (k == 1){
		get_geometry(icurr, jcurr, zcurr, face, &geom, gcov, gcon, gdet);

		//Get left state
		if (zoffset != 0 && dir == 3){
			#pragma unroll 9	
			for (k = 0; k < NPR; k++){
				p[k] = 0.5 * (pv[k * (ksize)+global_id] + pv[k * (ksize)+global_id - D3]);
			}
		}
		else{
			#pragma unroll 9	
			for (k = 0; k < NPR; k++){
				p[k] = dq2[k * (ksize)+global_id - idel * isize - jdel * (BS_3 + 2 * N3G) - zdel];
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
		primtoflux(p, &state, dir, &geom, temp1, &cmax_l, &cmin_l);
		primtoflux(p, &state, 0, &geom, temp2, &cmax_l, &cmin_l);
		#if(RAD_M1)
		get_state_rad(p, &geom, &state_rad);
		primtoflux_rad(p, &state_rad, dir, &geom, temp3);
		primtoflux_rad(p, &state_rad, 0, &geom, temp4);
		vchar_rad(p, &state, &state_rad, &geom, dir, &cmax_l_rad, &cmin_l_rad, dx);
		#endif

		//Get right state
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
		primtoflux(p, &state, dir, &geom, temp3, &cmax_r, &cmin_r);
		primtoflux(p, &state, 0, &geom, temp4, &cmax_r, &cmin_r);

		cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
		cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
		ctop = MY_MAX(cmax, cmin);

		#if(RAD_M1)
		if (flag == 1) {
			for (k = 0; k < NPR_U; k++) {
				#if(HLLF)
				F[k * (ksize)+global_id] = 0.5 * (F[k * (ksize)+global_id] + (cmax * temp1[k] + cmin * temp3[k] - cmax * cmin * (temp4[k] - temp2[k])) / (cmax + cmin + SMALL));
				#else
				F[k * (ksize)+global_id] = 0.5 * (F[k * (ksize)+global_id] + LAXF * (0.5 * (temp1[k] + temp3[k] - ctop * (temp4[k] - temp2[k]))));
				#endif
			}
		}
		else {
			for (k = 0; k < NPR_U; k++) {
				#if(HLLF)
				F[k * (ksize)+global_id] = (cmax * temp1[k] + cmin * temp3[k] - cmax * cmin * (temp4[k] - temp2[k])) / (cmax + cmin + SMALL);
				#else
				F[k * (ksize)+global_id] = (0.5 * (temp1[k] + temp3[k] - ctop * (temp4[k] - temp2[k])));
				#endif
			}
		}
		get_state_rad(p, &geom, &state_rad);
		primtoflux_rad(p, &state_rad, dir, &geom, temp3);
		primtoflux_rad(p, &state_rad, 0, &geom, temp4);
		
		vchar_rad(p, &state, &state_rad, &geom, dir, &cmax_r_rad, &cmin_r_rad, dx);
		
		cmax_rad = fabs(MY_MAX(MY_MAX(0., cmax_l_rad), cmax_r_rad));
		cmin_rad = fabs(MY_MAX(MY_MAX(0., -cmin_l_rad), -cmin_r_rad));
		ctop_rad = MY_MAX(cmax_rad, cmin_rad);
		if (flag == 1) {
			for (k = UU_RAD; k <= U3_RAD; k++) {
				#if(HLLF)
				F[k * (ksize)+global_id] = 0.5 * (F[k * (ksize)+global_id] + (cmax_rad * temp1[k] + cmin_rad * temp3[k] - cmax_rad * cmin_rad * (temp4[k] - temp2[k])) / (cmax_rad + cmin_rad + SMALL));
				#else
				F[k * (ksize)+global_id] = 0.5 * (F[k * (ksize)+global_id] + 0.5 * (temp1[k] + temp3[k] - ctop_rad * (temp4[k] - temp2[k])));
				#endif
			}
		}
		else {
			for (k = UU_RAD; k <= U3_RAD; k++) {
				#if(HLLF)
				F[k * (ksize)+global_id] = (cmax_rad * temp1[k] + cmin_rad * temp3[k] - cmax_rad * cmin_rad * (temp4[k] - temp2[k])) / (cmax_rad + cmin_rad + SMALL);
				#else
				F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_rad * (temp4[k] - temp2[k]));
				#endif
			}
		}
		#else
		for (k = 0; k < NPR_U; k++) {
			#if(HLLF)
			F[k * (ksize)+global_id] = (cmax * temp1[k] + cmin * temp3[k] - cmax * cmin * (temp4[k] - temp2[k])) / (cmax + cmin + SMALL);
			#else
			F[k * (ksize)+global_id] = (0.5 * (temp1[k] + temp3[k] - ctop * (temp4[k] - temp2[k])));
			#endif
		}
		#endif

		/* evaluate restriction on timestep */
		#if(RAD_M1)
		ctop = MY_MAX(ctop, ctop_rad);
		#endif

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

	#if(DOKTOT)
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * P * pow(pr[RHO], -GAMMA);
	#endif
	#endif
}

__device__ void vchar_FT(double * pr, double ucon[NDIM], double bcon[NDIM], int dir, double *vmax, double *vmin)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2;
	double Asq, Bsq, Au, Bu, Au2, Bu2, AuBu, A, B, C;

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

__device__ void vchar(double *pr, struct of_state *q, struct of_geom *geom, int dir, double *vmax, double *vmin)
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
	#if AMD
	w = fma(gam, pr[UU], pr[RHO]);
	#else
	w = pr[RHO] + GAMMA*pr[UU];
	#endif
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;
	cs2 = GAMMA*(GAMMA - 1.)*pr[UU] / w;
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

__global__ void consttransport2_M1_2(double* emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2)
{
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + D3) * (BS_2 + D2);
	zcurr = (global_id % (isize)) % (BS_3 + D3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + D3);
	icurr = (global_id - (jcurr * (BS_3 + D3) + zcurr)) / (isize);
	zcurr += (N3G)*D3;
	jcurr += (N2G)*D2;
	icurr += (N1G)*D1;
	if (global_id < (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) k = 1;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	int jsize = BS_3 + 2 * N3G;

	if (k == 1) {
		double dE_LEFT_13_1 = E_cent[1 * (ksize)+global_id] - F3[B2 * (ksize)+global_id];
		double dE_LEFT_13_2 = E_cent[1 * (ksize)+global_id - jsize * D2] - F3[B2 * (ksize)+global_id - jsize * D2];
		double dE_RIGHT_13_1 = F3[B2 * (ksize)+global_id + D3 - D3] - E_cent[1 * (ksize)+global_id - D3];
		double dE_RIGHT_13_2 = F3[B2 * (ksize)+global_id + D3 - jsize * D2 - D3] - E_cent[1 * (ksize)+global_id - jsize * D2 - D3];
		double dE_LEFT_12_1 = E_cent[1 * (ksize)+global_id] + F2[B3 * (ksize)+global_id];
		double dE_LEFT_12_2 = E_cent[1 * (ksize)+global_id - D3] + F2[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_12_1 = -F2[B3 * (ksize)+global_id + D2 * jsize - D2 * jsize] - E_cent[1 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_12_2 = -F2[B3 * (ksize)+global_id + D2 * jsize - D2 * jsize - D3] - E_cent[1 * (ksize)+global_id - D2 * jsize - D3];
		double dE_LEFT_21_1 = E_cent[2 * (ksize)+global_id] - F1[B3 * (ksize)+global_id];
		double dE_LEFT_21_2 = E_cent[2 * (ksize)+global_id - D3] - F1[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_21_1 = F1[B3 * (ksize)+global_id + D1 * isize - D1 * isize] - E_cent[2 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_21_2 = F1[B3 * (ksize)+global_id + D1 * isize - D1 * isize - D3] - E_cent[2 * (ksize)+global_id - D1 * isize - D3];
		double dE_LEFT_23_1 = E_cent[2 * (ksize)+global_id] + F3[B1 * (ksize)+global_id];
		double dE_LEFT_23_2 = E_cent[2 * (ksize)+global_id - D1 * isize] + F3[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_23_1 = -F3[B1 * (ksize)+global_id + D3 - D3] - E_cent[2 * (ksize)+global_id - D3];
		double dE_RIGHT_23_2 = -F3[B1 * (ksize)+global_id + D3 - isize * D1 - D3] - E_cent[2 * (ksize)+global_id - isize * D1 - D3];
		double dE_LEFT_31_1 = E_cent[3 * (ksize)+global_id] + F1[B2 * (ksize)+global_id];
		double dE_LEFT_31_2 = E_cent[3 * (ksize)+global_id - D2 * jsize] + F1[B2 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_31_1 = -F1[B2 * (ksize)+global_id + D1 * isize - D1 * isize] - E_cent[3 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_31_2 = -F1[B2 * (ksize)+global_id + D1 * isize - D1 * isize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		double dE_LEFT_32_1 = E_cent[3 * (ksize)+global_id] - F2[B1 * (ksize)+global_id];
		double dE_LEFT_32_2 = E_cent[3 * (ksize)+global_id - D1 * isize] - F2[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_32_1 = F2[B1 * (ksize)+global_id + D2 * jsize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_32_2 = F2[B1 * (ksize)+global_id + D2 * jsize - D1 * isize - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		
		emf[1 * (ksize)+global_id] *= 0.5;
		emf[2 * (ksize)+global_id] *= 0.5;
		emf[3 * (ksize)+global_id] *= 0.5;

		emf[1 * (ksize)+global_id] += 0.25 * 0.5 * ((-F2[B3 * (ksize)+global_id] - (dE_LEFT_13_1 * (double)(F2[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_13_2 * (double)(F2[RHO * (ksize)+global_id] > 0.0)))
			+ (-F2[B3 * (ksize)+global_id - D3] + (dE_RIGHT_13_1 * (double)(F2[RHO * (ksize)+global_id - D3] <= 0.0) + dE_RIGHT_13_2 * (double)(F2[RHO * (ksize)+global_id - D3] > 0.0))) +
			+(F3[B2 * (ksize)+global_id] - (dE_LEFT_12_1 * (double)(F3[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_12_2 * (double)(F3[RHO * (ksize)+global_id] > 0.0)))
			+ (F3[B2 * (ksize)+global_id - D2 * jsize] + (dE_RIGHT_12_1 * (double)(F3[RHO * (ksize)+global_id - D2 * jsize] <= 0.0) + dE_RIGHT_12_2 * (double)(F3[RHO * (ksize)+global_id - D2 * jsize] > 0.0))));
		emf[2 * (ksize)+global_id] += 0.25 * 0.5 * ((-F3[B1 * (ksize)+global_id] - (dE_LEFT_21_1 * (double)(F3[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_21_2 * (double)(F3[RHO * (ksize)+global_id] > 0.0)))
			+ (-F3[B1 * (ksize)+global_id - D1 * isize] + (dE_RIGHT_21_1 * (double)(F3[RHO * (ksize)+global_id - D1 * isize] <= 0.0) + dE_RIGHT_21_2 * (double)(F3[RHO * (ksize)+global_id - D1 * isize] > 0.0)))
			+ (F1[B3 * (ksize)+global_id] - (dE_LEFT_23_1 * (double)(F1[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_23_2 * (double)(F1[RHO * (ksize)+global_id] > 0.0)))
			+ (F1[B3 * (ksize)+global_id - D3] + (dE_RIGHT_23_1 * (double)(F1[RHO * (ksize)+global_id - D3] <= 0.0) + dE_RIGHT_23_2 * (double)(F1[RHO * (ksize)+global_id - D3] > 0.0))));
		emf[3 * (ksize)+global_id] += 0.25 * 0.5 * ((F2[B1 * (ksize)+global_id] - (dE_LEFT_31_1 * (double)(F2[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_31_2 * (double)(F2[RHO * (ksize)+global_id] > 0.0)))
			+ (F2[B1 * (ksize)+global_id - D1 * isize] + (dE_RIGHT_31_1 * (double)(F2[RHO * (ksize)+global_id - D1 * isize] <= 0.0) + dE_RIGHT_31_2 * (double)(F2[RHO * (ksize)+global_id - D1 * isize] > 0.0)))
			+ (-F1[B2 * (ksize)+global_id] - (dE_LEFT_32_1 * (double)(F1[RHO * (ksize)+global_id] <= 0.0) + dE_LEFT_32_2 * (double)(F1[RHO * (ksize)+global_id] > 0.0)))
			+ (-F1[B2 * (ksize)+global_id - D2 * jsize] + (dE_RIGHT_32_1 * (double)(F1[RHO * (ksize)+global_id - D2 * jsize] <= 0.0) + dE_RIGHT_32_2 * (double)(F1[RHO * (ksize)+global_id - D2 * jsize] > 0.0))));

		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == BS_2 + N2G)) {
			emf[3 * (ksize)+global_id] = 0.;
			emf[1 * (ksize)+global_id] += -0.5 * 0.5 * (F2[B3 * (ksize)+global_id] + F2[B3 * (ksize)+global_id - D3]);
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
	int zsize = 1, zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
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
	int zsize = 1,  zoffset=0, u;
	double temp;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
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

__global__ void Utoprim_M1_0( double* p_i, double* U_n, double* dU_RAD0, const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2)
{
	#if(RAD_M1)
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * N2G);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * N3G);
	icurr = (global_id - (jcurr * (BS_3 + 2 * N3G)+zcurr)) / (isize);
	if (global_id < (BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) k = 1;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	struct of_geom geom;
	struct of_state q;
	struct of_state_rad q_rad;
	double p[NPR], dU[NPR], U[NPR], U_0[NPR], cell_size;
	double y = 1.0 - 1.0 / sqrt(2.0);
	int zsize = 1, zoffset = 0, u;
	int pflag_local, pflag_rad_local;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001 + pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			p[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				p[k] += (1.0 / ((double)zsize)) * p_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(p, &geom, &q);
		primtoflux(p, &q, 0, &geom, U, NULL, NULL);
		get_state_rad(p, &geom, &q_rad);
		primtoflux_rad(p, &q_rad, 0, &geom, U);

		//Perform implicit solve
		cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(p, U_n, U_n, U_0, &pflag_local, &pflag_rad_local, &geom, dU, Dt * Y_IMEX, cell_size, y_max);

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(p, radius[icurr], &geom, &q)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			U_n[k * (ksize)+global_id] = U[k];
			dU_RAD0[k * (ksize)+global_id] = dU[k];
			p_i[k * (ksize)+global_id] = p[k];
		}
	}
	#endif
}

__global__ void Utoprim_M1_1(double* ph_i, const  double* __restrict__ p_i, const double* __restrict__ U_n, double* dU_MHD1, const double* __restrict__ dU_RAD0, double* dU_RAD1, const  double* __restrict__  psh, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2)
{
	#if(RAD_M1)
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3) * (BS_2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr * (BS_3)+zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	if (global_id < (BS_1) * (BS_2) * (BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	struct of_geom geom;
	struct of_state q;
	double p[NPR], dU[NPR], U_1[NPR], U_n_tmp[NPR], cell_size;
	int zsize = 1, zoffset = 0, u;
	int pflag_local, pflag_rad_local;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001 + pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			p[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				p[k] += (1.0 / ((double)zsize)) * p_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(p, &geom, &q);
		source(p, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]);

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			U_1[k] = 0.;
			for (u = 0; u < zsize; u++) {
				U_1[k] += U_n[k * (ksize)+global_id - zoffset + u] / ((double)zsize);
				#if( N1G > 0 )
				U_1[k] -= Dt * (F1[k * (ksize)+global_id + isize - zoffset + u] - F1[k * (ksize)+global_id - zoffset + u]) / (dx_1 * (double)zsize);
				#endif
				#if( N2G > 0 )
				U_1[k] -= Dt * (F2[k * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k * (ksize)+global_id - zoffset + u]) / (dx_2 * (double)zsize);
				#endif
				U_1[k] += Dt * (dU[k] + (1. - 2. * Y_IMEX) * dU_RAD0[k * (ksize)+global_id - zoffset + u] ) / ((double)zsize);
			}
			#if( N3G > 0 )
			U_1[k] -= Dt * (F3[k * (ksize)+global_id - zoffset + zsize] - F3[k * (ksize)+global_id - zoffset]) / (dx_3 * (double)zsize);
			#endif
			dU_MHD1[k * (ksize)+global_id + isize - zoffset] = dU[k];
		}

		#if(NSY)
		U_1[B1] = 0.0;
		U_1[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++) {
			U_1[B1] += (psh[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			U_1[B2] += (psh[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
		}
		#if(N3G>0)
		U_1[B3] = (psh[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset] + psh[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
		#endif
		#endif
		#else
		#if(STAGGERED)
		U_1[B1] = 0.0;
		U_1[B2] = 0.0;
		for (u = 0; u < zsize; u++) {
			U_1[B1] += (psh[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_2 + 2 * N2G) + jcurr]) / (2.0 * (double)zsize);
			U_1[B2] += (psh[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0 * (double)zsize);
		}
		#if(N3G>0)
		U_1[B3] = (psh[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif
		#endif

		//Set temporary variable
		for (k = 0; k < NPR; k++) {
			U_n_tmp[k] = U_n[k * (ksize)+global_id];
		}

		//Perform implicit solve
		cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(p, U_n_tmp, U_1, U_1, &pflag_local, &pflag_rad_local, &geom, dU, Y_IMEX * Dt, cell_size, y_max);

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(p, radius[icurr], &geom, &q)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			ph_i[k * (ksize)+global_id] = p[k];
			dU_RAD1[k * (ksize)+global_id] = dU[k];
		}
	}
	#endif
}

__global__ void Utoprim_M1_2(const  double* __restrict__ ph_i, double* p_i, const double* __restrict__ U_n, const double* __restrict__ dU_MHD1, const double* __restrict__ dU_RAD0, const double* __restrict__ dU_RAD1, const  double* __restrict__  ps, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2)
{
	#if(RAD_M1)
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3) * (BS_2);
	zcurr = (global_id % (isize)) % (BS_3);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3);
	icurr = (global_id - (jcurr * (BS_3)+zcurr)) / (isize);
	zcurr += N3G;
	jcurr += N2G;
	icurr += N1G;
	if (global_id < (BS_1) * (BS_2) * (BS_3)) k = 1;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(NSY)
	int fix_mem2 = fix_mem1;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	struct of_geom geom;
	struct of_state q;
	double ph[NPR], dU[NPR], U_2[NPR];
	int zsize = 1, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001 + pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			ph[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				ph[k] += (1.0 / ((double)zsize)) * ph_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(ph, &geom, &q);
		source(ph, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]);

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			U_2[k] = 0.;
			for (u = 0; u < zsize; u++) {
				U_2[k] += U_n[k * (ksize) + global_id - zoffset + u] / ((double)zsize);
				#if( N1G > 0 )
				U_2[k] -= Dt * (F1[k * (ksize) + global_id + isize - zoffset + u] - F1[k * (ksize) + global_id - zoffset + u]) / (dx_1 * (double)zsize);
				#endif
				#if( N2G > 0 )
				U_2[k] -= Dt * (F2[k * (ksize) + global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k * (ksize) + global_id - zoffset + u]) / (dx_2 * (double)zsize);
				#endif
				U_2[k] += 0.5 * Dt * (dU[k] + dU_MHD1[k * (ksize) + global_id - zoffset + u] + dU_RAD0[k * (ksize) + global_id - zoffset + u] + dU_RAD1[k * (ksize) + global_id - zoffset + u]) / ((double)zsize);
			}
			#if( N3G > 0 )
			U_2[k] -= Dt * (F3[k * (ksize)+global_id - zoffset + zsize] - F3[k * (ksize)+global_id - zoffset]) / (dx_3 * (double)zsize);
			#endif
		}

		#if(NSY)
		U_2[B1] = 0.0;
		U_2[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++) {
			U_2[B1] += (ps[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + ps[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			U_2[B2] += (ps[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
		}
		#if(N3G>0)
		U_2[B3] = (ps[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset] + ps[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
		#endif
		#endif
		#else
		#if(STAGGERED)
		U_2[B1] = 0.0;
		U_2[B2] = 0.0;
		for (u = 0; u < zsize; u++) {
			U_2[B1] += (ps[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + ps[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_2 + 2 * N2G) + jcurr]) / (2.0 * (double)zsize);
			U_2[B2] += (ps[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0 * (double)zsize);
		}
		#if(N3G>0)
		U_2[B3] = (ps[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + ps[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif
		#endif

		#if(NEWMAN)
		pflag[global_id] = Utoprim_NM(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC);
		#else
		pflag[global_id] = Utoprim_2d(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC);
		#endif

		#if( DO_FONT_FIX ) 
		if (pflag[global_id]) {
			failimage[global_id]++;
			#if DOKTOT
			pflag[global_id] = Utoprim_1dvsq2fix1(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC);
			#endif
			if (pflag[global_id]) {
				failimage[1 * (ksize)+global_id]++;
				pflag[global_id] = Utoprim_1dfix1(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC);
				if (pflag[global_id]) {
					pflag[0] = global_id;
					failimage[2 * (ksize)+global_id]++;
				}
			}
		}
		#endif
		pflag_rad[global_id] = Rtoprim(U_2, geom.gcov, geom.gcon, geom.g, ph, y_max, BASIC);

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(ph, radius[icurr], &geom, &q)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			p_i[k * (ksize)+global_id] = ph[k];
		}
	}
	#endif
}

//For P100/V100 GPUs replace Utoprim0, Utoprim1, Utoprim2, fixup by this kernel
__global__ void fixup(double* pi_i, double* pb_i, double* pf_i, double* storage2, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2)
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
	double pf[NPR],  dU[NPR], U[NPR];

	int zsize = 1, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
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
			primtoflux(pf, &q,0, &geom, U, NULL, NULL);
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
			get_state(pf, &geom, &q);
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

		source(pf, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]);

		#pragma unroll 9	
		for (k = 0; k< NPR; k++){
			U[k] += Dt*(dU[k]);
		}

		#if(NSY)
		U[B1] = 0.0;
		U[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++){
			U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + D2)*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
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
		pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
		#else
		pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
		#endif

		#if( DO_FONT_FIX ) 
		if (pflag[global_id]) {
			failimage[global_id]++;
			#if DOKTOT
			pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
			#endif
			if (pflag[global_id]) {
				failimage[1 * (ksize)+global_id]++;
				pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
				if (pflag[global_id]){
					pflag[0] = global_id;
					failimage[2 * (ksize)+global_id]++;
				}
			}
		}
		#endif

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(pf, radius[icurr], &geom, &q)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k< NPR; k++){
			pf_i[k*(ksize)+global_id] = pf[k];
		}
	}
}

__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2)
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
	double pf[NPR], U[NPR];
	int zsize = 1, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
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
			primtoflux(pf, &q, 0, &geom, U, NULL, NULL);

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
				U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
				U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + D2)*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
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
			pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
			#else
			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
			#endif

			//compute the square of fluid frame magnetic field (twice magnetic pressure)
			#if( DO_FONT_FIX ) 
			if (pflag[global_id]) {
				failimage[global_id]++;
				#if DOKTOT
				pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
				#endif
				if (pflag[global_id]) {
					failimage[1 * (ksize)+global_id]++;
					pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC);
					if (pflag[global_id]){
						pflag[0] = global_id;
						failimage[2 * (ksize)+global_id]++;
					}
				}
			}
			#endif

			//Apply floors in ZAMO frame or drift frame
			if (fixup_cell(pf, radius[icurr], &geom, &q)){
				pflag[global_id] = -333;
				pflag[0] = global_id;;
				failimage[3 * (ksize)+global_id]++;
			}

			#pragma unroll 9	
			for (k = 0; k < NPR; k++){
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
			pv[U2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[U2 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[U2_RAD * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

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
			pv[U2 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
			#endif
			pv[B2 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
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
			pv[U2 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[U2 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[U2_RAD * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

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
			pv[U2 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
			#endif
			pv[B2 * (ksize)+isize*icurr + j*(BS_3 + 2 * N3G) + zcurr] *= -1.;
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
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[U3_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			#endif
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
			#if(RAD_M1)
			pv[U2_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[U3_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			#endif
			pv[B2*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[B3*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] *= -1.0;

			#if(STAGGERED)
			ps[0 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			ps[2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -ps[2 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			#endif
		}
	}
}

__global__ void fluxcalc2D_FT(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet,
	const  double* __restrict__ Mud, const  double* __restrict__ Mud_inv, int lim, int dir, double cour, double*  dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag)
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
	int zsize = 1, zoffset = 0;
	double factor;
	double cmax_r, cmin_r, ctop, cmax_l, cmin_l, cmax[2], cmin[2], cmax_roe, cmin_roe;
	double p_l[NPR], p_r[NPR], F1[NPR], F_FT[2][NPR], F_HLL[2][NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR];
	double l_ucon[NDIM], r_ucon[NDIM], l_bcon[NDIM], r_bcon[NDIM], int_velocity;
	struct of_geom geom;
	struct of_state state_l, state_r;
	struct of_trans trans;
	local_dtij[local_id] = 1.e9;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001 + pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; factor = cour*dx; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; factor = cour*dx; }
	else if (dir == 3) {idel = 0; jdel = 0; zdel = 1; face = FACE3; factor = cour*dx*((double)zsize);}

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

__device__ void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double A, B, C, D, vcon, ptot;
	int k, fail_HLLC = 0;
	int UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		UGEN_1 = U1; UGEN_2 = U2; UGEN_3 = U3;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		UGEN_1 = U2; UGEN_2 = U3; UGEN_3 = U1;
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
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
	int GEN_1, GEN_2, GEN_3, UGEN_1, BGEN_1, BGEN_2, BGEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
		UGEN_1 = U1;
		BGEN_1 = B1; BGEN_2 = B2; BGEN_3 = B3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
		UGEN_1 = U2; 
		BGEN_1 = B2; BGEN_2 = B3; BGEN_3 = B1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
		UGEN_1 = U3;
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
	double ptot_old, de_dptot, d_ptot = 0.;
	error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

	while (keep_iterating) {
		//Calculate error and error/d_ptot
		error_2 = calc_error_HLLD(dir, 0, ptot + pow(10., -10.)*fabs(ptot), cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

		//Save old value of ptot
		ptot_old = ptot;

		//Make the newton step in log-space
		//de_dptot = (error_2 - error_1) / (pow(10., -8.)*ptot);
		//de_dlptot = de_dptot*ptot;
		//dlptot = error_1 / de_dlptot;
		//lptot = log(ptot_old) - dlptot;
		//ptot = exp(lptot);
		//d_ptot = ptot - ptot_old;

		de_dptot = (error_2 - error_1) / (pow(10., -10.) * fabs(ptot ));
		d_ptot = error_1 / de_dptot;
		ptot = ptot - d_ptot;


		if (ptot < 0.)ptot = 0.5 * fabs(ptot);

		//Calculate updated value of ptot
		error_2 = error_1;
		error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, vcon_cl, vcon_cr, eta_l, eta_r, w_al, w_ar);

		if ((fabs(ptot-ptot_old) <= pow(10., -8.) * fabs(ptot)) || n_iter > 10) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	//If Newton-Raphson solver did not converge, reset ptot to ptot_HLLC and tag fail_HLLD
	if (!(fabs(ptot) > 0.) || ((fabs(d_ptot) > pow(10., -8.) * fabs(ptot)))) {
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
	int GEN_1, GEN_2, GEN_3;

	if (dir == 1) {
		GEN_1 = 1; GEN_2 = 2; GEN_3 = 3;
	}
	else if (dir == 2) {
		GEN_1 = 2; GEN_2 = 3; GEN_3 = 1;
	}
	else if (dir == 3) {
		GEN_1 = 3; GEN_2 = 1; GEN_3 = 2;
	}

	vcon_cl[GEN_1] = (vcon_cl[GEN_1] + vcon_cr[GEN_1])*0.5;
	vcon_cr[GEN_1] = vcon_cl[GEN_1];
	vcon_cl[GEN_2] = (vcon_cl[GEN_2] + vcon_cr[GEN_2])*0.5;
	vcon_cr[GEN_2] = vcon_cl[GEN_2];
	vcon_cl[GEN_3] = (vcon_cl[GEN_3] + vcon_cr[GEN_3])*0.5;
	vcon_cr[GEN_3] = vcon_cl[GEN_3];

	//Check that contact wave lies between inner and outer Alfven speed
	if (vcon_cl[GEN_1] < K_al[GEN_1] || vcon_cl[GEN_1] < cmin_roe) fail_HLLD[0] = 1;
	if (vcon_cl[GEN_1] > K_ar[GEN_1] || vcon_cl[GEN_1] > cmax_roe) fail_HLLD[0] = 1;

	//Check that contact wave is going slower than v=0.99c
	vsq = (vcon_cl[1] * vcon_cl[1] + vcon_cl[2] * vcon_cl[2] + vcon_cl[3] * vcon_cl[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;

	//If wavefan inconsisten revert to HLLC
	if (fabs(w_al) <= fabs(ptot) || vcon_al[GEN_1] <= cmin_roe || K_al[GEN_1] <= cmin_roe || w_al <= 0.) fail_HLLD[0] = 1;
	if (fabs(w_ar) <= fabs(ptot) || vcon_ar[GEN_1] >= cmax_roe || K_ar[GEN_1] >= cmax_roe || w_ar <= 0.) fail_HLLD[0] = 1;

	//Check that v_al is going slower than v=0.99c
	vsq = (vcon_al[1] * vcon_al[1] + vcon_al[2] * vcon_al[2] + vcon_al[3] * vcon_al[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;

	//Check that v_ar is going slower than v=0.99c
	vsq = (vcon_ar[1] * vcon_ar[1] + vcon_ar[2] * vcon_ar[2] + vcon_ar[3] * vcon_al[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;

	//Check that left Alfven wave is going slower than v=0.99c
	vsq = (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;

	//Check that left Alfven wave is going slower than v=0.99c
	vsq = (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;
}

__device__ double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double vcon_cl[NDIM], double vcon_cr[NDIM], double *eta_l, double *eta_r, double  *w_al, double *w_ar) {
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;
	double A, C, G, X, Q, error = 0.;
	double delta_Kx, Y_l, Y_r, B_hat[NDIM];

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
	
	return (vcon_cr[GEN_1] - vcon_cl[GEN_1]);*/

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
	
	return error;
}
