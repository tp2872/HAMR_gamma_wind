#include "config.h"

/*Declarations of functions for Utoprim2D*/
#if(DOHELM)
__device__ int Utoprim_2d(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, const  double* __restrict__ gpu_eos_table);
__device__ int Utoprim_new_body(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double tolerance, int lim, const  double* __restrict__ gpu_eos_table);
__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance, const  double* __restrict__ gpu_eos_table);
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, const  double* __restrict__ gpu_eos_table);
__device__ int Utoprim_1dvsq2fix1(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, int full_entropy, const  double* __restrict__ gpu_eos_table);
__device__ double W_of_vsq2(double vsq, double* p, double* rho, double* u, double D, double K_atm, const  double* __restrict__ gpu_eos_table);
__device__ void dWdvsq_calc2_helmholtz(const  double* __restrict__ gpu_eos_table, double vsq, double D, double K_atm, double* W, double* dWdvsq);
__device__ int Utoprim_new_body2(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double K_atm, double tolerance, int lim, const  double* __restrict__ gpu_eos_table);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, const  double* __restrict__ gpu_eos_table);
__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance, const  double* __restrict__ gpu_eos_table);
__device__ int Utoprim_NM_calc(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, const  double* __restrict__ gpu_eos_table);
__device__ int Utoprim_NM(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, const  double* __restrict__ gpu_eos_table);
__device__ void vchar(double* pr, struct of_state* q, struct of_geom* geom, int dir, double* vmax, double* vmin, const  double* __restrict__ gpu_eos_table);
__device__ void primtoflux(double* pr, struct of_state* q, int dir, struct of_geom* geom, double* flux, double* vmax, double* vmin, const  double* __restrict__ gpu_eos_table);
__device__ int fixup_cell(double pf[NDIM], double r, struct of_geom* geom, const  double* __restrict__ gpu_eos_table);
__device__ void source(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double Dt, const  double* __restrict__ conn, struct of_state* q, double r, const  double* __restrict__ gpu_eos_table);
__device__ void mhd_calc(double* pr, int dir, struct of_state* q, double* mhd, const  double* __restrict__ gpu_eos_table);
__device__ void primtoflux_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR], const  double* __restrict__ gpu_eos_table);
__device__ void vchar_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double* vmax, double* vmin, const  double* __restrict__ gpu_eos_table);
__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table);
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table);

/*Declarations of functions related to (M1) radiation inversion scheme*/
__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged, const  double* __restrict__ gpu_eos_table);
__device__ int implicit_rad_solve_UMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged, const  double* __restrict__ gpu_eos_table);
__device__ int implicit_rad_solve_PRAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged, const  double* __restrict__ gpu_eos_table);
__device__ int implicit_rad_solve_URAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged, const  double* __restrict__ gpu_eos_table);
__device__ int implicit_rad_solve_EMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged, const  double* __restrict__ gpu_eos_table);

__device__ void source_rad(double* ph, struct of_geom* geom, struct of_state* q, struct of_state_rad* q_rad, double* dU, const  double* __restrict__ gpu_eos_table);
__device__ void calc_Gcon(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double mhd_rad[NDIM][NDIM],  double bsq,const  double* __restrict__ gpu_eos_table);

__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int js, double* vmax, double* vmin, double dx, const  double* __restrict__ gpu_eos_table);
__device__ double calc_kappa_abs(double* ph, double bsq, double Tr, const  double* __restrict__ gpu_eos_table
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_emmit(double* ph, double bsq, double Tr, const  double* __restrict__ gpu_eos_table
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_es(double* ph, const  double* __restrict__ gpu_eos_table);
#else 
__device__ int Utoprim_2d(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if(TWO_T)
	, double fel
	#endif
);
__device__ int Utoprim_new_body(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double tolerance, int lim
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_1dvsq2fix1(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, int full_entropy
	#if(TWO_T)
	, double fel
	#endif
);
__device__ double W_of_vsq2(double vsq, double* p, double* rho, double* u, double D, double K_atm
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_new_body2(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double K_atm, double tolerance, int lim
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_NM(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if(TWO_T)
	, double fel
	#endif
);
__device__ int Utoprim_NM_calc(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if(TWO_T)
	, double *S
	, double fel
	#endif
);
__device__ void vchar(double* pr, struct of_state* q, struct of_geom* geom, int dir, double* vmax, double* vmin);
__device__ void primtoflux(double* pr, struct of_state* q, int dir, struct of_geom* geom, double* flux, double* vmax, double* vmin
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ int fixup_cell(double pf[NDIM], double r, struct of_geom* geom);
__device__ void source(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double Dt, const  double* __restrict__ conn, struct of_state* q, double r
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ void mhd_calc(double* pr, int dir, struct of_state* q, double* mhd
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ void primtoflux_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR]);
__device__ void vchar_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double* vmax, double* vmin);
__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
);

/*Declarations of functions related to (M1) radiation inversion scheme*/
__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ int implicit_rad_solve_UMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ int implicit_rad_solve_PRAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ int implicit_rad_solve_URAD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ int implicit_rad_solve_EMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if(COOL_STOP)
	, double r
	#endif
);

__device__ void source_rad(double* ph, struct of_geom* geom, struct of_state* q, struct of_state_rad* q_rad, double* dU
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ void calc_Gcon(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double ucon_rad[NDIM], double ucov_rad[NDIM], double mhd_rad[NDIM][NDIM], double bsq
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double *source_photon
	#endif
	#if(COOL_STOP)
	, double r
	#endif
);
__device__ double calc_Tr(double* ph, double ucon[NDIM], double ucon_rad[NDIM], double ucov[NDIM]
	#if(P_NUM)
	, double *exp_xi
	#endif
);
__device__ double calc_Te(double* ph);
__device__ double calc_Ti(double* ph);
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int js, double* vmax, double* vmin, double dx
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ double calc_kappa_abs(double* ph, double bsq, double Tr
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_abs_ph(double* ph, double bsq, double Tr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_emmit(double* ph, double bsq, double Tr
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_emmit_ph(double* ph, double bsq, double Tr
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
);
__device__ double calc_kappa_es(double* ph
	#if(TWO_T)
	, double gamma_g
	#endif
);

__device__ void calc_kappa_new(double* ph, double bsq, double Tr, double Te, double* kappa_abs, double* kappa_emmit, double* kappa_es
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double* kappa_abs_ph
	, double* kappa_emmit_ph
	, double exp_xi
	#endif
);
#endif

__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq);
__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq);
__device__ double dWdvsq_calc2(double vsq, double rho, double p);
__device__ void validate_x2(double x[1], double x0[1]);

/*Declerations of functions for Utoprim_1dfix1*/
__device__ int Utoprim_1dfix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim, int full_entropy
	#if(TWO_T)
	, double fel
	#endif
);
__device__ int Utoprim_new_body3(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int general_newton_raphson3(double x[],  double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old, double tolerance
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int gnr2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double* f, double* df, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);

/*Declerations of functions related to (M1) radiation scheme*/
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim);
__device__ int Rtoprim_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim);
__device__ void mhd_calc_rad(double* pr, int dir, struct of_state_rad* q_rad, double* mhd_rad);
__device__ void ucon_calc_rad(double* pr, struct of_geom* geom, double* ucon_rad);
__device__ void get_state_rad(double* pr, struct of_geom* geom, struct of_state_rad* q_rad);

/*Matrix Inversion*/
__device__ int invert_matrix_4D(double Am[][NDIM], double Aminv[][NDIM]);
__device__ int LU_decompose(double A[][NDIM], int permute[]);
__device__ void LU_substitution(double A[][NDIM], double B[], int permute[]);
__device__ int invert_matrix_3D(double Am[][3], double Aminv[][3]);
__device__ int LU_decompose_3D(double A[][3], int permute[]);
__device__ void LU_substitution_3D(double A[][3], double B[], int permute[]);
__device__ int invert_matrix_5D(double Am[][5], double Aminv[][5]);
__device__ int LU_decompose_5D(double A[][5], int permute[]);
__device__ void LU_substitution_5D(double A[][5], double B[], int permute[]);
__device__ int invert_matrix_6D(double Am[][6], double Aminv[][6]);
__device__ int LU_decompose_6D(double A[][6], int permute[]);
__device__ void LU_substitution_6D(double A[][6], double B[], int permute[]);
__device__ int gamma_calc_rad(double* pr, struct of_geom* geom, double* gamma_rad);

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

__device__ void primtoflux_rad(double* pr, struct of_state_rad* q_rad, int dir, struct of_geom* geom, double* flux);
__device__ void misc_source(double *  ph, int icurr, int jcurr, struct of_geom *  geom, struct of_state *  q, double *  dU,	 double r, double Dt);
__device__ double calc_entropy(double* pr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ void inflow_check(double *  prim, int ii, int jj, int zz, int type, const  double* __restrict__ gcov1, const  double* __restrict__ gcoBS_2, const  double* __restrict__ gdet3);
__device__ double bsq_calc(double *  pr, struct of_geom *  geom);
__device__ double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double readImageDouble(int4 a);
__device__ void ucon_to_utcon(double *ucon, struct of_geom *geom, double *utcon);
__device__ void ut_calc_3vel(double *vcon, struct of_geom *geom, double *ut);
__device__ void para(double x1, double x2, double x3, double x4, double x5, double *lout, double *rout);

/*Advanced Riemann solver related functions*/
__device__ void calculate_flattener(double x1, double x2, double  x3, double  x4, double  x5, double* F);
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

// EOS functions
#if (EOS_LINEAR)
__device__ void interp_eostable_linear(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double* free, double* df_d, double* df_t, double* df_tt, double* df_dt, double* dpepdd, double* etaele);
#else
__device__ void interp_eostable(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double* free, double* df_d, double* df_t, double* df_tt, double* df_dt, double* dpepdd, double* etaele);
#endif
__device__ void eos_helm(const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double abar, double zbar, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2);
__device__ void eos_mode_rhou_pres(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* pres);
__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* pres, double* cs2);
__device__ void eos_mode_rhow_pres_dpdrho_dpde_d(const  double* __restrict__ gpu_eos_table, double den, double w_goal, double* pres, double* dpdrho, double* dpde_d);
__device__ void eos_mode_rhow_pres_u(const  double* __restrict__ gpu_eos_table, double den, double w_goal, double* pres, double* u);
__device__ void eos_mode_rhotemp_pres_min(const  double* __restrict__ gpu_eos_table, double den, double* pres);
__device__ void eos_mode_rhopres_u(const  double* __restrict__ gpu_eos_table, double den, double p_goal, double* u);
__device__ void eos_mode_rhos_upres(const  double* __restrict__ gpu_eos_table, double den, double entr_goal, double* pres, double* u, double* dpdrho, double* dudrho);
__device__ void eos_mode_rhou_entr(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* entr);
__device__ void eos_mode_rhou_temp(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* temp);

//Declare resistivity related functions
__device__ void primtoflux_res(double* pr, struct of_state_res* q_res, int dir, struct of_geom* geom, double* flux);
__device__ void econ_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* econ);
__device__ void bcon_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* bcon);
__device__ void mhd_calc_res(double* pr, int dir, struct of_geom* geom, struct of_state_res* q_res, double* mhd);
__device__ void source_res(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double* q, double Dt, const  double* __restrict__ conn_GPU, struct of_state_res* q_res, double r);
__device__ double bsq_calc_res(double* pr, struct of_geom* geom);
__device__ void get_state_res(double* pr, struct of_geom* geom, struct of_state_res* q_res);
__device__ void vchar_res(struct of_geom* geom, int js, double* vmax, double* vmin);
__device__ void vchar_res2(double* pr, struct of_state_res* q, struct of_geom* geom, int js, double* vmax, double* vmin);
__device__ double divE_calc(double* p, const  double* __restrict__ gdet, double _dx1, double _dx2, double _dx3, int ii, int jj, int zz);
__device__ void lower_3(double* ucon, double gcov[10], double* ucov);
__device__ double lvc4u(int i, int j, int k, int l);
__device__ double lvc3u(int i, int j, int k);

//Declare 2T related functions
__device__ double calc_delta(double* ph, double bsq);
__device__ void heating(double* ph, struct of_state* q);
__device__ double source_Coulomb(double* p);
__device__ double calc_gamma_gas_conserved(double* S, double rho);
__device__ double calc_gamma_gas_prim(double* pr);
__device__ double calc_gamma_gas_w(double* S, double rho, double w, double fel);
__device__ double set_S_w(double* S, double rho, double w, double fel);
__device__ void set_S_kappa(double rho, double K_atm, double* S, double fel);
__device__ double bessi0(double x);
__device__ double bessi1(double x);
__device__ double bessk0(double x);
__device__ double bessk1(double x);
__device__ double bessk(int n, double x);

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

struct of_state_res {
	double ucon[NDIM];
	double ucov[NDIM];
	double bcon[NDIM];
	double bcov[NDIM];
	double econ[NDIM];
	double ecov[NDIM];
};

struct of_state_rad {
	double ucon[NDIM];
	double ucov[NDIM];
};

#include <stdio.h>

__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
) {
	double error_t[2]; 
	int k;
	double delta_Ur, U_ft[NPR], pb_i[NPR], U_n_temp[NPR], U_i_temp[NPR];

	PLOOP{
		U_n_temp[k] = U_n[k];
		U_i_temp[k] = U_i[k];
	}

	//Initialize temporary variables
	PLOOP{
		dU[k] = 0.;
		U_ft[k] = U_i_temp[k];
		pb_i[k] = pb[k];
	}

	//Set initial values and error before attempting implicit solver
	implicit_rad_solve_init(pb_i, U_n_temp, U_i_temp, U_ft, geom, dU, Dt, error_t, cell_size, y_max
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
		#if(COOL_STOP)
		, r
		#endif
	);

	//If we've reached the tolerance level, exit immediately and update variables
	if (error_t[1] < 1.e-12) {
		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
	else {
		//Determine relative updae to radiation variables
		delta_Ur = (U_n[UU_RAD] - U_i[UU_RAD] + 1.e-150) / (fabs(U_i[UU_RAD]) + fabs(U_n[UU_RAD]));

		//Check if fluid is in extreme radiation subdominant regime
		//if (((U_n[UU_RAD] / U_n[UU]) < 1.e-5) || ((pb[UU_RAD] / pb[UU]) < 1.e-5) || (fabs(delta_Ur) < 1.e-5)) {
		if (0) {
		
		}
		else {
			//If error is below set margin, accept solution, otherwise try PMHD
			//if (error_t[1] > 1.e-9)implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0);

		//	if (error_t[1] > 1.e-9)implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0 
		//		#if(DOHELM)
		//		, gpu_eos_table
		//		#endif
		//	);

			//if (error_t[1] > 1.e-9)implicit_rad_solve_PRAD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0
			//	#if(DOHELM)
			//	, gpu_eos_table
			//	#endif
			//	#if(COOL_STOP)
			//	, r
			//	#endif
			//);

	

			if (error_t[1] > 1.e-9)implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
				#if(DOHELM)
				, gpu_eos_table
				#endif
				#if(COOL_STOP)
				, r
				#endif
			);

			//if (error_t[1] > 1.e-9)implicit_rad_solve_PRAD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0
			//	#if(DOHELM)
			//	, gpu_eos_table
			//	#endif
			//#if(COOL_STOP)
			//	, r
			//	#endif
			//);

			//if (error_t[1] > 1.e-9)implicit_rad_solve_EMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0
			//	#if(DOHELM)
			//	, gpu_eos_table
			//	#endif
			//	#if(COOL_STOP)
			//	, r
			//	#endif
			//);

			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t[1] > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 0, 0
			//	#if(DOHELM)
			//	, gpu_eos_table
			//	#endif
			//	#if(COOL_STOP)
			//	, r
			//	#endif
			//);

			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t[1] > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0
			//	#if(DOHELM)
			//	, gpu_eos_table
			//	#endif
			//);
			//If error is still below set margin, accept solution, otherwise try URAD
			//if (error_t[1] > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0);
			//if (error_t[1] > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 1, 0);

			//if (error_t[1] > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size, y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try UMHD
			//if (error_t[1] > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 0, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD
			//if (error_t[1] > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 0, 0);

			//If error is below set margin, accept solution, otherwise try PRAD with entropy
			//if (error_t[1] > 1.e-9) implicit_rad_solve_PMHD(pb_i, U_n_temp, U_i_temp, U_ft,pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 1, 0);

			/*/
			//If error is still below set margin, accept solution, otherwise try URAD with entropy
			//if (error_t[1] > 1.e-9) implicit_rad_solve_URAD(pb_i, U_n_temp, U_i_temp, U_ft,pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try URAD with entropy
			//if (error_t[1] > 1.e-9) implicit_rad_solve_PRAD(pb_i, U_n_temp, U_i_temp, U_ft,pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 1, 0);
			*/

			//If error is still below set margin, accept solution, otherwise try UMHD with entropy
			//if (error_t[1] > 1.e-9) implicit_rad_solve_UMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 1, 0);

			//If error is still below set margin, accept solution, otherwise try EMHD with entropy
			//if (error_t[1] > 1.e-9) implicit_rad_solve_EMHD(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, error_t, cell_size,y_max, 1, 0);
		}

		//As final resort attempt subcycling
		if (error_t[1] > 1.e-7) {
			//subcycle_rad_solve(pb_i, U_n_temp, U_i_temp, U_ft, pflag, pflag_rad, geom, dU, Dt, cell_size);
		}

		PLOOP{
			U_f[k] = U_ft[k];
			dU[k] = (U_ft[k] - U_i_temp[k]) / Dt;
			pb[k] = pb_i[k];
		}
	}
}
//Calculate initial error for source term and set initial guess values
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
) {
	double kappa_abs, kappa_es, tau, norm, bsq, Tr, Te, dK_dS, pb_old[NPR];
	int k, pflag=0, pflag_rad=0, do_entropy=0;
	struct of_state q;
	struct of_state_rad q_rad;
	#if(TWO_T)
	double gamma_g;// = calc_gamma_gas_prim(pb);
	#endif
	#if(P_NUM)
	double exp_xi;
	#endif

	//Calculate optical depth

	//Calculate opacities
	/*get_state(pb, geom, &q);
	get_state_rad(pb, geom, &q_rad);
	bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
	Tr = calc_Tr(pb, q.ucon, q_rad.ucon, q.ucov
	#if(P_NUM)
	, &exp_xi
	#endif
	);
	Te = calc_Te(pb);
	calc_kappa_new(pb, bsq, Tr, Te, &kappa_abs, NULL, &kappa_es
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(P_NUM)
		, NULL
		, NULL
		, NULL
		#endif
	);
	tau = (kappa_abs + kappa_es) * cell_size;

	//Store old values
	if (tau > 0.66)PLOOP pb_old[k] = pb[k];*/

	/*get_state(pb, geom, &q);
	bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
	get_state_rad(pb, geom, &q_rad);
	Tr = calc_Tr(pb, q.ucon, q_rad.ucon, q.ucov
		#if(P_NUM)
		, &exp_xi
		#endif
	);
	kappa_abs = calc_kappa_abs(pb, bsq, Tr
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, exp_xi
		#endif
	);
	kappa_es = calc_kappa_es(pb
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	tau = (kappa_abs + kappa_es) * cell_size;*/
	//tau = 0.0;

	//Set guess values for primitives after implicit step based on optical depth
	#if(NEWMAN)
	pflag = Utoprim_NM(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
	);
	#else
	pflag = Utoprim_2d(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, fel
		#endif
	);
	#endif
	#if(DO_FONT_FIX)
	if (pflag) {
		pflag = Utoprim_1dvsq2fix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
		);
		#if(!TWO_T)
		if (pflag) pflag = Utoprim_1dfix1(U_f, geom->gcov, geom->gcon, geom->g, pb, NEWT_TOL, BASIC, FULL_ENTROPY
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
			);
		#endif
		if (!pflag) do_entropy = 1;
	}
	#endif	 

	//Even if MHD inversion fails, use updated value of radiation variable as gues
	pflag_rad = Rtoprim(U_f, geom->gcov, geom->gcon, geom->g, pb, y_max, TYPE2);

	//Set electron entropy variables after inversion; Apply heating only if primary (energy based) inversion succeeds; Otherwise assume adiabatic evolution of electrons
	#if(TWO_T)
	if (pflag == 0) {
		U_i[ENTRE] = pb[ENTRE] * U_i[RHO];
		U_i[ENTRI] = pb[ENTRI] * U_i[RHO];

		double ue, ui;
		//Set for 2T fluid entropy of ions based on electron entropy
			#if(CONSTANTGAMMA)
			ue = pb[ENTRE] * pow(pb[RHO], GAMMA) / (GAMMA - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb[UU]) ue = (1.0 - FLOOR_ENTROPY) * pb[UU];
			if (ue < FLOOR_ENTROPY * pb[UU]) ue =FLOOR_ENTROPY * pb[UU];
			pb[ENTRE] = (GAMMA - 1.0) * ue * pow(pb[RHO], -GAMMA);
			ui = pb[UU] - ue;
			pb[ENTRI] = (GAMMA - 1.0) * ui * pow(pb[RHO], -GAMMA);
			#elif(FIXEDGAMMA)
			ue = pb[ENTRE] * pow(pb[RHO], GAMMAE) / (GAMMAE - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb[UU]) ue = (1.0 - FLOOR_ENTROPY) * pb[UU];
			if (ue < FLOOR_ENTROPY * pb[UU]) ue = FLOOR_ENTROPY * pb[UU];
			pb[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb[RHO], -GAMMAE);
			ui = pb[UU] - ue;
			pb[ENTRI] = (GAMMA - 1.0) * ui * pow(pb[RHO], -GAMMA);
			#elif(VARGAMMA)
			double Theta, gam, C;

			//Calculate ue
			Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO] * pb[ENTRE], 2. / 3.)) - 1.0));
			gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
			ue = Theta / (MU_E * MASS_RATIO) * pb[RHO] / (gam - 1.0);

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * pb[UU]) ue = (1.0 - FLOOR_ENTROPY) * pb[UU];
			if (ue < FLOOR_ENTROPY * pb[UU]) ue = FLOOR_ENTROPY * pb[UU];
			ui = pb[UU] - ue;

			//Set electron entropy
			C = ue / pb[RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			pb[ENTRE] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO];

			//Set ion entropy
			C = ui / pb[RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			pb[ENTRI] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb[RHO];
			#endif
		get_state(pb, geom, &q);
		U_i[ENTRE] = geom->g * pb[RHO] * q.ucon[0] * pb[ENTRE];
		U_i[ENTRI] = geom->g * pb[RHO] * q.ucon[0] * pb[ENTRI];
	}
	U_f[ENTRE] = U_i[ENTRE];
	U_f[ENTRI] = U_i[ENTRI];
	#endif

	//Reset guess for p
	//if (tau > 0.66) {
	//	//PLOOP if(k!=B1 && k!=B2 && k!=B3) pb[k] = pb_old[k];
	//}

	//Recompute T_t^mu for consistency
	U_f[RHO] = U_i[RHO];
	get_state(pb, geom, &q);
	#if(TWO_T)
	gamma_g = calc_gamma_gas_prim(pb);
	#endif
	mhd_calc(pb, 0, &q, &U_f[UU]
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	for (k = UU; k <= U3; k++)U_f[k] *= geom->g;
	U_f[UU] += U_f[RHO];

	//Recompute entropy for consistency
	#if(DOKTOT)
	U_f[KTOT] = U_f[RHO] * calc_entropy(pb
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	#endif

	//Recompute R_t^mu for consistency
	get_state_rad(pb, geom, &q_rad);
	mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
	for (k = UU_RAD; k <= U3_RAD; k++) U_f[k] *= geom->g;
	//if (pflag_rad) {
	//	for (k = UU_RAD; k <= U3_RAD; k++) U_i[k] = U_f[k];
	//	#if(P_NUM)
	//	U_i[PHOTON] = U_f[PHOTON];
	//	#endif
	//}

	//Recommpute photon number after inversion
	#if(P_NUM)
	U_f[PHOTON] = geom->g * pb[PHOTON] * q_rad.ucon[0];
	#endif

	//Calculate source term for U_i
	source_rad(pb, geom, &q, &q_rad, dU
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, r
		#endif
	);

	//Calculate iterated error at start of iteration
	norm = (fabs(U_i[UU]) + fabs(U_f[UU]) + fabs(Dt * dU[UU]));
	if (do_entropy == 0) error_t[0] = 0.25 * (fabs(U_f[UU] - U_i[UU] - Dt * dU[UU]) / norm);
	else {
		#if(FULL_ENTROPY)
		dK_dS = pb[RHO] / ((GAMMA - 1.) * pb[UU]);
		error_t[0] = 0.25 * (fabs(U_f[KTOT] - U_i[KTOT] - Dt * dU[KTOT])) / (norm * dK_dS);
		#else
		dK_dS = (GAMMA - 1.) / pow(pb[RHO], GAMMA - 1.0);
		error_t[0] = 0.25 * (fabs((U_f[KTOT] - U_i[KTOT] - Dt * dU[KTOT]))) / (norm * dK_dS);
		#endif
	}
	#if(TWO_T)
		#if(CONSTANTGAMMA)
		dK_dS = (GAMMA - 1.) / pow(pb[RHO], GAMMA - 1.0);
		#elif(FIXEDGAMMA)
		dK_dS = (GAMMAE - 1.) / pow(pb[RHO], GAMMAE - 1.0);
		#elif(VARGAMMA)
		//Notes
		//Q = P * uu * dS;
		//Q = P * uu * 1 / kappa * dkappa;
		//Q = P / rho  * 1 / kappa * d(rho * uu * kappa);

		//For old entropy
		//Q = P * uu * rho ^ gamma / P * dkappa / (gamma - 1);
		//Q = rho ^ (gamma - 1.0) * d(kappa * rho * uu) / (gamma - 1);
		double Theta_e;
			//For variable entropy
			#if(FULL_ENTROPY)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO] * exp(pb[ENTRE]), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb[RHO] * pb[ENTRE], 2. / 3.)) - 1.0);
			dK_dS = (pb[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			#endif
		#endif
	norm = (fabs(U_i[ENTRE]) + fabs(U_f[ENTRE]) + fabs(Dt * dU[ENTRE]));
	error_t[0] += 0.25 * (fabs(U_f[ENTRE] - U_i[ENTRE] - Dt * dU[ENTRE]) / ( norm));
	#endif
	#if(P_NUM)
	//norm = (fabs(U_i[PHOTON]) + fabs(U_f[PHOTON]) + fabs(Dt * dU[PHOTON]));
	//if (pflag_rad == 0)error_t[0] += 0.25 * (fabs(U_f[PHOTON] - U_i[PHOTON] - Dt * dU[PHOTON]) / norm);
	#endif
	norm = (fabs(sqrt(geom->gcon[4]) * U_i[U1]) + fabs(U_f[U1]) + fabs(Dt * dU[U1]));
	norm += (fabs(sqrt(geom->gcon[7]) * U_i[U2]) + fabs(U_f[U2]) + fabs(Dt * dU[U2]));
	norm += (fabs(sqrt(geom->gcon[9]) * U_i[U3]) + fabs(U_f[U3]) + fabs(Dt * dU[U3]));
	error_t[0] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1] - U_i[U1] - Dt * dU[U1]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2] - U_i[U2] - Dt * dU[U2]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3] - U_i[U3] - Dt * dU[U3]) / norm);
	
	//Set total error to iterated error
	error_t[1] = error_t[0];

	//Calculate total error at start of iteration
	norm = (fabs(U_i[UU_RAD]) + fabs(U_f[UU_RAD]) + fabs(Dt * dU[UU_RAD]));
	if (pflag_rad == 0)error_t[1] += 0.25 * (fabs(U_f[UU_RAD] - U_i[UU_RAD] - Dt * dU[UU]) / norm);
	norm = (fabs(sqrt(geom->gcon[4]) * U_i[U1_RAD]) + fabs(U_f[U1_RAD]) + fabs(Dt * dU[U1_RAD]));
	norm += (fabs(sqrt(geom->gcon[7]) * U_i[U2_RAD]) + fabs(U_f[U2_RAD]) + fabs(Dt * dU[U2_RAD]));
	norm += (fabs(sqrt(geom->gcon[9]) * U_i[U3_RAD]) + fabs(U_f[U3_RAD]) + fabs(Dt * dU[U3_RAD]));
	error_t[1] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_f[U1_RAD] - U_i[U1_RAD] - Dt * dU[U1_RAD]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_f[U2_RAD] - U_i[U2_RAD] - Dt * dU[U2_RAD]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_f[U3_RAD] - U_i[U3_RAD] - Dt * dU[U3_RAD]) / norm);
}

__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged 
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
) {
	double U_new[NPR], U_old[NPR], U_prev[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb,  dEdpb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[5*2], offset = 1.e-8;
	double T_GAS, dK_dS, norm, D;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase2 = 0;
	#if(TWO_T)
	int flag_floor_kappa;
	double gamma_g, ue, ui;
	#endif

	//Set error to previous value
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

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
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			#else
			T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
			#endif			
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP pb_new[k] = pb_old[k];
				if (i == UU) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU]);
					pb_new[i] = pb_old[i] + dpb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[ENTRE]);
					pb_new[ENTRE] = pb_old[ENTRE] + dpb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dpb;
				}
				#endif
				else {
					dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					pb_new[i] = pb_old[i] + dpb;
				}

				// Compute (new conserved vars) S u^t and T^+mu from gas P_i+1
				get_state(pb_new, geom, &q);
				pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0]; //Obtain rho0 = U_1 / u^t from newly updates P_i+1
				U_new[RHO] = U_i[RHO];
				#if(TWO_T)
					//Set for 2T fluid entropy of ions based on electron entropy
					#if(CONSTANTGAMMA)
					ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMA) / (GAMMA - 1.0);
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					pb_new[ENTRE] = (GAMMA - 1.0) * ue * pow(pb_new[RHO], -GAMMA);
					ui = pb_new[UU] - ue;
					pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
					#elif(FIXEDGAMMA)
					ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
					ui = pb_new[UU] - ue;
					pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
					#elif(VARGAMMA)
					double Theta, gam, C;
					
					//Calculate ue
					Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0));
					gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
					ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

					//Check limits
					if (ue > (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - 0.5 * FLOOR_ENTROPY) * pb_new[UU];
					if (ue < 0.5 * FLOOR_ENTROPY * pb_new[UU]) ue = 0.5 * FLOOR_ENTROPY * pb_new[UU];
					ui = pb_new[UU] - ue;

					//Set electron entropy
					C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
					pb_new[ENTRE] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO];

					//Set ion entropy
					C = ui / pb_new[RHO] * MU_I;
					Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
					pb_new[ENTRI] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO];
					#endif
				U_new[ENTRE] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRE];
				U_new[ENTRI] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRI];
				gamma_g = calc_gamma_gas_prim(pb_new);
				#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] = U_new[UU] + U_new[RHO];

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
				U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
				U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
				U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

				//Invert radiation variables
				Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif

				//Calculate radiative (including coulomb) source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
				);

				//Calculate Jacobian
				for (k = U1; k <= U3; k++) {
					E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
					dEdpb_inv[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dpb;
				}
				#if(TWO_T)
				E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
				dEdpb_inv[4][i - UU] = (E_new[4] - E_old[4]) / dpb;
				#endif
				#if(P_NUM)
				E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
				dEdpb_inv[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dpb;
				#endif
				if (do_entropy == 1) {
					#if(FULL_ENTROPY)
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					#else
					T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
					#endif
					E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
				}
				else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
				dEdpb_inv[0][i - UU] = (E_new[0] - E_old[0]) / dpb;
			}

			//Invert Jacobian
			#if(P_NUM && TWO_T)
			flag = invert_matrix_6D(dEdpb_inv, dEdpb_inv);
			#elif(P_NUM || TWO_T)
			flag = invert_matrix_5D(dEdpb_inv, dEdpb_inv);
			#else
			flag = invert_matrix_4D(dEdpb_inv, dEdpb_inv);
			#endif

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;
		
		//Set primitive variables before Newton step
		PLOOP pb_new[k] = pb_old[k];

		/* Make the newton step: */
		if (do_staged == 0) {
			D = 1.0;
			for (k = 0; k < 4; k++) {
				dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
					#if(TWO_T)
					 + E_old[4] * dEdpb_inv[k][4]
					#endif
					#if(P_NUM)
					+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
					#endif
					);
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
					dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			if (n_iter / 4 == 1) {
				for (k = 0; k < 4; k++) { //energy only step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0]
						#if(TWO_T)
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
			else {
				for (k = 0; k < 4; k++) { //full 4d step
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU] = pb_old[k + UU] + dpb;
				}
			}
		}
		#if(TWO_T)
		dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
			#if(P_NUM)
			+ E_old[4 + P_NUM] * dEdpb_inv[4][4 + P_NUM]
			#endif	
			);
			pb_new[ENTRE] = pb_old[ENTRE] + dpb;
		#endif
		#if(P_NUM)
		dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4] * dEdpb_inv[4 + TWO_T][4]
			#if(TWO_T)
			+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
			#endif			
			);
		U_new[PHOTON] = U_old[PHOTON] + dpb;
		#endif

		//Make sure that internal energy stays positive
		if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);
		
		//Make sure that electron entropy stays positive
		#if(TWO_T)
		if (pb_new[ENTRE] < 0.0) pb_new[ENTRE] = 0.5 * fabs(pb_new[ENTRE]);
		#endif
		
		//Make sure that photon number stays positive
		#if(P_NUM)
		if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
		#endif

		//Obtain new conserved quantaties from MHD variables
		get_state(pb_new, geom, &q);
		U_new[RHO] = U_i[RHO];
		pb_new[RHO] = (U_i[RHO] / geom->g) / q.ucon[0];
		#if(TWO_T)
			flag_floor_kappa = 0;
			#if(CONSTANTGAMMA)
			ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMA) / (GAMMA - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				flag_floor_kappa = 1;
			}
			pb_new[ENTRE] = (GAMMA - 1.0) * ue * pow(pb_new[RHO], -GAMMA);
			ui = pb_new[UU] - ue;
			pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
			#elif(FIXEDGAMMA)
			ue = pb_new[ENTRE] * pow(pb_new[RHO], GAMMAE) / (GAMMAE - 1.0);
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
			if (ue < FLOOR_ENTROPY * pb_new[UU]) ue = FLOOR_ENTROPY * pb_new[UU];
			pb_new[ENTRE] = (GAMMAE - 1.0) * ue * pow(pb_new[RHO], -GAMMAE);
			ui = pb_new[UU] - ue;
			pb_new[ENTRI] = (GAMMA - 1.0) * ui * pow(pb_new[RHO], -GAMMA);
			#elif(VARGAMMA)
			double Theta, gam, C;

			//Calculate ue
			Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0));
			gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
			ue = Theta / (MU_E * MASS_RATIO) * pb_new[RHO] / (gam - 1.0);

			//Check limits
			if (ue > (1.0 - FLOOR_ENTROPY) * pb_new[UU]) {
				ue = (1.0 - FLOOR_ENTROPY) * pb_new[UU];
				flag_floor_kappa = 1;
			}
			if (ue < FLOOR_ENTROPY * pb_new[UU]) {
				ue = FLOOR_ENTROPY * pb_new[UU];
				flag_floor_kappa = 1;
			}
			ui = pb_new[UU] - ue;
		
			//Set electron entropy
			C = ue / pb_new[RHO] * MU_E * MASS_RATIO;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			pb_new[ENTRE] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO];

			//Set ion entropy
			C = ui / pb_new[RHO] * MU_I;
			Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			pb_new[ENTRI] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pb_new[RHO];
			#endif
		U_new[ENTRE] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRE];
		U_new[ENTRI] = geom->g * pb_new[RHO] * q.ucon[0] * pb_new[ENTRI];
		gamma_g = calc_gamma_gas_prim(pb_new);
		#endif
		mhd_calc(pb_new, 0, &q, &U_new[UU]
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
		U_new[UU] = U_new[UU] + U_new[RHO];

		//Recalculate gas entropy for consistency
		#if(DOKTOT)
		U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		//Derive new conserved quantaties for radiation variables
		U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
		U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
		U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
		U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

		//Get new radiation primitives using TYPE2 limiter
		flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);
		if (flag_rad) {
			for (k = UU_RAD; k <= U3_RAD; k++) U_prev[k] = U_new[k];
			#if(P_NUM)
			U_prev[PHOTON] = U_new[PHOTON];
			#endif
		}

		//Recompute R_t^mu for consistency
		get_state_rad(pb_new, geom, &q_rad);
		mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
		for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;
		
		//Recompute photon number
		#if(P_NUM)
		U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
		#endif

		//Get radiative source term
		source_rad(pb_new, geom, &q, &q_rad, dU_new
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
			#if(COOL_STOP)
			, r
			#endif
		);

		//Calculate iterated error
		norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
		norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
		norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
		error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
		error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
		else {
			#if(FULL_ENTROPY)
			dK_dS = pb_new[RHO] / ((GAMMA - 1.) * pb_new[UU]);
			error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm * dK_dS);
			#else
			dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
			error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
			#endif
		}
		#if(TWO_T)
			#if(CONSTANTGAMMA)
			dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
			#elif(FIXEDGAMMA)
			dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
			#elif(VARGAMMA)
			double Theta_e;
			//For variable entropy
				#if(FULL_ENTROPY)
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
				dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
				#else
				Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
				dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
				#endif
			#endif
		norm = (fabs(U_i[ENTRE]) + fabs(U_new[ENTRE]) + fabs(Dt * dU_new[ENTRE]));
		if(flag_floor_kappa==0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (norm));
		#endif
		#if(P_NUM)
		//norm =  (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
		//if (flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
		#endif

		//Set correct offset for Jacobian for next iteration
		if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10;
		else offset = 1.e-8;

		//Set total error to iterated error
		error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		if (flag_rad == 0 && do_entropy == 0) error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
		norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
		norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
		norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
		error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);


		double bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];

		//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
		if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-10 && bsq / pb_new[RHO] <= 1.0) || (n_iter >= 20) || (fabs(error_new[n_iter % 5 + 5]) <= 1.e-8 && bsq / pb_new[RHO]>1.0)) {
			keep_iterating = 0;
		}

		//If residual drops below bound exit
		//double residual = 0.;
		//for (k = 0; k < NPR; k++)residual += fabs(pb_new[k]/pb_old[k]-1.0);
		//if (residual<1.0e-15) {
			//keep_iterating = 0;
		//}

		//If total error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
		//	keep_iterating = 0;
		}

		//If iterated error increasing stop iterating
		if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5] + error_new[(n_iter - 3) % 5] + error_new[(n_iter - 2) % 5]) < (error_new[(n_iter - 1) % 5] + error_new[(n_iter - 0) % 5]))) {
			//keep_iterating = 0;
		}

		//If total error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
			count_increase++;
			if (count_increase >= 5) keep_iterating = 0;
		}

		//If iterated error increased more than 4 times stop iterating
		if ((n_iter > 4) && (error_new[(n_iter - 1) % 5] < error_new[(n_iter) % 5])) {
			count_increase2++;
			if (count_increase2 >= 5) keep_iterating = 0;
		}

		//Reset variables if Newton step succesfull
		if (keep_iterating) {
			for (k = 0; k < NPR; k++) {
				U_old[k] = U_new[k];
				pb_old[k] = pb_new[k];
				dU_old[k] = dU_new[k];
			}
		}

		//If error decreased compared to start value, update variables
		if (fabs(error_new[n_iter % 5]) < error_t[0] && fabs(error_new[n_iter % 5 + 5])<0.01) {
			error_t[0] = error_new[n_iter % 5];
			error_t[1] = error_new[n_iter % 5 + 5];
			for (k = 0; k < NPR; k++) {
				pb[k] = pb_new[k];
				U_f[k] = U_new[k];
				dU[k] = dU_new[k];
			}
			if (flag_rad) {
				Rtoprim(U_prev, geom->gcov, geom->gcon, geom->g, pb, y_max, BASIC);

				//Recompute R_t^mu for consistency
				get_state_rad(pb, geom, &q_rad);
				mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_f[k] *= geom->g;

				//Recompute photon number
				#if(P_NUM)
				U_f[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif
			}
		}

		n_iter++;
	}

	return(0);
}

// This method iterates T^t_mu
__device__ int implicit_rad_solve_UMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged 
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0;
	#if(TWO_T)
	double gamma_g;
	#endif

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			#else
			T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
			#endif		
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]);
					U_new[i] = U_old[i] + dUb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					U_new[i] = U_old[i] + dUb;
				}

				//Set entropy based on values in previous iteration
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				//Invert gas conserved to gas primitives
				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);
				#if(DO_FONT_FIX)
				if (flag && (n_iter_jacob > 1)) {
					//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2, FULL_ENTROPY
					//#if (DOHELM)
					//, gpu_eos_table
					//#endif
					//#if(TWO_T)
					//, 0.0
					//#endif
					//);
				}
				#endif

				if (flag == 0) {
					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q);
					#if(TWO_T)
					gamma_g = calc_gamma_gas_prim(pb_new);
					#endif
					mhd_calc(pb_new, 0, &q, &U_new[UU]
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif

					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Set radiation conserved quantities
					U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
					U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
					U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
					U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

					//Invert radiation conserved variables to primitives
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					#if(P_NUM)
					U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					#if(TWO_T)
					E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(FULL_ENTROPY)
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						#else
						T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
						#endif		
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;
				}
			}

			//Invert Jacobian
			if(flag==0){
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif	
			}
			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
						);
					U_new[k + UU] = U_old[k + UU] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
			#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
			#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Invert gas conserved quantities to primitives
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);
			#if(DO_FONT_FIX)
			if (flag && (n_iter_fail > 1)) {
				//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2, FULL_ENTROPY
				//#if (DOHELM)
				//, gpu_eos_table
				//#endif
				//#if(TWO_T)
				//, 0.0
				//#endif
				//);
			}
			#endif

			if (flag == 0) {
				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

				//Make sure that the photon number stays positive
				#if(P_NUM)
				if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q);
				#if(TWO_T)
				gamma_g = calc_gamma_gas_prim(pb_new);
				#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Derive new conserved quantaties for MHD variables
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

				//Recompute photon number
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
				);

				//Calculate iterated error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				//norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				//norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				//norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(FULL_ENTROPY)
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (dK_dS * norm);
					#endif
				}
				#if(TWO_T)
					#if(CONSTANTGAMMA)
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					#elif(FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
						dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
#					endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < 1.e-9) offset = 1.e-10;
				else offset = 1.e-8;

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				//norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				//norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				//norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];
					for (k = 0; k < NPR; k++) {
						pb[k] = pb_new[k];
						U_f[k] = U_new[k];
						dU[k] = dU_new[k];
					}
					/*if (flag_rad) {
						Rtoprim(U_prev, geom->gcov, geom->gcon, geom->g, pb, y_max, BASIC);

						//Recompute R_t^mu for consistency
						get_state_rad(pb, geom, &q_rad);
						mhd_calc_rad(pb, 0, &q_rad, &U_f[UU_RAD]);
						for (k = UU_RAD; k <= U3_RAD; k++)U_f[k] *= geom->g;

						//Recompute photon number
						#if(P_NUM)
						U_f[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
						#endif
					}*/
				}
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates Su^t and T^t_i
__device__ int implicit_rad_solve_EMHD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM],  error_new[10], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase_gas = 0;
	#if(TWO_T)
	double gamma_g;
	#endif

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb_old[k];
		U_old[k] = U_f[k];
		U_new[k] = U_old[k];
		dU_old[k] = dU[k];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1; k <= U3; k++) E_old[k - UU] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			#else
			T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
			#endif		
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU] - U_i[UU] - Dt * dU_old[UU]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU; i <= U3 + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[KTOT]);
					U_new[KTOT] = U_old[KTOT] + dUb;
				}
				#if(TWO_T)
				else if (i == U3 + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3 + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU]) * sqrt(geom->gcov[4 * (i == U1) + 7 * (i == U2) + 9 * (i == U3)]);
					U_new[i] = U_old[i] + dUb;
				}

				//Invert conserved MHD quantities using entropy based methods
				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC, FULL_ENTROPY		
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);

				if (flag == 0) {
					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q);
					#if(TWO_T)
					gamma_g = calc_gamma_gas_prim(pb_new);
					#endif
					mhd_calc(pb_new, 0, &q, &U_new[UU]
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif

	
					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Set radiation conserved quantities
					U_new[UU_RAD] = U_i[UU_RAD] - (U_new[UU] - U_i[UU]);
					U_new[U1_RAD] = U_i[U1_RAD] - (U_new[U1] - U_i[U1]);
					U_new[U2_RAD] = U_i[U2_RAD] - (U_new[U2] - U_i[U2]);
					U_new[U3_RAD] = U_i[U3_RAD] - (U_new[U3] - U_i[U3]);

					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					#if(P_NUM)
					U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1; k <= U3; k++) {
						E_new[k - UU] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU][i - UU] = (E_new[k - UU] - E_old[k - UU]) / dUb;
					}
					#if(TWO_T)
					E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(FULL_ENTROPY)
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						#else
						T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
						#endif		
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU] - U_i[UU] - Dt * dU_new[UU]);
					dEdUb[0][i - UU] = (E_new[0] - E_old[0]) / dUb;
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
					);
					if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
					else U_new[k + UU] = U_old[k + UU] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
						);
						if (k == 0) U_new[KTOT] = U_old[KTOT] + dUb;
						else U_new[k + UU] = U_old[k + UU] + dUb;
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Invert conserved MHD quantities using entropy based methods
			flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC, FULL_ENTROPY
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);

			if (flag == 0) {
				//Make sure that internal energy stays positive
				if (pb_new[UU] < 0.0) pb_new[UU] = 0.5 * fabs(pb_new[UU]);

				//Make sure that the photon number stays positive
				#if(P_NUM)
				if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q);
				#if(TWO_T)
				gamma_g = calc_gamma_gas_prim(pb_new);
					#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Derive new conserved quantaties for MHD variables
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

				//Compute photon number
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
				);

				//Calculate iterated error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(FULL_ENTROPY)
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					error_new[n_iter % 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					double dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					error_new[n_iter % 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (dK_dS * norm);
					#endif
				}
				#if(TWO_T)
					#if(CONSTANTGAMMA)
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					#elif(FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
						dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				//norm =  (fabs(U_i[ENTRE]) + fabs(U_new[ENTRE]) + fabs(Dt * dU_new[ENTRE]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				//error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error	
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				if (do_entropy == 0 && flag_rad == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					count_increase++;
					if (count_increase >= 5) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];
					for (k = 0; k < NPR; k++) {
						pb[k] = pb_new[k];
						U_f[k] = U_new[k];
						dU[k] = dU_new[k];
					}
				}
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}

	return(0);
}

// This method iterates R^t_mu
__device__ int implicit_rad_solve_URAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged 
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dUb, dEdUb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdUb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, n_iter_jacob, flag = 0, flag_rad = 0, count_increase = 0, count_increase_gas = 0;
	#if(TWO_T)
	double gamma_g;
	#endif

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
		dU_new[k] = dU[k];
	}

	//Calculate iterated error
	norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
	norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
	norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
	error_t[0] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
	if (do_entropy == 0) {
		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		error_t[0] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
	}
	#if(TWO_T)
		norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
		#if(CONSTANTGAMMA)
		dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
		#elif(FIXEDGAMMA)
		dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
		#elif(VARGAMMA)
		double Theta_e;
		//For variable entropy
			#if(FULL_ENTROPY)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
			dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			#endif
		#endif
		error_t[0] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
	#endif
	#if(P_NUM)
	norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
	error_t[0] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
	#endif

	//Set correct offset for Jacobian for next iteration
	if (error_t[0] < pow(10., -9.))offset = pow(10., -10.);
	else offset = pow(10., -8.);

	//Set total to iterated error
	error_t[1] = error_t[0];

	//Calculate total error
	norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
	if (do_entropy == 0) error_t[1] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
	else {
		#if(FULL_ENTROPY)
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		error_t[1] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
		#else
		double dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
		error_t[1] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (dK_dS * norm);
		#endif
	}
	norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
	norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
	norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
	error_t[1] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Calculate reference error
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) * pb_old[UU] / pb_old[RHO];
			#else
			T_GAS = pow(pb_old[RHO], GAMMA - 1.0) / (GAMMA - 1.);
			#endif		
			E_old[0] = T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU_RAD; i <= U3_RAD + TWO_T + P_NUM; i++) {
				PLOOP U_new[k] = U_old[k];
				if (i == UU_RAD) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD] + U_old[UU]);
					U_new[i] = U_old[i] + dUb;
				}
				#if(TWO_T)
				else if (i == U3_RAD + TWO_T) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
					U_new[ENTRE] = U_old[ENTRE] + dUb;
				}
				#endif
				#if(P_NUM)
				else if (i == U3_RAD + TWO_T + P_NUM) {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
					U_new[PHOTON] = U_old[PHOTON] + dUb;
				}
				#endif
				else {
					dUb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[UU_RAD]) * sqrt(geom->gcov[4 * (i == U1_RAD) + 7 * (i == U2_RAD) + 9 * (i == U3_RAD)]);
					U_new[i] = U_old[i] + dUb;
				}

				U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
				U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
				U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
				U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
				U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

				tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
				flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC			
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);
				#if(DO_FONT_FIX)
				//if (flag && (n_iter_jacob > -1)) {
				//	flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2, FULL_ENTROPY			
				//		#if (DOHELM)
				//		, gpu_eos_table
				//		#endif
				//		#if(TWO_T)
				//		, fel
				//		#endif
				//	);
				//}
				#endif

				if (flag == 0) {
					Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

					//Recompute T_t^mu for consistency
					U_new[RHO] = U_i[RHO];
					get_state(pb_new, geom, &q);
					#if(TWO_T)
					gamma_g = calc_gamma_gas_prim(pb_new);
					#endif
					mhd_calc(pb_new, 0, &q, &U_new[UU]
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
					U_new[UU] += U_new[RHO];

					//Electron and ion entropies
					#if(TWO_T)
					U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
					U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
					#endif
	
					//Recalculate gas entropy for consistency
					#if(DOKTOT)
					U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
					);
					#endif

					//Recompute R_t^mu for consistency
					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

					//Recompute photon number
					#if(P_NUM)
					U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
					#endif

					//Calculate source term using new variables
					source_rad(pb_new, geom, &q, &q_rad, dU_new
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, gamma_g
						#endif
						#if(COOL_STOP)
						, r
						#endif
					);

					//Calculate source function and jacobian
					for (k = U1_RAD; k <= U3_RAD; k++) {
						E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
						dEdUb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dUb;
					}
					#if(TWO_T)
					E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
					dEdUb[4][i - UU_RAD] = (E_new[4] - E_old[4]) / dUb;
					#endif
					#if(P_NUM)
					E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
					dEdUb[4 + TWO_T][i - UU_RAD] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dUb;
					#endif
					if (do_entropy == 1) {
						#if(FULL_ENTROPY)
						T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
						#else
						T_GAS = pow(pb_new[RHO], GAMMA - 1.0) / (GAMMA - 1.);
						#endif		
						E_new[0] = T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
					}
					else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
					dEdUb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dUb;
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdUb, dEdUb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdUb, dEdUb_inv);
				#else
				flag = invert_matrix_4D(dEdUb, dEdUb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1.0 / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdUb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
						#endif
						);
					U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if ((n_iter / 4) == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dUb = -D * (E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
				if ((n_iter / 4) == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dUb = -D * (E_old[0] * dEdUb_inv[k][0] + E_old[1] * dEdUb_inv[k][1] + E_old[2] * dEdUb_inv[k][2] + E_old[3] * dEdUb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdUb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdUb_inv[k][4 + TWO_T]
							#endif
							);
						U_new[k + UU_RAD] = U_old[k + UU_RAD] + dUb;
					}
				}
			}

			#if(TWO_T)
			dUb = -D * (E_old[0] * dEdUb_inv[4][0] + E_old[1] * dEdUb_inv[4][1] + E_old[2] * dEdUb_inv[4][2] + E_old[3] * dEdUb_inv[4][3] + E_old[4] * dEdUb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdUb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dUb;
			#endif
			#if(P_NUM)
			dUb = -D * (E_old[0] * dEdUb_inv[4 + TWO_T][0] + E_old[1] * dEdUb_inv[4 + TWO_T][1] + E_old[2] * dEdUb_inv[4 + TWO_T][2] + E_old[3] * dEdUb_inv[4 + TWO_T][3] + E_old[4] * dEdUb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdUb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dUb;
			#endif

			//Derive new conserved quantaties for MHD variables
			U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
			U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
			U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
			U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

			//Estimate conserved entropy using prior primitives
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);
			#if(DO_FONT_FIX)
			//if (flag && (n_iter_fail > -1)) {
			//	flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2, FULL_ENTROPY
				//#if (DOHELM)
				//, gpu_eos_table
				//#endif
				//#if(TWO_T)
				//, 0.0
				//#endif
			//	);
			//}
			#endif

			if (flag == 0) {
				//Make sure that the photon number stays positive
				#if(P_NUM)
				if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);
				#endif

				//Get new radiation primitives using TYPE2 limiter
				flag_rad = Rtoprim(U_new, geom->gcov, geom->gcon, geom->g, pb_new, y_max, TYPE2);

				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q);
				#if(TWO_T)
				gamma_g = calc_gamma_gas_prim(pb_new);
				#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				for (k = UU; k <= U3; k++)U_new[k] *= geom->g;
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif

				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Recompute R_t^mu for consistency
				get_state_rad(pb_new, geom, &q_rad);
				mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
				for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

				//Recompute photon number
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
				);

				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				if (do_entropy == 0) {
					norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
					error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				}
				#if(TWO_T)
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				#if(CONSTANTGAMMA)
				dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
				#elif(FIXEDGAMMA)
				dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
				#elif(VARGAMMA)
				double Theta_e;
				//For variable entropy
					#if(FULL_ENTROPY)
					Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
					dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
					#else
					Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
					dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
					#endif
				#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 0) error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(FULL_ENTROPY)
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					error_new[n_iter % 5 + 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					error_new[n_iter % 5 + 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (dK_dS * norm);
					#endif
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
				//	keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					//count_increase++;
					//if (count_increase >= 5) keep_iterating = 0;
				}

				//If gas negative more than 2 times stop iterating
				if (pb_new[UU] < 0.) {
					count_increase_gas++;
					//if (count_increase > 2) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];
					for (k = 0; k < NPR; k++) {
						pb[k] = pb_new[k];
						U_f[k] = U_new[k];
						dU[k] = dU_new[k];
					}
				}
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}
	return(0);
}

// This method iterates E_RAD an U_rad
__device__ int implicit_rad_solve_PRAD(double pb[NPR], double U_n[NPR], double U_i[NPR], double U_f[NPR], int* pflag, int* pflag_rad, struct of_geom* geom, double dU[NPR], double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged 
#if(DOHELM)
, const double* __restrict__ gpu_eos_table
#endif
#if(COOL_STOP)
, double r
#endif
) {
	double U_new[NPR], U_old[NPR], pb_new[NPR], pb_old[NPR], dU_new[NPR], dU_old[NPR], E_old[NPR], E_new[NPR], dpb, dEdpb[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], dEdpb_inv[4 + TWO_T + P_NUM][4 + TWO_T + P_NUM], error_new[10], offset = pow(10., -8.);
	double T_GAS, norm, norm_S, D, tol, dK_dS;
	struct of_state q;
	struct of_state_rad q_rad;
	int i, k, n_iter = 0, n_iter_fail = 0, keep_iterating = 1, flag, n_iter_jacob, count_increase = 0, count_increase_gas = 0;
	#if(TWO_T)
	double gamma_g;
	#endif

	//Set variables to previously iterated values
	for (k = 0; k < NPR; k++) {
		pb_old[k] = pb[k];
		pb_new[k] = pb[k];
		U_old[k] = U_f[k];
		U_new[k] = U_f[k];
		dU_old[k] = dU[k];
		dU_new[k] = dU[k];
	}

	//Calculate iterated error
	norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
	norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
	norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
	error_t[0] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
	error_t[0] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
	if (do_entropy == 0) {
		norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
		error_t[0] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
	}
	#if(TWO_T)
		#if(CONSTANTGAMMA)
		dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
		#elif(FIXEDGAMMA)
		dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
		#elif(VARGAMMA)
		double Theta_e;
		//For variable entropy
			#if(FULL_ENTROPY)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
			dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			#endif
		#endif
		error_t[0] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
	#endif
	#if(P_NUM)
	norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
	error_t[0] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
	#endif

	//Set correct offset for Jacobian for next iteration
	if (error_t[0] < 1.e-9)offset = 1.e-10;
	else offset = 1.e-8;

	//Set total to iterated error
	error_t[1] = error_t[0];

	//Calculate total error
	norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
	if (do_entropy == 0) error_t[1] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
	else {
		#if(FULL_ENTROPY)
		T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
		error_t[1] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
		#else
		double dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
		error_t[1] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]) / dK_dS)) / (norm);
		#endif
	}
	norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
	norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
	norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
	error_t[1] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
	error_t[1] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

	//Set error to 0
	for (k = 0; k < 5; k++) {
		error_new[k] = error_t[0];
		error_new[k + 5] = error_t[1];
	}

	/* Start the Newton-Raphson iterations : */
	while (keep_iterating) {
		//Set reference error at start
		for (k = U1_RAD; k <= U3_RAD; k++) E_old[k - UU_RAD] = (U_old[k] - U_i[k] - Dt * dU_old[k]);
		#if(TWO_T)
		E_old[4] = (U_old[ENTRE] - U_i[ENTRE] - Dt * dU_old[ENTRE]);
		#endif
		#if(P_NUM)
		E_old[4 + TWO_T] = (U_old[PHOTON] - U_i[PHOTON] - Dt * dU_old[PHOTON]);
		#endif
		if (do_entropy == 1) {
			#if(FULL_ENTROPY)
			T_GAS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA);
			#else
			T_GAS = (GAMMA - 1.) / pow(pb_old[RHO], GAMMA - 1.0);
			#endif
			E_old[0] = 1.0 / T_GAS * (U_old[KTOT] - U_i[KTOT] - Dt * dU_old[KTOT]);
		}
		else E_old[0] = (U_old[UU_RAD] - U_i[UU_RAD] - Dt * dU_old[UU_RAD]);

		//Calculate jacobian dEdpb
		n_iter_jacob = 0;
		do {
			flag = 0;
			for (i = UU_RAD; i <= U3_RAD + TWO_T + P_NUM; i++) {
				if(flag == 0){
					PLOOP{
						pb_new[k] = pb_old[k];
						U_new[k] = U_old[k];
					}
					if (i == UU_RAD) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (pb_old[UU_RAD]);
						pb_new[i] = pb_old[i] + dpb;
					}
					#if(TWO_T)
					else if (i == U3_RAD + TWO_T) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[ENTRE]);
						U_new[ENTRE] = U_old[ENTRE] + dpb;
					}
					#endif
					#if(P_NUM)
					else if (i == U3_RAD + TWO_T + P_NUM) {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) * (U_old[PHOTON]);
						U_new[PHOTON] = U_old[PHOTON] + dpb;
					}
					#endif
					else {
						dpb = offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) / sqrt(geom->gcov[4 * (i == U1_RAD) + 7 * (i == U2_RAD) + 9 * (i == U3_RAD)]);
						pb_new[i] = pb_old[i] + dpb;
					}

					get_state_rad(pb_new, geom, &q_rad);
					mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
					for (k = UU_RAD; k <= U3_RAD; k++) U_new[k] *= geom->g;

					U_new[RHO] = U_i[RHO];
					U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
					U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
					U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
					U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);
					U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

					tol = NEWT_TOL;// 0.01 * offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2)));
					flag += Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, BASIC
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, 0.0
						#endif
					);
					#if(DO_FONT_FIX)
					if (flag && (n_iter_jacob > 1)) {
						//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, tol, TYPE2, FULL_ENTROPY
						//#if (DOHELM)
						//, gpu_eos_table
						//#endif
						//#if(TWO_T)
						//, 0.0
						//#endif
						//);
					}
					#endif

					if (flag == 0) {
						//Recompute T_t^mu for consistency
						U_new[RHO] = U_i[RHO];
						get_state(pb_new, geom, &q);
						#if(TWO_T)
						gamma_g = calc_gamma_gas_prim(pb_new);
						#endif
						mhd_calc(pb_new, 0, &q, &U_new[UU]
							#if(DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, gamma_g
							#endif
						);
						for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
						U_new[UU] += U_new[RHO];

						//Electron and ion entropies
						#if(TWO_T)
						U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
						U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
						#endif
	
						//Recalculate gas entropy for consistency
						#if(DOKTOT)
						U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
							#if (DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, gamma_g
							#endif
						);
						#endif

						//Recompute photon number
						#if(P_NUM)
						U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
						#endif

						//Calculate source function and jacobian
						source_rad(pb_new, geom, &q, &q_rad, dU_new
							#if(DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, gamma_g
							#endif
							#if(COOL_STOP)
							, r
							#endif
						);

						//Calculate Jacobian
						for (k = U1_RAD; k <= U3_RAD; k++) {
							E_new[k - UU_RAD] = (U_new[k] - U_i[k] - Dt * dU_new[k]);
							dEdpb[k - UU_RAD][i - UU_RAD] = (E_new[k - UU_RAD] - E_old[k - UU_RAD]) / dpb;
						}
						#if(TWO_T)
						E_new[4] = (U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]);
						dEdpb[4][i - UU_RAD] = (E_new[4] - E_old[4]) / dpb;
						#endif
						#if(P_NUM)
						E_new[4 + TWO_T] = (U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]);
						dEdpb[4 + TWO_T][i - UU_RAD] = (E_new[4 + TWO_T] - E_old[4 + TWO_T]) / dpb;
						#endif
						if (do_entropy == 1) {
							#if(FULL_ENTROPY)
							T_GAS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA);
							#else
							T_GAS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
							#endif					
							E_new[0] = 1.0 / T_GAS * (U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]);
						}
						else E_new[0] = (U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]);
						dEdpb[0][i - UU_RAD] = (E_new[0] - E_old[0]) / dpb;
					}
				}
			}
			
			if(flag==0){
				//Invert Jacobian
				#if(P_NUM && TWO_T)
				flag = invert_matrix_6D(dEdpb, dEdpb_inv);
				#elif(P_NUM || TWO_T)
				flag = invert_matrix_5D(dEdpb, dEdpb_inv);
				#else
				flag = invert_matrix_4D(dEdpb, dEdpb_inv);
				#endif
			}

			n_iter_jacob++;
		} while (flag && (offset * pow(10., (double)(1 - 2 * (n_iter_jacob % 2)) * ((double)(n_iter_jacob / 2))) < 0.00003));

		if (flag) return 1;

		n_iter_fail = 0;
		while (n_iter_fail < 5) {
			//Set primitive variables before Newton step
			PLOOP{
				pb_new[k] = pb_old[k];
				U_new[k] = U_old[k];
			}

			/* Make the newton step: */
			if (do_staged == 0) {
				D = 1. / pow(2.0, (double)n_iter_fail);
				for (k = 0; k < 4; k++) {
					dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
						#if(TWO_T)
						+ E_old[4] * dEdpb_inv[k][4]
						#endif
						#if(P_NUM)
						+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
						#endif
						);
					pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
				}
			}
			else {
				//Set damping factor for Newton Raphson method
				if (n_iter == 0 || n_iter == 4) D = 0.5 / pow(2.0, (double)n_iter_fail);
				else if (n_iter == 8) D = 0.25 / pow(2.0, (double)n_iter_fail);
				else D = 1. / pow(2.0, (double)n_iter_fail);

				if (n_iter / 4 == 0) { //momentum only step
					for (k = 0; k < 4; k++) {
						dpb = -D * (E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
				if (n_iter / 4 == 1) {
					for (k = 0; k < 4; k++) { //energy only step
						dpb = -D * (E_old[0] * dEdpb_inv[k][0]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
				else {
					for (k = 0; k < 4; k++) { //full 4d step
						dpb = -D * (E_old[0] * dEdpb_inv[k][0] + E_old[1] * dEdpb_inv[k][1] + E_old[2] * dEdpb_inv[k][2] + E_old[3] * dEdpb_inv[k][3]
							#if(TWO_T)
							+ E_old[4] * dEdpb_inv[k][4]
							#endif
							#if(P_NUM)
							+ E_old[4 + TWO_T] * dEdpb_inv[k][4 + TWO_T]
							#endif	
						);
						pb_new[k + UU_RAD] = pb_old[k + UU_RAD] + dpb;
					}
				}
			}

			#if(TWO_T)
			dpb = -D * (E_old[0] * dEdpb_inv[4][0] + E_old[1] * dEdpb_inv[4][1] + E_old[2] * dEdpb_inv[4][2] + E_old[3] * dEdpb_inv[4][3] + E_old[4] * dEdpb_inv[4][4]
				#if(P_NUM)
				+ E_old[4 + P_NUM] * dEdpb_inv[4][4 + P_NUM]
				#endif	
			);
			U_new[ENTRE] = U_old[ENTRE] + dpb;
			#endif
			#if(P_NUM)
			dpb = -D * (E_old[0] * dEdpb_inv[4 + TWO_T][0] + E_old[1] * dEdpb_inv[4 + TWO_T][1] + E_old[2] * dEdpb_inv[4 + TWO_T][2] + E_old[3] * dEdpb_inv[4 + TWO_T][3] + E_old[4] * dEdpb_inv[4 + TWO_T][4]
				#if(TWO_T)
				+ E_old[4 + TWO_T] * dEdpb_inv[4 + TWO_T][4 + TWO_T]
				#endif			
			);
			U_new[PHOTON] = U_old[PHOTON] + dpb;
			#endif

			//Make sure that radiation internal energy stays positive
			if (pb_new[UU_RAD] < 0.0) pb_new[UU_RAD] = 0.5 * fabs(pb_new[UU_RAD]);

			//Make sure that electron entropy stays positive
			#if(TWO_T)
			if (U_new[ENTRE] < 0.0) U_new[ENTRE] = 0.5 * fabs(U_new[ENTRE]);
			#endif

			//Make sure that photon number stays positive
			if (U_new[PHOTON] < 0.0) U_new[PHOTON] = 0.5 * fabs(U_new[PHOTON]);

			//Obtain new radiation conserved quantaties from radiation primitive variables
			get_state_rad(pb_new, geom, &q_rad);
			mhd_calc_rad(pb_new, 0, &q_rad, &U_new[UU_RAD]);
			for (k = UU_RAD; k <= U3_RAD; k++)U_new[k] *= geom->g;

			//Obtatin photon number primitive quantity
			#if(P_NUM)
			pb_new[PHOTON] = (1.0 / geom->g) * U_new[PHOTON] / q_rad.ucon[0];
			#endif

			//Derive new MHD conserved quantaties for radiation variables
			U_new[RHO] = U_i[RHO];
			U_new[UU] = U_i[UU] - (U_new[UU_RAD] - U_i[UU_RAD]);
			U_new[U1] = U_i[U1] - (U_new[U1_RAD] - U_i[U1_RAD]);
			U_new[U2] = U_i[U2] - (U_new[U2_RAD] - U_i[U2_RAD]);
			U_new[U3] = U_i[U3] - (U_new[U3_RAD] - U_i[U3_RAD]);

			//Estimate conserved entropy
			U_new[KTOT] = U_i[KTOT] + Dt * dU_old[KTOT];

			//Get MHD primitives
			flag = Utoprim_2d(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif
			);
			#if(DO_FONT_FIX)
			if (flag) {
				//flag = Utoprim_1dvsq2fix1(U_new, geom->gcov, geom->gcon, geom->g, pb_new, NEWT_TOL, TYPE2, FULL_ENTROPY
				//#if (DOHELM)
				//, gpu_eos_table
				//#endif
				//#if(TWO_T)
				//, 0.0
				//#endif
				//);
			}
			#endif

			if (flag == 0) {
				//Recompute T_t^mu for consistency
				U_new[RHO] = U_i[RHO];
				get_state(pb_new, geom, &q);
				#if(TWO_T)
				gamma_g = calc_gamma_gas_prim(pb_new);
				#endif
				mhd_calc(pb_new, 0, &q, &U_new[UU]
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				for (k = UU; k <= U3; k++) U_new[k] *= geom->g;
				U_new[UU] += U_new[RHO];

				//Electron and ion entropies
				#if(TWO_T)
				U_new[ENTRE] = U_new[RHO] * pb_new[ENTRE];
				U_new[ENTRI] = U_new[RHO] * pb_new[ENTRI];
				#endif
	
				//Recalculate gas entropy for consistency
				#if(DOKTOT)
				U_new[KTOT] = U_new[RHO] * calc_entropy(pb_new
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
				);
				#endif

				//Recompute photon number
				#if(P_NUM)
				U_new[PHOTON] = geom->g * pb_new[PHOTON] * q_rad.ucon[0];
				#endif

				//Get radiative source term
				source_rad(pb_new, geom, &q, &q_rad, dU_new
					#if(DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, gamma_g
					#endif
					#if(COOL_STOP)
					, r
					#endif
				);

				//Calculate iterated error
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1_RAD]) + fabs(U_new[U1_RAD]) + fabs(Dt * dU_new[U1_RAD]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2_RAD]) + fabs(U_new[U2_RAD]) + fabs(Dt * dU_new[U2_RAD]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3_RAD]) + fabs(U_new[U3_RAD]) + fabs(Dt * dU_new[U3_RAD]));
				error_new[n_iter % 5] = 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1_RAD] - U_i[U1_RAD] - Dt * dU_new[U1_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2_RAD] - U_i[U2_RAD] - Dt * dU_new[U2_RAD]) / norm);
				error_new[n_iter % 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3_RAD] - U_i[U3_RAD] - Dt * dU_new[U3_RAD]) / norm);
				norm = (fabs(U_i[UU_RAD]) + fabs(U_new[UU_RAD]) + fabs(Dt * dU_new[UU_RAD]));
				if (do_entropy == 0)error_new[n_iter % 5] += 0.25 * (fabs(U_new[UU_RAD] - U_i[UU_RAD] - Dt * dU_new[UU_RAD]) / norm);
				#if(TWO_T)
					norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
					#if(CONSTANTGAMMA)
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					#elif(FIXEDGAMMA)
					dK_dS = (GAMMAE - 1.) / pow(pb_new[RHO], GAMMAE - 1.0);
					#elif(VARGAMMA)
					double Theta_e;
					//For variable entropy
						#if(FULL_ENTROPY)
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * exp(pb_new[ENTRE]), 2. / 3.)) - 1.0);
						dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
						#else
						Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pb_new[RHO] * pb_new[ENTRE], 2. / 3.)) - 1.0);
						dK_dS = (pb_new[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
						#endif
					#endif
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[ENTRE] - U_i[ENTRE] - Dt * dU_new[ENTRE]) / (dK_dS * norm));
				#endif
				#if(P_NUM)
				norm = (fabs(U_i[PHOTON]) + fabs(U_new[PHOTON]) + fabs(Dt * dU_new[PHOTON]));
				error_new[n_iter % 5] += 0.25 * (fabs(U_new[PHOTON] - U_i[PHOTON] - Dt * dU_new[PHOTON]) / (norm));
				#endif

				//Set correct offset for Jacobian for next iteration
				if (error_new[n_iter % 5] < pow(10., -9.))offset = pow(10., -10.);
				else offset = pow(10., -8.);

				//Set total error to iterated error
				error_new[n_iter % 5 + 5] = error_new[n_iter % 5];

				//Calculate total error
				norm = (fabs(U_i[UU]) + fabs(U_new[UU]) + fabs(Dt * dU_new[UU]));
				if (do_entropy == 0)error_new[n_iter % 5 + 5] += 0.25 * (fabs(U_new[UU] - U_i[UU] - Dt * dU_new[UU]) / norm);
				else {
					#if(FULL_ENTROPY)
					T_GAS = (GAMMA - 1.) * pb_new[UU] / pb_new[RHO];
					error_new[n_iter % 5 + 5] += 0.25 * T_GAS * (fabs(U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT])) / (norm);
					#else
					dK_dS = (GAMMA - 1.) / pow(pb_new[RHO], GAMMA - 1.0);
					error_new[n_iter % 5 + 5] += 0.25 * (fabs((U_new[KTOT] - U_i[KTOT] - Dt * dU_new[KTOT]))) / (norm * dK_dS);
					#endif
				}
				norm = sqrt(geom->gcon[4]) * (fabs(U_i[U1]) + fabs(U_new[U1]) + fabs(Dt * dU_new[U1]));
				norm += sqrt(geom->gcon[7]) * (fabs(U_i[U2]) + fabs(U_new[U2]) + fabs(Dt * dU_new[U2]));
				norm += sqrt(geom->gcon[9]) * (fabs(U_i[U3]) + fabs(U_new[U3]) + fabs(Dt * dU_new[U3]));
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[4]) * (fabs(U_new[U1] - U_i[U1] - Dt * dU_new[U1]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[7]) * (fabs(U_new[U2] - U_i[U2] - Dt * dU_new[U2]) / norm);
				error_new[n_iter % 5 + 5] += 0.25 * sqrt(geom->gcon[9]) * (fabs(U_new[U3] - U_i[U3] - Dt * dU_new[U3]) / norm);

				//If we've reached the tolerance level or we exceeded more than 20 iterations, stop iterating
				if ((fabs(error_new[n_iter % 5 + 5]) <= 1.e-12) || (n_iter >= 20)) {
					keep_iterating = 0;
				}

				//If error increasing stop iterating
				if (n_iter >= 4 && (0.3333 * (error_new[(n_iter - 4) % 5 + 5] + error_new[(n_iter - 3) % 5 + 5] + error_new[(n_iter - 2) % 5 + 5]) < 0.5 * (error_new[(n_iter - 1) % 5 + 5] + error_new[(n_iter - 0) % 5 + 5]))) {
					//keep_iterating = 0;
				}

				//If error increased more than 4 times stop iterating
				if ((n_iter > 4) && (error_new[(n_iter - 1) % 5 + 5] < error_new[(n_iter) % 5 + 5])) {
					count_increase++;
					//if (count_increase >= 5) keep_iterating = 0;
				}

				//If gas negative more than 2 times stop iterating
				if (pb_new[UU] < 0.) {
					count_increase_gas++;
					//if (count_increase > 2) keep_iterating = 0;
				}

				//Reset variables if Newton step succesfull
				if (keep_iterating) {
					for (k = 0; k < NPR; k++) {
						U_old[k] = U_new[k];
						pb_old[k] = pb_new[k];
						dU_old[k] = dU_new[k];
					}
				}

				//If error decreased compared to start value, update variables
				if (fabs(error_new[n_iter % 5 + 5]) < error_t[1] && fabs(error_new[n_iter % 5 + 5]) < 0.01) {
					error_t[0] = error_new[n_iter % 5];
					error_t[1] = error_new[n_iter % 5 + 5];
					for (k = 0; k < NPR; k++) {
						pb[k] = pb_new[k];
						U_f[k] = U_new[k];
						dU[k] = dU_new[k];
					}
				}
				break;
			}
			else {
				n_iter_fail++;
				if (n_iter_fail == 5) return(1);
			}
		}
		n_iter++;
	}
	return(0);
}

//Calculate fraction of heat that goes into electrons on ions based on temperature ratio at previous timestep: 
__device__ double calc_delta(double* ph, double bsq) {
	double delta;
	#if(HEAT_HOWES)
	double fel, c1, c2, c3, Te, Ti, beta, ratio;
	
	Te = calc_Te(ph);
	Ti = calc_Ti(ph);

	ratio = fabs((Te * MU_E) / (Ti * MU_I));
	c1 = 0.92;
	if ((Ti * MU_I) > (Te * MU_E)) {
		c2 = 1.6 * ratio;
		c3 = 18.0 - 5.0 * log10(ratio);
	}
	else {
		c2 = 1.2 * ratio;
		c3 = 18.0;
	}

	beta = ((Te + Ti) * ph[RHO] + 0.3333333 * ph[UU_RAD]) / (0.5 * bsq);
	if (!isfinite(beta) || beta>10000.0 || beta<0.000001) beta = 10000.0;
	fel = c1 * (c2 * c2 + pow(beta, 2.0 + 0.2 * log10(ratio))) / (c3 * c3 + pow(beta, 2.0 + 0.2 * log10(ratio))) * sqrt((MH_CGS / ME_CGS) * (MU_I * Ti) / (MU_E * Te)) * exp(-1.0 / beta);

	//Calculate delta
	delta = 1. / (1. + fel);
	#elif(HEAT_ROWAN)
	double sigma_w, beta_i, beta_max, Te, Ti;

	Te = calc_Te(ph);
	Ti = calc_Ti(ph);

	sigma_w = bsq / (ph[RHO] + GAMMA * ph[UU]);
	beta_i = (Ti * ph[RHO] + Te * ph[RHO]) / (0.5 * bsq);
	beta_max = 1.0 / (4.0 * sigma_w);

	//Calculate delta
	delta = 0.5 * exp((beta_i / beta_max - 1.0)) / (0.8 + sqrt(sigma_w));
	#else
	//Set delta to constant value
	delta=0.5;
	#endif

	if (!isfinite(delta) || delta > 1.0 || delta < 0.0) delta = 0.5;

	return delta;
}

__device__ void heating(double* ph, struct of_state* q)
{
	
}

__device__ double source_Coulomb(double* p) {
	double th_mean, th_sum, Theta_e, Theta_i, coeff, n_cgs, ne_cgs, T_e, T_i;
	double K2e, K2i, K0, K1;
	double theta_min = 1.e-2;
	double coulog; 
	double res;

	#if(CONSTANTGAMMA)
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(p[ENTRE] * pow(p[RHO], GAMMA - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(p[ENTRI] * pow(p[RHO], GAMMA - 1.0)) * MU_I);
		#else
		Theta_e = fabs(p[ENTRE] * pow(p[RHO], GAMMA - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(p[ENTRI] * pow(p[RHO], GAMMA - 1.0) * MU_I);
		#endif
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(p[ENTRI] * pow(p[RHO], GAMMA - 1.0)) * MU_I);
		#else
		Theta_e = fabs(p[ENTRE] * pow(p[RHO], GAMMAE - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(p[ENTRI] * pow(p[RHO], GAMMA - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO] * exp(p[ENTRE]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(p[RHO] * exp(p[ENTRI]), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(p[RHO] * p[ENTRE]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(p[RHO] * p[ENTRI]), 2. / 3.)) - 1.0);
		#endif
	#endif

	//note that average number density in Sadowski+17 (eq (20)) is assumed to be n_ave = ne_cgs.this can be updated 
	ne_cgs = p[RHO] * MASS_DENSITY_SCALE / (MU_E * MH_CGS);    // calculation in cgs unit
	n_cgs = p[RHO] * MASS_DENSITY_SCALE / (MH_CGS);    // calculation in cgs unit

	T_e = Theta_e / BOLTZ_CGS * (ME_CGS * C_CGS * C_CGS);
	T_i = Theta_i / BOLTZ_CGS * (MH_CGS * C_CGS * C_CGS);

	coulog = 35.4 + log(T_e / (1.0e7) * sqrt(1.0e-3 / ne_cgs));// Coulomb logarithm ( ln Lambda )
	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
	coeff *= ne_cgs * n_cgs * (T_i - T_e);

	th_sum = Theta_e + Theta_i;
	th_mean = Theta_e * Theta_i / (Theta_e + Theta_i);

	if (Theta_i < theta_min && Theta_e < theta_min) // approximated equations at small theta
	{
		res = coeff / sqrt(0.5 * M_PI * th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (Theta_i < theta_min)
	{
		//bessel function
		K2e = bessk(2.0, 1. / Theta_e);
		res = coeff / K2e / exp(1. / Theta_e) * sqrt(Theta_e) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else if (Theta_e < theta_min)
	{
		//bessel function
		K2i = bessk(2.0, 1. / Theta_i);

		res = coeff / K2i / exp(1. / Theta_i) * sqrt(Theta_i) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else // general form in Sadowski+17 (eq 20)
	{
		//bessel functions
		K2e = bessk(2.0, 1. / Theta_e);
		K2i = bessk(2.0, 1. / Theta_i);
		K0 = bessk0(1.0 / th_mean);
		K1 = bessk1(1.0 / th_mean);

		res = coeff / (K2e * K2i) * ((2. * th_sum * th_sum + 1.) / th_sum * K1 + 2. * K0);
	}

	if (!isfinite(res)) res = 0.;

	res = res / ENERGY_DENSITY_SCALE * R_GOC_CGS;     // unit conversion from cgs to grid unit
	return (res);
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy and gas density
__device__ double calc_gamma_gas_conserved(double* S, double rho) {
	double gamg, game, gami, Theta_e, Theta_i;

	#if(CONSTANTGAMMA)
	gamg = GAMMA;
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(S[0] * pow(rho, game - 1.0)) * (MU_E * MASS_RATIO)); //Actually theta_e=MU_E*Te meant
		Theta_i = fabs((gami - 1.0) * exp(S[1] * pow(rho, gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(S[0] * pow(rho, game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(S[1] * pow(rho, gami - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(S[0]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(S[1]), 2. / 3.)) - 1.0);
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif
	
	#if(VARGAMMA || FIXEDGAMMA)
	if ((Theta_i < 0.00001) || (Theta_e < 0.00001)) {
		gamg = 5.0 / 3.0;
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	}
	#endif

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
__device__ double calc_gamma_gas_prim(double* pr) {
	double gamg, game, gami, Theta_e, Theta_i;

	#if(CONSTANTGAMMA)
	gamg = GAMMA;
	#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Theta_e = fabs((game - 1.0) * exp(pr[ENTRE] * pow(pr[RHO], game - 1.0)) * (MU_E * MASS_RATIO));
		Theta_i = fabs((gami - 1.0) * exp(pr[ENTRI] * pow(pr[RHO], gami - 1.0)) * MU_I);
		#else
		Theta_e = fabs(pr[ENTRE] * pow(pr[RHO], game - 1.0) * (MU_E * MASS_RATIO));
		Theta_i = fabs(pr[ENTRI] * pow(pr[RHO], gami - 1.0) * MU_I);
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * exp(pr[ENTRE]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * exp(pr[ENTRI]), 2. / 3.)) - 1.0);
		#else
		Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * fabs(pr[ENTRE]), 2. / 3.)) - 1.0);
		Theta_i = 0.2 * (sqrt(1.0 + 25.0 * pow(pr[RHO] * fabs(pr[ENTRI]), 2. / 3.)) - 1.0);
		#endif
	game = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e);
	gami = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i);
	#endif

	#if(VARGAMMA || FIXEDGAMMA)
	if ((Theta_i < 0.00001) || (Theta_e < 0.00001)) {
		gamg = 5.0 / 3.0;
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (MU_I / (MU_E * MASS_RATIO) + Theta_i / Theta_e)) / ((Theta_i / Theta_e) * (game - 1.0) + MU_I / (MU_E * MASS_RATIO) * (gami - 1.0));
	}
	#endif

	return gamg;
}

//Calculate EOS gamma based on electron (and ion or total entropy) based on conserved entropy, gas density and w=W*(1-vsq)
__device__ double calc_gamma_gas_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;
	#if(VARGAMMA)
	double ratio, C, Theta;
	#endif

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(CONSTANTGAMMA)
	game = GAMMA;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(FIXEDGAMMA)     
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0) / MU_I;
		#else
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(fabs(rho * S[1]), 2. / 3.)) - 1.0) / MU_I;
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	if ((Ti * MU_I < 0.00001) || (Te * MU_E * MASS_RATIO < 0.00001)) {
		gami = GAMMA;
		game = GAMMAE;
		gamg = 5.0 / 3.0;
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	}
	#else
	gamg = GAMMA;
	#endif

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = quantg / gamg - ughat;

	//Update internal energy of electrons
	if (dis == -100.0) {
		quante = game * u_e;
		quanti = gami * u_i;
		double factor = quantg / (quante + quanti);
		quante *= factor;
		quanti *= factor;
	}
	else {
		u_e += delta * dis;
		#if(VARGAMMA)
		C = u_e / rho * MU_E * MASS_RATIO;
		//Theta=(gam-1)*ue/rho*MU_E*MASS_RATIO
		//Theta=((10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)-1)*ue/rho*MU_E*MASS_RATIO
		//Theta=((10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)-1)*C --> Solved in Wolfram Alpha
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
		if (Theta < 0.00001) game = GAMMAE;
		else game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		#endif
		quante = game * u_e; //quant=(gam)/(gam-1)*p
	}

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	//if (quante > quantg) quante = quantg;
	//if (quante < 0.0) quante = 0.0;
	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//Use analytical inversions
	C = MU_E * MASS_RATIO * quante / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / (MU_E*MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I * quanti / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gas eos gammma
	#if(FIXEDGAMMA || VARGAMMA)
	if ((Ti * MU_I < 0.00001) || (Te * MU_E * MASS_RATIO < 0.00001)) {
		gamg = 5.0 / 3.0;
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	}
	#else
	gamg = GAMMA;
	#endif

	return gamg;
}

//Update electron and ion entropy based on found w in Newton Raphson solver
__device__ double set_S_w(double* S, double rho, double w, double delta) {
	double gamg, game, gami, Te, pe, pi, Ti, u_e, u_i, dis, ughat, quantg, quanti, quante;

	quantg = fabs(w - rho); //quant=gamma*ug=gamma/(gamma-1)*p

	//Figure out if electron quant_e energy is bigger than quant_g
	#if(CONSTANTGAMMA)
	game = GAMMA;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(FIXEDGAMMA)   
	game = GAMMAE;
	gami = GAMMA;
		#if(FULL_ENTROPY)
		Te = fabs(exp((game - 1.0) * S[0] * pow(rho, game - 1.0)));
		Ti = fabs(exp((gami - 1.0) * S[1] * pow(rho, gami - 1.0)));
		#else
		Te = fabs(S[0] * pow(rho, game - 1.0));
		Ti = fabs(S[1] * pow(rho, gami - 1.0));
		#endif
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
		#if(FULL_ENTROPY)
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * exp(S[1]), 2. / 3.)) - 1.0) / (MU_I);
		#else
		Te = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * fabs(S[1]), 2. / 3.)) - 1.0) / (MU_I);
		#endif
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);
	#endif

	//Calculate gamma assuming purely adiabatic evolution
	#if(FIXEDGAMMA || VARGAMMA)
	if ((Ti * MU_I < 0.00001) || (Te * MU_E * MASS_RATIO < 0.00001)) {
		gami = GAMMA;
		game = GAMMAE;
		gamg = 5.0 / 3.0;
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	}
	#else
	gamg = GAMMA;
	#endif

	//Calculate gas pressures
	pe = Te * rho;
	pi = Ti * rho;

	//Calculate internal energy
	u_e = pe / (game - 1.0);
	u_i = pi / (gami - 1.0);

	//Total adiabatic evolution of ions and electrons
	ughat = (u_e + u_i);

	//Calculate dissipation assuming gamg didn't change
	dis = quantg / gamg - ughat;

	//Update internal energy of electrons
	if (dis == -100.0) {
		quante = game * u_e;
		quanti = gami * u_i;
		double factor = quantg / (quante + quanti);
		quante *= factor;
		quanti *= factor;
	}
	else {
		u_e += delta * dis;
		#if(VARGAMMA)
		double C, Theta;
		C = u_e / rho * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
		if (Theta < 0.00001) game = GAMMAE;
		else game = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		#endif
		quante = game * u_e; //quant=(gam)/(gam-1)*p
	}

	if (quante > (1.0 - FLOOR_ENTROPY) * quantg) quante = (1.0 - FLOOR_ENTROPY) * quantg;
	if (quante < FLOOR_ENTROPY * quantg) quante = FLOOR_ENTROPY * quantg;
	//if (quante >  quantg) quante = quantg;
	//if (quante < 0.0) quante = 0.0;
	quanti = quantg - quante;

	#if(CONSTANTGAMMA || FIXEDGAMMA)
	pe = (game - 1.0) / game * quante;
	pi = (gami - 1.0) / gami * quanti;

	//Set entropy
		#if(FULL_ENTROPY)
		S[0] = 1.0 / (game - 1.0) * log(pe * pow(rho, -game));
		S[1] = 1.0 / (gami - 1.0) * log(pi * pow(rho, -gami));
		#else
		S[0] = pe * pow(rho, -game);
		S[1] = pi * pow(rho, -gami);
		#endif
	Te = pe / rho;
	Ti = pi / rho;
	#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
	//quant*C=(10.0 + 20.0 * x * C) / (6.0 + 15.0 * x * C)/((10.0 + 20.0 * x * C) / (6.0 + 15.0 * x * C)-1)*x*C
	//B=(10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)/((10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta)-1)*Theta --> Solved this analytically in Wolfram
	//C= (MU_E * MASS_RATIO) / RHO
	//B=quant*C
	//x=pe
	//Theta=x*C
	double C = MU_E * MASS_RATIO * quante / rho;
	Te = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / (MU_E * MASS_RATIO);
	game = (10.0 + 20.0 * Te * MU_E * MASS_RATIO) / (6.0 + 15.0 * Te * MU_E * MASS_RATIO);
	C = MU_I * quanti / rho;
	Ti = 1.0 / 40.0 * (sqrt(5.0) * sqrt(5.0 * C * C + 44.0 * C + 20.0) + 5.0 * C - 10.0) / MU_I;
	gami = (10.0 + 20.0 * Ti * MU_I) / (6.0 + 15.0 * Ti * MU_I);

	//Set entropy
		#if(FULL_ENTROPY)
		S[0] = log(pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
		S[1] = pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho;
		#endif
	#endif

	#if(FIXEDGAMMA || VARGAMMA)
	if ((Ti * MU_I < 0.00001) || (Te * MU_E * MASS_RATIO < 0.00001)) {
		gami = GAMMA;
		game = GAMMAE;
		gamg = 5.0 / 3.0;

		pe = (game - 1.0) / game * quante;
		pi = (gami - 1.0) / gami * quanti;

		Te = pe / rho;
		Ti = pi / rho;

		#if(FULL_ENTROPY)
		S[0] = log(pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = pow(Te * (MU_E * MASS_RATIO), 1.5) * pow(Te * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
		S[1] = pow(Ti * MU_I, 1.5) * pow(Ti * MU_I + 0.4, 1.5) / rho;
		#endif
	}
	else {
		gamg = 1.0 + ((game - 1.0) * (gami - 1.0) * (1.0 + Ti / Te)) / (Ti / Te * (game - 1.0) + 1.0 * (gami - 1.0));
	}
	#else
	gamg = GAMMA;
	#endif

	return gamg;
}

// Some bessel functions
__device__ double bessi0(double x) {
	double ax, ans, y;

	if ((ax = fabs(x)) < 3.75) {
		y = x / 3.75, y = y * y;
		ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
			+ y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
	}
	else {
		y = 3.75 / ax;
		ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
			+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
				+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
					+ y * 0.392377e-2))))))));
	}

	return ans;
}

__device__ double bessi1(double x) {
	double ax, ans, y;

	if ((ax = fabs(x)) < 3.75) {
		y = x / 3.75, y = y * y;
		ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
			+ y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
	}
	else {
		y = 3.75 / ax;
		ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
			- y * 0.420059e-2));
		ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
			+ y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
		ans *= (exp(ax) / sqrt(ax));
	}

	return (x < 0.0 ? -ans : ans);
}

__device__ double bessk0(double x) {
	double y, ans;

	if (x <= 2.0) {
		y = x * x / 4.0;
		ans = (-log(x / 2.0) * bessi0(x)) + (-0.57721566 + y * (0.42278420
			+ y * (0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2
				+ y * (0.10750e-3 + y * 0.74e-5))))));
	}
	else {
		y = 2.0 / x;
		ans = (exp(-x) / sqrt(x)) * (1.25331414 + y * (-0.7832358e-1
			+ y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2
				+ y * (-0.251540e-2 + y * 0.53208e-3))))));
	}

	return ans;
}

__device__ double bessk1(double x) {
	double y, ans;

	if (x <= 2.0) {
		y = x * x / 4.0;
		ans = (log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144
			+ y * (-0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1
				+ y * (-0.110404e-2 + y * (-0.4686e-4)))))));
	}
	else {
		y = 2.0 / x;
		ans = (exp(-x) / sqrt(x)) * (1.25331414 + y * (0.23498619
			+ y * (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2
				+ y * (0.325614e-2 + y * (-0.68245e-3)))))));
	}

	return ans;
}

__device__ double bessk(int n, double x) {
	int j;
	double bk, bkm, bkp, tox;

	tox = 2.0 / x;
	bkm = bessk0(x);
	bk = bessk1(x);
	for (j = 1; j < n; j++) {
		bkp = bkm + j * tox * bk;
		bkm = bk;
		bk = bkp;
	}

	return bk;
}

__device__ int Utoprim_3d_res(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double tolerance, int lim, double Dt);
__device__ int invert_3DU(double D, double sigma, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
__device__ void res_3du_der(double D, double sigma, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);
__device__ void getE_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac);
__device__ void getdEdu_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]);
__device__ int invert_3DU_entropy(double D, double sigma, double etares, double entropy, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
__device__ void res_3du_der_entropy(double D, double sigma, double etares, double entropy, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);

__device__ int Utoprim_3d_res(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double tolerance, int lim, double Dt) {
	double D, tau, S[3], B_guess[3], E_guess[3], ncov_0, ncon[NDIM], rho, ug;
	int i, j, retval = 1;
	double alpha, sqrtgamma;
	double vD_guess[3], ggamma[3][3], ggammainv[3][3];

	//Return if rho*gamma is negative
	if (U[0] <= 0.) return(-100);

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);
	sqrtgamma = gdet / alpha; //determinant for spatial part of metric
	ncov_0 = -alpha;
	ncon[0] = ncov_0 * gcon[0];
	ncon[1] = ncov_0 * gcon[1];
	ncon[2] = ncov_0 * gcon[2];
	ncon[3] = ncov_0 * gcon[3];

	//Calculate covariant and contravariant 3+1 metrics
	ggamma[0][0] = gcov[4]; 
	ggamma[0][1] = gcov[5];
	ggamma[0][2] = gcov[6];
	ggamma[1][0] = gcov[5];
	ggamma[1][1] = gcov[7];
	ggamma[1][2] = gcov[8];
	ggamma[2][0] = gcov[6];
	ggamma[2][1] = gcov[8];
	ggamma[2][2] = gcov[9];
	ggammainv[0][0] = gcon[4] + ncon[1] * ncon[1];
	ggammainv[0][1] = gcon[5] + ncon[1] * ncon[2];
	ggammainv[0][2] = gcon[6] + ncon[1] * ncon[3];
	ggammainv[1][0] = gcon[5] + ncon[2] * ncon[1];
	ggammainv[1][1] = gcon[7] + ncon[2] * ncon[2];
	ggammainv[1][2] = gcon[8] + ncon[2] * ncon[3];
	ggammainv[2][0] = gcon[6] + ncon[3] * ncon[1];
	ggammainv[2][1] = gcon[8] + ncon[3] * ncon[2];
	ggammainv[2][2] = gcon[9] + ncon[3] * ncon[3];

	//Transform the CONSERVED variables to 3+1
	D = alpha * U[RHO] / gdet;

	//Energy to 3+1
	tau = ncov_0 * (ncon[0] * (U[UU] - U[RHO]) + ncon[1] * U[U1] + ncon[2] * U[U2] + ncon[3] * U[U3]) / gdet - D;

	//Momentum to 3+1
	S[0] = -ncov_0 * (U[U1]) / gdet;
	S[1] = -ncov_0 * (U[U2]) / gdet;
	S[2] = -ncov_0 * (U[U3]) / gdet;

	//Magnetic field to 3+1
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
			vD_guess[i] += ggamma[i][j] * prim[U1 + j];
		}
	}

	//NR Step, you get back gamma*v_i and E
	retval = invert_3DU(D, Dt * alpha, ETA, tau, S, ggamma, ggammainv, sqrtgamma, &rho, &ug, B_guess, E_guess, vD_guess, tolerance);

	//Backup entropy inversion
	#if(DOKTOT)
	if (retval != 0) {
		#if(FULL_ENTROPY)
		double kappa = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
		#else
		double kappa = fabs(U[KTOT] / U[RHO]);
		#endif
		retval = invert_3DU_entropy(D, Dt * alpha, ETA, kappa, S, ggamma, ggammainv, sqrtgamma, &rho, &ug, B_guess, E_guess, vD_guess, tolerance);
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

		double vU_guess[3], lfac_guess;
		vD_guess[0] = 0.;
		vD_guess[1] = 0.;
		vD_guess[2] = 0.;
		for (i = 0; i < 3; i++) {
			vU_guess[i] = 0.;
			
			for (j = 0; j < 3; j++) {
				vU_guess[i] = vU_guess[i] + ggammainv[i][j] * vD_guess[j];
			}
		}
		lfac_guess = sqrt(1.0 + vU_guess[0] * vD_guess[0] + vU_guess[1] * vD_guess[1] + vU_guess[2] * vD_guess[2]);
		getE_resistive(E_guess, E_guess, vU_guess, vD_guess, B_D, Dt * alpha, ETA, ggammainv, sqrtgamma, lfac_guess);
		prim[E1] = E_guess[0] / alpha;
		prim[E2] = E_guess[1] / alpha;
		prim[E3] = E_guess[2] / alpha;
	}

	//Update B fields regardless to preserve Div.B==0 regardless if inversion is succesful
	for (i = BCON1; i <= BCON3; i++) prim[i] = U[i] / gdet;

	return(retval);
}

//gives back E and gamma*v_i
__device__ int invert_3DU(double D, double sigma, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance) {
	double vD[3], vU[3], vDprev[3], xk_3du[3], J_3du[3][3], J_3du_inv[3][3], f_3du[3], Enew[3], B_D[3], dvd[3], lfac;
	double Enew_D[3], Stilde_j[3], ExB[3], Stilde_uj[3], Ssqr, bsqr, esqr, tautilde, z, eps;
	int i, j, nit, ii;
	double er, er1, normV;
	int maxitr = 100;
	int retval = 1;
	int retval_matrix;

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
		res_3du_der(D, sigma, etares, tau, S, xk_3du, ggamma, ggammainv, sqrtgamma, B_guess, E_guess, J_3du, f_3du);

		//Store previous ucov_tilde
		for (i = 0; i < 3; i++) vDprev[i] = vD[i];

		//Find inverse of Jacobian 
		retval_matrix = invert_matrix_3D(J_3du, J_3du_inv);

		//Print error if jacobian is singular
		if (retval_matrix == 1) {
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
		retval = 1;
		return retval;
	}
	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//Exit if Lorent factor smaller than 1
	if (lfac < 1.0) {
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

	ExB[0] = sqrtgamma * (Enew[1] * B_guess[2] - Enew[2] * B_guess[1]);
	ExB[1] = sqrtgamma * (Enew[2] * B_guess[0] - Enew[0] * B_guess[2]);
	ExB[2] = sqrtgamma * (Enew[0] * B_guess[1] - Enew[1] * B_guess[0]);
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
		retval = 2;
		return retval;
	}
	if (ug[0] < 0.) {
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
__device__ void res_3du_der(double D, double sigma, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]) {
	double Enew[3], Enew_D[3], B_D[3], dEdu[3][3], Stilde_j[3], ExB[3], tautilde, vU[3], decrossb[3], Stilde_uj[3];
	double lfac, z, esqr, bsqr, Ssqr, eps, p, enth, edotde, depsdu, denthdu, dpdu, S_ujdotdecrossb;
	int i, j;

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

	//calculate bsqr and esqr
	bsqr = B[0] * B_D[0] + B[1] * B_D[1] + B[2] * B_D[2];
	esqr = Enew[0] * Enew_D[0] + Enew[1] * Enew_D[1] + Enew[2] * Enew_D[2];

	//calculate tautilde and stilde
	tautilde = tau - (bsqr + esqr) * 0.5;

	ExB[0] = sqrtgamma * (Enew[1] * B[2] - Enew[2] * B[1]);
	ExB[1] = sqrtgamma * (Enew[2] * B[0] - Enew[0] * B[2]);
	ExB[2] = sqrtgamma * (Enew[0] * B[1] - Enew[1] * B[0]);
	for (i = 0; i < 3; i++) Stilde_j[i] = S_j[i] - ExB[i];

	//raise Stilde
	for (i = 0; i < 3; i++) {
		Stilde_uj[i] = 0.;
		for (j = 0; j < 3; j++) {
			Stilde_uj[i] = Stilde_uj[i] + ggammainv[i][j] * Stilde_j[j];
		}
	}

	Ssqr = Stilde_uj[0] * Stilde_j[0] + Stilde_uj[1] * Stilde_j[1] + Stilde_uj[2] * Stilde_j[2];

	//compute pressure
	z = sqrt(lfac * lfac - 1.0);
	eps = lfac * tautilde / D - z * sqrt(Ssqr) / D + z * z / (1.0 + lfac);
	p = (GAMMA - 1.0) * D / lfac * eps;
	enth = 1.0 + GAMMA / (GAMMA - 1.0) * p / D * lfac;

	//compute residuals
	res[0] = vD[0] - Stilde_j[0] / D / enth;
	res[1] = vD[1] - Stilde_j[1] / D / enth;
	res[2] = vD[2] - Stilde_j[2] / D / enth;

	//compute Jacobian
	//1-direction
	decrossb[0] = sqrtgamma * (dEdu[0][1] * B[2] - dEdu[0][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[0][2] * B[0] - dEdu[0][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[0][0] * B[1] - dEdu[0][1] * B[0]);
	edotde = Enew_D[0] * dEdu[0][0] + Enew_D[1] * dEdu[0][1] + Enew_D[2] * dEdu[0][2];
	depsdu = vU[0] / lfac * tautilde / D - lfac * edotde / D + 2.0 * vU[0] / (1.0 + lfac) - (z * z / ((1.0 + lfac) * (1.0 + lfac))) * vU[0] / lfac;

	if (z != 0.0) depsdu = depsdu - vU[0] / z * sqrt(Ssqr) / D;

	S_ujdotdecrossb = Stilde_uj[0] * decrossb[0] + Stilde_uj[1] * decrossb[1] + Stilde_uj[2] * decrossb[2];
	if (Ssqr != 0.0) depsdu = depsdu + (z / D) * S_ujdotdecrossb / sqrt(Ssqr);

	dpdu = (GAMMA - 1.0) * D * (depsdu / lfac - eps / (lfac * lfac * lfac) * vU[0]);
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[0] / lfac) / D;

	Jac[0][0] = 1.0 + decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][0] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][0] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//2-direction	
	decrossb[0] = sqrtgamma * (dEdu[1][1] * B[2] - dEdu[1][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[1][2] * B[0] - dEdu[1][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[1][0] * B[1] - dEdu[1][1] * B[0]);
	edotde = Enew_D[0] * dEdu[1][0] + Enew_D[1] * dEdu[1][1] + Enew_D[2] * dEdu[1][2];
	depsdu = vU[1] / lfac * tautilde / D - lfac * edotde / D + 2.0 * vU[1] / (1.0 + lfac) - (z * z / ((1.0 + lfac) * (1.0 + lfac))) * vU[1] / lfac;

	if (z != 0.0) depsdu = depsdu - vU[1] / z * sqrt(Ssqr) / D;

	S_ujdotdecrossb = Stilde_uj[0] * decrossb[0] + Stilde_uj[1] * decrossb[1] + Stilde_uj[2] * decrossb[2];
	if (Ssqr != 0.0) depsdu = depsdu + (z / D) * S_ujdotdecrossb / sqrt(Ssqr);

	dpdu = (GAMMA - 1.0) * D * (depsdu / lfac - eps / (lfac * lfac * lfac) * vU[1]);
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[1] / lfac) / D;

	Jac[0][1] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][1] = 1.0 + decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][1] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//3-direction
	decrossb[0] = sqrtgamma * (dEdu[2][1] * B[2] - dEdu[2][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[2][2] * B[0] - dEdu[2][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[2][0] * B[1] - dEdu[2][1] * B[0]);
	edotde = Enew_D[0] * dEdu[2][0] + Enew_D[1] * dEdu[2][1] + Enew_D[2] * dEdu[2][2];
	depsdu = vU[2] / lfac * tautilde / D - lfac * edotde / D + 2.0 * vU[2] / (1.0 + lfac) - (z * z / ((1.0 + lfac) * (1.0 + lfac))) * vU[2] / lfac;

	if (z != 0.0) depsdu = depsdu - vU[2] / z * sqrt(Ssqr) / D;

	S_ujdotdecrossb = Stilde_uj[0] * decrossb[0] + Stilde_uj[1] * decrossb[1] + Stilde_uj[2] * decrossb[2];
	if (Ssqr != 0.0) depsdu = depsdu + (z / D) * S_ujdotdecrossb / sqrt(Ssqr);

	dpdu = (GAMMA - 1.0) * D * (depsdu / lfac - eps / (lfac * lfac * lfac) * vU[2]);
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[2] / lfac) / D;

	Jac[0][2] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][2] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][2] = 1.0 + decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

}

//Recover E
__device__ void getE_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac) {
	double vxbU[3], e0dotv;
	int i;

	// ucov x B_D
	vxbU[0] = 1.0 / sqrtgamma * (vD[1] * B_D[2] - vD[2] * B_D[1]);
	vxbU[1] = 1.0 / sqrtgamma * (vD[2] * B_D[0] - vD[0] * B_D[2]);
	vxbU[2] = 1.0 / sqrtgamma * (vD[0] * B_D[1] - vD[1] * B_D[0]);

	// ImEx: Eold_upper.ucov
	e0dotv = E[0] * vD[0] + E[1] * vD[1] + E[2] * vD[2];

	//eta<1 case
	for (i = 0; i < 3; i++) {
		Enew[i] = etares * E[i] / (etares + lfac * sigma) - sigma / (etares + lfac * sigma) * (vxbU[i] - etares * e0dotv / (etares * lfac + sigma) * vU[i]);
	}
}

//Recover E and DE/du
__device__ void getdEdu_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]) {
	double vxbU[3], kxbU[3];
	int i;
	double e0dotv, denom1, denom2;

	// ucov x B_D
	vxbU[0] = 1.0 / sqrtgamma * (vD[1] * B_D[2] - vD[2] * B_D[1]);
	vxbU[1] = 1.0 / sqrtgamma * (vD[2] * B_D[0] - vD[0] * B_D[2]);
	vxbU[2] = 1.0 / sqrtgamma * (vD[0] * B_D[1] - vD[1] * B_D[0]);

	// ImEx: Eold_upper.ucov
	e0dotv = E[0] * vD[0] + E[1] * vD[1] + E[2] * vD[2];

	//eta<1 case
	for (i = 0; i < 3; i++) {
		Enew[i] = etares * E[i] / (etares + lfac * sigma) - sigma / (etares + lfac * sigma) * (vxbU[i] - etares * e0dotv / (etares * lfac + sigma) * vU[i]);
	}
	denom1 = etares + lfac * sigma;
	denom2 = etares * lfac + sigma;

	// Derivative of u x B: dE/dv1
	kxbU[0] = 0.;
	kxbU[1] = 1.0 / sqrtgamma * (-B_D[2]);
	kxbU[2] = 1.0 / sqrtgamma * (B_D[1]);

	//Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[0][i] = -E[i] * etares / (denom1 * denom1) * sigma * vU[0] / lfac
			- (-sigma * sigma / (denom1 * denom1) * vU[0] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i])
				+ sigma / denom1 * (kxbU[i]
					+ etares * (-E[0] / denom2 * vU[i] + etares * e0dotv / (denom2 * denom2) * vU[0] / lfac * vU[i] - e0dotv / denom2 * ggammainv[0][i])));
	}

	//Derivative of u x B: dE/dv2
	kxbU[0] = 1.0 / sqrtgamma * (B_D[2]);
	kxbU[1] = 0.;
	kxbU[2] = 1.0 / sqrtgamma * (-B_D[0]);

	// Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[1][i] = -E[i] * etares / (denom1 * denom1) * sigma * vU[1] / lfac
			- (-(sigma * sigma / (denom1 * denom1)) * vU[1] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i])
				+ sigma / denom1 * (kxbU[i]
					+ etares * (-E[1] / denom2 * vU[i] + etares * e0dotv / (denom2 * denom2) * vU[1] / lfac * vU[i] - e0dotv / denom2 * ggammainv[1][i])));
	}

	// Derivative of u x B: dE/dv3
	kxbU[0] = 1.0 / sqrtgamma * (-B_D[1]);
	kxbU[1] = 1.0 / sqrtgamma * (B_D[0]);
	kxbU[2] = 0.;

	// Build derivative
	for (i = 0; i < 3; i++) {
		dEdu[2][i] = -E[i] * etares / (denom1 * denom1) * sigma * vU[2] / lfac
			- (-(sigma * sigma / (denom1 * denom1)) * vU[2] / lfac * (vxbU[i] - etares * e0dotv / denom2 * vU[i])
				+ sigma / denom1 * (kxbU[i]
					+ etares * (-E[2] / denom2 * vU[i] + etares * e0dotv / (denom2 * denom2) * vU[2] / lfac * vU[i] - e0dotv / denom2 * ggammainv[2][i])));
	}
}

//gives back E and gamma*v_i
__device__ int invert_3DU_entropy(double D, double sigma, double etares, double kappa, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance) {
	double vD[3], vU[3], vDprev[3], xk_3du[3], J_3du[3][3], J_3du_inv[3][3], f_3du[3], Enew[3], B_D[3], dvd[3], lfac;
	int i, j, nit, ii;
	double er, er1, normV;
	int maxitr = 100;
	int retval = 1;
	int retval_matrix;

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
		retval = 1;
		return retval;
	}
	//calculate Lorentz factor
	lfac = sqrt(1.0 + vU[0] * vD[0] + vU[1] * vD[1] + vU[2] * vD[2]);

	//Exit if Lorent factor smaller than 1
	if (lfac < 1.0) {
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
		retval = 2;
		return retval;
	}
	if (ug[0] < 0.) {
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
__device__ void res_3du_der_entropy(double D, double sigma, double etares, double kappa, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]) {
	double Enew[3], B_D[3], dEdu[3][3], Stilde_j[3], ExB[3], vU[3], decrossb[3];
	double lfac, p, enth, denthdu, dpdu;
	int i, j;

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

	ExB[0] = sqrtgamma * (Enew[1] * B[2] - Enew[2] * B[1]);
	ExB[1] = sqrtgamma * (Enew[2] * B[0] - Enew[0] * B[2]);
	ExB[2] = sqrtgamma * (Enew[0] * B[1] - Enew[1] * B[0]);
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
	decrossb[0] = sqrtgamma * (dEdu[0][1] * B[2] - dEdu[0][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[0][2] * B[0] - dEdu[0][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[0][0] * B[1] - dEdu[0][1] * B[0]);
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[0];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[0] / lfac) / D;

	Jac[0][0] = 1.0 + decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][0] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][0] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//2-direction	
	decrossb[0] = sqrtgamma * (dEdu[1][1] * B[2] - dEdu[1][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[1][2] * B[0] - dEdu[1][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[1][0] * B[1] - dEdu[1][1] * B[0]);
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[1];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[1] / lfac) / D;

	Jac[0][1] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][1] = 1.0 + decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][1] = decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;

	//3-direction
	decrossb[0] = sqrtgamma * (dEdu[2][1] * B[2] - dEdu[2][2] * B[1]);
	decrossb[1] = sqrtgamma * (dEdu[2][2] * B[0] - dEdu[2][0] * B[2]);
	decrossb[2] = sqrtgamma * (dEdu[2][0] * B[1] - dEdu[2][1] * B[0]);
	dpdu = -GAMMA * kappa * pow(D, GAMMA) / pow(lfac, GAMMA + 2.0) * vU[2];
	denthdu = GAMMA / (GAMMA - 1.0) * (dpdu * lfac + p * vU[2] / lfac) / D;

	Jac[0][2] = decrossb[0] / (D * enth) + Stilde_j[0] / (D * enth * enth) * denthdu;
	Jac[1][2] = decrossb[1] / (D * enth) + Stilde_j[1] / (D * enth * enth) * denthdu;
	Jac[2][2] = 1.0 + decrossb[2] / (D * enth) + Stilde_j[2] / (D * enth * enth) * denthdu;
}


//4D Matrix Inversion
__device__ int invert_matrix_4D(double Am[][NDIM], double Aminv[][NDIM]){
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

//3D Matrix inversion
__device__ int invert_matrix_3D(double Am[][3], double Aminv[][3])
{

	int i, j;
	int n = 3;
	int permute[3];
	double dxm[3], Amtmp[3][3];

	for (i = 0; i < 3 * 3; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_3D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_3D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//5D Matrix inversion
__device__ int invert_matrix_5D(double Am[][5], double Aminv[][5])
{

	int i, j;
	int n = 5;
	int permute[5];
	double dxm[5], Amtmp[5][5];

	for (i = 0; i < 5 * 5; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_5D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_5D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//6D Matrix inversion
__device__ int invert_matrix_6D(double Am[][6], double Aminv[][6])
{

	int i, j;
	int n = 6;
	int permute[6];
	double dxm[6], Amtmp[6][6];

	for (i = 0; i < 6 * 6; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_6D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_6D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//4D LU-decomposition
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
			if(!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
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
}

//5D LU-decomposition
__device__ int LU_decompose_5D(double A[][5], int permute[])
{
	double row_norm[5];
	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */
	double  absmax, maxtemp;
	int i, j, k, max_row;
	int n = 5;

	max_row = 0;
	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);
			if (!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}

	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		for (i = j; i < n; i++) {
			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}
		}

		if (max_row != j) {
			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}
				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}

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

//6D LU-decomposition
__device__ int LU_decompose_6D(double A[][6], int permute[])
{
	double row_norm[6];
	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */
	double  absmax, maxtemp;
	int i, j, k, max_row;
	int n = 6;

	max_row = 0;
	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);
			if (!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}

	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		for (i = j; i < n; i++) {
			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}
		}

		if (max_row != j) {
			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}
				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}

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
}


//3D LU-Substitution
__device__ void LU_substitution_3D(double A[][3], double B[], int permute[])
{
	int i, j;
	int n = 3;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}

//5D LU-Substitution
__device__ void LU_substitution_5D(double A[][5], double B[], int permute[])
{
	int i, j;
	int n = 5;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}

//6D LU-Substitution
__device__ void LU_substitution_6D(double A[][6], double B[], int permute[])
{
	int i, j;
	int n = 6;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}

//3D
//Inversion from radiation conserved to primitive quantities
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim){
	double U_tmp[NPR_R + P_NUM], prim_tmp[NPR_R+P_NUM];
	int i, ret;
	double alpha;

	//Set the geometry variables
	alpha = 1.0 / sqrt(-gcon[0]);

	//Transform the CONSERVED variables into eulerian observers frame nu_Mu=alpha 
	for (i = 0; i < NPR_R; i++) U_tmp[i] = alpha * U[i + UU_RAD] / gdet;
	#if(P_NUM)
	U_tmp[4] = alpha * U[PHOTON] / gdet;
	#endif

	//Transform the PRIMITIVE variables into the new system
	for (i = 0; i < NPR_R; i++) prim_tmp[i] = prim[i + UU_RAD]; //radiation prims
	#if(P_NUM)
	prim_tmp[4] = prim[PHOTON];
	#endif

	//Do inversion
	ret = Rtoprim_calc(U_tmp, gcov, gcon, gdet, prim_tmp, y_max, lim);

	//Transform new primitive variables back if there was no problem
	for (i = 0; i < NPR_R; i++) {
		prim[i + UU_RAD] = prim_tmp[i];
	}
	#if(P_NUM)
	prim[PHOTON] = prim_tmp[4];
	#endif

	return(ret);
}

__device__ int Rtoprim_calc(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim) {
	double Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq = 0., Qtcon[NDIM], Qtsq, Qdotn;
	double Uabs, qsq;
	double gammasq, y, pressure, f;
	#if(P_NUM)
	double Tr;
	#endif
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

	#if(P_NUM)
	prim[4] = U[4] / sqrt(gammasq);
	#endif
	
	/*if (0) {
		prim[0] = 1.e-30;
		prim[1] = 0.;
		prim[2] = 0.;
		prim[3] = 0.;

		// Get Ebar and p_rad as usual
		if (!isnan(Qdotn) && Qdotn < 0.0) {
			pressure = -Qdotn / (4. - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad
		}

		//Floor on photon number+
		#if(P_NUM)
		Tr = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif

		return 0;
	}*/
	if (y > y_max || isnan(Qdotn) || prim[0] < 0. || Qdotn > 0.0 || isnan(y) || y < 0. || isnan(prim[1]) || isnan(prim[2]) || isnan(prim[3])) {
		Uabs = 0.5 * (fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++)prim[i] = Qtcon[i];

		qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
			+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
		//if (qsq < 0. && fabs(qsq) < 1.E-10) qsq = 1.E-10; // set floor
		if (qsq < 0.) {
		 qsq = 1.E-10; // set floor
		}

		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;


		//if (y < 1. - 100. * NUMEPSILON) {
		if ((Qtsq>0.0) && ((prim[1]*prim[1])>0.0) && ((prim[2] * prim[2]) > 0.0) && ((prim[3] * prim[3]) > 0.0)) {
			if (lim==TYPE2) {
				if (Qdotn<0.0) {
					// Get Ebar and p_rad as usual
					Qdotn = -(1.e-150 + sqrt(Qtsq / y_max));
					pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
					prim[0] = pressure * 3.; // Erad = 3*p_rad
					prim[1] = 0.;
					prim[2] = 0.;
					prim[3] = 0.;
					returnval = 1;
				}
				else{
					prim[0] = 1.e-30;
					prim[1] = 0.;
					prim[2] = 0.;
					prim[3] = 0.;
					returnval = 1;
					//pressure = Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
					///prim[0] = pressure * 3.; // Erad = 3*p_rad	
				}
				//Qdotn = -(1.e-150 + sqrt(Qtsq / y_max));
				//pressure = -Qdotn / (4. * GAMMAMAX_RAD*GAMMAMAX_RAD - 1.);
				//prim[0] = pressure * 3.; // Erad = 3*p_rad	
				//prim[1] = 0.;
				//prim[2] = 0.;
				//prim[3] = 0.;
			}
			else {
				prim[0] = 1.e-30;
				prim[1] = 0.;
				prim[2] = 0.;
				prim[3] = 0.;
				returnval = 1;
			}

		}
		else {
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;

			returnval = 1;
		}

		//Floor on photon number+
		#if(P_NUM)
		Tr = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif
	}

	#if(P_NUM)
	if (prim[4] < 0.0) {
		Tr = pow(prim[0] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		prim[4] = prim[0] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		returnval = 1;
	}
	#endif

	return(returnval);


	/*
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

	//Check for bad values of -Erad and U_tilde^2; If values are nan floor them
	if (!isfinite(Qdotn)) Qdotn = -(1.e-30);
	if (!isfinite(Qtsq)) Qtsq = 0.0;
	if (!isfinite(Qtcon[1])) Qtcon[1] = 0.0;
	if (!isfinite(Qtcon[2])) Qtcon[2] = 0.0;
	if (!isfinite(Qtcon[3])) Qtcon[3] = 0.0;

	y = Qtsq / (Qdotn * Qdotn + 1.e-30); //Definition from McKinney2013. Should only range [0,1].
	if (y < 0. || isnan(y)) y = 0.;
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	#if(P_NUM)
	prim[4] = U[4] / sqrt(gammasq);
	#endif

	if (Qdotn > 0.) { //Negative internal energy
		if (lim == TYPE2) {
			Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
			for (i = 1; i < 4; i++) {
				if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
				prim[i] = Qtcon[i] / Uabs;
			}
			qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
				+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
			if (qsq < 1.E-10) qsq = 1.E-10; // set floor
			gammasq = 1. + qsq;

			f = 0.;// sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
			if (f < 10000000.0) {
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			}
			Qdotn = -(1.e-150 + sqrt(fabs(Qtsq) / y_max));
			prim[0] = pressure * 3.; // Erad = 3*p_rad

			returnval = 1;
		}
		else {
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
			gammasq = 1.0;
		}

		#if(P_NUM)
		prim[4] = U[4] / sqrt(gammasq);
		#endif
	}
	else if (y > y_max) {
		Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++) {
			if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
			prim[i] = Qtcon[i] / Uabs;
		}
		qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
			+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
		if (qsq < 1.E-10 || !isfinite(qsq)) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		//if (f < 10000000.0 && y<1.0-0.000000001){
		prim[1] *= f;
		prim[2] *= f;
		prim[3] *= f;
		//}
		//else {
		//	prim[1] = 0;
		//	prim[2] = 0;
		//	prim[3] = 0;
	//	}
		if (lim == TYPE2) {
			Qdotn = -(1.e-30 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = pressure * 3.; // Erad = 3*p_rad
			returnval = 1;
		}
		else{
			prim[0] = pressure * 3.; // Erad = 3*p_rad
		}

		#if(P_NUM)
		prim[4] = U[4] / sqrt(GAMMAMAX_RAD * GAMMAMAX_RAD);
		#endif
	}
	return returnval;
	*/
	/*
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

	//Check for bad values of -Erad and U_tilde^2; If values are nan floor them
	//if (!isfinite(Qdotn)) Qdotn = -(1.e-30);
	//if (!isfinite(Qtsq)) Qtsq = 0.0;

	y = Qtsq / (Qdotn * Qdotn + 1.e-30); //Definition from McKinney2013. Should only range [0,1].
	//if (y < 0. || isnan(y)) y = 0.;
	gammasq = (2. - y + sqrt(4. - 3. * y)) / (4. - 4. * y);

	// Get Ebar and p_rad as usual
	pressure = -Qdotn / (4. * gammasq - 1.);
	prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad

	// utilde ^i _rad = gam_rad * Utilde^i / (4 * p * gam_rad^2)
	for (i = 1; i < 4; i++) prim[i] = sqrt(gammasq) * Qtcon[i] / (4. * pressure * gammasq);

	#if(P_NUM)
	prim[4] = U[4] / sqrt(gammasq);
	#endif



	if (Qdotn > 0. || isnan(Qdotn)) { //Negative internal energy
		if (lim == -10) {
			Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
			for (i = 1; i < 4; i++) {
				if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
				prim[i] = Qtcon[i] / Uabs;
			}

			qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
				+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
			if (qsq < 1.E-10) qsq = 1.E-10; // set floor
			gammasq = 1. + qsq;

			f = 0.0;// sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
			//if (f < 10000000.0) {
				prim[1] *= f;
				prim[2] *= f;
				prim[3] *= f;
			//}
			Qdotn = -(1.e-150 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad

			returnval = 1;
		}
		else {
			prim[0] = 1.e-30;
			prim[1] = 0.;
			prim[2] = 0.;
			prim[3] = 0.;
			gammasq = 1.0;
		}

		#if(P_NUM)
		prim[4] = U[4] / sqrt(gammasq);
		#endif
	}
	else if (y > y_max || isnan(y)) {
		Uabs = 0.5 * (sqrt(fabs(Qtsq)) + fabs(Qdotn) + 1.e-150);
		for (i = 1; i < 4; i++) {
			if (!isfinite(Qtcon[i]))Qtcon[i] = 0.;
			prim[i] = Qtcon[i] / Uabs;
		}
		qsq = gcov[4] * prim[1] * prim[1] + gcov[7] * prim[2] * prim[2] + gcov[9] * prim[3] * prim[3]
			+ 2. * (gcov[5] * prim[1] * prim[2] + gcov[6] * prim[1] * prim[3] + gcov[8] * prim[2] * prim[3]);
		if (qsq < 1.E-10 || !isfinite(qsq)) qsq = 1.E-10; // set floor
		gammasq = 1. + qsq;

		f = 0.0;// sqrt((GAMMAMAX_RAD * GAMMAMAX_RAD - 1.) / (gammasq - 1.));
		//if (f < 10000000.0 && y<1.0-0.000000001){
			prim[1] = 0.0;
			prim[2] = 0.0;
			prim[3] = 0.0;
		//}
		//else {
		//	prim[1] = 0;
		//	prim[2] = 0;
		//	prim[3] = 0;
	//	}
		if (lim == -10) {
			Qdotn = -(1.e-30 + sqrt(fabs(Qtsq) / y_max));
			pressure = -Qdotn / (4. * GAMMAMAX_RAD * GAMMAMAX_RAD - 1.);
			prim[0] = MY_MAX(pressure * 3., 1.e-30); // Erad = 3*p_rad
			returnval = 1;
		}
		else{
			prim[0] = 1.e-30;
		}
	
		#if(P_NUM)
		prim[4] = U[4] / sqrt(GAMMAMAX_RAD * GAMMAMAX_RAD);
		#endif
	}
	return returnval;
	*/
}

__device__ int Utoprim_NM(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha;
	#if(TWO_T)
	double S[NPR_2T];
	#endif

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

	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_NM_calc(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
    );

	//Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
		#if(TWO_T)
		prim[ENTRE] = S[0];
		prim[ENTRI] = S[1];
		#endif
	}

	return(ret);
}

__device__ int Utoprim_NM_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double *S
	, double fel
	#endif
) {
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u,  w,  gamma,   vsq, errx, gamma_eos;
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

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    // -- to get min. pressure for a given density, set T = T_min = 1e4 K
    double xpres;
    eos_mode_rhotemp_pres_min (gpu_eos_table, prim[RHO], &xpres);
    p_array[0] = xpres;
	#else
		// Ideal gas EOS
		#if(TWO_T)
		gamma_eos = calc_gamma_gas_conserved(S, prim[RHO]);
		#else
		gamma_eos = GAMMA;
		#endif
	p_array[0] = (gamma_eos - 1.) * prim[UU];
	#endif
	
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

		// DANAT: add this - therefore rho0 is nan
		if (fabs(vsq) < 1e-15) vsq = 0.0;
		if (fabs(vsq) >= 1.0) return(1);

		Wsq = 1. / (1. - vsq);
		w = z * (1. - vsq);
		gamma = 1. / sqrt(1. - vsq);
		rho0 = U[RHO] / gamma;

		// EOS-specific calls:
		#if (DOHELM)
		// 1. Helmholtz EOS
		double xpres;
		eos_mode_rhow_pres_u(gpu_eos_table, rho0, w, &xpres, &u);
		#else
			// Ideal gas EOS
			#if(TWO_T)
			gamma_eos = calc_gamma_gas_w(S, rho0, w, fel);
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
		p_array[iter] = (gamma_eos - 1.) * u;
		#endif

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

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        eos_mode_rhow_pres_u (gpu_eos_table, rho0, w, &p_new, &u);
		#else
			#if(TWO_T)
			gamma_eos = set_S_w(S, rho0, w, fel);
			#else
			gamma_eos = GAMMA;
			#endif
		// Ideal gas EOS
		u = (w - rho0) / gamma_eos;
		p_new = (gamma_eos - 1.) * u;
		#endif
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

__device__ int Utoprim_1dfix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim, int full_entropy
	#if(TWO_T)
	, double fel
	#endif
){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha, K_atm;
	#if(TWO_T)
	double S[2];
	#endif

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
	if(full_entropy)K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	else K_atm = U[KTOT] / U[RHO];
	#endif

	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_new_body3(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	if (ret == 0) {
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}
	}

	return(ret);
}

__device__ int Utoprim_new_body3(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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
	retval = general_newton_raphson3(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old, tolerance
		#if(TWO_T)
		, S
		, fel
		#endif
	);

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

__device__ int general_newton_raphson3(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old, double tolerance
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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
		func_1d_orig1(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2, rho_for_gnr2, W_for_gnr2_old, rho_for_gnr2_old
			#if(TWO_T)
			, S
			, fel
			#endif
		);  /* returns with new dx, f, df */
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

__device__ int gnr2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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
		func_gnr2_rho(x, dx, resid, jac, &f, &df, D, K_atm, W_for_gnr2
			#if(TWO_T)
			,  S
			, fel
			#endif
		);  /* returns with new dx, f, df */
		
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
__device__ void func_1d_orig1(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double W_for_gnr2, double rho_for_gnr2, double W_for_gnr2_old, double rho_for_gnr2_old
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	int ntries;
	double  Dc, t1, t10, t2, t21, t23, t26, t29, t3, t30;
	double  t32, t33, t34, t38, t5, t51, t67, t8,  x_rho[1], rho, rho_g;

	//W = x[0];
	W_for_gnr2 = x[0];

	// get rho from NR:
	rho_g = x_rho[0] = rho_for_gnr2;

	ntries = 0;
	while ((gnr2(x_rho, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, W_for_gnr2
		#if(TWO_T)
		,  S
		, fel
		#endif
		)) && (ntries++ < 10)) {
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
__device__ void func_gnr2_rho(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double D, double K_atm, double W_for_gnr2
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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

__device__ int Utoprim_1dvsq2fix1(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim, int full_entropy
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
){
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha, K_atm;
	#if(TWO_T)
	double S[2];
	#endif

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

	//Set entropy to kappa from log(kappa) if necessary
	if (full_entropy) K_atm = exp((U[KTOT] / U[RHO]) * (GAMMA - 1.));
	else K_atm = U[KTOT] / U[RHO];

	//Set electron and ion entropies
	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_new_body2(U_tmp, gcov, gcon, gdet, prim_tmp, K_atm, tolerance, lim
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

	//Transform new primitive variables back if there was no problem
	if (ret == 0) {
		#pragma unroll 5
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

__device__ int Utoprim_new_body2(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double K_atm, double tolerance, int lim
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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
	#if(DOHELM)
	// 1. Helmholtz EOS
	double dpdrho, dudrho;
	eos_mode_rhos_upres(gpu_eos_table, rho0, K_atm, &p, &u, &dpdrho, &dudrho);
	#elif(TWO_T)
	double gamma_g = calc_gamma_gas_conserved(S, prim[RHO]);
	u = prim[UU];
	p = (gamma_g - 1.) * u;
	#else
	// 2. Gamma EOS
	u = prim[UU];
	p = (GAMMA - 1.)*u;
	#endif

	//Initialize independent variables for Newton-Raphson:
	x_1d[0] = 1. - 1. / gammasq;

	//Find vsq via Newton-Raphson:
	retval = general_newton_raphson2(x_1d, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm, tolerance
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

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
	W = W_of_vsq2(vsq, &p, &rho0, &u, D, K_atm
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	
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
	#if(TWO_T)
	set_S_kappa(rho0, K_atm, S, fel);
	#endif

	#pragma unroll 3
	for (i = 1; i < 4; i++) {
		Qtcon[i] = Qcon[i] + ncon[i] * Qdotn;
		prim[UTCON1 + i - 1] = gamma / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
	}

	/* done! */
	return(retval);
}

__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
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
		func_1d_gnr2(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D, K_atm
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			,  S
			, fel
			#endif
		);/* returns with new dx, f, df */

		//Set old values
		errx = 0.;
		x_old[0] = x[0];

		//Make Newton step
		x[0] += dx[0];

		validate_x2(x, x_old);

		//Calculate W=w*gamma
		W_old = W;
		W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, S
			, fel
			#endif
		);
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

__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double W, Wsq, dWdvsq, fact_tmp, rho, p, u;
	//vsq = x[0];

	// Calculate best value for W given current guess for vsq: 
	#if(DOHELM)
	// Helmholtz EOS
	dWdvsq_calc2_helmholtz(gpu_eos_table, x[0], D, K_atm, &W, &dWdvsq);
	Wsq = W * W;
	#else
	//Gamma EOS
	W = W_of_vsq2(x[0], &p, &rho, &u, D, K_atm
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);
	Wsq = W * W;

		// Doing this assuming  P = (G-1) u :
		#if(TWO_T)
		double p_new, u_new, rho_new, vsq_new, W_new, dvsq;
		dvsq = MY_MIN(1.e-8, 1.0 - (x[0] + fabs(1.e-8)));
		vsq_new = x[0] + dvsq;
		W_new=W_of_vsq2(vsq_new, &p_new, &rho_new, &u_new, D, K_atm
			#if(DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, S
			, fel
			#endif
		);
		dWdvsq = (W_new - W) / dvsq;
		#else
		dWdvsq = dWdvsq_calc2(x[0], rho, p);
		#endif
	#endif

	fact_tmp = (Bsq + W);
	resid[0] = Qtsq - x[0] * fact_tmp * fact_tmp + QdotBsq * (Bsq + 2.*W) / Wsq;
	jac[0][0] = -fact_tmp * (fact_tmp + 2. * dWdvsq * (x[0] + QdotBsq / (W*Wsq)));
	dx[0] = -resid[0] / jac[0][0];
	*f = 0.5*resid[0] * resid[0];
	*df = -2. * (*f);
}

__device__ double W_of_vsq2(double vsq, double *p, double *rho, double *u, double D, double K_atm
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
){
	double gtmp;
	gtmp = (1. - vsq);
	rho[0] = D * sqrt(gtmp);
	#if(DOHELM)
	// 1. Helmholtz EOS
	double dpdrho, dudrho;
	eos_mode_rhos_upres(gpu_eos_table, *rho, K_atm, p, u, &dpdrho, &dudrho);
	// 2. Gamma EOS
	#elif(TWO_T)
		//Calculate EOS gamma based on electron (and ion or total entropy)  based on primitive variables
		double gamg, game, gami, pe, pi, T_e, T_i, T_g;

		#if(CONSTANTGAMMA)
		game = GAMMA;
		gami = GAMMA;
			#if(FULL_ENTROPY)
			T_e = fabs((game - 1.0) * exp(S[0] * pow(*rho, game - 1.0)));
			T_i = fabs((gami - 1.0) * exp(S[1] * pow(*rho, gami - 1.0)));
			#else
			T_e = fabs(S[0] * pow(*rho, game - 1.0));
			T_i = fabs(S[1] * pow(*rho, gami - 1.0));
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
			#if(FULL_ENTROPY)
			T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho[0] * pow(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
			T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho[0] * pow(S[1]), 2. / 3.)) - 1.0) / MU_I;
			#else
			T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho[0] * S[0], 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
			T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho[0] * S[1], 2. / 3.)) - 1.0) / MU_I;
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
	p[0] = K_atm * pow(rho[0], GAMMA);
	u[0] = p[0] / (GAMMA - 1.);
	#endif
	return((rho[0] + u[0] + p[0]) / gtmp);
}

__device__ void set_S_kappa(double rho, double K_atm, double* S, double fel) {
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
		#if(FULL_ENTROPY)
		T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * pow(S[0]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * pow(S[1]), 2. / 3.)) - 1.0) / MU_I;
		#else
		T_e = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * S[0], 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
		T_i = 0.2 * (sqrt(1.0 + 25.0 * pow(rho * S[1], 2. / 3.)) - 1.0) / MU_I;
		#endif
	#endif

	//Calculate gas pressures
	pe = T_e * rho;
	pi = T_i * rho;

	//Update internal energy of electrons
	p = (pe + pi);

	//Limit temperature ratios
	if (pe > (1.0 - 0.5 * FLOOR_ENTROPY) * p) pe = (1.0 - FLOOR_ENTROPY) * p;
	if (pe < 0.5 * FLOOR_ENTROPY * p) pe = FLOOR_ENTROPY * p;
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
		#if(FULL_ENTROPY)
		S[0] = log(pow(T_e * (MU_E * MASS_RATIO), 1.5) * pow(T_e * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho);
		S[1] = log(pow(T_i * MU_I, 1.5) * pow(T_i * MU_I + 0.4, 1.5) / rho);
		#else
		S[0] = pow(T_e * (MU_E * MASS_RATIO), 1.5) * pow(T_e * (MU_E * MASS_RATIO) + 0.4, 1.5) / rho;
		S[1] = pow(T_i * MU_I, 1.5) * pow(T_i * MU_I + 0.4, 1.5) / rho;
		#endif
	#endif
}

//W=((rho+u+p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+u+p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+gam/(gam-1)*p)/(1-vsq))
//W=((D*sqrt(1.0-vsq)+gam/(gam-1)*kappa*rho^gamma)/(1-vsq))
//dWdvsq=(0.5*rho+u+p)/(1-vsq)^2 + d(u+p)/dvsq/(1-vsq)
//dWdvsq=(0.5*rho+u+p)/(1-vsq)^2 +
__device__ double dWdvsq_calc2(double vsq, double rho, double p){
	return((GAMMA*(2. - GAMMA)*p + (GAMMA - 1.)*rho) / (2.*(GAMMA - 1.)*(1. - vsq)*(1. - vsq)));
}

// Entropy inversion, Helmholtz EOS
#if(DOHELM)
__device__ void dWdvsq_calc2_helmholtz(const double* __restrict__ gpu_eos_table, double vsq, double D, double K_atm, double* W, double* dWdvsq)
{
	double gtmp;
	gtmp = (1. - vsq);
	double rho = D * sqrt(gtmp);

	double p, u, dpdrho, dudrho;
	eos_mode_rhos_upres(gpu_eos_table, rho, K_atm, &p, &u, &dpdrho, &dudrho);
	*W = (rho + u + p) / gtmp;

	*dWdvsq = ((0.5 * rho * (1.0 - dpdrho - dudrho) + p + u) / (gtmp * gtmp));
}
#endif

__device__ int Utoprim_2d(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double fel
	#endif
) {
	double U_tmp[NPR_U], prim_tmp[NPR_HD];
	int i, ret;
	double alpha;
	#if(TWO_T)
	double S[NPR_2T];
	#endif

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

	//Calculate entropy variable for 2T fluids to recover EOS gamma
	#if(TWO_T)
	S[0] = U[ENTRE] / U[RHO];
	S[1] = U[ENTRI] / U[RHO];
	#endif

	ret = Utoprim_new_body(U_tmp, gcov, gcon, gdet, prim_tmp, tolerance, lim
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

	/* Transform new primitive variables back if there was no problem : */
	if (ret == 0) {
		#pragma unroll 5
		for (i = 0; i < BCON1; i++) {
			prim[i] = prim_tmp[i];
		}

		#if(TWO_T)
		prim[ENTRE] = S[0];
		prim[ENTRI] = S[1];
		#endif
	}

	return(ret);
}

__device__ int Utoprim_new_body(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double tolerance, int lim
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double *S
	, double fel
	#endif
){
	double x_2d[NEWT_DIM_2];
	double QdotB, Bcon[NDIM], Bcov[NDIM], Qcov[NDIM], Qcon[NDIM], ncov, ncon[NDIM], Qsq, Qtcon[NDIM];
	double rho0, u, p, w, gammasq, gamma, gtmp, W_last, W, utsq, vsq;
	int i, retval=0, i_increase;
	double Bsq, QdotBsq, Qtsq, Qdotn, D;
	#if(TWO_T)
	double gamma_g;
	#endif

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
	#elif(TWO_T)
	gamma_g = calc_gamma_gas_conserved(S, prim[RHO]);
	p = (gamma_g - 1.) * u;
    #else
    // 2. Ideal gas EOS
	p = (GAMMA - 1.)*u;
    #endif

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
	retval = general_newton_raphson(x_2d, Bsq, Qtsq, QdotBsq, Qdotn, D, tolerance
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, S
		, fel
		#endif
	);

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

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    eos_mode_rhow_pres_u (gpu_eos_table, rho0, w, &p, &u);
	#elif(TWO_T)
	gamma_g = set_S_w(S, rho0, w
		#if(TWO_T)
		, fel
		#endif
	);
	u = (w - rho0) / gamma_g;
	p = (gamma_g - 1.0) * u;
    #else
    // 2. Ideal gas EOS
	p = (GAMMA - 1.)*(w - rho0) / GAMMA;
	u = w - (rho0 + p);
    #endif

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

__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double *S
	, double fel
	#endif
){
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
		func_vsq(x, dx, resid, jac, &f, &df, Bsq, Qtsq, QdotBsq, Qdotn, D
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, S
			, fel
			#endif
		);  /* returns with new dx, f, df */

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

__device__ void func_vsq(double x[], double dx[], double resid[], double jac[][NEWT_DIM_2], double *f, double *df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double *S
	, double fel
	#endif
){
	double Wsq, p_tmp, dPdvsq, dPdW, gtmp;
	double t11, t16, t18, t2, t21, t23,t24, t25, t3, t35, t36, t4, t40, t9;

	//W = x[0];
	//vsq = x[1];

	Wsq = x[0] * x[0];
	gtmp = 1. - x[1];

	// EOS-specific calls:
	#if (DOHELM)
	// 1. Helmholtz EOS
	double rho = D * sqrt(gtmp);
	double dpdrho, dpde_d;

	eos_mode_rhow_pres_dpdrho_dpde_d(gpu_eos_table, rho, (x[0]*gtmp), &p_tmp, &dpdrho, &dpde_d);

	double dpdeps_o_rho = dpde_d / rho;
	double dpdvsq_1 = -0.5 * D / sqrt(gtmp) * dpdrho;
	double dpdvsq_2 = -0.5 * (x[0] + p_tmp / sqrt(gtmp)) / rho;
	dPdW = (dpdeps_o_rho / (1.0 + dpdeps_o_rho)) * gtmp;
	dPdvsq = (dpdvsq_1 + dpde_d * dpdvsq_2) / (1.0 + dpdeps_o_rho);
	#elif(TWO_T)
	double gamma_eos1, gamma_eos2, w, rho, dgamma, factor, dvsq, dW;

	//Temporary variables
	gtmp = 1. - x[1];
	w = x[0] * gtmp;
	rho = D * sqrt(gtmp);
	gamma_eos1 = calc_gamma_gas_w(S, rho, w, fel);
	factor = (gamma_eos1 - 1.) / gamma_eos1;
	p_tmp = factor * (x[0] * gtmp - D * sqrt(gtmp));

	//Offset sizes
	dW = 1.e-8 * rho;
	dvsq = MY_MIN(1.e-8, fabs(1.0 - (x[1] + 1.e-8)));

	//Calculate dPdW
	gamma_eos2 = calc_gamma_gas_w(S, rho, (x[0] + dW) * gtmp, fel);
	dgamma = (gamma_eos2 - gamma_eos1) / dW;
	dPdW = factor * gtmp + (x[0] * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;

	//Calculate dPdvsq
	gamma_eos2 = calc_gamma_gas_w(S, D * sqrt(fabs(1.0 - (x[1] + dvsq))), x[0] * (1.0 - (x[1] + dvsq)), fel);
	dgamma =  (gamma_eos2 - gamma_eos1) / dvsq;
	dPdvsq = factor * (0.5 * D / sqrt(gtmp) - x[0]) + (x[0] * gtmp - D * sqrt(gtmp)) * pow(gamma_eos1, -2.0) * dgamma;
	#else
	// 2. Ideal gas EOS
	#if(AMD)
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#else
	p_tmp = (GAMMA - 1.) * (fma(x[0], gtmp, -D * sqrt(gtmp))) / GAMMA;
	dPdW = (GAMMA - 1.) * (1. - x[1]) / GAMMA;
	dPdvsq = (GAMMA - 1.) * (fma(0.5, D / sqrt(1. - x[1]), -x[0])) / GAMMA;
	#endif
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
__device__ int fixup_cell(double* pf, double r, struct of_geom* geom
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
) {
	#if(!CARTESIAN)
	double rhoscal, uuscal, rhoflr, uuflr, bsq, wold, wnew, QdotB, trans, vpar, one_over_ucondr_t, x, f;
	double pf_prefloor[NPR_U], betapar, betasq, betasqmax, gamma, ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], vcon[NDIM], ucon[NDIM], utcon[NDIM], B, Bsq, udotB, ut;
	#if(RESISTIVE)
	struct of_state_res q;
	#else
	struct of_state q;
	#endif
	int dofloor=0, flag = 0, m, k;

	rhoscal = pow(r, -POWRHO);
	uuscal = pow(rhoscal, GAMMA);

	rhoflr = RHOMIN * rhoscal;
	uuflr = UUMIN * uuscal;

	#if(RESISTIVE)
	get_state_res(pf, geom, &q);
	#else
	get_state(pf, geom, &q);
	#endif
	bsq = dot(q.bcon, q.bcov);

	//tie floors to the local values of magnetic field and internal energy density
	if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);	
	#if(RAD_M1)
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	if (rhoflr < (pf[UU]+pf[UU_RAD]) / UORHOMAX)  rhoflr = (pf[UU] + pf[UU_RAD]) / (UORHOMAX);
	#else
	if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
	if (rhoflr < pf[UU] / UORHOMAX) rhoflr = pf[UU] / (UORHOMAX);
	#endif
	if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
	if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;

	//Store old values
	#pragma unroll 9
	for (k = 0; k < NPR_U; k++) pf_prefloor[k] = pf[k];

	//floor on density 
	if (pf[RHO] < rhoflr) {
		pf[RHO] = rhoflr;
		dofloor = 1;
	}

	//Internal energy floor
	#if(RAD_M1)
	if (pf[UU] + pf[UU_RAD] < uuflr) {
		pf[UU] = uuflr - pf[UU_RAD];
		dofloor = 1;
	}
	if (pf[UU] < 0.0001*uuflr) {
		pf[UU] = 0.0001 * uuflr;
		dofloor = 1;
	}
	#else
	if (pf[UU] < uuflr) {
		pf[UU] = uuflr;
		dofloor = 1;
	}
	#endif

	//Floor on radiation energy density
	#if(RAD_M1)
	if (pf[UU_RAD] < pow(10., -30.)) {
		pf[UU_RAD] = pow(10., -30.);
		
		//Floor on photon number
		#if(P_NUM)
		double Tr;
		Tr = pow(pf[UU_RAD] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
		pf[PHOTON] = pf[UU_RAD] * C_CGS * C_CGS / (2.701178 * BOLTZ_CGS * Tr);
		#endif
	}
	#endif

	#if(DRIFT_FLOOR)
	trans = 10. * bsq / MY_MIN(pf[RHO], pf[UU]) - 1.;
	if (dofloor && (trans) > 0.) {
		if (trans > 1.) trans = 1.;
		betapar = -q.bcon[0] / ((bsq + SMALL) * q.ucon[0]);
		betasq = betapar * betapar * bsq;
		betasqmax = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		if (betasq > betasqmax) betasq = betasqmax;

		gamma = 1. / sqrt(1 - betasq);
		#pragma unroll 4
		for (m = 0; m < NDIM; m++) ucondr[m] = gamma * (q.ucon[m] + betapar * q.bcon[m]);

		Bcon[0] = 0.;
		#pragma unroll 3
		for (m = 1; m < NDIM; m++) Bcon[m] = pf[B1 - 1 + m];

		lower(Bcon, geom->gcov, Bcov);
		udotB = dot(q.ucon, Bcov);
		Bsq = dot(Bcon, Bcov);
		B = sqrt(Bsq);

		//enthalpy before the floors
		#if (DOHELM)
		double xP;
		eos_mode_rhou_pres(gpu_eos_table, pf_prefloor[RHO], pf_prefloor[UU], &xP);
		wold = pf_prefloor[RHO] + pf_prefloor[UU] + xP;
		#elif(TWO_T)
		wold = pf_prefloor[RHO] + pf_prefloor[UU] * GAMMA;
		#else
		wold = pf_prefloor[RHO] + pf_prefloor[UU] * GAMMA;
		#endif

		//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
		QdotB = udotB * wold * q.ucon[0];

		//enthalpy after the floors
		#if (DOHELM)
		eos_mode_rhou_pres(gpu_eos_table, pf[RHO], pf[UU], &xP);
		wnew = pf[RHO] + pf[UU] + xP;
		#elif(TWO_T)
		wnew = pf[RHO] + pf[UU] * GAMMA;
		#else
		wnew = pf[RHO] + pf[UU] * GAMMA;
		#endif

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
		double dpf[NPR_U], U_prefloor[NPR_U], dU[NPR_U], U[NPR_U], Xtransone_over_ucondr;
		struct of_state_rad q_rad;
		#pragma unroll 9
		for (k = 0; k < NPR_U; k++) dpf[k] = pf[k] - pf_prefloor[k];

		//compute the conserved quantity associated with floor addition
		get_state(dpf, geom, q);
		primtoflux(dpf, q, 0, geom, dU, NULL, NULL
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

		//compute the prefloor conserved quantity
		get_state(pf_prefloor, geom, q);
		primtoflux(pf_prefloor, q, 0, geom, U_prefloor, NULL, NULL
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if (DOHELM)
			, gamma_g
			#endif
		);

		//add U_added to the current conserved quantity
		#pragma unroll 9
		for (k = 0; k < NPR_U; k++) U[k] = U_prefloor[k] + dU[k];

		#if(NEWMAN)
		flag = Utoprim_NM(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, 0.0
			#endif
		);
		#else
		flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, 0.0
			#endif
		);
		#endif
		if (0) {
			#if( DO_FONT_FIX ) 
			flag = Utoprim_1dvsq2fix1(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC, 0
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, 0.0
				#endif	
			);
			if (flag) {
				flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC, 0
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);
			}
			#endif	
		}
	}
	#endif

	#if(TWO_T)
	if(dofloor) {
		#if(CONSTANTGAMMA || FIXEDGAMMA)
			#if(FULL_ENTROPY)
			pf[ENTRE] = 1. / (GAMMAE - 1.) * log(0.5 * (GAMMAE - 1.0) * pf[UU] * pow(pf[RHO], -GAMMAE));
			pf[ENTRI] = 1. / (GAMMA - 1.) * log(0.5 * (GAMMA - 1.0) * pf[UU] * pow(pf[RHO], -GAMMA));
			#else
			pf[ENTRE] = 0.5 * (GAMMAE - 1.0) * pf[UU] * pow(pf[RHO], -GAMMAE);
			pf[ENTRI] = 0.5 * (GAMMA - 1.0) * pf[UU] * pow(pf[RHO], -GAMMA);
			#endif
		#elif(VARGAMMA)
		double Theta, u, C;
		u = 0.5 * pf[UU];
		C = u / pf[RHO] * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY)
			pf[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO]);
			#else
			pf[ENTRE] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO];
			#endif

		C = u / pf[RHO] * MU_I;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY)
			pf[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO]);
			#else
			pf[ENTRI] = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO];
			#endif
		#endif
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
	#else 
	return(0);
	#endif
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
__device__ void source(double *  ph, struct of_geom *  geom, int icurr, int jcurr, int zcurr, double *  dU, double Dt, const  double* __restrict__ conn_GPU, struct of_state *  q, double r
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if (TWO_T)
	, double gamma_g
	#endif
) {
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
	#elif(TWO_T)
	P = (gamma_g - 1.) * ph[UU];
	#else
	// Ideal gas EOS
	P = (GAMMA - 1.)*ph[UU];
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
	double Y = (GAMMA - 1.)*epsilon / T_target; // HELMEOS
	double lambda = om_kepler*ph[UU] * sqrt(Y - 1. + fabs(Y - 1.));
	double int_energy = q->ucov[0] * q->ucon[0] * ph[UU];
	double bsq = dot(q->bcon,q->bcov);
	#if(WHICHPROBLEM==TRUNC_PROBLEM)
	if (r > 40.) {
		if (fabs(q->ucov[0] * lambda) * Dt < 0.1 * fabs(int_energy)) {
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) * (GAMMA - 1.) * lambda;
		}
		else {
			lambda *= (0.1 * fabs(int_energy)) / (fabs(q->ucov[0] * lambda) * Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) * (GAMMA - 1.) * lambda;
		}
	}
	#else
	if (bsq / ph[RHO]<1. || r<10.){
		if (fabs(q->ucov[0] * lambda)*Dt<0.1*fabs(int_energy)){
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda; // HELMEOS
		}
		else{
			lambda *= (0.1*fabs(int_energy)) / (fabs(q->ucov[0] * lambda)*Dt);
			dU[UU] += -q->ucov[0] * lambda;
			dU[U1] += -q->ucov[1] * lambda;
			dU[U2] += -q->ucov[2] * lambda;
			dU[U3] += -q->ucov[3] * lambda;
			dU[KTOT] += -pow(ph[RHO], 1. - GAMMA) *(GAMMA - 1.) * lambda; // HELMEOS
		}
	}
	#endif
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc(double *  pr, int dir, struct of_state * q, double * mhd
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	int j;
	double r, u, P, w, bsq, eta, ptot;

	r = pr[RHO];
	u = pr[UU];

    // EOS-specific calls:
    #if (DOHELM)
    // 1. Helmholtz EOS
    eos_mode_rhou_pres (gpu_eos_table, r, u, &P);
	#elif(TWO_T)
	P = (gamma_g - 1.) * u;
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

__device__ void mhd_calc_rad(double * pr, int dir, struct of_state_rad * q_rad, double * mhd_rad){
	int j;
	/* single row of mhd stress tensor, first index up, second index down */
	DLOOPA mhd_rad[j] = 4. / 3. * pr[UU_RAD] * q_rad->ucon[dir] * q_rad->ucov[j] + 1. / 3. * pr[UU_RAD] * delta(dir, j);
}

__device__ void source_rad(double *  ph, struct of_geom *  geom, struct of_state* q, struct of_state_rad* q_rad, double * dU
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
)
{
	#if(RAD_M1)
	double mhd_rad[NDIM][NDIM], Gcov[NDIM], Gcon[NDIM],dK_dS, bsq;
	int k;

	PLOOP dU[k] = 0.;

	//Add M1 radiation terms
	mhd_calc_rad(ph, 0, q_rad, mhd_rad[0]);
	mhd_calc_rad(ph, 1, q_rad, mhd_rad[1]);
	mhd_calc_rad(ph, 2, q_rad, mhd_rad[2]);
	mhd_calc_rad(ph, 3, q_rad, mhd_rad[3]);

	//Add radiation 4-force
	bsq = q->bcon[0] * q->bcov[0] + q->bcon[1] * q->bcov[1] + q->bcon[2] * q->bcov[2] + q->bcon[3] * q->bcov[3];

	calc_Gcon(ph, Gcon, q->ucon, q->ucov, q_rad->ucon, q_rad->ucov, mhd_rad, bsq
		#if(DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
		#if(P_NUM)
		, &(dU[PHOTON])
		#endif
		#if(COOL_STOP)
		, r
		#endif
	);
	lower(Gcon, geom->gcov, Gcov);

	dU[UU] = Gcov[0];
	dU[U1] = Gcov[1];
	dU[U2] = Gcov[2];
	dU[U3] = Gcov[3];

	dU[UU_RAD] = -Gcov[0];
	dU[U1_RAD] = -Gcov[1];
	dU[U2_RAD] = -Gcov[2];
	dU[U3_RAD] = -Gcov[3];

	//Entropy source term
	#if(DOKTOT)
		#if (DOHELM)
		eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &dK_dS);
		#elif(TWO_T)
			#if(VARGAMMA || FIXEDGAMMA)
			double Theta, C;
			//For variable entropy
			C = ph[UU] / ph[RHO] * MU_G;
			Theta= (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
				#if(FULL_ENTROPY)
				dK_dS = (1.0 / Theta) * (MU_G);
				#else
				dK_dS = (ph[KTOT] / Theta) * (MU_G);
				#endif
			#else
				#if(FULL_ENTROPY)
				dK_dS = ph[RHO] / (gamma_g - 1.) * ph[UU]);
				#else
				dK_dS = (gamma_g - 1.) / pow(ph[RHO], gamma_g - 1.0);
				#endif
			#endif
		#else
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / (GAMMA - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMA - 1.) / pow(ph[RHO], GAMMA - 1.0); 
			#endif
		#endif
		dU[KTOT] = -dK_dS * (Gcov[0] * q->ucon[0] + Gcov[1] * q->ucon[1] + Gcov[2] * q->ucon[2] + Gcov[3] * q->ucon[3]);
	#endif

	//Electron entropy source term for radiative cooling and coulomb coupling
	#if(TWO_T)
		#if(FIXEDGAMMA || CONSTANTGAMMA)
			#if(FULL_ENTROPY)
			dK_dS = ph[RHO] / (GAMMAE - 1.) * ph[UU]);
			#else
			dK_dS = (GAMMAE - 1.) / pow(ph[RHO], GAMMAE - 1.0);
			#endif
		#elif(VARGAMMA)
			double Theta_e;
			//For variable entropy
			#if(FULL_ENTROPY)
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * exp(ph[ENTRE]), 2. / 3.)) - 1.0);
			dK_dS = (1.0 / Theta_e) * (MU_E * MASS_RATIO);
			#else
			Theta_e = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * ph[ENTRE], 2. / 3.)) - 1.0);
			dK_dS = (ph[ENTRE] / Theta_e) * (MU_E * MASS_RATIO);
			#endif
		#endif
		dU[ENTRE] = -dK_dS * (Gcov[0] * q->ucon[0] + Gcov[1] * q->ucon[1] + Gcov[2] * q->ucon[2] + Gcov[3] * q->ucon[3]);
		dU[ENTRE] += dK_dS * source_Coulomb(ph);
	#endif

	#pragma ivdep
	PLOOP dU[k] *= geom->g;
	#endif
}

//Calculate radiation 4-force
__device__ void calc_Gcon(double * ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double ucon_rad[NDIM], double ucov_rad[NDIM], double mhd_rad[NDIM][NDIM], double bsq
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double *source_photon
	#endif
	#if(COOL_STOP)
	, double r
	#endif
) {
	#if(RAD_M1)
	int i;
	double lambda, kappa_abs, kappa_emmit, kappa_es, R_dot_ucon[NDIM], Tr, Te;
	#if(P_NUM || COMPTON)
	double exp_xi, kappa_abs_ph, kappa_emmit_ph;
	double Ehat, Nhat, u_dot_urad, u_dot_u;
	#endif
	#if(COMPTON)
	double G0, Theta_e, Theta_r;
	#endif

	//Calculate radiation temperature in rest frame of fluid
	Tr = calc_Tr(ph, ucon, ucon_rad, ucov
		#if(P_NUM)
		, &exp_xi
		#endif
	);
	#if (DOHELM)
	eos_mode_rhou_temp(gpu_eos_table, ph[RHO], ph[UU], &Te);
	#elif(TWO_T)
	Te = calc_Te(ph) * MU_E * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#else
	Te = calc_Te(ph) * MMW * MH_CGS * C_CGS * C_CGS / (BOLTZ_CGS);
	#endif

	//Calculate opacities
	calc_kappa_new(ph, bsq, Tr, Te, &kappa_abs, &kappa_emmit, &kappa_es
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, r
		#endif
		#if(P_NUM)
		, &kappa_abs_ph
		, &kappa_emmit_ph
		, exp_xi
		#endif
	);

	//Calculate emmission rate
	lambda = kappa_emmit * (ARAD / ENERGY_DENSITY_SCALE) * Te * Te * Te * Te; //in units of erg/(Rg/c)/cm^3

	//Calculate non-Compton scattering source term
	for (i = 0; i < NDIM; i++) R_dot_ucon[i] = (mhd_rad[i][0] * ucon[0] + mhd_rad[i][1] * ucon[1] + mhd_rad[i][2] * ucon[2] + mhd_rad[i][3] * ucon[3]);
	for (i = 0; i < NDIM; i++) {
		Gcon[i] = -(kappa_abs * R_dot_ucon[i] + lambda * ucon[i]) - kappa_es * (R_dot_ucon[i] + (R_dot_ucon[0] * ucov[0] + R_dot_ucon[1] * ucov[1] + R_dot_ucon[2] * ucov[2] + R_dot_ucon[3] * ucov[3]) * ucon[i]);
	}

		//Evaluate comptonization term
		//Misc variables-->Merge with calc_Tr
		#if(P_NUM || COMPTON)
		u_dot_urad = ucov[0] * ucon_rad[0] + ucov[1] * ucon_rad[1] + ucov[2] * ucon_rad[2] + ucov[3] * ucon_rad[3];
		u_dot_u = ucon[0] * ucov[0] + ucon[1] * ucov[1] + ucon[2] * ucov[2] + ucon[3] * ucov[3];
		Ehat = ((4. / 3.) * ph[UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * ph[UU_RAD] * u_dot_u);
		#endif

		#if(P_NUM)
		Nhat = -ph[PHOTON] * u_dot_urad;
		
		//Source term for photons
		source_photon[0] = -kappa_abs_ph * Nhat + (kappa_emmit_ph / MASS_DENSITY_SCALE * ARAD * Te * Te * Te * Te / (BOLTZ_CGS * Te * 2.701178));
		#endif

		//Compton scattering term is added
		#if(COMPTON)
		Theta_e = Te * 1.6863687454173171e-10;
		Theta_r = Tr * 1.6863687454173171e-10;
		G0 = -kappa_es * Ehat * 4.0 * (Theta_e - Theta_r) * (1.0 + 3.683 * Theta_e + 4.0 * Theta_e * Theta_e) / ((1.0 + Theta_e));
		for (i = 0; i < NDIM; i++) Gcon[i] += ucon[i] * G0;
		#endif
	#endif
}

//Calculate radiation temperature in rest frame of fluid
__device__ double calc_Tr(double* ph, double ucon[NDIM], double ucon_rad[NDIM], double ucov[NDIM]
	#if(P_NUM)
	, double *exp_xi
	#endif
) {
	double Tr, u_dot_urad, u_dot_u, Ehat;

	u_dot_urad = ucov[0] * ucon_rad[0] + ucov[1] * ucon_rad[1] + ucov[2] * ucon_rad[2] + ucov[3] * ucon_rad[3];
	u_dot_u = ucon[0] * ucov[0] + ucon[1] * ucov[1] + ucon[2] * ucov[2] + ucon[3] * ucov[3];
	Ehat = ENERGY_DENSITY_SCALE * ((4. / 3.) * ph[UU_RAD] * u_dot_urad * u_dot_urad + (1. / 3.) * ph[UU_RAD] * u_dot_u);

	//Get radiation temperature either assuming blackbody or diluted blackbody
	#if(P_NUM)
	double  Nhat;
	Nhat = fabs(-ph[PHOTON] * MASS_DENSITY_SCALE * u_dot_urad);
	//Tr = Ehat / (BOLTZ_CGS * Nhat * (3. - 2.449724 * Nhat * Nhat * Nhat * Nhat / (CK_CGS * Ehat * Ehat * Ehat)));
	//Tr = Ehat / (BOLTZ_CGS * Nhat * (0.33333 + 0.060725 / (0.646756 + 0.121982 * CK_CGS * Ehat * Ehat * Ehat / (Nhat * Nhat * Nhat * Nhat))));
	Tr = Ehat / (BOLTZ_CGS * Nhat * 2.701);
	exp_xi[0] = MY_MIN(1.64676 / (0.646756 + 0.121982 * CK_CGS * Ehat * Ehat * Ehat / (Nhat * Nhat * Nhat * Nhat)), 1.0);
	#else
	Tr = pow(Ehat / ARAD, 0.25);
	#endif

	return Tr;
}

__device__ double calc_Te(double* ph) {
	double Te;

	#if(TWO_T)
		#if(CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Te = exp((GAMMA - 1.0) * ph[ENTRE]) * pow(ph[RHO], GAMMA - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(FIXEDGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Te = exp((GAMMAE - 1.0) * ph[ENTRE]) * pow(ph[RHO], GAMMAE - 1.0);
			#else
			Te = ph[ENTRE] * pow(ph[RHO], GAMMAE - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY)
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * exp(ph[ENTRE]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
			#else
			Te = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * fabs(ph[ENTRE]), 2. / 3.)) - 1.0) / (MU_E * MASS_RATIO);
			#endif
		#endif
	#else
	Te = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Te;
}

__device__ double calc_Ti(double* ph) {
	double Ti;

	#if(TWO_T)
		#if(FIXEDGAMMA || CONSTANTGAMMA)   // fixed gamma: Ressler+15 & Ryan+17
			#if(FULL_ENTROPY)
			Ti = exp((GAMMA - 1.0) * ph[ENTRI]) * pow(ph[RHO], GAMMA - 1.0);
			#else
			Ti = ph[ENTRI] * pow(ph[RHO], GAMMA - 1.0);
			#endif
		#elif(VARGAMMA)     // variable gamma: Sadowski+17 & Chael+19
			#if(FULL_ENTROPY)
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * exp(ph[ENTRI]), 2. / 3.)) - 1.0) / MU_I;
			#else
			Ti = 0.2 * (sqrt(1.0 + 25.0 * pow(ph[RHO] * fabs(ph[ENTRI]), 2. / 3.)) - 1.0) / MU_I;
			#endif
		#endif
	#else
	Ti = (GAMMA - 1.) * ph[UU] / ph[RHO];
	#endif

	return Ti;
}

__device__ void primtoflux_rad(double* pr, struct of_state_rad* q_rad, int dir, struct of_geom* geom, double* flux){
	#if(RAD_M1)
	int k;

	//Radiation energy tensor
	mhd_calc_rad(pr, dir, q_rad, &flux[UU_RAD]);
	for (k = UU_RAD; k <= U3_RAD; k++) flux[k] *= geom->g;

		//Flux of photon number
		#if(P_NUM)
		flux[PHOTON] = pr[PHOTON] * q_rad->ucon[dir];
		flux[PHOTON] *= geom->g;
		#endif
	#endif
	return;
}

__device__ void primtoflux(double *  pr, struct of_state *  q,  int dir, struct of_geom *  geom, double *  flux, double *  vmax, double *  vmin
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if (TWO_T)
	, double gamma_g
	#endif
) {
	int j, k;
	double mhd[NDIM];
	double P, w, bsq, eta, ptot;

	/*Calculate misc quantities*/
    // EOS-specific calls:
    #if (DOHELM)
    double cs2_helm;
    eos_mode_rhou_pres_cs2 (gpu_eos_table, pr[RHO], pr[UU], &P, &cs2_helm);
	#elif(TWO_T)
	P = (gamma_g - 1.) * pr[UU];
    #else
	P = (GAMMA - 1.) * pr[UU];
    #endif

	w = pr[RHO] + P + pr[UU];
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

	//Flux of electron and ion entropies
	#if(TWO_T)
	flux[ENTRE] = flux[RHO] * pr[ENTRE];
	flux[ENTRI] = flux[RHO] * pr[ENTRI];
	#endif

	#if(DOKTOT)
	flux[KTOT] = flux[RHO] * calc_entropy(pr
		#if (DOHELM)
		, gpu_eos_table
		#endif
		#if(TWO_T)
		, gamma_g
		#endif
	);
	#endif

	#pragma unroll 9
	for (k = 0; k < NPR_U;k++) flux[k] *= geom->g;
	#if(TWO_T)
	flux[ENTRE] *= geom->g;
	flux[ENTRI] *= geom->g;
	#endif

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

		/* find fast magnetosonic speed */

        // EOS-specific calls:
        #if (DOHELM)
        // 1. Helmholtz EOS
        // cs2 was already calculated above
        cs2 = cs2_helm;
		#elif(TWO_T)
		cs2 = gamma_g * (gamma_g - 1.) * pr[UU] / w;
        #else
        // 2. Ideal gas EOS
		cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / w;
        #endif

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

//Calculate gas entropy
__device__ double calc_entropy(double* pr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	double entr;
	#if(DOHELM)
	eos_mode_rhou_entr(gpu_eos_table, pr[RHO], pr[UU], &entr);
	entr = xentr;
	//entr = exp(KTOT_FACTOR * entr);
	#elif(TWO_T)
		#if(FIXEDGAMMA || VARGAMMA)
		double Theta;
		//For variable entropy
		Theta = (gamma_g - 1.0) * pr[UU] / pr[RHO] * MU_G;
			#if(FULL_ENTROPY)
			entr = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO]);
			#else
			entr = pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pr[RHO];
			#endif
		#else
		double P = (gamma_g - 1.0) * pr[UU];
			#if(FULL_ENTROPY)
			entr = 1. / (gamma_g - 1.) * log(P * pow(pr[RHO], -gamma_g));
			#else
			entr = P * pow(pr[RHO], -gamma_g);
			#endif
		#endif
	#else 
	double P = (GAMMA - 1.0) * pr[UU];
		#if(FULL_ENTROPY)
		entr = 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
		#else
		entr = P * pow(pr[RHO], -GAMMA);
		#endif
	#endif

	return entr;
}

//Calculate radiative wave velocity
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int dir, double* vmax, double* vmin, double dx
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
) {
	#if(RAD_M1)
	double discr, vp, vm, tau, kappa_abs, kappa_es, kappa_tot, crad2, cmin_rad, cmax_rad, cmin_mhd, cmax_mhd, bsq, Tr, Te;
	double Acon_0, Acon_js;
	double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
	#if(P_NUM)
	double exp_xi, kappa_abs_ph;
	#endif

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

	/* find radiation wave speed at 1./3. speed of light (==isotrpic in radiation frame) */
	crad2 = 1.0 / 3.0;

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

	/* find radiation wave speed in fluid frame based on optical depth */
	//Calculate optical depth
	bsq = q->bcon[0] * q->bcov[0] + q->bcon[1] * q->bcov[1] + q->bcon[2] * q->bcov[2] + q->bcon[3] * q->bcov[3];
	Tr = calc_Tr(pr, q->ucon, q_rad->ucon, q->ucov
		#if(P_NUM)
		,  &exp_xi
		#endif
	);
	Te = calc_Te(pr) * MU_E * MH_CGS * C_CGS * C_CGS / BOLTZ_CGS;

	//Calculate opacities
	calc_kappa_new(pr, bsq, Tr, Te, &kappa_abs, NULL, &kappa_es
		#if(TWO_T)
		, gamma_g
		#endif
		#if(COOL_STOP)
		, Tr //Fake value for r; We do not need to know kappa_emmit
		#endif
		#if(P_NUM)
		, &kappa_abs_ph
		, NULL
		, exp_xi
		#endif
	);
	#if(P_NUM)
	kappa_tot = MY_MIN(kappa_abs, kappa_abs_ph) + kappa_es;
	#else
	kappa_tot = kappa_abs + kappa_es;
	#endif
	tau = kappa_tot * sqrt(geom->gcov[(dir == 1) * 4 + (dir == 2) * 7 + (dir == 3) * 9]) * dx;
	crad2 = 16. / (9. * tau * tau);

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
__device__ void calc_kappa_new(double* ph, double bsq, double Tr, double Te, double *kappa_abs, double *kappa_emmit, double *kappa_es
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double *kappa_abs_ph 
	, double *kappa_emmit_ph
	, double exp_xi
	#endif
) {
	double kappa_m, kappa_h, kappa_chianti, kappa_bf, kappa_ff_abs, kappa_ff_emmit, kappa_HOPAL, kappa_COPAL, kappa_fe, kappa_ff_unity, kappa_sy_abs, kappa_sy_emmit, kappa_dc,  ne, p_theta, scaling_factor;
	double Ree, Rei, Theta_e, Theta_gamma, zeta, nu_mu, phi;
	#if(P_NUM)
	double one_exp_xi, a, b, c, d, e;
	one_exp_xi = 1.0 - exp_xi;
	#endif

	ne = ph[RHO] * MASS_DENSITY_SCALE / (MU_E * MH_CGS);

	#if(OP_EXTRA)
	Theta_e = Te * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
	Theta_gamma = Tr * BOLTZ_CGS / (ME_CGS * C_CGS * C_CGS);
	zeta = Tr / Te;
	nu_mu = 1.5 * E_CGS * sqrt(bsq * 4. * M_PI + 0.00000001 * ph[RHO]) * MAGNETIC_DENSITY_SCALE * Theta_e * Theta_e / (2.0 * M_PI * ME_CGS * C_CGS);

	//Calc free-free absorption opacity
	if (kappa_abs != NULL || kappa_emmit != NULL) {
		if (Theta_e <= 1.0) {
			Rei = 1. + 1.76 * pow(Theta_e, 1.34);
			Ree = 1.7 * Theta_e * (1.0 + 1.1 * Theta_e + Theta_e * Theta_e - 1.06 * pow(Theta_e, 2.5));
		}
		else {
			Rei = 1.4 * sqrt(Theta_e) * (log(1.12 * Theta_e + 0.48) + 1.5);
			Ree = 1.7 * sqrt(Theta_e) * (1.46 * (1.28 + log(1.12 * Theta_e)));
		}
		#if(P_NUM)
		a = 0.188 * pow(exp_xi, 13.9) - 0.2 * pow(one_exp_xi, 0.565) + 0.356;
		b = 0.0722 * pow(exp_xi, 1.36) + 0.255 * pow(one_exp_xi, 0.313) + 3.06;
		c = -1.41 * pow(exp_xi, 3.08) - 1.44 * pow(one_exp_xi, 0.128) + 5.99;
		kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * a * pow(zeta, -b) * log(1.0 + c * zeta);
		kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * log(1.0 + 4.52);
		#else
		kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1.0 - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * pow(zeta, -3.14) * log(1.0 + 4.52 * zeta);
		kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1.0 - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * 0.532 * log(1.0 + 4.52);
		#endif
		scaling_factor = kappa_ff_abs / kappa_ff_emmit;
	}

	//Calculate synchrotron opacities
	#if(P_NUM)
		#if(0)
		//AGN
		if (kappa_abs != NULL){
			a = -0.0295 * pow(exp_xi, 2.29) - 0.143 * pow(one_exp_xi, 0.251) + 0.236;
			b = 0.00977 * pow(exp_xi, 730.0) + 0.0291 * pow(one_exp_xi, 0.48) + 2.58;
			c = 1.29 * pow(exp_xi, 1.59) + 3.46 * pow(one_exp_xi, 0.234) + 2.15;
			d = -78.1 * pow(exp_xi, 66.0) - 40.3 * pow(one_exp_xii, 0.899) + 87.4;
			e = 0.415 * pow(exp_xi, 0.399) + 1.04 * pow(one_exp_xi, 0.252) + 2.68;

			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}

		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		#else
		//XRB
		if (kappa_abs != NULL){
			a = -2.31e-8 * pow(exp_xi, 34.) - 8.24e-9 * pow(one_exp_xi, 2.42) + 1.27;
			b = -0.0261 * pow(exp_xi, 738.0) - 0.00475 * pow(one_exp_xi, 1.55) + 1.06;
			c = 0.000179 * pow(exp_xi, 432.0) + 0.0000411 * pow(one_exp_xi, 0.372) + 0.000584;
			d = -17.7 * pow(exp_xi, 49.4) - 3.33 * pow(one_exp_xi, 2.76) + 18.3;
			e = 0.427 * pow(exp_xi, 0.654) + 1.23 * pow(one_exp_xi, 0.214) + 2.49;

			phi = BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}
		if (kappa_emmit != NULL) {
			phi = BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		#endif
	#else
		#if(0)
		//AGN
		//a = 0.206;
		//b = 2.59;
		//c = 3.44;
		//d = 9.33;
		//e = 3.09;

		if (kappa_abs != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_abs *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		if (kappa_emmit != NULL) {
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.206 * pow(phi, -2.59) * log(1.0 + 3.44 * phi)) + 1.0 / (9.33 * pow(phi, -3.09)));
		}
		#else
		//XRB
		//a = 1.27;
		//b = 1.03;
		//c = 0.000763;
		//d = 0.616;
		//e = 2.91;
		if (kappa_abs != NULL) {
			phi = BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu);
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_abs *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		if (kappa_emmit != NULL) {
			phi = BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (1.27 * pow(phi, -1.03) * log(1.0 + 0.000763 * phi)) + 1.0 / (0.616 * pow(phi, -2.91)));
		}
		#endif
	#endif
	
	//Calculate double compton opacity
	/*
	#if(1)
		#if(P_NUM)
		//Absorption opacity
		a = 6.7 * pow(exp_xi, 0.942) + 4.16 * pow(one_exp_xi, 1.69) + 3.1e-8;
		b = -0.0021 * pow(exp_xi, 0.0217) - 0.0334 * pow(one_exp_xi, 0.469) + 0.042;
		c = -0.18 * pow(exp_xi, 33.0) + 0.201 * pow(one_exp_xi, 0.258) + 3.8;
		d = 0.0169 * pow(exp_xi, 35.4) - 0.0626 * pow(one_exp_xi, 0.35) + 0.118;
		p_theta = pow(1.0 + Theta_e, -3.0);
		kappa_dc_abs = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
		kappa_dc_abs *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
	
		//Emmission opacity
		a = 0.488 * pow(exp_xi, 1.75) - 0.0589 * pow(one_exp_xi, 10.7) + 6.34;
		b = 0.0282 * pow(exp_xi, 1.56) + 0.0142 * pow(one_exp_xi, 0.361) + 0.00875;
		c = -0.16 * pow(exp_xi, 15.4) + 0.184 * pow(one_exp_xi, 0.366) + 3.78;
		d = 0.015 * pow(exp_xi, 26.3) - 0.0256 * pow(one_exp_xi, 0.398) + 0.119;
		p_theta = pow(1.0 + Theta_gamma, -3.0);
		kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
		kappa_dc_emmit *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		#else
		//Absorption opacity
		p_theta = pow(1.0 + Theta_e, -3.0);
		kappa_dc_abs = 7.36e-46 * ne * Tr * Tr * 1.0 * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
		kappa_dc_abs *= 1.0 / ((1.0 / 6.83 + 1.0 / (0.0374 * pow(Theta_gamma, -3.63))) + 1.0 / (0.134 * pow(Theta_gamma, -3.63 / 3.0)));

		//Emmission opacity
		p_theta = pow(1.0 + Theta_gamma, -3.0);
		kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * 1.0 * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
		kappa_dc_emmit *= 1.0 / ((1.0 / 6.83 + 1.0 / (0.0374 * pow(Theta_gamma, -3.63))) + 1.0 / (0.134 * pow(Theta_gamma, -3.63 / 3.0)));
		#endif
	#endif
	*/

	//Calculate molecular opacity
	kappa_m = 3.0 * Z_AB; //No scaling factor

	//Calculate H- opacity
	kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, 7.7);

	//Calculate Chianti opacity
	kappa_chianti = 3.0e34 * ph[RHO] * MASS_DENSITY_SCALE * (0.1 + Z_AB / 0.02) * X_AB * (1 + X_AB) * pow(Te, -4.7);

	//Calculate iron opacity
	kappa_fe = 0.3 * (Z_AB / 0.02) * exp(-6.0 * pow(-12.0 + log(Te), 2.0)); //No scaling factor

	//Calculate bound-free opacity
	kappa_bf = 1.2e24 * 750.0 * Z_AB * (1.0 + X_AB + 0.75 * Y_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Te, -3.5);

	//Calculate COPAL terms conform Mckinney+2017
	kappa_COPAL = 3.0e-13 * kappa_chianti * pow(Te, 1.6) * pow(ph[RHO] * MASS_DENSITY_SCALE, -0.4);

	//Calculate HOPAL terms conform Mckinney+2017
	kappa_HOPAL = 1.0e4 * pow(Te, -1.2) * kappa_h;

	//Calculate total absorption opacity
	if (kappa_abs != NULL) {
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_HOPAL * scaling_factor) + 1.0 / (kappa_COPAL * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_h * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;

		if (!isfinite(kappa_abs[0])) kappa_abs[0] = 0.0;
		else kappa_abs[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;
	}
	if (kappa_emmit != NULL) {
		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_HOPAL) + 1.0 / kappa_COPAL + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;

		if (!isfinite(kappa_emmit[0])) kappa_emmit[0] = 0.0;
		else kappa_emmit[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;
	}

		//Calculate number absorption and emmission opacities
		#if(P_NUM)
		//Calc free-free opacity
		if (kappa_abs_ph != NULL || kappa_emmit_ph != NULL) {
			a = 21.0 * pow(exp_xi, 5.0) - 2.06 * one_exp_xi + 4.0;
			b = -0.412 * pow(exp_xi, 59.1) + 0.000894 * pow(one_exp_xi, 10.2) + 3.15;
			c = 5.27 * pow(exp_xi, 69.2) + 2.39 * pow(one_exp_xi, 0.552);
			kappa_ff_abs = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * a * pow(zeta, -b) * log(1 + c * zeta);
			kappa_ff_emmit = 1.2e24 * (1. + X_AB) * (1. - Z_AB) * (ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, -3.5) * (Rei + Ree) * 25.0 * log(1.0 + 5.27); //Watch out with coefficients
			scaling_factor = kappa_ff_abs / kappa_ff_emmit;
		}

		//Calculate synchrotron opacities
		if (kappa_abs_ph != NULL) {
			#if(0)
			//AGN
			phi = MY_MAX(BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu), 1.0e-5);
			a = 10.8 * pow(exp_xi, 172.0) - 20.4 * pow(one_exp_xi, 0.699) + 29.2;
			b = -0.18 * pow(exp_xi, 31.9) + 0.425 * pow(one_exp_xi, 0.179) + 2.76;
			c = 0.0207 * pow(exp_xi, 9.69) + 0.0506 * pow(one_exp_xi, 0.804) + 0.0314;
			d = 1.51e-6 * pow(exp_xi, 2830.0) - 1.4e-5 * pow(one_exp_xii, 3.06e-12) + 1.4e-5;
			e = 0.1 * pow(exp_xi, 1.95) + 1.57 * pow(one_exp_xi, 0.124);
			#else
			//XRB
			phi = BOLTZ_CGS * Tr / (PLANCK_CGS * nu_mu);
			a = -0.000359 * pow(exp_xi, 1.31) - 0.000552 * pow(one_exp_xi, 0.135) + 0.00209;
			b = 0.035 * pow(exp_xi, 5.43) + 0.0433 * pow(one_exp_xi, 0.159) + 0.948;
			c = -0.122 * pow(exp_xi, 37.1) - 0.0685 * pow(one_exp_xi, 2.8) + 1.04;
			d = -8.59 * pow(exp_xi, 155.0) - 6.47 * pow(one_exp_xi, 0.436) + 8.71;
			e = -0.447 * pow(exp_xi, 394.0) + 0.506 * pow(one_exp_xi, 0.155) + 2.45;
			#endif
			kappa_sy_abs = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Tr) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_abs *= 1.0 / (1.0 / (a * pow(phi, -b) * log(1.0 + c * phi)) + 1.0 / (d * pow(phi, -e)));
		}
		if (kappa_emmit_ph != NULL) {
			#if(0)
			//AGN
			phi = MY_MAX(BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu), 1.0e-5);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (40.0 * pow(phi, -2.58) * log(1.0 + 0.0522 * phi)) + 1.0 / (1.65e6 * pow(phi, -0.1)));
			#else
			//XRB
			phi = BOLTZ_CGS * Te / (PLANCK_CGS * nu_mu);
			kappa_sy_emmit = 5.85374e-14 * ne * phi / (Theta_e * Theta_e * Theta_e * Te) / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_sy_emmit *= 1.0 / (1.0 / (0.00173 * pow(phi, -0.983) * log(1.0 + 0.921 * phi)) + 1.0 / (0.123 * pow(phi, -2.0)));
			#endif
		}

		//Calculate double compton number absorption opacity
		/*
		#if(1)
		if (kappa_abs_ph != NULL) {
			p_theta = pow(1.0 + Theta_e, -3.0);
			a = 29.4 * pow(exp_xi, 285.0) - 76.4 * pow(one_exp_xi, 0.136) + 87.5;
			b = 0.196 * pow(exp_xi, 18.1) - 1.12 * pow(one_exp_xi, 0.134) + 1.16;
			c = -0.8 * pow(exp_xi, 21.6) + 0.0427 * pow(one_exp_xi, 182.0) + 3.93;
			d = 1.87 * pow(exp_xi, 309.0) - 2.72 * pow(one_exp_xi, 0.106) + 2.86;
			kappa_dc = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_dc *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		}

		//Calculate double compton number emmission opacity
		if (kappa_emmit_ph != NULL) {
			p_theta = pow(1.0 + Theta_gamma, -3.0);
			a = -81.7 * pow(exp_xi, 1.01) - 94.8 * pow(one_exp_xi, 0.925) + 198.0;
			b = 1.31 * pow(exp_xi, 1.12) + 1.05 * pow(one_exp_xi, 0.249) + 4.8e-11;
			c = -0.418 * pow(exp_xi, 14.8) + 0.442 * pow(one_exp_xi, 0.361) + 3.44;
			d = 1.37 * pow(exp_xi, 31.3) - 1.38 * pow(one_exp_xi, 0.316) + 3.38;
			kappa_dc_emmit = 7.36e-46 * ne * Tr * Tr * exp_xi * p_theta / (ph[RHO] * MASS_DENSITY_SCALE);
			kappa_dc_emmit *= 1.0 / ((1.0 / a + 1.0 / (b * pow(Theta_gamma, -c))) + 1.0 / (d * pow(Theta_gamma, -c / 3.0)));
		}
		#endif
		*/

		//Calculate total number absorption opacity
		if (kappa_abs_ph != NULL) {
			kappa_abs_ph[0] = 1. / (1. / (kappa_m + kappa_HOPAL * scaling_factor) + 1.0 / (kappa_COPAL * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
			kappa_abs_ph[0] = 1. / (1. / (kappa_m + kappa_h * scaling_factor) + 1. / (kappa_chianti * scaling_factor + kappa_bf * scaling_factor + kappa_ff_abs)) + kappa_sy_abs;
			if (!isfinite(kappa_abs_ph[0])) kappa_abs_ph[0] = 0.0;
			else kappa_abs_ph[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;
		}
		if (kappa_emmit_ph != NULL) {
			kappa_emmit_ph[0] = 1. / (1. / (kappa_m + kappa_HOPAL) + 1.0 / kappa_COPAL + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
			kappa_emmit_ph[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;

			if (!isfinite(kappa_emmit_ph[0])) kappa_emmit_ph[0] = 0.0;
			else kappa_emmit_ph[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;
		}
		#endif
	#else
	kappa_m = 30.0 * 0.1 * Z_AB;
	if (kappa_abs != NULL) {
		zeta = 4. * M_PI * ME_CGS * ME_CGS * ME_CGS * pow(C_CGS, 5.0) * Tr / (3.0 * E_CGS * BOLTZ_CGS * PLANCK_CGS * sqrt(bsq * 4. * M_PI + 0.00000001*ph[RHO]) * MAGNETIC_DENSITY_SCALE * Te * Te);	
		kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, 7.7);
		kappa_chianti = 30.0e33 * ph[RHO] * MASS_DENSITY_SCALE * (0.1 + Z_AB / 0.02) * X_AB * (1.0 + X_AB) * pow(Te, -1.7) * pow(Tr, -3.);
		kappa_bf = 30.0 * 3.0e25 * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te));
		kappa_ff_abs = 30.0 * 4.0e22 * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Te, -0.5) * pow(Tr, -3.0) * log(1. + 1.6 * (Tr / Te)) * (1. + 4.4e-10 * Te);
		kappa_sy_abs = 1.59e-30 * ne * 4. * M_PI * bsq * ENERGY_DENSITY_SCALE * Te * pow(Tr, -3.) / (ph[RHO] * MASS_DENSITY_SCALE); /// (1. + 5.444 * pow(zeta, -0.666666) + 7.218 * pow(zeta, -1.3333333));;
		kappa_abs[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_abs)) + kappa_sy_abs;
		if (!isfinite(kappa_abs[0])) kappa_abs[0] = 0.0;
		else kappa_abs[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;

		#if(P_NUM)
		if (kappa_abs_ph != NULL) {
			kappa_abs_ph[0] = kappa_abs[0] - kappa_sy_abs * ((ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS);
			kappa_sy_abs = 1.59e-30 * ne * 4. * M_PI * bsq * ENERGY_DENSITY_SCALE * Te * pow(Tr, -3.) / (ph[RHO] * MASS_DENSITY_SCALE) * 0.868 * zeta;// / (1.0 + 0.589 * pow(zeta, -1.0 / 3.0) + 0.087 * pow(zeta, -2.0 / 3.0));
			kappa_abs_ph[0] += (kappa_sy_abs * ((ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS));
			if (!isfinite(kappa_abs_ph[0])) kappa_abs_ph[0] = 0.0;

		}
		#endif
	}
	if (kappa_emmit != NULL) {
		zeta = 4. * M_PI * ME_CGS * ME_CGS * ME_CGS * pow(C_CGS, 5.0) * Te / (3.0 * E_CGS * BOLTZ_CGS * PLANCK_CGS * sqrt(bsq * 4. * M_PI + 0.00000001 * ph[RHO]) * MAGNETIC_DENSITY_SCALE * Te * Te);
		kappa_h = 33.0e-25 * sqrt(Z_AB * ph[RHO] * MASS_DENSITY_SCALE) * pow(Te, 7.7);
		kappa_chianti = 30.0e33 * ph[RHO] * MASS_DENSITY_SCALE * (0.1 + Z_AB / 0.02) * X_AB * (1.0 + X_AB) * pow(Te, -4.7);
		kappa_bf = 30.0 * 3.0e25 * Z_AB * (1. + X_AB + 0.75 * Y_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Te, -3.5) * log(1. + 1.6);
		kappa_ff_emmit = 30.0 * 4.0e22 * (1. + X_AB) * (1. - Z_AB) * ph[RHO] * MASS_DENSITY_SCALE * pow(Te, -3.5) * log(1. + 1.6) * (1. + 4.4e-10 * Te);
		kappa_sy_emmit = 1.59e-30 * ne * 4. * M_PI * bsq * ENERGY_DENSITY_SCALE * pow(Te, -2.) / (ph[RHO] * MASS_DENSITY_SCALE);// / (1. + 5.444 * pow(zeta, -0.666666) + 7.218 * pow(zeta, -1.3333333));
		kappa_emmit[0] = 1. / (1. / (kappa_m + kappa_h) + 1. / (kappa_chianti + kappa_bf + kappa_ff_emmit)) + kappa_sy_emmit;
		if (!isfinite(kappa_emmit[0])) kappa_emmit[0] = 0.0;
		else kappa_emmit[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;

		#if(P_NUM)
		if (kappa_emmit_ph != NULL) {
			kappa_emmit_ph[0] = kappa_emmit[0] - kappa_sy_emmit * ((ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS);
			kappa_sy_emmit = 1.59e-30 * ne * 4. * M_PI * bsq * ENERGY_DENSITY_SCALE * pow(Te, -2.) / (ph[RHO] * MASS_DENSITY_SCALE) * 0.868 * zeta;// / (1.0 + 0.589 * pow(zeta, -1.0 / 3.0) + 0.087 * pow(zeta, -2.0 / 3.0));
			kappa_emmit_ph[0] += (kappa_sy_emmit * ((ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS));
			if (!isfinite(kappa_emmit_ph[0])) kappa_emmit_ph[0] = 0.0;
		}
		#endif	
	}
	#endif

	#if(COOL_STOP)
		#if(TWO_T)
		double epsilon = ((gamma_g - 1.) * ph[UU] + 0.3333 * ph[UU_RAD]) / ph[RHO];
		#else
		double epsilon = ((GAMMA - 1.) * ph[UU] + 0.3333 * ph[UU_RAD]) / ph[RHO];
		#endif	
	double om_kepler = 1. / (pow(r, 3. / 2.) + BH_SPIN);
	double T_target = M_PI / 2. * pow(STOP_SCALEHEIGHT * r * om_kepler, 2.);
		#if(TWO_T)
		double Y = (gamma_g - 1.) * epsilon / T_target; 
		#else
		double Y = (GAMMA - 1.) * epsilon / T_target;
		#endif
	if (Y < 1.0) {
		if (kappa_emmit != NULL) kappa_emmit[0] *= pow(Y, 4.0);
		#if(P_NUM)
		if (kappa_emmit_ph != NULL) kappa_emmit_ph[0] *= pow(Y, 4.0);
		#endif
	}
	#endif

	//Calculate electron scattering opacity
	if (kappa_es != NULL) {
		//kappa_es[0] = 0.2 * (1 + X_AB) / (1. + pow(Te / (4.5 * pow(10., 8.)), 0.86));
		kappa_es[0] = 0.2 * (1 + X_AB);
		if (!isfinite(kappa_es[0])) kappa_es[0] = 0.0;
		else kappa_es[0] *= (ph[RHO] * MASS_DENSITY_SCALE) * R_G_CGS;
	}
}

//Calculate total absorption opacity
__device__ double calc_kappa_abs(double* ph, double bsq, double Tr 
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
) {
	return 1.0;
}

//Calculate total absorption opacity
__device__ double calc_kappa_abs_ph(double* ph, double bsq, double Tr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
) {
	return 1.0;
}

//Calculate total emmission opacity
__device__ double calc_kappa_emmit(double* ph, double bsq, double Tr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
) {
	return 1.0;
}

//Calculate total emmission opacity
__device__ double calc_kappa_emmit_ph(double* ph, double bsq, double Tr
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(P_NUM)
	, double exp_xi
	#endif
) {
	return 1.0;
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
		alpha = 1. / sqrt(-geom.gcon[0]);
		beta1 = geom.gcon[1] * alpha * alpha;

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
			vsq_rad = 1. - 1. / (GAMMAMAX_RAD * GAMMAMAX_RAD);
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
	Dqm = (1.0)*(y2 - y1);
	Dqp = (1.0)*(y3 - y2);
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
			for (k = 0; k < NPR; k++) temp[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				for (k = 0; k < NPR; k++) temp[k] += p[k * (ksize)+global_id - zoffset + u] / ((double)zsize);
			}
			for (u = 0; u < zsize; u++){
				for (k = 0; k < NPR; k++) p[k*ksize + global_id - zoffset + u] = temp[k] + (((double)u + 0.5) - 0.5*(double)zsize) / ((double)zsize)*(dq2[k*(ksize)+global_id - zoffset] - dq1[k*(ksize)+global_id - zoffset]);
			}

			temp[0] = 0.0;
			for (u = 0; u < zsize; u++) {
				temp[0] += ps[0 * (ksize)+global_id - zoffset + u] / ((double)zsize);
			}
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

__global__ void fluxcalc2D2(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir, double cour, double*  dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
) {
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
	double cmax_r, cmin_r, cmax, cmin, cmax_l, cmin_l, ctop;
	double temp1[NPR], temp2[NPR], temp3[NPR], temp4[NPR], p[NPR];
	struct of_geom geom;
	#if(RESISTIVE)
	struct of_state_res state;
	#else
	struct of_state state;
	#endif
	#if(RAD_M1)
	double cmax_r_rad, cmin_r_rad, cmax_l_rad, cmin_l_rad, cmax_rad, cmin_rad, ctop_rad;
	struct of_state_rad state_rad;
	#endif
	#if(TWO_T)
	double gamma_g;
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

		#if(RESISTIVE)
		get_state_res(p, &geom, &state);
		primtoflux_res(p, &state, dir, &geom, temp1);
		primtoflux_res(p, &state, 0, &geom, temp2);
		vchar_res(&geom, dir, &cmax_l, &cmin_l);
		#else
		get_state(p, &geom, &state);
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(p);
		#endif
		primtoflux(p, &state, dir, &geom, temp1, &cmax_l, &cmin_l
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		primtoflux(p, &state, 0, &geom, temp2, &cmax_l, &cmin_l
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		#if(RAD_M1)
		get_state_rad(p, &geom, &state_rad);
		primtoflux_rad(p, &state_rad, dir, &geom, temp1);
		primtoflux_rad(p, &state_rad, 0, &geom, temp2);
		vchar_rad(p, &state, &state_rad, &geom, dir, &cmax_l_rad, &cmin_l_rad, factor/cour
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
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

		#if(RESISTIVE)
		get_state_res(p, &geom, &state);
		primtoflux_res(p, &state, dir, &geom, temp3);
		primtoflux_res(p, &state, 0, &geom, temp4);
		vchar_res(&geom, dir, &cmax_r, &cmin_r);
		#else
		get_state(p, &geom, &state);
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(p);
		#endif
		primtoflux(p, &state, dir, &geom, temp3, &cmax_r, &cmin_r
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		primtoflux(p, &state, 0, &geom, temp4, &cmax_r, &cmin_r
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		cmax = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
		cmin = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
		ctop = MY_MAX(cmax, cmin);

		#if(RAD_M1)
		get_state_rad(p, &geom, &state_rad);
		primtoflux_rad(p, &state_rad, dir, &geom, temp3);
		primtoflux_rad(p, &state_rad, 0, &geom, temp4);
		
		vchar_rad(p, &state, &state_rad, &geom, dir, &cmax_r_rad, &cmin_r_rad, factor/cour
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		cmax_rad = fabs(MY_MAX(MY_MAX(0., cmax_l_rad), cmax_r_rad));
		cmin_rad = fabs(MY_MAX(MY_MAX(0., -cmin_l_rad), -cmin_r_rad));
		ctop_rad = MY_MAX(cmax_rad, cmin_rad);

		for (k = 0; k < NPR; k++) {
			if (k == UU_RAD || k == U1_RAD || k == U2_RAD || k == U3_RAD) {
				F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_rad * (temp4[k] - temp2[k]));
			}
			#if(P_NUM)
			else if (k == PHOTON) {
				F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_rad * (temp4[k] - temp2[k]));
			}
			#endif
			else {
				#if(HLLF)
				F[k * (ksize)+global_id] = (cmax * temp1[k] + cmin * temp3[k] - cmax * cmin * (temp4[k] - temp2[k])) / (cmax + cmin + SMALL);
				#else
				F[k * (ksize)+global_id] = (0.5 * (temp1[k] + temp3[k] - ctop * (temp4[k] - temp2[k])));
				#endif
			}
		}
		#else
		for (k = 0; k < NPR; k++) {
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

__device__ void primtoflux_FT(double *pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR]
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
{
	int j, k;
	double  P, w, bsq, eta, ptot;

	/* particle number flux */
	flux[RHO] = pr[RHO] * ucon[dir];

	/* MHD stress tensor, with first index up, second index down */

	// EOS-specific calls:
	#if (DOHELM)
	// 1. Helmholtz EOS
	eos_mode_rhou_pres(gpu_eos_table, pr[RHO], pr[UU], &P);
	w = pr[RHO] + pr[UU] + P;
	#else
	// 2. Ideal gas EOS
	P = (GAMMA - 1.) * pr[UU];
	#endif

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
	#if(DOHELM)
	double xentr;
	eos_mode_rhou_entr(gpu_eos_table, pr[RHO], pr[UU], &xentr);
	flux[KTOT] = flux[RHO] * xentr;
	//flux[KTOT] = flux[RHO] * exp(KTOT_FACTOR * xentr);
	#else 
	#if(FULL_ENTROPY)
	// DIMARK: entropy test
	//double ENTROPY_CONST = 2.5 * (1. - log(MASS_DENSITY_SCALE * avo / MMW)) + 1.5 * log(PRESSURE_SCALE * 2. * M_PI * MH_CGS / (PLANCK_CGS * PLANCK_CGS));
	//flux[KTOT] = flux[RHO] * (1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA)) + ENTROPY_CONST);
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log(P * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * P * pow(pr[RHO], -GAMMA);
	#endif
	#endif
	#endif
}

__device__ void vchar_FT(double * pr, double ucon[NDIM], double bcon[NDIM], int dir, double *vmax, double *vmin
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2;
	double Asq, Bsq, Au, Bu, Au2, Bu2, AuBu, A, B, C;

	/* find fast magnetosonic speed */
	bsq = -bcon[0] * bcon[0] + bcon[1] * bcon[1] + bcon[2] * bcon[2] + bcon[3] * bcon[3];
	
	#if (DOHELM)
	// 1. Helmholtz EOS
	double xpres;
	eos_mode_rhou_pres_cs2(gpu_eos_table, pr[RHO], pr[UU], &xpres, &cs2);
	EF = pr[RHO] + pr[UU] + xpres;
	#else
	// 2. Ideal gas EOS
	#if AMD
	EF = fma(gam, pr[UU], pr[RHO]);
	#else
	EF = pr[RHO] + GAMMA * pr[UU];
	#endif
	/* find fast magnetosonic speed */
	cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / EF;
	#endif
	EE = bsq + EF;
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

	if (k==1){
		for (k = 0; k<NPR; k++){
			pb[k] = pb_i[k*(ksize)+global_id];
		}
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		#if(RESISTIVE)
		double alpha, beta[3], E_cov[3];
		//Lapse in 3+1
		alpha = 1.0 / sqrt(-geom.gcon[0]);

		//Beta in 3+1
		beta[0] = geom.gcon[1] * alpha * alpha;
		beta[1] = geom.gcon[2] * alpha * alpha;
		beta[2] = geom.gcon[3] * alpha * alpha;

		/* dual of Maxwell tensor */
		lower_3(&(pb[E1]), geom.gcov, E_cov);

		//calculate the cell center values of the E-field
		#if(N3G>0)
		E_cent[1 * ksize + global_id] = geom.g * (beta[1] * pb[B3] - beta[2] * pb[B2] + (alpha * alpha / geom.g) * E_cov[0]); //-F2[B3], F3[B2]
		E_cent[2 * ksize + global_id] = geom.g * (beta[2] * pb[B1] - beta[0] * pb[B3] + (alpha * alpha / geom.g) * E_cov[1]); //-F3[B1], F1[B3]
		#endif
		E_cent[3 * ksize + global_id] = geom.g * (beta[0] * pb[B2] - beta[1] * pb[B1] + (alpha * alpha / geom.g) * E_cov[2]); //-F1[B2], F2[B1]
		#else
		struct of_state q;
		ucon_calc(pb, &geom, q.ucon);
		lower(q.ucon, geom.gcov, q.ucov);
		bcon_calc(pb, q.ucon, q.ucov, q.bcon);

		#if(N3G>0)
		E_cent[1 * ksize + global_id] = -geom.g * (q.ucon[2] * q.bcon[3] - q.ucon[3] * q.bcon[2]);
		E_cent[2 * ksize + global_id] = -geom.g * (q.ucon[3] * q.bcon[1] - q.ucon[1] * q.bcon[3]);
		#endif
		E_cent[3 * ksize + global_id] = -geom.g * (q.ucon[1] * q.bcon[2] - q.ucon[2] * q.bcon[1]);
		#endif
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
		#if(RESISTIVE || CARTESIAN)
		double dE_LEFT_13_1 = 0.0;
		double dE_LEFT_13_2 = 0.0;
		double dE_RIGHT_13_1 = 0.0;
		double dE_RIGHT_13_2 = 0.0;
		double dE_LEFT_12_1 = 0.0;
		double dE_LEFT_12_2 = 0.0;
		double dE_RIGHT_12_1 = 0.0;
		double dE_RIGHT_12_2 = 0.0;
		double dE_LEFT_21_1 = 0.0;
		double dE_LEFT_21_2 = 0.0;
		double dE_RIGHT_21_1 = 0.0;
		double dE_RIGHT_21_2 = 0.0;
		double dE_LEFT_23_1 = 0.0;
		double dE_LEFT_23_2 = 0.0;
		double dE_RIGHT_23_1 = 0.0;
		double dE_RIGHT_23_2 = 0.0;
		double dE_LEFT_31_1 = 0.0;
		double dE_LEFT_31_2 = 0.0;
		double dE_RIGHT_31_1 = 0.0;
		double dE_RIGHT_31_2 = 0.0;
		double dE_LEFT_32_1 = 0.0;
		double dE_LEFT_32_2 = 0.0;
		double dE_RIGHT_32_1 = 0.0;
		double dE_RIGHT_32_2 = 0.0;
		#else
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
		#endif

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
	/*int global_id = blockDim.x * blockIdx.x + threadIdx.x;
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
	int zsize0 = 1, zsize1 = 1;
	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel0 = 0;
	int zoffset0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs((jcurr-D2) - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN((jcurr - D2) - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize0 = (int)(0.001 + pow(2.0, (double)zlevel0));
	zoffset0 = (zcurr - N3G) % zsize0;
	int zlevel1 = 0;
	int zoffset1;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize1 = (int)(0.001 + pow(2.0, (double)zlevel1));
	zoffset1 = (zcurr - N3G) % zsize1;
	#endif

	if (k == 1) {
		double dE_LEFT_13_1 = E_cent[1 * (ksize)+global_id] - F3[B2 * (ksize)+global_id];
		double dE_LEFT_13_2 = E_cent[1 * (ksize)+global_id - jsize * D2] - F3[B2 * (ksize)+global_id - jsize * D2];
		double dE_RIGHT_13_1 = F3[B2 * (ksize)+global_id ] - E_cent[1 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_13_2 = F3[B2 * (ksize)+global_id - jsize * D2] - E_cent[1 * (ksize)+global_id - jsize * D2 - D3 * zsize0];
		double dE_LEFT_12_1 = E_cent[1 * (ksize)+global_id] + F2[B3 * (ksize)+global_id];
		double dE_LEFT_12_2 = E_cent[1 * (ksize)+global_id - D3 * zsize1] + F2[B3 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_12_1 = -F2[B3 * (ksize)+global_id] - E_cent[1 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_12_2 = -F2[B3 * (ksize)+global_id - D3 * zsize0] - E_cent[1 * (ksize)+global_id - D2 * jsize - D3 * zsize0];
		double dE_LEFT_21_1 = E_cent[2 * (ksize)+global_id] - F1[B3 * (ksize)+global_id];
		double dE_LEFT_21_2 = E_cent[2 * (ksize)+global_id - D3] - F1[B3 * (ksize)+global_id - D3];
		double dE_RIGHT_21_1 = F1[B3 * (ksize)+global_id] - E_cent[2 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_21_2 = F1[B3 * (ksize)+global_id - D3 * zsize1] - E_cent[2 * (ksize)+global_id - D1 * isize - D3*zsize1];
		double dE_LEFT_23_1 = E_cent[2 * (ksize)+global_id] + F3[B1 * (ksize)+global_id];
		double dE_LEFT_23_2 = E_cent[2 * (ksize)+global_id - D1 * isize] + F3[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_23_1 = -F3[B1 * (ksize)+global_id] - E_cent[2 * (ksize)+global_id - D3 * zsize1];
		double dE_RIGHT_23_2 = -F3[B1 * (ksize)+global_id - isize * D1] - E_cent[2 * (ksize)+global_id - isize * D1 - D3 * zsize1];
		double dE_LEFT_31_1 = E_cent[3 * (ksize)+global_id] + F1[B2 * (ksize)+global_id];
		double dE_LEFT_31_2 = E_cent[3 * (ksize)+global_id - D2 * jsize] + F1[B2 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_31_1 = -F1[B2 * (ksize)+global_id] - E_cent[3 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_31_2 = -F1[B2 * (ksize)+global_id - D2 * jsize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		double dE_LEFT_32_1 = E_cent[3 * (ksize)+global_id] - F2[B1 * (ksize)+global_id];
		double dE_LEFT_32_2 = E_cent[3 * (ksize)+global_id - D1 * isize] - F2[B1 * (ksize)+global_id - D1 * isize];
		double dE_RIGHT_32_1 = F2[B1 * (ksize)+global_id] - E_cent[3 * (ksize)+global_id - D2 * jsize];
		double dE_RIGHT_32_2 = F2[B1 * (ksize)+global_id - D1 * isize] - E_cent[3 * (ksize)+global_id - D1 * isize - D2 * jsize];
		
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

		#if(!CARTESIAN)
		if ((POLE_1 == 1 && jcurr == N2G) || (POLE_2 == 1 && jcurr == BS_2 + N2G)) {
			emf[3 * (ksize)+global_id] = 0.;
			emf[1 * (ksize)+global_id] += -0.5 * 0.5 * (F2[B3 * (ksize)+global_id] + F2[B3 * (ksize)+global_id - D3]);
		}
		#endif
	}*/
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
	#if(N3G>0)
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr;
	#endif
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#if(N3G>0)
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif
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
	#if(N3G>0)
	int index3 = FACE3*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + (zcurr + (k==6));
	#endif
	#else
	int index1 = FACE1*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + (k == 2))*(BS_2 + 2 * N2G) + jcurr;
	int index2 = FACE2*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + (jcurr + (k == 4));
	#if(N3G>0)
	int index3 = FACE3*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_2 + 2 * N2G) + jcurr;
	#endif
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
				#endif
				for (u = 0; u < zsize; u++)psf[global_id + (k == 2)*isize - zoffset + u] = temp / gdet_GPU[index1 - NSY*(zoffset - u)];
			}
		}

		if (icurr >= N1G && jcurr >= N2G + (k == 1 || k == 2) && zcurr >= N3G + D3 * (k != 5) && icurr < BS_1 + N1G && jcurr < BS_2 + N2G - (k == 1 || k == 2) && zcurr < BS_3 + N3G + D3 - (k != 6)){
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
				#if(N3G>0)
				temp += -Dt / ((double)zsize*dx_3)*(E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset + D3*zsize] - E_corn[1 * ksize + global_id + (k == 4)*(BS_3 + 2 * N3G) - zoffset]);
				#endif
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

__global__ void Utoprim_M1_0( double* p_i, double* U_n, double* U_0, double* dU_RAD0, const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
{
	#if(RAD_M1)
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr, k = 0;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	zcurr = (global_id % (isize)) % (BS_3 + 2 * N3G);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3 + 2 * N3G);
	icurr = (global_id - (jcurr * (BS_3 + 2 * N3G)+zcurr)) / (isize);
	if (global_id < (BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) k = 1;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	struct of_geom geom;
	struct of_state q;
	struct of_state_rad q_rad;
	double p[NPR], dU[NPR], U[NPR], UU0[NPR], cell_size;
	int zsize = 1, zoffset = 0, u;
	int pflag_local, pflag_rad_local;
	#if(TWO_T)
	double gamma_g, fel;
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
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(p);
		#endif
		primtoflux(p, &q, 0, &geom, U, NULL, NULL
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		get_state_rad(p, &geom, &q_rad);
		primtoflux_rad(p, &q_rad, 0, &geom, U);

		//Perform implicit solve
		#if(TWO_T)
		fel = calc_delta(p, dot(q.bcon, q.bcov));
		#endif
		cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(p, U, U, UU0, &pflag_local, &pflag_rad_local, &geom, dU, Dt * Y_IMEX, cell_size, y_max
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
			#if(COOL_STOP)
			, radius[icurr]
			#endif
		);

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			U_n[k * (ksize)+global_id] = U[k];
			U_0[k * (ksize)+global_id] = UU0[k];
			dU_RAD0[k * (ksize)+global_id] = dU[k];
			p_i[k * (ksize)+global_id] = p[k];
		}
	}
	#endif
}

__global__ void Utoprim_M1_1(double* ph_i, const  double* __restrict__ p_i, const double* __restrict__ U_n, const double* __restrict__ U_0, double*  U_1, const double* __restrict__ dU_RAD0, double* dU_RAD1, const  double* __restrict__  psh, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
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
	double p[NPR], dU[NPR], UU1[NPR], U_n_tmp[NPR], cell_size;
	int zsize = 1, zoffset = 0, u;
	int pflag_local, pflag_rad_local;
	#if(TWO_T)
	double gamma_g, fel;
	#endif

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
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(p);
		#endif
		source(p, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);

		for (k = 0; k < NPR; k++) {
			UU1[k] = 0.;
			for (u = 0; u < zsize; u++) {
				UU1[k] += (((3.0 * Y_IMEX - 1.0) / Y_IMEX) * U_n[k * (ksize)+global_id - zoffset + u] + ((1.0 - 2.0 * Y_IMEX) / Y_IMEX) * U_0[k * (ksize)+global_id - zoffset + u]) / ((double)zsize);
				#if( N1G > 0 )
				UU1[k] -= Dt * (F1[k * (ksize)+global_id + isize - zoffset + u] - F1[k * (ksize)+global_id - zoffset + u]) / (dx_1 * (double)zsize);
				#endif
				#if( N2G > 0 )
				UU1[k] -= Dt * (F2[k * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k * (ksize)+global_id - zoffset + u]) / (dx_2 * (double)zsize);
				#endif
				UU1[k] += Dt * (dU[k]) / ((double)zsize);
			}
			#if( N3G > 0 )
			UU1[k] -= Dt * (F3[k * (ksize)+global_id - zoffset + zsize] - F3[k * (ksize)+global_id - zoffset]) / (dx_3 * (double)zsize);
			#endif
		}

		#if(NSY)
		UU1[B1] = 0.0;
		UU1[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++) {
			UU1[B1] += (psh[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			UU1[B2] += (psh[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
		}
		#if(N3G>0)
		UU1[B3] = (psh[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset] + psh[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
		#endif
		#endif
		#else
		#if(STAGGERED)
		UU1[B1] = 0.0;
		UU1[B2] = 0.0;
		for (u = 0; u < zsize; u++) {
			UU1[B1] += (psh[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_2 + 2 * N2G) + jcurr]) / (2.0 * (double)zsize);
			UU1[B2] += (psh[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0 * (double)zsize);
		}
		#if(N3G>0)
		UU1[B3] = (psh[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psh[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif
		#endif

		//Set temporary variable
		for (k = 0; k < NPR; k++) U_n_tmp[k] = U_n[k * (ksize)+global_id];

		//Perform implicit solve
		#if(TWO_T)
		fel = calc_delta(p, dot(q.bcon, q.bcov));
		#endif
		cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(p, U_n_tmp, UU1, UU1, &pflag_local, &pflag_rad_local, &geom, dU, Y_IMEX * Dt, cell_size, y_max
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
			#if(COOL_STOP)
			, radius[icurr]
			#endif
		);

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(p, radius[icurr], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
		)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			ph_i[k * (ksize)+global_id] = p[k];
			U_1[k * (ksize)+global_id] = UU1[k];
			dU_RAD1[k * (ksize)+global_id] = dU[k];
		}
	}
	#endif
}

__global__ void Utoprim_M1_2(const  double* __restrict__ ph_i, double* p_i, const double* __restrict__ U_n, const double* __restrict__ U_0, const double* __restrict__ U_1, const double* __restrict__ dU_RAD0, const double* __restrict__ dU_RAD1, const  double* __restrict__  ps, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
)
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
	#if(TWO_T)
	double gamma_g, fel;
	#endif

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
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(ph);
		#endif
		source(ph, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);

		for (k = 0; k < NPR; k++) {
			U_2[k] = 0.;
			for (u = 0; u < zsize; u++) {
				U_2[k] += 0.5*(U_n[k * (ksize)+global_id - zoffset + u] + U_1[k * (ksize)+global_id - zoffset + u] )/ ((double)zsize);
				#if( N1G > 0 )
				U_2[k] -= 0.5 * Dt * (F1[k * (ksize)+global_id + isize - zoffset + u] - F1[k * (ksize)+global_id - zoffset + u]) / (dx_1 * (double)zsize);
				#endif
				#if( N2G > 0 )
				U_2[k] -= 0.5 * Dt * (F2[k * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k * (ksize)+global_id - zoffset + u]) / (dx_2 * (double)zsize);
				#endif
				U_2[k] += Dt * (0.5 * dU[k] + Y_IMEX * dU_RAD0[k * (ksize)+global_id - zoffset + u] + 0.5 * (1.0 - Y_IMEX) * dU_RAD1[k * (ksize)+global_id - zoffset + u]) / ((double)zsize);
			}
			#if( N3G > 0 )
			U_2[k] -= 0.5 * Dt * (F3[k * (ksize)+global_id - zoffset + zsize] - F3[k * (ksize)+global_id - zoffset]) / (dx_3 * (double)zsize);
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

		#if(TWO_T)
		fel = calc_delta(ph, dot(q.bcon, q.bcov));
		#endif

		#if(NEWMAN)
		pflag[global_id] = Utoprim_NM(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
		);
		#else
		pflag[global_id] =  Utoprim_2d(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC
		#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
		);
		#endif

		#if(DO_FONT_FIX) 
		if (pflag[global_id]) {
			failimage[global_id]++;
			pflag[global_id] = Utoprim_1dvsq2fix1(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC, FULL_ENTROPY
				#if (DOHELM)
				,gpu_eos_table
				#endif
				#if(TWO_T)
				, fel
				#endif
			);			
			if (pflag[global_id]) {
				failimage[1 * (ksize)+global_id]++;
				pflag[global_id] = Utoprim_1dfix1(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC, FULL_ENTROPY
					#if(DOHELM==10)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, fel
					#endif
				);
				if (pflag[global_id]) {
					pflag[0] = global_id;
					failimage[2 * (ksize)+global_id]++;
				}
			}
		}
		#endif
		pflag_rad[global_id] = Rtoprim(U_2, geom.gcov, geom.gcon, geom.g, ph, y_max, BASIC);

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(ph, radius[icurr], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
		)) {
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
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2, double y_max
	#if (DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
)
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
	double pf[NPR], dU[NPR], U[NPR];
	#if(RESISTIVE)
	struct of_state_res q;
	#else
	struct of_state q;
	#endif
	#if(TWO_T)
	double gamma_g, fel;
	#endif
	int zsize = 1, zoffset = 0, u;

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		if (full_step == 0) {
			for (k = 0; k < NPR; k++) {
				pf[k] = 0.0;
				for (u = 0; u < zsize; u++) {
					pf[k] += (1.0 / ((double)zsize)) * pi_i[k * (ksize)+global_id - zoffset + u];
				}
			}

			#if(TWO_T)
			gamma_g = calc_gamma_gas_prim(pf);
			#endif
			#if(RESISTIVE)
			get_state_res(pf, &geom, &q);
			primtoflux_res(pf, &q, 0, &geom, U);
			#else
			get_state(pf, &geom, &q);
			primtoflux(pf, &q, 0, &geom, U, NULL, NULL
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, gamma_g
				#endif
			);
			#endif

			#if(RAD_M1)
			struct of_state_rad q_rad;
			get_state_rad(pf, &geom, &q_rad);
			primtoflux_rad(pf, &q_rad, 0, &geom, U);
			#endif
			#pragma unroll 9	
			for (k = 0; k < NPR; k++) {
				storage2[k * (ksize)+global_id] = U[k];
			}
		}
		else {
			#pragma unroll 9
			for (k = 0; k < NPR; k++) {
				U[k] = storage2[k * (ksize)+global_id];
			}
			for (k = 0; k < NPR; k++) {
				pf[k] = 0.0;
				for (u = 0; u < zsize; u++) {
					pf[k] += (1.0 / ((double)zsize)) * pb_i[k * (ksize)+global_id - zoffset + u];
				}
			}
			#if(TWO_T)
			gamma_g = calc_gamma_gas_prim(pf);
			#endif
			#if(RESISTIVE)
			get_state_res(pf, &geom, &q);
			#else
			get_state(pf, &geom, &q);
			#endif
		}

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			for (u = 0; u < zsize; u++) {
				#if( N1G > 0 )
				U[k] -= Dt * (F1[k * (ksize)+global_id + isize - zoffset + u] - F1[k * (ksize)+global_id - zoffset + u]) / (dx_1 * (double)zsize);
				#endif
				#if( N2G > 0 )
				U[k] -= Dt * (F2[k * (ksize)+global_id + (BS_3 + 2 * N3G) - zoffset + u] - F2[k * (ksize)+global_id - zoffset + u]) / (dx_2 * (double)zsize);
				#endif 
			}
			#if( N3G > 0 )
			U[k] -= Dt * (F3[k * (ksize)+global_id - zoffset + zsize] - F3[k * (ksize)+global_id - zoffset]) / (dx_3 * (double)zsize);
			#endif
		}

		#if(RESISTIVE)
		double q_charge;
		q_charge = divE_calc(pb_i, gdet, dx_1, dx_2, dx_3, icurr, jcurr, zcurr);
		source_res(pf, &geom, icurr, jcurr, zcurr, dU, &q_charge, Dt, conn, &q, radius[icurr]);
		#else
		source(pf, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr]
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		#pragma unroll 9
		for (k = 0; k < NPR; k++) {
			U[k] += Dt * (dU[k]);
		}

		#if(NSY)
		U[B1] = 0.0;
		U[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++) {
			U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
			U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / 2.0;
		}
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + (zcurr - zoffset + zsize * D3)]) / 2.0;
		#endif
		#endif
		#else
		#if(STAGGERED)
		U[B1] = 0.0;
		U[B2] = 0.0;
		for (u = 0; u < zsize; u++) {
			U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_2 + 2 * N2G) + jcurr]) / (2.0 * (double)zsize);
			U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + (jcurr + D2)]) / (2.0 * (double)zsize);
		}
		#if(N3G>0)
		U[B3] = (psf[2 * ksize + global_id - zoffset] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr] + psf[2 * ksize + global_id - zoffset + zsize * D3] * gdet[FACE3 * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_2 + 2 * N2G) + jcurr]) / 2.0;
		#endif
		#endif
		#endif

		#if(TWO_T)
		fel = calc_delta(pf, dot(q.bcon, q.bcov));
		#endif

		#if(RAD_M1)
		double U_0[NPR];
		int pflag_local, pflag_rad_local;
		PLOOP dU[k] = 0.;

		//Perform implicit solve
		double cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(pf, U, U, U_0, &pflag_local, &pflag_rad_local, &geom, dU, Dt, cell_size, y_max
			#if(TWO_T)
			, fel
			#endif
			#if(COOL_STOP)
			, radius[icurr]
			#endif
		);
		#else
			#if(RESISTIVE)
			pflag[global_id] = Utoprim_3d_res(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, Dt);
			#else
				#if(NEWMAN)
					pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, fel
					#endif
				);
				#else
				pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, fel
				#endif
				);
				#endif
				if (pflag[global_id]) {
					failimage[global_id]++;
					pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, fel
						#endif
					);
					if (pflag[global_id] && !TWO_T) {
						failimage[1 * (ksize)+global_id]++;
						#if(!DOHELM)
						pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, fel
						#endif
						);
						#endif
						if (pflag[global_id]){
							pflag[0] = global_id;
							failimage[2 * (ksize)+global_id]++;
						}
					}
				}
			#endif
		#endif

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(pf, radius[icurr], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
		)) {
			pflag[global_id] = -333;
			pflag[0] = global_id;
			failimage[3 * (ksize)+global_id]++;
		}

		#pragma unroll 9	
		for (k = 0; k< NPR; k++){
			pf_i[k*(ksize)+global_id] = pf[k];
		}
	}
}


__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2
	#if (DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
)
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
	#if(TWO_T)
	double gamma_g, fel;
	#endif

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
			#if(TWO_T)
			gamma_g = calc_gamma_gas_prim(pf);
			#endif
			primtoflux(pf, &q, 0, &geom, U, NULL, NULL
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, gamma_g
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

			#if(TWO_T)
			fel = calc_delta(pf, dot(q.bcon, q.bcov));
			#endif

			#if(NEWMAN)
			pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, fel
				#endif
			);
			#else
			pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(TWO_T)
				, fel
				#endif
			);
			#endif
			//compute the square of fluid frame magnetic field (twice magnetic pressure)
			#if( DO_FONT_FIX )
			if (pflag[global_id]) {
				failimage[global_id]++;
				#if DOKTOT
				pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, fel
					#endif
				);
				#endif
				if (pflag[global_id]) {
					failimage[1 * (ksize)+global_id]++;
					#if(!DOHELM)
					pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, fel
						#endif
					);
					#endif
					if (pflag[global_id]){
						pflag[0] = global_id;
						failimage[2 * (ksize)+global_id]++;
					}
				}
			}
			#endif

			//Apply floors in ZAMO frame or drift frame
			if (fixup_cell(pf, radius[icurr], &geom
				#if (DOHELM)
				, gpu_eos_table
				#endif
			)){
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
			pv[UU_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[UU_RAD * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[U1_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[U1_RAD * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[U3_RAD * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[U3_RAD * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(TWO_T)
			pv[ENTRE * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[ENTRE * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[ENTRI * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[ENTRI * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(P_NUM)
			pv[PHOTON * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[PHOTON * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
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
			#if(RESISTIVE)
			pv[E2 * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
			#endif
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
			pv[UU_RAD * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[UU_RAD * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[U1_RAD * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[U1_RAD * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[U3_RAD * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[U3_RAD * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(TWO_T)
			pv[ENTRE * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[ENTRE * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[ENTRI * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[ENTRI * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(P_NUM)
			pv[PHOTON * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[PHOTON * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
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
			#if(RESISTIVE)
			pv[E2 * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
			#endif
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
			#if(RESISTIVE)
			pv[E2 * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[E3 * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			#endif

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
			#if(RESISTIVE)
			pv[E2 * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			pv[E3 * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
			#endif

			#if(STAGGERED)
			ps[0 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			ps[2 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = -ps[2 * (ksize)+isize*icurr + (2 * BS_2 - j - 1 + N2G)*(BS_3 + 2 * N3G) + (zcurr - N3G + BS_3 / 2) % BS_3 + N3G];
			#endif
		}
	}
}

__global__ void fluxcalc2D_FT(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet,
	const  double* __restrict__ Mud, const  double* __restrict__ Mud_inv, int lim, int dir, double cour, double*  dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag
	#if(DOHELM)
	, const double* __restrict__ gpu_eos_table
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

		vchar(p_l, &state_l, &geom, dir, &(cmax_l), &(cmin_l)
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		vchar(p_r, &state_r, &geom, dir, &(cmax_r), &(cmin_r)
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

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

		primtoflux_FT(p_l, l_ucon, l_bcon, dir, F_l
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		primtoflux_FT(p_r, r_ucon, r_bcon, dir, F_r
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		primtoflux_FT(p_l, l_ucon, l_bcon, 0, U_l
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		primtoflux_FT(p_r, r_ucon, r_bcon, 0, U_r
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

		vchar_FT(p_l, l_ucon, l_bcon, dir, &(cmax_l), &(cmin_l)
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);
		vchar_FT(p_r, r_ucon, r_bcon, dir, &(cmax_r), &(cmin_r)
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

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


// EOS function calls
#if (DOHELM)

#if (EOS_LINEAR)
__device__ void interp_eostable_linear(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double* free, double* df_d, double* df_t, double* df_tt, double* df_dt, double* dpepdd, double* etaele) {
	int iat, jat;
	double xt, xd, mxt, mxd;
	int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;

	//  hash locate this temperature and density
	jat = (int)((log10(btemp) - eos_tlo) * (double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
	jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
	iat = (int)((log10(din) - eos_dlo) * (double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
	iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

	double tstp = (eos_thi - eos_tlo) / (double)(EOSJMAX - 1);
	double dstp = (eos_dhi - eos_dlo) / (double)(EOSIMAX - 1);
	double eos_t_jat = pow(10.0, (eos_tlo + jat * tstp));
	double eos_d_iat = pow(10.0, (eos_dlo + iat * dstp));
	double eos_dt_jat = pow(10.0, (eos_tlo + (jat + 1) * tstp)) - pow(10.0, (eos_tlo + jat * tstp));
	double eos_dd_iat = pow(10.0, (eos_dlo + (iat + 1) * dstp)) - pow(10.0, (eos_dlo + iat * dstp));

	//  various differences
	xt = MY_MAX((btemp - eos_t_jat) / eos_dt_jat, 0.0); 
	xd = MY_MAX((din - eos_d_iat) / eos_dd_iat, 0.0); 
	mxt = 1.0 - xt;
	mxd = 1.0 - xd;

	// the free energy
	*free = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
			gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
			gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
			gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	// derivative with respect to density
	*df_d = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
			gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
			gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
			gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	// derivative with respect to temperature
	*df_t = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
			gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
			gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
			gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	// second derivative with respect to temperature
	*df_tt =	gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
				gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
				gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
				gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	//  second derivative with respect to temperature and density
	*df_dt =	gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
				gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
				gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
				gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	// now get the pressure derivative with density, chemical potential, and
	// electron positron number densities
	// get the interpolation weight functions

	//  pressure derivative with density
	*dpepdd =	gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
				gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
				gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
				gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;

	*dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

	//  electron chemical potential etaele
	*etaele =	gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat)] * mxt * mxd +
				gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * mxt * xd +
				gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat)*EOSJMAX + (jat + 1)] * xt * mxd +
				gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * xt * xd;
}

#else

__device__ void interp_eostable(const  double* __restrict__ gpu_eos_table, double den, double btemp, double din, double ye, double *free, double *df_d, double *df_t, double *df_tt, double *df_dt, double *dpepdd, double *etaele) {
    int iat, jat;
    double fi[36];
    double xt, xd, mxt, mxd;
    double si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md;
    double dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt;
	int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;

	//  hash locate this temperature and density
	jat = (int)((log10(btemp) - eos_tlo) * (double)(EOSJMAX - 1) / (eos_thi - eos_tlo)) + 1;
	jat = MY_MAX(1, MY_MIN(jat, EOSJMAX - 1)) - 1;
	iat = (int)((log10(din) - eos_dlo) * (double)(EOSIMAX - 1) / (eos_dhi - eos_dlo)) + 1;
	iat = MY_MAX(1, MY_MIN(iat, EOSIMAX - 1)) - 1;

    //  access the table locations only once
    fi[0] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[1] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[2] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[3] = gpu_eos_table[0 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[4] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[5] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[6] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[7] = gpu_eos_table[2 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[8] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[9] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[10] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[11] = gpu_eos_table[4 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[12] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat) ];
    fi[13] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[14] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[15] = gpu_eos_table[1 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[16] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[17] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[18] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[19] = gpu_eos_table[3 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[20] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[21] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[22] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[23] = gpu_eos_table[5 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[24] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[25] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[26] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[27] = gpu_eos_table[6 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[28] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[29] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[30] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[31] = gpu_eos_table[7 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

    fi[32] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)];
    fi[33] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)];
    fi[34] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)];
    fi[35] = gpu_eos_table[8 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)];

	double tstp = (eos_thi - eos_tlo) / (double)(EOSJMAX - 1);
	double dstp = (eos_dhi - eos_dlo) / (double)(EOSIMAX - 1);
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
    *dpepdd =   gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si0t +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si0t +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si0mt +
                gpu_eos_table[9 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si0mt +

                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si1t +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si1t +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si1mt +
                gpu_eos_table[11 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si1mt +

                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si0t +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si0t +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si0mt +
                gpu_eos_table[10 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si0mt +

                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si1t +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si1t +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si1mt +
                gpu_eos_table[12 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si1mt;

    // h3dpd(iat,jat,
    //      si0t,   si1t,   si0mt,   si1mt,
    //      si0d,   si1d,   si0md,   si1md,
    //      eos_dpdf, eos_dpdft, eos_dpdfd, eos_dpdfdt);

    *dpepdd = MY_MAX(ye * (*dpepdd), 0.0);

    //  electron chemical potential etaele
    *etaele =   gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si0t +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si0t +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si0mt +
                gpu_eos_table[13 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si0mt +

                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si0d * si1t +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si0md * si1t +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si0d * si1mt +
                gpu_eos_table[15 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si0md * si1mt +

                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si0t +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si0t +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si0mt +
                gpu_eos_table[14 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si0mt +

                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat)] * si1d * si1t +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat)] * si1md * si1t +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat) * EOSJMAX + (jat + 1)] * si1d * si1mt +
                gpu_eos_table[16 * (EOSIMAX * EOSJMAX + eos_offset) + (iat + 1) * EOSJMAX + (jat + 1)] * si1md * si1mt;

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
#endif


__device__ void eos_helm (const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double abar, double zbar, double *pres, double *ener, double* entr, double *dpresdt, double *denerdt, double *dentrdt, double *dpresdd, double *denerdd, double *cs2)
{
    // Local variables
	double prad, dpraddt, erad, deraddt, srad, dsraddt;
	double pion, dpiondt, eion, deiondt, sion, dsiondt;
	double pele, dpepdt, eele, deepdt, sele, dsepdt;

	// Danat: out of all derivatives w.r.t. density we only need dpdrho so far; commented out the others for the sake of optimizing the code
	double dpraddd, dpiondd, dpepdd;
	double deraddd, deiondd, deepdd;
	double dsepdd; //dentrdd, dsraddd, dsiondd, ;

    // For the coulomb corrections
#if (EOS_COULOMB_CORR)
	double x4, x5, y1, y2, y3, z4, z5;
	double ktinv, dxnidd, dsdd, lami, inv_lami, lamidd, s1, s2, s3, plasg, plasg_inv, plasgdd, plasgdt;
	double ecoul, decouldd, decouldt, pcoul, dpcouldd, dpcouldt, scoul, dscouldd, dscouldt;
#endif

    // Convert from code units to cgs units (EOS table units)
    btemp *= conv_T_CODE2CGS;
    den *= conv_dens_CODE2CGS;

	// If density is below the minimum supplied by the table:
	double den_low = den;
	int is_density_low = 0;
	if (den < eos_dens_low) {
		den = eos_dens_low;
		is_density_low = 1;
	}

	double deni = 1.0 / den;
	double tempi = 1.0 / btemp;

    // Useful relations
    double ytot1 = 1.0 / abar;
    double ye = ytot1 * zbar;
    double kt = kerg * btemp;
    double din = zbar * ytot1 * den;
	double kavoy = kergavo * ytot1;

    //Look up the desired quantities in the eos table
	double free, df_d, df_t, df_tt, df_dt, etaele;
#if (EOS_LINEAR)
	interp_eostable_linear(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, &etaele);
#else
    interp_eostable(gpu_eos_table, den, btemp, din, ye, &free, &df_d, &df_t, &df_tt, &df_dt, &dpepdd, &etaele);
#endif

    // the desired electron-positron thermodynamic quantities
    pele = din * din * df_d;
    sele = -df_t * ye;
    eele = ye * free + btemp * sele;

	// ion portion of the gas:
	double xni = avo * ytot1 * den;
    pion = xni * kt; 
	eion = 1.5 * pion * deni;
	sion = (pion * deni + eion) * tempi + kavoy * log(pow(abar, 2.5) * deni * avoinv * pow(sioncon * btemp, 1.5));

    // uniform background corrections & only the needed parts for speed
    // plasg is the plasma coupling parameter
    // split up calculations below -- they all used to depend upon a redefined z
#if (EOS_COULOMB_CORR)
	s1 = 4.0 / 3.0 * M_PI * xni;
	lami = 1.0 / pow(s1, third);
	inv_lami = 1.0 / lami;
	dxnidd = avo * ytot1;
	ktinv = 1.0 / kt;
	plasg = zbar * zbar * esqu * ktinv * inv_lami;
    if (plasg >= 1.0) {
		// yakovlev & shalybkov 1989 equations 82, 85, 86, 87
		x4 = pow(plasg, 0.25);
        z4 = eos_c1 / x4;
        ecoul = dxnidd * kt * (eos_a1*plasg + eos_b1*x4 + z4 + d1cc);
        pcoul = third * den * ecoul;
        scoul = -kavoy*(3.0*eos_b1*x4 - 5.0*z4 + d1cc*(log(plasg) - 1.0) - e1cc);
	}
    else if (plasg < 1.0) {
		// yakovlev & shalybkov 1989 equations 102, 103, 104
		x5 = plasg * sqrt(plasg);
        y3 = pow(plasg, eos_b2);
        z5 = eos_c2 * x5 - third * eos_a2 * y3;
		pcoul = -pion * z5;
        ecoul = 3.0 * pcoul * deni;
        scoul = -kavoy*(eos_c2*x5 - eos_a2*(eos_b2 - 1.0) / eos_b2*y3);
	}
#endif

    // radiation section:
	#if (RAD_M1)
	prad = erad = srad = 0.;
	#else
    prad = asoli3 * btemp * btemp * btemp * btemp;
    double x1 = prad * deni;
    erad = 3.0 * x1;
    srad = (x1 + erad)*tempi;
	#endif

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
#if (EOS_COULOMB_CORR)
    *pres = prad + pion + pele + pcoul * eos_coulombMult;
    *ener = erad + eion + eele + ecoul * eos_coulombMult;
	*entr = srad + sion + sele + scoul * eos_coulombMult;
#else
	*pres = prad + pion + pele;
	*ener = erad + eion + eele;
	*entr = srad + sion + sele;
#endif



    if (calc_derivatives) {
        // Calculate pressure derivatives
        dpraddt = 4.0 * prad * tempi;
        dpraddd = 0.0;

        dpiondd = avo * ytot1 * kt;
        dpiondt = xni * kerg;

        dpepdt = din * din * df_dt;

#if (EOS_COULOMB_CORR)
		plasg_inv = 1.0 / plasg;
        if (plasg >= 1.0) {
			// yakovlev & shalybkov 1989 equations 82, 85, 86, 87
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
        else if (plasg < 1.0) {
			// yakovlev & shalybkov 1989 equations 102, 103, 104
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
#endif

#if (EOS_COULOMB_CORR)
        *dpresdd = dpraddd + dpiondd + dpepdd + dpcouldd * eos_coulombMult; // pressure derivative vs density
        *dpresdt = dpraddt + dpiondt + dpepdt + dpcouldt * eos_coulombMult; // pressure derivative vs temperature
#else
		*dpresdd = dpraddd + dpiondd + dpepdd; // pressure derivative vs density
		*dpresdt = dpraddt + dpiondt + dpepdt; // pressure derivative vs temperature
#endif
        // Calculate energy derivatives
        deiondd = (1.5 * dpiondd - eion)*deni;
        deiondt = 1.5 * xni * kerg *deni;
		deraddd = -erad*deni;
        deraddt = 4.0 * erad * tempi;
        
		dsepdt = -df_tt * ye;
        dsepdd = -df_dt * ye * ye;
        deepdt = btemp * dsepdt;
        deepdd = ye*ye*df_d + btemp*dsepdd;

#if (EOS_COULOMB_CORR)
        *denerdd = deraddd + deiondd + deepdd + decouldd * eos_coulombMult;  // energy derivative vs density
        *denerdt = deraddt + deiondt + deepdt + decouldt * eos_coulombMult; // energy derivative vs temperature
#else 
		*denerdd = deraddd + deiondd + deepdd;  // energy derivative vs density
		*denerdt = deraddt + deiondt + deepdt; // energy derivative vs temperature
#endif

        // Calculate entropy derivatives
        //dsraddd = (dpraddd*deni - x1*deni + deraddd)*tempi;
        dsraddt = (dpraddt*deni + deraddt - srad)*tempi;
        //dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - kavoy * deni;
        dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion) * tempi*tempi + 1.5 * kavoy * tempi;

#if (EOS_COULOMB_CORR)
        if (plasg >= 1.0) {
			// yakovlev & shalybkov 1989 equations 82, 85, 86, 87
			y2 = -kavoy*plasg_inv*(0.75*eos_b1*x4 + 1.25*z4 + d1cc);
            dscouldd = y2 * plasgdd;
            dscouldt = y2 * plasgdt;
        }
        else if (plasg < 1.0) {
			// yakovlev & shalybkov 1989 equations 102, 103, 104
			s3 = -kavoy*plasg_inv*(1.5*eos_c2*x5 - eos_a2*(eos_b2 - 1.0)*y3);
			dscouldd = s3 * plasgdd;
            dscouldt = s3 * plasgdt;
        }
#endif

#if (EOS_COULOMB_CORR)
		//dentrdd = dsraddd + dsiondd + dsepdd + dscouldd * eos_coulombMult; // entropy derivative vs density and density
        *dentrdt = dsraddt + dsiondt + dsepdt + dscouldt * eos_coulombMult; // entropy derivative vs density and time
#else
		//dentrdd = dsraddd + dsiondd + dsepdd; // entropy derivative vs density and density
		*dentrdt = dsraddt + dsiondt + dsepdt; // entropy derivative vs density and time
#endif 
        // calculate relativistic soundspeeds
        double chit, z;
        chit = btemp / (*pres) * (*dpresdt);
        z = 1.0 + ((*ener) + (c_light * c_light)) * den / (*pres);
        *cs2 = (chit * chit * (*pres) * deni  * tempi / (*denerdt) + (*dpresdd) * den / (*pres)) / z; // already in the units of the code (c = 1)
    }

	double density_factor = den / den_low;
	if (is_density_low) {
		*pres *= density_factor;
		//*ener unchanged;
		*entr *= density_factor;
		*dpresdt *= density_factor;
		//*denerdt unchanged;
		*dentrdt *= density_factor;
		//*dpresdd unchanged
		//*denerdd unchanged or = 0, I don't know yet
		//*cs2 unchanged
		// for now
	}

    // Convert from cgs to code units
    *pres *= conv_pres_CGS2CODE;
    *ener *= conv_ener_CGS2CODE;
	*entr *= conv_entr_CGS2CODE;

    *dpresdt *= conv_pres_CGS2CODE * conv_T_CODE2CGS;
    *denerdt *= conv_ener_CGS2CODE * conv_T_CODE2CGS;
	*dentrdt *= conv_entr_CGS2CODE;
    *dpresdd *= conv_pres_CGS2CODE * conv_dens_CODE2CGS;
	*denerdd *= conv_ener_CGS2CODE * conv_dens_CODE2CGS;

    return;
}

__device__ void validate_T(double* temp);
__device__ void eos_NR_temp_guess(double rho, double u, double* temp);

__device__ void validate_T(double* temp) {
	if (*temp < eos_temp_low) *temp = eos_temp_low;
	if (*temp > eos_temp_up) *temp = eos_temp_up;
	return;
}

__device__ int eos_check_input_u(double rho, double u) {
	if (u <= 0.) return 1;
	else return 0;
}

__device__ void eos_NR_temp_guess(double rho, double u, double* temp) {
	double gam = 5. / 3.;

	if (u < 0. || rho < 0.) {
		*temp = eos_temp_low;
		return;
	}

	#if (RAD_M1)
	//*temp = fabs(MMW * MH_CGS * (gam - 1.) * (u * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * rho * MASS_DENSITY_SCALE));
	*temp = pow(u * PRESSURE_SCALE / ARAD, 0.25);
	#else
	*temp = pow(u * PRESSURE_SCALE / ARAD, 0.25);
	#endif

	validate_T(temp);
	return;
}

// __device__ double eos_newton_raphson(const double* __restrict__ gpu_eos_table, int mode, double den, double temp_ini, double q_goal, double *pres, double *ener, double *entr, double *cs2, double *dpdt, double *dedt, double *dsdt, double *dpdd, double *dedd) {
// 	// double pres, ener, entr, dpdt, dedt, dsdt, dpdd, dedd, cs2;
// 	double enth, dhdt;
// 	double temp_old, temp_new;
// 	double errT, errQ;
// 
// 	int more_iterations = 2; // number of additional iterations, if reached desired tolerance
// 	temp_old = temp_ini;
// 
// 	for (int i = 0; i < EOS_ITERATIONS; i++) {
// 		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, ener, entr, dpdt, dedt, dsdt, dpdd, dedd, cs2);
// 
// 		if (mode == EOS_NR_ENER) {
// 			temp_new = temp_old - (*ener - q_goal) / *dedt;
// 			errQ = fabs((*ener - q_goal) / q_goal);
// 		}
// 		else if (mode == EOS_NR_PRES) {
// 			temp_new = temp_old - (*pres - q_goal) / *dpdt;
// 			errQ = fabs((*pres - q_goal) / q_goal);
// 		}
// 		else if (mode == EOS_NR_ENTR) {
// 			temp_new = temp_old - (*entr - q_goal) / *dsdt;
// 			errQ = fabs((*entr - q_goal) / q_goal);
// 		}
// 		else if (mode == EOS_NR_ENTH) {
// 			enth = *ener + *pres / den;
// 			dhdt = *dedt + *dpdt / den;
// 			temp_new = temp_old - (*enth - q_goal) / *dhdt;
// 			errQ = fabs((*enth - q_goal) / q_goal);
// 		}
// 		else
// 			printf("EOS NEWTON RAPHSON: WRONG MODE CHOSEN!\n");
// 
// 		//do not allow temp to change more than 10. times in one iteration
// 		if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
// 		if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;
// 
// 		errT = fabs((temp_new - temp_old) / temp_old);
// 		validate_T(&temp_new);
// 
// 		temp_old = temp_new;
// 
// 		// more iterations after reached below tolerance
// 		if (errT < EOS_TEMP_TOL && errQ < EOS_TOL) {
// 			more_iterations -= 1;
// 			if (more_iterations == 0) break;
// 		}
// 	}
// 
// 	return errQ;
// }

__device__ void eos_mode_rhou_pres (const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres) {	
	// initial guess : temperature
    double temp_ini_guess;
	eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

    double ener_goal = u_goal / den;
    double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
	double entr, dsdt, dedrho;
    double cs2;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
        eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
		validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
        if(error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double fA, fB, fC;
	int flag = 1;

	if (error_e > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fA = enerA - ener_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fB = enerB - ener_goal;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
			fC = enerC - ener_goal;
			error_e = fabs(fC / ener_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}

	if (error_e > EOS_TOL) {
		// Use GAMMA EOS in this case
		*pres = (GAMMA - 1.0) * u_goal;
		error_e = 9.99e-12;
	}
	

	if (error_e > EOS_TOL) printf("1 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);

}

__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double *pres, double *cs2) {
	// initial guess : temperature
    double temp_ini_guess;
	eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

	double ener_goal = u_goal / den;
	double temp_new, temp_old;
    double ener_tmp;
    double dpdt, dedt, dpdrho;
	double entr, dsdt, dedrho;

    double error, error_e;
    int i;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2);
        temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

        //do not allow temp to change more than 2. times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_e = fabs((ener_tmp - ener_goal) / ener_goal);
		validate_T(&temp_new);

        temp_old = temp_new;

        // more iterations after reached below tolerance
		if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double fA, fB, fC;
	int flag = 1;

	if (error_e > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2);
		fA = enerA - ener_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2);
		fB = enerB - ener_goal;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, cs2);
			fC = enerC - ener_goal;
			error_e = fabs(fC / ener_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}

	if (error_e > EOS_TOL) {
		*pres = (GAMMA - 1.0) * u_goal;
		*cs2 = GAMMA * (GAMMA - 1.0) * u_goal / (den + GAMMA * u_goal);
		error_e = 9.99e-12;
	}

	if (error_e > EOS_TOL) printf("2 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);

}

__device__ void eos_mode_rhow_pres_dpdrho_dpde_d (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *dpdrho, double *dpde_d) {
	// initial guess : temperature
    double temp_ini_guess;
	eos_NR_temp_guess(den, w_goal-den, &temp_ini_guess);
	
	double deni = 1.0 / den;
	double h_goal = w_goal * deni;
    double temp_new, temp_old;
    double dpdt, dedt, dhdt;
	double entr, dsdt, dedrho;
    double h_tmp;
    double cs2;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xener = 0.0;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &xener, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);

        h_tmp = xener + (*pres) * deni;
        dhdt = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdt * xenth;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
		validate_T(&temp_new);

        temp_old = temp_new;
		if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double presA, presB, presC;
	double fA, fB, fC;
	int flag = 1;

	if (error_h > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
		fA = enerA + presA * deni - xenth;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
		fB = enerB + presB * deni - xenth;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
			fC = enerC + presC * deni - xenth;
			error_h = fabs(fC / xenth);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
		*pres = presC;
	}

	//if (error_h > EOS_TOL) {
	//	*pres = (GAMMA - 1.0) * (w_goal - den) / (GAMMA);
	//}

	if (error_h > EOS_TOL) printf("3 %g %g %g %g %g\n", error_h, temp_old, den, w_goal-den, temp_ini_guess);

    *dpde_d = dpdt / dedt;
}

__device__ void eos_mode_rhow_pres_u (const  double* __restrict__ gpu_eos_table, double den, double w_goal, double *pres, double *u) {
    // implementation in Newman-Hamlin inversion
	// initial guess : temperature
    double temp_ini_guess;
	eos_NR_temp_guess(den, w_goal - den, &temp_ini_guess);

	double deni = 1.0 / den;
    double h_goal = w_goal * deni;
	double temp_new, temp_old;
    double dhdtemp;
    double h_tmp;
    double cs2;
    double dpdrho, dpdt, dedt;
	double entr, dsdt, dedrho;

    double error, error_h;
    int i;

    double xenth = h_goal - 1.0; // Helmholtz EOS takes non-relativistic enthalpy
    double xener;

    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);

        h_tmp = xener + (*pres) * deni;
        dhdtemp = dedt + dpdt * deni;
        temp_new = temp_old - (h_tmp / xenth - 1.0) / dhdtemp * xenth;

		// do not allow temp to change more than 2 times in one iteration
		if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
		if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_h = fabs((h_tmp - xenth) / xenth);
		validate_T(&temp_new);

        temp_old = temp_new;
		if (error < EOS_TEMP_TOL && error_h < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }
	*u = xener * den;

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double presA, presB, presC;
	double fA, fB, fC;
	int flag = 1;

	if (error_h > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, &presA, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fA = enerA + presA * deni - xenth;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, &presB, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fB = enerB + presB * deni - xenth;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, &presC, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
			fC = enerC + presC * deni - xenth;
			error_h = fabs(fC / xenth);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_h < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;

			i++;
		}
		*pres = presC;
		*u = enerC * den;
	}

	if (error_h > EOS_TOL) {
		*u = (w_goal - den) / GAMMA;
		*pres = *u * (GAMMA - 1.);
		error_h = 9.99e-12;
	}

	if (error_h > EOS_TOL) printf("4 %g %g %g %g %g\n", error_h, temp_old, den, w_goal-den, temp_ini_guess);
}

__device__ void eos_mode_rhotemp_pres_min (const  double* __restrict__ gpu_eos_table, double den, double *pres) {
    // implementation in Newman-Hamlin inversion
    // Parameters of Newton-Raphson iterations
    double temp = eos_temp_low;
    double ener, dpdt, dedt, dpdrho;
	double entr, dsdt, dedrho;
	double cs2;

	eos_helm(gpu_eos_table, 1, temp, den, 1.0, 1.0, pres, &ener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
}

__device__ void eos_mode_rhopres_u (const  double* __restrict__ gpu_eos_table, double den, double p_goal, double *u) {
	// initial guess : temperature
    double temp_ini_guess;
	eos_NR_temp_guess(den, p_goal, &temp_ini_guess);

    double temp_new, temp_old;
    double p_tmp;
    double cs2;
    double xener;
    double error, error_p;
    int i;

	double entr, dsdt, dedrho;
    double dpdt, dedt, dpdrho;
    int more_iterations = 2; // number of additional iterations, if reached desired tolerance

    temp_old = temp_ini_guess;
    for(i = 0; i < EOS_ITERATIONS; i++){
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, &p_tmp, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);

        temp_new = temp_old - (p_tmp - p_goal) / dpdt;

        // do not allow temp to change more than 2 times in one iteration
        if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
        if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

        error = fabs((temp_new - temp_old) / temp_old);
        error_p = fabs((p_tmp - p_goal) / p_goal);
		validate_T(&temp_new);

        temp_old = temp_new;
        // more iterations after reached below tolerance
		if (error < EOS_TEMP_TOL && error_p < EOS_TOL) {
            more_iterations -= 1;
            if (more_iterations == 0) break;
        }
    }

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double presA, presB, presC;
	double fA, fB, fC;
	int flag = 1;

	if (error_p > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, &presA, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fA = presA - p_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, &presB, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fB = presB - p_goal;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, &presC, &xener, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
			fC = presC - p_goal;
			error_p = fabs(fC / p_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}

	if (error_p > EOS_TOL) {
		*u = p_goal / (GAMMA - 1.);
		error_p = 9.99e-12;
	}

	if (error_p > EOS_TOL) printf("5 %g %g %g %g %g\n", error_p, temp_old, den, p_goal, temp_ini_guess);
	*u = xener * den;
}


// Entropy inversion
__device__ void eos_mode_rhos_upres(const double* __restrict__ gpu_eos_table, double den, double entr_goal, double *pres, double* u, double *dpdrho, double *dudrho) {
	// initial guess : temperature
	double temp_ini_guess = 1.0e9; // Danat: random guess
	
	// Convert kappa to entropy
	//entr_goal = log(entr_goal / KTOT_FACTOR);

	double temp_new, temp_old;
	double cs2;
	double xener, xentr;
	double dedrho;
	double error, error_p;
	int i;
	double dpdt, dedt, dsdt;
	int more_iterations = 2; // number of additional iterations, if reached desired tolerance

	temp_old = temp_ini_guess;
	for (i = 0; i < EOS_ITERATIONS; i++) {
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, pres, &xener, &xentr, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);

		temp_new = temp_old - (xentr - entr_goal) / dsdt;

		// do not allow temp to change more than 2 times in one iteration
		if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
		if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

		error = fabs((temp_new - temp_old) / temp_old);
		error_p = fabs((xentr - entr_goal) / entr_goal);
		validate_T(&temp_new);

		temp_old = temp_new;
		// more iterations after reached below tolerance
		if (error < EOS_TEMP_TOL && error_p < EOS_TOL) {
			more_iterations -= 1;
			if (more_iterations == 0) break;
		}
	}

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double entrA, entrB, entrC;
	double fA, fB, fC;
	int flag = 1;

	if (error_p > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, pres, &xener, &entrA, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
		fA = entrA - entr_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, pres, &xener, &entrB, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
		fB = entrB - entr_goal;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, pres, &xener, &entrC, &dpdt, &dedt, &dsdt, dpdrho, &dedrho, &cs2);
			fC = entrC - entr_goal;
			error_p = fabs(fC / entr_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_p < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}

	*u = xener * den;
	*dudrho = dedrho * den + xener;
	if (error_p > EOS_TOL) printf("6 %g %g %g %g %g\n", error_p, temp_old, den, entr_goal, temp_ini_guess);

	//if (isnan(entr_goal)) printf("[helm] u: %g, iters: %d, T: %g err: %g den: %g s: %g\n", (*u), i, temp_old, error_p, den, entr_goal);
}


__device__ void eos_mode_rhou_entr(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* entr) {
	// Check the input:
	if (eos_check_input_u(den, u_goal)) {
		*entr = 1e-30;
		return;
	}

	// initial guess : temperature
	double temp_ini_guess;
	eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

	double ener_goal = u_goal / den;
	double temp_new, temp_old;
	double ener_tmp;
	double dpdt, dedt, dpdrho;
	double dsdt, dedrho;
	double pres, cs2;
	double error, error_e;
	int i;
	int more_iterations = 2; // number of additional iterations, if reached desired tolerance

	temp_old = temp_ini_guess;
	for (i = 0; i < EOS_ITERATIONS; i++) {
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, &pres, &ener_tmp, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

		//do not allow temp to change more than 2. times in one iteration
		if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
		if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

		error = fabs((temp_new - temp_old) / temp_old);
		error_e = fabs((ener_tmp - ener_goal) / ener_goal);
		validate_T(&temp_new);

		temp_old = temp_new;
		// more iterations after reached below tolerance
		if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
			more_iterations -= 1;
			if (more_iterations == 0) break;
		}
	}

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double fA, fB, fC;
	int flag = 1;

	if (error_e > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, &pres, &enerA, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fA = enerA - ener_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, &pres, &enerB, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fB = enerB - ener_goal;

		if (fA * fB >= 0.0) flag = 0; 

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, &pres, &enerC, entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
			fC = enerC - ener_goal;
			error_e = fabs(fC / ener_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}

	if (error_e > EOS_TOL) printf("7 %g %g %g %g %g\n", error_e, temp_old, den, u_goal, temp_ini_guess);
}

__device__ void eos_mode_rhou_temp(const  double* __restrict__ gpu_eos_table, double den, double u_goal, double* temp) {
	// initial guess : temperature
	double temp_ini_guess;
	eos_NR_temp_guess(den, u_goal, &temp_ini_guess);

	double ener_goal = u_goal / den;
	double temp_new, temp_old;
	double ener_tmp;
	double dpdt, dedt, dpdrho;
	double dsdt, dedrho;
	double pres, cs2, entr;
	double error, error_e;
	int i;
	int more_iterations = 2; // number of additional iterations, if reached desired tolerance

	temp_old = temp_ini_guess;
	for (i = 0; i < EOS_ITERATIONS; i++) {
		eos_helm(gpu_eos_table, 1, temp_old, den, 1.0, 1.0, &pres, &ener_tmp, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		temp_new = temp_old - (ener_tmp - ener_goal) / dedt;

		//do not allow temp to change more than 2. times in one iteration
		if (temp_new / temp_old > 10.0) temp_new = 10.0 * temp_old;
		if (temp_old / temp_new > 10.0) temp_new = 0.1 * temp_old;

		error = fabs((temp_new - temp_old) / temp_old);
		error_e = fabs((ener_tmp - ener_goal) / ener_goal);	
		validate_T(&temp_new);

		temp_old = temp_new;
		// more iterations after reached below tolerance
		if (error < EOS_TEMP_TOL && error_e < EOS_TOL) {
			more_iterations -= 1;
			if (more_iterations == 0) break;
		}
	}
	*temp = temp_old;

	// Bisection method as backup rootfinder
	double tempA, tempB, tempC;
	double enerA, enerB, enerC;
	double fA, fB, fC;
	int flag = 1;

	if (error_e > EOS_TOL) {
		tempA = eos_temp_low;
		eos_helm(gpu_eos_table, 1, tempA, den, 1.0, 1.0, &pres, &enerA, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fA = enerA - ener_goal;

		tempB = eos_temp_up;
		eos_helm(gpu_eos_table, 1, tempB, den, 1.0, 1.0, &pres, &enerB, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
		fB = enerB - ener_goal;

		if (fA * fB >= 0.0) flag = 0;

		i = 0;
		while (i < 2 * EOS_ITERATIONS && flag) {
			tempC = 0.5 * ((tempA)+(tempB));

			eos_helm(gpu_eos_table, 1, tempC, den, 1.0, 1.0, &pres, &enerC, &entr, &dpdt, &dedt, &dsdt, &dpdrho, &dedrho, &cs2);
			fC = enerC - ener_goal;
			error_e = fabs(fC / ener_goal);

			if (fC == 0.0 || 0.5 * (tempB - tempA) < EOS_TEMP_TOL || error_e < EOS_TOL) {
				break;
			}

			if (fC * fA >= 0.0) tempA = tempC;
			else tempB = tempC;
			i++;
		}
	}
	*temp = tempC;

	if (error_e > EOS_TOL) {
		// Revert back to GAMMA law
		*temp = fabs(MMW * MH_CGS * (GAMMA - 1.) * (ener_goal * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * MASS_DENSITY_SCALE));
		error_e = 9.99e-12;
	}

	if (error_e > EOS_TOL) printf("8 %g %g %g %g %g\n", error_e, *temp, den, u_goal, temp_ini_guess);
}
#endif

__device__ void primtoflux_res(double* pr, struct of_state_res* q_res, int dir, struct of_geom* geom, double* flux)
{
	#if(RESISTIVE)
	int k;
	double alpha, beta[NDIM], Ecov[3], Bcov[3];
	double sqrtgamma_inv;

	/* particle number flux */
	flux[RHO] = pr[RHO] * q_res->ucon[dir];

	/* MHD stress-energy tensor w/ first index up, * second index down. */
	mhd_calc_res(pr, dir, geom, q_res, &flux[UU]);
	flux[UU] += flux[RHO];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-(geom->gcon[0]));

	//Beta in 3+1
	beta[0] = 0;
	beta[1] = geom->gcon[1] * alpha * alpha;
	beta[2] = geom->gcon[2] * alpha * alpha;
	beta[3] = geom->gcon[3] * alpha * alpha;

	//Inverse of 3-metric
	sqrtgamma_inv = alpha / (geom->g);

	/*Maxwell tensor */
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	if (dir == 0) {
		flux[E1] = pr[E1];
		flux[E2] = pr[E2];
		flux[E3] = pr[E3];
	}
	else {
		flux[E1] = beta[1] * pr[E1 + (dir - 1)] - beta[dir] * pr[E1];
		flux[E2] = beta[2] * pr[E1 + (dir - 1)] - beta[dir] * pr[E2];
		flux[E3] = beta[3] * pr[E1 + (dir - 1)] - beta[dir] * pr[E3];
		flux[E1] -= lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 0 - (dir - 1))]);
		flux[E2] -= lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 1 - (dir - 1))]);
		flux[E3] -= lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Bcov[(3 - 2 - (dir - 1))]);
	}

	/* dual of Maxwell tensor */
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	if (dir == 0) {
		flux[B1] = pr[B1];
		flux[B2] = pr[B2];
		flux[B3] = pr[B3];
	}
	else {
		flux[B1] = beta[1] * pr[B1 + (dir - 1)] - beta[dir] * pr[B1];
		flux[B2] = beta[2] * pr[B1 + (dir - 1)] - beta[dir] * pr[B2];
		flux[B3] = beta[3] * pr[B1 + (dir - 1)] - beta[dir] * pr[B3];
		flux[B1] += lvc3u(0, dir - 1, (3 - 0 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 0 - (dir - 1))]);
		flux[B2] += lvc3u(1, dir - 1, (3 - 1 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 1 - (dir - 1))]);
		flux[B3] += lvc3u(2, dir - 1, (3 - 2 - (dir - 1))) * (alpha * sqrtgamma_inv) * (Ecov[(3 - 2 - (dir - 1))]);
	}

	//Entropy advection
	#if(FULL_ENTROPY)
	flux[KTOT] = flux[RHO] * 1. / (GAMMA - 1.) * log((GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA));
	#else
	flux[KTOT] = flux[RHO] * (GAMMA - 1.) * pr[UU] * pow(pr[RHO], -GAMMA);
	#endif

	PLOOP flux[k] *= geom->g;
	#endif
}

/* calculate magnetic field four-vector */
__device__ void econ_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* econ)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], E_dot_v, Bcov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[1] * alpha;
	ncon[2] = -geom->gcon[2] * alpha;
	ncon[3] = -geom->gcon[3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA!
	lower_3(&(pr[B1]), geom->gcov, Bcov);
	E_dot_v = alpha * (pr[E1] * ucov[1] + pr[E2] * ucov[2] + pr[E3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	econ[0] = (E_dot_v)*ncon[0];
	econ[1] = (E_dot_v)*ncon[1] + gamma * (alpha * pr[E1]) + (alpha * alpha / geom->g) * (ucov[2] * Bcov[2] - ucov[3] * Bcov[1]);
	econ[2] = (E_dot_v)*ncon[2] + gamma * (alpha * pr[E2]) + (alpha * alpha / geom->g) * (ucov[3] * Bcov[0] - ucov[1] * Bcov[2]);
	econ[3] = (E_dot_v)*ncon[3] + gamma * (alpha * pr[E3]) + (alpha * alpha / geom->g) * (ucov[1] * Bcov[1] - ucov[2] * Bcov[0]);

	return;
	#endif
}

/* calculate magnetic field four-vector */
__device__ void bcon_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* bcon)
{
	#if(RESISTIVE)
	double alpha, gamma, ncon[NDIM], B_dot_v, Ecov[3];

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//4-velocity Eulerian observer
	ncon[0] = 1.0 / alpha;
	ncon[1] = -geom->gcon[1] * alpha;
	ncon[2] = -geom->gcon[2] * alpha;
	ncon[3] = -geom->gcon[3] * alpha;

	//Gamma in 3+1
	gamma = ucon[0] * alpha;

	//Dot product between magnetic field and velocity 3-vector time GAMMA
	lower_3(&(pr[E1]), geom->gcov, Ecov);
	B_dot_v = alpha * (pr[B1] * ucov[1] + pr[B2] * ucov[2] + pr[B3] * ucov[3]);

	//Final calculation of rest frame magnetic field
	bcon[0] = (B_dot_v)*ncon[0];
	bcon[1] = (B_dot_v)*ncon[1] + gamma * (alpha * pr[B1]) - (alpha * alpha / geom->g) * (ucov[2] * Ecov[2] - ucov[3] * Ecov[1]);
	bcon[2] = (B_dot_v)*ncon[2] + gamma * (alpha * pr[B2]) - (alpha * alpha / geom->g) * (ucov[3] * Ecov[0] - ucov[1] * Ecov[2]);
	bcon[3] = (B_dot_v)*ncon[3] + gamma * (alpha * pr[B3]) - (alpha * alpha / geom->g) * (ucov[1] * Ecov[1] - ucov[2] * Ecov[0]);

	return;
	#endif
}

/* MHD stress tensor, with first index up, second index down */
__device__ void mhd_calc_res(double* pr, int dir, struct of_geom* geom, struct of_state_res* q_res, double* mhd)
{
	#if(RESISTIVE)
	int j, lambda, beta, kappa;
	double P, w, bsq, esq, eta, ptot, mhd_u[NDIM], mhd_d[NDIM];

	//Calculate contraction term
	DLOOPA{
		mhd_u[j] = 0.;
		for (lambda = 0; lambda < NDIM; lambda++)for (beta = 0; beta < NDIM; beta++)for (kappa = 0; kappa < NDIM; kappa++) {
			mhd_u[j] += q_res->ucov[lambda] * q_res->ecov[beta] * q_res->bcov[kappa] * (q_res->ucon[dir] * (1.0 / geom->g) * lvc4u(j, lambda, beta, kappa) + q_res->ucon[j] * (1.0 / geom->g) * lvc4u(dir, lambda, beta, kappa));
		}
	}
	lower(mhd_u, geom->gcov, mhd_d);

	#if DOHELM   
	eos_mode_rhou_pres(pr[RHO], pr[UU], &P); // Helmholtz EOS
	#else
	P = (GAMMA - 1.) * pr[UU]; // Ideal gas EOS
	#endif

	w = P + pr[RHO] + pr[UU];
	bsq = dot(q_res->bcon, q_res->bcov);
	esq = dot(q_res->econ, q_res->ecov);
	eta = w + (bsq + esq);
	ptot = P + 0.5 * (bsq + esq);

	//single row of mhd stress tensor, first index up, second index down
	DLOOPA mhd[j] = eta * q_res->ucon[dir] * q_res->ucov[j] + ptot * delta(dir, j) - q_res->bcon[dir] * q_res->bcov[j] - q_res->econ[dir] * q_res->ecov[j] + mhd_d[j];
	#endif
}

/* add in geometrical and cooling source terms to equations of motion */
__device__ void source_res(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double* q, double Dt, const  double* __restrict__ conn_GPU, struct of_state_res* q_res, double r) {
	double conn, mhd_res[NDIM][NDIM];
	int k;
	double alpha, beta[4], gamma;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id = icurr * (BS_2 + 2 * N2G) + jcurr;
	#endif

	mhd_calc_res(ph, 0, geom, q_res, mhd_res[0]);
	mhd_calc_res(ph, 1, geom, q_res, mhd_res[1]);
	mhd_calc_res(ph, 2, geom, q_res, mhd_res[2]);
	mhd_calc_res(ph, 3, geom, q_res, mhd_res[3]);

	/* contract mhd stress tensor with connection */
	PLOOP dU[k] = 0.;

	#pragma unroll 4
	for (k = 0; k < NDIM; k++) {
		#if(NSY)
		dU[UU] += mhd_res[0][k] * conn_GPU[0 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[1][k] * conn_GPU[4 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[2][k] * conn_GPU[7 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd_res[3][k] * conn_GPU[9 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[1][k] * conn;
		dU[U1] += mhd_res[0][k] * conn;
		conn = conn_GPU[2 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[0][k] * conn;
		conn = conn_GPU[3 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[0][k] * conn;
		conn = conn_GPU[5 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[1][k] * conn;
		conn = conn_GPU[6 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[1][k] * conn;
		conn = conn_GPU[8 * NDIM * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[2][k] * conn;
		#else
		dU[UU] += mhd_res[0][k] * conn_GPU[0 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[1][k] * conn_GPU[4 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[2][k] * conn_GPU[7 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U3] += mhd_res[3][k] * conn_GPU[9 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		conn = conn_GPU[1 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[1][k] * conn;
		dU[U1] += mhd_res[0][k] * conn;
		conn = conn_GPU[2 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[0][k] * conn;
		conn = conn_GPU[3 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[UU] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[0][k] * conn;
		conn = conn_GPU[5 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[2][k] * conn;
		dU[U2] += mhd_res[1][k] * conn;
		conn = conn_GPU[6 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U1] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[1][k] * conn;
		conn = conn_GPU[8 * NDIM * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + k * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id];
		dU[U2] += mhd_res[3][k] * conn;
		dU[U3] += mhd_res[2][k] * conn;
		#endif
	}

	//Lapse in 3+1
	alpha = 1.0 / sqrt(-geom->gcon[0]);

	//Beta in 3+1
	beta[1] = geom->gcon[1] * alpha * alpha;
	beta[2] = geom->gcon[2] * alpha * alpha;
	beta[3] = geom->gcon[3] * alpha * alpha;

	//Calculate relative Lorentz factor
	gamma = q_res->ucon[0] * alpha;

	//Calculate explicit part of electric current J sourceterm
	dU[E1] = -alpha * q[0] * ph[U1] / gamma + beta[1] * q[0];
	dU[E2] = -alpha * q[0] * ph[U2] / gamma + beta[2] * q[0];
	dU[E3] = -alpha * q[0] * ph[U3] / gamma + beta[3] * q[0];

	//Add cooling term if needed
	#if (COOL_DISK)
	misc_source(ph, icurr, jcurr, geom, q, dU, r, Dt);
	#endif

	PLOOP dU[k] *= geom->g;
}

//returns b^2 (i.e., twice magnetic pressure)
__device__ double bsq_calc_res(double* pr, struct of_geom* geom)
{
	double ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM];
	ucon_calc(pr, geom, ucon);
	lower(ucon, geom->gcov, ucov);
	bcon_calc_res(pr, geom, ucon, ucov, bcon);
	lower(bcon, geom->gcov, bcov);

	return(dot(bcon, bcov));
}

//find ucon, ucov, bcon, bcov from primitive variables */
__device__ void get_state_res(double* pr, struct of_geom* geom, struct of_state_res* q_res)
{
	#if(RESISTIVE)
	//get ucon
	ucon_calc(pr, geom, q_res->ucon);
	lower(q_res->ucon, geom->gcov, q_res->ucov);

	//get bcon
	bcon_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->bcon);
	lower(q_res->bcon, geom->gcov, q_res->bcov);

	//get econ
	econ_calc_res(pr, geom, q_res->ucon, q_res->ucov, q_res->econ);
	lower(q_res->econ, geom->gcov, q_res->ecov);
	#endif
}

//Calculate wavespeed assuming it is c
__device__ void vchar_res(struct of_geom* geom, int js, double* vmax, double* vmin) {
	double sqrtgamma, ncon_js, alpha, beta, vm, vp;
	alpha = 1. / sqrt(-geom->gcon[0]);
	beta = geom->gcon[js] * alpha * alpha;
	ncon_js = -alpha * geom->gcon[js];

	if (js == 1) sqrtgamma = sqrt(geom->gcon[4] + ncon_js * ncon_js);
	else if (js == 2) sqrtgamma = sqrt(geom->gcon[7] + ncon_js * ncon_js);
	else sqrtgamma = sqrt(geom->gcon[9] + ncon_js * ncon_js);

	vp = alpha * sqrtgamma - beta;
	vm = -alpha * sqrtgamma - beta;

	if (vp > vm) {
		*vmax = vp;
		*vmin = vm;
	}
	else {
		*vmax = vm;
		*vmin = vp;
	}
}

//Calculate wavespeed in ideal limit
__device__ void vchar_res2(double* pr, struct of_state_res* q, struct of_geom* geom, int dir, double* vmax, double* vmin)
{
	double discr, vp, vm, va2, cs2, cms2;
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

	double w, bsq, eta;
	// EOS-specific calls:
	#if (DOHELM)
	// 1. Helmholtz EOS
	double cs2_helm;
	eos_mode_rhou_pres_cs2(gpu_eos_table, pr[RHO], pr[UU], &P, &cs2_helm);
	w = pr[RHO] + pr[UU] + P;
	#else
	// 2. Ideal gas EOS
	#if(AMD)
	w = fma(GAMMA, pr[UU], pr[RHO]);
	#else
	w = pr[RHO] + GAMMA * pr[UU];
	#endif
	#endif
	bsq = dot(q->bcon, q->bcov);
	eta = w + bsq;

	/* find fast magnetosonic speed */
	// EOS-specific calls:
	#if (DOHELM)
	// 1. Helmholtz EOS
	// cs2 was already calculated above
	cs2 = cs2_helm;
	#else
	// 2. Ideal gas EOS
	cs2 = GAMMA * (GAMMA - 1.) * pr[UU] / w;
	#endif

	va2 = bsq / eta;
	cms2 = cs2 + va2 - cs2 * va2;	/* and there it is... */

	//check on it!
	if (cms2 < 0.) cms2 = SMALL;
	if (cms2 > 1.) cms2 = 1.;

	//now require that speed of wave measured by observer q->ucon is cms2
	Asq = Acon_js;
	Bsq = geom->gcon[0];// dot(Bcon, Bcov);
	Au = q->ucon[dir];
	Bu = q->ucon[0];
	AB = Acon_0;
	Au2 = Au * Au;
	Bu2 = Bu * Bu;
	AuBu = Au * Bu;
	#if AMD
	A = fma(-(Bsq + Bu2), cms2, Bu2);
	B = 2. * fma(-(AB + AuBu), cms2, AuBu);
	C = fma(-(Asq + Au2), cms2, Au2);
	discr = fma(B, B, -4. * A * C);
	#else
	A = Bu2 - (Bsq + Bu2) * cms2;
	B = 2. * (AuBu - (AB + AuBu) * cms2);
	C = Au2 - (Asq + Au2) * cms2;
	discr = B * B - 4. * A * C;
	#endif

	if ((discr < 0.0) && (discr > -1.e-10)) discr = 0.0;
	else if (discr < -1.e-10) discr = 0.;
	discr = sqrt(discr);
	vp = -(-B + discr) / (2. * A);
	vm = -(-B - discr) / (2. * A);

	*vmax = MY_MAX(vp, vm);
	*vmin = MY_MIN(vp, vm);

	return;
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower_3(double* ucon, double gcov[10], double* ucov)
{
	#if(RESISTIVE)
	ucov[0] = gcov[4] * ucon[0] + gcov[5] * ucon[1] + gcov[6] * ucon[2];
	ucov[1] = gcov[5] * ucon[0] + gcov[7] * ucon[1] + gcov[8] * ucon[2];
	ucov[2] = gcov[6] * ucon[0] + gcov[8] * ucon[1] + gcov[9] * ucon[2];
	return;
	#endif
}

//4D Levi-cevita symbol (not tensor)
__device__ double lvc4u(int i, int j, int k, int l) {
	double lvc4u;

	if ((i == j) || (i == k) || (i == l) || (j == k) || (j == l) || (k == l)) {
		lvc4u = 0.0;
	}
	else if ((i + j == 1) || (i + j == 5)) {
		if ((j + k) % 4 == 3) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if ((i + j == 2) || (i + j == 4)) {
		if ((j + k) % 4 == 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else if (i + j == 3) {
		if ((j + k) % 4 != 1) lvc4u = 1.0;
		else lvc4u = -1.0;
	}
	else lvc4u = 0.0;

	return (lvc4u);
}

//3D Levi-cevita symbol (not tensor)
__device__ double lvc3u(int i, int j, int k) {
	double lvc3u;

	if ((i == j) || (j == k) || (k == i)) lvc3u = 0.;
	else if ((i + 1 == j) || (i - 2 == j)) lvc3u = 1.;
	else lvc3u = -1.;

	return (lvc3u);
}

__device__ double divE_calc(double* p, const  double* __restrict__ gdet, double _dx1, double _dx2, double _dx3, int ii, int jj, int zz) {
	#if(RESISTIVE)
	double divE;
	int isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	int jsize = (BS_3 + 2 * N3G);
	int global_id = ii * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jj * (BS_3 + 2 * N3G) + zz;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	#if(NSY)
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ind0 = CENT * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id;
	#else
	int fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int global_id_2D = ii * (BS_2 + 2 * N2G) + jj;
	int ind0 = CENT * ((BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + global_id_2D;
	#endif

	int zsize = 1, zoffset = 0, zlevel = 0;

	#if(N_LEVELS_1D_INT>0 && D3>0 && GPU_ENABLED==1)
	if ((block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) && j < N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(j - N2_GPU_offset[n]) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if ((block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) && j >= N2_GPU_offset[n] + BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(j - N2_GPU_offset[n], BS_2 - D2)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = round(pow(2.0, (double)zlevel));
	zoffset = (z - N3_GPU_offset[n]) % zsize;
	#endif

	/* Constrained transport defn */

	/* Flux-ct defn */
	divE = (
		#if(N1G>1)
		(p[E1 * ksize + global_id + isize] * gdet[ind0 + (!NSY) * (BS_1 + 2 * N1G) + NSY * isize] - p[E1 * ksize + global_id - isize] * gdet[ind0 - (!NSY) * (BS_1 + 2 * N1G) - NSY * isize]) / (2.0 * _dx1)
		#endif
		#if(N2G>1)
		+ (p[E2 * ksize + global_id + jsize] * gdet[ind0 + (!NSY) + jsize * NSY] - p[E2 * ksize + global_id - jsize] * gdet[ind0 - (!NSY) - jsize * NSY]) / (2.0 * _dx2)
		#endif
		#if(N3G>1)
		+ (p[E3 * ksize + global_id + zsize] * gdet[ind0 + NSY * zsize] - p[E3 * ksize + global_id - zsize] * gdet[ind0 - NSY*zsize]) / (2.0 * (double)(zsize)*_dx3)
		#endif
	);
	return (divE / gdet[ind0]);
	#else
	return(0.0);
	#endif
}