
/*Declarations of functions for Utoprim2D*/
__device__ int Utoprim_2d(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double fel
	#endif
);
__device__ int Utoprim_new_body(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double tolerance, int lim
	 #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int general_newton_raphson(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DOHELM_TEMPERATURE)
    //, double temp_guess
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_vsq(double[], double[], double[], double[][NEWT_DIM_2], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DOHELM_TEMPERATURE)
    //, double temp_guess
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif	
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_1dvsq2fix1(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim, int full_entropy
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double fel
	#endif
);
__device__ double W_of_vsq2(double vsq, double* p, double* rho, double* u, double D, double K_atm
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DOHELM_TEMPERATURE)
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_new_body2(double U[], double gcov[10], double gcon[10], double gdet, double prim[], double K_atm, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if(DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ void func_1d_gnr2(double x[], double dx[], double resid[], double jac[][NEWT_DIM_1], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if(DOHELM_TEMPERATURE)
    //, double temp_guess
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif	
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int general_newton_raphson2(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double K_atm, double tolerance
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if(DOHELM_TEMPERATURE)
    //, double temp_guess
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif	
	#if(TWO_T)
	, double* S
	, double fel
	#endif
);
__device__ int Utoprim_NM(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double fel
	#endif
);
__device__ int Utoprim_NM_calc(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
    #if (DO_YE)
    , double ye
    #endif
	#if(TWO_T)
	, double *S
	, double fel
	#endif
);

__device__ int Utoprim_3D_T(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
);
__device__ int Utoprim_new_3D_T(double* U, double gcov[10], double gcon[10], double gdet, double* prim, double tolerance, int lim
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
	#if (DO_YE)
    , double ye
	#endif
);
__device__ int general_newton_raphson_3D_T(double x[], double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D, double tolerance
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
	#if (DO_YE)
    , double ye
	#endif
);
__device__ void func_vsq_3D_T(double x[], double dx[], double resid[], double jac[][3], double* f, double* df, double Bsq, double Qtsq, double QdotBsq, double Qdotn, double D
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
	#if (DO_YE)
    , double ye
	#endif
);
__device__ int get_safe_guess_NR_3D_T(double x[3], double D, double Bsq, double Qdotn, double ye
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
);
__device__ void validate_x_3D_T(double x[3], double x0[3]);
__device__ void EP_dEdW_dEdZ_dEdT(double* Eprim, double* Pprim, double* dEdvsq, double* dEdW, double* dEdT, double* dpdrho, double* dpdT, double* x, double D
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
	#endif
	#if (DO_YE)
    , double ye
	#endif
);

__device__ void vchar(double* pr, struct of_state* q, struct of_geom* geom, int dir, double* vmax, double* vmin
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
);
__device__ void primtoflux(double* pr, struct of_state* q, int dir, struct of_geom* geom, double* flux, double* vmax, double* vmin
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ int fixup_cell(double pf[NDIM], double r, struct of_geom* geom
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(CALC_MDOT)
	,  double magnetic_density_scale
	#endif
);
__device__ void source(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double Dt, const  double* __restrict__ conn, struct of_state* q, double r
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ void mhd_calc(double* pr, int dir, struct of_state* q, double* mhd
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(TWO_T)
	, double gamma_g
	#endif
);
__device__ void primtoflux_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double flux[NPR]
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
);
__device__ void vchar_FT(double* pr, double ucon[NDIM], double bcon[NDIM], int dir, double* vmax, double* vmin
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
);
__device__ void implicit_rad_solve(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ void implicit_rad_solve_init(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int* pflag_rad
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(TWO_T)
	, double fel
	#endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);

/*Declarations of functions related to (M1) radiation inversion scheme*/
__device__ int implicit_rad_solve_PMHD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max,int do_entropy, int do_staged
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int implicit_rad_solve_PMHD_fast(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max,int do_entropy, int do_staged
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int implicit_rad_solve_UMHD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int implicit_rad_solve_PRAD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
	#if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int implicit_rad_solve_URAD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int implicit_rad_solve_EMHD(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_rad, struct of_geom* geom, double* dU, double Dt, double* error_t, double cell_size, double y_max, int do_entropy, int do_staged
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(COOL_STOP)
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);

__device__ void source_rad(double* ph, struct of_geom* geom, struct of_state* q, struct of_state_rad* q_rad, double* dU
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(COOL_STOP)
	, double r
	, double r
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ void calc_Gcon(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucov[NDIM], double ucon_rad[NDIM], double ucov_rad[NDIM], double mhd_rad[NDIM][NDIM], double bsq
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
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ double calc_Tr(double* ph, double ucon[NDIM], double ucon_rad[NDIM], double ucov[NDIM]
	#if(P_NUM)
	, double *exp_xi
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ double calc_Te(double* ph);
__device__ double calc_Ti(double* ph);
__device__ void vchar_rad(double* pr, struct of_state* q, struct of_state_rad* q_rad, struct of_geom* geom, int js, double* vmax, double* vmin, double dx
    #if (DOHELM)
    , const  double* __restrict__ gpu_eos_table
    #endif	
	#if(TWO_T)
	, double gamma_g
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);


__device__ void calc_kappa_new(double* ph, double bsq, double Tr, double Te, double* kappa_abs, double* kappa_emmit, double* kappa_es	
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
	, double* kappa_abs_ph
	, double* kappa_emmit_ph
	, double exp_xi
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);

__device__ double vsq_calc(double W, double Bsq, double Qtsq, double QdotBsq);
__device__ double x1_of_x0(double x0, double Bsq, double Qtsq, double QdotBsq);
#if (DOHELM)
__device__ void dWdvsq_calc2_helmholtz(const  double* __restrict__ gpu_eos_table, double vsq, double D, double K_atm, double* W, double* dWdvsq
    #if(DOHELM_TEMPERATURE)
    , double* temp_prev
    #endif
    #if (DO_YE)
    , double ye
    #endif
    #if (DONUCLEAR)
    , double x_alpha, double x_atm
    #endif
);
#endif
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
__device__ int Rtoprim(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__device__ int Rtoprim_calc(double *U, double gcov[10], double gcon[10], double gdet, double *prim, double y_max, int lim
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
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
__device__ int invert_matrix_1D(double Am[][1], double Aminv[][1]);
__device__ int invert_matrix_2D(double Am[][2], double Aminv[][2]);

/*Declare other functions*/
__device__ void get_state(double *  pr, struct of_geom *  geom, struct of_state *  q
	#if(CALC_MDOT)
	,  double magnetic_density_scale
	#endif
);
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
__device__ void inflow_check(double *  prim, int ii, int jj, int zz, int type, const  double* __restrict__ gcov1, const  double* __restrict__ gcoBS_2, const  double* __restrict__ gdet3, int dir);
__device__ double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq);
__device__ double readImageDouble(int a);
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
__device__ void eos_helm(const  double* __restrict__ gpu_eos_table, int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_helm_backup_nondegenerate(int calc_derivatives, double btemp, double den, double ye, double* pres, double* ener, double* entr, double* dpresdt, double* denerdt, double* dentrdt, double* dpresdd, double* denerdd, double* cs2, double* etaele);
__device__ void eos_mode_rhou_pres(const  double* __restrict__ gpu_eos_table, double rho, double u_goal, double ye, double* pres);
__device__ void eos_mode_rhou_pres_cs2(const  double* __restrict__ gpu_eos_table, double rho, double u_goal, double ye, double* pres, double* cs2);
__device__ void eos_mode_rhow_pres_dpdrho_dpde_d(const  double* __restrict__ gpu_eos_table, double rho, double w_goal, double ye, double* pres, double* dpdrho, double* dpde_d);
__device__ void eos_mode_rhow_pres_u(const  double* __restrict__ gpu_eos_table, double rho, double w_goal, double ye, double* pres, double* u);
__device__ void eos_mode_rhotemp_pres_min(const  double* __restrict__ gpu_eos_table, double rho, double ye, double* pres);
__device__ void eos_mode_rhos_upres(const  double* __restrict__ gpu_eos_table, double rho, double s_goal, double ye, double* pres, double* u, double* dpdrho, double* dudrho);
__device__ void eos_mode_rhou_entr(const  double* __restrict__ gpu_eos_table, double rho, double u_goal, double ye, double* entr);
__device__ void eos_mode_rhou_temp(const  double* __restrict__ gpu_eos_table, double rho, double u_goal, double ye, double* temp);

// DITEMP: eos wrapper functions 
#if (DOHELM_TEMPERATURE)
__device__ void eos_mode_rhotemp_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* u
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_mode_rhotemp_pres_u_cs2(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* u, double* cs2
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_mode_rhotemp_pres_u_3D_T(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres, double* ener, double* dPdrho, double* dPdT, double* dEdrho, double* dEdT);
__device__ void eos_mode_rhotemp_pres(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* pres
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_mode_rhotemp_entr(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* entr
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);

// Rootfinding functions based on w and s (3x)
__device__ void eos_mode_rhotemp_w_pres_dpdrho_dpde_d(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double w, double* pres, double* dpdrho, double* dpde_d
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_mode_rhotemp_w_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double w, double* pres, double* u
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
__device__ void eos_mode_rhotemp_s_pres_u(const  double* __restrict__ gpu_eos_table, double dens, double* temp, double ye, double entr, double* pres, double* u, double* dpdrho, double* dudrho
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
// Fixup
__device__ int eos_mode_rhotemp_u_pres_floor(const  double* gpu_eos_table, double dens, double* temp, double ye, double u, double* pres
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
// For neutrinos
#if(NEUTRINOS_M1)
__device__ void eos_mode_rhotemp_etaele(const  double* __restrict__ gpu_eos_table, double dens, double temp, double ye, double* mu_ele
#if (DONUCLEAR)
    , double x_alpha, double x_atm
#endif
);
#endif
#endif

#if(DONUCLEAR)
__device__ double get_xp(double ye, double x_alpha);
__device__ double get_xn(double ye, double x_alpha);
__device__ void nse_abundances(double rho, double tgas, double ye, double* x_n, double* x_p, double* x_alpha);
__device__ void nse_derivatives(double rho, double tgas, double ye, double x_n, double x_p, double x_alpha, double* xn_d, double* xn_t, double* xn_y, double* xp_d, double* xp_t, double* xp_y, double* xa_d, double* xa_t, double* xa_y);
__device__ void nse_nucevol(double rho, double tgas, double ye, double* x_alpha, double* x_atm, double* xa_t);
__device__ void nuc_evol(const double* __restrict__ gpu_eos_table, double* ph);
#endif

//Declare resistivity related functions
__device__ void primtoflux_res(double* pr, struct of_state_res* q_res, int dir, struct of_geom* geom, double* flux);
__device__ void econ_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* econ);
__device__ void bcon_calc_res(double* pr, struct of_geom* geom, double* ucon, double* ucov, double* bcon);
__device__ void mhd_calc_res(double* pr, int dir, struct of_geom* geom, struct of_state_res* q_res, double* mhd);
__device__ void source_res(double* ph, struct of_geom* geom, int icurr, int jcurr, int zcurr, double* dU, double* q, double Dt, const  double* __restrict__ conn_GPU, struct of_state_res* q_res, double r);
__device__ double bsq_calc_res(double* pr, struct of_geom* geom);
__device__ void get_state_res(double* pr, struct of_geom* geom, struct of_state_res* q_res
	#if(CALC_MDOT)
	,  double magnetic_density_scale
	#endif
);
__device__ void vchar_res(struct of_geom* geom, int js, double* vmax, double* vmin);
__device__ void vchar_res2(double* pr, struct of_state_res* q, struct of_geom* geom, int js, double* vmax, double* vmin);
__device__ double divE_calc(double* p, const  double* __restrict__ gdet, double _dx1, double _dx2, double _dx3, int ii, int jj, int zz);
__device__ void lower_3(double* ucon, double gcov[10], double* ucov);
__device__ double lvc4u(int i, int j, int k, int l);
__device__ double lvc3u(int i, int j, int k);

//Declare 2T related functions
__device__ double calc_delta(double* ph, double bsq);
__device__ void heating(double* ph, struct of_state* q);
__device__ double source_Coulomb(double* p
	#if(CALC_MDOT)
	, double mass_density_scale
	#endif
);
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

#if(NEUTRINOS_M1) // DINU: 3 species
// Declarations 
__device__ int semiimplicit_solve_nu(double* pb, double* U_n, double* U_i, double* U_f, int* pflag, int* pflag_nu, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table
#if (NU_INNER_STOP)
	, double radius
#endif
#if(NEUTRINOS_DEBUG)
	, double* error_nu0, double* error_nu1, double* error_nu2
#endif
);
__device__ int implicit_solve_nu(double* pb, double* U_n, double* U_i, double* U_f, double* U_prev, int* pflag, int* pflag_nu, struct of_geom* geom, double* dU, double Dt, double cell_size, double y_max, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table
#if(NEUTRINOS_DEBUG)
	, double* error_nu0, double* error_nu1, double* error_nu2
#endif
#if (NU_INNER_STOP)
	, double radius
#endif
);
__device__ void source_linearized_nu(double* ph, struct of_geom* geom, double* ncon, double ncov0, double* U_old, double* U_new, double Dt, const  double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, int species
#if (NU_KEEP_COEFF_CONST)
	, double eta_0, double kappa_abs0, double kappa_s0, double eta_N0, double kappa_N0
#endif
);
__device__ int calc_linearized_error(double* ncon, double ncov0, double gcon[10], double* U_1, double* U_2, double* U_old, double* U_new, int species, double y_max, double* error_tmp);
__device__ void implicit_evolve_neutrino_num(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, struct of_geom* geom, double* ucon, double* ucov, double Ncon0_i, double* Ncon0_f, double Dt, int species);

// Neutrino functions declarations
__device__ int Rtoprim_nu(double* U, struct of_geom* geom, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim);
__device__ int Rtoprim_nu_calc(double* U, double* ucon, double* ucov, double gcov[10], double gcon[10], double gdet, double* prim, double y_max, int lim);
__device__ void Rtoprim_nu_number(double UN, struct of_geom* geom, double* prim, double* primN, int sp);
__device__ void primtoflux_nu(double* pr, struct of_state_nu* q_nu, int dir, struct of_geom* geom, double* flux);
__device__ void primtoflux_nu_number(double* ph, double* ucon, double* ucov, int dir, struct of_geom* geom, double* flux);
__device__ void vchar_nu(double* pr, struct of_state* q, struct of_state_nu* q_nu, struct of_geom* geom, int dir, double* vmax, double* vmin, double dx, const  double* __restrict__ gpu_eos_table, const  double* __restrict__ gpu_nulib_table);
__device__ void mhd_calc_nu(double* pr, int dir, struct of_state_nu* q_nu, double* mhd_nu, int species);
__device__ void ucon_calc_nu(double* pr, struct of_geom* geom, double* ucon_nu, int species);
__device__ int gamma_calc_nu(double* pr, struct of_geom* geom, double* gamma_nu, int species);
__device__ void get_state_nu(double* pr, struct of_geom* geom, struct of_state_nu* q_nu, int species);
__device__ void calc_source_numdens_nu(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double J, double ener_nu_avg, double* source_nu_num, int species);
__device__ double calc_nu_kappa_emiss(const double* __restrict__ gpu_nulib_table, double* ph, int sp);
__device__ double calc_nu_kappa_abs(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp);
__device__ double calc_nu_kappa_scatt(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp);
__device__ double calc_nu_number_emiss(const double* __restrict__ gpu_nulib_table, double* ph, int sp);
__device__ double calc_nu_number_abs(const double* __restrict__ gpu_eos_table, const double* __restrict__ gpu_nulib_table, double* ph, double ener_nu_avg, int sp);
__device__ void interp_nulib_check_bounds(const double* __restrict__ gpu_nulib_table, double* ph, int species, int quantity, double* opacity);
__device__ void interp_nulib_table(const double* __restrict__ gpu_nulib_table, double rho, double Tgas, double ye, int species, int quantity, double* opacity);
__device__ void calc_neutrino_temperature(const double* __restrict__ gpu_eos_table, double* ph, double ener_nu_avg, double* Tnu_over_Tgas, int species);
__device__ void calc_mu_np(double rho, double T_gas, double x_n, double x_p, double* mu_n, double* mu_p);

// explicit part:
__device__ void source_nu(double* ph, struct of_geom* geom, double* dU, double* U_i, double* U_f, double Dt, double y_max, const  double* __restrict__ gpu_eos_table, const  double* __restrict__ gpu_nulib_table
#if (NU_KEEP_COEFF_CONST)
	, double eta_0[NU_SPECIES], double kappa_abs0[NU_SPECIES], double kappa_s0[NU_SPECIES], double eta_N0[NU_SPECIES], double kappa_N0[NU_SPECIES]
#endif
);
__device__ void calc_Gcon_nu(double* ph, double Gcon[NDIM], double ucon[NDIM], double ucon_nu[NDIM], double ucov[NDIM], double mhd_nu[NDIM][NDIM], double Ncon0, const  double* __restrict__ gpu_eos_table, const  double* __restrict__ gpu_nulib_table, double* source_number_nu, double* source_ye, int species
#if (NU_KEEP_COEFF_CONST)
	, double eta_0, double kappa_abs0, double kappa_s0, double eta_N0, double kappa_N0
#endif
);
#endif

//Moved up by Matthew
__device__ void eos_NR_temp_guess(double rho, double u, double* temp);
__device__ void extrapolate_gdet_innerBC(double* pr_B, double* pr_ghost, const double gdet_B, const double gdet_ghost, double dr_over_r);

// Fermi integrals from Takahashi, El Eid & Hillebrandt '78
__device__ double calc_fermiint2(double x);
__device__ double calc_fermiint3(double x);
__device__ void validate_ye(double* ye);
__device__ void validate_T(double* temp);

//Resistive 3D contoprim inversion
__device__ int Utoprim_3d_res(double U[NPR], double gcov[10], double gcon[10], double gdet, double prim[NPR], double tolerance, int lim, double Dt);
__device__ int invert_3DU(double D, double sigma, double etares, double tau, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
__device__ void res_3du_der(double D, double sigma, double etares, double tau, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);
__device__ void getE_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac);
__device__ void getdEdu_resistive(double Enew[3], double E[3], double vU[3], double vD[3], double B_D[3], double sigma, double etares, double ggammainv[3][3], double sqrtgamma, double lfac, double dEdu[3][3]);
__device__ int invert_3DU_entropy(double D, double sigma, double etares, double entropy, double S[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double* rho, double* ug, double B_guess[3], double E_guess[3], double vD_guess[3], double tolerance);
__device__ void res_3du_der_entropy(double D, double sigma, double etares, double entropy, double S_j[3], double vD[3], double ggamma[3][3], double ggammainv[3][3], double sqrtgamma, double B[3], double E[3], double Jac[3][3], double res[3]);


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

struct of_state_nu {
    double ucon[NDIM];
    double ucov[NDIM];
};