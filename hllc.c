/* restart functions; restart_init and restart_dump */
#include "decs_MPI.h"
void primtoflux_FT(double * restrict pr, double ucon[NDIM], double bcon[NDIM], int dir, double restrict flux[NPR]);
void vchar_FT(double * restrict pr, double ucon[NDIM], double bcon[NDIM], int dir, double  restrict *vmax, double restrict *vmin);
double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq);
double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq);
void calc_HLLC(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
void calc_HLLD(int dir, double cmin_roe, double cmax_roe, double int_velocity, double l_ucon[NDIM], double r_ucon[NDIM], double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]);
double calc_HLLD_pres(int dir, int *fail_HLLC, int *fail_HLLD, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double *eta_l, double *eta_r, double *w_al, double *w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
void calc_HLLD_state(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double ptot, double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
	double B_al[NDIM], double K_ar[NDIM], double  B_ar[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double eta_l, double eta_r, double w_al, double w_ar, double vcon_cl[NDIM], double vcon_cr[NDIM],
	double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR], double R_l[NPR], double R_r[NPR], double B_c[NDIM]);
void check_HLLD_par(int dir, int * fail_HLLD, double cmin_roe, double cmax_roe, double ptot, double w_al, double w_ar, double eta_l, double eta_r, double vcon_cl[NDIM], double vcon_cr[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double B_c[NDIM]);
double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double *eta_l, double *eta_r, double  *w_al, double *w_ar);


void set_Mud(int n){
	int i, j, z;
	double A, B, C, D, E, F, G, H;
	struct of_geom geom;
	#if(!NSY)
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]) {
	#else
	ZSLOOP3D(-N1G + N1_GPU_offset[n], BS_1 + N1_GPU_offset[n] - 1 + N1G, -N2G + N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1 + N2G, -N3G + N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1 + N3G) {
	#endif
		//dir 1
		get_geometry(n, i, j, z, FACE1, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[1][1] - geom.gcon[0][1] * geom.gcon[0][1]), -0.5);
		C = pow(geom.gcov[3][3], -0.5);
		D = pow((geom.gcov[3][3])*(geom.gcov[2][2] * geom.gcov[3][3] - geom.gcov[2][3] * geom.gcov[2][3]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][1] = B*(geom.gcon[0][1] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[1][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][1][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][1] = B*(geom.gcon[0][1] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[1][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][2] = D*geom.gcov[3][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][2][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][1] = B*(geom.gcon[0][1] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[1][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][2] = -D*geom.gcov[2][3];
		Mud[nl[n]][index_2D(n, i, j, z)][1][3][3] = C;

		E = geom.gcon[0][1] * geom.gcon[1][2] - geom.gcon[1][1] * geom.gcon[0][2];
		F = geom.gcon[0][1] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[1][2];
		G = geom.gcon[0][1] * geom.gcon[1][3] - geom.gcon[1][1] * geom.gcon[0][3];
		H = geom.gcon[0][1] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[1][3];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][0] = B*geom.gcon[0][1];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][1] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][1][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][1] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[3][3]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][2] = 1./(D*geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][2][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[2][3]/geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][1] = (B*B / C)*geom.gcon[0][0]*(H+F*geom.gcov[2][3]/geom.gcov[3][3]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][2] = (1./C)*geom.gcov[2][3]/geom.gcov[3][3];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][1][3][3] = 1./C;

		//dir 2
		get_geometry(n, i, j, z, FACE2, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[2][2] - geom.gcon[0][2] * geom.gcon[0][2]), -0.5);
		C = pow(geom.gcov[1][1], -0.5);
		D = pow((geom.gcov[1][1])*(geom.gcov[3][3] * geom.gcov[1][1] - geom.gcov[3][1] * geom.gcov[3][1]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][2] = B*(geom.gcon[0][2] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[2][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][2][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][2] = B*(geom.gcon[0][2] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[2][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][3] = D*geom.gcov[1][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][3][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][2] = B*(geom.gcon[0][2] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[2][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][3] = -D*geom.gcov[3][1];
		Mud[nl[n]][index_2D(n, i, j, z)][2][1][1] = C;

		E = geom.gcon[0][2] * geom.gcon[2][3] - geom.gcon[2][2] * geom.gcon[0][3];
		F = geom.gcon[0][2] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[2][3];
		G = geom.gcon[0][2] * geom.gcon[2][1] - geom.gcon[2][2] * geom.gcon[0][1];
		H = geom.gcon[0][2] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[2][1];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][0] = B*geom.gcon[0][2];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][2] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][2][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][2] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[1][1]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][3] = 1. / (D*geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][3][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[3][1] / geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][2] = (B*B / C)*geom.gcon[0][0] * (H + F*geom.gcov[3][1] / geom.gcov[1][1]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][3] = (1. / C)*geom.gcov[3][1] / geom.gcov[1][1];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][2][1][1] = 1. / C;

		//dir 3
		get_geometry(n, i, j, z, FACE3, &geom);
		A = -pow(-geom.gcon[0][0], -0.5);
		B = pow((geom.gcon[0][0])*(geom.gcon[0][0] * geom.gcon[3][3] - geom.gcon[0][3] * geom.gcon[0][3]), -0.5);
		C = pow(geom.gcov[2][2], -0.5);
		D = pow((geom.gcov[2][2])*(geom.gcov[1][1] * geom.gcov[2][2] - geom.gcov[1][2] * geom.gcov[1][2]), -0.5);
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][0] = A*geom.gcon[0][0];
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][3] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][0][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][0] = A*geom.gcon[0][3];
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][3] = B*(geom.gcon[0][3] * geom.gcon[0][3] - geom.gcon[0][0] * geom.gcon[3][3]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][1] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][3][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][0] = A*geom.gcon[0][1];
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][3] = B*(geom.gcon[0][3] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[3][1]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][1] = D*geom.gcov[2][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][1][2] = 0;
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][0] = A*geom.gcon[0][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][3] = B*(geom.gcon[0][3] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[3][2]);
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][1] = -D*geom.gcov[1][2];
		Mud[nl[n]][index_2D(n, i, j, z)][3][2][2] = C;

		E = geom.gcon[0][3] * geom.gcon[3][1] - geom.gcon[3][3] * geom.gcon[0][1];
		F = geom.gcon[0][3] * geom.gcon[0][1] - geom.gcon[0][0] * geom.gcon[3][1];
		G = geom.gcon[0][3] * geom.gcon[3][2] - geom.gcon[3][3] * geom.gcon[0][2];
		H = geom.gcon[0][3] * geom.gcon[0][2] - geom.gcon[0][0] * geom.gcon[3][2];

		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][0] = -A;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][3] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][0][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][0] = B*geom.gcon[0][3];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][3] = -B*geom.gcon[0][0];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][1] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][3][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][0] = B*B*E*geom.gcon[0][0] / (D*geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][3] = B*B*F*geom.gcon[0][0] / (D*geom.gcov[2][2]);;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][1] = 1. / (D*geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][1][2] = 0;
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][0] = (B*B / C)*geom.gcon[0][0] * (G + E*geom.gcov[1][2] / geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][3] = (B*B / C)*geom.gcon[0][0] * (H + F*geom.gcov[1][2] / geom.gcov[2][2]);
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][1] = (1. / C)*geom.gcov[1][2] / geom.gcov[2][2];
		Mud_inv[nl[n]][index_2D(n, i, j, z)][3][2][2] = 1. / C;
	}
}

void primtoflux_FT(double * restrict pr, double ucon[NDIM], double bcon[NDIM], int dir,double restrict flux[NPR])
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
	#pragma ivdep
	for (j = 1; j < NDIM;j++)flux[UU + j] = eta*ucon[dir] * ucon[j] + ptot*delta(dir, j) - bcon[dir] * bcon[j];

	/* dual of Maxwell tensor */
	#pragma ivdep
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

void vchar_FT(double * restrict pr, double ucon[NDIM], double bcon[NDIM], int dir, double  restrict *vmax, double restrict *vmin)
{
	double discr, vp, vm, bsq, EE, EF, va2, cs2, cms2;
	double Asq, Bsq, Au, Bu, Au2, Bu2, AuBu, A, B, C;
	int j;

	/* find fast magnetosonic speed */
	bsq = -bcon[0]*bcon[0]+bcon[1]*bcon[1] + bcon[2] * bcon[2] + bcon[3] * bcon[3];
	#if AMD
	EF = fma(gam, pr[UU], pr[RHO]);
	#else
	EF = pr[RHO] + GAMMA* pr[UU];
	#endif
	EE = bsq + EF;
	cs2 = GAMMA*(GAMMA - 1.)* pr[UU] / EF;
	va2 = bsq / EE;

	/* find fast magnetosonic speed */
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
	B = 2.* fma(-( AuBu), cms2, AuBu);
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

/***********************************************************************************************/
/***********************************************************************************************
fluxcalc():
---------
-- sets the numerical fluxes, avaluated at the cell boundaries using the slope limiter
slope_lim();

-- only has HLL and Lax-Friedrichs  approximate Riemann solvers implemented;

***********************************************************************************************/
double fluxcalc_hlld(double(*restrict pr[NB_LOCAL])[NPR], double(*restrict F[NB_LOCAL])[NPR], int dir, int flag, int n)
{
	int i, j, z, k, idel, jdel, zdel, face, i1, j1, i2, j2;
	double p_l[NPR], p_r[NPR], F_l[NPR], F_r[NPR], U_l[NPR], U_r[NPR], F_HLL[2][NPR], U_HLL[NPR], vcon, U_i[NPR], F1[NPR], F_FT[2][NPR], ptot, bcon[NDIM], ucon[NDIM], l_bcon[NDIM], l_ucon[NDIM], r_bcon[NDIM], r_ucon[NDIM], R_l[NPR], R_r[NPR];
	double cmax_l, cmax_r, cmin_l, cmin_r, cmax[NDIM], cmin[NDIM], cmax_roe, cmin_roe, ndt, ndt_thread, dtij;
	double ctop;
	struct of_geom geom;
	struct of_state state_l, state_r, state_l_FT, state_r_FT, state_roe;
	struct of_trans trans;
	double bsq;
	int max_i, max_j, max_z;
	double val;
	ndt = 1.e9;
	int ind0, ind1;
	int fail_HLLC = 0;
	int fail_HLLD;
	int counter0 = 0;
	int counter1 = 0;
	double test;

	if (dir == 1) { idel = 1; jdel = 0; zdel = 0;  face = FACE1; }
	else if (dir == 2) { idel = 0; jdel = 1; zdel = 0; face = FACE2; }
	else if (dir == 3) { idel = 0; jdel = 0; zdel = 1; face = FACE3; }
	else { exit(10); }

	double A, B, C, D, v_dot_B;
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
	int N_HLL = 0;
	int N_HLLC = 0;
	int N_HLLD = 0;

	#pragma omp parallel private(i,j,z,k, ndt_thread, p_l, p_r, geom, state_l, state_r, state_roe, state_l_FT, state_r_FT,F_l, F_r,U_l, U_r, cmax_l, cmax_r, cmin_l, cmin_r, cmax, cmin, cmax_roe, cmin_roe, ctop, dtij, ind0, ind1, U_HLL, F_HLL, vcon, A, B, C, D, v_dot_B, U_i, bsq, fail_HLLC,fail_HLLD, test, ptot, F_FT,i1,i2,j1,j2,bcon, ucon, l_bcon, l_ucon, r_bcon, r_ucon, trans, F1, R_l, R_r)
	{
		ndt_thread = 1.e9;

		/* then evaluate slopes */
		#pragma omp for collapse(2) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - D1, N1_GPU_offset[n] + BS_1 - 1 + D1, N2_GPU_offset[n] - D2, N2_GPU_offset[n] + BS_2 - 1 + D2, N3_GPU_offset[n] - D3, N3_GPU_offset[n] + BS_3 - 1 + D3) {
			// #pragma ivdep
			PLOOP{
				dq[nl[n]][index_3D(n, i, j, z)][k] = slope_lim(pr[nl[n]][index_3D(n, i - idel, j - jdel, z - zdel)][k], pr[nl[n]][index_3D(n, i, j, z)][k], pr[nl[n]][index_3D(n, i + idel, j + jdel, z + zdel)][k]);
			}
		}

		#pragma omp for collapse(2) schedule(static,(BS_1+jdel+zdel+1)*(BS_2+idel+zdel+1)/nthreads)
		ZSLOOP((N1_GPU_offset[n] - jdel - zdel)*D1, (N1_GPU_offset[n] + BS_1)*D1, (N2_GPU_offset[n] - idel - zdel)*D2, (N2_GPU_offset[n] + BS_2)*D2) {
			for (z = (N3_GPU_offset[n] - idel - jdel)*D3; z <= (N3_GPU_offset[n] + BS_3)*D3; z++) {
				fail_HLLD = 0;
				fail_HLLC = 0;
				get_geometry(n, i, j, z, face, &geom);
				get_trans(n, i, j, z, dir, &trans);

				ind0 = index_3D(n, i, j, z);
				ind1 = index_3D(n, i - idel, j - jdel, z - zdel);

				#pragma ivdep
				PLOOP{
					p_l[k] = pr[nl[n]][ind1][k] + 0.5*dq[nl[n]][ind1][k];
					p_r[k] = pr[nl[n]][ind0][k] - 0.5*dq[nl[n]][ind0][k];
					#if(STAGGERED)
					if ((dir == 1 && k == B1)) {
						if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
						else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
						p_r[k] = p_l[k];
					}
					if ((dir == 2 && k == B2)) {
						if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
						else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
						p_r[k] = p_l[k];
					}
					if ((dir == 3 && k == B3)) {
						if (flag == 0) p_l[k] = ps[nl[n]][ind0][k - (B1 - 1)];
						else p_l[k] = psh[nl[n]][ind0][k - (B1 - 1)];
						p_r[k] = p_l[k];
					}
					#endif
				}

				#if(STAGGERED)
				if ((dir == 2) && ((j == 0 && (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3)) || (j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2])) && (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3)))) {
					p_r[B1] = 0.;
					p_l[B1] = 0.;
				}
				#endif

				//First calculate HLL fluxes for F[B1], F[B2] and F[B3]
				get_state(p_l, &geom, &state_l);
				get_state(p_r, &geom, &state_r);

				vchar(p_l, &state_l, &geom, dir, &(cmax_l), &(cmin_l), i, j, z);
				vchar(p_r, &state_r, &geom, dir, &(cmax_r), &(cmin_r), i, j, z);
				cmax[1] = fabs(MY_MAX(MY_MAX(0., cmax_l), cmax_r));
				cmin[1] = fabs(MY_MAX(MY_MAX(0., -cmin_l), -cmin_r));
				ctop = MY_MAX(cmax[1], cmin[1]);

				//Transform 4 velocities and 4 magnetic fields to orthonormal frame
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
				double int_velocity = geom.gcon[0][dir] / (sqrt(geom.gcon[0][dir] * geom.gcon[0][dir] - geom.gcon[0][0] * geom.gcon[dir][dir]));

				//Get HLL fluxes and conserved states if flow is superfast
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
					for (k = 0; k < NPR; k++) F_HLL[1][k] = HLLF*((cmax_roe * F_l[k] - cmin_roe * F_r[k] + cmax_roe * cmin_roe * (U_r[k] - U_l[k])) / (cmax_roe - cmin_roe + SMALL));

					int do_hydro = (fabs(F_HLL[0][dir + B1 - 1] * F_HLL[0][dir + B1 - 1] * l_ucon[0] * r_ucon[0]) < pow(10., -8.)*fabs(F_HLL[0][UU]));

					if (do_hydro) {
						calc_HLLC_hydro(dir, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, F_FT, F_HLL, F_l, F_r, U_l, U_r);
						for (k = 0; k < NPR; k++) {
							F_FT[0][k] = F_HLL[0][k];
							F_FT[1][k] = F_HLL[1][k];
						}
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

				//Normalize with gdet
				PLOOP F[nl[n]][ind0][k] = geom.g*F1[k];

				// evaluate restriction on timestep
				dtij = fabs(cour*dx[nl[n]][dir] / ctop);
				if (dtij < ndt_thread) {
					ndt_thread = dtij;
				}

				#if(!TRANS_BOUND && !CARTESIAN)
				if (dir == 2 && (j == 0 || j == N2 * pow(1 + REF_2, block[n][AMR_LEVEL]))) {
					//#pragma ivdep
					PLOOP F[nl[n]][ind0][k] = 0.;
				}
				#endif
			}
		}
		#pragma omp critical
		{
			if (ndt_thread < ndt) {
				ndt = ndt_thread;
			}
		}
	}
	//if (nstep % 640 == AMR_MAXTIMELEVEL - 1)printf("N_HLL: %d N_HLLC: %d N_HLLD :%d \n", N_HLL, N_HLLC, N_HLLD);
	return(ndt);
}

void calc_HLLC_hydro(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double A, B, C, D, vcon, ptot;
	int k, fail_HLLC=0;
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
	if (!(fabs(ptot) > 0.) || (vcon < cmin_roe) || (vcon > cmax_roe) || !(fabs(vcon) > 0.)) {
		fail_HLLC = 1;
	}

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

void calc_HLLC(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
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

	//Convert to 4-velocity
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

void calc_HLLD(int dir, double cmin_roe, double cmax_roe, double int_velocity, double l_ucon[NDIM], double r_ucon[NDIM], double F_FT[2][NPR], double F_HLL[2][NPR], double F_l[NPR], double F_r[NPR], double U_l[NPR], double U_r[NPR]) {
	double K_al[NDIM], B_al[NDIM], K_ar[NDIM], B_ar[NDIM], vcon_al[NDIM], vcon_ar[NDIM], eta_l, eta_r, w_al, w_ar, vcon_cl[NDIM], vcon_cr[NDIM], B_c[NDIM], R_l[NPR], R_r[NPR], ptot;
	int k, fail_HLLC=0, fail_HLLD=0;

	for (k = 0; k < NPR; k++) R_l[k] = (cmin_roe*U_l[k] - F_l[k]);
	for (k = 0; k < NPR; k++) R_r[k] = (cmax_roe*U_r[k] - F_r[k]);

	//Calculate necessary pressure using Newton Raphson solve
	ptot=calc_HLLD_pres(dir, &fail_HLLC, &fail_HLLD, l_ucon, r_ucon, int_velocity, cmin_roe, cmax_roe, K_al, B_al, K_ar, B_ar, vcon_al, vcon_ar, &eta_l, &eta_r, &w_al, &w_ar, vcon_cl, vcon_cr, F_FT, F_HLL, F_l, F_r, U_l, U_r, R_l, R_r, B_c);

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

double calc_HLLD_pres(int dir, int *fail_HLLC, int *fail_HLLD, double l_ucon[NDIM], double r_ucon[NDIM], double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
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
	vcon[GEN_1] = (-B - sqrt(D)) / (2.*A);

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
		ptot = (-B + sqrt(D)) / (2.*A);
		if (!(fabs(ptot) > 0.)) {
			fail_HLLD[0] = 1;
			return -10.;
		}
	}

	//Newton Raphson loop to find pressure of intermediate states in HLLD solver
	double error_1, error_2;
	double ptot_old, de_dptot, de_dlptot, dlptot, lptot, d_ptot = 0.;
	error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, eta_l, eta_r, w_al, w_ar);

	while (keep_iterating) {
		//Calculate error and error/d_ptot
		error_2 = calc_error_HLLD(dir, 0, ptot + pow(10., -11.)*(ptot+ F_HLL[0][UU]), cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, eta_l, eta_r, w_al, w_ar);

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
		error_1 = calc_error_HLLD(dir, 0, ptot, cmin_roe, cmax_roe, F_HLL[0][BGEN_1], R_l, R_r, B_al, B_ar, B_c, vcon_al, vcon_ar, K_al, K_ar, eta_l, eta_r, w_al, w_ar);

		if ((fabs(d_ptot) <= pow(10., -7.) * (fabs(ptot) + fabs(F_HLL[0][UU]))) || n_iter > 10) {
			keep_iterating = 0;
		}

		n_iter++;
	}

	//If Newton-Raphson solver did not converge, reset ptot to ptot_HLLC and tag fail_HLLD
	if (!(fabs(ptot) > 0.) || (fabs(d_ptot) > pow(10., -6.)*fabs(ptot))) {
		ptot = ptot_HLLC;
		fail_HLLD[0] = 1;
	}

	return ptot;
}

void calc_HLLD_state(int dir, double l_ucon[NDIM], double r_ucon[NDIM], double ptot, double int_velocity, double cmin_roe, double cmax_roe, double K_al[NDIM],
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

	if ((cmax_roe > int_velocity) && (vcon_cr[dir] <= int_velocity)) {
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
void check_HLLD_par(int dir, int * fail_HLLD, double cmin_roe, double cmax_roe, double ptot, double w_al, double w_ar, double eta_l, double eta_r, double vcon_cl[NDIM], double vcon_cr[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double B_c[NDIM]) {
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

	//Calculate state around contact discontiuity according to equations 47, 50-52
	vcon_cl[GEN_1] = (K_al[GEN_1] - (B_c[GEN_1] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_2] = (K_al[GEN_2] - (B_c[GEN_2] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cl[GEN_3] = (K_al[GEN_3] - (B_c[GEN_3] * (1. - (K_al[1] * K_al[1] + K_al[2] * K_al[2] + K_al[3] * K_al[3]))) / (eta_l - (K_al[1] * B_c[1] + K_al[2] * B_c[2] + K_al[3] * B_c[3])));
	vcon_cr[GEN_1] = (K_ar[GEN_1] - (B_c[GEN_1] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_2] = (K_ar[GEN_2] - (B_c[GEN_2] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));
	vcon_cr[GEN_3] = (K_ar[GEN_3] - (B_c[GEN_3] * (1. - (K_ar[1] * K_ar[1] + K_ar[2] * K_ar[2] + K_ar[3] * K_ar[3]))) / (eta_r - (K_ar[1] * B_c[1] + K_ar[2] * B_c[2] + K_ar[3] * B_c[3])));

	vcon_cl[dir] = (vcon_cl[dir] + vcon_cr[dir])*0.5;
	vcon_cr[dir] = vcon_cl[dir];

	//Check that contact wave lies between inner and outer Alfven speed
	if (vcon_cl[dir] < K_al[dir] || vcon_cl[dir] < cmin_roe) fail_HLLD[0] = 1;
	if (vcon_cl[dir] > K_ar[dir] || vcon_cl[dir] > cmax_roe) fail_HLLD[0] = 1;

	//Check that contact wave is going slower than v=0.99c
	vsq = (vcon_cl[1] * vcon_cl[1]+ vcon_cl[2] * vcon_cl[2]+ vcon_cl[3] * vcon_cl[3]);
	if (!(vsq < 0.999)) fail_HLLD[0] = 1;

	//If wavefan inconsisten revert to HLLC
	if (fabs(w_al) <= fabs(ptot) || vcon_al[dir] <= cmin_roe || K_al[dir] <= cmin_roe || w_al <= 0.) fail_HLLD[0] = 1;
	if (fabs(w_ar) <= fabs(ptot) || vcon_ar[dir] >= cmax_roe || K_ar[dir] >= cmax_roe || w_ar <= 0.) fail_HLLD[0] = 1;

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

double calc_error_HLLD(int dir, int do_hydro, double ptot, double cmin_roe, double cmax_roe, double BX, double R_l[NPR], double R_r[NPR], double B_al[NDIM], double B_ar[NDIM], double B_c[NDIM], double vcon_al[NDIM], double vcon_ar[NDIM], double K_al[NDIM], double K_ar[NDIM], double *eta_l, double *eta_r, double  *w_al, double *w_ar) {
	int GEN_1, GEN_2, GEN_3, UGEN_1, UGEN_2, UGEN_3, BGEN_1, BGEN_2, BGEN_3;
	double A, C, G, X, Q, error = 0.;
	double delta_Kx, B_hat[NDIM], Y_r, Y_l;

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

	return error;
}

double NewtonRaphson(double start, int max_count, int dir, double *  ucon, double *  bcon, double E, double vasq, double csq)
{
	int count = 0;
	int keep_looping = 1;
	double dx;
	double x;
	double error_1, error_2, derror_dx;

	x = start;
	error_1 = Drel(dir, x, ucon, bcon, E, vasq, csq);

	while(keep_looping){
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

double Drel(int dir, double v, double *  ucon, double *  bcon, double E, double vasq, double csq) {
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
