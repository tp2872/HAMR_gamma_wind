__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NPR], double *  ps[3], double *  send, const  double* __restrict__ gdet_GPU[NPG], int work_size);
__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, double *  pv[NPR], double *  ph[NPR],
	double *  ps[3], double *  psh[3], double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, double *  pv[NPR], double *  ph[NPR],
	double *  ps[3], double *  psh[3], double *  receive, double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, double *  pv[NPR], double *  ph[NPR],
	double *  ps[3], double *  psh[3], double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double * p, double * ph, double * ps[3], double * psh[3], double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double * p, double * ph, double * ps[3], double * psh[3], double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double * p, double * ph, double * ps[3], double * psh[3], double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU[NPG], int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NPR], double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NPR], double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NPR], double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NPR], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void packsendE1corn(int i1, int i2, int j, int z, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corn(int i, int j1, int j2, int z, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corn(int i, int j, int z1, int z2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE1corncourse(int i1, int i2, int j, int z, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corncourse(int i, int j1, int j2, int z, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corncourse(int i, int j, int z1, int z2, double *  pv[NDIM], double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, double *  prim[NDIM], double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);

__global__ void fluxcalcprep(const  double* __restrict__   F[NPR], double *  dq1[NPR], double *  dq2[NPR], const  double* __restrict__  pv[NPR], int dir, int lim, int number, const  double* __restrict__  V[NPR], int poststep_p);
__global__ void fluxcalc2D2(double *  F[NPR], const  double* __restrict__  dq1[NPR], const  double* __restrict__ dq2[NPR], const  double* __restrict__  pv[NPR], const  double* __restrict__  ps[3], const  double* __restrict__ gcov[10 * NPG], const  double* __restrict__ gcon[10 * NPG], const  double* __restrict__ gdet[NPG], int lim, int dir,
	double gam, double cour, double*  dtij, int POLE_1, int POLE_2, double dx_1, double dx_2, double dx_3, int poststep_p, int calc_time);
__global__ void fix_flux(double *  F1[NPR], double *  F2[NPR], double *  F3[NPR], int NBR_1, int NBR_2, int NBR_3, int NBR_4);
__global__ void consttransport1(const  double* __restrict__  pb_i[NPR], double *  E_cent, const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], int poststep_p);
__global__ void consttransport2(double *  emf[NDIM], const  double* __restrict__  E_cent, const  double* __restrict__  F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__  F3[NPR],
	const  double* __restrict__  pb_i[NPR], const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], int POLE_1, int POLE_2, int poststep_p);
__global__ void consttransport3(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU[NPG], double *  psi[3], double *  psf[3],
	const  double* __restrict__  E_corn[NDIM], double Dt, int poststep_p);
__global__ void consttransport3_post(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU[NPG], double *  psi[3], double *  psf[3],
	const  double* __restrict__  E_corn[NDIM], double Dt);
__global__ void flux_ct1(const  double* __restrict__  F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__  F3[NPR], double *  emf[NDIM]);
__global__ void flux_ct2(double *  F1[NPR], double *  F2[NPR], double *  F3[NPR], const  double* __restrict__  emf[NDIM]);
__global__ void Utoprim0(const  double* __restrict__ pi_i[NPR], const  double* __restrict__ pb_i[NPR], double* pf_i[NPR], double *  psf[3],
	const  double* __restrict__  F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__  F3[NPR], double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], const  double* __restrict__ conn[10*NDIM], double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void Utoprim1(double* pi_i[NPR], double* pb_i[NPR], double* pf_i[NPR], double *  psf[3],
	double *  F1[NPR], double *  F2[NPR], double *  F3[NPR], double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], const  double* __restrict__ conn[10*NDIM], double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void Utoprim2(double* __restrict__ pi_i[NPR], double* pb_i[NPR], double* pf_i[NPR], const  double* __restrict__  psf[3],
	const  double* __restrict__  F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__  F3[NPR], double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], const  double* __restrict__ conn[10*NDIM], double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void fixup(double* pi_i[NPR], double* pb_i[NPR], double* pf_i[NPR], double* storage_2[NPR], const  double* __restrict__  psf[3],
	const  double* __restrict__ F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__ F3[NPR], const  double* __restrict__ U_i[NPR], const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], const  double* __restrict__ conn[10*NDIM], double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void fixup_post(double* pi_i[NPR], double* pb_i[NPR], double* pf_i[NPR], const  double* __restrict__  psf[3],
	const  double* __restrict__ F1[NPR], const  double* __restrict__  F2[NPR], const  double* __restrict__ F3[NPR], const  double* __restrict__ U_i[NPR], const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], const  double* __restrict__ conn[10*NDIM], double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step);
__global__ void cleanup_post(double* F1[NPR], double* F2[NPR], double* F3[NPR], double* E_corn[NDIM]);
__global__ void fixuputoprim(double *  pv[NPR], int *  pflag, int *  failimage, const  double* __restrict__ gcov[10*NPG], const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet);
__global__ void boundprim1(double * pv[NPR], const  double* __restrict__ gcov[10*NPG],const  double* __restrict__ gcon[10*NPG], const  double* __restrict__ gdet[NPG], int NBR_2, int NBR_4, double * ps[3]);
__global__ void boundprim2(double * pv[NPR], const  double* __restrict__ gdet[NPG], int NBR_1, int NBR_3, double * ps[3]);
__global__ void boundprim_trans(double * pv[NPR], const  double* __restrict__ gdet[NPG], int NBR_1, int NBR_3, double * ps[3]);

		