__global__ void packsend1(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend2(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend3(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage1(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage2(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage3(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void unpackreceive1(int n_blocks, int n, int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive2(int n_blocks, int n, int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive3(int n_blocks, int n, int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse1(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse2(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse3(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void packsend1flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive2flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive3flux(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void packsendfluxaverage1(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage2(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage3(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend1E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage1(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage2(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage3(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive2E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive3E(int n_blocks, int n, int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void packsendE1corn(int n_blocks, int n, int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corn(int n_blocks, int n, int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corn(int n_blocks, int n, int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE1corncourse(int n_blocks, int n, int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corncourse(int n_blocks, int n, int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corncourse(int n_blocks, int n, int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceiveE1corn(int n_blocks, int n, int i1, int i2, int j, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE2corn(int n_blocks, int n, int i, int j1, int j2, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE3corn(int n_blocks, int n, int i, int j, int z1, int z2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);

__global__ void fluxcalcprep(int n_blocks, int n, int **block, const  double* __restrict__   F, double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int lim, int number, const  double* __restrict__  V, int poststep_p);
__global__ void fluxcalc2D2(int n_blocks, int n, int **block, double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir,
	double gam, double cour, double*  dtij, int POLE_1, int POLE_2, double dx_1, double dx_2, double dx_3, int poststep_p, int calc_time);
__global__ void fix_flux(int n_blocks, int n, int **block, double *  F1, double *  F2, double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4);
__global__ void consttransport1(int n_blocks, int n, int **block, const  double* __restrict__  pb_i, double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int poststep_p);
__global__ void consttransport2(int n_blocks, int n, int **block, double *  emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2, int poststep_p);
__global__ void consttransport3(int n_blocks, int n, int **block, double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int poststep_p);
__global__ void consttransport3_post(int n_blocks, int n, int **block, double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt);
__global__ void flux_ct1(int n_blocks, int n, int **block, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double *  emf);
__global__ void flux_ct2(int n_blocks, int n, int **block, double *  F1, double *  F2, double *  F3, const  double* __restrict__  emf);
__global__ void Utoprim0(int n_blocks, int n, int **block, const  double* __restrict__ pi_i, const  double* __restrict__ pb_i, double* pf_i, double *  psf,
	const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void Utoprim1(int n_blocks, int n, int **block, double* pi_i, double* pb_i, double* pf_i, double *  psf,
	double *  F1, double *  F2, double *  F3, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void Utoprim2(int n_blocks, int n, int **block, double* __restrict__ pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double* U_i, double* radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void fixup(int n_blocks, int n, int **block, double* pi_i, double* pb_i, double* pf_i, double* storage_2, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step, int poststep_p);
__global__ void fixup_post(int n_blocks, int n, int **block, double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int full_step);
__global__ void cleanup_post(int n_blocks, int n, int **block, double* F1, double* F2, double* F3, double* E_corn);
__global__ void fixuputoprim(int n_blocks, int n, int **block, double *  pv, int *  pflag, int *  failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet);
__global__ void boundprim1(int n_blocks, int n, int **block, double *   pv, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double *  ps);
__global__ void boundprim2(int n_blocks, int n, int **block, double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps);
__global__ void boundprim_trans(int n_blocks, int n, int **block, double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps);

		