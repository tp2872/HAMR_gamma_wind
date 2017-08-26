__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  ps, __global double *  send, const  double* __restrict__ gdet_GPU);
__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, __global double *  p, __global double *  ph,
	__global double *  ps, __global double *  psh, __global double *  receive, __global double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, __global double *  p, __global double *  ph,
	__global double *  ps, __global double *  psh, __global double *  receive, __global double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, __global double *  p, __global double *  ph,
	__global double *  ps, __global double *  psh, __global double *  receive, __global double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double * p, __global double * ph, __global double * ps, __global double * psh, __global double * prim,
	__global double *  receive, __global double *  temp1receive, __global double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double * p, __global double * ph, __global double * ps, __global double * psh, __global double * prim,
	__global double *  receive, __global double *  temp1receive, __global double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double * p, __global double * ph, __global double * ps, __global double * psh, __global double * prim,
	__global double *  receive, __global double *  temp1receive, __global double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  receive,
	__global double *  temp1, __global double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  receive,
	__global double *  temp1, __global double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  receive,
	__global double *  temp1, __global double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  pv, __global double *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void packsendE1corn(int i1, int i2, int j, int z, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corn(int i, int j1, int j2, int z, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corn(int i, int j, int z1, int z2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE1corncourse(int i1, int i2, int j, int z, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corncourse(int i, int j1, int j2, int z, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corncourse(int i, int j, int z1, int z2, __global double *  pv, __global double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, __global double *  prim, __global double *  receive, __global double *  temp1, __global double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void fluxcalcprep(int N1, int N2, int N3, __global double *   F, __global double *  dq, __global double *  p, int dir, int lim, int number, __global double *  V);
__global__ void fluxcalc2D2(int N1, int N2, int N3, __global double *  F, __global double *  dq, __global double *  pv, __global double *  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir,
	double gam, double cour, __global double*  dtij, int POLE_1, int POLE_2, __global double* storage1, __global double* storage2, __global double* storage3, __global double* storage4, double dx_1, double dx_2, double dx_3);
__global__ void fix_flux(int N1, int N2, int N3, __global double *  F1, __global double *  F2, __global double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4);
__global__ void consttransport1(int N1, int N2, int N3, __global double *  pb_i, __global double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet);
__global__ void consttransport2(int N1, int N2, int N3, __global double *  emf, __global double *  E_cent, __global double *  F1, __global double *  F2, __global double *  F3,
	__global double *  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2);
__global__ void consttransport3(int N1, int N2, int N3, double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, __global double *  psi, __global double *  psf,
	__global double *  E_corn, double Dt);
__global__ void flux_ct1(int N1, int N2, int N3, __global double *  F1, __global double *  F2, __global double *  F3, __global double *  emf);
__global__ void flux_ct2(int N1, int N2, int N3, __global double *  F1, __global double *  F2, __global double *  F3, __global double *  emf);
__global__ void fixup(int N1, int N2, int N3, __global double* pi_i, __global double* pb_i, __global double* pf_i, __global double *  psf,
	__global double *  F1, __global double *  F2, __global double *  F3, __global double* radius, __global int* pflag, __global int* failimage,
	const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, __global double* Katm, double gam, double dx_1, double dx_2, double dx_3, double a, double Dt, int flag);
__global__ void fixuputoprim(int N1, int N2, int N3, __global double *  pv, __global int *  pflag, __global int *  failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet);
__global__ void boundprim1(int N1, int N2, int N3, __global double *   pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, __global double *  ps);
__global__ void boundprim2(int N1, int N2, int N3, __global double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, __global double *  ps);

		