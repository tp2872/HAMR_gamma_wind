__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU, int work_size);
__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU, int work_size);
__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU, int work_size);
__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  ps, __global FTYPE2 *  send, const  FTYPE2* __restrict__ gdet_GPU);
__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int reverse, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, __global FTYPE2 *  p, __global FTYPE2 *  ph,
	__global FTYPE2 *  ps, __global FTYPE2 *  psh, __global FTYPE2 *  receive, __global FTYPE2 *  tempreceive, int update_staggered, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 * p, __global FTYPE2 * ph, __global FTYPE2 * ps, __global FTYPE2 * psh, __global FTYPE2 * prim,
	__global FTYPE2 *  receive, __global FTYPE2 *  temp1receive, __global FTYPE2 *  temp2receive, const  FTYPE2* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size);
__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  receive,
	__global FTYPE2 *  temp1, __global FTYPE2 *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int mode, int work_size);
__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void packsendE1corn(int i1, int i2, int j, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corn(int i, int j1, int j2, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corn(int i, int j, int z1, int z2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE1corncourse(int i1, int i2, int j, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corncourse(int i, int j1, int j2, int z, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corncourse(int i, int j, int z1, int z2, __global FTYPE2 *  pv, __global FTYPE2 *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, __global FTYPE2 *  prim, __global FTYPE2 *  receive, __global FTYPE2 *  temp1, __global FTYPE2 *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void fluxcalcprep(int N1, int N2, int N3, __global FTYPE2 *   F, __global FTYPE2 *  dq, __global FTYPE2 *  p, int dir, int lim, int number, __global FTYPE2 *  V);
__global__ void fluxcalc2D2(int N1, int N2, int N3, __global FTYPE2 *  F, __global FTYPE2 *  dq, __global FTYPE2 *  pv, __global FTYPE2 *  ps, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, int lim, int dir,
	FTYPE2 gam, FTYPE2 cour, __global FTYPE2*  dtij, int POLE_1, int POLE_2, __global FTYPE2* storage1, __global FTYPE2* storage2, __global FTYPE2* storage3, __global FTYPE2* storage4, double dx_1, double dx_2, double dx_3);
__global__ void fix_flux(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4);
__global__ void consttransport1(int N1, int N2, int N3, __global FTYPE2 *  pb_i, __global FTYPE2 *  E_cent, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet);
__global__ void consttransport2(int N1, int N2, int N3, __global FTYPE2 *  emf, __global FTYPE2 *  E_cent, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3,
	__global FTYPE2 *  pb_i, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, int POLE_1, int POLE_2);
__global__ void consttransport3(int N1, int N2, int N3, double dx_1, double dx_2, double dx_3, const  FTYPE2* __restrict__ gdet_GPU, __global FTYPE2 *  psi, __global FTYPE2 *  psf,
	__global FTYPE2 *  E_corn, double Dt);
__global__ void flux_ct1(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2 *  emf);
__global__ void flux_ct2(int N1, int N2, int N3, __global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2 *  emf);
__global__ void fixup(int N1, int N2, int N3, __global FTYPE2* pi_i, __global FTYPE2* pb_i, __global FTYPE2* pf_i, __global FTYPE2 *  psf,
	__global FTYPE2 *  F1, __global FTYPE2 *  F2, __global FTYPE2 *  F3, __global FTYPE2* radius, __global int* pflag, __global int* failimage,
	const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, const  FTYPE2* __restrict__ conn, __global FTYPE2* Katm, FTYPE2 gam, FTYPE2 dx_1, FTYPE2 dx_2, FTYPE2 dx_3, FTYPE2 a, FTYPE2 Dt, int flag);
__global__ void fixuputoprim(int N1, int N2, int N3, __global FTYPE2 *  pv, __global int *  pflag, __global int *  failimage, const  FTYPE2* __restrict__ gcov, const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet);
__global__ void boundprim1(int N1, int N2, int N3, __global FTYPE2 *   pv, const  FTYPE2* __restrict__ gcov,const  FTYPE2* __restrict__ gcon, const  FTYPE2* __restrict__ gdet, int NBR_2, int NBR_4, __global FTYPE2 *  ps);
__global__ void boundprim2(int N1, int N2, int N3, __global FTYPE2 *  pv, const  FTYPE2* __restrict__ gdet, int NBR_1, int NBR_3, __global FTYPE2 *  ps);

		