#include "config.h"

__global__ void packsend1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size);
__global__ void packsend3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size,int POLE_1, int POLE_2);
__global__ void packsendaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  ps, double *  send, const  double* __restrict__ gdet_GPU, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceive1(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int jsize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceive2(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int zsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int reverse, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceive3(int i1, int i2, int i_offset, int j1, int j2, int j_offset, int z1, int z2, int z_offset, int isize2, int jsize2, double *  p, double *  ph,
	double *  ps, double *  psh, double *  receive, double *  tempreceive, int update_staggered, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceivecoarse1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceivecoarse2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceivecoarse3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double * p, double * ph, double * ps, double * psh, double * prim, double * psim,
	double *  receive, double *  temp1receive, double *  temp2receive, const  double* __restrict__ gdet_GPU, int nstep, double dt, int timelevel, int timelevel_rec, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsend1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void unpackreceive1flux(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive2flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceive3flux(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  receive,
	double *  temp1, double *  temp2, int calc_corr, int nstep, int nstep2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void packsendfluxaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendfluxaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendfluxaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsend1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsend3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendEaverage1(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendEaverage2(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void packsendEaverage3(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1, int ref_2, int ref_3);
__global__ void unpackreceive1E(int i1, int i2, int j1, int j2, int z1, int z2, int jsize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive2E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int zsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void unpackreceive3E(int i1, int i2, int j1, int j2, int z1, int z2, int isize2, int jsize2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int d1, int d2, int e1, int e2, int work_size);
__global__ void packsendE1corn(int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE2corn(int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE3corn(int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size);
__global__ void packsendE1corncourse(int i1, int i2, int j, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_1);
__global__ void packsendE2corncourse(int i, int j1, int j2, int z, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_2);
__global__ void packsendE3corncourse(int i, int j, int z1, int z2, double *  pv, double *  send, double factor, int first_timestep, int work_size, int ref_3);
__global__ void unpackreceiveE1corn(int i1, int i2, int j, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE2corn(int i, int j1, int j2, int z, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void unpackreceiveE3corn(int i, int j, int z1, int z2, double *  prim, double *  receive, double *  temp1, double *  temp2,
	int calc_corr, int nstep, int nstep_2, int timelevel, int timelevel_rec, double factor, int work_size);
__global__ void fluxcalcprep(const  double* __restrict__   F, double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int lim, int number, const  double* __restrict__  V, int POLE_1, int POLE_2);
__global__ void interpolate(double *  dq1, double *  dq2, const  double* __restrict__  p, int dir, int POLE_1, int POLE_2);
__global__ void fluxcalc2D2(double* F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir, double cour, double* dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(NEUTRINOS_M1)
	, const double* __restrict__ gpu_nulib_table
	#endif	
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
);
__global__ void fluxcalc2D_FT(double* F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet,
	const  double* __restrict__ Mud_GPU, const  double* __restrict__ Mud_inv_GPU, int lim, int dir, double cour, double* dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
);
__global__ void fixup(double* pi_i, double* pb_i, double* pf_i, double* storage_2, const  double* __restrict__  psf, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2, double y_max
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if (NEUTRINOS_M1)
	, const  double* __restrict__ gpu_nulib_table, int* pflag_nu
	#if (NEUTRINOS_DEBUG)
	, double* allflags_nu
	#endif
	#endif
	#if(RAD_M1)
	, int *pflag_rad
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);
__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2	
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(RAD_M1)
	, int *pflag_rad
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);
__global__ void Utoprim_M1_0(double* p_i, double* U_n, double* U_0, double* dU_RAD0, const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);
__global__ void Utoprim_M1_1(double* ph_i, const  double* __restrict__ p_i, const double* __restrict__ U_n, const double* __restrict__ U_0, double* U_1, const double* __restrict__ dU_RAD0, double* dU_RAD1, const  double* __restrict__  psh, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);
__global__ void Utoprim_M1_2(const  double* __restrict__ ph_i, double* p_i, const double* __restrict__ U_n, const double* __restrict__ U_0, const double* __restrict__ U_1, const double* __restrict__ dU_RAD0, const double* __restrict__ dU_RAD1, const  double* __restrict__  ps, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
	const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif	
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);

__global__ void reconstruct_internal(double* p, double* ps, const  double* __restrict__ dq1, const  double* __restrict__ dq2, const  double* __restrict__ gdet, int POLE_1, int POLE_2);
__global__ void fix_flux(double *  F1, double *  F2, double *  F3, int NBR_1, int NBR_2, int NBR_3, int NBR_4);
__global__ void consttransport1(const  double* __restrict__  pb_i, double *  E_cent, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet);
__global__ void consttransport2(double *  emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2
	#if(CALC_MDOT)
	, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
);
__global__ void consttransport2_M1_2(double* emf, const  double* __restrict__  E_cent, const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3,
	const  double* __restrict__  pb_i, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int POLE_1, int POLE_2);
__global__ void consttransport3(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2);
__global__ void consttransport3_post(double dx_1, double dx_2, double dx_3, const  double* __restrict__ gdet_GPU, double *  psi, double *  psf,
	const  double* __restrict__  E_corn, double Dt, int POLE_1, int POLE_2);
__global__ void flux_ct1(const  double* __restrict__  F1, const  double* __restrict__  F2, const  double* __restrict__  F3, double *  emf);
__global__ void flux_ct2(double *  F1, double *  F2, double *  F3, const  double* __restrict__  emf);
__global__ void Utoprim0(void);
__global__ void cleanup_post(double* F1, double* F2, double* F3, double* E_corn);
__global__ void fixuputoprim(double* pv, const  double* __restrict__ radius, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int* pflag, int* failimage
#if (DOHELM)
	, const double* __restrict__ gpu_eos_table
#endif
#if(CALC_MDOT)
	, double mass_density_scale
	, double magnetic_density_scale
#endif
);
__global__ void fixuputoprim_rad(double* pv, int* pflag_rad, int* failimage);
__global__ void boundprim1_outflow(double* pv, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double* ps
	#if(DANAT_GDET_INTERP)	
	, const double* __restrict__ radius
	#endif
);
__global__ void boundprim2_outflow(double* pv, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double* ps);
__global__ void boundprim2_reflective(double* pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double* ps);
__global__ void boundprim2_trans(double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps);
__global__ void boundprim3_outflow(double* pv, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_5, int NBR_6, double* ps);
__global__ void boundprim_cart(double* pv, double* ps, int* pflag_cart, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet);
#if (NEUTRINOS_M1)
__global__ void fixuputoprim_nu(double* pv, int* pflag_nu, int* failimage);
#endif

		
