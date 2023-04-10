#include "config.h"
#if(SCUDA)
#include <cuda.h>
#elif(SHIP)
#include "hip/hip_runtime.h"
#endif
#include <stdio.h>

//Include declerations
#include "GPU_decs.h"

//Include radiation functions
#include "GPU_radiation.h"

//Include neutrino radiation functions
#include "GPU_radiation_nu.h"

//Include 2T functions
#include "GPU_2T.h"

//Include Helmholtz functions
#include "GPU_eos.h"

//Include matrix inversion functions
#include "GPU_inv.h"

//Include Bart's resistive inversion
#include "GPU_utoprim_3d_res.h"

//Include Danat's radiative inversion
#include "GPU_utoprim_3d_T.h"

//Include newman inversion
#include "GPU_utoprim_nm.h"

//Include Noble's 1D inversion
#include "GPU_utoprim_1dfix1.h"

//Include Noble's 1DVSQ2 inversion
#include "GPU_utoprim_1dvsq2fix1.h"

//Include Noble's 2D inversion
#include "GPU_utoprim_2d.h"

//Include physics functions
#include "GPU_phys.h"

//Include resistive physics functions
#include "GPU_phys_res.h"

//Include radiation physics functions
#include "GPU_phys_rad.h"

//Include neutrino radiation physics functions
#include "GPU_phys_nu.h"

//Include HLLC/HLLD solver
#include "GPU_hllc.h"

//Include fixup functions
#include "GPU_fixup.h"

//Include metric functions
#include "GPU_metric.h"

//Include boundary functions
#include "GPU_bounds.h"

//Include interpolation functions
#include "GPU_interp.h"

//Include nuclear physics functions
#include "GPU_nuclear.h"

//Include interpolation functions
#include "GPU_const_trans.h"

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

__global__ void fluxcalc2D2(double *  F, const  double* __restrict__  dq1, const  double* __restrict__ dq2, const  double* __restrict__  pv, const  double* __restrict__  ps, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int lim, int dir, double cour, double*  dtij, int POLE_1, int POLE_2, double dx, int calc_time, int flag
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if (NEUTRINOS_M1)
    , const  double* __restrict__ gpu_nulib_table
    #endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
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
	#if(NEUTRINOS_M1)
    double cmax_r_nu[NU_SPECIES], cmin_r_nu[NU_SPECIES], cmax_l_nu[NU_SPECIES], cmin_l_nu[NU_SPECIES], cmax_nu[NU_SPECIES], cmin_nu[NU_SPECIES], ctop_nu[NU_SPECIES];
    struct of_state_nu state_nu[NU_SPECIES];
    int sp;
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
		get_state_res(p, &geom, &state
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		primtoflux_res(p, &state, dir, &geom, temp1);
		primtoflux_res(p, &state, 0, &geom, temp2);
		vchar_res(&geom, dir, &cmax_l, &cmin_l);
		#else
		get_state(p, &geom, &state
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
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
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);

		#endif

		#if(NEUTRINOS_M1)
        for (sp = 0; sp < NU_SPECIES; sp++) get_state_nu(p, &geom, &state_nu[sp], sp);
        primtoflux_nu(p, state_nu, dir, &geom, temp1);
        primtoflux_nu(p, state_nu, 0, &geom, temp2);
        #if (NU_NUMBER_DENSITY_FLUID_EVOLVE)
        //primtoflux_nu_number(p, state.ucon, state.ucov, dir, &geom, temp1);
        //primtoflux_nu_number(p, state.ucon, state.ucov, 0, &geom, temp2);
        #endif
        vchar_nu(p, &state, state_nu, &geom, dir, &cmax_l_nu[0], &cmin_l_nu[0], factor/cour, gpu_eos_table, gpu_nulib_table);
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
		get_state_res(p, &geom, &state
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		primtoflux_res(p, &state, dir, &geom, temp3);
		primtoflux_res(p, &state, 0, &geom, temp4);
		vchar_res(&geom, dir, &cmax_r, &cmin_r);
		#else
		get_state(p, &geom, &state
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
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
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
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
		#elif(NEUTRINOS_M1)
        for (sp = 0; sp < NU_SPECIES; sp++) get_state_nu(p, &geom, &state_nu[sp], sp);
        primtoflux_nu(p, state_nu, dir, &geom, temp3);
        primtoflux_nu(p, state_nu, 0, &geom, temp4);
        #if (NU_NUMBER_DENSITY_FLUID_EVOLVE)
        //primtoflux_nu_number(p, state.ucon, state.ucov, dir, &geom, temp3);
        //primtoflux_nu_number(p, state.ucon, state.ucov, 0, &geom, temp4);
        #endif

        vchar_nu(p, &state, state_nu, &geom, dir, &cmax_r_nu[0], &cmin_r_nu[0], factor / cour, gpu_eos_table, gpu_nulib_table);
        for (sp = 0; sp < NU_SPECIES; sp++) {
            cmax_nu[sp] = fabs(MY_MAX(MY_MAX(0., cmax_l_nu[sp]), cmax_r_nu[sp]));
            cmin_nu[sp] = fabs(MY_MAX(MY_MAX(0., -cmin_l_nu[sp]), -cmin_r_nu[sp]));
            ctop_nu[sp] = MY_MAX(cmax_nu[sp], cmin_nu[sp]);
            // Danat debug:
            //ctop_nu[sp] = 1.0;
        }

        for (k = 0; k < NPR; k++) {
            if (k == UU_NU || k == U1_NU || k == U2_NU || k == U3_NU || k == NUMBER_NU) {
                F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_nu[0] * (temp4[k] - temp2[k]));
            }
            #if (NU_SPECIES > 1)
            else if (k == index_nu(UU_NU, 1) || k == index_nu(U1_NU, 1) || k == index_nu(U2_NU, 1) || k == index_nu(U3_NU, 1) || k == index_nu(NUMBER_NU, 1)) {
                F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_nu[1] * (temp4[k] - temp2[k]));
            }
            else if (k == index_nu(UU_NU, 2) || k == index_nu(U1_NU, 2) || k == index_nu(U2_NU, 2) || k == index_nu(U3_NU, 2) || k == index_nu(NUMBER_NU, 2)) {
                F[k * (ksize)+global_id] = 0.5 * (temp1[k] + temp3[k] - ctop_nu[2] * (temp4[k] - temp2[k]));
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
        #if(NEUTRINOS_M1)
			#if (NU_SPECIES > 1)
			ctop = MY_MAX(ctop, MY_MAX(ctop_nu[0], MY_MAX(ctop_nu[1], ctop_nu[2])));
			#else
			ctop = MY_MAX(ctop, ctop_nu[0]);
			#endif
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

__global__ void Utoprim_M1_0( double* p_i, double* U_n, double* U_0, double* dU_RAD0, const  double* __restrict__ radius, int* pflag, int* pflag_rad, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, double dx_1, double dx_2, double dx_3, double Dt, double y_max, int POLE_1, int POLE_2
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale, double magnetic_density_scale
	#endif
	#if(CARTESIAN_GR)
	, int* pflag_cart
	#endif
)
{
#if(DO_IMEX)
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
	#if(TWO_T)
	double gamma_g, fel;
	#endif

	#if(CARTESIAN_GR)
	if (k == 1 && pflag_cart[global_id] == 1) k = 0;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			p[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				p[k] += (1.0 / ((double)zsize)) * p_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(p, &geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
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
		implicit_rad_solve(p, U, U, UU0, &pflag[global_id], &pflag_rad[global_id], &geom, dU, Dt * Y_IMEX, cell_size, y_max
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, 0.0
			#endif
			#if(COOL_STOP)
			, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
			#endif
			#if(CALC_MDOT)
			,  mass_density_scale
			#endif
		);

		#if(CALC_MDOT)
		p[B1] /= magnetic_density_scale;
		p[B2] /= magnetic_density_scale;
		p[B3] /= magnetic_density_scale;
		#endif

		#pragma unroll 9	
		for (k = 0; k < NPR; k++) {
			U_n[k * (ksize)+global_id] = U[k];
			U_0[k * (ksize)+global_id] = UU0[k];
			dU_RAD0[k * (ksize)+global_id] = dU[k];
			p_i[k * (ksize)+global_id] = p[k];
		}
	}
	#endif
#endif
}

__global__ void Utoprim_M1_1(double* ph_i, const  double* __restrict__ p_i, const double* __restrict__ U_n, const double* __restrict__ U_0, double*  U_1, const double* __restrict__ dU_RAD0, double* dU_RAD1, const  double* __restrict__  psh, const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3,
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
)
{
#if(DO_IMEX)
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

	#if(CARTESIAN_GR)
	if (k == 1 && pflag_cart[global_id] == 1) k = 0;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			p[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				p[k] += (1.0 / ((double)zsize)) * p_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(p, &geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(p);
		#endif
		source(p, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
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
			UU1[B1] += (psh[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
			UU1[B2] += (psh[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psh[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
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

		#if(CALC_MDOT)
		UU1[B1] *= magnetic_density_scale;
		UU1[B2] *= magnetic_density_scale;
		UU1[B3] *= magnetic_density_scale;
		#endif

		//Set temporary variable
		for (k = 0; k < NPR; k++) U_n_tmp[k] = UU1[k];

		//Perform implicit solve
		#if(TWO_T)
		fel = calc_delta(p, dot(q.bcon, q.bcov));
		#endif
		cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(p, U_n_tmp, U_n_tmp, UU1, &pflag[global_id], &pflag_rad[global_id], &geom, dU, Y_IMEX * Dt, cell_size, y_max
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, fel
			#endif
			#if(COOL_STOP)
			, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
			#endif
			#if(CALC_MDOT)
			,  mass_density_scale
			#endif
		);

		#if(CALC_MDOT)
		p[B1] /= magnetic_density_scale;
		p[B2] /= magnetic_density_scale;
		p[B3] /= magnetic_density_scale;
		#endif

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(p, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(CALC_MDOT)
			, magnetic_density_scale
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
#endif
}

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
)
{
#if(DO_IMEX)
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

	#if(CARTESIAN_GR)
	if (k == 1 && pflag_cart[global_id] == 1) k = 0;
	#endif

	if (k == 1) {
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		for (k = 0; k < NPR; k++) {
			ph[k] = 0.0;
			for (u = 0; u < zsize; u++) {
				ph[k] += (1.0 / ((double)zsize)) * ph_i[k * (ksize)+global_id - zoffset + u];
			}
		}

		get_state(ph, &geom, &q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		#if(TWO_T)
		gamma_g = calc_gamma_gas_prim(ph);
		#endif
		source(ph, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
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
			U_2[B1] += (ps[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + ps[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
			U_2[B2] += (ps[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + ps[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
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

		#if(CALC_MDOT)
		U_2[B1] *= magnetic_density_scale;
		U_2[B2] *= magnetic_density_scale;
		U_2[B3] *= magnetic_density_scale;
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
				#if(!TWO_T)
				pflag[global_id] = Utoprim_1dfix1(U_2, geom.gcov, geom.gcon, geom.g, ph, NEWT_TOL, BASIC, FULL_ENTROPY
					#if(TWO_T)
					, fel
					#endif
				);
				#endif
				if (pflag[global_id]) {
					pflag[0] = global_id;
					failimage[2 * (ksize)+global_id]++;
				}
			}
		}
		#endif
		pflag_rad[global_id] = Rtoprim(U_2, geom.gcov, geom.gcon, geom.g, ph, y_max, BASIC
			#if(CALC_MDOT)
			, mass_density_scale, magnetic_density_scale
			#endif
		);

		#if(CALC_MDOT)
		ph[B1] /= magnetic_density_scale;
		ph[B2] /= magnetic_density_scale;
		ph[B3] /= magnetic_density_scale;
		#endif

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(ph, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(CALC_MDOT)
			, magnetic_density_scale
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
#endif
}

