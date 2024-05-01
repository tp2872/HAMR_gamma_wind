#include "config.h"


//For P100/V100 GPUs replace Utoprim0, Utoprim1, Utoprim2, fixup by this kernel
__global__ void fixup(double* pi_i, double* pb_i, double* pf_i, double* storage2, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2, double y_max
    #if (DOHELM)
    , const double* __restrict__ gpu_eos_table
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
	#if(NEUTRON_STAR)
	, double fixupWeight
	#if(!NS_TAPERED_FLOORS)
	, const  double* __restrict__ theta
	#endif
	#endif
)
{
#if(1)
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
	#if(NEUTRINOS_M1)
    int sp;
    #endif

	#if(N_LEVELS_1D_INT>0 && D3>0)
	int zlevel = 0;
	if (POLE_1 == 1 && jcurr - N2G < BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (abs(jcurr - N2G) + D2))) / log(2.)), N_LEVELS_1D_INT);
	if (POLE_2 == 1 && jcurr - N2G >= BS_2 / 2) zlevel = MY_MIN((int)(0.001 + log((double)(BS_2 / (BS_2 - MY_MIN(jcurr - N2G, BS_2 - 1)))) / log(2.)), N_LEVELS_1D_INT);
	zsize = (int)(0.001+pow(2.0, (double)zlevel));
	zoffset = (zcurr - N3G) % zsize;
	#endif

	#if(CARTESIAN_GR)
	//if (k == 1 && pflag_cart[global_id] == 1) k = 0;
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
				get_state_res(pf, &geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
				primtoflux_res(pf, &q, 0, &geom, U);
			#else
				get_state(pf, &geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
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
			#if(NEUTRINOS_M1)
            struct of_state_nu q_nu[NU_SPECIES];
            for (sp = 0; sp < NU_SPECIES; sp++) get_state_nu(pf, &geom, &q_nu[sp], sp);
            primtoflux_nu(pf, q_nu, 0, &geom, U);
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
				get_state_res(pf, &geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
			#else
				get_state(pf, &geom, &q
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				);
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
		source_res(pf, &geom, icurr, jcurr, zcurr, dU, &q_charge, Dt, conn, &q, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]);
		#else
		source(pf, &geom, icurr, jcurr, zcurr, dU, Dt, conn, &q, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(TWO_T)
			, gamma_g
			#endif
		);
		#endif

		#pragma unroll 12
		for (k = 0; k < NPR; k++) {
			U[k] += Dt * (dU[k]);
		}

		#if(NSY)
		U[B1] = 0.0;
		U[B2] = 0.0;
		#if(STAGGERED)
		for (u = 0; u < zsize; u++) {
			U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
			U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + jcurr * (BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2 * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem2) + icurr * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (jcurr + D2) * (BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
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

		#if(NEUTRON_STAR && USE_PS1START)
		U[PS1START] = 0.0;
		dU[PS1START] = 0.0;
		#endif
	
		#if(CALC_MDOT)
		U[B1] *= magnetic_density_scale;
		U[B2] *= magnetic_density_scale;
		U[B3] *= magnetic_density_scale;
		#endif

		#if(TWO_T)
		fel = calc_delta(pf, dot(q.bcon, q.bcov));
		#endif

		#if(RAD_M1)
		double U_0[NPR];
		PLOOP dU[k] = 0.;



		//Perform implicit solve
		double cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
		implicit_rad_solve(pf, U, U, U_0, &pflag[global_id], &pflag_rad[global_id], &geom, dU, Dt, cell_size, y_max
			#if(TWO_T)
			, fel
			#endif
			#if(COOL_STOP)
			, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)]
			#endif
			#if(CALC_MDOT)
			,  mass_density_scale, magnetic_density_scale
			#endif
		);
		#elif(NEUTRINOS_M1)
        double U_0[NPR];
        PLOOP dU[k] = 0.;

        //Perform implicit solve
        double cell_size = MY_MAX(MY_MAX(dx_1 * sqrt(geom.gcov[4]), dx_2 * sqrt(geom.gcov[7])), dx_3 * sqrt(geom.gcov[9]));
        semiimplicit_solve_nu(pf, U, U, U_0, &pflag[global_id], &pflag_nu[global_id], &geom, dU, Dt, cell_size, y_max, gpu_eos_table, gpu_nulib_table
            #if(NU_INNER_STOP)
            , radius[icurr]
            #endif
            #if(NEUTRINOS_DEBUG)
            , &allflags_nu[0 * ksize + global_id], &allflags_nu[1 * ksize + global_id], &allflags_nu[2 * ksize + global_id]
            #endif
        ); // DIMARK: nusolve
        if (pflag[global_id]) failimage[0 * (ksize)+global_id]++;
        if (pflag_nu[global_id]) failimage[1 * (ksize)+global_id]++;

        /*if (icurr==11 && jcurr==65 && pf[RHO]>1.0) {
            printf("\n\t\t ############################################");
            PLOOP{
                printf("\n\t\t pf[%d] = %e, U[%d] = %e, dU[%d] = %e", k, pf[k], k, U[k], k, dU[k]);
            }
        }*/
        #else
			#if(RESISTIVE)
			pflag[global_id] = Utoprim_3d_res(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, Dt);
			#else
				#if(JET_ENTROPY)
				double bsq = q.bcon[0] * q.bcov[0] + q.bcon[1] * q.bcov[1] + q.bcon[2] * q.bcov[2] + q.bcon[3] * q.bcov[3];
				if(bsq/pf[RHO]<1.){ //Criterion for entropy evolution
					#if(NEWMAN)
						pflag[global_id] = Utoprim_NM(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
						#if (DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, fel
						#endif
					);
					#elif (USE_3D_INV)
					pflag[global_id] = Utoprim_3D_T(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
						#if (DOHELM)
						, gpu_eos_table
						#endif
					);
					if (pflag[global_id]) {
						pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
							#if (DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, fel
							#endif
						);
					}
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
				}
				else pflag[global_id]=1;
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
					#elif (USE_3D_INV)
					pflag[global_id] = Utoprim_3D_T(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
						#if (DOHELM)
						, gpu_eos_table
						#endif
					);
					if (pflag[global_id]) {
						pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
							#if (DOHELM)
							, gpu_eos_table
							#endif
							#if(TWO_T)
							, fel
							#endif
						);
					}
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
				#endif

				if (pflag[global_id]){
					failimage[global_id]++;
					pflag[global_id] = Utoprim_1dvsq2fix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
						#if(DOHELM)
						, gpu_eos_table
						#endif
						#if(TWO_T)
						, fel
						#endif
					);
					if (pflag[global_id] && !TWO_T && !DOHELM) {
						failimage[1 * (ksize)+global_id]++;
						pflag[global_id] = Utoprim_1dfix1(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC, FULL_ENTROPY
						#if(TWO_T)
						, fel
						#endif
						);
						if (pflag[global_id]){
							pflag[0] = global_id;
							failimage[2 * (ksize)+global_id]++;
						}
					}
				}
			#endif
		#endif

		//#if (DONUCLEAR && DOHELM && DOHELM_TEMPERATURE)
		#if (0)
        // Compute the effect of the alpha particle recombination on the gas temperature, Xalpha and Xatm (keeping rho and ye fixed)
        nuc_evol(gpu_eos_table, pf);
        #endif

		#if(CALC_MDOT)
		pf[B1] = U[B1] / geom.g / magnetic_density_scale;
		pf[B2] = U[B2] / geom.g / magnetic_density_scale;
		pf[B3] = U[B3] / geom.g / magnetic_density_scale;
		#endif

		//Apply floors in ZAMO frame or drift frame
		if (fixup_cell(pf, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)], &geom
			#if (DOHELM)
			, gpu_eos_table
			#endif
			#if(CALC_MDOT)
			, magnetic_density_scale
			#endif
			#if(NEUTRON_STAR)
			, fixupWeight
			#if(!NS_TAPERED_FLOORS)
			, Dt, theta[jcurr]
			#endif
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
#endif
}


__global__ void fixup_post(double* pi_i, double* pb_i, double* pf_i, const  double* __restrict__  psf,
	const  double* __restrict__ F1, const  double* __restrict__  F2, const  double* __restrict__ F3, const  double* __restrict__ U_i, const  double* __restrict__ radius, int* pflag, int* failimage, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, const  double* __restrict__ conn, double dx_1, double dx_2, double dx_3, double Dt, int full_step, int POLE_1, int POLE_2
	#if (DOHELM)
	, const double* __restrict__ gpu_eos_table
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
	#if(DO_RBOUND || (NEUTRON_STAR && 0))
	, int* pflag_rbound
	#endif
	#if(NEUTRON_STAR)
	, double fixupWeight
	#if(!NS_TAPERED_FLOORS)
	, const  double* __restrict__ theta
	#endif
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

	#if(CARTESIAN_GR)
	if (k == 1 && pflag_cart[global_id] == 1) k = 0;
	#endif

	/* Check if cell is marked for inflow because it is within RBOUND. */
	#if(DO_RBOUND || (NEUTRON_STAR && 0))
    if (k == 1 && pflag_rbound[global_id] == 1) k = 0;
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
			get_state(pf, &geom, &q
			#if(CALC_MDOT)
			, magnetic_density_scale
			#endif
			);
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
				U[B1] += (psf[0 * ksize + global_id - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[0 * ksize + global_id + isize - zoffset + u] * gdet[FACE1*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + (icurr + D1)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
				U[B2] += (psf[1 * ksize + global_id - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + jcurr*(BS_3 + 2 * N3G) + zcurr - zoffset + u] + psf[1 * ksize + global_id + (BS_3 + 2 * N3G) - zoffset + u] * gdet[FACE2*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2) + icurr*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (jcurr + D2)*(BS_3 + 2 * N3G) + zcurr - zoffset + u]) / (2.0 * (double)zsize);
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

			#if(CALC_MDOT)
			U[B1] *= magnetic_density_scale;
			U[B2] *= magnetic_density_scale;
			U[B3] *= magnetic_density_scale;
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
			#elif (USE_3D_INV)
			pflag[global_id] = Utoprim_3D_T(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
			);
			if (pflag[global_id]) {
				pflag[global_id] = Utoprim_2d(U, geom.gcov, geom.gcon, geom.g, pf, NEWT_TOL, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, fel
					#endif
				);
			}
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
			if (fixup_cell(pf, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)], &geom
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				#if(NEUTRON_STAR)
				, fixupWeight
				#if(!NS_TAPERED_FLOORS)
				, Dt, theta[jcurr]
				#endif
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

__global__ void fixuputoprim(double *  pv, const  double* __restrict__ radius, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int *  pflag, int *  failimage
	#if (DOHELM)
	, const double* __restrict__ gpu_eos_table
	#endif
	#if(CALC_MDOT)
	, double mass_density_scale
	, double magnetic_density_scale
	#endif
#if(NEUTRON_STAR)
	, double fixupWeight
#if(!NS_TAPERED_FLOORS)
	, double Dt, const  double* __restrict__ theta
#endif
#endif
)
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
	//double avg[NPR];
	//int counter = 0;
	double pf[NPR];
	struct of_geom geom;

	/* Fix the interior points first */
	if (k==1) {
		if (pflag[global_id] != 0) {
			get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);

			for (k = 0; k < NPR; k++) {
				pf[k] = pv[k * (ksize)+global_id];
			}

			pf[RHO] = RHOMINLIMIT;
			pf[UU] = UUMINLIMIT;
			pf[U1] = 0.0;
			pf[U2] = 0.0;
			pf[U3] = 0.0;

			fixup_cell(pf, radius[icurr * (SPHERICAL || SPHERICAL_GR) + global_id * (CARTESIAN || CARTESIAN_GR)], &geom
				#if (DOHELM)
				, gpu_eos_table
				#endif
				#if(CALC_MDOT)
				, magnetic_density_scale
				#endif
				#if(NEUTRON_STAR)
				, fixupWeight
				#if(!NS_TAPERED_FLOORS)
				, Dt, theta[jcurr]
				#endif
				#endif
				);

			for (k = 0; k < NPR; k++) {
				pv[k * (ksize)+global_id] = pf[k];
			}

			/*for (k = 0; k < NPR; k++) avg[k] = 0.;
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
			*/
		}
	}
}

__global__ void fixuputoprim_rad(double* pv, int* pflag_rad, int* failimage)
{
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
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR];
	int counter = 0;

	/* Fix the interior points first */
	if (k == 1) {
		if (pflag_rad[global_id] != 0) {
			for (k = 0; k < NPR; k++) avg[k] = 0.;
			if (icurr - 1 >= N1G) {
				if (pflag_rad[global_id - isize] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++)  avg[k] += pv[k * (ksize)+global_id - isize];
					counter++;
				}
			}
			if (icurr + 1 < BS_1 + N1G) {
				if (pflag_rad[global_id + isize] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++) avg[k] += pv[k * (ksize)+global_id + isize];
					counter++;
				}
			}
			if (jcurr - 1 >= N2G) {
				if (pflag_rad[global_id - (BS_3 + 2 * N3G)] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++) avg[k] += pv[k * (ksize)+global_id - (BS_3 + 2 * N3G)];
					counter++;
				}
			}
			if (jcurr + 1 < BS_2 + N2G) {
				if (pflag_rad[global_id + (BS_3 + 2 * N3G)] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++) avg[k] += pv[k * (ksize)+global_id + (BS_3 + 2 * N3G)];
					counter++;
				}
			}
			if (zcurr - 1 >= N3G) {
				if (pflag_rad[global_id - D3] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++) avg[k] += pv[k * (ksize)+global_id - D3];
					counter++;
				}
			}
			if (zcurr + 1 < BS_3 + N3G) {
				if (pflag_rad[global_id + D3] == 0) {
					for (k = UU_RAD; k <= U3_RAD; k++) avg[k] += pv[k * (ksize)+global_id + D3];
					counter++;
				}
			}
			if (counter > 0) {
				//for (k = UU_RAD; k <= U3_RAD; k++) pv[k * (ksize)+global_id] = 1. / ((double)counter) * avg[k];
			}
		}
	}
}

__global__ void fixuputoprim_nu(double* pv, int* pflag_nu, int* failimage)
{
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
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	double avg[NPR];
	int counter = 0;
	int sp;
	/* Fix the interior points first */
	if (k == 1) {
		if (pflag_nu[global_id] != 0) {
			for (k = 0; k < NPR; k++) avg[k] = 0.;
			for (sp = 0; sp < NU_SPECIES; sp++) {
				if (icurr - 1 >= N1G) {
					if (pflag_nu[global_id - isize] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id - isize];
						counter++;
					}
				}
				if (icurr + 1 < BS_1 + N1G) {
					if (pflag_nu[global_id + isize] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id + isize];
						counter++;
					}
				}
				if (jcurr - 1 >= N2G) {
					if (pflag_nu[global_id - (BS_3 + 2 * N3G)] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id - (BS_3 + 2 * N3G)];
						counter++;
					}
				}
				if (jcurr + 1 < BS_2 + N2G) {
					if (pflag_nu[global_id + (BS_3 + 2 * N3G)] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id + (BS_3 + 2 * N3G)];
						counter++;
					}
				}
				if (zcurr - 1 >= N3G) {
					if (pflag_nu[global_id - D3] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id - D3];
						counter++;
					}
				}
				if (zcurr + 1 < BS_3 + N3G) {
					if (pflag_nu[global_id + D3] == 0) {
						for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) avg[k] += pv[k * (ksize)+global_id + D3];
						counter++;
					}
				}
				if (counter > 0) {
					//for (k = index_nu(UU_NU, sp); k <= index_nu(U3_NU, sp); k++) pv[k * (ksize)+global_id] = 1. / ((double)counter) * avg[k];
				}
			}
		}
	}
}

//Apply floors to a cell
__device__ int fixup_cell(double* pf, double r, struct of_geom* geom
	#if (DOHELM)
	, const  double* __restrict__ gpu_eos_table
	#endif
	#if(CALC_MDOT)
	, double magnetic_density_scale
	#endif
	#if(NEUTRON_STAR)
	, double fixupWeight
	#if(!NS_TAPERED_FLOORS)
	, double dt, double th
	#endif
	#endif
) {
	#if(!CARTESIAN)
	double rhoscal, uuscal, rhoflr, uuflr, bsq, wold, wnew, QdotB, trans, vpar, one_over_ucondr_t, x, f;
	double pf_prefloor[NPR], betapar, betasq, betasqmax, gamma, ucondr[NDIM], Bcon[NDIM], Bcov[NDIM], vcon[NDIM], ucon[NDIM], utcon[NDIM], B, Bsq, udotB, ut ,u;
	#if(RESISTIVE)
	struct of_state_res q;
	#else
	struct of_state q;
	#endif
	#if(TWO_T)
	double ue, ui, dis;
		#if(VARGAMMA)
		double Theta, gam, C;
		#endif
	#endif
#if(NEUTRON_STAR)
		double Rlc, rho_b, rho_g;
#endif
	int dofloor=0, flag = 0, m, k, j;

	rhoscal = pow(MY_MAX(r, 1.0), -POWRHO);
	uuscal = pow(rhoscal, GAMMA);

	rhoflr = 1.e-5 * RHOMIN * rhoscal;
	uuflr = 1.e-5 * UUMIN * uuscal;

	#if(RESISTIVE)
	get_state_res(pf, geom, &q
	#if(CALC_MDOT)
	, magnetic_density_scale
	#endif
	);
	#else
	get_state(pf, geom, &q
	#if(CALC_MDOT)
	, magnetic_density_scale
	#endif
	);
	#endif
	bsq = dot(q.bcon, q.bcov);

	
    //DI: floor on electron fraction
    #if (DO_YE)
    validate_ye(&pf[YE]);
    #endif

    #if (DOHELM)
    double xP;
		#if (DOHELM_TEMPERATURE)
		// Floor on temperature
		if (pf[UU] < eos_temp_low) pf[UU] = eos_temp_low;
		// Get the value of internal energy for other floors
		eos_mode_rhotemp_pres_u(gpu_eos_table, pf[RHO], pf[UU],
			#if (DO_YE)
			pf[YE],
			#else 
			1.0,
			#endif
			&xP, &u
			#if (DONUCLEAR)
			, &pf[XALPHA], &pf[XATM]
			#endif
		);
    
		// Pre-floor value of internal energy
		double prefloor_u = u;
		#else
		u = pf[UU];
		#endif
    #else
    u = pf[UU];
    #endif

	//Store old values
#pragma unroll 12
	for (k = 0; k < (NPR_U + NEUTRON_STAR * (1 + DOFLR + USE_PS1START)); k++) pf_prefloor[k] = pf[k];

#if (NEUTRON_STAR)
#if(NS_TAPERED_FLOORS)

	double smooth, smooth_geom;
	double etacon[NDIM], bhatcon[NDIM], bhatcov[NDIM];
	double uprllcon[NDIM], uperpcon[NDIM], uprllcov[NDIM], uperpcov[NDIM];
	double uDotEta, bDotEta, bhatsq, uDotBhat, uprllsq, uperpsq;
	double rhopre, uintpre, rhopost, uintpost;
	double Kprll, Kcal, K2, uprllDotBhat;
	double XNS, YNS, ZNS, aminusNS, aplusNS, aNS, normalizerNS;
	double u1con[NDIM], beta[NDIM];

	if (OMEGA_NS > 0.0)
		Rlc = 1.0 / OMEGA_NS;
	else
		Rlc = 1.0e6;


	double rho0 = RHO0_HYDROSTAT_ATM_NS * pow(MU_NS / 10.0, 2.0);
	double alpha1_NS, alpha2_NS, n_NS, rb_NS, constant_NS;
	rb_NS = 1.0 * Rlc; // break radius for smooth broken power law
	n_NS = 2.0;       // smaller n = smoother break
	alpha2_NS = 6.0;    // At large r, want rho, u to be propto r^(-alpha2)
					  // For hydrostatic atmosphere to large radii, set alpha2 = alpha1
					  // in each case below.

	/* Density */
	alpha1_NS = 1.0 / (GAMMA - 1.0);
	constant_NS = rho0 * pow(R_NS, alpha1_NS);
	/* Function for smooth broken power law */
	rhoflr = constant_NS * pow(rb_NS, -alpha1_NS) * pow(pow(r / rb_NS, alpha1_NS * n_NS) + pow(r / rb_NS, alpha2_NS * n_NS), -1.0 / n_NS);

	/* Internal energy */
	alpha1_NS = GAMMA / (GAMMA - 1.0);
	constant_NS = (rho0 / GAMMA) * (1.0 / R_NS) * pow(R_NS, alpha1_NS);
	uuflr = constant_NS * pow(rb_NS, -alpha1_NS) * pow(pow(r / rb_NS, alpha1_NS * n_NS) + pow(r / rb_NS, alpha2_NS * n_NS), -1.0 / n_NS);
	double mod_bsq_over_rho_max, mod_bsq_over_uu_max;
	double log_mod_bsq_rho, log_mod_bsq_uu, log_bsq_rho, log_bsq_uu, log_profile;
	/*** Dynamic floors ***/
	/* Use dynamis floor everywhere, but increase acceptable bsq/rho and bsq/u inside the light-sphere,
	 * so that the hydrostatic region doesn't trigger this unless bsq increases enormously              */
	 //double bsq_over_rho_surf = 1e6;
	 //double bsq_over_uu_surf  = 1e7;
	if (r < Rlc)
	{
		log_bsq_rho = log10(MAX_BSQ_OVER_RHO);
		log_bsq_uu = log10(MAX_BSQ_OVER_UINT);
		log_profile = pow((Rlc - r) / (Rlc - R_NS), 2.0);
		log_mod_bsq_rho = log_bsq_rho + (SURF_MAX_BSQ_RHO_LOG - log_bsq_rho) * log_profile;
		log_mod_bsq_uu = log_bsq_uu + (SURF_MAX_BSQ_UINT_LOG - log_bsq_uu) * log_profile;
		mod_bsq_over_rho_max = pow(10.0, log_mod_bsq_rho);
		mod_bsq_over_uu_max = pow(10.0, log_mod_bsq_uu);
	}
	else
	{
		mod_bsq_over_rho_max = MAX_BSQ_OVER_RHO;
		mod_bsq_over_uu_max = MAX_BSQ_OVER_UINT;
	}
	/* tie floors to the local values of magnetic field and internal energy density */
	if (rhoflr < bsq / mod_bsq_over_rho_max)
		rhoflr = bsq / mod_bsq_over_rho_max;

	if (uuflr < bsq / mod_bsq_over_uu_max)
		uuflr = bsq / mod_bsq_over_uu_max;
#else
	if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
	if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;
#endif
#else
    //tie floors to the local values of magnetic field and internal energy density
    if (rhoflr < bsq / BSQORHOMAX) rhoflr = bsq / (BSQORHOMAX);	
    if (uuflr < bsq / BSQOUMAX) uuflr = bsq / (BSQOUMAX);
    #if(RAD_M1)
    if (rhoflr < (u+pf[UU_RAD]) / UORHOMAX) rhoflr = (u + pf[UU_RAD]) / (UORHOMAX);
    #elif (NEUTRINOS_M1)
		#if (NU_SPECIES > 1)
		if (rhoflr < (u+pf[UU_NU]+pf[index_nu(UU_NU, 1)]+pf[index_nu(UU_NU, 2)]) / UORHOMAX)  rhoflr = (u + pf[UU_NU] + pf[index_nu(UU_NU, 1)] + pf[index_nu(UU_NU, 2)]) / (UORHOMAX);
		#else
		if (rhoflr < (u+pf[UU_NU]) / UORHOMAX)  rhoflr = (u + pf[UU_NU]) / (UORHOMAX);
		#endif
    #else
    if (rhoflr < u / UORHOMAX) rhoflr = u / (UORHOMAX);
    #endif
    if (rhoflr < RHOMINLIMIT) rhoflr = RHOMINLIMIT;
    if (uuflr < UUMINLIMIT) uuflr = UUMINLIMIT;
#endif


#if(NEUTRON_STAR)

	if (OMEGA_NS > 0.0)
		Rlc = 1.0 / OMEGA_NS;
	else
		Rlc = 1.e6; //10.0; //maybe move to a large value for omega=0

	if (pf[FLRFRAC] < 0.0)
		pf[FLRFRAC] = 0.0;
	if (pf[FLRFRAC] > 1.0)
		pf[FLRFRAC] = 1.0;

	rho_b = pf[RHO] * pf[FLRFRAC];
	rho_g = pf[RHO] - rho_b;

#if(!NS_TAPERED_FLOORS)

	double r1 = 0.5 * Rlc;
	double r2 = 1.0 * Rlc;
	double Period = 2. * M_PI * Rlc;
	double tau0 = 0.005 * Period;
	double tau = tau0 * pow((r - r1) * (r2 - R_NS) / (((r2 - r1) * (r2 - r))), 1.0);
	double rho_t = bsq / FREEZE_BSQORHO;
	double uu_t = 0.2 * rho_t;
	double b2 = fabs(cos(th)) / (tau + SMALL);
	if (r <= 0.5 * Rlc) {
		pf[RHO] = rho_t;
		pf[UU] = uu_t;
	}
	if (r > 0.5 * Rlc && r < Rlc) {
		pf[RHO] = rho_t + (pf[RHO] - rho_t) * exp(-dt * fixupWeight * b2);
		pf[UU] = uu_t + (pf[UU] - uu_t) * exp(-dt * fixupWeight * b2);
	}

#else

	//if (pf[RHO] < 1.0001 * rhoflr)
	//	pf[FLRFRAC] = 1.0;
	//else
	//	pf[FLRFRAC] = 0.0;

	if (r > Rlc)
		smooth_geom = 1.0;
	else if (r < R_NS)
		smooth_geom = 0.0;
	else
		smooth_geom = pow((1.0) * 0.5 * (1.0 - cos(M_PI * (r - R_NS) / (Rlc - R_NS))), 2.0);
	smooth_geom = pow(smooth_geom, fixupWeight); //fixupWeight=0.5 or 1
	/* This variable is 1 beyond Rlc, or if FLRFRAC = 0 */
	smooth = 1.0 - pf[FLRFRAC] * (1.0 - smooth_geom);  // Don't want to do anything to real gas
#endif
#endif

	#if(TWO_T)
	pf_prefloor[ENTRE] = pf[ENTRE];
	pf_prefloor[ENTRI] = pf[ENTRI];
	#endif

	//floor on density 
	if (pf[RHO] < rhoflr) {
#if(NEUTRON_STAR && NS_TAPERED_FLOORS)
		rho_b = rhoflr - rho_g;
#endif
		pf[RHO] = rhoflr;
		dofloor = 1;
	}
#if(NEUTRON_STAR && NS_TAPERED_FLOORS)
	else if (rho_b > rhoflr && r < Rlc)
	{
		pf[RHO] = rhoflr + smooth_geom * (rho_b - rhoflr) + rho_g;
	}
#endif

	#if (DONUCLEAR)
	validate_abund(&pf[XALPHA]);
	validate_abund(&pf[XATM]);
    #endif

	//Internal energy floor

#if(NEUTRON_STAR && NS_TAPERED_FLOORS)
/*** Cool rapidly in high-floor zones inside the LC ***/
	if (pf[UU] > uuflr && r < Rlc)
		pf[UU] = uuflr + smooth * (pf[UU] - uuflr); // Full smoothing fn: don't cool good gas
#endif

	#if(RAD_M1)
	if (u + pf[UU_RAD] < uuflr) {
		pf[UU] = uuflr - pf[UU_RAD];
		dofloor = 1;
	}
	if (u < 0.0001*uuflr) {
		u = 0.0001 * uuflr;
		dofloor = 1;
	}
	#elif(NEUTRINOS_M1)
		#if (NU_SPECIES > 1)
		if (u + pf[UU_NU] + pf[index_nu(UU_NU, 1)] + pf[index_nu(UU_NU, 2)] < uuflr) {
			u = uuflr - pf[UU_NU] + pf[index_nu(UU_NU, 1)] + pf[index_nu(UU_NU, 2)];
			dofloor = 1;
		}
		#else
		if (u + pf[UU_NU] < uuflr) {
			u = uuflr - pf[UU_NU];
			dofloor = 1;
		}
		#endif
	#else
		#if(DOHELM_TEMPERATURE)
		if (u < uuflr) {
			u = uuflr;
			dofloor = 1;
		}
		#else
		if (u < uuflr) {
			pf[UU] = uuflr;
			u = uuflr;
			dofloor = 1;
		}
		#endif
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

	// Floor on neutrino energy density
    #if(NEUTRINOS_M1)
    double Tnu;
    for (int sp = 0; sp < NU_SPECIES; sp++) {
        if (pf[index_nu(UU_NU, sp)] < pow(10., -30.)) {
            pf[index_nu(UU_NU, sp)] = pow(10., -30.);

            //Floor on neutrino number
            //pf[NUMBER_NU] = 1e-30;
            Tnu = pow(pf[index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / ARAD, 0.25);
            pf[index_nu(NUMBER_NU, sp)] = pf[index_nu(UU_NU, sp)] * ENERGY_DENSITY_SCALE / (2.701178 * MASS_DENSITY_SCALE * BOLTZ_CGS * Tnu);
        }
    }
    #endif

	//Divide internal energy inject between electrons and ions 1:1
	#if(TWO_T)
	if(dofloor) {	
		#if(CONSTANTGAMMA || FIXEDGAMMA)
		//Calculate electron entropy
		ue = pf[ENTRE] * pow(pf[RHO], GAMMAE) / (GAMMAE - 1.0);

		//Calculate ion entropy
		ui = pf[ENTRI] * pow(pf[RHO], GAMMA) / (GAMMA - 1.0);
		
		//Calculate total dissipation
		dis = pf[UU] - (ue + ui);
		ue += 0.5 * dis;

		//Check limits
		if (ue > (1.0 - FLOOR_ENTROPY) * pf[UU]) {
			ue = (1.0 - FLOOR_ENTROPY) * pf[UU];
		}
		if (ue < FLOOR_ENTROPY * pf[UU]) {
			ue = FLOOR_ENTROPY * pf[UU];
		}
		ui = pf[UU] - ue;

			//Calculate electron and ion entropies
			#if(FULL_ENTROPY)
			pf[ENTRE] = 1. / (GAMMAE - 1.) * log(0.5 * (GAMMAE - 1.0) * pf[UU] * pow(pf[RHO], -GAMMAE));
			pf[ENTRI] = 1. / (GAMMA - 1.) * log(0.5 * (GAMMA - 1.0) * pf[UU] * pow(pf[RHO], -GAMMA));
			#else
			pf[ENTRE] = 0.5 * (GAMMAE - 1.0) * pf[UU] * pow(pf[RHO], -GAMMAE);
			pf[ENTRI] = 0.5 * (GAMMA - 1.0) * pf[UU] * pow(pf[RHO], -GAMMA);
			#endif
		#elif(VARGAMMA)
		
		//Calculate ue
		#if(FULL_ENTROPY_VARGAMMA)
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pf[RHO] * exp(pf[ENTRE])), 2. / 3.)) - 1.0));
		#else
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pf[RHO], 2. / 3.) * fabs(pf[ENTRE])) - 1.0));
		#endif
		gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		ue = Theta / (MU_E * MASS_RATIO) * pf[RHO] / (gam - 1.0);

		//Calculate ui
		#if(FULL_ENTROPY_VARGAMMA)
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(fabs(pf[RHO] * exp(pf[ENTRI])), 2. / 3.)) - 1.0));
		#else
		Theta = fabs(0.2 * (sqrt(1.0 + 25.0 * pow(pf[RHO], 2. / 3.) * fabs(pf[ENTRI])) - 1.0));
		#endif
		gam = (10.0 + 20.0 * Theta) / (6.0 + 15.0 * Theta);
		ui = Theta / (MU_I) * pf[RHO] / (gam - 1.0);

		//Calculate total dissipation
		dis = pf[UU] - (ue + ui);
		ue += 0.5 * dis;
		ue = 0.5 * pf[UU];

		//Check limits
		if (ue > (1.0 - FLOOR_ENTROPY) * pf[UU]) {
			ue = (1.0 - FLOOR_ENTROPY) * pf[UU];
		}
		if (ue < FLOOR_ENTROPY * pf[UU]) {
			ue = FLOOR_ENTROPY * pf[UU];
		}
		ui = pf[UU] - ue;

		//Calculate electron entropy
		C = ue / pf[RHO] * MU_E * MASS_RATIO;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pf[ENTRE] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO]);
			#else
			pf[ENTRE] = Theta * (Theta + 0.4) / pow(pf[RHO], 2. / 3.);
			#endif

		//Calculate ion entropy
		C = ui / pf[RHO] * MU_I;
		Theta = (1.0 / 30.0) * (sqrt(25.0 * C * C + 180.0 * C + 36.0) + 5.0 * C - 6.0);
			#if(FULL_ENTROPY_VARGAMMA)
			pf[ENTRI] = log(pow(Theta, 1.5) * pow(Theta + 0.4, 1.5) / pf[RHO]);
			#else
			pf[ENTRI] = Theta * (Theta + 0.4) / pow(pf[RHO], 2. / 3.);
			#endif
		#endif
	}
	#endif
#if(NEUTRON_STAR)
#if(!NS_TAPERED_FLOORS)
	get_state(pf, geom, &q
#if(CALC_MDOT)
		, magnetic_density_scale
#endif
	);
	Bcon[0] = 0.;
	for (k = 1; k < NDIM; k++) {
		Bcon[k] = pf[B1 - 1 + k];
	}

	Bcov[1] = geom->gcov[1] * Bcon[0] + geom->gcov[4] * Bcon[1] + geom->gcov[5] * Bcon[2] + geom->gcov[6] * Bcon[3];
	Bcov[2] = geom->gcov[2] * Bcon[0] + geom->gcov[5] * Bcon[1] + geom->gcov[7] * Bcon[2] + geom->gcov[8] * Bcon[3];
	Bcov[3] = geom->gcov[3] * Bcon[0] + geom->gcov[6] * Bcon[1] + geom->gcov[8] * Bcon[2] + geom->gcov[9] * Bcon[3];

	udotB = dot(q.ucon, Bcov);
	Bsq = dot(Bcon, Bcov);

	double ucon_old[NDIM], ucon_par_old[NDIM];

	for (k = 0; k < NDIM; k++) {
		ucon_old[k] = q.ucon[k];
	}
	for (k = 0; k < NDIM; k++) {
		ucon_par_old[k] = udotB * Bcon[k] / (Bsq + SMALL);
	}

	//new parallel velocity
	if (r <= 0.5 * Rlc) {
		for (k = 0; k < NDIM; k++) {
			ucon[k] = ucon_old[k] - ucon_par_old[k];
		}
	}
	else if (r > 0.5 * Rlc && r < Rlc) {
		for (k = 0; k < NDIM; k++) {
			ucon[k] = ucon_old[k] - ucon_par_old[k] + ucon_par_old[k] * exp(-fixupWeight * dt / (tau + SMALL));
		}
	}
	else {
		for (k = 0; k < NDIM; k++) {
			ucon[k] = ucon_old[k];
		}
	}

	#pragma unroll 3
	SLOOPA	pf[j + U1 - 1] = ucon[j] - ucon[0] * geom->gcon[j] / geom->gcon[0];

#else


	//kyles_unified_4D_velocityAdjust(pv_prefloor,  pv, &geom, smooth)

	get_state(pf_prefloor, geom, &q
#if(CALC_MDOT)
		, magnetic_density_scale
#endif
	);
	if (pf_prefloor[RHO] < pf[RHO] || pf_prefloor[UU] < pf[UU] || smooth < 1.0) {
		/* 4-velocity of observer which is static wrt the coordinates */
		etacon[0] = sqrt(-1.0 / geom->gcov[0]);
		#pragma unroll 3
		SLOOPA etacon[j] = 0.0;

		uDotEta = dot(q.ucov, etacon);
		bDotEta = dot(q.bcov, etacon);

		/* Construct B-field 4-vector in coordinate-static frame */
		#pragma unroll 4
		DLOOPA bhatcon[j] = bDotEta * q.ucon[j] - uDotEta * q.bcon[j];
		lower(bhatcon, geom->gcov, bhatcov);
		bhatsq = dot(bhatcov, bhatcon);

		/* Construct parallel and perpendicular (unnormalized) 4-velocities by projection */
		uDotBhat = dot(q.ucov, bhatcon);
		#pragma unroll 4
		DLOOPA
		{
			uprllcon[j] = (uDotBhat / bhatsq) * bhatcon[j];
			uperpcon[j] = q.ucon[j] - uprllcon[j]; // identical to using projection tensor
		}
		lower(uprllcon, geom->gcov, uprllcov);
		lower(uperpcon, geom->gcov, uperpcov);
		uprllsq = dot(uprllcov, uprllcon);
		uperpsq = dot(uperpcov, uperpcon);

		rhopre = pf_prefloor[RHO];
		uintpre = pf_prefloor[UU];

		/***** Modification due to flooring ******/
		if (pf[RHO] >= rhopre || pf[UU] >= uintpre)
		{
			/* Find final enthalpy */
			if (pf[RHO] > rhopre)
				rhopost = pf[RHO];
			else
				rhopost = rhopre;

			if (pf[UU] > uintpre)
				uintpost = pf[UU];
			else
				uintpost = uintpre;

			// Conserved mom'm along coord-static frame magnetic field
			Kprll = (rhopre + uintpre + (GAMMA - 1.0) * uintpre) * q.ucon[0] * dot(bhatcon, q.ucov) + (GAMMA - 1.0) * uintpre * bhatcon[0];

			// Calligraphic K from notes: subtract off pressure term with *post-floor* value
			Kcal = Kprll - (GAMMA - 1.0) * uintpost * bhatcon[0];

			uprllDotBhat = dot(uprllcov, bhatcon);

			// Final form: K_2
			K2 = Kcal / ((rhopost + uintpost + (GAMMA - 1.0) * uintpost) * uprllDotBhat);

			/* X a^2 + Y a + Z = 0 */
			XNS = uprllcon[0] + K2 * uprllsq;
			YNS = uperpcon[0];
			ZNS = K2 * uperpsq;

			aminusNS = (-YNS - sqrt(YNS * YNS - 4 * XNS * ZNS)) / (2 * XNS);
			aplusNS = (-YNS + sqrt(YNS * YNS - 4 * XNS * ZNS)) / (2 * XNS);

			// Want a to be positive
			if ((aminusNS >= 0.0 && aminusNS <= 1.0) && (aplusNS < 0 || aplusNS > 1))
				aNS = aminusNS;
			else if ((aplusNS >= 0.0 && aplusNS <= 1.0) && (aminusNS < 0 || aminusNS > 1))
				aNS = aplusNS;
			else            // Both or neither values in range --- always seems to be neither
				aNS = 1.0;   // Seems to happen when the "true" solution is just over 1 due to numerical errors
				//fprintf(stderr, "BOTH or NEITHER A VALUES IN RANGE: %e   %e\n", aminus, aplus);
		}
		else
			aNS = 1.0;

		/***** Modification due to reduce parallel velocity inside the light-sphere *****/
		if (smooth < 1.0)
			aNS *= smooth;

		/***** Construct final 4-velocity *****/
		normalizerNS = sqrt(-(aNS * aNS * uprllsq + uperpsq));
		#pragma unroll 4
		DLOOPA ucon[j] = (aNS * uprllcon[j] + uperpcon[j]) / normalizerNS;

		#pragma unroll 3
		SLOOPA	pf[j + U1 - 1] = ucon[j] - ucon[0] * geom->gcon[j] / geom->gcon[0];


	}
#endif
	#elif(DRIFT_FLOOR)
	trans = 10. * bsq / MY_MIN(pf[RHO], u) - 1.;
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
		#if(CALC_MDOT)
		#pragma unroll 3
		for (m = 1; m < NDIM; m++) Bcon[m] = magnetic_density_scale*pf[B1 - 1 + m];
		#else
		#pragma unroll 3
		for (m = 1; m < NDIM; m++) Bcon[m] = pf[B1 - 1 + m];
		#endif
		lower(Bcon, geom->gcov, Bcov);
		udotB = dot(q.ucon, Bcov);
		Bsq = dot(Bcon, Bcov);
		B = sqrt(Bsq);

		//enthalpy before the floors
        #if (DOHELM)
			#if (DOHELM_TEMPERATURE)
			wold = pf_prefloor[RHO] + prefloor_u + xP;
			#else
			eos_mode_rhou_pres(gpu_eos_table, pf_prefloor[RHO], pf_prefloor[UU]
				#if (DO_YE)
				, pf[YE]
				#else
				, 1.0
				#endif
				, &xP);
			wold = pf_prefloor[RHO] + pf_prefloor[UU] + xP;
			#endif
		#elif(TWO_T)
		double gamma_g;
		gamma_g = calc_gamma_gas_prim(pf_prefloor);
		wold = pf_prefloor[RHO] + pf_prefloor[UU] * gamma_g;
		#else
		wold = pf_prefloor[RHO] + pf_prefloor[UU] * GAMMA;
		#endif

		//B^\mu Q_\mu = (B^\mu u_\mu) (\rho+u+p) u^t (eq. (26) divided by alpha; Noble et al. 2006)
		QdotB = udotB * wold * q.ucon[0];

		//enthalpy after the floors
	    #if (DOHELM)
			#if (DOHELM_TEMPERATURE)
			// Find the temperature and pressure for new ugas:
			eos_mode_rhotemp_u_pres_floor(gpu_eos_table, pf[RHO], &pf[UU],
				#if (DO_YE)
				pf[YE],
				#else 
				1.0,
				#endif
				u, &xP
				#if (DONUCLEAR)
				, &pf[XALPHA], &pf[XATM]
				#endif
			);
			wnew = pf[RHO] + u + xP;
			#else
			eos_mode_rhou_pres(gpu_eos_table, pf[RHO], pf[UU]
				#if (DO_YE)
				, pf[YE]
				#else
				, 1.0
				#endif
				, &xP);
			wnew = pf[RHO] + pf[UU] + xP;
			#endif
		#elif(TWO_T)
		gamma_g = calc_gamma_gas_prim(pf);
		wnew = pf[RHO] + pf[UU] * gamma_g;
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
		get_state(dpf, geom, q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
		primtoflux(dpf, q, 0, geom, dU, NULL, NULL
			#if (DOHELM)
			, gpu_eos_table
			#endif
		);

		//compute the prefloor conserved quantity
		get_state(pf_prefloor, geom, q
		#if(CALC_MDOT)
		, magnetic_density_scale
		#endif
		);
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
			#if (USE_3D_INV)
			flag = Utoprim_3D_T(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC
				#if (DOHELM)
				, gpu_eos_table
				#endif
			);
			if (flag) {
				flag = Utoprim_2d(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC
					#if (DOHELM)
					, gpu_eos_table
					#endif
					#if(TWO_T)
					, 0.0
					#endif
				);
			}
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
        #endif
		if (flag) {
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
				#if (!DOHELM)
				flag = Utoprim_1dfix1(U, geom->gcov, geom->gcon, geom->g, pf, NEWT_TOL, BASIC, 0
					#if(TWO_T)
					, 0.0
					#endif
				);
				#endif
			}
			#endif	
		}
	}
	#endif
	#if(NEUTRON_STAR)
	/*** Reset floor fraction following either flooring or draining ***/
	pf[FLRFRAC] = rho_b / (rho_b + rho_g);

	/*** Enforce some sanity ***/
	if (pf[FLRFRAC] < 0.0)
		pf[FLRFRAC] = 0.0;
	else if (pf[FLRFRAC] > 1.0)
		pf[FLRFRAC] = 1.0;
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
			//pf[FLRFRAC] = 1.0;
		}
	}

	return flag;
	#else 
	return(0);
	#endif
}
