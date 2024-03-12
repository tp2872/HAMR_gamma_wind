

__global__ void boundprim1_outflow(double *   pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double *  ps
    #if(DANAT_GDET_INTERP)	
    , const double* __restrict__ radius
    #endif
)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G);
	int k;
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];
	#if(DANAT_GDET_INTERP)	
	struct of_geom geom1, geom2, geom3, geom5;
	double dr_over_r = (radius[N1G] - radius[N1G - 1]) / radius[N1G];
	#endif

	// inner r boundary condition: u, gdet extrapolation
	if (jcurr >= 0 && jcurr<BS_2 + 2 * N2G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_4 == -1){
		
		#if(DANAT_GDET_INTERP)	
		get_geometry(0, jcurr, zcurr, CENT, &geom1, gcov, gcon, gdet);
		get_geometry(1, jcurr, zcurr, CENT, &geom2, gcov, gcon, gdet);
		#if(N1G==3)
		get_geometry(2, jcurr, zcurr, CENT, &geom3, gcov, gcon, gdet);
		#endif
		get_geometry(N1G, jcurr, zcurr, CENT, &geom5, gcov, gcon, gdet);
		#endif

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
		inflow_check(prim1, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		inflow_check(prim2, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		#if(N1G==3)
		inflow_check(prim3, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		#endif
		inflow_check(prim1, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		inflow_check(prim2, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		#if(N1G==3)
		inflow_check(prim3, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		#endif

		// Extrapolate in the ghost cells as in Gammie et al. (gdet extrapolation)
        #if(DANAT_GDET_INTERP)	
        extrapolate_gdet_innerBC(prim5, prim1, geom5.g, geom1.g, dr_over_r);
        extrapolate_gdet_innerBC(prim5, prim2, geom5.g, geom2.g, dr_over_r);
			#if(N1G==3)
			extrapolate_gdet_innerBC(prim5, prim3, geom5.g, geom3.g, dr_over_r);
			#endif
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
	#if(!CONSTANT_BC)
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

		//Make sure there is no inflow at outer boundary
		inflow_check(prim3, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		inflow_check(prim4, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		#endif
		inflow_check(prim3, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		inflow_check(prim4, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
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
	#endif
}

__global__ void boundprim2_outflow(double * pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G);
	int gridsize= (BS_1 + 2 * N1G) * (BS_3 + 2 * N3G);
	int k;
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];

	// inner r boundary condition: u, gdet extrapolation
	if (icurr >= 0 && icurr < BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_1 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim5[k] = pv[k * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
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
		inflow_check(prim1, icurr, 0, zcurr, 0, gcov, gcon, gdet, 2);
		inflow_check(prim2, icurr, 0, zcurr, 0, gcov, gcon, gdet, 2);
		#if(N2G==3)
		inflow_check(prim3, icurr, 0, zcurr, 0, gcov, gcon, gdet, 2);
		#endif
		inflow_check(prim1, icurr, 1, zcurr, 0, gcov, gcon, gdet, 2);
		inflow_check(prim2, icurr, 1, zcurr, 0, gcov, gcon, gdet, 2);
		#if(N2G==3)
		inflow_check(prim3, icurr, 1, zcurr, 0, gcov, gcon, gdet, 2);
		#endif

		/*Write primitives back to global memory*/
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k * (ksize)+icurr * isize + 0 * (BS_3 + 2 * N3G) + zcurr] = prim2[k];
			pv[k * (ksize)+icurr * isize + 1 * (BS_3 + 2 * N3G) + zcurr] = prim1[k];
			#if(N2G==3)
			pv[k * (ksize)+icurr * isize + 2 * (BS_3 + 2 * N3G) + zcurr] = prim3[k];
			#endif
		}

		#if(STAGGERED)
		ps[0 * (ksize)+icurr * isize + 0 * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+icurr * isize + 1 * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + 0 * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + 1 * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+icurr * isize + 2 * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + 2 * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + N3G * (BS_3 + 2 * N3G) + zcurr];
		#endif
		#endif

		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}

	if (global_id<gridsize){
		global_id = -10;
		icurr = -10;
		zcurr = -10;
	}
	else if (global_id >= gridsize){
		global_id = global_id - gridsize;
		zcurr = global_id % (BS_3 + 2 * N3G);
		icurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	}

	// outer r BC: outflow
	if (icurr >= 0 && icurr < BS_1 + 2 * N1G && zcurr >= 0 && zcurr<BS_3 + 2 * N3G && NBR_3 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim6[k] = pv[k * (ksize) + icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		}

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}

		//Make sure there is no inflow at outer boundary
		inflow_check(prim3, icurr, BS_2 + N2G, zcurr, 1, gcov, gcon, gdet, 2);
		inflow_check(prim4, icurr, BS_2 + N2G, zcurr, 1, gcov, gcon, gdet, 2);
		#if(N2G==3)
		inflow_check(prim5, icurr, BS_2 + N2G, zcurr, 1, gcov, gcon, gdet, 2);
		#endif
		inflow_check(prim3, icurr, BS_2 + N2G + 1, zcurr, 1, gcov, gcon, gdet, 2);
		inflow_check(prim4, icurr, BS_2 + N2G + 1, zcurr, 1, gcov, gcon, gdet, 2);
		#if(N2G==3)
		inflow_check(prim5, icurr, BS_2 + N2G + 1, zcurr, 1, gcov, gcon, gdet, 2);
		#endif

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k * (ksize)+icurr * isize + (BS_2 + N2G) * (BS_3 + 2 * N3G) + zcurr] = prim3[k];
			pv[k * (ksize)+icurr * isize + (BS_2 + N2G + 1) * (BS_3 + 2 * N3G) + zcurr] = prim4[k];
			#if(N2G==3)
			pv[k * (ksize)+icurr * isize + (BS_2 + N2G + 2) * (BS_3 + 2 * N3G) + zcurr] = prim5[k];
			#endif
		}
		#if(STAGGERED)
		ps[0 * (ksize)+icurr * isize + (BS_2 + N2G) * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		ps[0 * (ksize)+icurr * isize + (BS_2 + N2G + 1) * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + (BS_2 + N2G) * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + (BS_2 + N2G + 1) * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		#if(N2G==3)
		ps[0 * (ksize)+icurr * isize + (BS_2 + N2G + 2) * (BS_3 + 2 * N3G) + zcurr] = ps[0 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		ps[2 * (ksize)+icurr * isize + (BS_2 + N2G + 2) * (BS_3 + 2 * N3G) + zcurr] = ps[2 * (ksize)+icurr * isize + (BS_2 + N2G - 1) * (BS_3 + 2 * N3G) + zcurr];
		#endif
		#endif
	}
}

__global__ void boundprim2_reflective(double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
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

			#if(NEUTRINOS_M1)
            for (int sp = 0; sp < NU_SPECIES; sp++) {
                pv[index_nu(U2_NU, sp) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[index_nu(U2_NU, sp) * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
                pv[index_nu(UU_NU, sp) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(UU_NU, sp) * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
                pv[index_nu(U1_NU, sp) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(U1_NU, sp) * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
                pv[index_nu(U3_NU, sp) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(U3_NU, sp) * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
                pv[index_nu(NUMBER_NU, sp) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(NUMBER_NU, sp) * (ksize)+isize * icurr + (jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
            }
            #endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			#if(DOKTOT)
			pv[KTOT*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(DO_YE)
			pv[YE*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[YE*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif
			
			#if(DONUCLEAR)
			pv[XALPHA*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[XALPHA*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[XATM*(ksize)+isize*icurr + (j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[XATM*(ksize)+isize*icurr + (jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
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
			#if(NEUTRINOS_M1)
            pv[U2_NU * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            #if (NU_SPECIES > 1)
            pv[index_nu(U2_NU, 1) * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            pv[index_nu(U2_NU, 2) * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            #endif
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

			#if(NEUTRINOS_M1)
			for (int sp = 0; sp < NU_SPECIES; sp++) {
				pv[index_nu(U2_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = (j + 0.5) / (jref + 0.5) * pv[index_nu(U2_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
				pv[index_nu(UU_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(UU_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
				pv[index_nu(U1_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(U1_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
				pv[index_nu(U3_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(U3_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
				pv[index_nu(NUMBER_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[index_nu(NUMBER_NU, sp) * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			}
			#endif

			//everything else copy (both poles)
			pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[RHO * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[UU * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U1 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[U3 * (ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];

			#if DOKTOT
			pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - j + N2G)*(BS_3 + 2 * N3G) + zcurr] = pv[KTOT*(ksize)+isize*icurr + (BS_2 - 1 - jref + N2G)*(BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(DO_YE)
			pv[YE * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[YE * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			#endif

			#if(DONUCLEAR)
			pv[XALPHA * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[XALPHA * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
			pv[XATM * (ksize)+isize * icurr + (BS_2 - 1 - j + N2G) * (BS_3 + 2 * N3G) + zcurr] = pv[XATM * (ksize)+isize * icurr + (BS_2 - 1 - jref + N2G) * (BS_3 + 2 * N3G) + zcurr];
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
			#if(NEUTRINOS_M1)
            pv[U2_NU * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            #if (NU_SPECIES > 1)
            pv[index_nu(U2_NU, 1) * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            pv[index_nu(U2_NU, 2) * (ksize)+isize * icurr + j * (BS_3 + 2 * N3G) + zcurr] *= -1.;
            #endif
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

__global__ void boundprim2_trans(double *  pv, const  double* __restrict__ gdet, int NBR_1, int NBR_3, double *  ps)
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
			#if(NEUTRINOS_M1)
            pv[U2_NU * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[U3_NU * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            #if (NU_SPECIES > 1)
            pv[index_nu(U2_NU, 1) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U2_NU, 2) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U3_NU, 1) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U3_NU, 2) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            #endif
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
			#if(NEUTRINOS_M1)
            pv[U2_NU * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[U3_NU * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            #if (NU_SPECIES > 1)
            pv[index_nu(U2_NU, 1) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U2_NU, 2) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U3_NU, 1) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            pv[index_nu(U3_NU, 2) * (ksize)+isize * icurr + (j + N2G) * (BS_3 + 2 * N3G) + zcurr] *= -1.0;
            #endif
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

__global__ void boundprim3_outflow(double * pv, const  double* __restrict__ gcov,const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_5, int NBR_6, double *  ps)
{
	int global_id=blockDim.x*blockIdx.x+threadIdx.x;
	int isize = (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G);
	int gridsize= (BS_1 + 2 * N1G) * (BS_2 + 2 * N2G);
	int k;
	int jcurr = global_id % (BS_2 + 2 * N2G);
	int icurr = (global_id - jcurr) / (BS_2 + 2 * N2G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize*(BS_1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];

	// inner r boundary condition: u, gdet extrapolation
	if (icurr >= 0 && icurr < BS_1 + 2 * N1G && jcurr >= 0 && jcurr<BS_2 + 2 * N2G && NBR_6 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim5[k] = pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + N3G];
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
		inflow_check(prim1, icurr, jcurr, 0, 0, gcov, gcon, gdet, 3);
		inflow_check(prim2, icurr, jcurr, 0, 0, gcov, gcon, gdet, 3);
		#if(N3G==3)
		inflow_check(prim3, icurr, jcurr, 0, 0, gcov, gcon, gdet, 3);
		#endif
		inflow_check(prim1, icurr, jcurr, 1, 0, gcov, gcon, gdet, 3);
		inflow_check(prim2, icurr, jcurr, 1, 0, gcov, gcon, gdet, 3);
		#if(N3G==3)
		inflow_check(prim3, icurr, jcurr, 1, 0, gcov, gcon, gdet, 3);
		#endif

		/*Write primitives back to global memory*/
		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G)] = prim2[k];
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + 1] = prim1[k];
			#if(N3G==3)
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + 2] = prim3[k];
			#endif
		}

		#if(STAGGERED)
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +0] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +1] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +0] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +1] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		#if(N3G==3)
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +2] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +2] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +N3G];
		#endif
		#endif

		global_id = -10;
		icurr = -10;
		jcurr = -10;
	}

	if (global_id<gridsize){
		global_id = -10;
		icurr = -10;
		jcurr = -10;
	}
	else if (global_id >= gridsize){
		global_id = global_id - gridsize;
		jcurr = global_id % (BS_2 + 2 * N2G);
		icurr = (global_id - jcurr) / (BS_2 + 2 * N2G);
	}

	// outer r BC: outflow
	if (icurr >= 0 && icurr < BS_1 + 2 * N1G && jcurr >= 0 && jcurr<BS_2 + 2 * N2G && NBR_5 == -1){
		#pragma unroll 9
		for (k = 0; k< NPR; k++){
			prim6[k] = pv[k * (ksize) + icurr * isize + jcurr * (BS_3 + 2 * N3G) + (BS_3 + N3G - 1)];
		}

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}

		//Make sure there is no inflow at outer boundary
		inflow_check(prim3, icurr, jcurr, BS_3 + N3G, 1, gcov, gcon, gdet, 3);
		inflow_check(prim4, icurr, jcurr, BS_3 + N3G, 1, gcov, gcon, gdet, 3);
		#if(N3G==3)
		inflow_check(prim5, icurr, jcurr, BS_3 + N3G, 1, gcov, gcon, gdet, 3);
		#endif
		inflow_check(prim3, icurr, jcurr, BS_3 + N3G + 1, 1, gcov, gcon, gdet, 3);
		inflow_check(prim4, icurr, jcurr, BS_3 + N3G + 1, 1, gcov, gcon, gdet, 3);
		#if(N3G==3)
		inflow_check(prim5, icurr, jcurr, BS_3 + N3G + 1, 1, gcov, gcon, gdet, 3);
		#endif

		#pragma unroll 9
		for (k = 0; k<NPR; k++){
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + (BS_3 + N3G)] = prim3[k];
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + (BS_3 + N3G + 1)] = prim4[k];
			#if(N3G==3)
			pv[k * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) + (BS_3 + N3G + 2)] = prim5[k];
			#endif
		}
		#if(STAGGERED)
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G)] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G - 1)];
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G + 1)] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_1 + N3G - 1)];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G)] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G - 1)];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G + 1)] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G - 1)];
		#if(N3G==3)
		ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G + 2)] = ps[0 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G - 1)];
		ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G + 2)] = ps[1 * (ksize)+icurr * isize + jcurr * (BS_3 + 2 * N3G) +(BS_3 + N3G - 1)];
		#endif
		#endif
	}
}

__global__ void boundprim_rbound(double * pv, double *  ps, int * pflag_rbound, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);  
	zcurr = (global_id % (isize)) % (BS_3+2*N3G);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3+2*N3G);
	icurr = (global_id - (jcurr * (BS_3+2*N3G)+zcurr)) / (isize);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	int k=0, tag=0, i2;
	double p_local[NPR];
	struct of_geom geom;

	if (global_id < (BS_1+2*N1G) * (BS_2+2*N2G) * (BS_3+2*N3G)) k = 1;

	if (k==1 && pflag_rbound[global_id] == 1) {
		#if(RBOUND_INFLOW)
		tag = 0;
		for (i2 = 1; i2 <= N1G; i2++) {
			if ((icurr+i2<BS_1+2*N1G) && (pflag_rbound[global_id+i2*isize] == 0)) {
				PLOOP pv[k * ksize + global_id] = pv[k * ksize + global_id + i2 * isize];
				#if(STAGGERED)
				ps[0 * ksize + global_id] = 0.0;
				ps[1 * ksize + global_id] = ps[1 * ksize + global_id + i2 * isize];
				ps[2 * ksize + global_id] = ps[2 * ksize + global_id + i2 * isize];
				#endif
				tag = 1;
				break;
			}
		}
		#endif		
		if(tag==0){
			//Get metric
			get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);

			//Set density and internal energy
			p_local[RHO] = RHOMIN;
			p_local[UU] = UUMIN;

			//Set other scalars
			#if(DOKTOT)
			p_local[KTOT] = 0.0;
			#endif
			#if(TWO_T)
			p_local[ENTRE] = 0.0;
			p_local[ENTRI] = 0.0;
			#endif
			#if(P_NUM)
			p_local[PHOTON] = 1.e-30;
			#endif
			#if(RAD_M1)
			p_local[UU_RAD] = 1.e-30;
			#endif

			//Set fluid velocities to 0
			p_local[U1] = 0;
			p_local[U2] = 0;
			p_local[U3] = 0;

			#if(RAD_M1)
			p_local[U1_RAD] = p_local[U1];
			p_local[U2_RAD] = p_local[U2];
			p_local[U3_RAD] = p_local[U3];
			#endif

			//Set B-fields to 0
			p_local[B1] = 0;
			p_local[B2] = 0;
			p_local[B3] = 0;

			//Export results to global memory
			for (k = 0; k < NPR; k++) {
				pv[k * ksize + global_id] = p_local[k];
			}

			if (pflag_rbound[global_id - D1 * isize * ((icurr - D1) >= 0)] == 1) { //B1
				ps[0 * ksize + global_id] = 0.0;
			}
			if (pflag_rbound[global_id  - D2 * (BS_3 + 2 * N3G) * ((jcurr - D2) >= 0)] == 1) { //B2
				ps[1 * ksize + global_id] = 0.0;
			}
			if (pflag_rbound[global_id - D3 * ((zcurr - D3) >= 0)] == 1) { //B3
				ps[2 * ksize + global_id] = 0.0;
			}
		}
	}
}

__global__ void boundprim_cart(double * pv, double *  ps, int * pflag_cart, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet)
{
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize, icurr, jcurr, zcurr;
	isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);  
	zcurr = (global_id % (isize)) % (BS_3+2*N3G);
	jcurr = ((global_id - zcurr) % (isize)) / (BS_3+2*N3G);
	icurr = (global_id - (jcurr * (BS_3+2*N3G)+zcurr)) / (isize);
	global_id = isize * icurr + (BS_3 + 2 * N3G) * jcurr + zcurr;
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	int k=0;
	double p_local[NPR], alpha, vsq, gamma;
	struct of_geom geom;

	if (global_id < (BS_1+2*N1G) * (BS_2+2*N2G) * (BS_3+2*N3G)) k = 1;

	if (k==1 && pflag_cart[global_id] == 1) {
		//Get metric
		get_geometry(icurr, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);

		//Set density and internal energy
		p_local[RHO] = RHOMIN;
		p_local[UU] = UUMIN;

		//Set other scalars
		#if(DOKTOT)
		p_local[KTOT] = 0.0;
		#endif
		#if(TWO_T)
		p_local[ENTRE] = 0.0;
		p_local[ENTRI] = 0.0;
		#endif
		#if(P_NUM)
		p_local[PHOTON] = 1.e-30;
		#endif
		#if(RAD_M1)
		p_local[UU_RAD] = 1.e-30;
		#endif

		//Set fluid velocities to 0
		alpha = 1. / sqrt(-geom.gcon[0]);
		p_local[U1] = geom.gcon[1] * alpha;
		p_local[U2] = geom.gcon[2] * alpha;
		p_local[U3] = geom.gcon[3] * alpha;

		// now find new gamma and put it back in
		vsq = geom.gcov[4] * p_local[UTCON1 + 1 - 1] * p_local[UTCON1 + 1 - 1]; //1,1
		vsq += 2. * geom.gcov[5] * p_local[UTCON1 + 2 - 1] * p_local[UTCON1 + 1 - 1]; //1,2
		vsq += 2. * geom.gcov[6] * p_local[UTCON1 + 3 - 1] * p_local[UTCON1 + 1 - 1]; //1,3
		vsq += geom.gcov[7] * p_local[UTCON1 + 2 - 1] * p_local[UTCON1 + 2 - 1]; //2,2
		vsq += 2 * geom.gcov[8] * p_local[UTCON1 + 3 - 1] * p_local[UTCON1 + 2 - 1]; //2,3
		vsq += geom.gcov[9] * p_local[UTCON1 + 3 - 1] * p_local[UTCON1 + 3 - 1]; //3,3
		vsq = MY_MAX(1.e-13, vsq);
		if (vsq >= 1.) {
			vsq = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		}
		gamma = 1. / sqrt(1. - vsq);
		p_local[U1] *= gamma;
		p_local[U2] *= gamma;
		p_local[U3] *= gamma;
		#if(RAD_M1)
		p_local[U1_RAD] = p_local[U1];
		p_local[U2_RAD] = p_local[U2];
		p_local[U3_RAD] = p_local[U3];
		#endif

		//Set magnetic fields
		p_local[B1]=  pv[B1 * ksize + global_id];
		p_local[B2] = pv[B2 * ksize + global_id];
		p_local[B3] = pv[B3 * ksize + global_id];

		//Export results to global memory
		for (k = 0; k < NPR; k++) {
			pv[k * ksize + global_id] = p_local[k];
		}

		if (pflag_cart[global_id - D1 * isize * ((icurr - D1) >= 0)] == 1) { //B1
			//ps[0 * ksize + global_id] = 0.0;
		}
		if (pflag_cart[global_id  - D2 * (BS_3 + 2 * N3G) * ((jcurr - D2) >= 0)] == 1) { //B2
			//ps[1 * ksize + global_id] = 0.0;
		}
		if (pflag_cart[global_id - D3 * ((zcurr - D3) >= 0)] == 1) { //B3
			//ps[2 * ksize + global_id] = 0.0;
		}
	}
}



__global__ void boundprim1_NS(double* pv, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int NBR_2, int NBR_4, double* ps, const double* __restrict__ radius, const double* __restrict__ scaleCENT, const double* __restrict__ scaleFACE)
{
#if(NEUTRON_STAR)
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	int isize = (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G);
	int k, ii;
	int zcurr = global_id % (BS_3 + 2 * N3G);
	int jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	int fix_mem1 = LOCAL_WORK_SIZE - (isize * (BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	int ksize = isize * (BS_1 + 2 * N1G) + fix_mem1;
	double prim1[NPR], prim2[NPR], prim3[NPR], prim4[NPR], prim5[NPR], prim6[NPR];
	struct of_geom geom;


	// inner r boundary condition: u, gdet extrapolation
	if (jcurr >= 0 && jcurr < BS_2 + 2 * N2G && zcurr >= 0 && zcurr < BS_3 + 2 * N3G && NBR_4 == -1) {


#if(STAGGERED)
		ps[1 * (ksize)+0 * isize + global_id] = ps[1 * (ksize)+N1G * isize + global_id];
		ps[1 * (ksize)+1 * isize + global_id] = ps[1 * (ksize)+N1G * isize + global_id];
		ps[2 * (ksize)+0 * isize + global_id] = ps[2 * (ksize)+N1G * isize + global_id];
		ps[2 * (ksize)+1 * isize + global_id] = ps[2 * (ksize)+N1G * isize + global_id];
#if(N1G==3)
		ps[1 * (ksize)+2 * isize + global_id] = ps[1 * (ksize)+N1G * isize + global_id];
		ps[2 * (ksize)+2 * isize + global_id] = ps[2 * (ksize)+N1G * isize + global_id];
#endif
#endif


#pragma unroll NPR
		for (k = 0; k < NPR; k++) {
			prim5[k] = pv[k * (ksize)+N1G * isize + global_id]; //pFAZ
		}
		get_geometry(N1G, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
		int accreting, forcefree, useForcefreeBC;
		double ucon[NDIM], gamma, qsq;
		qsq = geom.gcov[4] * prim5[U1] * prim5[U1] + geom.gcov[7] * prim5[U2] * prim5[U2] + geom.gcov[9] * prim5[U3] * prim5[U3] + 2. * (geom.gcov[5] * prim5[U1] * prim5[U2] + geom.gcov[6] * prim5[U1] * prim5[U3] + geom.gcov[8] * prim5[U2] * prim5[U3]);
		gamma = sqrt(1. + qsq);
		ucon[0] = gamma * sqrt(-geom.gcon[0]);
		for (k = 1; k < NDIM; k++) {
			ucon[k] = prim5[k + U1 - 1] + ucon[0] * geom.gcon[k] / geom.gcon[0];
		}
		if (ucon[1] < 0.0)
			accreting = 1;
		else
			accreting = 0;

		if (prim5[FLRFRAC] > FFE_ZONE_FLRFRAC_THRESHOLD)
			forcefree = 1;
		else
			forcefree = 0;

		if (forcefree || (accreting == 0))
			useForcefreeBC = 1;
		else
			useForcefreeBC = 0;

		if (useForcefreeBC)
		{

			//basic_hydroStatic_atm(r_ghost1, &rho_temp, &uu_temp);
			pv[RHO * (ksize)+(N1G - 1) * isize + global_id] = RHO0_HYDROSTAT_ATM_NS * pow(MU_NS / 10.0, 2.0) * pow(radius[N1G-1] / R_NS, -1.0 / (GAMMA - 1.0));
			pv[UU * (ksize)+(N1G - 1) * isize + global_id] = (RHO0_HYDROSTAT_ATM_NS / (GAMMA * R_NS)) * pow(MU_NS / 10.0, 2.0) * pow(radius[N1G-1] / R_NS, GAMMA / (1.0 - GAMMA));

			//basic_hydroStatic_atm(r_ghost2, &rho_temp, &uu_temp);
			for (ii = 0; ii < N1G-1; ii++) {
				pv[RHO * (ksize)+ii * isize + global_id] = RHO0_HYDROSTAT_ATM_NS * pow(MU_NS / 10.0, 2.0) * pow(radius[N1G - 2] / R_NS, -1.0 / (GAMMA - 1.0));
				pv[UU * (ksize)+ii * isize + global_id] = (RHO0_HYDROSTAT_ATM_NS / (GAMMA * R_NS)) * pow(MU_NS / 10.0, 2.0) * pow(radius[N1G - 2] / R_NS, GAMMA / (1.0 - GAMMA));
			}
		}
#pragma unroll N1G
		for (ii = 0; ii < N1G; ii++)
		{
#if OBLIQUE_NS
			prim[nl[n]][index_3D(n, i, j, z)][B1] = calcRadialField(i, j, k, CENT, &geom);

#else /* aligned rotator: can store normal field */ 

			/*Here*/
			pv[B1 * (ksize)+ii * isize + global_id] = pv[B1 * (ksize)+N1G * isize + global_id] * scaleCENT[ii* isize + global_id];

			/*usually we do not set a bounds condition on ps[1]*/
			ps[1 * (ksize)+ii * isize + global_id] = ps[1 * (ksize)+N1G * isize + global_id] * scaleFACE[ii * isize + global_id];
#endif
			if (useForcefreeBC) {
				pv[FLR * (ksize)+ii * isize + global_id] = 1.0;
				pv[FLRFRAC * (ksize)+ii * isize + global_id] = 1.0;
				pv[KTOT * (ksize)+ii * isize + global_id] = 0.0;
			}
			else {
				pv[FLR * (ksize)+ii * isize + global_id] = pv[FLR * (ksize)+N1G * isize + global_id];
				pv[FLRFRAC * (ksize)+ii * isize + global_id] = pv[FLRFRAC * (ksize)+N1G * isize + global_id];
				pv[KTOT * (ksize)+ii * isize + global_id] = pv[KTOT * (ksize)+N1G * isize + global_id];
			}

		}

		//simple_extrap_prim(j, k, B2, prim);
		//simple_extrap_prim(j, k, B3, prim);

		double df;
		df = pv[B2 * (ksize)+(N1G+1)* isize + global_id] - pv[B2 * (ksize)+(N1G) * isize + global_id];
		pv[B2 * (ksize)+(N1G - 1) * isize + global_id] = pv[B2 * (ksize)+(N1G) * isize + global_id] - df;
		pv[B2 * (ksize)+(N1G - 2) * isize + global_id] = pv[B2 * (ksize)+(N1G) * isize + global_id] - 2.0 * df;
		pv[B2 * (ksize)+(N1G - 3) * isize + global_id] = pv[B2 * (ksize)+(N1G) * isize + global_id] - 3.0 * df;

		df = pv[B3 * (ksize)+(N1G + 1) * isize + global_id] - pv[B3 * (ksize)+(N1G)*isize + global_id];
		pv[B3 * (ksize)+(N1G - 1) * isize + global_id] = pv[B3 * (ksize)+(N1G)*isize + global_id] - df;
		pv[B3 * (ksize)+(N1G - 2) * isize + global_id] = pv[B3 * (ksize)+(N1G)*isize + global_id] - 2.0 * df;
		pv[B3 * (ksize)+(N1G - 3) * isize + global_id] = pv[B3 * (ksize)+(N1G)*isize + global_id] - 3.0 * df;

		/* Now do velocities */

		if (useForcefreeBC)
		{
			double bncon[NDIM], bscon[NDIM], uscon[NDIM], etacon[NDIM], etacov[NDIM];
			double bccon[NDIM], bccov[NDIM], uperpcon[NDIM], uperpcov[NDIM];
			double bs_dot_eta, us_dot_eta, bcsq, bc_dot_us, uperpsq;
			/*set_boundary_velocities_FFE_4Dmethod(n, i, j, k, CENT, prim[nl[n]][index_3D(n, i, j, z)]);*/
#pragma unroll 3
			for (ii = 0; ii < N1G; ii++) {
				get_geometry(ii, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
				bncon[0] = 0.0;
//#pragma unroll 3 //may test unroll later
				for (k = 1; k < NDIM; k++) {
					bncon[k] = pv[(B1 + k - 1) * (ksize)+ii * isize + global_id] / sqrt(-geom.gcon[0]); 
				}
				/* Surface-observer 4-velocity and magnetic field */
				get_surface_4velocity(gcov, uscon);
				get_surface_magneticField(gcov, gcon, bncon, uscon, bscon);

				/* Coordinate-observer 4-velocity and magnetic field */
				etacon[0] = sqrt(-1.0 / geom.gcov[0]); // i.e. eta = u_c

				for (k = 1; k < NDIM; k++) {
					etacon[k] = 0.0;
				}
				lower_KC(etacon, gcov, etacov);

				bs_dot_eta = dot(bscon, etacov);
				us_dot_eta = dot(uscon, etacov);


				for (k = 0; k < NDIM; k++) {
					bccon[k] = uscon[k] * bs_dot_eta - bscon[k] * us_dot_eta;
				}
				lower_KC(bccon, gcov, bccov);
				bcsq = dot(bccon, bccov);

				/* Project surface velocity us orthogonal to coordinate-observer magnetic field bc */
				bc_dot_us = dot(bccov, uscon);


				for (k = 0; k < NDIM; k++) {
					uperpcon[k] = uscon[k] - bccon[k] * bc_dot_us / bcsq;
				}

				/* Normalize: u = u_p / sqrt(- u_p^2) */
				lower_KC(uperpcon, gcov, uperpcov);
				uperpsq = dot(uperpcon, uperpcov);


				for (k = 0; k < NDIM; k++) {
					ucon[k] = uperpcon[k] / sqrt(-uperpsq);
				}

				/* Just use surface 4-velocity directly */
				//DLOOPA
				//    ucon[j] = uscon[j] ;

				for (k = 1; k < NDIM; k++) {
					pv[(U1 + k - 1) * (ksize)+ii * isize + global_id] = ucon[k] - geom.gcon[k] * ucon[0] / geom.gcon[0];
				}
			}


		}
		else
		{
			double bncon[NDIM], bscon[NDIM], bscov[NDIM], uscon[NDIM], bsmag;
			double uprllcon[NDIM], uprllsq;
			double udotb[3], d_udotb, udotb_ghost[N1G], beta_NS;
			//find_udotb_first3(n, j, k, prim, udotb);
#pragma unroll 3
			for (ii = 0; ii < 3; ii++) {
				get_geometry(ii, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
				for (k = 0; k < NPR; k++) {
					prim1[k] = pv[k * (ksize)+ii * isize + global_id]; 
				}
				ucon_calc(prim1, &geom, ucon);
				/* Normal-observer magnetic field */
				bncon[0] = 0.0;
				for (k = 1; k < NDIM; k++) {
					bncon[k] = -1.0 * pv[(B1 + k - 1) * (ksize)+ii * isize + global_id] / (geom.gcon[0]);
				}
				get_surface_4velocity(gcov, uscon);
				get_surface_magneticField(gcov, gcon, bncon, uscon, bscon);
				lower_KC(bscon, gcov, bscov);
				bsmag = sqrt(dot(bscon, bscov));
				udotb[ii] = dot(ucon, bscov) / bsmag;  // Store u.b/|b|
			}
			d_udotb = slope_lim(udotb[0], udotb[1], udotb[2], 0);
#pragma unroll N1G
			for (ii = 0; ii < N1G; ii++)
				udotb_ghost[ii] = udotb[0] + (ii - N1G) * d_udotb;
			//udotb_surface[j][k] = udotb[0] - 0.5 * d_udotb;

#pragma unroll N1G
			for (ii = 0; ii < N1G; ii++) {
				// set_boundary_velocities_surfaceFrame_4Dmethod(n, i, j, k, CENT, prim[i][j][k], udotb_ghost[i + N1G]);
				get_geometry(ii, jcurr, zcurr, CENT, &geom, gcov, gcon, gdet);
				bncon[0] = 0.0;
				for (k = 1; k < NDIM; k++) {
					bncon[k] = -1.0 * pv[(B1 + k - 1) * (ksize)+ii * isize + global_id] / (geom.gcon[0]);
				}
				/* Surface-observer 4-velocity and magnetic field */
				get_surface_4velocity(gcov, uscon);
				get_surface_magneticField(gcov, gcon, bncon, uscon, bscon);
				lower_KC(bscon, gcov, bscov);
				bsmag = sqrt(dot(bscon, bscov));
				beta_NS = udotb_ghost[ii] / sqrt(1.0 + udotb_ghost[ii] * udotb_ghost[ii]);
				for (k = 0; k < NDIM; k++) {
					ucon[k] = (uscon[k] + beta_NS * bscon[k] / bsmag) / sqrt(1.0 - beta_NS * beta_NS);
				}

				for (k = 1; k < NDIM; k++) {
					pv[(U1 + k - 1) * (ksize)+ii * isize + global_id] = ucon[k] - geom.gcon[k] * ucon[0] / geom.gcon[0];
				}
			}
		}

#pragma unroll NPR
		for (k = 0; k < NPR; k++) {
			prim1[k] = pv[k * (ksize)+1 * isize + global_id];
			prim2[k] = pv[k * (ksize)+global_id];
#if(N1G==3)
			prim3[k] = pv[k * (ksize)+2 * isize + global_id];
#endif
		}
		/*Make sure there is no inflow at inner boundary*/
		inflow_check(prim1, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		inflow_check(prim2, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
#if(N1G==3)
		inflow_check(prim3, 0, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
#endif
		inflow_check(prim1, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
		inflow_check(prim2, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
#if(N1G==3)
		inflow_check(prim3, 1, jcurr, zcurr, 0, gcov, gcon, gdet, 1);
#endif

		/*Write primitives back to global memory*/
#pragma unroll NPR
		for (k = 0; k < NPR; k++) {
			pv[k * (ksize)+global_id] = prim2[k];
			pv[k * (ksize)+1 * isize + global_id] = prim1[k];
#if(N1G==3)
			pv[k * (ksize)+2 * isize + global_id] = prim3[k];
#endif
		}



		global_id = -10;
		jcurr = -10;
		zcurr = -10;
	}

	if (global_id < isize) {
		global_id = -10;
		jcurr = -10;
		zcurr = -10;
	}
	else if (global_id >= isize) {
		global_id = global_id - isize;
		zcurr = global_id % (BS_3 + 2 * N3G);
		jcurr = (global_id - zcurr) / (BS_3 + 2 * N3G);
	}

	// outer r BC: outflow
#if(!CONSTANT_BC)
	if (jcurr >= 0 && jcurr < BS_2 + 2 * N2G && zcurr >= 0 && zcurr < BS_3 + 2 * N3G && NBR_2 == -1) {
#pragma unroll 9
		for (k = 0; k < NPR; k++) {
			prim6[k] = pv[k * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
		}

#pragma unroll 9
		for (k = 0; k < NPR; k++) {
			prim3[k] = prim6[k];
			prim4[k] = prim6[k];
			prim5[k] = prim6[k];
		}

		//Make sure there is no inflow at outer boundary
		inflow_check(prim3, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		inflow_check(prim4, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
#endif
		inflow_check(prim3, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
		inflow_check(prim4, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
#if(N1G==3)
		inflow_check(prim5, BS_1 + N1G + 1, jcurr, zcurr, 1, gcov, gcon, gdet, 1);
#endif

#pragma unroll 9
		for (k = 0; k < NPR; k++) {
			pv[k * (ksize)+(BS_1 + N1G) * isize + global_id] = prim3[k];
			pv[k * (ksize)+(BS_1 + N1G + 1) * isize + global_id] = prim4[k];
#if(N1G==3)
			pv[k * (ksize)+(BS_1 + N1G + 2) * isize + global_id] = prim5[k];
#endif
		}
#if(STAGGERED)
		ps[1 * (ksize)+(BS_1 + N1G) * isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
		ps[1 * (ksize)+(BS_1 + N1G + 1) * isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G) * isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G + 1) * isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
#if(N1G==3)
		ps[1 * (ksize)+(BS_1 + N1G + 2) * isize + global_id] = ps[1 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
		ps[2 * (ksize)+(BS_1 + N1G + 2) * isize + global_id] = ps[2 * (ksize)+(BS_1 + N1G - 1) * isize + global_id];
#endif
#endif
	}
#endif
#endif
}



/*** Find the contravariant components of the 4-velocity of the rotating stellar surface */
__device__ void get_surface_4velocity(const  double* __restrict__ gcov, double uscon[NDIM])
{
	double omega, omega0, t0, delta_t;
	t0 = SPINUP_START_TIME_NS;
	delta_t = SPINUP_TIME_NS;
	omega0 = OMEGA_NS;
	#if(0)
	if (t < t0)
		omega = 0.0;
	else if (t > t0 + delta_t)
		omega = omega0;
	else
		omega = omega0 * (t - t0) / delta_t;
	//omega = 0.5 * (1.0 - cos((t-t0)*M_PI/delta_t)) * omega0 ;
	#else
	omega = omega0;
	#endif

	uscon[0] = 1.0 / sqrt(-(gcov[0] + 2.0 * gcov[3] * omega + gcov[9] * omega * omega));
	uscon[1] = 0.0;
	uscon[2] = 0.0;
	uscon[3] = omega * uscon[0];

	return;
}

__device__ void get_surface_magneticField(const  double* __restrict__ gcov, const  double* __restrict__ gcon, double bncon[NDIM], double uscon[NDIM], double bscon[NDIM])
{
	double lapse, us_dot_n, bn_dot_us, bncov[NDIM];
	int j;

	lower_KC(bncon, gcov, bncov);

	lapse = sqrt(-1.0 / gcon[0]);
	us_dot_n = -lapse * uscon[0]; // Since normal observer n_mu = (-lapse, 0, 0, 0)
	bn_dot_us = dot(bncov, uscon);

	for (j = 0; j < NDIM; j++) {
		bscon[j] = -(bncon[j] + uscon[j] * bn_dot_us) / us_dot_n;
	}

	return;
}


__device__ void inflow_check(double* pr, int ii, int jj, int zz, int type, const  double* __restrict__ gcov, const  double* __restrict__ gcon, const  double* __restrict__ gdet, int dir)
{
	struct of_geom geom;
	double ucon[NDIM];
	double alpha, beta1, gamma, vsq;
	get_geometry(ii, jj, zz, CENT, &geom, gcov, gcon, gdet);
	ucon_calc(pr, &geom, ucon);

	if (((ucon[dir] > 0.) && (type == 0)) || ((ucon[dir] < 0.) && (type == 1))) {
		// find gamma and remove it from primitives 
		gamma_calc(pr, &geom, &gamma);
		pr[U1] /= gamma;
		pr[U2] /= gamma;
		pr[U3] /= gamma;
		alpha = 1. / sqrt(-geom.gcon[0]);
		beta1 = geom.gcon[dir] * alpha * alpha;

		// reset radial velocity so radial 4-velocity is zero
		pr[UU + dir] = beta1 / alpha;

		// now find new gamma and put it back in
		vsq = geom.gcov[4] * pr[UTCON1 + 1 - 1] * pr[UTCON1 + 1 - 1]; //1,1
		vsq += 2. * geom.gcov[5] * pr[UTCON1 + 2 - 1] * pr[UTCON1 + 1 - 1]; //1,2
		vsq += 2. * geom.gcov[6] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 1 - 1]; //1,3
		vsq += geom.gcov[7] * pr[UTCON1 + 2 - 1] * pr[UTCON1 + 2 - 1]; //2,2
		vsq += 2 * geom.gcov[8] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 2 - 1]; //2,3
		vsq += geom.gcov[9] * pr[UTCON1 + 3 - 1] * pr[UTCON1 + 3 - 1]; //3,3
		vsq = MY_MAX(1.e-13, vsq);
		if (vsq >= 1.) {
			vsq = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		}
		gamma = 1. / sqrt(1. - vsq);
		pr[U1] *= gamma;
		pr[U2] *= gamma;
		pr[U3] *= gamma;
	}

#if(0)
	double ucon_rad[NDIM], gamma_rad, vsq_rad;
	ucon_calc_rad(pr, &geom, ucon_rad);
	if (((ucon_rad[dir] > 0.) && (type == 0)) || ((ucon_rad[dir] < 0.) && (type == 1))) {
		/* find gamma and remove it from primitives */
		gamma_calc_rad(pr, &geom, &gamma_rad);
		pr[U1_RAD] /= gamma_rad;
		pr[U2_RAD] /= gamma_rad;
		pr[U3_RAD] /= gamma_rad;
		alpha = 1. / sqrt(-geom.gcon[0]);
		beta1 = geom.gcon[dir] * alpha * alpha;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[UU_RAD + dir] = beta1 / alpha;

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

#if(0)
	double ucon_nu[NDIM], gamma_nu, vsq_nu;
	for (int sp = 0; sp < NU_SPECIES; sp++) {
		ucon_calc_nu(pr, &geom, ucon_nu, sp);
		if (((ucon_nu[1] > 0.) && (type == 0)) || ((ucon_nu[1] < 0.) && (type == 1))) {
			/* find gamma and remove it from primitives */
			gamma_calc_nu(pr, &geom, &gamma_nu, sp);
			pr[index_nu(U1_NU, sp)] /= gamma_nu;
			pr[index_nu(U2_NU, sp)] /= gamma_nu;
			pr[index_nu(U3_NU, sp)] /= gamma_nu;
			alpha = 1. / sqrt(-geom.gcon[0]);
			beta1 = geom.gcon[1] * alpha * alpha;

			/* reset radial velocity so radial 4-velocity is zero */
			pr[index_nu(U1_NU, sp)] = beta1 / alpha;

			// now find new gamma and put it back in 		
			vsq_nu = geom.gcov[4] * pr[index_nu(U1_NU, sp) + 1 - 1] * pr[index_nu(U1_NU, sp) + 1 - 1]; //1,1
			vsq_nu += 2. * geom.gcov[5] * pr[index_nu(U1_NU, sp) + 2 - 1] * pr[index_nu(U1_NU, sp) + 1 - 1]; //1,2
			vsq_nu += 2. * geom.gcov[6] * pr[index_nu(U1_NU, sp) + 3 - 1] * pr[index_nu(U1_NU, sp) + 1 - 1]; //1,3
			vsq_nu += geom.gcov[7] * pr[index_nu(U1_NU, sp) + 2 - 1] * pr[index_nu(U1_NU, sp) + 2 - 1]; //2,2
			vsq_nu += 2 * geom.gcov[8] * pr[index_nu(U1_NU, sp) + 3 - 1] * pr[index_nu(U1_NU, sp) + 2 - 1]; //2,3
			vsq_nu += geom.gcov[9] * pr[index_nu(U1_NU, sp) + 3 - 1] * pr[index_nu(U1_NU, sp) + 3 - 1]; //3,3

			vsq_nu = MY_MAX(1.e-13, vsq_nu);
			if (vsq_nu >= 1.) {
				vsq_nu = 1. - 1. / (GAMMAMAX_NU * GAMMAMAX_NU);
			}
			gamma_nu = 1. / sqrt(1. - vsq_nu);
			pr[index_nu(U1_NU, sp)] *= gamma_nu;
			pr[index_nu(U2_NU, sp)] *= gamma_nu;
			pr[index_nu(U3_NU, sp)] *= gamma_nu;

			/* done */
		}
	}
#endif
}

__device__ void extrapolate_gdet_innerBC(double* pr_B, double* pr_ghost, const double gdet_B, const double gdet_ghost, double dr_over_r)
{
	// Boundary conditions at the innermost boundary (gdet interpolation)
#if(DANAT_GDET_INTERP)	
// Extrapolate the density, temperature, number density and energy density of the neutrinos
	pr_ghost[RHO] = pr_B[RHO] * gdet_B / gdet_ghost;
	pr_ghost[UU] = pr_B[UU] * gdet_B / gdet_ghost;
	// Radial velocity
	pr_ghost[U1] = pr_B[U1] * (1. + dr_over_r);
	// Theta, phi velocity
	pr_ghost[U2] = pr_B[U2] * (1. - dr_over_r);
	pr_ghost[U3] = pr_B[U3] * (1. - dr_over_r);
	// B-field 
	pr_ghost[B1] = pr_B[B1] * (1. + dr_over_r);
	// Theta, phi velocity 
	pr_ghost[B2] = pr_B[B2] * (1. - dr_over_r);
	pr_ghost[B3] = pr_B[B3] * (1. - dr_over_r);
#if(DO_YE)
	pr_ghost[YE] = pr_B[YE] * gdet_B / gdet_ghost;
#endif

#if(NEUTRINOS_M1)
	for (int sp = 0; sp < NU_SPECIES; sp++) {
		pr_ghost[index_nu(UU_NU, sp)] = pr_B[index_nu(UU_NU, sp)] * gdet_B / gdet_ghost;
		pr_ghost[index_nu(U1_NU, sp)] = pr_B[index_nu(U1_NU, sp)] * (1. + dr_over_r);
		pr_ghost[index_nu(U2_NU, sp)] = pr_B[index_nu(U2_NU, sp)] * (1. - dr_over_r);
		pr_ghost[index_nu(U3_NU, sp)] = pr_B[index_nu(U3_NU, sp)] * (1. - dr_over_r);
		pr_ghost[index_nu(NUMBER_NU, sp)] = pr_B[index_nu(NUMBER_NU, sp)] * gdet_B / gdet_ghost;
	}
#endif
#endif
}

/* Lowers a contravariant rank-1 tensor to a covariant one */
__device__ void lower_KC(double ucon[NDIM], const  double* __restrict__ gcov, double ucov[NDIM])
{
#if AMD
	ucov[0] = fma(gcov[0], ucon[0], fma(gcov[1], ucon[1], fma(gcov[2], ucon[2], gcov[3] * ucon[3])));
	ucov[1] = fma(gcov[1], ucon[0], fma(gcov[4], ucon[1], fma(gcov[5], ucon[2], gcov[6] * ucon[3])));
	ucov[2] = fma(gcov[2], ucon[0], fma(gcov[5], ucon[1], fma(gcov[7], ucon[2], gcov[8] * ucon[3])));
	ucov[3] = fma(gcov[3], ucon[0], fma(gcov[6], ucon[1], fma(gcov[8], ucon[2], gcov[9] * ucon[3])));
	return;
#else
	ucov[0] = gcov[0] * ucon[0] + gcov[1] * ucon[1] + gcov[2] * ucon[2] + gcov[3] * ucon[3];
	ucov[1] = gcov[1] * ucon[0] + gcov[4] * ucon[1] + gcov[5] * ucon[2] + gcov[6] * ucon[3];
	ucov[2] = gcov[2] * ucon[0] + gcov[5] * ucon[1] + gcov[7] * ucon[2] + gcov[8] * ucon[3];
	ucov[3] = gcov[3] * ucon[0] + gcov[6] * ucon[1] + gcov[8] * ucon[2] + gcov[9] * ucon[3];
#endif
}