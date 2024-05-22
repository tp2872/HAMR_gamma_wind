#include "include.h"
#include "decs_MPI.h"
void bound_prim1_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_reflective(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim3_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim1_NS(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n, double t);

/* bound array containing entire set of primitive variables */
void bound_prim(double(*restrict prim[NB_LOCAL])[NPR], int bound_force, double t)
{
	int i, n, flag;
	double temp = nstep;
	if (bound_force == 1) nstep = -1;

#if(BOUND_TYPE1==NEUTRON_STAR_BC)	
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim1_NS(p, ps, n_ord[n], t);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim1_NS(ph, psh, n_ord[n], t);
	}
#elif(BOUND_TYPE1==OUTFLOW)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim1_outflow(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim1_outflow(ph, psh, n_ord[n]);
	}
#endif

#if(BOUND_TYPE2==OUTFLOW)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2_outflow(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2_outflow(ph, psh, n_ord[n]);
	}
#elif(BOUND_TYPE2==REFLECTIVE)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2_reflective(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2_reflective(ph, psh, n_ord[n]);
	}
#endif

#if(BOUND_TYPE3==OUTFLOW)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim3_outflow(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim3_outflow(ph, psh, n_ord[n]);
	}
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	rc = 0;
	gpu = 0;
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++) {
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++) {
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	if (N3 > 1) {
		for (n = 0; n < n_active; n++) {
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		set_iprobe(0, &flag);
		do {
			for (n = 0; n < n_active; n++) {
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
			}
			set_iprobe(1, &flag);
		} while (flag);
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");

#if(BOUND_TYPE2==TRANSMISSIVE && NB_3==1)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2_trans(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2_trans(ph, psh, n_ord[n]);
	}
#endif

	//#if(CARTESIAN_GR)
	//for (n = 0; n < n_active; n++) {
	//	bound_prim_cart(p, ps, 1, n_ord[n]);
	//	bound_prim_cart(ph, psh, 1, n_ord[n]);
	//}
	//#endif

#if(DO_RBOUND)
	for (n = 0; n < n_active; n++) {
		bound_prim_rbound(p, ps, 1, n_ord[n]);
		bound_prim_rbound(ph, psh, 1, n_ord[n]);
	}
#endif

#if (STAGGERED && COPY_BFIELD)
	rc = 0;
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1 || bound_force == 1) { //watch out does this for both half and full timestep while only needed for full timestep
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send2(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1) {
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send3(ps, Bufferps_1, n_ord[n]);
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) B_rec3(ps, Bufferps_1, n_ord[n]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB \n");
#endif

	if (bound_force == 1) nstep = temp;
}

void set_iprobe(int mode, int* flag) {
#if(TASK_BASED)
	int i, n;
	*flag = 0;
	for (n = 0; n < n_active; n++) {
		if (mode == 0) {
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++) block[n_ord[n]][i] = 0;
		}
		else {
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++) {
				if (block[n_ord[n]][i] == -1) {
					block[n_ord[n]][i] = 0;
					*flag = 1;
				}
				else  block[n_ord[n]][i] = 1;
			}
		}
	}
#else
	* flag = 0;
#endif
	return;
}

void bound_prim1_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n) {
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	if (block[n][AMR_NBR4] == -1) {
#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
					//#pragma omp   simd
					for (i = -N1G; i < 0; i++) {
						for (k = 0; k < NPR - USE_PS1START; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, 0, j, z)][k];
						}
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, 0, j, z)][2];
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, 0, j, z)][3];
#endif
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, 0, j, z)];
					}
				}
			}
		}
	}

#if(!CONSTANT_BC)
	if (block[n][AMR_NBR2] == -1) {
		// outer r BC: outflow 		
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
					for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G; i++) {
						for (k = 0; k < NPR - USE_PS1START; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
						}

						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)];
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][2];
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][3];
#endif
					}
				}
			}
		}
	}
#endif

	// make sure there is no inflow at the inner boundary 
	if (block[n][AMR_NBR4] == -1) {
		for (i = -N1G; i <= -1; i++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, -1, j, z)], n, i, j, z, 0, 1);
						inflow_check(prim[nl[n]][index_3D(n, -2, j, z)], n, i, j, z, 0, 1);
#if(N1G==3)
						inflow_check(prim[nl[n]][index_3D(n, -3, j, z)], n, i, j, z, 0, 1);
#endif
					}
				}
			}
		}
	}

	// make sure there is no inflow at the outer boundary
#if(!CONSTANT_BC)
	if (block[n][AMR_NBR2] == -1) {
		for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i <= N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G - 1; i++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]), j, z)], n, i, j, z, 1, 1);
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 1, j, z)], n, i, j, z, 1, 1);
#if(N1G==3)
						inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 2, j, z)], n, i, j, z, 1, 1);
#endif
					}
				}
			}
		}
	}
#endif
}

void bound_prim2_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n) {
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	if (block[n][AMR_NBR1] == -1) {
#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
					//#pragma omp   simd
					for (j = -N2G; j < 0; j++) {
						for (k = 0; k < NPR; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, 0, z)][k];
						}
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, 0, z)][1];
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i, 0, z)][3];
#endif
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, i, 0, z)];
					}
				}
			}
		}
	}

	if (block[n][AMR_NBR3] == -1) {
		// outer r BC: outflow 		
#pragma omp parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)];
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][1];
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][3];
#endif
					}
				}
			}
		}
	}

	// make sure there is no inflow at the inner boundary 
	if (block[n][AMR_NBR1] == -1) {
		for (j = -N2G; j <= -1; j++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, j) private(i,z)
			{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, i, -1, z)], n, i, j, z, 0, 2);
						inflow_check(prim[nl[n]][index_3D(n, i, -2, z)], n, i, j, z, 0, 2);
#if(N2G==3)
						inflow_check(prim[nl[n]][index_3D(n, i, -2, z)], n, i, j, z, 0, 2);
#endif
					}
				}
			}
		}
	}

	// make sure there is no inflow at the outer boundary
	if (block[n][AMR_NBR3] == -1) {
		for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j <= N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G - 1; j++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, j) private(i,z)
			{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
					for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
						inflow_check(prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)], n, i, j, z, 1, 2);
						inflow_check(prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)], n, i, j, z, 1, 2);
#if(N2G==3)
						inflow_check(prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)], n, i, j, z, 1, 2);
#endif
					}
				}
			}
		}
	}
}

void bound_prim2_reflective(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n) {
	int i, j, z, k, jref;

	//copy all densities and B^phi in; interpolate linearly transverse velocity
#if(POLEFIX && POLEFIX < N2/2)
	jref = POLEFIX;
	if (block[n][AMR_NBR1] == -1) {
#pragma omp   parallel shared(n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2 || (N3 > 1 && k == B3))
							//don't touch magnetic fields
							continue;
							#if(RESISTIVE)
							if (k == E1 || k == E2 || (N3 > 1 && k == E3))
								//don't touch electric fields
								continue;
								#endif
								#if(NEUTRON_STAR && USE_PS1START)
								if (k == PS1START)
									continue;
								#endif
								else if (k == U2) {
									//linear interpolation of transverse velocity (both poles)
									prim[nl[n]][index_3D(n, i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, jref, z)][k];
								}
								#if(RAD_M1)
								else if (k == U2_RAD) {
									//linear interpolation of transverse velocity (both poles)
									prim[nl[n]][index_3D(n, i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, jref, z)][k];
								}
								#endif
								#if(NEUTRINOS_M1)
								else if (k == U2_NU) {
									//linear interpolation of transverse velocity (both poles)
									prim[nl[n]][index_3D(n, i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, jref, z)][k];
								}
								#if (NU_SPECIES > 1)
								else if (k == index_nu(U2_NU, 1) || k == index_nu(U2_NU, 2)) {
									//linear interpolation of transverse velocity (both poles)
									prim[nl[n]][index_3D(n, i, j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, jref, z)][k];
								}
								#endif
								#endif
								else {
									//everything else copy (both poles)
									prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, jref, z)][k];
								}
						}
					}
				}
			}
		}
	}
	if (block[n][AMR_NBR3] == -1) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = 0; j < jref; j++) {
						PLOOP{
							if (k == B1 || k == B2 || (N3 > 1 && k == B3))
							//don't touch magnetic fields
							continue;
							#if(RESISTIVE)
							if (k == E1 || k == E2 || (N3 > 1 && k == E3))
								//don't touch electric fields
								continue;
							#endif
							#if(NEUTRON_STAR && USE_PS1START)
							if (k == PS1START)
								continue;
							#endif
							else if (k == U2) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
							#if(RAD_M1)
							else if (k == U2_RAD) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
							#endif
							#if(NEUTRINOS_M1)
							else if (k == U2_NU) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
							#if (NU_SPECIES > 1)
							else if (k == index_nu(U2_NU, 1) || k == index_nu(U2_NU, 2)) {
								//linear interpolation of transverse velocity (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = (j + 0.5) / (jref + 0.5) * prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
							#endif
							#endif
							else {
								//everything else copy (both poles)
								prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - jref, z)][k];
							}
						}
					}
				}
			}
		}
	}
#endif

	// polar BCs 
	if (block[n][AMR_NBR1] == -1) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					for (k = 0; k < NPR - USE_PS1START; k++) {
						prim[nl[n]][index_3D(n, i, -1, z)][k] = prim[nl[n]][index_3D(n, i, 0, z)][k];
						prim[nl[n]][index_3D(n, i, -2, z)][k] = prim[nl[n]][index_3D(n, i, 1, z)][k];
#if(N1G==3)
						prim[nl[n]][index_3D(n, i, -3, z)][k] = prim[nl[n]][index_3D(n, i, 2, z)][k];
#endif
					}
					pflag[nl[n]][index_3D(n, i, -1, z)] = pflag[nl[n]][index_3D(n, i, 0, z)];
#if(STAGGERED)
					k = 1;
					ps[nl[n]][index_3D(n, i, -1, z)][k] = ps[nl[n]][index_3D(n, i, 0, z)][k];
					ps[nl[n]][index_3D(n, i, -2, z)][k] = ps[nl[n]][index_3D(n, i, 1, z)][k];
#if(N2G==3)
					ps[nl[n]][index_3D(n, i, -3, z)][k] = ps[nl[n]][index_3D(n, i, 2, z)][k];
#endif
#if(N3>1)
					k = 3;
					ps[nl[n]][index_3D(n, i, -1, z)][k] = ps[nl[n]][index_3D(n, i, 0, z)][k];
					ps[nl[n]][index_3D(n, i, -2, z)][k] = ps[nl[n]][index_3D(n, i, 1, z)][k];
#if(N2G==3)
					ps[nl[n]][index_3D(n, i, -3, z)][k] = ps[nl[n]][index_3D(n, i, 2, z)][k];
#endif
#endif			
#endif
				}
			}
		}
	}

	if (block[n][AMR_NBR3] == -1) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					for (k = 0; k < NPR - USE_PS1START; k++) {
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
#if(N1G==3)
						prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
#endif
					}
					pflag[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)] = pflag[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)];
#if(STAGGERED)
					k = 1;
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
#if(N2G==3)
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
#endif
#if(N3>1)
					k = 3;
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]), z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1, z)][k];
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 1, z)][k] = ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 2, z)][k];
#if(N2G==3)
					ps[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + 2, z)][k] = prim[nl[n]][index_3D(n, i, N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - 3, z)][k];
#endif
#endif			
#endif
				}
			}
		}
	}

	// make sure b and u are antisymmetric at the poles 
	if (block[n][AMR_NBR1] == -1) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.;
#endif
#if(NEUTRINOS_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_NU] *= -1.;
#if (NU_SPECIES > 1)
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 1)] *= -1.;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 2)] *= -1.;
#endif
#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
	if (block[n][AMR_NBR3] == -1) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.;
#endif
#if(NEUTRINOS_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_NU] *= -1.;
#if(NU_SPECIES > 1)
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 1)] *= -1.;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 2)] *= -1.;
#endif
#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
}

void bound_prim2_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n) {
	int i, j, z, k;

	// polar BCs 
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++) {
						//#pragma omp   simd
						for (k = 0; k < NPR - USE_PS1START; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][k];
						}
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] *= -1.0;
#endif
#if(NEUTRINOS_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_NU] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_NU] *= -1.0;
#if (NU_SPECIES > 1)
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 1)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 2)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U3_NU, 1)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U3_NU, 2)] *= -1.0;
#endif
#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B3] *= -1.0;

#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][1];
#if(N3>1)
						ps[nl[n]][index_3D(n, i, j, z)][3] = -ps[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][3];
#endif			
#endif
					}
				}
			}
		}
	}

	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++) {
						//#pragma omp   simd
						for (k = 0; k < NPR - USE_PS1START; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1, (z + BS_3 / 2) % BS_3)][k];
						}
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] *= -1.0;
#endif
#if(NEUTRINOS_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_NU] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_NU] *= -1.0;
#if (NU_SPECIES > 1)
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 1)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U2_NU, 2)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U3_NU, 1)] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][index_nu(U3_NU, 2)] *= -1.0;
#endif
#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B3] *= -1.0;
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1, (z + BS_3 / 2) % BS_3)][1];
#if(N3>1)
						ps[nl[n]][index_3D(n, i, j, z)][3] = -ps[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1, (z + BS_3 / 2) % BS_3)][3];
#endif			
#endif
					}
				}
			}
		}
	}
}

void bound_prim3_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n) {
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	if (block[n][AMR_NBR6] == -1) {
#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
					//#pragma omp   simd
					for (z = -N3G; z < 0; z++) {
						for (k = 0; k < NPR; k++) {
							prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, j, 0)][k];
						}
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, j, 0)][1];
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i, j, 0)][2];
#endif
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, i, j, 0)];
					}
				}
			}
		}
	}

	if (block[n][AMR_NBR5] == -1) {
		// outer r BC: outflow 		
#pragma omp parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
					for (z = N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]); z < N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + N3G; z++) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1)][k];
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1)];
#if(STAGGERED)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1)][1];
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) - 1)][2];
#endif
					}
				}
			}
		}
	}

	// make sure there is no inflow at the inner boundary 
	if (block[n][AMR_NBR6] == -1) {
		for (z = -N3G; z <= -1; z++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, z) private(j,i)
			{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
					for (j = -N2G + N2_GPU_offset[n]; j < BS_2 + N2_GPU_offset[n] + N2G; j++) {
						inflow_check(prim[nl[n]][index_3D(n, i, j, -1)], n, i, j, z, 0, 3);
						inflow_check(prim[nl[n]][index_3D(n, i, j, -2)], n, i, j, z, 0, 3);
#if(N3G==3)
						inflow_check(prim[nl[n]][index_3D(n, i, j, -3)], n, i, j, z, 0, 3);
#endif
					}
				}
			}
		}
	}

	// make sure there is no inflow at the outer boundary
	if (block[n][AMR_NBR5] == -1) {
		for (z = N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]); z <= N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + N3G - 1; z++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, z) private(j,i)
			{
#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
					for (j = -N2G + N2_GPU_offset[n]; j < BS_2 + N2_GPU_offset[n] + N2G; j++) {
						inflow_check(prim[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]))], n, i, j, z, 1, 3);
						inflow_check(prim[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + 1)], n, i, j, z, 1, 3);
#if(N3G==3)
						inflow_check(prim[nl[n]][index_3D(n, i, j, N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + 2)], n, i, j, z, 1, 3);
#endif
					}
				}
			}
		}
	}
}

//Set Cartesian boundary conditions
void bound_prim_cart(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int dir, int n) {
	int i, j, z, k;
	struct of_geom geom;
	double alpha, vsq, gamma;

	ZSLOOP3D(N1_GPU_offset[n] - N1G, BS_1 + N1_GPU_offset[n] + N1G - 1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G - 1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 + N3G - 1) {
		if (pflag_cart[nl[n]][index_3D(n, i, j, z)] == 1) {
			//Get metric
			get_geometry(n, i, j, z, CENT, &geom);

			//Set density and internal energy
			prim[nl[n]][index_3D(n, i, j, z)][RHO] = RHOMIN;
			prim[nl[n]][index_3D(n, i, j, z)][UU] = UUMIN;

			//Set other scalars
#if(DOKTOT)
			prim[nl[n]][index_3D(n, i, j, z)][KTOT] = 0.0;
#endif
#if(TWO_T)
			prim[nl[n]][index_3D(n, i, j, z)][ENTRE] = 0.0;
			prim[nl[n]][index_3D(n, i, j, z)][ENTRI] = 0.0;
#endif
#if(P_NUM)
			prim[nl[n]][index_3D(n, i, j, z)][PHOTON] = 1.e-30;
#endif
#if(RAD_M1)
			prim[nl[n]][index_3D(n, i, j, z)][UU_RAD] = 1.e-30;
#endif

			//Set fluid velocities to 0
			alpha = 1. / sqrt(-geom.gcon[0][0]);
			prim[nl[n]][index_3D(n, i, j, z)][U1] = geom.gcon[0][1] * alpha;
			prim[nl[n]][index_3D(n, i, j, z)][U2] = geom.gcon[0][2] * alpha;
			prim[nl[n]][index_3D(n, i, j, z)][U3] = geom.gcon[0][3] * alpha;

			// now find new gamma and put it back in
			SLOOP vsq += geom.gcov[j][k] * prim[nl[n]][index_3D(n, i, j, z)][U1 + j - 1] * prim[nl[n]][index_3D(n, i, j, z)][U1 + k - 1];
			vsq = MY_MAX(1.e-13, vsq);
			if (vsq >= 1.) {
				vsq = 1. - 1. / (GAMMAMAX * GAMMAMAX);
			}
			gamma = 1. / sqrt(1. - vsq);
			prim[nl[n]][index_3D(n, i, j, z)][U1] *= gamma;
			prim[nl[n]][index_3D(n, i, j, z)][U2] *= gamma;
			prim[nl[n]][index_3D(n, i, j, z)][U3] *= gamma;
#if(RAD_M1)
			prim[nl[n]][index_3D(n, i, j, z)][U1_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U1];
			prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U2];
			prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U3];
#endif

			//Set (staggered) magnetic field components
			//prim[nl[n]][index_3D(n, i, j, z)][B1] = 0.0;
			//prim[nl[n]][index_3D(n, i, j, z)][B2] = 0.0;
			//prim[nl[n]][index_3D(n, i, j, z)][B3] = 0.0;

			if (pflag_cart[nl[n]][index_3D(n, i - D1 * ((i - D1) >= 0), j, z)] == 1) { //B1
				//ps[nl[n]][index_3D(n, i, j, z)][1] = 0.0;
			}
			if (pflag_cart[nl[n]][index_3D(n, i, j - D2 * ((j - D2) >= 0), z)] == 1) { //B2
				//ps[nl[n]][index_3D(n, i, j, z)][2] =  0.0;
			}
			if (pflag_cart[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= 0))] == 1) { //B3
				//ps[nl[n]][index_3D(n, i, j, z)][3] = 0.0;
			}
		}
	}
}

// For spherical coords, set inflow to cells beneath RBOUND when DO_RBOUND is 1
void bound_prim_rbound(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int dir, int n) {
	int i, i2, j, z, k, tag = 0;

	ZSLOOP3D(N1_GPU_offset[n] - N1G, BS_1 + N1_GPU_offset[n] + N1G - 1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G - 1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 + N3G - 1) {
		if (pflag_rbound[nl[n]][index_3D(n, i, j, z)] == 1) {
#if(RBOUND_INFLOW)
			tag = 0;
			for (i2 = 1; i2 <= N1G; i2++) {
				if ((i + i2 < BS_1 + N1_GPU_offset[n] + N1G) && (pflag_rbound[nl[n]][index_3D(n, i + i2, j, z)] == 0)) {
					PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i + i2, j, z)][k];
#if(STAGGERED)
					ps[nl[n]][index_3D(n, i, j, z)][1] = 0.0;
					ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i + i2, j, z)][2];
					ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i + i2, j, z)][3];
#endif
					tag = 1;
					break;
				}
			}
#endif		
			if (tag == 0) {
				//Set density and internal energy
				prim[nl[n]][index_3D(n, i, j, z)][RHO] = RHOMIN;
				prim[nl[n]][index_3D(n, i, j, z)][UU] = UUMIN;

				//Set other scalars
#if(DOKTOT)
				prim[nl[n]][index_3D(n, i, j, z)][KTOT] = 0.0;
#endif
#if(TWO_T)
				prim[nl[n]][index_3D(n, i, j, z)][ENTRE] = 0.0;
				prim[nl[n]][index_3D(n, i, j, z)][ENTRI] = 0.0;
#endif
#if(P_NUM)
				prim[nl[n]][index_3D(n, i, j, z)][PHOTON] = 1.e-30;
#endif
#if(RAD_M1)
				prim[nl[n]][index_3D(n, i, j, z)][UU_RAD] = 1.e-30;
#endif

#if(NEUTRON_STAR)
				prim[nl[n]][index_3D(n, i, j, z)][FLR] = 1.0;
#if(DOFLR)
				prim[nl[n]][index_3D(n, i, j, z)][FLRFRAC] = 1.0;
#endif
#endif

				//Set fluid velocities to 0
				prim[nl[n]][index_3D(n, i, j, z)][U1] = 0;
				prim[nl[n]][index_3D(n, i, j, z)][U2] = 0;
				prim[nl[n]][index_3D(n, i, j, z)][U3] = 0;

#if(RAD_M1)
				prim[nl[n]][index_3D(n, i, j, z)][U1_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U1];
				prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U2];
				prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] = prim[nl[n]][index_3D(n, i, j, z)][U3];
#endif

				//Set magnetic fields to 0
				prim[nl[n]][index_3D(n, i, j, z)][B1] = 0;
				prim[nl[n]][index_3D(n, i, j, z)][B2] = 0;
				prim[nl[n]][index_3D(n, i, j, z)][B3] = 0;

				if (pflag_rbound[nl[n]][index_3D(n, i - D1 * ((i - D1) >= N1_GPU_offset[n] - N1G), j, z)] == 1) { //B1
					ps[nl[n]][index_3D(n, i, j, z)][1] = 0.0;
				}
				if (pflag_rbound[nl[n]][index_3D(n, i, j - D2 * ((j - D2) >= N2_GPU_offset[n] - N2G), z)] == 1) { //B2
					ps[nl[n]][index_3D(n, i, j, z)][2] = 0.0;
				}
				if (pflag_rbound[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= N3_GPU_offset[n] - N3G))] == 1) { //B3
					ps[nl[n]][index_3D(n, i, j, z)][3] = 0.0;
				}
			}
		}
	}
}

void inflow_check(double* restrict pr, int n, int ii, int jj, int zz, int type, int dir) {
	struct of_geom geom;
	double ucon[NDIM];
	int j, k;
	double alpha, beta1, gamma, vsq;

	get_geometry(n, ii, jj, zz, CENT, &geom);
	ucon_calc(pr, &geom, ucon);

	if (((ucon[dir] > 0.) && (type == 0)) || ((ucon[dir] < 0.) && (type == 1))) {
		/* find gamma and remove it from primitives */
		if (gamma_calc(pr, &geom, &gamma)) {
			fprintf(stderr, "\ninflow_check(): gamma failure \n");
			fail(FAIL_GAMMA);
		}
		pr[U1] /= gamma;
		pr[U2] /= gamma;
		pr[U3] /= gamma;
		alpha = 1. / sqrt(-geom.gcon[0][0]);
		beta1 = geom.gcon[0][dir] * alpha * alpha;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[UU + dir] = beta1 / alpha;

		/* now find new gamma and put it back in */
		vsq = 0.;
		SLOOP vsq += geom.gcov[j][k] * pr[U1 + j - 1] * pr[U1 + k - 1];
		if (fabs(vsq) < 1.e-13)  vsq = 1.e-13;
		if (vsq >= 1.) {
			vsq = 1. - 1. / (GAMMAMAX * GAMMAMAX);
		}
		gamma = 1. / sqrt(1. - vsq);
		pr[U1] *= gamma;
		pr[U2] *= gamma;
		pr[U3] *= gamma;

		/* done */
	}

#if(0)
	double ucon_rad[NDIM], gamma_rad, vsq_rad;
	ucon_calc_rad(pr, &geom, ucon_rad);
	if (((ucon_rad[dir] > 0.) && (type == 0)) || ((ucon_rad[dir] < 0.) && (type == 1))) {
		/* find gamma and remove it from primitives */
		if (gamma_calc_rad(pr, &geom, &gamma_rad)) {
			fprintf(stderr, "\ninflow_check(): gamma failure \n");
			fail(FAIL_GAMMA);
		}
		pr[U1_RAD] /= gamma_rad;
		pr[U2_RAD] /= gamma_rad;
		pr[U3_RAD] /= gamma_rad;
		alpha = 1. / sqrt(-geom.gcon[0][0]);
		beta1 = geom.gcon[0][dir] * alpha * alpha;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[UU_RAD + dir] = beta1 / alpha;

		/* now find new gamma and put it back in */
		vsq_rad = 0.;
		SLOOP vsq_rad += geom.gcov[j][k] * pr[U1_RAD + j - 1] * pr[U1_RAD + k - 1];
		if (fabs(vsq_rad) < 1.e-13)  vsq_rad = 1.e-13;
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

#if(NEUTRINOS_M1)
	double ucon_nu[NDIM], gamma_nu, vsq_nu;
	int sp;
	for (sp = 0; sp < NU_SPECIES; sp++) {
		ucon_calc_nu(pr, &geom, ucon_nu, sp);
		if (((ucon_nu[1] > 0.) && (type == 0)) || ((ucon_nu[1] < 0.) && (type == 1))) {
			/* find gamma and remove it from primitives */
			if (gamma_calc_nu(pr, &geom, &gamma_nu, sp)) {
				fprintf(stderr, "\ninflow_check(): gamma failure \n");
				fail(FAIL_GAMMA);
			}
			pr[index_nu(U1_NU, sp)] /= gamma_nu;
			pr[index_nu(U2_NU, sp)] /= gamma_nu;
			pr[index_nu(U3_NU, sp)] /= gamma_nu;
			alpha = 1. / sqrt(-geom.gcon[0][0]);
			beta1 = geom.gcon[0][1] * alpha * alpha;

			/* reset radial velocity so radial 4-velocity is zero */
			pr[index_nu(U1_NU, sp)] = beta1 / alpha;

			/* now find new gamma and put it back in */
			vsq_nu = 0.;
			SLOOP vsq_nu += geom.gcov[j][k] * pr[index_nu(U1_NU, sp) + j - 1] * pr[index_nu(U1_NU, sp) + k - 1];
			if (fabs(vsq_nu) < 1.e-13)  vsq_nu = 1.e-13;
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

double angleRotated(double t)
{
	double phi, omega0, t0, delta_t;
	t0 = SPINUP_START_TIME_NS;
	delta_t = SPINUP_TIME_NS;
	omega0 = OMEGA_NS;

	if (t < t0)
		phi = 0.0;
	else if (t > t0 + delta_t)
		phi = omega0 * (t - t0 - 0.5 * delta_t);
	else
		phi = 0.5 * omega0 * (t - t0) * (t - t0) / delta_t;

	return phi;
}

/* Assume that radial coord lines are straight at and inside the stellar surface */
/* i.e. dxpdx[1][2] = 0                                                          */
/* Schwarzschild static coords only for now                                      */
#if(OBLIQUE_NS)

double vpotns_flux(double r, double th1, double th2, double ph1, double ph2)
{
	double alpha = OBL_ANGLE_NS;
	double sinth1 = sin(th1);
	double sinth2 = sin(th2);
	double sinth1sq = sinth1 * sinth1;
	double sinth2sq = sinth2 * sinth2;
	double int_Ath_dth = -((th2 - th1) * (sin(ph2) - sin(ph1)) * sin(alpha));
	double int_Aph_dph = ((ph2 - ph1) * (sinth2sq - sinth1sq) * cos(alpha)
		- (sin(2 * th2) - sin(2 * th1)) * (sin(ph2) - sin(ph1)) * sin(alpha) * 0.5);


	double radFactor;
	double z = 2.0 / r;
	double zinv = 1.0 / z;

	double schwFactor = 0.5 + zinv + zinv * zinv * log(1.0 - z);


	radFactor = -schwFactor * 3.0 * MU_NS / 2.0 * pow(R_NS / 4.0, 3.0);



	return(radFactor * (-int_Ath_dth + int_Aph_dph));
	//return( (-int_Ath_dth + int_Aph_dph)/r );
}

double dfluxns(double r, double Omega, double phi, double th1, double th2, double t, double dt)
{
	double vpotns_flux(double r, double th1, double th2, double ph1, double ph2);
	double phi2 = phi - angleRotated(t);
	double phi1 = phi2 - Omega * dt;
	return(vpotns_flux(r, th1, th2, phi1, phi2));
}
#endif /* OBLIQUE_NS */
double calcRadialField(int n, int i, int j, int z, int loc, struct of_geom* geom, double t)
{
	double Br, B1_code;

	double r, theta, phi, phiRot, Br_ang;
	double zmetric, zinv, g_11_Schw;
	//double rdetg_IEF, rdetg_KS, rdetg_ratio ;
	double schwFactor, lapse;
	double dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM];
	double X[NDIM];

	//get_KS_metric(i, j, k, geom, CENT) ;

	coord(n, i, j, z, loc, X);
	bl_coord(X, &r, &theta, &phi);

	zmetric = 2.0 / r;
	zinv = 1.0 / zmetric;
	g_11_Schw = 1.0 / (1.0 - zmetric);
	lapse = sqrt(1.0 - zmetric); // Schw.; lapse function of metric actually in use: Schw, BL, ...

	//rdetg_IEF = sqrt(g_11_IEF) * r*r * sin(theta) ;  // These are sqrts of the spatial metric determinant
	//rdetg_KS  = geom.g / lapse ; // code gdet is sqrt(abs(g)), g = - alpha^2 gamma


	/* Magnetic fields as measured by the Schwarzschild normal observer (fido) */
	schwFactor = zinv * zinv * (zinv * log(1.0 - zmetric) + 1.0 + 0.5 * zmetric);
	if (fabs(OBL_ANGLE_NS) > 0.0)	phiRot = angleRotated(t);
	else phiRot = 0.0;
	Br_ang = cos(OBL_ANGLE_NS) * cos(theta) + sin(OBL_ANGLE_NS) * sin(theta) * cos(phi - phiRot);
	Br = -(6.0 * MU_NS * Br_ang / (r * r * r * sqrt(g_11_Schw))) * schwFactor * pow(R_NS / 4.0, 3.0);

	/* Modify so that the field has zero divergence in KS coordinates: *
	 * gives field as measured by the KS normal observer               */
	 //rdetg_ratio = rdetg_IEF / rdetg_KS ;
	 //Br *= rdetg_ratio;

	 /* Take account of factor of lapse,                                *
	  * since code uses B^i = *F^it = B^i_fido / alpha                  */
	Br *= 1.0 / lapse;

	/* transform to code coords                                         */
	/* dr^\mu/dx^\nu jacobian, where x^\nu are internal coords          */
	dxdxp_func(X, dxdxp);
	invert_matrix(dxdxp, dxpdx);

	B1_code = dxpdx[1][1] * Br;


	return B1_code;
}


static double omega_star(double t)
{
	double omega, omega0, t0, delta_t;
	t0 = SPINUP_START_TIME_NS;
	delta_t = SPINUP_TIME_NS;
	omega0 = OMEGA_NS;

	if (t < t0)
		omega = 0.0;
	else if (t > t0 + delta_t)
		omega = omega0;
	else
		omega = omega0 * (t - t0) / delta_t;
	//omega = 0.5 * (1.0 - cos((t-t0)*M_PI/delta_t)) * omega0 ;

	return omega;
}

/*** Find the contravariant components of the 4-velocity of the rotating stellar surface */
static void get_surface_4velocity(struct of_geom* geom, double* uscon, double t)
{
	double omega = omega_star(t);
	uscon[0] = 1.0 / sqrt(-(geom->gcov[0][0] + 2.0 * geom->gcov[0][3] * omega + geom->gcov[3][3] * omega * omega));
	uscon[1] = 0.0;
	uscon[2] = 0.0;
	uscon[3] = omega * uscon[0];

	return;
}

static void get_surface_magneticField(struct of_geom* geom, double* bncon, double* uscon, double* bscon)
{
	double lapse, us_dot_n, bn_dot_us, bncov[NDIM];
	int j;

	lower(bncon, geom, bncov);

	lapse = sqrt(-1.0 / geom->gcon[0][0]);
	us_dot_n = -lapse * uscon[0]; // Since normal observer n_mu = (-lapse, 0, 0, 0)
	bn_dot_us = dot(bncov, uscon);

	DLOOPA
		bscon[j] = -(bncon[j] + uscon[j] * bn_dot_us) / us_dot_n;

	return;
}


void bound_prim1_NS(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n, double t) {
	int i, j, z, k;
	double r, th, phi, X[NDIM];
	struct of_geom geom;
	int accreting, forcefree, useForcefreeBC;
	double ucon[NDIM], ucov[NDIM], bcon[NDIM], bcov[NDIM];
	double ucon_FAZ[NDIM], vcon_FAZ[NDIM]; // "FAZ" : first active zone
	double* pFAZ;
	double dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM];

	// inner r boundary condition: u, gdet extrapolation
	if (block[n][AMR_NBR4] == -1) {
		/*
		fprintf(stderr, "Before bounds.c \n");
		coord(n_ord[n], CELLS_IN_STAR, 500, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		double r_FAZ = r;
		for (i = 0; i < CELLS_IN_STAR + 1; i++) {
			coord(n_ord[n], i, 500, 0, CENT, X);
			bl_coord(X, &r, &th, &phi);
			//fprintf(stderr, "change: r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", i, r, 500, th, 0, phi);
			fprintf(stderr, "B1: %g, B1 expected:%g, at r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", prim[nl[n]][index_3D(n, i, 500, 0)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR, 500, 0)][B1] * pow(r_FAZ / r, 4.0), i, r, 500, th, 0, phi);
		}
		fprintf(stderr, "change \n");
		*/

#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
					//#pragma omp   simd
					for (i = -N1G; i < CELLS_IN_STAR; i++) {
						pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)];
					}
					get_geometry(n, CELLS_IN_STAR, j, z, CENT, &geom);
					pFAZ = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)];
					ucon_calc(prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)], &geom, ucon_FAZ);
					for (k = 1; k < NDIM; k++) {
						vcon_FAZ[k] = ucon_FAZ[k] / ucon_FAZ[0];
					}
					if (vcon_FAZ[1] < 0.0)
						accreting = 1;
					else
						accreting = 0;

					if (pFAZ[FLRFRAC] > FFE_ZONE_FLRFRAC_THRESHOLD)
						forcefree = 1;
					else
						forcefree = 0;

					if (forcefree || (accreting == 0))
						useForcefreeBC = 1;
					else
						useForcefreeBC = 0;


					if (useForcefreeBC)
					{
						//basic_hydroStatic_atm
						for (i = -N1G; i < 0; i++) {
							coord(n, i, 0, 0, CENT, X);
							bl_coord(X, &r, &th, &phi);
							//prim[nl[n]][index_3D(n, i, j, z)][RHO] = 1.9225588e-4 * pow(10.0, SURF_MAX_BSQ_RHO_LOG - 2.1) * pow(MU_NS / 10.0, 2.0) * pow(r / R_NS, -4.0);
							//prim[nl[n]][index_3D(n, i, j, z)][UU] = 1.9225588e-4 * pow(10.0, SURF_MAX_BSQ_UINT_LOG - 2.1) * pow(MU_NS / 10.0, 2.0) * pow(r / R_NS, -4.0);
							prim[nl[n]][index_3D(n, i, j, z)][RHO] = RHO0_HYDROSTAT_ATM_NS * pow(MU_NS / 10.0, 2.0) * pow(r / R_NS, -1.0 / (GAMMA - 1.0));
							prim[nl[n]][index_3D(n, i, j, z)][UU] = (RHO0_HYDROSTAT_ATM_NS / (GAMMA * R_NS)) * pow(MU_NS / 10.0, 2.0) * pow(r / R_NS, GAMMA / (1.0 - GAMMA));
						}
					}
					//ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] = calcRadialField(n, CELLS_IN_STAR, j, k, FACE1, &geom, t);
					//prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][PS1START] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1];
					coord(n, CELLS_IN_STAR, j, z, FACE1, X);
					bl_coord(X, &r, &th, &phi);
					double r_surf = r;
					dxdxp_func(X, dxdxp);
					double dxdxp_surf = dxdxp[1][1];
					get_geometry(n, CELLS_IN_STAR, j, z, FACE1, &geom);
					double gcon_surf = geom.gcon[0][0];
					coord(n, CELLS_IN_STAR, j, z, CENT, X);
					bl_coord(X, &r, &th, &phi);
					double r_FAZ = r;
					dxdxp_func(X, dxdxp);
					double dxdxp_FAZ = dxdxp[1][1];
					get_geometry(n, CELLS_IN_STAR, j, z, CENT, &geom);
					double gcon_FAZ = geom.gcon[0][0];
					for (int i = -N1G; i < 0; i++)
					{						
						coord(n, i, j, z, CENT, X);
						bl_coord(X, &r, &th, &phi);
						double r_cell = r;
						//if (j == 500 && z == 0) fprintf(stderr, "change: r(%d)=%g %g %g, th(%d)=%g, phi(%d)=%g \n", i, r, r_cell, r_FAZ, j, th, z, phi);
						//if (j == 500 && z == 0) fprintf(stderr, "chonge: r(%d)=%g %g %g, th(%d)=%g, phi(%d)=%g \n", i, r, r_cell, r_FAZ, j, th, z, phi);
						dxdxp_func(X, dxdxp);
						invert_matrix(dxdxp, dxpdx);
						double dxpdx11 = dxpdx[1][1];
						get_geometry(n, i, j, z, CENT, &geom);
#if OBLIQUE_NS
						prim[nl[n]][index_3D(n, i, j, z)][B1] = calcRadialField(n, i, j, k, CENT, &geom, t);
						ps[nl[n]][index_3D(n, i, j, z)][1] = calcRadialField(n, i, j, k, FACE1, &geom, t);
						prim[nl[n]][index_3D(n, i, j, z)][PS1START] = ps[nl[n]][index_3D(n, i, j, z)][1];

#else /* aligned rotator: can store normal field */ 

#if(1)
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] * pow(r_surf / r_cell, 3.0) * dxpdx[1][1] * dxdxp_surf;// * pow(geom.gcon[0][0] / gcon_surf, 3. / 4.);
						prim[nl[n]][index_3D(n, i, j, z)][PS1START] = ps[nl[n]][index_3D(n, i, j, z)][1];

#if(1)
						prim[nl[n]][index_3D(n, i, j, z)][B1] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] * pow(r_surf / r_cell, 3.0) * dxpdx11 * dxdxp_surf;// * pow(geom.gcon[0][0] / gcon_surf, 3. / 4.);
#else
						prim[nl[n]][index_3D(n, i, j, z)][B1] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * pow(r_FAZ / r_cell, 3.0) * dxpdx11 * dxdxp_FAZ *pow(geom.gcon[0][0] / gcon_FAZ, 3. / 4.);
#endif
						//if (j == 500 && z == 0) fprintf(stderr, "PS1: %g, PS1 expected:%g, at r(%d)=%g %g, th(%d)=%g, phi(%d)=%g \n", ps[nl[n]][index_3D(n, i, j, z)][1], ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] * pow(r_surf / r_cell, 4.0) * dxpdx[1][1] * dxdxp_surf * pow(geom.gcon[0][0] / gcon_surf, 3. / 4.), i, r, r_cell, j, th, z, phi);
						//if (j == 500 && z == 0) fprintf(stderr, "B1: %g, B1 expected:%g, at r(%d)=%g %g, th(%d)=%g, phi(%d)=%g \n", prim[nl[n]][index_3D(n, i, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * pow(r_FAZ / r_cell, 4.0), i, r, r_cell, j, th, z, phi);
#else
						double df_B1 = 0.0;
						df_B1 = prim[nl[n]][index_3D(n, 1 + CELLS_IN_STAR, j, z)][B1] - prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1];
						//df_B1 = ps[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][1] - ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1];
						//if ((ps[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][1] * (ps[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][1] - (double)(N1G + CELLS_IN_STAR) * df_B1)) < 0.0 || (ps[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][1] * (ps[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][1] - df_B1) < 0.0)) df_B1 = 0.0;
						//df_B1 = slope_lim_BC(prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 2, j, z)][B1]);
						if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(N1G + CELLS_IN_STAR) * df_B1)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - df_B1) < 0.0)) df_B1 = 0.0;
						prim[nl[n]][index_3D(n, i, j, z)][B1] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(CELLS_IN_STAR - i) * df_B1;
						ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] - (double)(CELLS_IN_STAR - i) * df_B1;
						prim[nl[n]][index_3D(n, i, j, z)][PS1START] = ps[nl[n]][index_3D(n, i, j, z)][1];
#endif
						
						//coord(n, i, j, z, FACE1, X);
						//bl_coord(X, &r, &th, &phi);
						//double r_cell = r;
						//if (j == 500 && z == 0) fprintf(stderr, "change: r(%d)=%g %g, th(%d)=%g, phi(%d)=%g \n", i, r, r_cell, j, th, z, phi);
						//get_geometry(n, i, j, z, FACE1, &geom);
						//if (j == 500 && z == 0) fprintf(stderr, "chonge: r(%d)=%g %g, th(%d)=%g, phi(%d)=%g \n", i, r, r_cell, j, th, z, phi);
						//dxdxp_func(X, dxdxp);
						//invert_matrix(dxdxp, dxpdx);
#endif					
						//if(j ==500 && z ==0) fprintf(stderr, "B1: %g, B1 expected:%g, at r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", prim[nl[n]][index_3D(n, i, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * pow(r_FAZ / r, 4.0), i, r, j, th, z, phi);
#if(SIMPLE_NS_BC_EXTRAPOLATE)
						//pv[B1 * (ksize)+(N1G + CELLS_IN_STAR) * isize + global_id] = pv[PS1START * (ksize)+(N1G + CELLS_IN_STAR) * isize + global_id] * scaleFACE[(N1G + CELLS_IN_STAR) * isize + global_id];
						double df_B1 = 0.0, df_B2 = 0.0, df_B3 = 0.0;
						//df_B1 = prim[nl[n]][index_3D(n, 1 + CELLS_IN_STAR, j, z)][B1] - prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1];
						df_B2 = prim[nl[n]][index_3D(n, 1 + CELLS_IN_STAR, j, z)][B2] - prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2];
						df_B3 = prim[nl[n]][index_3D(n, 1 + CELLS_IN_STAR, j, z)][B3] - prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3];

						//if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(N1G + CELLS_IN_STAR) * df_B1)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - df_B1) < 0.0)) df_B1 = 0.0;
						if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - (double)(N1G + CELLS_IN_STAR) * df_B2)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - df_B2) < 0.0)) df_B2 = 0.0;
						if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - (double)(N1G + CELLS_IN_STAR) * df_B3)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - df_B3) < 0.0)) df_B3 = 0.0;
						//prim[nl[n]][index_3D(n, i, j, z)][B1] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(CELLS_IN_STAR - i) * df_B1;
						prim[nl[n]][index_3D(n, i, j, z)][B2] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - (double)(CELLS_IN_STAR - i) * df_B2;
						prim[nl[n]][index_3D(n, i, j, z)][B3] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - (double)(CELLS_IN_STAR - i) * df_B3;

#if(STAGGERED)
						//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] - (double)(CELLS_IN_STAR - i) * df_B1;
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][2] - (double)(CELLS_IN_STAR - i) * df_B2;
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][3] - (double)(CELLS_IN_STAR - i) * df_B3;
#endif

#elif(SLOPELIM_NS_BC_EXTRAPOLATE)
						//pv[B1 * (ksize)+(N1G + CELLS_IN_STAR) * isize + global_id] = pv[PS1START * (ksize)+(N1G + CELLS_IN_STAR) * isize + global_id] * scaleFACE[(N1G + CELLS_IN_STAR) * isize + global_id];
						double df_B1 = 0.0, df_B2 = 0.0, df_B3 = 0.0;
						//df_B1 = slope_lim_BC(prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 2, j, z)][B1]);
						df_B2 = slope_lim_BC(prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][B2], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 2, j, z)][B2]);
						df_B3 = slope_lim_BC(prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 1, j, z)][B3], prim[nl[n]][index_3D(n, CELLS_IN_STAR + 2, j, z)][B3]);

						//if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(N1G + CELLS_IN_STAR) * df_B1)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - df_B1) < 0.0)) df_B1 = 0.0;
						if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - (double)(N1G + CELLS_IN_STAR) * df_B2)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - df_B2) < 0.0)) df_B2 = 0.0;
						if ((prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - (double)(N1G + CELLS_IN_STAR) * df_B3)) < 0.0 || (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] * (prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - df_B3) < 0.0)) df_B3 = 0.0;

						//prim[nl[n]][index_3D(n, i, j, z)][B1] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B1] - (double)(CELLS_IN_STAR - i) * df_B1;
						prim[nl[n]][index_3D(n, i, j, z)][B2] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] - (double)(CELLS_IN_STAR - i) * df_B2;
						prim[nl[n]][index_3D(n, i, j, z)][B3] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] - (double)(CELLS_IN_STAR - i) * df_B3;

#if(STAGGERED)
						//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][1] - (double)(CELLS_IN_STAR - i) * df_B1;
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][2] - (double)(CELLS_IN_STAR - i) * df_B2;
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][3] - (double)(CELLS_IN_STAR - i) * df_B3;
#endif	
#else			
#if(STAGGERED)
						coord(n, CELLS_IN_STAR, j, z, CENT, X);
						bl_coord(X, &r, &th, &phi);
						r_FAZ = r;
						get_geometry(n, CELLS_IN_STAR, j, z, CENT, &geom);
						gcon_FAZ = geom.gcon[0][0];
						coord(n, CELLS_IN_STAR, j, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						double r_FACE2 = r;
						get_geometry(n, CELLS_IN_STAR, j, z, FACE2, &geom);
						double gcon_FACE2 = geom.gcon[0][0];
						coord(n, i, j, z, FACE2, X);
						bl_coord(X, &r, &th, &phi);
						r_cell = r;
						get_geometry(n, i, j, z, FACE2, &geom);
						ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][2] * pow(r_FACE2 / r_cell, 4.0);// * pow(geom.gcon[0][0] / gcon_FACE2, 3. / 2.);

						coord(n, CELLS_IN_STAR, j, z, FACE3, X);
						bl_coord(X, &r, &th, &phi);
						double r_FACE3 = r;
						coord(n_ord[n], i, j, z, FACE3, X);
						bl_coord(X, &r, &th, &phi);
						r_cell = r;
						ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][3] * pow(r_FACE3 / r_cell, 2.0);

#endif
						coord(n, i, j, z, CENT, X);
						bl_coord(X, &r, &th, &phi);
						r_cell = r;
						get_geometry(n, i, j, z, CENT, &geom);
						prim[nl[n]][index_3D(n, i, j, z)][B2] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B2] * pow(r_FAZ / r_cell, 4.0);// * pow(geom.gcon[0][0] / gcon_FAZ, 3. / 2.);
						prim[nl[n]][index_3D(n, i, j, z)][B3] = prim[nl[n]][index_3D(n, CELLS_IN_STAR, j, z)][B3] * pow(r_FAZ / r_cell, 2.0);
#endif

						//if (i < 5 && z == 0 && (j<5) && r < 4.2) {
						//	fprintf(stderr, "ps1: %g at r=%g(%d), th=%g(%d), phi=%g(%d) \n", ps[nl[n]][index_3D(n, i, j, z)][1],r, i, th, j, phi, z);
						//}

						if (useForcefreeBC) {
							prim[nl[n]][index_3D(n, i, j, z)][FLR] = 1.0;
							prim[nl[n]][index_3D(n, i, j, z)][FLRFRAC] = 1.0;
							prim[nl[n]][index_3D(n, i, j, z)][KTOT] = 0.0;
						}
						else {
							prim[nl[n]][index_3D(n, i, j, z)][FLR] = prim[nl[n]][index_3D(n, 0, j, z)][FLR];
							prim[nl[n]][index_3D(n, i, j, z)][FLRFRAC] = prim[nl[n]][index_3D(n, 0, j, z)][FLRFRAC];
							prim[nl[n]][index_3D(n, i, j, z)][KTOT] = prim[nl[n]][index_3D(n, 0, j, z)][KTOT];
						}

					}


					/* Now do velocities */

					if (useForcefreeBC)
					{
						double bncon[NDIM], bscon[NDIM], uscon[NDIM], etacon[NDIM], etacov[NDIM];
						double bccon[NDIM], bccov[NDIM], uperpcon[NDIM], uperpcov[NDIM];
						double bs_dot_eta, us_dot_eta, bcsq, bc_dot_us, uperpsq;
						/*set_boundary_velocities_FFE_4Dmethod(n, i, j, k, CENT, prim[nl[n]][index_3D(n, i, j, z)]);*/
						for (i = -N1G; i < 0 + CELLS_IN_STAR; i++) {
							get_geometry(n, i, j, z, CENT, &geom);
							bncon[0] = 0.0;
							for (k = 1; k < NDIM; k++) {
								bncon[k] = prim[nl[n]][index_3D(n, i, j, z)][B1 + k - 1] / sqrt(-geom.gcon[0][0]);
							}
							/* Surface-observer 4-velocity and magnetic field */
							get_surface_4velocity(&geom, uscon, t);
							get_surface_magneticField(&geom, bncon, uscon, bscon);

							/* Coordinate-observer 4-velocity and magnetic field */
							etacon[0] = sqrt(-1.0 / geom.gcov[0][0]); // i.e. eta = u_c
							for (k = 1; k < NDIM; k++) {
								etacon[k] = 0.0;
							}
							lower(etacon, &geom, etacov);

							bs_dot_eta = dot(bscon, etacov);
							us_dot_eta = dot(uscon, etacov);

							for (k = 0; k < NDIM; k++) {
								bccon[k] = uscon[k] * bs_dot_eta - bscon[k] * us_dot_eta;
							}
							lower(bccon, &geom, bccov);
							bcsq = dot(bccon, bccov) + SMALL;

							/* Project surface velocity us orthogonal to coordinate-observer magnetic field bc */
							bc_dot_us = dot(bccov, uscon);
							for (k = 0; k < NDIM; k++) {
								uperpcon[k] = uscon[k] - bccon[k] * bc_dot_us / bcsq;
							}

							/* Normalize: u = u_p / sqrt(- u_p^2) */
							lower(uperpcon, &geom, uperpcov);
							uperpsq = dot(uperpcon, uperpcov) + SMALL;
							for (k = 0; k < NDIM; k++) {
								ucon[k] = uperpcon[k] / sqrt(-uperpsq);
							}

							/* Just use surface 4-velocity directly */
							//DLOOPA
							//    ucon[j] = uscon[j] ;

							for (k = 1; k < NDIM; k++) {
								prim[nl[n]][index_3D(n, i, j, z)][U1 + k - 1] = ucon[k] - geom.gcon[0][k] * ucon[0] / geom.gcon[0][0];
							}
						}


					}
					else
					{
						double bncon[NDIM], bscon[NDIM], bscov[NDIM], uscon[NDIM], bsmag;
						double uprllcon[NDIM], uprllsq;
						double udotb[3], d_udotb, udotb_ghost[N1G + 0], beta_NS;
						//find_udotb_first3(n, j, k, prim, udotb);
						for (i = 0; i < 3; i++) {
							get_geometry(n, i + 0, j, z, CENT, &geom);
							ucon_calc(prim[nl[n]][index_3D(n, i + 0, j, z)], &geom, ucon);
							/* Normal-observer magnetic field */
							bncon[0] = 0.0;
							for (k = 1; k < NDIM; k++) {
								bncon[k] = prim[nl[n]][index_3D(n, i + 0, j, z)][B1 + k - 1] / sqrt(-geom.gcon[0][0]);
							}
							get_surface_4velocity(&geom, uscon, t);
							get_surface_magneticField(&geom, bncon, uscon, bscon);
							lower(bscon, &geom, bscov);
							bsmag = sqrt(dot(bscon, bscov));
							udotb[i] = dot(ucon, bscov) / bsmag;  // Store u.b/|b|
						}
						d_udotb = slope_lim_BC(udotb[0], udotb[1], udotb[2]);
						for (i = -N1G; i < 0 + 0; i++)
							udotb_ghost[i + N1G] = udotb[0] - (0 - i) * d_udotb;
						//udotb_surface[j][k] = udotb[0] - 0.5 * d_udotb;


						for (i = -N1G; i < 0 + 0; i++) {
							// set_boundary_velocities_surfaceFrame_4Dmethod(n, i, j, k, CENT, prim[i][j][k], udotb_ghost[i + N1G]);
							get_geometry(n, i, j, z, CENT, &geom);
							bncon[0] = 0.0;
							for (k = 1; k < NDIM; k++) {
								bncon[k] = prim[nl[n]][index_3D(n, i, j, z)][B1 + k - 1] / sqrt(-geom.gcon[0][0]);
							}
							/* Surface-observer 4-velocity and magnetic field */
							get_surface_4velocity(&geom, uscon, t);
							get_surface_magneticField(&geom, bncon, uscon, bscon);
							lower(bscon, &geom, bscov);
							bsmag = sqrt(dot(bscon, bscov));
							beta_NS = udotb_ghost[i + N1G] / sqrt(1.0 + udotb_ghost[i + N1G] * udotb_ghost[i + N1G]);
							for (k = 0; k < NDIM; k++) {
								ucon[k] = (uscon[k] + beta_NS * bscon[k] / bsmag) / sqrt(1.0 - beta_NS * beta_NS);
							}

							for (k = 1; k < NDIM; k++) {
								prim[nl[n]][index_3D(n, i, j, z)][U1 + k - 1] = ucon[k] - geom.gcon[0][k] * ucon[0] / geom.gcon[0][0];
							}
						}
					}
				}
			}
			/*
			fprintf(stderr, "After bounds.c \n");
			coord(n_ord[n], CELLS_IN_STAR, 500, 0, CENT, X);
			bl_coord(X, &r, &th, &phi);
			double r_FAZ = r;
			coord(n_ord[n], CELLS_IN_STAR, 500, 0, FACE1, X);
			bl_coord(X, &r, &th, &phi);
			double r_surf = r;
			for (i = 0; i < CELLS_IN_STAR + 1; i++) {
				coord(n_ord[n], i, 500, 0, FACE1, X);
				bl_coord(X, &r, &th, &phi);
				fprintf(stderr, "ps1: %g, ps1 expected:%g, at r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", ps[nl[n]][index_3D(n, i, 500, 0)][1], ps[nl[n]][index_3D(n, CELLS_IN_STAR, 500, 0)][1] * pow(r_surf / r, 4.0), i, r, 500, th, 0, phi);
				coord(n_ord[n], i, 500, 0, CENT, X);
				bl_coord(X, &r, &th, &phi);
				//fprintf(stderr, "change: r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", i, r, 500, th, 0, phi);
				fprintf(stderr, "B1: %g, B1 expected:%g, at r(%d)=%g, th(%d)=%g, phi(%d)=%g \n", prim[nl[n]][index_3D(n, i, 500, 0)][B1], prim[nl[n]][index_3D(n, CELLS_IN_STAR, 500, 0)][B1] * pow(r_FAZ / r, 4.0), i, r, 500, th, 0, phi);

			}
			*/





		}

#if(!CONSTANT_BC)
		if (block[n][AMR_NBR2] == -1) {
			// outer r BC: outflow 		
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
			{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
					for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++) {
						for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G; i++) {
							PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][k];
							pflag[nl[n]][index_3D(n, i, j, z)] = pflag[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)];
#if(STAGGERED)
							ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][2];
							ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) - 1, j, z)][3];
#endif
						}
					}
				}
			}
		}
#endif

		// make sure there is no inflow at the inner boundary 
		if (block[n][AMR_NBR4] == -1) {
			for (i = -N1G; i <= -1; i++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
				{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
					for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
						for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
							inflow_check(prim[nl[n]][index_3D(n, -1, j, z)], n, i, j, z, 0, 1);
							inflow_check(prim[nl[n]][index_3D(n, -2, j, z)], n, i, j, z, 0, 1);
#if(N1G==3)
							inflow_check(prim[nl[n]][index_3D(n, -3, j, z)], n, i, j, z, 0, 1);
#endif
						}
					}
				}
			}
		}

		// make sure there is no inflow at the outer boundary
#if(!CONSTANT_BC)
		if (block[n][AMR_NBR2] == -1) {
			for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i <= N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G - 1; i++) {
#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
				{
#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
					for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++) {
						for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
							inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]), j, z)], n, i, j, z, 1, 1);
							inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 1, j, z)], n, i, j, z, 1, 1);
#if(N1G==3)
							inflow_check(prim[nl[n]][index_3D(n, N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + 2, j, z)], n, i, j, z, 1, 1);
#endif
						}
					}
				}
			}
		}
#endif
	}
}






