#include "include.h"
#include "decs_MPI.h"
void bound_prim1_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_reflective(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim3_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);
void bound_prim2_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n);

/* bound array containing entire set of primitive variables */
void bound_prim(double(*restrict prim[NB_LOCAL])[NPR], int bound_force)
{
	int i, n, flag;
	double temp=nstep;
	if (bound_force == 1) nstep = -1;

	#if(BOUND_TYPE1==OUTFLOW)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim1_outflow(p,ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim1_outflow(ph, psh, n_ord[n]);
	}
	#endif

	#if(BOUND_TYPE2==OUTFLOW)
	for (n = 0; n < n_active; n++) {
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2_outflow(p, ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2_outflow(ph, psh, n_ord[n]);
	}
	#elif(BOUND_TYPE2==REFLECTIVE)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_prim2_reflective(p,ps, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_prim2_reflective(ph,psh, n_ord[n]);
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
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do {
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	if (N3 > 1){
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		set_iprobe(0, &flag);
		do {
			for (n = 0; n < n_active; n++){
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, bound_force, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, bound_force, n_ord[n]);
			}
			set_iprobe(1, &flag);
		} while (flag);
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");

	#if(BOUND_TYPE2==TRANSMISSIVE && NB_3==1)
	for (n = 0; n < n_active; n++){
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

	#if (STAGGERED && COPY_BFIELD)
	rc = 0;
	if (nstep % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1 || bound_force == 1){ //watch out does this for both half and full timestep while only needed for full timestep
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send2(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_rec2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1)B_send3(ps, Bufferps_1, n_ord[n]);
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || bound_force == 1) B_rec3(ps, Bufferps_1, n_ord[n]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB \n");
	#endif

	if (bound_force == 1) nstep = temp;
}

void set_iprobe(int mode, int * flag){
	#if(TASK_BASED)
	int i ,n;
	*flag = 0;
	for (n = 0; n < n_active; n++){
		if (mode == 0){
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++) block[n_ord[n]][i] = 0;
		}
		else{
			for (i = AMR_IPROBE1; i <= AMR_IPROBE6_4; i++){
				if (block[n_ord[n]][i] == -1){
					block[n_ord[n]][i] = 0;
					*flag = 1;
				}
				else  block[n_ord[n]][i] = 1;
			}
		}
	}
	#else
	*flag = 0;
	#endif
	return;
}

void bound_prim1_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR4] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n]-N2G; j < N2_GPU_offset[n] + BS_2+N2G; j++){
				for (z = N3_GPU_offset[n]-N3G; z < N3_GPU_offset[n] + BS_3+N3G; z++){
					//#pragma omp   simd
					for (i = -N1G; i < 0; i++){
						for (k = 0; k < NPR; k++){
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
	#endif

	#if(!CONSTANT_BC)
	if (block[n][AMR_NBR2] == -1){
		// outer r BC: outflow 		
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
			for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++){
					for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i < N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G; i++){
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
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR4] == -1){
		for (i = -N1G; i <= -1; i++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
				#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
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
	#endif

	// make sure there is no inflow at the outer boundary
	#if(!CONSTANT_BC)
	if (block[n][AMR_NBR2] == -1){
		for (i = N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]); i <= N1 * pow(1 + REF_1, block[n][AMR_LEVEL1]) + N1G - 1; i++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, i) private(j,z)
			{
				#pragma omp for collapse(2) schedule(static, (BS_2+2*N2G)*(BS_3+2*N3G)/nthreads)	
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
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

void bound_prim2_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n]-N1G; i < N1_GPU_offset[n] + BS_1+N1G; i++){
				for (z = N3_GPU_offset[n]-N3G; z < N3_GPU_offset[n] + BS_3+N3G; z++){
					//#pragma omp   simd
					for (j = -N2G; j < 0; j++){
						for (k = 0; k < NPR; k++){
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
	#endif

	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR3] == -1){
		// outer r BC: outflow 		
		#pragma omp parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (z = N3_GPU_offset[n] - N3G; z < N3_GPU_offset[n] + BS_3 + N3G; z++){
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++){
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
	#endif

	// make sure there is no inflow at the inner boundary 
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR1] == -1){
		for (j = -N2G; j <= -1; j++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, j) private(i,z)
			{
				#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
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
	#endif

	// make sure there is no inflow at the outer boundary
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR3] == -1){
		for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j <= N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G - 1; j++){
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
	#endif
}

void bound_prim2_reflective(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k, jref;

	//copy all densities and B^phi in; interpolate linearly transverse velocity
	#if(POLEFIX && POLEFIX < N2/2)
	jref = POLEFIX;
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
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
	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, jref,gdet) private(i,j,z,k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
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
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					PLOOP{
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

	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					//#pragma omp   simd
					PLOOP{
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
	if (block[n][AMR_NBR1] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
						#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.;
						#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
	if (block[n][AMR_NBR3] == -1){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim) private(i,j,z)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++) {
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.;
						#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.;
						#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.;
					}
				}
			}
		}
	}
}

void bound_prim2_trans(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;

	// polar BCs 
	if (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag,gdet) private(i,j,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = -N2G; j < 0; j++){
						//#pragma omp   simd
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, -j - 1, (z + BS_3 / 2) % BS_3)][k];
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
						#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] *= -1.0;
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

	if (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3){
		#pragma omp   parallel shared(block,n,n_ord,n_active,prim, pflag, gdet) private(i,z, k)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_3+2*N3G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
				for (z = -N3G + N3_GPU_offset[n]; z < BS_3 + N3_GPU_offset[n] + N3G; z++) {
					for (j = N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]); j < N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) + N2G; j++){
						//#pragma omp   simd
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, 2 * N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]) - j - 1 , (z + BS_3 / 2) % BS_3)][k];
						prim[nl[n]][index_3D(n, i, j, z)][U2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3] *= -1.0;
						#if(RAD_M1)
						prim[nl[n]][index_3D(n, i, j, z)][U2_RAD] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][U3_RAD] *= -1.0;
						#endif
						prim[nl[n]][index_3D(n, i, j, z)][B2] *= -1.0;
						prim[nl[n]][index_3D(n, i, j, z)][B3] *= -1.0;
						#
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

void bound_prim3_outflow(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int n){
	int i, j, z, k;
	struct of_geom geom;

	// inner r boundary condition: u, gdet extrapolation
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR6] == -1){
		#pragma omp   parallel shared(n,n_ord,n_active,prim, pflag,gdet) private(i,j,z,k,geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
			for (i = N1_GPU_offset[n]-N1G; i < N1_GPU_offset[n] + BS_1+N1G; i++){
				for (j = N2_GPU_offset[n]-N2G; j < N2_GPU_offset[n] + BS_2+N2G; j++){
					//#pragma omp   simd
					for (z = -N3G; z < 0; z++){
						for (k = 0; k < NPR; k++){
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
	#endif

	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR5] == -1){
		// outer r BC: outflow 		
		#pragma omp parallel shared(block,n,n_ord,n_active,prim, pflag) private(i,j,k,z, geom)
		{
			#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
			for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++) {
				for (j = N2_GPU_offset[n] - N2G; j < N2_GPU_offset[n] + BS_2 + N2G; j++){
					for (z = N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]); z < N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + N3G; z++){
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
	#endif

	// make sure there is no inflow at the inner boundary 
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR6] == -1){
		for (z = -N3G; z <= -1; z++){
			#pragma omp   parallel shared(block,n,n_ord,n_active,prim, z) private(j,i)
			{
				#pragma omp for collapse(2) schedule(static, (BS_1+2*N1G)*(BS_2+2*N2G)/nthreads)	
				for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
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
	#endif

	// make sure there is no inflow at the outer boundary
	#if(!(CONSTANT_BC && (CARTESIAN || CARTESIAN_GR)))
	if (block[n][AMR_NBR5] == -1){
		for (z = N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]); z <= N3 * pow(1 + REF_3, block[n][AMR_LEVEL3]) + N3G - 1; z++){
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
	#endif
}

//Set Cartesian boundary conditions
void bound_prim_cart(double(*restrict prim[NB_LOCAL])[NPR], double(*restrict ps[NB_LOCAL])[NDIM], int dir, int n){
	int i, j, z, k;
	struct of_geom geom;

	ZSLOOP3D(N1_GPU_offset[n] - N1G, BS_1 + N1_GPU_offset[n] + N1G - 1, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 + N2G - 1, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 + N3G - 1) {
		if (pflag_cart[nl[n]][index_3D(n, i, j, z)] == 1) {
			if (dir == 1) {
				int itest, i_add;
				for (i_add = D1; i_add <= N1G; i_add++) {
					itest = MY_MIN(i + i_add, N1_GPU_offset[n] + BS_1 + N1G - 1);
					if (pflag_cart[nl[n]][index_3D(n, itest, j, z)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, itest, j, z)][k];
						if (pflag_cart[nl[n]][index_3D(n, i, j - D2 * ((j - D2) >= 0), z)] == 1) { //B2
							//ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, itest, j, z)][2];
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= 0))] == 1) { //B3
							//ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, itest, j, z)][3];
						}
						break;
					}

					itest = MY_MAX(i - i_add, N1_GPU_offset[n] - N1G);
					if (pflag_cart[nl[n]][index_3D(n, itest, j, z)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, itest, j, z)][k];
						if (pflag_cart[nl[n]][index_3D(n, i, j - D2 * ((j - D2) >= 0), z)] == 1) { //B2
							//ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, itest, j, z)][2];
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= 0))] == 1) { //B3
							//ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, itest, j, z)][3];
						}
						break;
					}
				}
			}
			else if (dir == 2) {
				int jtest, j_add;
				for (j_add = D2; j_add <= N2G; j_add++) {
					jtest = MY_MIN(j + j_add, N2_GPU_offset[n] + BS_2 + N2G - 1);
					if (pflag_cart[nl[n]][index_3D(n, i, jtest, z)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, jtest, z)][k];
						if (pflag_cart[nl[n]][index_3D(n, i - D1 * ((i - D1) >= 0), j, z)] == 1) { //B1
							//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, jtest, z)][1];
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= 0))] == 1) { //B3
							//ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i, jtest, z)][3];
						}
						break;
					}

					jtest = MY_MAX(j - j_add, N2_GPU_offset[n] - N2G);
					if (pflag_cart[nl[n]][index_3D(n, i, jtest, z)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, jtest, z)][k];
						if (pflag_cart[nl[n]][index_3D(n, i - D1 * ((i - D1) >= 0), j, z)] == 1) { //B1
							//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, jtest, z)][1];
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j, z - D3 * ((z - D3) >= 0))] == 1) { //B3
							//ps[nl[n]][index_3D(n, i, j, z)][3] = ps[nl[n]][index_3D(n, i, jtest, z)][3];
						}
						break;
					}
				}
			}
			else if (dir == 3) {
				int ztest, z_add;
				for (z_add = D3; z_add <= N3G; z_add++) {
					ztest = MY_MIN(z + z_add, N3_GPU_offset[n] + BS_3 + N3G - 1);
					if (pflag_cart[nl[n]][index_3D(n, i, j, ztest)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, j, ztest)][k];
						if (pflag_cart[nl[n]][index_3D(n, i - D1 * ((i - D1) >= 0), j, z)] == 1) { //B1
							//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, j, ztest)][1];
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j - D2 * ((i - D2) >= 0), z)] == 1) { //B2
							//ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i, j, ztest)][2];
						}
						break;
					}

					ztest = MY_MAX(z - z_add, N3_GPU_offset[n] - N3G);
					if (pflag_cart[nl[n]][index_3D(n, i, j, ztest)] == 0) {
						PLOOP prim[nl[n]][index_3D(n, i, j, z)][k] = prim[nl[n]][index_3D(n, i, j, ztest)][k];
						if (pflag_cart[nl[n]][index_3D(n, i - D1 * ((i - D1) >= 0), j, z)] == 1) { //B1
							//ps[nl[n]][index_3D(n, i, j, z)][1] = ps[nl[n]][index_3D(n, i, j, ztest)][1]; 
						}
						if (pflag_cart[nl[n]][index_3D(n, i, j - D2 * ((i - D2) >= 0), z)] == 1) { //B2
							//ps[nl[n]][index_3D(n, i, j, z)][2] = ps[nl[n]][index_3D(n, i, j, ztest)][2];
						}
						break;
					}
				}
			}
		}
	}
}

void inflow_check(double * restrict pr, int n, int ii, int jj, int zz, int type, int dir){
    struct of_geom geom ;
    double ucon[NDIM];
    int j,k ;
    double alpha,beta1,gamma,vsq ;

    get_geometry(n, ii,jj,zz,CENT,&geom) ;
    ucon_calc(pr, &geom, ucon) ;

    if( ((ucon[dir] > 0.) && (type==0)) || ((ucon[dir] < 0.) && (type==1)) ) { 
		/* find gamma and remove it from primitives */
		if( gamma_calc(pr,&geom,&gamma) ) { 
			fprintf(stderr,"\ninflow_check(): gamma failure \n");
			fail(FAIL_GAMMA);
		}
		pr[U1] /= gamma ;
		pr[U2] /= gamma ;
		pr[U3] /= gamma ;
		alpha = 1./sqrt(-geom.gcon[0][0]) ;
		beta1 = geom.gcon[0][dir]*alpha*alpha ;

		/* reset radial velocity so radial 4-velocity is zero */
		pr[UU+dir] = beta1/alpha ;

		/* now find new gamma and put it back in */
		vsq = 0. ;
		SLOOP vsq += geom.gcov[j][k]*pr[U1+j-1]*pr[U1+k-1] ;
		if( fabs(vsq) < 1.e-13 )  vsq = 1.e-13;
		if( vsq >= 1. ) { 
			vsq = 1. - 1./(GAMMAMAX*GAMMAMAX) ;
		}
		gamma = 1./sqrt(1. - vsq) ;
		pr[U1] *= gamma ;
		pr[U2] *= gamma ;
		pr[U3] *= gamma ;

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
		pr[UU_RAD+dir] = beta1 / alpha;

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
}

