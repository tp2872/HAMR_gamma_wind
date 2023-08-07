#include "decsCUDA.h"
#include "include.h"
extern "C" {
#include "decs.h"
}

#include "GPU_mem.h"

void GPU_init(void)
{
	int ranks_per_node;

	//Do some checks first
	if (N_GPU>numdevices){
		fprintf(stderr, "N_GPU is bigger than the number of devices! \n");
		exit(0);
	}
	size_t mem_int, mem_tot;
	gpuMemGetInfo(&mem_int, &mem_tot);
	gpu_mem = mem_tot / (1.e9);

	//Enable peer access
	ranks_per_node = numdevices / N_GPU;
	gpu_offset = (rank % (ranks_per_node))*N_GPU;
	#if(N_GPU>1)
	int i, j;
	for (i = gpu_offset; i < gpu_offset + N_GPU; i++){
		gpuSetDevice(i);
		for (j = gpu_offset; j < gpu_offset + N_GPU; j++){
			if (i!=j) gpuDeviceEnablePeerAccess(j, 0);
		}
		//#if (DOHELM)
		//eos_init_GPU(i + rank / ranks_per_node);
		//#endif
		//#if(NEUTRINOS_M1)
		//nulib_init_GPU(i + rank / ranks_per_node);
		//#endif
	}
	#endif

	status = gpuGetLastError();
	if (gpuSuccess != status){
		fprintf(stderr, "Error in setting peeraccess: %d \n", status);
		exit(0);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	//gpuDeviceSetCacheConfig(gpuFuncCachePreferL1);
	//gpuDeviceSetSharedMemConfig(gpuSharedMemBankSizeEightByte);

	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting cache: %d \n", status);
}

#if(NEUTRINOS_M1)
void nulib_init_GPU(int n) {
	int i, j, k, l;
	#if(N_GPU>1)
	cudaSetDevice(block[n][AMR_GPU]);
	#endif

	// Danat: 3 is for emiss, kappa_abs, kappa_es
	gpuMalloc((void**)&GPU_nulib_table[0], (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES * NULIB_VARS) * sizeof(double));
	gpuMallocHost((void**)&nulib_table[0], (NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES * NULIB_VARS) * sizeof(double));

	// Check for errors: Nulib array allocation
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting Nulib tables: %d \n", status);

	// fill in the host array
	for (i = 0; i < NULIB_RHO; i++) for (j = 0; j < NULIB_TEMP; j++) for (k = 0; k < NULIB_YE; k++) for (l = 0; l < NU_SPECIES; l++) {
		nulib_table[0][(0 * (NULIB_RHO * NULIB_TEMP * NULIB_YE) + i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l] = nu_kappa_emiss[(i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l];
		nulib_table[0][(1 * (NULIB_RHO * NULIB_TEMP * NULIB_YE) + i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l] = nu_kappa_abs[(i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l];
		nulib_table[0][(2 * (NULIB_RHO * NULIB_TEMP * NULIB_YE) + i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l] = nu_kappa_scatt[(i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l];
		nulib_table[0][(3 * (NULIB_RHO * NULIB_TEMP * NULIB_YE) + i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l] = nu_kappa_emiss_N[(i * NULIB_TEMP * NULIB_YE + j * NULIB_YE + k) * NU_SPECIES + l];
	}
	gpuMemcpy(GPU_nulib_table[0], nulib_table[0], ((NULIB_RHO * NULIB_TEMP * NULIB_YE * NU_SPECIES * NULIB_VARS) * sizeof(double)), gpuMemcpyHostToDevice);
	gpuDeviceSynchronize();
}
#endif

double check = 1.0;

#if(DOHELM)
void eos_init_GPU(int n) {
	int i, j;
	int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	// Setting EOS arrays: the dumbest way - to copy EOS table for each block individually
	gpuMallocHost((void**)&eos_table[n], (EOSIMAX * EOSJMAX + eos_offset) * 21 * sizeof(double)); // should I add? OFFSET = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX * 21) % LOCAL_WORK_SIZE
	gpuMalloc((void**)&GPU_eos_table[n], (EOSIMAX * EOSJMAX + eos_offset) * 21 * sizeof(double)); // same here regarding the OFFSET
	// Check for errors: EOS array allocation
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting EOS: %d \n", status);

	// Write EOS table for each block individually
	// fill in the host array
	for (i = 0; i < EOSIMAX; i++) for (j = 0; j < EOSJMAX; j++) {
		// helmholtz free energy table (total: 9 items)
		eos_table[n][0 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_f[i * EOSJMAX + j];
		eos_table[n][1 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fd[i * EOSJMAX + j];
		eos_table[n][2 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ft[i * EOSJMAX + j];
		eos_table[n][3 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdd[i * EOSJMAX + j];
		eos_table[n][4 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ftt[i * EOSJMAX + j];
		eos_table[n][5 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdt[i * EOSJMAX + j];
		eos_table[n][6 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fddt[i * EOSJMAX + j];
		eos_table[n][7 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdtt[i * EOSJMAX + j];
		eos_table[n][8 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fddtt[i * EOSJMAX + j];

		// pressure derivative with density table (total: 4 items)
		eos_table[n][9 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdf[i * EOSJMAX + j];
		eos_table[n][10 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdfd[i * EOSJMAX + j];
		eos_table[n][11 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdft[i * EOSJMAX + j];
		eos_table[n][12 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdfdt[i * EOSJMAX + j];

		// electron chemical potential table (total: 4 items)
		eos_table[n][13 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ef[i * EOSJMAX + j];
		eos_table[n][14 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_efd[i * EOSJMAX + j];
		eos_table[n][15 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_eft[i * EOSJMAX + j];
		eos_table[n][16 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_efdt[i * EOSJMAX + j];

		// number denisty table (total: 4 items)
		eos_table[n][17 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xf[i * EOSJMAX + j];
		eos_table[n][18 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xfd[i * EOSJMAX + j];
		eos_table[n][19 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xft[i * EOSJMAX + j];
		eos_table[n][20 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xfdt[i * EOSJMAX + j];
	}

	gpuMemcpy(GPU_eos_table[n], eos_table[n], (EOSIMAX * EOSJMAX + eos_offset) * 21 * sizeof(double), gpuMemcpyHostToDevice);
	gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in eos_init_GPU: %d\n", status);
}
#endif

void GPU_write(int n)
{
	int i, j, z, k;
	double r, th, phi, X[NDIM];
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	#if(SPHERICAL|| SPHERICAL_GR)
	for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
		coord(n, i, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		radius_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G)] = r;
	}
	#endif

	#pragma omp parallel private(i, j, z, k, X, r, th, phi)
	{
		#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = p[nl[n]][index_3D(n, i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[nl[n]][index_3D(n, i, j, z)][k];
				#endif
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[nl[n]][index_3D(n, i, j, z)][k];
				#if(GPU_DEBUG)
				psh_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = psh[nl[n]][index_3D(n, i, j, z)][k];
				#endif
			}
			#endif
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
			}
			#if(CARTESIAN_GR)
			pflag_CART_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = pflag_cart[nl[n]][index_3D(n, i, j, z)];
			#endif

			#if(DO_RBOUND)
			pflag_RBOUND_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = pflag_rbound[nl[n]][index_3D(n, i, j, z)];
			#endif

			#if(CARTESIAN|| CARTESIAN_GR)
			coord(n, i, j, z, CENT, X);
			bl_coord(X, &r, &th, &phi);
			radius_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G) * (BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G) * (BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = r;
			#endif
		}
	}

	status = 0;
	#if (ELLIPTICAL2)
	for (i = N1_GPU_offset[n] - N1G; i<N1_GPU_offset[n] + BS_1 + N1G; i++){
		for (j = N2_GPU_offset[n] - N2G; j<N2_GPU_offset[n] + BS_2 + N2G; j++){
			for (k = 0; k < NPR; k++){
				dU_GPU[nl[n]][k*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = dU_s[nl[n]][index_2D(n, i, j, 0)][k];
			}
		}
	}
	status = gpuMemcpyAsync(BufferdU[nl[n]], dU_GPU[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	/*Initialize memory items that have to be passed on to the GPU*/
	gpuMemcpyAsync(Bufferp_1[nl[n]], p_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Bufferph_1[nl[n]], p_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#if(STAGGERED)
	gpuMemcpyAsync(Bufferps_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Bufferpsh_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	//gpuMemcpyAsync(Bufferpflag[nl[n]], pflag_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int), gpuMemcpyHostToDevice,commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Bufferfailimage[nl[n]], failimage_GPU[nl[n]], NFAIL*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#if(CARTESIAN || CARTESIAN_GR)
	gpuMemcpyAsync(Bufferradius[nl[n]], radius_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#else
	gpuMemcpyAsync(Bufferradius[nl[n]], radius_GPU[nl[n]], (BS_1 + 2 * N1G)*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	#if(CARTESIAN_GR)
	gpuMemcpyAsync(Bufferpflag_CART[nl[n]], pflag_CART_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	#if(DO_RBOUND)
	gpuMemcpyAsync(Bufferpflag_RBOUND[nl[n]], pflag_RBOUND_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif

	/*Copy metric to GPU*/
	int pg;
	#pragma omp parallel private(i, j, z, pg)
	{
		#if(!NSY)
		#pragma omp for collapse(2) schedule(static, (BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]){
		#else
		#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G){
		#endif
			for (pg = 0; pg < NPG; pg++){
				#if(!NSY)
				gdet_GPU[nl[n]][pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gdet[nl[n]][index_2D(n, i, j, z)][pg];
				#else
				gdet_GPU[nl[n]][pg*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gdet[nl[n]][index_2D(n, i, j, z)][pg];
				#endif
				#if(!NSY)
				gcov_GPU[nl[n]][0*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				gcon_GPU[nl[n]][0*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				gcov_GPU[nl[n]][1*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				gcon_GPU[nl[n]][1*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				gcov_GPU[nl[n]][2*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				gcon_GPU[nl[n]][2*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				gcov_GPU[nl[n]][3*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				gcon_GPU[nl[n]][3*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				gcov_GPU[nl[n]][4*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				gcon_GPU[nl[n]][4*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				gcov_GPU[nl[n]][5*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				gcon_GPU[nl[n]][5*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				gcov_GPU[nl[n]][6*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				gcon_GPU[nl[n]][6*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				gcov_GPU[nl[n]][7*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				gcon_GPU[nl[n]][7*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				gcov_GPU[nl[n]][8*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				gcon_GPU[nl[n]][8*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				gcov_GPU[nl[n]][9*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				gcon_GPU[nl[n]][9*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				#else
				gcov_GPU[nl[n]][0 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				gcon_GPU[nl[n]][0 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				gcov_GPU[nl[n]][1 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				gcon_GPU[nl[n]][1 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				gcov_GPU[nl[n]][2 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				gcon_GPU[nl[n]][2 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				gcov_GPU[nl[n]][3 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				gcon_GPU[nl[n]][3 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				gcov_GPU[nl[n]][4 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				gcon_GPU[nl[n]][4 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				gcov_GPU[nl[n]][5 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				gcon_GPU[nl[n]][5 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				gcov_GPU[nl[n]][6 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				gcon_GPU[nl[n]][6 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				gcov_GPU[nl[n]][7 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				gcon_GPU[nl[n]][7 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				gcov_GPU[nl[n]][8 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				gcon_GPU[nl[n]][8 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				gcov_GPU[nl[n]][9 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				gcon_GPU[nl[n]][9 * NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				#endif
			}

			#if(FRAME_TRANSFORM)
			int i1, j1;
			for (pg = 0; pg < NSOLVER; pg++) {
				#if(!NSY)
				for (i1 = 0; i1 < NDIM; i1++)for (j1 = 0; j1 < NDIM; j1++) {
					Mud_GPU[nl[n]][(i1*NDIM + j1) * NSOLVER*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = Mud[nl[n]][index_2D(n, i, j, z)][pg][i1][j1];
					Mud_inv_GPU[nl[n]][(i1*NDIM + j1)  * NSOLVER*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = Mud_inv[nl[n]][index_2D(n, i, j, z)][pg][i1][j1];
				}
				#else
				for (i1 = 0; i1 < NDIM; i1++)for (j1 = 0; j1 < NDIM; j1++) {
					Mud_GPU[nl[n]][(i1*NDIM + j1) * NSOLVER*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = Mud[nl[n]][index_2D(n, i, j, z)][pg][i1][j1];
					Mud_inv_GPU[nl[n]][(i1*NDIM + j1) * NSOLVER*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = Mud_inv[nl[n]][index_2D(n, i, j, z)][pg][i1][j1];
				}
				#endif
			}
			#endif

			for (pg = 0; pg < NDIM; pg++){
				#if(!NSY)
				conn_GPU[nl[n]][0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				conn_GPU[nl[n]][1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				conn_GPU[nl[n]][2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				conn_GPU[nl[n]][3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				conn_GPU[nl[n]][4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				conn_GPU[nl[n]][5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				conn_GPU[nl[n]][6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				conn_GPU[nl[n]][7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				conn_GPU[nl[n]][8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				conn_GPU[nl[n]][9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				#else
				conn_GPU[nl[n]][0 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][0];
				conn_GPU[nl[n]][1 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][1];
				conn_GPU[nl[n]][2 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][2];
				conn_GPU[nl[n]][3 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][0][3];
				conn_GPU[nl[n]][4 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][1];
				conn_GPU[nl[n]][5 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][2];
				conn_GPU[nl[n]][6 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][1][3];
				conn_GPU[nl[n]][7 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][2][2];
				conn_GPU[nl[n]][8 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][2][3];
				conn_GPU[nl[n]][9 * NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][3][3];
				#endif
			}
			#if(LEER)
			for (k = 0; k < 6; k++){
				dq_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = V[nl[n]][index_3D(n, i, j, z)][k];
			}
			#endif
		}
	}

	#if(!NSY)
	gpuMemcpyAsync(Buffergdet[nl[n]], gdet_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Buffergcov[nl[n]], gcov_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Buffergcon[nl[n]], gcon_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#if(FRAME_TRANSFORM)
	gpuMemcpyAsync(BufferMud[nl[n]], Mud_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(BufferMud_inv[nl[n]], Mud_inv_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	gpuMemcpyAsync(Bufferconn[nl[n]], conn_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#else
	gpuMemcpyAsync(Buffergdet[nl[n]], gdet_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Buffergcov[nl[n]], gcov_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(Buffergcon[nl[n]], gcon_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#if(FRAME_TRANSFORM)
	gpuMemcpyAsync(BufferMud[nl[n]], Mud_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	gpuMemcpyAsync(BufferMud_inv[nl[n]], Mud_inv_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	gpuMemcpyAsync(Bufferconn[nl[n]], conn_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10*sizeof(double), gpuMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in GPU_write: %d \n", status);
}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	/*Set arguments of kernel*/
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;

	if (dir == 1){
		if (flag == 1){
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
	}
	else if (dir == 2){
		if (flag == 1){
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
	}
	else{
		if (flag == 1){
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(fluxcalcprep, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], POLE_1, POLE_2);
			#endif
		}
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error Fluxcalcprep %d \n", status);
}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) / LOCAL_WORK_SIZE;

	//If on an Cartesion grid, first set boundary conditions in the hole in the middle (or holes anywhere else in case of binary metric)
	#if(CARTESIAN_GR)
	//GPU_boundprim_cart(dir, 0, n);
	//GPU_boundprim_cart(dir, 1, n);
	#endif
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){
		if (dir == 1){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][1], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][1], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
		if (dir == 2){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], 
					lim, dir, cour, dtij2_GPU[nl[n]], POLE_1, POLE_2, 
					dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], 
					lim, dir, cour, dtij2_GPU[nl[n]], POLE_1, POLE_2, 
					dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
		if (dir == 3){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
	}
	else{
		if (dir == 1){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
		if (dir == 2){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir,  cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
		if (dir == 3){
			#if(FRAME_TRANSFORM)
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D_FT, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D_FT << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					BufferMud[nl[n]], BufferMud_inv[nl[n]], lim, dir, cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					);
				#endif
			#else
				#if(SHIP)
				hipLaunchKernelGGL(fluxcalc2D2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir, cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					  dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir, cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					  dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(NEUTRINOS_M1)
					, GPU_nulib_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu, magnetic_density_scale_cpu
					#endif
					#if(DO_RBOUND)
					, Bufferpflag_RBOUND[nl[n]]
					#endif
					);
				#endif
			#endif
		}
	}
	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error Fluxcalc2D2 %d\n", status);
}

void GPU_reconstruct_internal(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	if ((block[n][AMR_POLE] != 0) || (block[n][AMR_NBR1] < 0) || (block[n][AMR_NBR3] < 0)) {
		int nr_workgroups_local[1];
		int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
		int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);

		/*Calculate reconstructed left state*/
		if (flag == 1) {
			nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;
			#if(SHIP)
			hipLaunchKernelGGL(interpolate, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], 3, POLE_1, POLE_2);
			#elif(SCUDA)
			interpolate << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], 3, POLE_1, POLE_2);
			#endif

			nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1)* (BS_2 + 2 * D2)* (BS_3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1)* (BS_2 + 2 * D2)* (BS_3)) / LOCAL_WORK_SIZE;
			#if(SHIP)
			hipLaunchKernelGGL(reconstruct_internal, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
			#endif
		}
		else {
			nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;
			#if(SHIP)
			hipLaunchKernelGGL(interpolate, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], 3, POLE_1, POLE_2);
			#elif(SCUDA)
			interpolate << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], 3, POLE_1, POLE_2);
			#endif

			nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1)* (BS_2 + 2 * D2)* (BS_3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1)* (BS_2 + 2 * D2)* (BS_3)) / LOCAL_WORK_SIZE;
			#if(SHIP)
			hipLaunchKernelGGL(reconstruct_internal, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
			#elif(SCUDA)
			reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status) fprintf(stderr, "Error GPU_reconstruct_internal %d\n", status);
	}
}

/*Start reading timestep from GPU*/
void read_time_GPU(void){

}

/*Do last step of reduction of timestep on CPU*/
double fluxcalc_GPU(int n, int dir)
{
	double ndt;
	int y;
	ndt = 1.e9;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	gpuStreamSynchronize(commandQueueGPU[nl[n]]);
	status = gpuGetLastError();
	if (status != 0) fprintf(stderr, "Error fluxcalc_GPU %d\n", status);
	if (dir == 1) {
		for (y = 0; y <  nr_workgroups2_1[nl[n]]; y++) {
			if (dtij1_GPU[nl[n]][y] < ndt && dtij1_GPU[nl[n]][y] < 1.e9 && dtij1_GPU[nl[n]][y] > 1.e-6) {
				ndt = dtij1_GPU[nl[n]][y];
			}
		}
	}
	else if (dir == 2) {
		for (y = 0; y < nr_workgroups2_2[nl[n]]; y++) {
			if (dtij2_GPU[nl[n]][y] < ndt && dtij2_GPU[nl[n]][y] < 1.e9 && dtij2_GPU[nl[n]][y] > 1.e-6) {
				ndt = dtij2_GPU[nl[n]][y];
			}
		}
	}
	else if (dir == 3) {
		for (y = 0; y < nr_workgroups2_3[nl[n]]; y++) {
			if (dtij3_GPU[nl[n]][y] < ndt && dtij3_GPU[nl[n]][y] < 1.e9 && dtij3_GPU[nl[n]][y] > 1.e-6) {
				ndt = dtij3_GPU[nl[n]][y];
			}
		}
	}
	return(ndt);
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	#if(SHIP)
	hipLaunchKernelGGL(fix_flux, nr_workgroups_special[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	#elif(SCUDA)
	fix_flux << < nr_workgroups_special[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	#endif

	// gpuDeviceSynchronize();
	 status = gpuGetLastError();
	if (gpuSuccess != status)fprintf(stderr, "Error fixflux %d \n", status);
}

void GPU_consttransport_bound(void){
	int n, flag;

	gpu = 1;
	#if(BOUND_TYPE2 == TRANSMISSIVE)
	E_average();
	#endif
	set_iprobe(0, &flag);

	#if(PRESTEP)
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		E_send1(E_corn, BufferE_1, n_ord[n]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
		#if(!TIMESTEP_JET)
		#if(N3G>0)
		E_send3(E_corn, BufferE_1, n_ord[n]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
	}

	//For last timestep synchronize electric fields immediately
	do{
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			E_rec1(E_corn, BufferE_1, n_ord[n], 5);
			E_rec2(E_corn, BufferE_1, n_ord[n], 5);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 5);
			#endif
			#endif
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		#if(!TIMESTEP_JET)
		#if(N3G>0)
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
	}

	//For first timestep do not synchronize electrice fields
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
		#if(!TIMESTEP_JET)
		#if(N3G>0)
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
	}
	#elif(PRESTEP2)
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		E_send1(E_corn, BufferE_1, n_ord[n]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
		#if(!TIMESTEP_JET)
		#if(N3G>0)
		E_send3(E_corn, BufferE_1, n_ord[n]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
	}
	#else
		#if(AVG_EMF)
		int nstep_temp;
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				nstep_temp = nstep;
				nstep = -100;
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				E_send1(E_corn, BufferE_1, n_ord[n]);
				E_send2(E_corn, BufferE_1, n_ord[n]);
				#if(D3>0)
				E_send3(E_corn, BufferE_1, n_ord[n]);
				#endif
				nstep = nstep_temp;
			}
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
				nstep_temp = nstep;
				nstep = -100;
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				#if(D3>0)
				E1_send_corn(E_corn, BufferE_1, n_ord[n]);
				E2_send_corn(E_corn, BufferE_1, n_ord[n]);
				#endif
				E3_send_corn(E_corn, BufferE_1, n_ord[n]);
				nstep = nstep_temp;
			}
		#endif

		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			E_send1(E_corn, BufferE_1, n_ord[n]);
			E_send2(E_corn, BufferE_1, n_ord[n]);
			#if(D3>0)
			E_send3(E_corn, BufferE_1, n_ord[n]);
			#endif
		}
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1) {
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			#if(D3>0)
			E1_send_corn(E_corn, BufferE_1, n_ord[n]);
			E2_send_corn(E_corn, BufferE_1, n_ord[n]);
			#endif
			E3_send_corn(E_corn, BufferE_1, n_ord[n]);
		}
		do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				E_rec1(E_corn, BufferE_1, n_ord[n], 1);
			}
			set_iprobe(1, &flag);
		} while (flag);
		set_iprobe(0, &flag);

		do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				E_rec2(E_corn, BufferE_1, n_ord[n], 1);
			}
			set_iprobe(1, &flag);
		} while (flag);
		set_iprobe(0, &flag);

		#if(N3G>0)
		do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				E_rec3(E_corn, BufferE_1, n_ord[n], 1);
			}
			set_iprobe(1, &flag);
		} while (flag);
		set_iprobe(0, &flag);
		#endif

		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			#if(D3>0)
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
		}
	#endif
}

void GPU_consttransport1(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) % LOCAL_WORK_SIZE) + (BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	if (flag == 1){
		#if(SHIP)
		hipLaunchKernelGGL(consttransport1, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
		#elif(SCUDA)
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(consttransport1, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
		#elif(SCUDA)
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
		#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status)fprintf(stderr, "Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	if (flag == 1){
		#if(SHIP)
		hipLaunchKernelGGL(consttransport2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2
			#if(CALC_MDOT)
			, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#elif(SCUDA)
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2
			#if(CALC_MDOT)
			, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#endif
	}
	else{
	#if(SHIP)
	hipLaunchKernelGGL(consttransport2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]],  BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2
		#if(CALC_MDOT)
		, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#elif(SCUDA)
	consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2
		#if(CALC_MDOT)
		, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status)fprintf(stderr, "Error constransport2 %d \n", status);
}

void GPU_consttransport2_M1_2(int flag, double Dt, int n) {
	#if(RAD_M1)
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	if (flag == 1) {
		#if(SHIP)
		hipLaunchKernelGGL(consttransport2_M1_2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport2_M1_2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#endif
	}
	else {
		#if(SHIP)
		hipLaunchKernelGGL(consttransport2_M1_2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport2_M1_2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#endif
	}
	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status)fprintf(stderr, "Error constransport2 %d \n", status);
	#endif
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	if (flag == 1){
		#if(SHIP)
		hipLaunchKernelGGL(consttransport3, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(consttransport3, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
		#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status )fprintf(stderr, "Error constransport3 %d \n", status);
}

void GPU_consttransport3_post(double Dt, int n){
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2) * (BS_3 + D3) + 2 * (BS_1 + D1) * (BS_3 + D3) + 2 * (BS_1 + D1) * (BS_2 + D2)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2) * (BS_3 + D3) + 2 * (BS_1 + D1) * (BS_3 + D3) + 2 * (BS_1 + D1) * (BS_2 + D2)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	#if(SHIP)
	hipLaunchKernelGGL(consttransport3_post, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
	#elif(SCUDA)
	consttransport3_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, POLE_1, POLE_2);
	#endif

	status = gpuGetLastError();
	if (gpuSuccess != status)fprintf(stderr, "Error constransport3_post %d \n", status);
}

void GPU_flux_ct1(int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	#if(SHIP)
	hipLaunchKernelGGL(flux_ct1, nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	#elif(SCUDA)
	flux_ct1 << < nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	#endif

	// gpuDeviceSynchronize();
	 status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	/*Run kernel*/
	#if(SHIP)
	hipLaunchKernelGGL(flux_ct2, nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	#elif(SCUDA)
	flux_ct2 << < nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	#endif

	 //gpuDeviceSynchronize();
	 status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error fluxct2 %d\n", status);
}

void GPU_Utoprim_M1_0(int n, double Dt)
{
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	#if(SHIP)
	hipLaunchKernelGGL(Utoprim_M1_0, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferdU_RAD0[nl[n]], Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#elif(SCUDA)
	Utoprim_M1_0 << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferdU_RAD0[nl[n]], Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error GPU_Utoprim_M1_0 %d\n", status);
}

void GPU_Utoprim_M1_1(int n, double Dt)
{
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);

	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1) * (BS_2) * (BS_3)) % LOCAL_WORK_SIZE) + (BS_1) * (BS_2) * (BS_3)) / LOCAL_WORK_SIZE;

	#if(SHIP)
	hipLaunchKernelGGL(Utoprim_M1_1, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferU_1[nl[n]], BufferdU_RAD0[nl[n]], BufferdU_RAD1[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#elif(SCUDA)
	Utoprim_M1_1 << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferU_1[nl[n]], BufferdU_RAD0[nl[n]], BufferdU_RAD1[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error GPU_Utoprim_M1_1 %d\n", status);
}

void GPU_Utoprim_M1_2(int n, double Dt)
{
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1) * (BS_2) * (BS_3)) % LOCAL_WORK_SIZE) + (BS_1) * (BS_2) * (BS_3)) / LOCAL_WORK_SIZE;

	#if(SHIP)
	hipLaunchKernelGGL(Utoprim_M1_2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferU_1[nl[n]], BufferdU_RAD0[nl[n]], BufferdU_RAD1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#elif(SCUDA)
	Utoprim_M1_2 << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferp_1[nl[n]], BufferU_n[nl[n]], BufferU_0[nl[n]], BufferU_1[nl[n]], BufferdU_RAD0[nl[n]], BufferdU_RAD1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, y_max, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error GPU_Utoprim_M1_2 %d\n", status);
}

void GPU_fixup(int flag, int n, double Dt)
{
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)*(BS_2)*(BS_3)) % LOCAL_WORK_SIZE) + (BS_1)*(BS_2)*(BS_3)) / LOCAL_WORK_SIZE;

	if (flag == 0){
		#if(SHIP)
		hipLaunchKernelGGL(fixup, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferstorage2[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, flag, POLE_1, POLE_2, y_max
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if (NEUTRINOS_M1) 
			, GPU_nulib_table[0], Bufferpflag_NU[nl[n]]
				#if (NEUTRINOS_DEBUG)
				, Bufferallflags_NU[nl[n]]
				#endif
			#endif
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#elif(SCUDA)
		fixup << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferstorage2[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, flag, POLE_1, POLE_2, y_max
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if (NEUTRINOS_M1) 
			, GPU_nulib_table[0], Bufferpflag_NU[nl[n]]
				#if (NEUTRINOS_DEBUG)
				, Bufferallflags_NU[nl[n]]
				#endif
			#endif
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(fixup, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferstorage2[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, flag, POLE_1, POLE_2, y_max
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if (NEUTRINOS_M1) 
			, GPU_nulib_table[0], Bufferpflag_NU[nl[n]]
				#if (NEUTRINOS_DEBUG)
				, Bufferallflags_NU[nl[n]]
				#endif
			#endif
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#elif(SCUDA)
		fixup << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferstorage2[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, flag, POLE_1, POLE_2, y_max
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if (NEUTRINOS_M1) 
			, GPU_nulib_table[0], Bufferpflag_NU[nl[n]]
			#if (NEUTRINOS_DEBUG)
			, Bufferallflags_NU[nl[n]]
			#endif
			#endif
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu, magnetic_density_scale_cpu
			#endif
			#if(CARTESIAN_GR)
			, Bufferpflag_CART[nl[n]]
			#endif
			);
		#endif
	}
	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error fixup %d\n", status);
}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)*(BS_2)*(BS_3)) % LOCAL_WORK_SIZE) + (BS_1)*(BS_2)*(BS_3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	if (flag == 1){
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferradius[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
			, magnetic_density_scale_cpu
			#endif
		);
		#elif(SCUDA)
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferradius[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
			, magnetic_density_scale_cpu
			#endif
		);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferradius[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
			, magnetic_density_scale_cpu
			#endif
		);
		#elif(SCUDA)
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferradius[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]
			#if (DOHELM) 
			, GPU_eos_table[0]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
			, magnetic_density_scale_cpu
			#endif
		);
		#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_fixuputoprim_rad(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1) * (BS_2) * (BS_3)) % LOCAL_WORK_SIZE) + (BS_1) * (BS_2) * (BS_3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	if (flag == 1) {
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim_rad, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim_rad << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]]);
		#endif
	}
	else {
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim_rad, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim_rad << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpflag_RAD[nl[n]], Bufferfailimage[nl[n]]);
		#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_fixuputoprim_nu(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1) * (BS_2) * (BS_3)) % LOCAL_WORK_SIZE) + (BS_1) * (BS_2) * (BS_3)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	if (flag == 1) {
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim_nu, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferpflag_NU[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim_nu << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferpflag_NU[nl[n]], Bufferfailimage[nl[n]]);
		#endif
	}
	else {
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim_nu, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpflag_NU[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim_nu << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpflag_NU[nl[n]], Bufferfailimage[nl[n]]);
		#endif
	}

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_cleanup_post(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	#if(SHIP)
	hipLaunchKernelGGL(cleanup_post, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], BufferE_1[nl[n]]);
	#elif(SCUDA)
	cleanup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], BufferE_1[nl[n]]);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error cleanup_post %d\n", status);
}

void GPU_fixup_post(int n, double Dt)
{
	int nr_workgroups_local[1];
	int POLE_1 = block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3);
	int POLE_2 = block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2) % LOCAL_WORK_SIZE) + 2 * (BS_2)*(BS_3) + (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2)) / LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	#if(SHIP)
	hipLaunchKernelGGL(fixup_post, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, 1, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(RAD_M1)
		, Bufferpflag_RAD[nl[n]]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		#if(DO_RBOUND)
		, Bufferpflag_RBOUND[nl[n]]
		#endif
		);
	#elif(SCUDA)
		fixup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Dt, 1, POLE_1, POLE_2
		#if (DOHELM) 
		, GPU_eos_table[0]
		#endif
		#if(RAD_M1)
		, Bufferpflag_RAD[nl[n]]
		#endif
		#if(CALC_MDOT)
		, mass_density_scale_cpu, magnetic_density_scale_cpu
		#endif
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
		#endif
		#if(DO_RBOUND)
		, Bufferpflag_RBOUND[nl[n]]
		#endif
		);
	#endif

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error fixup_post %d\n", status);
}

void GPU_boundprim(int bound_force)
{
	int n, flag;
	int temp = nstep;
	gpu = 1;

	if (bound_force == 1) nstep = -1;
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif

	#if(BOUND_TYPE1==OUTFLOW)
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) {
				GPU_boundprim1_outflow(1, n_ord[n]);
				if (nstep == -1) GPU_boundprim1_outflow(0, n_ord[n]);
			}
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1_outflow(0, n_ord[n]);
		}
	#endif

	#if(BOUND_TYPE2==OUTFLOW)
		#if(GPU_OPENMP)
		//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) {
				GPU_boundprim2_outflow(1, n_ord[n]);
				if (nstep == -1) GPU_boundprim2_outflow(0, n_ord[n]);
			}
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2_outflow(0, n_ord[n]);
		}
	#elif(BOUND_TYPE2==REFLECTIVE)
		#if(GPU_OPENMP)
		//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) {
				GPU_boundprim2_reflective(1, n_ord[n]);
				if (nstep == -1) GPU_boundprim2_reflective(0, n_ord[n]);
			}
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2_reflective(0, n_ord[n]);
		}
	#endif

	#if(BOUND_TYPE3==OUTFLOW)
		for (n = 0; n < n_active; n++){
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) {
				GPU_boundprim3_outflow(1, n_ord[n]);
				if (nstep == -1) GPU_boundprim3_outflow(0, n_ord[n]);
			}
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim3_outflow(0, n_ord[n]);
		}
	#endif

	#if(PRESTEP)
	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		set_iprobe(0, &flag);
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			flux_rec1(F1, BufferF1_1, n_ord[n], 3);
			flux_rec2(F2, BufferF2_1, n_ord[n], 3);
			flux_rec3(F3, BufferF3_1, n_ord[n], 3);

			E_rec1(E_corn, BufferE_1, n_ord[n], 3);
			E_rec2(E_corn, BufferE_1, n_ord[n], 3);
			#if(!TIMESTEP_JET)
			#if(N3G>0)
			E_rec3(E_corn, BufferE_1, n_ord[n], 3);
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
		}

		do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				flux_rec1(F1, BufferF1_1, n_ord[n], 1);
				flux_rec2(F2, BufferF2_1, n_ord[n], 1);
				flux_rec3(F3, BufferF3_1, n_ord[n], 1);
			}
			set_iprobe(1, &flag);
		}while(flag);
		set_iprobe(0, &flag);
		do{
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				E_rec1(E_corn, BufferE_1, n_ord[n], 1);
				E_rec2(E_corn, BufferE_1, n_ord[n], 1);
				#if(!TIMESTEP_JET)
				#if(N3G>0)
				E_rec3(E_corn, BufferE_1, n_ord[n], 1);
				#endif
				#endif
			}
			set_iprobe(1, &flag);
		}while(flag);
		set_iprobe(0, &flag);
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			#if(!TIMESTEP_JET)
			#if(N3G>0)
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE/BoundcomF \n");
	#endif

	//For last timestep do not receive synchronized electrice fields
	//for (n = gpu_offset; n < gpu_offset + N_GPU; n++) {
		//#if(N_GPU>1)
		//gpuSetDevice(n);
		//#endif
		//gpuDeviceSynchronize();
	//}
	mpi_synch(bound_force);

	if (rank == 0) begin2 = get_wall_time();
	rc = 0;
	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do{
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	#if(GPU_OPENMP)
	#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		#if(N_GPU>1)
		gpuSetDevice(block[n_ord[n]][AMR_GPU]);
		#endif
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do{
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);
	if (N3 > 1){
		#if(GPU_OPENMP)
		#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
		#endif
		for (n = 0; n < n_active; n++){
			#if(N_GPU>1)
			gpuSetDevice(block[n_ord[n]][AMR_GPU]);
			#endif
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		set_iprobe(0, &flag);
		do {
			#if(GPU_OPENMP)
			#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
			#endif
			for (n = 0; n < n_active; n++){
				#if(N_GPU>1)
				gpuSetDevice(block[n_ord[n]][AMR_GPU]);
				#endif
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
			}
			set_iprobe(1, &flag);
		} while (flag);
	}

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");

	#if(BOUND_TYPE2==TRANSMISSIVE && NB_3==1)
	#if(GPU_OPENMP)
	//#pragma omp parallel for schedule(static,n_active/nthreads) private(n,status)
	#endif
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) {
			GPU_boundprim2_trans(1, n_ord[n]);
			if(nstep==-1) GPU_boundprim2_trans(0, n_ord[n]);
		}
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2_trans(0, n_ord[n]);
	}
	#endif

	#if(CARTESIAN_GR)
	for (n = 0; n < n_active; n++) {
		GPU_boundprim_cart(1, 0, n_ord[n]);
		GPU_boundprim_cart(1, 1, n_ord[n]);
	}
	#endif

	#if(DO_RBOUND)
	for (n = 0; n < n_active; n++) {
		GPU_boundprim_rbound(1, 0, n_ord[n]);
		GPU_boundprim_rbound(1, 1, n_ord[n]);
	}
	#endif

	//MPI communication
	//for (n = gpu_offset; n < gpu_offset + N_GPU; n++) {
		//#if(N_GPU>1)
		//gpuSetDevice(n);
		//#endif
		//gpuDeviceSynchronize();
	//}
	//#if(PRESTEP2)
	mpi_synch(bound_force);
	//#endif

	if (rank == 0){
		end2 = get_wall_time();
		time_spent3 += (double)(end2 - begin2);
	}

	nstep = temp;
}

void GPU_boundprim1_outflow(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim1_outflow, nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]
				#if(DANAT_GDET_INTERP)	
				, Bufferradius[nl[n]]
				#endif
			);
			#elif(SCUDA)
			boundprim1_outflow << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]
				#if(DANAT_GDET_INTERP)	
				, Bufferradius[nl[n]]
				#endif
			);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim1_outflow, nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]
				#if(DANAT_GDET_INTERP)	
				, Bufferradius[nl[n]]
				#endif
			);
			#elif(SCUDA)
			boundprim1_outflow << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]
				 #if(DANAT_GDET_INTERP)	
				 , Bufferradius[nl[n]]
				 #endif
			);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim1_outflow %d\n", status);
	}
}

void GPU_boundprim2_reflective(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_reflective, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_reflective << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_reflective, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_reflective << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status) fprintf(stderr, "Error boundprim2_reflective %d\n", status);
	}
}

void GPU_boundprim2_outflow(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_outflow, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_outflow << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_outflow, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_outflow << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim2_outflow %d\n", status);
	}
}

void GPU_boundprim2_trans(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_POLE] != 0 ){
		if (flag == 0){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_trans, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferpsh_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_trans << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferpsh_1[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim2_trans, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferps_1[nl[n]]);
			#elif(SCUDA)
			boundprim2_trans << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferps_1[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status) fprintf(stderr, "Error boundprim2_trans %d\n", status);
	}
}

void GPU_boundprim3_outflow(int flag, int n)
{
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
	if (block[n][AMR_NBR5] == -1 || block[n][AMR_NBR6] == -1){
		if (flag == 0){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim3_outflow, nr_workgroups_special3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR5], block[n][AMR_NBR6], Bufferpsh_1[nl[n]]);
			#elif(SCUDA)
			boundprim3_outflow << < nr_workgroups_special3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR5], block[n][AMR_NBR6], Bufferpsh_1[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim3_outflow, nr_workgroups_special3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR5], block[n][AMR_NBR6], Bufferps_1[nl[n]]);
			#elif(SCUDA)
			boundprim3_outflow << < nr_workgroups_special3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR5], block[n][AMR_NBR6], Bufferps_1[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim3_outflow %d\n", status);
	}
}

void GPU_boundprim_cart(int dir, int flag, int n)
{
	if(block[n][AMR_CARTFLAG]==1){
		int nr_workgroups_local = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G))) / LOCAL_WORK_SIZE;

		#if(N_GPU>1)
		gpuSetDevice(block[n][AMR_GPU]);
		#endif
		if (flag == 1){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim_cart, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_CART[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#elif(SCUDA)
			boundprim_cart << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_CART[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim_cart, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_CART[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#elif(SCUDA)
			boundprim_cart << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_CART[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim_cart %d\n", status);
	}
}

void GPU_boundprim_rbound(int dir, int flag, int n)
{
	if(block[n][AMR_RBOUNDFLAG]==1){
		int nr_workgroups_local = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G))) / LOCAL_WORK_SIZE;

		#if(N_GPU>1)
		gpuSetDevice(block[n][AMR_GPU]);
		#endif
		if (flag == 1){
			#if(SHIP)
			hipLaunchKernelGGL(boundprim_rbound, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_RBOUND[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#elif(SCUDA)
			boundprim_rbound << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_RBOUND[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim_rbound, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_RBOUND[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#elif(SCUDA)
			boundprim_rbound << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_RBOUND[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim_rbound %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k;

	//gpuDeviceSynchronize();
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	#if(N_LEVELS_1D_INT>20)
	int nr_workgroups_local[1];

	/*Calculate gradients for reconstruction*/
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;
	#if(SHIP)
	hipLaunchKernelGGL(interpolate, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], 3, (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	#elif(SCUDA)
	interpolate << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], 3, (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	#endif

	/*Reconstruct derefined region around pole for Ray-Tracing and store in temporary variable (used for fluxes normally)*/
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)* (BS_2)* (BS_3)) % LOCAL_WORK_SIZE) + (BS_1)* (BS_2)* (BS_3)) / LOCAL_WORK_SIZE;
	gpuMemcpyAsync(BufferF1_1[nl[n]], Bufferp_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToDevice, commandQueueGPU[nl[n]]);
	#if(SHIP)
	hipLaunchKernelGGL(reconstruct_internal, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	#elif(SCUDA)
	reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	#endif

	gpuMemcpyAsync(p_1[nl[n]], BufferF1_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#else
	gpuMemcpyAsync(p_1[nl[n]], Bufferp_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	#if(GPU_DEBUG)
	gpuMemcpyAsync(ph_1[nl[n]], Bufferph_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	#if(STAGGERED)
	gpuMemcpyAsync(ps_1[nl[n]], Bufferps_1[nl[n]], (int)(3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#if(GPU_DEBUG)
	gpuMemcpyAsync(psh_1[nl[n]], Bufferpsh_1[nl[n]], (int)(3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	#endif
	gpuMemcpyAsync(failimage_GPU[nl[n]], Bufferfailimage[nl[n]], (int)((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#if (NEUTRINOS_DEBUG)
	gpuMemcpyAsync(allflags_NU_GPU[nl[n]], Bufferallflags_NU[nl[n]], (double)((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NEUTRINOS_DEBUG_NFLAGS * NU_SPECIES * sizeof(double), gpuMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	gpuDeviceSynchronize();

	if (n == n_ord[0]) {
		for (k = 0; k < NFAIL; k++) failimage_counter[k] = 0;
	}

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(3) schedule(static, (BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[nl[n]][index_3D(n, i, j, z)][k] = p_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				#if(GPU_DEBUG)
				ph[nl[n]][index_3D(n, i, j, z)][k] = ph_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				#endif
			}
			for (k = 0; k < NFAIL; k++){
				failimage[nl[n]][index_3D(n, i, j, z)][k] = failimage_GPU[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				if ((failimage[nl[n]][index_3D(n, i, j, z)][k] != 0) && (i >= N1_GPU_offset[n]) && (j >= N2_GPU_offset[n]) && (z >= N3_GPU_offset[n]) && (i < N1_GPU_offset[n] + BS_1) && (j < N2_GPU_offset[n] + BS_2) && (z < N3_GPU_offset[n] + BS_3)) {
					#pragma omp critical
					{
						if(block[n][AMR_POLE]==0) failimage_counter[k] += failimage[nl[n]][index_3D(n, i, j, z)][k];
					}
				}
			}
			#if (NEUTRINOS_DEBUG)
			for (k = 0; k < NU_SPECIES; k++) {
				allflags_NU[nl[n]][index_3D(n, i, j, z)][k] = allflags_NU_GPU[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#endif

			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps[nl[n]][index_3D(n, i, j, z)][k] = ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				#if(GPU_DEBUG)
				psh[nl[n]][index_3D(n, i, j, z)][k] = psh_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				#endif
			}
			#endif
		}
	}
	status = gpuGetLastError();
	if (gpuSuccess != status )fprintf(stderr, "Error in GPU_read: %d \n", status);
}

