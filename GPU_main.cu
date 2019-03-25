#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}
int gpuAlloc(void **devPtr, int size);
void alloc_bounds_GPU(int n);
void free_bound_gpu(int n);
int gpuFree(void *devPtr);

//Wrapper for allocation of boundary cells
#define gpuAlloc cudaMalloc
/*void gpuAlloc(void **devPtr, int size){
	#if(GPU_DIRECT)
	return cudaMalloc(devPtr, size);
	#else
	return cudaHostAlloc(devPtr, size,0);
	#endif
}*/

void GPU_init(void)
{
	int i,j,ranks_per_node;
	
	//Do some checks first
	if (N_GPU>numdevices){
		fprintf(stderr, "N_GPU is bigger than the number of devices! \n");
		exit(0);
	}

	//Enable peer access
	ranks_per_node = numdevices / N_GPU;
	gpu_offset = (rank % (ranks_per_node))*N_GPU;
	for (i = gpu_offset; i < gpu_offset + N_GPU; i++){
		//cudasetDevice(i);
		for (j = gpu_offset; j < gpu_offset + N_GPU; j++){
			if (i!=j) cudaDeviceEnablePeerAccess(j, 0);
		}
	}
	status = cudaGetLastError();
	if (cudaSuccess != status){
		fprintf(stderr, "Error in setting peeraccess: %d \n", status);
		exit(0);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting cache: %d \n", status);
}

void set_arrays_GPU(int n, int device){
	int i;

	if (mem_spot_gpu[nl[n]] == device){
		block[n][AMR_GPU] = device;	
		alloc_bounds_GPU(n);
		return;
	}
	else if (mem_spot_gpu[nl[n]] != device && mem_spot_gpu[nl[n]] != -1){
		GPU_finish(n, 1);
	}
	block[n][AMR_GPU] = device;
	//cudasetDevice(device);
	mem_spot_gpu[nl[n]] = device;

	/*Set the global work size and make sure that it is a multiple of the group size. The Nvidia OpenCL framework crashes otherwise!*/
	fix_mem[nl[n]] = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#if(!NSY)
	fix_mem2[nl[n]] = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#else
	fix_mem2[nl[n]] = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	#endif
	local_work_size[0] = LOCAL_WORK_SIZE;

	global_work_size_special[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) +
		(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G);
	global_work_size_special1[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * NG;
	global_work_size_special2[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * NG;
	global_work_size_special3[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * NG;
	nr_workgroups_special[nl[n]] = (int)ceil((double)global_work_size_special[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special1[nl[n]] = (int)ceil((double)global_work_size_special1[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special2[nl[n]] = (int)ceil((double)global_work_size_special2[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special3[nl[n]] = (int)ceil((double)global_work_size_special3[nl[n]][0] / (double)LOCAL_WORK_SIZE);

	global_work_size1[nl[n]][0] = (LOCAL_WORK_SIZE - (BS_1 * BS_2 * BS_3) % LOCAL_WORK_SIZE) + (BS_1 * BS_2 * BS_3); //utoprim
	global_work_size2[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3); //fluxcalc_prerp
	global_work_size2_1[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * D1 - 1) * (BS_2 + 2 * D2 ) * (BS_3 + 2 * D3 )) % LOCAL_WORK_SIZE)+ (BS_1 + 2 * D1 - 1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3); //fluxcalc
	global_work_size2_2[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * D1 ) * (BS_2 + 2 * D2 - 1) * (BS_3 + 2 * D3 )) % LOCAL_WORK_SIZE)+ (BS_1 + 2 * D1) * (BS_2 + 2 * D2 - 1) * (BS_3 + 2 * D3); //fluxcalc
	global_work_size2_3[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3 - 1)) % LOCAL_WORK_SIZE)+ (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3 - 1); //fluxcalc
	global_work_size3[nl[n]][0] = (LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3); //flux_ct

	nr_workgroups[nl[n]] = (int)ceil((double)global_work_size2[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups1[nl[n]] = (int)ceil((double)global_work_size1[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2[nl[n]] = (int)ceil((double)global_work_size2[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_1[nl[n]] = (int)ceil((double)global_work_size2_1[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_2[nl[n]] = (int)ceil((double)global_work_size2_2[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_3[nl[n]] = (int)ceil((double)global_work_size2_3[nl[n]][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups3[nl[n]] = (int)ceil((double)global_work_size3[nl[n]][0] / (double)LOCAL_WORK_SIZE);

	//Select correct CUDA device
	cudaStreamCreate(&commandQueueGPU[nl[n]]);

	//Create events
	for (i = 0; i < 600; i++) cudaEventCreate(&boundevent[nl[n]][i]);
	for (i = 0; i < 100; i++) cudaEventCreate(&boundevent1[nl[n]][i]);

	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in creating events: %d \n", status);

	/*Allocate memory to 1D arrays*/
	p_1[nl[n]] = (double(*))calloc(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(double));
	dq_1[nl[n]] = (double(*))calloc(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(double)); //array to store temporary data
	#if(STAGGERED)
	ps_1[nl[n]] = (double(*))calloc(NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(double));
	psh_1[nl[n]] = (double(*))calloc(NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(double));
	#endif
	ph_1[nl[n]] = (double(*))calloc(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(double));
	//dU_GPU[nl[n]] = (double(*))calloc(NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)), sizeof(double));
	#if(!NSY)
	gcov_GPU[nl[n]] = (double(*))calloc(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10, sizeof(double));
	gcon_GPU[nl[n]] = (double(*))calloc(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10, sizeof(double));
	conn_GPU[nl[n]] = (double(*))calloc(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10, sizeof(double));
	gdet_GPU[nl[n]] = (double(*))calloc(((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG, sizeof(double));
	#else
	gcov_GPU[nl[n]] = (double(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10, sizeof(double));
	gcon_GPU[nl[n]] = (double(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10, sizeof(double));
	conn_GPU[nl[n]] = (double(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10, sizeof(double));
	gdet_GPU[nl[n]] = (double(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG , sizeof(double));
	#endif
	//pflag_GPU[nl[n]] = (int(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(int));
	failimage_GPU[nl[n]] = (int(*))calloc(((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL ,sizeof(int));
	Katm_GPU[nl[n]] = (double(*))calloc((BS_1 + 2 * N1G) , sizeof(double));

	/*Allocate memory to buffers on GPU*/
	cudaMalloc(&BufferF1_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&BufferF2_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&BufferF3_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&Bufferdq_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&BufferE_1[nl[n]], NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&Bufferp_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&Bufferph_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#if(LEER)
	cudaMalloc(&BufferV[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#endif
	cudaMalloc(&Bufferradius[nl[n]], (BS_1 + 2 * N1G)*sizeof(double));
	cudaMalloc(&Bufferstorage1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));

	#if(N_LEVELS_1D_INT>0)
	cudaMalloc(&Bufferstorage2[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&Bufferstorage3[nl[n]], NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#else
	Bufferstorage2[nl[n]] = Bufferp_1[nl[n]];
	Bufferstorage3[nl[n]] = Bufferdq_1[nl[n]];
	#endif
	#if(STAGGERED)
	cudaMalloc(&Bufferps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	cudaMalloc(&Bufferpsh_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#endif
	#if(!NSY)
	cudaMalloc(&Buffergdet[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double));
	cudaMalloc(&Buffergcov[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	cudaMalloc(&Buffergcon[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	cudaMalloc(&Bufferconn[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM * 10 * sizeof(double));
	#else
	cudaMalloc(&Buffergdet[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double));
	cudaMalloc(&Buffergcov[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	cudaMalloc(&Buffergcon[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	cudaMalloc(&Bufferconn[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM * 10 * sizeof(double));
	#endif
	cudaMalloc(&Bufferpflag[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int));
	cudaMalloc(&Bufferfailimage[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int));
	cudaMalloc(&BufferKatm[nl[n]], (BS_1 + 2 * N1G)*sizeof(double));
	//cudaMalloc(&BufferdU[nl[n]], NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double));
	if (cudaSuccess != cudaGetLastError() ) fprintf(stderr, "Error in setting kernel arguments 3: %d \n", cudaGetLastError());
	cudaHostAlloc(&dtij_GPU[nl[n]], (nr_workgroups[nl[n]] + 1)*sizeof(double), 0);
	//cudaMalloc(&Bufferdtij[nl[n]], (nr_workgroups[nl[n]] + 1)*sizeof(double));
	
	alloc_bounds_GPU(n);
	
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting kernel arguments 4.6: %d \n", status);
}

//Set arrays for boundary cell transfer on GPU
void alloc_bounds_GPU(int n){
	int ref1_1, ref1_3, ref1_5, ref1_6;
	int ref2_2, ref2_4, ref2_5, ref2_6;
	int ref3_1, ref3_2, ref3_3, ref3_4;
	int ref1_1s, ref1_3s, ref1_5s, ref1_6s;
	int ref2_2s, ref2_4s, ref2_5s, ref2_6s;
	int ref3_1s, ref3_2s, ref3_3s, ref3_4s;
	
	ref1_1 = REF_1; ref1_3 = REF_1; ref1_5 = REF_1; ref1_6 = REF_1;
	ref2_2 = REF_2; ref2_4 = REF_2; ref2_5 = REF_2; ref2_6 = REF_2;
	ref3_1 = REF_3; ref3_2 = REF_3; ref3_3 = REF_3; ref3_4 = REF_3;
	ref1_1s = REF_1; ref1_3s = REF_1; ref1_5s = REF_1; ref1_6s = REF_1;
	ref2_2s = REF_2;ref2_4s = REF_2; ref2_5s = REF_2; ref2_6s = REF_2;
	ref3_1s = REF_3; ref3_2s = REF_3; ref3_3s = REF_3; ref3_4s = REF_3;

	if (block[n][AMR_LEVEL] != N_LEVELS - 1){
		if (block[n][AMR_NBR1_3] >= 0) ref1_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR3_1] >= 0) ref1_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL1] - block[n][AMR_LEVEL1];
		if (block[n][AMR_NBR1_3] >= 0) ref3_1 = block[block[n][AMR_NBR1_3]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
		if (block[n][AMR_NBR3_1] >= 0) ref3_3 = block[block[n][AMR_NBR3_1]][AMR_LEVEL3] - block[n][AMR_LEVEL3];
	}
	ref1_1s = ref1_1;
	ref1_3s = ref1_3;
	ref3_1s = ref3_1;
	ref3_3s = ref3_3;

	if (block[n][AMR_NBR1P] >= 0)ref1_1s = MY_MIN(ref1_1, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR1P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR3P] >= 0)ref1_3s = MY_MIN(ref1_3, block[n][AMR_LEVEL1] - block[block[n][AMR_NBR3P]][AMR_LEVEL1]);
	if (block[n][AMR_NBR1P] >= 0)ref3_1s = MY_MIN(ref3_1, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR1P]][AMR_LEVEL3]);
	if (block[n][AMR_NBR3P] >= 0)ref3_3s = MY_MIN(ref3_3, block[n][AMR_LEVEL3] - block[block[n][AMR_NBR3P]][AMR_LEVEL3]);
	if ((block[n][AMR_COORD2] == 0 || block[n][AMR_COORD2] == NB_2*(int)pow(1 + REF_2, block[n][AMR_LEVEL]) - 1) && DEREFINE_POLE){
		ref3_2s = 0;
		ref3_4s = 0;
	}

	gpuAlloc(&Buffersend1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend1_3[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend1_4[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend1_7[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend1_8[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_1s) + 2 * N1G)*(BS_3 / (1 + ref3_1s) + 2 * N3G)*sizeof(double));
	#endif
	gpuAlloc(&Buffersend2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend2_1[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_2s) + 2 * N2G)*(BS_3 / (1 + ref3_2s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend2_2[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_2s) + 2 * N2G)*(BS_3 / (1 + ref3_2s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend2_3[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_2s) + 2 * N2G)*(BS_3 / (1 + ref3_2s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend2_4[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_2s) + 2 * N2G)*(BS_3 / (1 + ref3_2s) + 2 * N3G)*sizeof(double));
	#endif
	gpuAlloc(&Buffersend3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend3_1[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend3_2[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend3_5[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend3_6[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_3s) + 2 * N1G)*(BS_3 / (1 + ref3_3s) + 2 * N3G)*sizeof(double));
	#endif
	gpuAlloc(&Buffersend4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend4_5[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_4s) + 2 * N2G)*(BS_3 / (1 + ref3_4s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend4_6[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_4s) + 2 * N2G)*(BS_3 / (1 + ref3_4s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend4_7[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_4s) + 2 * N2G)*(BS_3 / (1 + ref3_4s) + 2 * N3G)*sizeof(double));
	gpuAlloc(&Buffersend4_8[nl[n]], NG * (NPR + 3)*(BS_2 / (1 + ref2_4s) + 2 * N2G)*(BS_3 / (1 + ref3_4s) + 2 * N3G)*sizeof(double));
	#endif
	#if(N3G>0)
	gpuAlloc(&Buffersend5[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend5_1[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_5s) + 2 * N1G)*(BS_2 / (1 + ref2_5s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend5_3[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_5s) + 2 * N1G)*(BS_2 / (1 + ref2_5s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend5_5[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_5s) + 2 * N1G)*(BS_2 / (1 + ref2_5s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend5_7[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_5s) + 2 * N1G)*(BS_2 / (1 + ref2_5s) + 2 * N2G)*sizeof(double));
	#endif
	gpuAlloc(&Buffersend6[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuAlloc(&Buffersend6_2[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_6s) + 2 * N1G)*(BS_2 / (1 + ref2_6s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend6_4[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_6s) + 2 * N1G)*(BS_2 / (1 + ref2_6s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend6_6[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_6s) + 2 * N1G)*(BS_2 / (1 + ref2_6s) + 2 * N2G)*sizeof(double));
	gpuAlloc(&Buffersend6_8[nl[n]], NG * (NPR + 3)*(BS_1 / (1 + ref1_6s) + 2 * N1G)*(BS_2 / (1 + ref2_6s) + 2 * N2G)*sizeof(double));
	#endif
	#endif
	gpuAlloc(&Bufferrec1[nl[n]], (1 + ref1_3)*(1 + ref3_3)* NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G)*sizeof(double));
	Bufferrec1_3[nl[n]] = Bufferrec1[nl[n]];
	Bufferrec1_4[nl[n]] = Bufferrec1[nl[n]] + ref3_3*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	Bufferrec1_7[nl[n]] = Bufferrec1[nl[n]] + (ref3_3 + ref1_3)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	Bufferrec1_8[nl[n]] = Bufferrec1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	gpuAlloc(&Bufferrec2[nl[n]], (1 + ref2_4)*(1 + ref3_4)* NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G)*sizeof(double));
	Bufferrec2_1[nl[n]] = Bufferrec2[nl[n]];
	Bufferrec2_2[nl[n]] = Bufferrec2[nl[n]] + ref3_4*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	Bufferrec2_3[nl[n]] = Bufferrec2[nl[n]] + (ref3_4 + ref2_4)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	Bufferrec2_4[nl[n]] = Bufferrec2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	gpuAlloc(&Bufferrec3[nl[n]], (1 + ref1_1)*(1 + ref3_1)* NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G)*sizeof(double));
	Bufferrec3_1[nl[n]] = Bufferrec3[nl[n]];
	Bufferrec3_2[nl[n]] = Bufferrec3[nl[n]] + ref3_1*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	Bufferrec3_5[nl[n]] = Bufferrec3[nl[n]] + (ref3_1 + ref1_1)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	Bufferrec3_6[nl[n]] = Bufferrec3[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	gpuAlloc(&Bufferrec4[nl[n]], (1 + ref2_2)*(1 + ref3_2)* NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G)*sizeof(double));
	Bufferrec4_5[nl[n]] = Bufferrec4[nl[n]];
	Bufferrec4_6[nl[n]] = Bufferrec4[nl[n]] + ref3_2*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	Bufferrec4_7[nl[n]] = Bufferrec4[nl[n]] + (ref3_2 + ref2_2)*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	Bufferrec4_8[nl[n]] = Bufferrec4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)
	gpuAlloc(&Bufferrec5[nl[n]], (1 + ref1_6)*(1 + ref2_6)* NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G)*sizeof(double));
	Bufferrec5_1[nl[n]] = Bufferrec5[nl[n]];
	Bufferrec5_3[nl[n]] = Bufferrec5[nl[n]] + ref2_6*(NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	Bufferrec5_5[nl[n]] = Bufferrec5[nl[n]] + (ref2_6 + ref1_6)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	Bufferrec5_7[nl[n]] = Bufferrec5[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NG * (NPR + 3) * (BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	gpuAlloc(&Bufferrec6[nl[n]], (1 + ref1_5)*(1 + ref2_5)* NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G)*sizeof(double));
	Bufferrec6_2[nl[n]] = Bufferrec6[nl[n]];
	Bufferrec6_4[nl[n]] = Bufferrec6[nl[n]] + ref2_5*(NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	Bufferrec6_6[nl[n]] = Bufferrec6[nl[n]] + (ref2_5 + ref1_5)*(NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	Bufferrec6_8[nl[n]] = Bufferrec6[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NG * (NPR + 3) * (BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	#endif
	#if(PRESTEP || PRESTEP2)
	gpuAlloc(&tempBufferrec1[nl[n]], (1 + ref1_3)*(1 + ref3_3)* 2 * NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G)*sizeof(double));
	tempBufferrec1_3[nl[n]] = tempBufferrec1[nl[n]];
	tempBufferrec1_4[nl[n]] = tempBufferrec1[nl[n]] + ref3_3*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	tempBufferrec1_7[nl[n]] = tempBufferrec1[nl[n]] + (ref3_3 + ref1_3)*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	tempBufferrec1_8[nl[n]] = tempBufferrec1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_3) + 2 * N1G)*(BS_3 / (1 + ref3_3) + 2 * N3G));
	gpuAlloc(&tempBufferrec2[nl[n]], (1 + ref2_4)*(1 + ref3_4)* 2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G)*sizeof(double));
	tempBufferrec2_1[nl[n]] = tempBufferrec2[nl[n]];
	tempBufferrec2_2[nl[n]] = tempBufferrec2[nl[n]] + ref3_4*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempBufferrec2_3[nl[n]] = tempBufferrec2[nl[n]] + (ref3_4 + ref2_4)*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	tempBufferrec2_4[nl[n]] = tempBufferrec2[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_4) + 2 * N2G)*(BS_3 / (1 + ref3_4) + 2 * N3G));
	gpuAlloc(&tempBufferrec3[nl[n]], (1 + ref1_1)*(1 + ref3_1)* 2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G)*sizeof(double));
	tempBufferrec3_1[nl[n]] = tempBufferrec3[nl[n]];
	tempBufferrec3_2[nl[n]] = tempBufferrec3[nl[n]] + ref3_1*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	tempBufferrec3_5[nl[n]] = tempBufferrec3[nl[n]] + (ref3_1 + ref1_1)*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	tempBufferrec3_6[nl[n]] = tempBufferrec3[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_1) + 2 * N1G)*(BS_3 / (1 + ref3_1) + 2 * N3G));
	gpuAlloc(&tempBufferrec4[nl[n]], (1 + ref2_2)*(1 + ref3_2)* 2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G)*sizeof(double));
	tempBufferrec4_5[nl[n]] = tempBufferrec4[nl[n]];
	tempBufferrec4_6[nl[n]] = tempBufferrec4[nl[n]] + ref3_2*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempBufferrec4_7[nl[n]] = tempBufferrec4[nl[n]] + (ref3_2 + ref2_2)*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	tempBufferrec4_8[nl[n]] = tempBufferrec4[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 *NG * (NPR + 3)*(BS_2 / (1 + ref2_2) + 2 * N2G)*(BS_3 / (1 + ref3_2) + 2 * N3G));
	#if(N3G>0)
	gpuAlloc(&tempBufferrec5[nl[n]], (1 + ref1_6)*(1 + ref2_6)*2 * NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G)*sizeof(double));
	tempBufferrec5_1[nl[n]] = tempBufferrec5[nl[n]];
	tempBufferrec5_3[nl[n]] = tempBufferrec5[nl[n]] + ref2_6*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	tempBufferrec5_5[nl[n]] = tempBufferrec5[nl[n]] + (ref2_6 + ref1_6)*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	tempBufferrec5_7[nl[n]] = tempBufferrec5[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(2 *NG * (NPR + 3) * (BS_1 / (1 + ref1_6) + 2 * N1G)*(BS_2 / (1 + ref2_6) + 2 * N2G));
	gpuAlloc(&tempBufferrec6[nl[n]], (1 + ref1_5)*(1 + ref2_5)*2 * NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G)*sizeof(double));
	tempBufferrec6_2[nl[n]] = tempBufferrec6[nl[n]];
	tempBufferrec6_4[nl[n]] = tempBufferrec6[nl[n]] + ref2_5*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	tempBufferrec6_6[nl[n]] = tempBufferrec6[nl[n]] + (ref2_5 + ref1_5)*(2 *NG * (NPR + 3)*(BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	tempBufferrec6_8[nl[n]] = tempBufferrec6[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(2 *NG * (NPR + 3) * (BS_1 / (1 + ref1_5) + 2 * N1G)*(BS_2 / (1 + ref2_5) + 2 * N2G));
	#endif
	#endif
	gpuAlloc(&Buffersend1flux[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Buffersend2flux[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuAlloc(&Buffersend3flux[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Buffersend4flux[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Buffersend5flux[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuAlloc(&Buffersend6flux[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	gpuAlloc(&Bufferrec1flux[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec2flux[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec3flux[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec4flux[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Bufferrec5flux[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuAlloc(&Bufferrec6flux[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	Bufferrec1_3flux[nl[n]] = Bufferrec1flux[nl[n]];
	Bufferrec1_4flux[nl[n]] = Bufferrec1flux[nl[n]] + ref3_3*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec1_7flux[nl[n]] = Bufferrec1flux[nl[n]] + (ref3_3 + ref1_3)*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec1_8flux[nl[n]] = Bufferrec1flux[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec2_1flux[nl[n]] = Bufferrec2flux[nl[n]];
	Bufferrec2_2flux[nl[n]] = Bufferrec2flux[nl[n]] + ref3_4*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec2_3flux[nl[n]] = Bufferrec2flux[nl[n]] + (ref3_4 + ref2_4)*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec2_4flux[nl[n]] = Bufferrec2flux[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec3_1flux[nl[n]] = Bufferrec3flux[nl[n]];
	Bufferrec3_2flux[nl[n]] = Bufferrec3flux[nl[n]] + ref3_1*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec3_5flux[nl[n]] = Bufferrec3flux[nl[n]] + (ref3_1 + ref1_1)*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec3_6flux[nl[n]] = Bufferrec3flux[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec4_5flux[nl[n]] = Bufferrec4flux[nl[n]];
	Bufferrec4_6flux[nl[n]] = Bufferrec4flux[nl[n]] + ref3_2*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	Bufferrec4_7flux[nl[n]] = Bufferrec4flux[nl[n]] + (ref3_2 + ref2_2)*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	Bufferrec4_8flux[nl[n]] = Bufferrec4flux[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	Bufferrec5_1flux[nl[n]] = Bufferrec5flux[nl[n]];
	Bufferrec5_3flux[nl[n]] = Bufferrec5flux[nl[n]] + ref2_6*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec5_5flux[nl[n]] = Bufferrec5flux[nl[n]] + (ref2_6 + ref1_6)*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec5_7flux[nl[n]] = Bufferrec5flux[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec6_2flux[nl[n]] = Bufferrec6flux[nl[n]];
	Bufferrec6_4flux[nl[n]] = Bufferrec6flux[nl[n]] + ref2_5*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	Bufferrec6_6flux[nl[n]] = Bufferrec6flux[nl[n]] + (ref2_5 + ref1_5)*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	Bufferrec6_8flux[nl[n]] = Bufferrec6flux[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	gpuAlloc(&Bufferrec1flux1[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec2flux1[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec3flux1[nl[n]], NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuAlloc(&Bufferrec4flux1[nl[n]], NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Bufferrec5flux1[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuAlloc(&Bufferrec6flux1[nl[n]], NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	Bufferrec1_3flux1[nl[n]] = Bufferrec1flux1[nl[n]];
	Bufferrec1_4flux1[nl[n]] = Bufferrec1flux1[nl[n]] + ref3_3*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec1_7flux1[nl[n]] = Bufferrec1flux1[nl[n]] + (ref3_3 + ref1_3)*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec1_8flux1[nl[n]] = Bufferrec1flux1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3)));
	Bufferrec2_1flux1[nl[n]] = Bufferrec2flux1[nl[n]];
	Bufferrec2_2flux1[nl[n]] = Bufferrec2flux1[nl[n]] + ref3_4*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec2_3flux1[nl[n]] = Bufferrec2flux1[nl[n]] + (ref3_4 + ref2_4)*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec2_4flux1[nl[n]] = Bufferrec2flux1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(NPR*(BS_2 / (1 + ref2_4))*(BS_3 / (1 + ref3_4)));
	Bufferrec3_1flux1[nl[n]] = Bufferrec3flux1[nl[n]];
	Bufferrec3_2flux1[nl[n]] = Bufferrec3flux1[nl[n]] + ref3_1*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec3_5flux1[nl[n]] = Bufferrec3flux1[nl[n]] + (ref3_1 + ref1_1)*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec3_6flux1[nl[n]] = Bufferrec3flux1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1)));
	Bufferrec4_5flux1[nl[n]] = Bufferrec4flux1[nl[n]];
	Bufferrec4_6flux1[nl[n]] = Bufferrec4flux1[nl[n]] + ref3_2*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	Bufferrec4_7flux1[nl[n]] = Bufferrec4flux1[nl[n]] + (ref3_2 + ref2_2)*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	Bufferrec4_8flux1[nl[n]] = Bufferrec4flux1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(NPR*(BS_2 / (1 + ref2_2))*(BS_3 / (1 + ref3_2)));
	#if(N3G>0)
	Bufferrec5_1flux1[nl[n]] = Bufferrec5flux1[nl[n]];
	Bufferrec5_3flux1[nl[n]] = Bufferrec5flux1[nl[n]] + ref2_6*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec5_5flux1[nl[n]] = Bufferrec5flux1[nl[n]] + (ref2_6 + ref1_6)*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec5_7flux1[nl[n]] = Bufferrec5flux1[nl[n]] + (ref2_6 + ref1_6 + (ref2_6 && ref1_6))*(NPR*(BS_2 / (1 + ref2_6))*(BS_1 / (1 + ref1_6)));
	Bufferrec6_2flux1[nl[n]] = Bufferrec6flux1[nl[n]];
	Bufferrec6_4flux1[nl[n]] = Bufferrec6flux1[nl[n]] + ref2_5*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	Bufferrec6_6flux1[nl[n]] = Bufferrec6flux1[nl[n]] + (ref2_5 + ref1_5)*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	Bufferrec6_8flux1[nl[n]] = Bufferrec6flux1[nl[n]] + (ref2_5 + ref1_5 + (ref2_5 && ref1_5))*(NPR*(BS_2 / (1 + ref2_5))*(BS_1 / (1 + ref1_5)));
	#endif
	#if(N_LEVELS>1)
	gpuAlloc(&Bufferrec1_3flux2[nl[n]], NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3))*sizeof(double));
	gpuAlloc(&Bufferrec1_4flux2[nl[n]], NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3))*sizeof(double));
	gpuAlloc(&Bufferrec1_7flux2[nl[n]], NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3))*sizeof(double));
	gpuAlloc(&Bufferrec1_8flux2[nl[n]], NPR*(BS_1 / (1 + ref1_3))*(BS_3 / (1 + ref3_3))*sizeof(double));
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec2_1flux2[nl[n]], NPR*(BS_2 / (1 + ref2_4s))*(BS_3 / (1 + ref3_4s))*sizeof(double));
	gpuAlloc(&Bufferrec2_2flux2[nl[n]], NPR*(BS_2 / (1 + ref2_4s))*(BS_3 / (1 + ref3_4s))*sizeof(double));
	gpuAlloc(&Bufferrec2_3flux2[nl[n]], NPR*(BS_2 / (1 + ref2_4s))*(BS_3 / (1 + ref3_4s))*sizeof(double));
	gpuAlloc(&Bufferrec2_4flux2[nl[n]], NPR*(BS_2 / (1 + ref2_4s))*(BS_3 / (1 + ref3_4s))*sizeof(double));
	#endif
	gpuAlloc(&Bufferrec3_1flux2[nl[n]], NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1))*sizeof(double));
	gpuAlloc(&Bufferrec3_2flux2[nl[n]], NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1))*sizeof(double));
	gpuAlloc(&Bufferrec3_5flux2[nl[n]], NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1))*sizeof(double));
	gpuAlloc(&Bufferrec3_6flux2[nl[n]], NPR*(BS_1 / (1 + ref1_1))*(BS_3 / (1 + ref3_1))*sizeof(double));
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec4_5flux2[nl[n]], NPR*(BS_2 / (1 + ref2_2s))*(BS_3 / (1 + ref3_2s))*sizeof(double));
	gpuAlloc(&Bufferrec4_6flux2[nl[n]], NPR*(BS_2 / (1 + ref2_2s))*(BS_3 / (1 + ref3_2s))*sizeof(double));
	gpuAlloc(&Bufferrec4_7flux2[nl[n]], NPR*(BS_2 / (1 + ref2_2s))*(BS_3 / (1 + ref3_2s))*sizeof(double));
	gpuAlloc(&Bufferrec4_8flux2[nl[n]], NPR*(BS_2 / (1 + ref2_2s))*(BS_3 / (1 + ref3_2s))*sizeof(double));
	#endif
	#if(N3G>0)
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec5_1flux2[nl[n]], NPR*(BS_1 / (1 + ref1_6))*(BS_2 / (1 + ref2_6))*sizeof(double));
	gpuAlloc(&Bufferrec5_3flux2[nl[n]], NPR*(BS_1 / (1 + ref1_6))*(BS_2 / (1 + ref2_6))*sizeof(double));
	gpuAlloc(&Bufferrec5_5flux2[nl[n]], NPR*(BS_1 / (1 + ref1_6))*(BS_2 / (1 + ref2_6))*sizeof(double));
	gpuAlloc(&Bufferrec5_7flux2[nl[n]], NPR*(BS_1 / (1 + ref1_6))*(BS_2 / (1 + ref2_6))*sizeof(double));
	gpuAlloc(&Bufferrec6_2flux2[nl[n]], NPR*(BS_1 / (1 + ref1_5))*(BS_2 / (1 + ref2_5))*sizeof(double));
	gpuAlloc(&Bufferrec6_4flux2[nl[n]], NPR*(BS_1 / (1 + ref1_5))*(BS_2 / (1 + ref2_5))*sizeof(double));
	gpuAlloc(&Bufferrec6_6flux2[nl[n]], NPR*(BS_1 / (1 + ref1_5))*(BS_2 / (1 + ref2_5))*sizeof(double));
	gpuAlloc(&Bufferrec6_8flux2[nl[n]], NPR*(BS_1 / (1 + ref1_5))*(BS_2 / (1 + ref2_5))*sizeof(double));
	#endif
	#endif
	#endif

	cudaHostAlloc(&Buffersend1fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Buffersend3fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Bufferrec1fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Bufferrec3fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);

	gpuAlloc(&Buffersend1E[nl[n]], 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&Buffersend2E[nl[n]], 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&Buffersend3E[nl[n]], 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&Buffersend4E[nl[n]], 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Buffersend5E[nl[n]], 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&Buffersend6E[nl[n]], 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuAlloc(&Bufferrec1E[nl[n]], (1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec2E[nl[n]], (1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec3E[nl[n]], (1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec4E[nl[n]], (1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Bufferrec5E[nl[n]], (1 + ref2_6)*(1 + ref1_6) * 2 * (BS_1 / (1 + ref2_6) + 2 * D1)*(BS_2 / (1 + ref1_6) + 2 * D2)*sizeof(double));
	gpuAlloc(&Bufferrec6E[nl[n]], (1 + ref2_5)*(1 + ref1_5) * 2 * (BS_1 / (1 + ref2_5) + 2 * D1)*(BS_2 / (1 + ref1_5) + 2 * D2)*sizeof(double));
	#endif
	Bufferrec1_3E[nl[n]] = Bufferrec1E[nl[n]];
	Bufferrec1_4E[nl[n]] = Bufferrec1E[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec1_7E[nl[n]] = Bufferrec1E[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec1_8E[nl[n]] = Bufferrec1E[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec2_1E[nl[n]] = Bufferrec2E[nl[n]];
	Bufferrec2_2E[nl[n]] = Bufferrec2E[nl[n]] + ref3_4*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec2_3E[nl[n]] = Bufferrec2E[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec2_4E[nl[n]] = Bufferrec2E[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec3_1E[nl[n]] = Bufferrec3E[nl[n]];
	Bufferrec3_2E[nl[n]] = Bufferrec3E[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec3_5E[nl[n]] = Bufferrec3E[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec3_6E[nl[n]] = Bufferrec3E[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec4_5E[nl[n]] = Bufferrec4E[nl[n]];
	Bufferrec4_6E[nl[n]] = Bufferrec4E[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	Bufferrec4_7E[nl[n]] = Bufferrec4E[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	Bufferrec4_8E[nl[n]] = Bufferrec4E[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	Bufferrec5_1E[nl[n]] = Bufferrec5E[nl[n]];
	Bufferrec5_3E[nl[n]] = Bufferrec5E[nl[n]] + ref1_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec5_5E[nl[n]] = Bufferrec5E[nl[n]] + (ref1_6 + ref2_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec5_7E[nl[n]] = Bufferrec5E[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec6_2E[nl[n]] = Bufferrec6E[nl[n]];
	Bufferrec6_4E[nl[n]] = Bufferrec6E[nl[n]] + ref1_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	Bufferrec6_6E[nl[n]] = Bufferrec6E[nl[n]] + (ref1_5 + ref2_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	Bufferrec6_8E[nl[n]] = Bufferrec6E[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	gpuAlloc(&Bufferrec1E1[nl[n]], (1 + ref1_3)*(1 + ref3_3) * 2 * (BS_1 / (1 + ref1_3) + 2 * D1)*(BS_3 / (1 + ref3_3) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec2E1[nl[n]], (1 + ref2_4)*(1 + ref3_4) * 2 * (BS_2 / (1 + ref2_4) + 2 * D2)*(BS_3 / (1 + ref3_4) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec3E1[nl[n]], (1 + ref1_1)*(1 + ref3_1) * 2 * (BS_1 / (1 + ref1_1) + 2 * D1)*(BS_3 / (1 + ref3_1) + 2 * D3)*sizeof(double));
	gpuAlloc(&Bufferrec4E1[nl[n]], (1 + ref2_2)*(1 + ref3_2) * 2 * (BS_2 / (1 + ref2_2) + 2 * D2)*(BS_3 / (1 + ref3_2) + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&Bufferrec5E1[nl[n]], (1 + ref2_6)*(1 + ref1_6) * 2 * (BS_1 / (1 + ref2_6) + 2 * D1)*(BS_2 / (1 + ref1_6) + 2 * D2)*sizeof(double));
	gpuAlloc(&Bufferrec6E1[nl[n]], (1 + ref2_5)*(1 + ref1_5) * 2 * (BS_1 / (1 + ref2_5) + 2 * D1)*(BS_2 / (1 + ref1_5) + 2 * D2)*sizeof(double));
	#endif
	Bufferrec1_3E1[nl[n]] = Bufferrec1E1[nl[n]];
	Bufferrec1_4E1[nl[n]] = Bufferrec1E1[nl[n]] + ref3_3*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec1_7E1[nl[n]] = Bufferrec1E1[nl[n]] + (ref3_3 + ref1_3)*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec1_8E1[nl[n]] = Bufferrec1E1[nl[n]] + (ref3_3 + ref1_3 + (ref3_3 && ref1_3))*(2 * ((BS_1 / (1 + ref1_3) + 2 * D1))*((BS_3 / (1 + ref3_3) + 2 * D3)));
	Bufferrec2_1E1[nl[n]] = Bufferrec2E1[nl[n]];
	Bufferrec2_2E1[nl[n]] = Bufferrec2E1[nl[n]] + ref3_4*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec2_3E1[nl[n]] = Bufferrec2E1[nl[n]] + (ref3_4 + ref2_4)*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec2_4E1[nl[n]] = Bufferrec2E1[nl[n]] + (ref3_4 + ref2_4 + (ref3_4 && ref2_4))*(2 * ((BS_2 / (1 + ref2_4) + 2 * D2))*((BS_3 / (1 + ref3_4) + 2 * D3)));
	Bufferrec3_1E1[nl[n]] = Bufferrec3E1[nl[n]];
	Bufferrec3_2E1[nl[n]] = Bufferrec3E1[nl[n]] + ref3_1*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec3_5E1[nl[n]] = Bufferrec3E1[nl[n]] + (ref3_1 + ref1_1)*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec3_6E1[nl[n]] = Bufferrec3E1[nl[n]] + (ref3_1 + ref1_1 + (ref3_1 && ref1_1))*(2 * ((BS_1 / (1 + ref1_1) + 2 * D1))*((BS_3 / (1 + ref3_1) + 2 * D3)));
	Bufferrec4_5E1[nl[n]] = Bufferrec4E1[nl[n]];
	Bufferrec4_6E1[nl[n]] = Bufferrec4E1[nl[n]] + ref3_2*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	Bufferrec4_7E1[nl[n]] = Bufferrec4E1[nl[n]] + (ref3_2 + ref2_2)*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	Bufferrec4_8E1[nl[n]] = Bufferrec4E1[nl[n]] + (ref3_2 + ref2_2 + (ref3_2 && ref2_2))*(2 * ((BS_2 / (1 + ref2_2) + 2 * D2))*((BS_3 / (1 + ref3_2) + 2 * D3)));
	#if(N3G>0)
	Bufferrec5_1E1[nl[n]] = Bufferrec5E1[nl[n]];
	Bufferrec5_3E1[nl[n]] = Bufferrec5E1[nl[n]] + ref1_6*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec5_5E1[nl[n]] = Bufferrec5E1[nl[n]] + (ref1_6 + ref2_6)*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec5_7E1[nl[n]] = Bufferrec5E1[nl[n]] + (ref1_6 + ref2_6 + (ref1_6 && ref2_6))*(2 * ((BS_2 / (1 + ref2_6) + 2 * D2))*((BS_1 / (1 + ref1_6) + 2 * D1)));
	Bufferrec6_2E1[nl[n]] = Bufferrec6E1[nl[n]];
	Bufferrec6_4E1[nl[n]] = Bufferrec6E1[nl[n]] + ref1_5*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	Bufferrec6_6E1[nl[n]] = Bufferrec6E1[nl[n]] + (ref1_5 + ref2_5)*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	Bufferrec6_8E1[nl[n]] = Bufferrec6E1[nl[n]] + (ref1_5 + ref2_5 + (ref1_5 && ref2_5))*(2 * ((BS_2 / (1 + ref2_5) + 2 * D2))*((BS_1 / (1 + ref1_5) + 2 * D1)));
	#endif
	#if(N_LEVELS>1)
	gpuAlloc(&Bufferrec1_3E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_3) *(BS_3 + 2 * D3) / (1 + ref3_3) *sizeof(double));
	if (ref3_3)gpuAlloc(&Bufferrec1_4E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_3) *(BS_3 + 2 * D3) / (1 + ref3_3) *sizeof(double));
	if (ref1_3)gpuAlloc(&Bufferrec1_7E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_3) *(BS_3 + 2 * D3) / (1 + ref3_3) *sizeof(double));
	if (ref1_3 && ref3_3)gpuAlloc(&Bufferrec1_8E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_3) *(BS_3 + 2 * D3) / (1 + ref3_3) *sizeof(double));
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec2_1E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_4s) *(BS_3 + 2 * D3) / (1 + ref3_4s) *sizeof(double));
	if (MY_MAX(ref3_4, ref3_4s))gpuAlloc(&Bufferrec2_2E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_4s) *(BS_3 + 2 * D3) / (1 + ref3_4s) *sizeof(double));
	if (MY_MAX(ref2_4, ref2_4s))gpuAlloc(&Bufferrec2_3E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_4s) *(BS_3 + 2 * D3) / (1 + ref3_4s) *sizeof(double));
	if (MY_MAX(ref2_4, ref2_4s) && MY_MAX(ref3_4, ref3_4s))gpuAlloc(&Bufferrec2_4E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_4s) *(BS_3 + 2 * D3) / (1 + ref3_4s) *sizeof(double));
	#endif
	gpuAlloc(&Bufferrec3_1E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_1) *(BS_3 + 2 * D3) / (1 + ref3_1) *sizeof(double));
	if (ref3_1)gpuAlloc(&Bufferrec3_2E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_1) *(BS_3 + 2 * D3) / (1 + ref3_1) *sizeof(double));
	if (ref1_1)gpuAlloc(&Bufferrec3_5E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_1) *(BS_3 + 2 * D3) / (1 + ref3_1) *sizeof(double));
	if (ref1_1 && ref3_1)gpuAlloc(&Bufferrec3_6E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_1) *(BS_3 + 2 * D3) / (1 + ref3_1) *sizeof(double));
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec4_5E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_2s) *(BS_3 + 2 * D3) / (1 + ref3_2s) *sizeof(double));
	if (MY_MAX(ref3_2,ref3_2s))gpuAlloc(&Bufferrec4_6E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_2s) *(BS_3 + 2 * D3) / (1 + ref3_2s) *sizeof(double));
	if (MY_MAX(ref2_2,ref2_2s))gpuAlloc(&Bufferrec4_7E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_2s) *(BS_3 + 2 * D3) / (1 + ref3_2s) *sizeof(double));
	if (MY_MAX(ref2_2,ref2_2s) && MY_MAX(ref3_2,ref3_2s))gpuAlloc(&Bufferrec4_8E2[nl[n]], 2 * (BS_2 + 2 * D2) / (1 + ref2_2s) *(BS_3 + 2 * D3) / (1 + ref3_2s) *sizeof(double));
	#endif
	#if(N3G>0)
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	gpuAlloc(&Bufferrec5_1E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_6) *(BS_2 + 2 * D2) / (1 + ref2_6) *sizeof(double));
	if (ref2_6)gpuAlloc(&Bufferrec5_3E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_6) *(BS_2 + 2 * D2) / (1 + ref2_6) *sizeof(double));
	if (ref1_6)gpuAlloc(&Bufferrec5_5E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_6) *(BS_2 + 2 * D2) / (1 + ref2_6) *sizeof(double));
	if (ref1_6 && ref2_6)gpuAlloc(&Bufferrec5_7E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_6) *(BS_2 + 2 * D2) / (1 + ref2_6) *sizeof(double));
	gpuAlloc(&Bufferrec6_2E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_5) *(BS_2 + 2 * D2) / (1 + ref2_5) *sizeof(double));
	if (ref2_5)gpuAlloc(&Bufferrec6_4E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_5) *(BS_2 + 2 * D2) / (1 + ref2_5) *sizeof(double));
	if (ref1_5)gpuAlloc(&Bufferrec6_6E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_5) *(BS_2 + 2 * D2) / (1 + ref2_5) *sizeof(double));
	if (ref1_5 && ref2_5)gpuAlloc(&Bufferrec6_8E2[nl[n]], 2 * (BS_1 + 2 * D1) / (1 + ref1_5) *(BS_2 + 2 * D2) / (1 + ref2_5) *sizeof(double));
	#endif
	#endif
	#endif
	#if(N3G>0)
	gpuAlloc(&BuffersendE1corn9[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BuffersendE1corn10[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BuffersendE1corn11[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BuffersendE1corn12[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BuffersendE2corn5[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BuffersendE2corn6[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BuffersendE2corn7[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BuffersendE2corn8[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuAlloc(&BuffersendE3corn1[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BuffersendE3corn2[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BuffersendE3corn3[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BuffersendE3corn4[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuAlloc(&BufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&BufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&BufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuAlloc(&BufferrecE3corn1[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BufferrecE3corn2[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BufferrecE3corn3[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BufferrecE3corn4[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	gpuAlloc(&BufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	#endif
	gpuAlloc(&BufferrecE3corn1_3[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn1_4[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn2_1[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn2_2[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn3_5[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn3_6[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn4_7[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	gpuAlloc(&BufferrecE3corn4_8[nl[n]], 1 * (BS_3 + N3G) *sizeof(double));
	#endif
	#if(N3G>0)
	gpuAlloc(&tempBufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&tempBufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&tempBufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&tempBufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuAlloc(&tempBufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&tempBufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&tempBufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuAlloc(&tempBufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuAlloc(&tempBufferrecE3corn1[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&tempBufferrecE3corn2[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&tempBufferrecE3corn3[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&tempBufferrecE3corn4[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	gpuAlloc(&tempBufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&tempBufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	#endif
	gpuAlloc(&tempBufferrecE3corn1_3[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn1_4[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn2_1[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn2_2[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn3_5[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn3_6[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn4_7[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&tempBufferrecE3corn4_8[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	#if(N3G>0)
	gpuAlloc(&BufferrecE1corn9_32[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn9_72[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn10_12[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn10_52[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn11_22[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn11_62[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn12_42[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE1corn12_82[nl[n]], 1 * (BS_1 + 2 * D1) *sizeof(double));
	gpuAlloc(&BufferrecE2corn5_22[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn5_42[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn6_12[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn6_32[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn7_52[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn7_72[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn8_62[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	gpuAlloc(&BufferrecE2corn8_82[nl[n]], 1 * (BS_2 + 2 * D2) *sizeof(double));
	#endif
	gpuAlloc(&BufferrecE3corn1_32[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&BufferrecE3corn1_42[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BufferrecE3corn2_12[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&BufferrecE3corn2_22[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&BufferrecE3corn3_52[nl[n]], 1 * (BS_3 + 2 * D3)  *sizeof(double));
	gpuAlloc(&BufferrecE3corn3_62[nl[n]], 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuAlloc(&BufferrecE3corn4_72[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	gpuAlloc(&BufferrecE3corn4_82[nl[n]], 1 * (BS_3 + 2 * D3) *sizeof(double));
	#endif

}
double check = 1.0;

void GPU_write(int n)
{
	int i, j, z, k;
	double radius_GPU[(N1 + 2 * N1G)], r, th, phi, X[NDIM];
	//cudasetDevice(block[n][AMR_GPU]);

	for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + BS_1 + N1G; i++){
		coord(n, i, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		radius_GPU[(i - N1_GPU_offset[n] + N1G)] = r;
		Katm_GPU[nl[n]][i - N1_GPU_offset[n] + N1G] = Katm[nl[n]][i - N1_GPU_offset[n] + N1G];
	}

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = p[nl[n]][index_3D(n, i, j, z)][k];
				ph_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[nl[n]][index_3D(n, i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[nl[n]][index_3D(n, i, j, z)][k];
				#endif
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[nl[n]][index_3D(n, i, j, z)][k];
				psh_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = psh[nl[n]][index_3D(n, i, j, z)][k];

			}
			#endif
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
			}
			//pflag_GPU[nl[n]][(i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
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
	status = cudaMemcpyAsync(BufferdU[nl[n]], dU_GPU[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	/*Initialize memory items that have to be passed on to the GPU*/
	cudaMemcpyAsync(Bufferp_1[nl[n]], p_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), cudaMemcpyHostToDevice,commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferph_1[nl[n]], p_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#if(STAGGERED)
	cudaMemcpyAsync(Bufferps_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferpsh_1[nl[n]], ps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif
	//cudaMemcpyAsync(Bufferpflag[nl[n]], pflag_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int), cudaMemcpyHostToDevice,commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferfailimage[nl[n]], failimage_GPU[nl[n]], NFAIL*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(BufferKatm[nl[n]], Katm_GPU[nl[n]], (BS_1 + 2 * N1G)*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferradius[nl[n]], radius_GPU, (BS_1 + 2 * N1G)*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);

	/*Copy metric to GPU*/
	int pg;
	#pragma omp parallel private(i, j, z, pg)
	{
		#if(!NSY)
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]){
		#else
		#pragma omp for collapse(2) schedule(dynamic)
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
	cudaMemcpyAsync(Buffergdet[nl[n]], gdet_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcov[nl[n]], gcov_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcon[nl[n]], gcon_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferconn[nl[n]], conn_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#else
	cudaMemcpyAsync(Buffergdet[nl[n]], gdet_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcov[nl[n]], gcov_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcon[nl[n]], gcon_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferconn[nl[n]], conn_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#endif

	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in GPU_write: %d \n", status);
}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	/*Set arguments of kernel*/
	//cudasetDevice(block[n][AMR_GPU]);

	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;

	if (dir == 1){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
	}
	else if (dir == 2){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
	}
	else{
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]],  block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error Fluxcalcprep %d \n", status);
}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	//cudasetDevice(block[n][AMR_GPU]);

	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) / LOCAL_WORK_SIZE;
	
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),  
				dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
	}
	else{
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Fluxcalc2D2 %d\n", status);
}

void GPU_reconstruct_internal(int flag, int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	if ((block[n][AMR_POLE] != 0) || (block[n][AMR_NBR1] < 0) || (block[n][AMR_NBR3] < 0)){
		int nr_workgroups_local[1];
		nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)* (BS_2)* (BS_3)) % LOCAL_WORK_SIZE) + (BS_1)* (BS_2)* (BS_3)) / LOCAL_WORK_SIZE;

		/*Calculate reconstructed left state*/
		if (flag == 1){
			reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
		else{
			reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1] < 0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3] < 0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error GPU_reconstruct_internal %d\n", status);
	}
}

/*Start reading timestep from GPU*/
void read_time_GPU(void){
	//int n;
	//for (n = 0; n < n_active; n++){
	//	if (prestep_full[nl[n_ord[n]]] == 1){
	//		cudaMemcpyAsync(dtij_GPU[n_ord[n]], Bufferdtij[n_ord[n]], (int)((nr_workgroups[n_ord[n]]) * sizeof(double)), cudaMemcpyDeviceToHost, commandQueueGPU[n_ord[n]]);
	//	}
	//}
}

/*Do last step of reduction of timestep on CPU*/
double fluxcalc_GPU(int n, int dir)
{
	double ndt;
	int y;
	ndt = 1.e9;
	int nr;
	if (dir == 1) nr = nr_workgroups2_1[nl[n]];
	else if (dir == 2) nr = nr_workgroups2_2[nl[n]];
	else if (dir == 3) nr = nr_workgroups2_3[nl[n]];
	//cudasetDevice(block[n][AMR_GPU]);
	cudaStreamSynchronize(commandQueueGPU[nl[n]]);
	status = cudaGetLastError();
	if (status != 0) fprintf(stderr, "Error fluxcalc_GPU %d\n", status);
	for (y = 0; y < nr; y++){
		if (dtij_GPU[nl[n]][y] < ndt && dtij_GPU[nl[n]][y] < 1.e9 && dtij_GPU[nl[n]][y] > 1.e-6){
			ndt = dtij_GPU[nl[n]][y];
		}
	}
	return(ndt);
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	//cudasetDevice(block[n][AMR_GPU]);
	 fix_flux << < nr_workgroups_special[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error fixflux %d \n", status);
}

void GPU_consttransport_bound(void){
	int n, flag;
	
	gpu = 1;
	#if(TRANS_BOUND)
	E_average();
	#endif
	set_iprobe(0, &flag);

	#if(PRESTEP)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);

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
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		
		#if(!TIMESTEP_JET)
		#if(N3G>0)
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
	}

	//For first timestep do not synchronize electrice fields 
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);

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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_send1(E_corn, BufferE_1, n_ord[n]);
	}
	do{
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			E_rec1(E_corn, BufferE_1, n_ord[n], 1);
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
	}
	do{
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			E_rec2(E_corn, BufferE_1, n_ord[n], 1);
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
	}
	#if(!TIMESTEP_JET)
	#if(N3G>0)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_send3(E_corn, BufferE_1, n_ord[n]);
	}
	do{
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			E_rec3(E_corn, BufferE_1, n_ord[n], 1);
		}
		set_iprobe(1, &flag);
	} while (flag);
	set_iprobe(0, &flag);

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}	
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	#endif
}

void GPU_consttransport1(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) % LOCAL_WORK_SIZE) + (BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) / LOCAL_WORK_SIZE;
	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
	}
	else{
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;

	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport2 %d \n", status);
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;

	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], BufferE_1[nl[n]], Dt, block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status )fprintf(stderr, "Error constransport3 %d \n", status);
}

void GPU_consttransport3_post(double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) / LOCAL_WORK_SIZE;
	
	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	consttransport3_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt);
	
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport3_post %d \n", status);
}

void GPU_flux_ct1(int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct1 << < nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct2 << < nr_workgroups3[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	 //cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct2 %d\n", status);
}

void GPU_Utoprim(int flag, int n, double Dt)
{
}

void GPU_fixup(int flag, int n, double Dt)
{
	int nr_workgroups_local[1];
	//cudasetDevice(block[n][AMR_GPU]);
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)*(BS_2)*(BS_3)) % LOCAL_WORK_SIZE) + (BS_1)*(BS_2)*(BS_3)) / LOCAL_WORK_SIZE;

	if (flag == 0){
		fixup << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferstorage2[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag, block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		fixup << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferstorage2[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag, block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fixup %d\n", status);
}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)*(BS_2)*(BS_3)) % LOCAL_WORK_SIZE) + (BS_1)*(BS_2)*(BS_3)) / LOCAL_WORK_SIZE;
	//cudasetDevice(block[n][AMR_GPU]);

	if (flag == 1){
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
	}
	else{
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_cleanup_post(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)*(BS_2)*(BS_3)) % LOCAL_WORK_SIZE) + (BS_1)*(BS_2)*(BS_3)) / LOCAL_WORK_SIZE;
	//cudasetDevice(block[n][AMR_GPU]);
	cleanup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], BufferE_1[nl[n]]);
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error cleanup_post %d\n", status);
}

void GPU_fixup_post(int n, double Dt)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2) % LOCAL_WORK_SIZE) + 2 * (BS_2)*(BS_3) + (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2)) / LOCAL_WORK_SIZE;
	//cudasetDevice(block[n][AMR_GPU]);
	fixup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, 1);
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error fixup_post %d\n", status);
}

void GPU_boundprim(int bound_force)
{
	int n, flag;
	int temp = nstep;
	gpu = 1;

	if (bound_force == 1) nstep = -1;
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim1(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim2(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	//For last timestep do not receive synchronized electrice fields 
	cudaDeviceSynchronize();
	mpi_synch();
	if (rank == 0) begin2 = get_wall_time();

	#if(PRESTEP)
	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		set_iprobe(0, &flag);
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
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
			//#pragma omp parallel for schedule(dynamic,1) private(n,status)
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				//cudasetDevice(block[n_ord[n]][AMR_GPU]);
				flux_rec1(F1, BufferF1_1, n_ord[n], 1);
				flux_rec2(F2, BufferF2_1, n_ord[n], 1);
				flux_rec3(F3, BufferF3_1, n_ord[n], 1);
			}
			set_iprobe(1, &flag);
		}while(flag);
		set_iprobe(0, &flag);
		do{
			//#pragma omp parallel for schedule(dynamic,1) private(n,status)
			for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
				//cudasetDevice(block[n_ord[n]][AMR_GPU]);
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
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
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

	rc = 0;
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do{
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudasetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	set_iprobe(0, &flag);
	do{
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
		set_iprobe(1, &flag);
	} while (flag);
	if (N3 > 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			//cudasetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		set_iprobe(0, &flag);
		do {
			//#pragma omp parallel for schedule(dynamic,1) private(n,status)
			for (n = 0; n < n_active; n++){
				//cudasetDevice(block[n_ord[n]][AMR_GPU]);
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
			}
			set_iprobe(1, &flag);
		} while (flag);
	}

	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");

	#if(TRANS_BOUND && NB_3==1)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim_trans(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim_trans(0, n_ord[n]);
	}
	#endif

	//MPI communication
	//cudaDeviceSynchronize();
	mpi_synch();

	if (rank == 0){
		end2 = get_wall_time();
		time_spent3 += (double)(end2 - begin2);
	}

	nstep = temp;
}

void GPU_boundprim1(int flag, int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			 boundprim1 << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]);
		}
		else{
			 boundprim1 << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status ) fprintf(stderr, "Error boundprim1 %d\n", status);
	}
}

void GPU_boundprim2(int flag, int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			 boundprim2 << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[nl[n]]);
		}
		else{
			 boundprim2 << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error boundprim2.1 %d\n", status);
	}
}

void GPU_boundprim_trans(int flag, int n)
{
	//cudasetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_POLE] != 0 ){
		if (flag == 0){
			boundprim_trans << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferpsh_1[nl[n]]);
		}
		else{
			boundprim_trans << < nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error boundprim2.1 %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k;
	int nr_workgroups_local[1];

	//cudaDeviceSynchronize();
	//cudasetDevice(block[n][AMR_GPU]);

	#if(N_LEVELS_1D_INT>10)
	/*Calculate gradients for reconstruction*/
	GPU_fluxcalcprep(3, 1, 1, n);
	
	/*Reconstruct derefined region around pole for Ray-Tracing and store in temporary variable (used for fluxes normally)*/
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1)* (BS_2)* (BS_3)) % LOCAL_WORK_SIZE) + (BS_1)* (BS_2)* (BS_3)) / LOCAL_WORK_SIZE;
	cudaMemcpyAsync(BufferF1_1[nl[n]], Bufferp_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToDevice, commandQueueGPU[nl[n]]);
	reconstruct_internal << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	cudaMemcpyAsync(p_1[nl[n]], Bufferph_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#else
	cudaMemcpyAsync(p_1[nl[n]], Bufferp_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	cudaMemcpyAsync(ph_1[nl[n]], Bufferph_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#if(STAGGERED)
	cudaMemcpyAsync(ps_1[nl[n]], Bufferps_1[nl[n]], (int)(3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(psh_1[nl[n]], Bufferpsh_1[nl[n]], (int)(3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	#endif
	cudaMemcpyAsync(failimage_GPU[nl[n]], Bufferfailimage[nl[n]], (int)((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
	cudaDeviceSynchronize();

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + BS_3 - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[nl[n]][index_3D(n, i, j, z)][k] = p_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				ph[nl[n]][index_3D(n, i, j, z)][k] = ph_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			for (k = 0; k < NFAIL; k++){
				failimage[nl[n]][index_3D(n, i, j, z)][k] = failimage_GPU[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps[nl[n]][index_3D(n, i, j, z)][k] = ps_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				psh[nl[n]][index_3D(n, i, j, z)][k] = psh_1[nl[n]][(k - 1) * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#endif
		}
	}
	status = cudaGetLastError();
	if (cudaSuccess != status )fprintf(stderr, "Error in GPU_read: %d \n", status);
}

void GPU_finish(int n, int force_delete)
{
	int i;

	//Tell code no GPU
	block[n][AMR_GPU] = -1;
	if (mem_spot[nl[n]] == 1) fprintf(stderr, "Error, tries to deallocate GPU memory before deaalocating RAM! \n");
	if (mem_spot[nl[n]] == 0 && force_delete==0){
		free_bound_gpu(n);
		return;
	}
	else if (mem_spot_gpu[nl[n]] == -1){
		fprintf(stderr, "Error, tries to deallocate empty GPU memory! \n");
		return;
	}
	else{
		//Select correct CUDA device
		//cudasetDevice(mem_spot_gpu[nl[n]]);
		
		//Tell the code that memory is deallocated on the GPU
		mem_spot_gpu[nl[n]] = -1;
	}

	//Make sure all events are finished
	for (i = 0; i < 600; i++) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent[nl[n]][i], 0);
	for (i = 0; i < 100; i++) cudaStreamWaitEvent(commandQueueGPU[nl[n]], boundevent1[nl[n]][i], 0);

	//Destroy CUDA events associated with block
	for (i = 0; i < 600; i++) cudaEventDestroy(boundevent[nl[n]][i]);
	for (i = 0; i < 100; i++) cudaEventDestroy(boundevent1[nl[n]][i]);
	cudaStreamDestroy(commandQueueGPU[nl[n]]);

	free(p_1[nl[n]]);
	#if(STAGGERED)
	free(ps_1[nl[n]]);
	free(psh_1[nl[n]]);
	#endif
	free(ph_1[nl[n]]);
	//free(dU_GPU[nl[n]]);
	//free(pflag_GPU[nl[n]]);
	free(failimage_GPU[nl[n]]);
	free(Katm_GPU[nl[n]]);
	free(dq_1[nl[n]]);
	free(gcov_GPU[nl[n]]);
	free(gcon_GPU[nl[n]]);
	free(conn_GPU[nl[n]]);
	free(gdet_GPU[nl[n]]);

	status += cudaFreeHost(dtij_GPU[nl[n]]);
	//status += cudaFree(Bufferdtij[nl[n]]);
	status += cudaFree(BufferF1_1[nl[n]]);
	status += cudaFree(BufferF2_1[nl[n]]);
	status += cudaFree(BufferF3_1[nl[n]]);
	status += cudaFree(Bufferdq_1[nl[n]]);
	status += cudaFree(BufferE_1[nl[n]]);
	#if(LEER)
	status += cudaFree(BufferV[nl[n]]);
	#endif
	status += cudaFree(Bufferradius[nl[n]]);
	status += cudaFree(Bufferstorage1[nl[n]]);
	#if(N_LEVELS_1D_INT>0)
	status += cudaFree(Bufferstorage2[nl[n]]);
	status += cudaFree(Bufferstorage3[nl[n]]);
	#endif
	status += cudaFree(Bufferp_1[nl[n]]);
	status += cudaFree(Bufferph_1[nl[n]]);
	#if(STAGGERED)
	status += cudaFree(Bufferps_1[nl[n]]);
	status += cudaFree(Bufferpsh_1[nl[n]]);
	#endif
	status += cudaFree(Bufferpflag[nl[n]]);
	status += cudaFree(Bufferfailimage[nl[n]]);
	status += cudaFree(BufferKatm[nl[n]]);
	//status += cudaFree(BufferdU[nl[n]]);
	status += cudaFree(Buffergcov[nl[n]]);
	status += cudaFree(Buffergcon[nl[n]]);
	status += cudaFree(Bufferconn[nl[n]]);
	status += cudaFree(Buffergdet[nl[n]]);
	
	free_bound_gpu(n);

	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in GPU_finish_1: %d \n", status);
}

void free_bound_gpu(int n){
	//status += gpuFree(NULL_POINTER[nl[n]]);
	status += gpuFree(Buffersend1[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend1_3[nl[n]]);
	status += gpuFree(Buffersend1_4[nl[n]]);
	status += gpuFree(Buffersend1_7[nl[n]]);
	status += gpuFree(Buffersend1_8[nl[n]]);
	#endif
	status += gpuFree(Buffersend2[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend2_1[nl[n]]);
	status += gpuFree(Buffersend2_2[nl[n]]);
	status += gpuFree(Buffersend2_3[nl[n]]);
	status += gpuFree(Buffersend2_4[nl[n]]);
	#endif
	status += gpuFree(Buffersend3[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend3_1[nl[n]]);
	status += gpuFree(Buffersend3_2[nl[n]]);
	status += gpuFree(Buffersend3_5[nl[n]]);
	status += gpuFree(Buffersend3_6[nl[n]]);
	#endif
	status += gpuFree(Buffersend4[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend4_5[nl[n]]);
	status += gpuFree(Buffersend4_6[nl[n]]);
	status += gpuFree(Buffersend4_7[nl[n]]);
	status += gpuFree(Buffersend4_8[nl[n]]);
	#endif
	#if(N3G>0)
	status += gpuFree(Buffersend5[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend5_1[nl[n]]);
	status += gpuFree(Buffersend5_3[nl[n]]);
	status += gpuFree(Buffersend5_5[nl[n]]);
	status += gpuFree(Buffersend5_7[nl[n]]);
	#endif
	status += gpuFree(Buffersend6[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Buffersend6_2[nl[n]]);
	status += gpuFree(Buffersend6_4[nl[n]]);
	status += gpuFree(Buffersend6_6[nl[n]]);
	status += gpuFree(Buffersend6_8[nl[n]]);
	#endif
	#endif
	status += gpuFree(Bufferrec1[nl[n]]);
	status += gpuFree(Bufferrec2[nl[n]]);
	status += gpuFree(Bufferrec3[nl[n]]);
	status += gpuFree(Bufferrec4[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Bufferrec5[nl[n]]);
	status += gpuFree(Bufferrec6[nl[n]]);
	#endif
	#if(PRESTEP || PRESTEP2)
	status += gpuFree(tempBufferrec1[nl[n]]);
	status += gpuFree(tempBufferrec2[nl[n]]);
	status += gpuFree(tempBufferrec3[nl[n]]);
	status += gpuFree(tempBufferrec4[nl[n]]);
	#if(N3G>0)
	status += gpuFree(tempBufferrec5[nl[n]]);
	status += gpuFree(tempBufferrec6[nl[n]]);
	#endif
	#endif	
	status += gpuFree(Buffersend1flux[nl[n]]);
	status += gpuFree(Buffersend2flux[nl[n]]);
	status += gpuFree(Buffersend3flux[nl[n]]);
	status += gpuFree(Buffersend4flux[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Buffersend5flux[nl[n]]);
	status += gpuFree(Buffersend6flux[nl[n]]);
	#endif
	status += gpuFree(Bufferrec1flux[nl[n]]);
	status += gpuFree(Bufferrec2flux[nl[n]]);
	status += gpuFree(Bufferrec3flux[nl[n]]);
	status += gpuFree(Bufferrec4flux[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Bufferrec5flux[nl[n]]);
	status += gpuFree(Bufferrec6flux[nl[n]]);
	#endif

	status += gpuFree(Bufferrec1flux1[nl[n]]);
	status += gpuFree(Bufferrec2flux1[nl[n]]);
	status += gpuFree(Bufferrec3flux1[nl[n]]);
	status += gpuFree(Bufferrec4flux1[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Bufferrec5flux1[nl[n]]);
	status += gpuFree(Bufferrec6flux1[nl[n]]);
	#endif
	status += gpuFree(Bufferrec1_3flux2[nl[n]]);
	#if(N_LEVELS>1)
	status += gpuFree(Bufferrec1_4flux2[nl[n]]);
	status += gpuFree(Bufferrec1_7flux2[nl[n]]);
	status += gpuFree(Bufferrec1_8flux2[nl[n]]);
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec2_1flux2[nl[n]]);
	status += gpuFree(Bufferrec2_2flux2[nl[n]]);
	status += gpuFree(Bufferrec2_3flux2[nl[n]]);
	status += gpuFree(Bufferrec2_4flux2[nl[n]]);
	#endif
	status += gpuFree(Bufferrec3_1flux2[nl[n]]);
	status += gpuFree(Bufferrec3_2flux2[nl[n]]);
	status += gpuFree(Bufferrec3_5flux2[nl[n]]);
	status += gpuFree(Bufferrec3_6flux2[nl[n]]);
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec4_5flux2[nl[n]]);
	status += gpuFree(Bufferrec4_6flux2[nl[n]]);
	status += gpuFree(Bufferrec4_7flux2[nl[n]]);
	status += gpuFree(Bufferrec4_8flux2[nl[n]]);
	#endif
	#if(N3G>0)
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec5_1flux2[nl[n]]);
	status += gpuFree(Bufferrec5_3flux2[nl[n]]);
	status += gpuFree(Bufferrec5_5flux2[nl[n]]);
	status += gpuFree(Bufferrec5_7flux2[nl[n]]);
	status += gpuFree(Bufferrec6_2flux2[nl[n]]);
	status += gpuFree(Bufferrec6_4flux2[nl[n]]);
	status += gpuFree(Bufferrec6_6flux2[nl[n]]);
	status += gpuFree(Bufferrec6_8flux2[nl[n]]);
	#endif
	#endif
	#endif
	status += cudaFreeHost(Buffersend1fine[nl[n]]);
	status += cudaFreeHost(Buffersend3fine[nl[n]]);
	status += cudaFreeHost(Bufferrec1fine[nl[n]]);
	status += cudaFreeHost(Bufferrec3fine[nl[n]]);

	status += gpuFree(Buffersend1E[nl[n]]);
	status += gpuFree(Buffersend2E[nl[n]]);
	status += gpuFree(Buffersend3E[nl[n]]);
	status += gpuFree(Buffersend4E[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Buffersend5E[nl[n]]);
	status += gpuFree(Buffersend6E[nl[n]]);
	#endif
	status += gpuFree(Bufferrec1E[nl[n]]);
	status += gpuFree(Bufferrec2E[nl[n]]);
	status += gpuFree(Bufferrec3E[nl[n]]);
	status += gpuFree(Bufferrec4E[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Bufferrec5E[nl[n]]);
	status += gpuFree(Bufferrec6E[nl[n]]);
	#endif
	status += gpuFree(Bufferrec1E1[nl[n]]);
	status += gpuFree(Bufferrec2E1[nl[n]]);
	status += gpuFree(Bufferrec3E1[nl[n]]);
	status += gpuFree(Bufferrec4E1[nl[n]]);
	#if(N3G>0)
	status += gpuFree(Bufferrec5E1[nl[n]]);
	status += gpuFree(Bufferrec6E1[nl[n]]);
	#endif
	#if(N_LEVELS>1)
	status += gpuFree(Bufferrec1_3E2[nl[n]]);
	status += gpuFree(Bufferrec1_4E2[nl[n]]);
	status += gpuFree(Bufferrec1_7E2[nl[n]]);
	status += gpuFree(Bufferrec1_8E2[nl[n]]);
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec2_1E2[nl[n]]);
	status += gpuFree(Bufferrec2_2E2[nl[n]]);
	status += gpuFree(Bufferrec2_3E2[nl[n]]);
	status += gpuFree(Bufferrec2_4E2[nl[n]]);
	#endif
	status += gpuFree(Bufferrec3_1E2[nl[n]]);
	status += gpuFree(Bufferrec3_2E2[nl[n]]);
	status += gpuFree(Bufferrec3_5E2[nl[n]]);
	status += gpuFree(Bufferrec3_6E2[nl[n]]);
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec4_5E2[nl[n]]);
	status += gpuFree(Bufferrec4_6E2[nl[n]]);
	status += gpuFree(Bufferrec4_7E2[nl[n]]);
	status += gpuFree(Bufferrec4_8E2[nl[n]]);
	#endif
	#if(N3G>0)
	#if(DEREFINE_POLE==0 || N_LEVELS_3D>1)
	status += gpuFree(Bufferrec5_1E2[nl[n]]);
	status += gpuFree(Bufferrec5_3E2[nl[n]]);
	status += gpuFree(Bufferrec5_5E2[nl[n]]);
	status += gpuFree(Bufferrec5_7E2[nl[n]]);
	status += gpuFree(Bufferrec6_2E2[nl[n]]);
	status += gpuFree(Bufferrec6_4E2[nl[n]]);
	status += gpuFree(Bufferrec6_6E2[nl[n]]);
	status += gpuFree(Bufferrec6_8E2[nl[n]]);
	#endif
	#endif
	#endif
	#if(N3G>0)
	status += gpuFree(BuffersendE1corn9[nl[n]]);
	status += gpuFree(BuffersendE1corn10[nl[n]]);
	status += gpuFree(BuffersendE1corn11[nl[n]]);
	status += gpuFree(BuffersendE1corn12[nl[n]]);
	status += gpuFree(BuffersendE2corn5[nl[n]]);
	status += gpuFree(BuffersendE2corn6[nl[n]]);
	status += gpuFree(BuffersendE2corn7[nl[n]]);
	status += gpuFree(BuffersendE2corn8[nl[n]]);
	#endif
	status += gpuFree(BuffersendE3corn1[nl[n]]);
	status += gpuFree(BuffersendE3corn2[nl[n]]);
	status += gpuFree(BuffersendE3corn3[nl[n]]);
	status += gpuFree(BuffersendE3corn4[nl[n]]);
	#if(N3G>0)
	status += gpuFree(BufferrecE1corn9[nl[n]]);
	status += gpuFree(BufferrecE1corn10[nl[n]]);
	status += gpuFree(BufferrecE1corn11[nl[n]]);
	status += gpuFree(BufferrecE1corn12[nl[n]]);
	status += gpuFree(BufferrecE2corn5[nl[n]]);
	status += gpuFree(BufferrecE2corn6[nl[n]]);
	status += gpuFree(BufferrecE2corn7[nl[n]]);
	status += gpuFree(BufferrecE2corn8[nl[n]]);
	#endif
	status += gpuFree(BufferrecE3corn1[nl[n]]);
	status += gpuFree(BufferrecE3corn2[nl[n]]);
	status += gpuFree(BufferrecE3corn3[nl[n]]);
	status += gpuFree(BufferrecE3corn4[nl[n]]);
	#if(N_LEVELS>1)
	#if(N3G>0)
	status += gpuFree(BufferrecE1corn9_3[nl[n]]);
	status += gpuFree(BufferrecE1corn9_7[nl[n]]);
	status += gpuFree(BufferrecE1corn10_1[nl[n]]);
	status += gpuFree(BufferrecE1corn10_5[nl[n]]);
	status += gpuFree(BufferrecE1corn11_2[nl[n]]);
	status += gpuFree(BufferrecE1corn11_6[nl[n]]);
	status += gpuFree(BufferrecE1corn12_4[nl[n]]);
	status += gpuFree(BufferrecE1corn12_8[nl[n]]);
	status += gpuFree(BufferrecE2corn5_2[nl[n]]);
	status += gpuFree(BufferrecE2corn5_4[nl[n]]);
	status += gpuFree(BufferrecE2corn6_1[nl[n]]);
	status += gpuFree(BufferrecE2corn6_3[nl[n]]);
	status += gpuFree(BufferrecE2corn7_5[nl[n]]);
	status += gpuFree(BufferrecE2corn7_7[nl[n]]);
	status += gpuFree(BufferrecE2corn8_6[nl[n]]);
	status += gpuFree(BufferrecE2corn8_8[nl[n]]);
	#endif
	status += gpuFree(BufferrecE3corn1_3[nl[n]]);
	status += gpuFree(BufferrecE3corn1_4[nl[n]]);
	status += gpuFree(BufferrecE3corn2_1[nl[n]]);
	status += gpuFree(BufferrecE3corn2_2[nl[n]]);
	status += gpuFree(BufferrecE3corn3_5[nl[n]]);
	status += gpuFree(BufferrecE3corn3_6[nl[n]]);
	status += gpuFree(BufferrecE3corn4_7[nl[n]]);
	status += gpuFree(BufferrecE3corn4_8[nl[n]]);
	#if(N3G>0)
	status += gpuFree(tempBufferrecE1corn9[nl[n]]);
	status += gpuFree(tempBufferrecE1corn10[nl[n]]);
	status += gpuFree(tempBufferrecE1corn11[nl[n]]);
	status += gpuFree(tempBufferrecE1corn12[nl[n]]);
	status += gpuFree(tempBufferrecE2corn5[nl[n]]);
	status += gpuFree(tempBufferrecE2corn6[nl[n]]);
	status += gpuFree(tempBufferrecE2corn7[nl[n]]);
	status += gpuFree(tempBufferrecE2corn8[nl[n]]);
	#endif
	#endif
	status += gpuFree(tempBufferrecE3corn1[nl[n]]);
	status += gpuFree(tempBufferrecE3corn2[nl[n]]);
	status += gpuFree(tempBufferrecE3corn3[nl[n]]);
	status += gpuFree(tempBufferrecE3corn4[nl[n]]);
	#if(N_LEVELS>1)
	#if(N3G>0)
	status += gpuFree(tempBufferrecE1corn9_3[nl[n]]);
	status += gpuFree(tempBufferrecE1corn9_7[nl[n]]);
	status += gpuFree(tempBufferrecE1corn10_1[nl[n]]);
	status += gpuFree(tempBufferrecE1corn10_5[nl[n]]);
	status += gpuFree(tempBufferrecE1corn11_2[nl[n]]);
	status += gpuFree(tempBufferrecE1corn11_6[nl[n]]);
	status += gpuFree(tempBufferrecE1corn12_4[nl[n]]);
	status += gpuFree(tempBufferrecE1corn12_8[nl[n]]);
	status += gpuFree(tempBufferrecE2corn5_2[nl[n]]);
	status += gpuFree(tempBufferrecE2corn5_4[nl[n]]);
	status += gpuFree(tempBufferrecE2corn6_1[nl[n]]);
	status += gpuFree(tempBufferrecE2corn6_3[nl[n]]);
	status += gpuFree(tempBufferrecE2corn7_5[nl[n]]);
	status += gpuFree(tempBufferrecE2corn7_7[nl[n]]);
	status += gpuFree(tempBufferrecE2corn8_6[nl[n]]);
	status += gpuFree(tempBufferrecE2corn8_8[nl[n]]);
	#endif
	status += gpuFree(tempBufferrecE3corn1_3[nl[n]]);
	status += gpuFree(tempBufferrecE3corn1_4[nl[n]]);
	status += gpuFree(tempBufferrecE3corn2_1[nl[n]]);
	status += gpuFree(tempBufferrecE3corn2_2[nl[n]]);
	status += gpuFree(tempBufferrecE3corn3_5[nl[n]]);
	status += gpuFree(tempBufferrecE3corn3_6[nl[n]]);
	status += gpuFree(tempBufferrecE3corn4_7[nl[n]]);
	status += gpuFree(tempBufferrecE3corn4_8[nl[n]]);
	#if(N3G>0)
	status += gpuFree(BufferrecE1corn9_32[nl[n]]);
	status += gpuFree(BufferrecE1corn9_72[nl[n]]);
	status += gpuFree(BufferrecE1corn10_12[nl[n]]);
	status += gpuFree(BufferrecE1corn10_52[nl[n]]);
	status += gpuFree(BufferrecE1corn11_22[nl[n]]);
	status += gpuFree(BufferrecE1corn11_62[nl[n]]);
	status += gpuFree(BufferrecE1corn12_42[nl[n]]);
	status += gpuFree(BufferrecE1corn12_82[nl[n]]);
	status += gpuFree(BufferrecE2corn5_22[nl[n]]);
	status += gpuFree(BufferrecE2corn5_42[nl[n]]);
	status += gpuFree(BufferrecE2corn6_12[nl[n]]);
	status += gpuFree(BufferrecE2corn6_32[nl[n]]);
	status += gpuFree(BufferrecE2corn7_52[nl[n]]);
	status += gpuFree(BufferrecE2corn7_72[nl[n]]);
	status += gpuFree(BufferrecE2corn8_62[nl[n]]);
	status += gpuFree(BufferrecE2corn8_82[nl[n]]);
	#endif
	status += gpuFree(BufferrecE3corn1_32[nl[n]]);
	status += gpuFree(BufferrecE3corn1_42[nl[n]]);
	status += gpuFree(BufferrecE3corn2_12[nl[n]]);
	status += gpuFree(BufferrecE3corn2_22[nl[n]]);
	status += gpuFree(BufferrecE3corn3_52[nl[n]]);
	status += gpuFree(BufferrecE3corn3_62[nl[n]]);
	status += gpuFree(BufferrecE3corn4_72[nl[n]]);
	status += gpuFree(BufferrecE3corn4_82[nl[n]]);
	#endif
}

int gpuFree(void *devPtr){
	#if(GPU_DIRECT)
	return cudaFree(devPtr);
	#else
	return cudaFreeHost(devPtr);
	#endif
}