#include "decsCUDA.h"
extern "C" {
#include "decs.h" 
/*Start reading timestep from GPU*/
void read_time_GPU(void){
	int n;
	for (n = 0; n < n_active; n++){
		if (prestep_full[n_ord[n]] == 1){
			cudaMemcpyAsync(dtij_GPU[n_ord[n]], Bufferdtij[n_ord[n]], (int)((nr_workgroups[n_ord[n]] - 1) * sizeof(FTYPE2)), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
		}
	}
}

/*Do last step of reduction of timestep on CPU*/
double fluxcalc_GPU(int n)
{
	double ndt;
	int y;
	ndt = 1.e9;

	cudaStreamSynchronize(commandQueueGPU[n]);
	for (y = 0; y < nr_workgroups[n] - 1; y++){
		if (dtij_GPU[n][y] < ndt && dtij_GPU[n][y] < 1.e9){
			ndt = dtij_GPU[n][y];
		}
	}

	return(ndt);
}

void GPU_init(void)
{
	int i, j;
	//Create concurrent commandqueues
	for (i = 0; i < N_GPU; i++){
		cudaSetDevice(i);
		for (j = 0; j < NQ; j++) cudaStreamCreate(&commandQueue[i*NQ + j]);
		for (j = 0; j < N_GPU;j++) if(i != j) cudaDeviceEnablePeerAccess(j, 0);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

void set_arrays_GPU(int n, int device){
	int i, j, z;
	
	/*N1 and N2 values to be exported to GPU memory*/
	int offset = 0;

	/*Set the global work size and make sure that it is a multiple of the group size. The Nvidia OpenCL framework crashes otherwise!*/
	fix_mem[n] = LOCAL_WORK_SIZE - ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)) % LOCAL_WORK_SIZE;
	fix_mem2[n] = LOCAL_WORK_SIZE - ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)) % LOCAL_WORK_SIZE;
	local_work_size[0] = LOCAL_WORK_SIZE;

	global_work_size_special[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) + (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) +
		(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G);
	global_work_size_special1[n][0] = (LOCAL_WORK_SIZE - ((N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G) * NG;
	global_work_size_special2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) * NG) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G) * NG;
	global_work_size_special3[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G) * NG;
	nr_workgroups_special[n] = (int)ceil((double)global_work_size_special[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special1[n] = (int)ceil((double)global_work_size_special1[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special2[n] = (int)ceil((double)global_work_size_special2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups_special3[n] = (int)ceil((double)global_work_size_special3[n][0] / (double)LOCAL_WORK_SIZE);

	global_work_size1[n][0] = (LOCAL_WORK_SIZE - (N1_GPU[n] * N2_GPU[n] * N3_GPU[n]) % LOCAL_WORK_SIZE) + (N1_GPU[n] * N2_GPU[n] * N3_GPU[n]); //utoprim
	global_work_size2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3); //fluxcalc_prerp
	global_work_size2_1[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1 - 1) * (N2_GPU[n] + 2 * D2 ) * (N3_GPU[n] + 2 * D3 )) % LOCAL_WORK_SIZE)+ (N1_GPU[n] + 2 * D1 - 1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3); //fluxcalc
	global_work_size2_2[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1 ) * (N2_GPU[n] + 2 * D2 - 1) * (N3_GPU[n] + 2 * D3 )) % LOCAL_WORK_SIZE)+ (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2 - 1) * (N3_GPU[n] + 2 * D3); //fluxcalc
	global_work_size2_3[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3 - 1)) % LOCAL_WORK_SIZE)+ (N1_GPU[n] + 2 * D1) * (N2_GPU[n] + 2 * D2) * (N3_GPU[n] + 2 * D3 - 1); //fluxcalc
	global_work_size3[n][0] = (LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3); //flux_ct

	nr_workgroups[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups1[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_1[n] = (int)ceil((double)global_work_size2_1[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_2[n] = (int)ceil((double)global_work_size2_2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_3[n] = (int)ceil((double)global_work_size2_3[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups3[n] = (int)ceil((double)global_work_size3[n][0] / (double)LOCAL_WORK_SIZE);

	//Select correct CUDA device
	cudaSetDevice(device);

	//Create events
	for (i = 0; i < 600; i++) cudaEventCreate(&boundevent[n][i]);
	for (i = 0; i < 100; i++) cudaEventCreate(&boundevent1[n][i]);
	for (i = 0; i < 100; i++) cudaEventCreate(&boundevent2[n][i]);

	/*Allocate memory to 1D arrays*/
	p_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	dq_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2)); //array to store temporary data
	#if(STAGGERED)
	ps_1[n] = (FTYPE2(*))calloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	psh_1[n] = (FTYPE2(*))calloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	#endif
	ph_1[n] = (FTYPE2(*))calloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(FTYPE2));
	dU_GPU[n] = (FTYPE2(*))calloc(NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G)), sizeof(FTYPE2));
	pbound_1[n] = (FTYPE2(*))calloc(N_POINTS*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n]), sizeof(FTYPE2));
	#if(!NONSYMMETRIC)
	gcov_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	gcon_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	conn_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM, sizeof(FTYPE2));
	gdet_GPU[n] = (FTYPE2(*))calloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG, sizeof(FTYPE2));
	#else
	gcov_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	gcon_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM, sizeof(FTYPE2));
	conn_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NDIM*NDIM*NDIM, sizeof(FTYPE2));
	gdet_GPU[n] = (FTYPE2(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG, sizeof(FTYPE2));
	#endif
	pflag_GPU[n] = (int(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]), sizeof(int));
	failimage_GPU[n] = (int(*))calloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL, sizeof(int));
	Katm_GPU[n] = (FTYPE2(*))calloc((N1_GPU[n] + 2 * N1G), sizeof(FTYPE2));

	/*Allocate memory to buffers on GPU*/
	cudaMalloc(&BufferF1_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&BufferF2_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&BufferF3_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferdq_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&BufferE_1[n], NDIM*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#if(LEER)
	cudaMalloc(&BufferV[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferradius[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferstorage1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferstorage2[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferstorage3[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferstorage4[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferp_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#if(STAGGERED)
	cudaMalloc(&Bufferps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferpsh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferpbound_1[n], N_POINTS*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n])*sizeof(FTYPE2));
	cudaMalloc(&Bufferdtij[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) / LOCAL_WORK_SIZE*sizeof(FTYPE2));
	cudaMalloc(&Bufferpflag[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int));
	cudaMalloc(&Bufferfailimage[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(int));
	cudaMalloc(&Bufferdiagflux[n], 3 * MY_MAX((N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G), (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G))*sizeof(FTYPE2));
	cudaMalloc(&BufferKatm[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferdU[n], NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G))*sizeof(FTYPE2));

	cudaMalloc(&dtij_GPU[n], (nr_workgroups[n] + 1) * sizeof(FTYPE2));
	cudaHostGetDevicePointer(&Bufferdtij[n], dtij_GPU[n], 0);

	cudaMalloc(&Buffersend1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Buffersend2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Buffersend3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Buffersend4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif

	#if(N3G>0)
	cudaMalloc(&Buffersend5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend5_1[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend5_3[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend5_5[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend5_7[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Buffersend6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend6_2[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend6_4[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend6_6[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend6_8[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaMalloc(&Bufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#endif
	#if(N3G>0)
	cudaMalloc(&Bufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaMalloc(&tempBufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	#if(N3G>0)
	cudaMalloc(&tempBufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaMalloc(&Buffersend1flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend2flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend3flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend4flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Buffersend5flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend6flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec1flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaMalloc(&Bufferrec1flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec1_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaHostRegister(send1_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2),0);
	//cudaMalloc(&Buffersend2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaHostRegister(send3_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2),0);
	//cudaMalloc(&Buffersend4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	//cudaMalloc(&Buffersend5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	//cudaMalloc(&Buffersend6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaHostRegister(receive1_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2),0);
	//cudaMalloc(&Bufferrec2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaHostRegister(receive3_fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2),0);
	//cudaMalloc(&Bufferrec4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	//cudaMalloc(&Bufferrec5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	//cudaMalloc(&Bufferrec6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	/*#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif*/
	cudaMalloc(&Buffersend1E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend2E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend3E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend4E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Buffersend5E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Buffersend6E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec1E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4E[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6E[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8E[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8E[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	cudaMalloc(&Bufferrec1E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2E1[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4E1[n], 2 * (N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6E1[n], 2 * (N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8E1[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8E1[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&Bufferrec1_3E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_4E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_7E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec1_8E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_1E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_2E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_3E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec2_4E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_1E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_2E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_5E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec3_6E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_5E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_6E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_7E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec4_8E2[n], 2 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_3E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_5E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec5_7E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_2E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_4E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_6E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&Bufferrec6_8E2[n], 2 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	#endif
	#if(N3G>0)
	cudaMalloc(&BuffersendE1corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE1corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE1corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE1corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE2corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE2corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE2corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE2corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&BuffersendE3corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE3corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE3corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BuffersendE3corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&BufferrecE3corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9_3[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn9_7[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_5[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_6[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_4[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_8[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_4[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_3[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_5[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_7[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_6[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_8[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&BufferrecE3corn1_3[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn1_4[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn2_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn2_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn3_5[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn3_6[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn4_7[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn4_8[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12[n], 1 * (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8[n], 1 * (N2_GPU[n] + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrecE3corn1[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn2[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn3[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn4[n], 1 * (N3_GPU[n] + 2 * N3G)*sizeof(FTYPE2));
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9_3[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn9_7[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_1[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_5[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_2[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_6[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_4[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_8[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_2[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_4[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_1[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_3[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_5[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_7[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_6[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_8[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&tempBufferrecE3corn1_3[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn1_4[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn2_1[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn2_2[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn3_5[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn3_6[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn4_7[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&tempBufferrecE3corn4_8[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9_32[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn9_72[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_12[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn10_52[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_22[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn11_62[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_42[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE1corn12_82[n], 1 * (N1_GPU[n] / (1 + REF_1) + 2 * N1G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_22[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn5_42[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_12[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn6_32[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_52[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn7_72[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_62[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE2corn8_82[n], 1 * (N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(FTYPE2));
	#endif
	cudaMalloc(&BufferrecE3corn1_32[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn1_42[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn2_12[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn2_22[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn3_52[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn3_62[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn4_72[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	cudaMalloc(&BufferrecE3corn4_82[n], 1 * (N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(FTYPE2));
	#endif
	/*
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend1_3[n], send1_3[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_4[n], send1_4[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_7[n], send1_7[n], 0);
	cudaHostGetDevicePointer(&Buffersend1_8[n], send1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend2[n], send2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend2_1[n], send2_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend2_2[n], send2_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend2_3[n], send2_3[n],0);
	cudaHostGetDevicePointer(&Buffersend2_4[n], send2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend3[n], send3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend3_1[n], send3_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_2[n], send3_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_5[n], send3_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend3_6[n], send3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend4[n], send4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend4_5[n], send4_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_6[n], send4_6[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_7[n], send4_7[n], 0);
	cudaHostGetDevicePointer(&Buffersend4_8[n], send4_8[n], 0);
	#endif

	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5[n], send5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend5_1[n], send5_1[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_3[n], send5_3[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_5[n], send5_5[n], 0);
	cudaHostGetDevicePointer(&Buffersend5_7[n], send5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&Buffersend6[n], send6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Buffersend6_2[n], send6_2[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_4[n], send6_4[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_6[n], send6_6[n], 0);
	cudaHostGetDevicePointer(&Buffersend6_8[n], send6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1[n], receive1[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3[n], receive1_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4[n], receive1_4[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7[n], receive1_7[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8[n], receive1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec2[n], receive2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec2_1[n], receive2_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2[n], receive2_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3[n], receive2_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4[n], receive2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec3[n], receive3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec3_1[n], receive3_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2[n], receive3_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5[n], receive3_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6[n], receive3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec4[n], receive4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec4_5[n], receive4_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6[n], receive4_6[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7[n], receive4_7[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8[n], receive4_8[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5[n], receive5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec5_1[n], receive5_1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3[n], receive5_3[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5[n], receive5_5[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7[n], receive5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec6[n], receive6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec6_2[n], receive6_2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4[n], receive6_4[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6[n], receive6_6[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8[n], receive6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&tempBufferrec1[n], tempreceive1[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec1_3[n], tempreceive1_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_4[n], tempreceive1_4[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_7[n], tempreceive1_7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec1_8[n], tempreceive1_8[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec2[n], tempreceive2[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec2_1[n], tempreceive2_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_2[n], tempreceive2_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_3[n], tempreceive2_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec2_4[n], tempreceive2_4[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec3[n], tempreceive3[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec3_1[n], tempreceive3_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_2[n], tempreceive3_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_5[n], tempreceive3_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec3_6[n], tempreceive3_6[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec4[n], tempreceive4[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec4_5[n], tempreceive4_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_6[n], tempreceive4_6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_7[n], tempreceive4_7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec4_8[n], tempreceive4_8[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrec5[n], tempreceive5[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec5_1[n], tempreceive5_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_3[n], tempreceive5_3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_5[n], tempreceive5_5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec5_7[n], tempreceive5_7[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrec6[n], tempreceive6[n], 0);
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&tempBufferrec6_2[n], tempreceive6_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_4[n], tempreceive6_4[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_6[n], tempreceive6_6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrec6_8[n], tempreceive6_8[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1flux[n], send1_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend2flux[n], send2_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend3flux[n], send3_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend4flux[n], send4_flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5flux[n], send5_flux[n], 0);
	cudaHostGetDevicePointer(&Buffersend6flux[n], send6_flux[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1flux[n], receive1_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2flux[n], receive2_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3flux[n], receive3_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4flux[n], receive4_flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5flux[n], receive5_flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6flux[n], receive6_flux[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3flux[n], receive1_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux[n], receive1_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux[n], receive1_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux[n], receive1_8flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux[n], receive2_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux[n], receive2_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux[n], receive2_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux[n], receive2_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux[n], receive3_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux[n], receive3_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux[n], receive3_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux[n], receive3_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux[n], receive4_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux[n], receive4_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux[n], receive4_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux[n], receive4_8flux[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux[n], receive5_1flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux[n], receive5_3flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux[n], receive5_5flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux[n], receive5_7flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux[n], receive6_2flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux[n], receive6_4flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux[n], receive6_6flux[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux[n], receive6_8flux[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1flux1[n], receive1_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2flux1[n], receive2_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3flux1[n], receive3_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4flux1[n], receive4_flux1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5flux1[n], receive5_flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6flux1[n], receive6_flux1[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3flux1[n], receive1_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux1[n], receive1_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux1[n], receive1_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux1[n], receive1_8flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux1[n], receive2_1flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux1[n], receive2_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux1[n], receive2_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux1[n], receive2_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux1[n], receive3_1flux1[n],  0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux1[n], receive3_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux1[n], receive3_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux1[n], receive3_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux1[n], receive4_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux1[n], receive4_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux1[n], receive4_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux1[n], receive4_8flux1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux1[n], receive5_1flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux1[n], receive5_3flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux1[n], receive5_5flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux1[n], receive5_7flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux1[n], receive6_2flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux1[n], receive6_4flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux1[n], receive6_6flux1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux1[n], receive6_8flux1[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1_3flux2[n], receive1_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4flux2[n], receive1_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7flux2[n], receive1_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8flux2[n], receive1_8flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1flux2[n], receive2_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2flux2[n], receive2_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3flux2[n], receive2_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4flux2[n], receive2_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1flux2[n], receive3_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2flux2[n], receive3_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5flux2[n], receive3_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6flux2[n], receive3_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5flux2[n], receive4_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6flux2[n], receive4_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7flux2[n], receive4_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8flux2[n], receive4_8flux2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1flux2[n], receive5_1flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3flux2[n], receive5_3flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5flux2[n], receive5_5flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7flux2[n], receive5_7flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2flux2[n], receive6_2flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4flux2[n], receive6_4flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6flux2[n], receive6_6flux2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8flux2[n], receive6_8flux2[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1fine[n], send1_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend2fine[n], send2_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend3fine[n], send3_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend4fine[n], send4_fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5fine[n], send5_fine[n], 0);
	cudaHostGetDevicePointer(&Buffersend6fine[n], send6_fine[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1fine[n], receive1_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2fine[n], receive2_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3fine[n], receive3_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4fine[n], receive4_fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5fine[n], receive5_fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6fine[n], receive6_fine[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3fine[n], receive1_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4fine[n], receive1_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7fine[n], receive1_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8fine[n], receive1_8fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1fine[n], receive2_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2fine[n], receive2_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3fine[n], receive2_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4fine[n], receive2_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1fine[n], receive3_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2fine[n], receive3_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5fine[n], receive3_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6fine[n], receive3_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5fine[n], receive4_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6fine[n], receive4_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7fine[n], receive4_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8fine[n], receive4_8fine[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1fine[n], receive5_1fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3fine[n], receive5_3fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5fine[n], receive5_5fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7fine[n], receive5_7fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2fine[n], receive6_2fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4fine[n], receive6_4fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6fine[n], receive6_6fine[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8fine[n], receive6_8fine[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Buffersend1E[n], send1_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend2E[n], send2_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend3E[n], send3_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend4E[n], send4_E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Buffersend5E[n], send5_E[n], 0);
	cudaHostGetDevicePointer(&Buffersend6E[n], send6_E[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1E[n], receive1_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2E[n], receive2_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3E[n], receive3_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4E[n], receive4_E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5E[n], receive5_E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6E[n], receive6_E[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3E[n], receive1_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E[n], receive1_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E[n], receive1_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E[n], receive1_8E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E[n], receive2_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E[n], receive2_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E[n], receive2_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E[n], receive2_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E[n], receive3_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E[n], receive3_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E[n], receive3_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E[n], receive3_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E[n], receive4_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E[n], receive4_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E[n], receive4_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E[n], receive4_8E[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E[n], receive5_1E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E[n], receive5_3E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E[n], receive5_5E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E[n], receive5_7E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E[n], receive6_2E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E[n], receive6_4E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E[n], receive6_6E[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E[n], receive6_8E[n], 0);
	#endif
	#endif
	cudaHostGetDevicePointer(&Bufferrec1E1[n], receive1_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2E1[n], receive2_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3E1[n], receive3_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4E1[n], receive4_E1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5E1[n], receive5_E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6E1[n], receive6_E1[n], 0);
	#endif
	#if(N_LEVELS>1)
	cudaHostGetDevicePointer(&Bufferrec1_3E1[n], receive1_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E1[n], receive1_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E1[n], receive1_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E1[n], receive1_8E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E1[n], receive2_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E1[n], receive2_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E1[n], receive2_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E1[n], receive2_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E1[n], receive3_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E1[n], receive3_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E1[n], receive3_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E1[n], receive3_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E1[n], receive4_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E1[n], receive4_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E1[n], receive4_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E1[n], receive4_8E1[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E1[n], receive5_1E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E1[n], receive5_3E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E1[n], receive5_5E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E1[n], receive5_7E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E1[n], receive6_2E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E1[n], receive6_4E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E1[n], receive6_6E1[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E1[n], receive6_8E1[n], 0);
	#endif
	cudaHostGetDevicePointer(&Bufferrec1_3E2[n], receive1_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_4E2[n], receive1_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_7E2[n], receive1_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec1_8E2[n], receive1_8E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_1E2[n], receive2_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_2E2[n], receive2_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_3E2[n], receive2_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec2_4E2[n], receive2_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_1E2[n], receive3_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_2E2[n], receive3_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_5E2[n], receive3_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec3_6E2[n], receive3_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_5E2[n], receive4_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_6E2[n], receive4_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_7E2[n], receive4_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec4_8E2[n], receive4_8E2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&Bufferrec5_1E2[n], receive5_1E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_3E2[n], receive5_3E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_5E2[n], receive5_5E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec5_7E2[n], receive5_7E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_2E2[n], receive6_2E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_4E2[n], receive6_4E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_6E2[n], receive6_6E2[n], 0);
	cudaHostGetDevicePointer(&Bufferrec6_8E2[n], receive6_8E2[n], 0);
	#endif
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&BuffersendE1corn9[n], send_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn10[n], send_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn11[n], send_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&BuffersendE1corn12[n], send_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn5[n], send_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn6[n], send_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn7[n], send_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&BuffersendE2corn8[n], send_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&BuffersendE3corn1[n], send_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn2[n], send_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn3[n], send_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&BuffersendE3corn4[n], send_E3_corn4[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9[n], receive_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10[n], receive_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11[n], receive_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12[n], receive_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5[n], receive_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6[n], receive_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7[n], receive_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8[n], receive_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1[n], receive_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2[n], receive_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3[n], receive_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4[n], receive_E3_corn4[n], 0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9_3[n], receive_E1_corn9_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn9_7[n], receive_E1_corn9_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_1[n], receive_E1_corn10_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_5[n], receive_E1_corn10_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_4[n], receive_E1_corn11_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_6[n], receive_E1_corn11_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_2[n], receive_E1_corn12_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_8[n], receive_E1_corn12_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_2[n], receive_E2_corn5_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_4[n], receive_E2_corn5_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_1[n], receive_E2_corn6_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_3[n], receive_E2_corn6_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_5[n], receive_E2_corn7_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_7[n], receive_E2_corn7_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_6[n], receive_E2_corn8_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_8[n], receive_E2_corn8_2[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1_3[n], receive_E3_corn1_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn1_4[n], receive_E3_corn1_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_1[n], receive_E3_corn2_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_2[n], receive_E3_corn2_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_5[n], receive_E3_corn3_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_6[n], receive_E3_corn3_2[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_7[n], receive_E3_corn4_1[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_8[n], receive_E3_corn4_2[n], 0);
	#endif
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrecE1corn9[n], tempreceive_E1_corn9[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10[n], tempreceive_E1_corn10[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11[n], tempreceive_E1_corn11[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12[n], tempreceive_E1_corn12[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5[n], tempreceive_E2_corn5[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6[n], tempreceive_E2_corn6[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7[n], tempreceive_E2_corn7[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8[n], tempreceive_E2_corn8[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrecE3corn1[n], tempreceive_E3_corn1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2[n], tempreceive_E3_corn2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3[n], tempreceive_E3_corn3[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4[n], tempreceive_E3_corn4[n], 0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostGetDevicePointer(&tempBufferrecE1corn9_3[n], tempreceive_E1_corn9_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn9_7[n], tempreceive_E1_corn9_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10_1[n], tempreceive_E1_corn10_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn10_5[n], tempreceive_E1_corn10_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11_2[n], tempreceive_E1_corn11_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn11_6[n], tempreceive-E1_corn11_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12_4[n], tempreceive_E1_corn12_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE1corn12_8[n], tempreceive_E1_corn12_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5_2[n], tempreceive_E2_corn5_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn5_4[n], tempreceive_E2_corn5_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6_1[n], tempreceive_E2_corn6_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn6_3[n], tempreceive_E2_corn6_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7_5[n], tempreceive_E2_corn7_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn7_7[n], tempreceive_E2_corn7_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8_6[n], tempreceive_E2_corn8_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE2corn8_8[n], tempreceive_E2_corn8_2[n], 0);
	#endif
	cudaHostGetDevicePointer(&tempBufferrecE3corn1_3[n], tempreceive_E3_corn1_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn1_4[n], tempreceive_E3_corn1_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2_1[n], tempreceive_E3_corn2_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn2_2[n], tempreceive_E3_corn2_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3_5[n], tempreceive_E3_corn3_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn3_6[n], tempreceive_E3_corn3_2[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4_7[n], tempreceive_E3_corn4_1[n], 0);
	cudaHostGetDevicePointer(&tempBufferrecE3corn4_8[n], tempreceive_E3_corn4_2[n], 0);
	#if(N3G>0)
	cudaHostGetDevicePointer(&BufferrecE1corn9_32[n], receive_E1_corn9_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn9_72[n], receive_E1_corn9_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_12[n], receive_E1_corn10_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn10_52[n], receive_E1_corn10_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_22[n], receive_E1_corn11_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn11_62[n], receive_E1_corn11_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_42[n], receive_E1_corn12_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE1corn12_82[n], receive_E1_corn12_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_22[n], receive_E2_corn5_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn5_42[n], receive_E2_corn5_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_12[n], receive_E2_corn6_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn6_32[n], receive_E2_corn6_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_52[n], receive_E2_corn7_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn7_72[n], receive_E2_corn7_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_62[n], receive_E2_corn8_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE2corn8_82[n], receive_E2_corn8_22[n], 0);
	#endif
	cudaHostGetDevicePointer(&BufferrecE3corn1_32[n], receive_E3_corn1_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn1_42[n], receive_E3_corn1_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_12[n], receive_E3_corn2_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn2_22[n], receive_E3_corn2_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_52[n], receive_E3_corn3_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn3_62[n], receive_E3_corn3_22[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_72[n], receive_E3_corn4_12[n], 0);
	cudaHostGetDevicePointer(&BufferrecE3corn4_82[n], receive_E3_corn4_22[n], 0);
	#endif
	*/

	/*Set arguments of kernel*/
	int pg, d1, d2, k;
	#pragma omp parallel private(i, j, z, k, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (pg = 0; pg < NPG; pg++){
				#if(!NONSYMMETRIC)
				gdet_GPU[n][pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gdet[n][index2(n, i, j, z)][pg];
				#else
				#endif
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						gcov_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[n][index2(n, i, j, z)][pg][d1][d2];
						gcon_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[n][index2(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}

			for (pg = 0; pg < NDIM; pg++){
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						conn_GPU[n][d1*NDIM*NDIM*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + d2*NDIM*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) 
							+ (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[n][index2(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}
			#if(LEER)
			for (k = 0; k < 6; k++){
				dq_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = V[n][index(n, i, j, z)][k];
			}
			#endif
		}
	}
	
	status=0;
	#if(LEER)
	status =cudaMemcpy(BufferV[n], dq_1[n], 6 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);;
	#endif
	
	#if(!NONSYMMETRIC)
	cudaMalloc(&Buffergdet[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*sizeof(FTYPE2));
	cudaMalloc(&Buffergcov[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(FTYPE2));
	cudaMalloc(&Buffergcon[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(FTYPE2));
	cudaMalloc(&Bufferconn[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM*sizeof(FTYPE2));
	status += cudaMemcpy(Buffergdet[n], gdet_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Buffergcov[n], gcov_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Buffergcon[n], gcov_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferconn[n], conn_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#else
	#endif

	if (status != 0) printf("Error in setting kernel arguments 2: %d \n", status);
	free(gcov_GPU[n]);
	free(gcon_GPU[n]);
	free(conn_GPU[n]);
	free(gdet_GPU[n]);
}
double check = 1.0;

void GPU_write(int n)
{
	int i, j, z, k, l, pg, d1, d2;
	double radius_GPU[(N1 + 2 * N1G)], r, th, phi, X[NDIM];
	for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		coord(n, i, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		radius_GPU[(i - N1_GPU_offset[n] + N1G)] = r;
		Katm_GPU[n][i - N1_GPU_offset[n] + N1G] = Katm[n][i - N1_GPU_offset[n] + N1G];
	}

	#pragma omp parallel private(i, j, z, k, l, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = p[n][index(n, i, j, z)][k];
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index(n, i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index(n, i, j, z)][k];
				#endif
				#if(ibound)
				if (i == N1_GPU_offset[n]){
					for (l = 0; l < N_POINTS; l++){
						pbound_1[n][l*NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + fix_mem[n]) + k*((N2_GPU[n] + 2 * N2G) + fix_mem[n]) + (j - N2_GPU_offset[n] + N2G)] = pbound[j*(N3_GPU[n] + 2 * N3G) + z][k][l];
					}
				}
				#endif
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[n][index(n, i, j, z)][k];
				psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = psh[n][index(n, i, j, z)][k];

			}
			#endif
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
			}
			pflag_GPU[n][(i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
		}
	}

	status = 0;
	#if (ELLIPTICAL2)
	for (i = N1_GPU_offset[n] - N1G; i<N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		for (j = N2_GPU_offset[n] - N2G; j<N2_GPU_offset[n] + N2_GPU[n] + N2G; j++){
			for (k = 0; k < NPR; k++){
				dU_GPU[n][k*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = dU_s[n][index2(n, i, j, 0)][k];
			}
		}
	}
	status = cudaMemcpy(BufferdU[n], dU_GPU[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#endif
	cudaSetDevice(block[n][AMR_GPU]);
	/*Initialize memory items that have to be passed on to the GPU*/
	status = cudaMemcpy(Bufferp_1[n], p_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferph_1[n], ph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#if(STAGGERED)
	status += cudaMemcpy(Bufferps_1[n], ps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferpsh_1[n], psh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	#endif
	status += cudaMemcpy(Bufferpflag[n], pflag_GPU[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferfailimage[n], failimage_GPU[n], NFAIL*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int), cudaMemcpyHostToDevice);
	status += cudaMemcpy(BufferKatm[n], Katm_GPU[n], (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaMemcpyHostToDevice);
	status += cudaMemcpy(Bufferradius[n], radius_GPU, (N1_GPU[n] + 2 * N1G)*sizeof(FTYPE2), cudaMemcpyHostToDevice);

	if (status != 0) printf("Error in GPU_write: %d \n", status);
}

void GPU_hcor(int n){

}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	__global__ void fluxcalcprep(int i, int j, int z, double *   F, double *  dq, double *  p, int dir, int lim, int number, double *  V);

	/*Set arguments of kernel*/
	if (dir == 1){
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else if (dir == 2){
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else{
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	if (status != 0) printf("Error Fluxcalcprep %d\n", status);

}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	cudaSetDevice(block[n][AMR_GPU]);
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){
		if (dir == 1){
			 fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			 fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			 fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	else{
		if (dir == 1){
			 fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			 fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			 fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, Bufferdtij[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	if (status != 0) printf("Error Fluxcalc2D2 %d\n", status);

	/*Calculate reconstructed right state*/
	#if(PPM || LEER)
	GPU_fluxcalcprep(dir, flag, 2, n);
	printf("PPM and Leer not yet fully implemented this way... \n");
	#endif
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	cudaSetDevice(block[n][AMR_GPU]);
	 fix_flux << < nr_workgroups_special[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	if (status != 0)printf("Error fixflux %d \n", status);
}

void GPU_consttransport_bound(void){
	int n;

	gpu = 1;
	#if(PRESTEP)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#if(!TIMESTEP_JET)
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		E_send1(E_corn, BufferE_1, n_ord[n]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_send3(E_corn, BufferE_1, n_ord[n]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
		#endif
		#endif
	}

	//For last timestep synchronize electric fields immediately
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#if(!TIMESTEP_JET)
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		E_rec1(E_corn, BufferE_1, n_ord[n], 5);
		E_rec2(E_corn, BufferE_1, n_ord[n], 5);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_rec3(E_corn, BufferE_1, n_ord[n], 5);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 5);
		#endif
		#endif
	}

	//For first timestep do not synchronize electrice fields 
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		#if(!TIMESTEP_JET)
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
		#if(N3G>0)
		#if(!TIMESTEP_JET)
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
		#endif
		#endif
	}
	#else
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send1(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 1);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);

		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
	}

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec2(E_corn, BufferE_1, n_ord[n], 1);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
	}

	#if(N3G>0)
	#if(!TIMESTEP_JET)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send3(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 1);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
	}

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}	
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}

	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	#endif
	#endif
	#if(TRANS_BOUND)
	E_average();
	#endif
}

void GPU_consttransport1(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	if (status != 0)printf("Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	if (status != 0)printf("Error constransport2 %d \n", status);
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferps_1[n], BufferE_1[n], Dt);
	}
	else{
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferpsh_1[n], BufferE_1[n], Dt);
	}
	if (status != 0)printf("Error constransport3 %d \n", status);
}

void GPU_flux_ct1(int n)
{
	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct1 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	if (status != 0) printf("Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct2 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	if (status != 0) printf("Error fluxct2 %d\n", status);
}

void GPU_Utoprim(int flag, int n)
{

}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) / LOCAL_WORK_SIZE;
	cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 1){
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	if (status != 0) printf("Error fixuputoprim %d\n", status);
}

void GPU_fixup(int flag, int n, double Dt)
{
	cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 0){
		 fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		 fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	if (status != 0) printf("Error fixup %d\n", status);
}

void GPU_boundprim(int bound_force)
{
	int i, n;
	int temp = nstep;
	gpu = 1;

	if (bound_force == 1) nstep = -1;
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim1(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim2(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	//For last timestep do not receive synchronized electrice fields 
	#if(PRESTEP)
	rc = 0;
	//MPI communication
	for (i = log(AMR_MAXTIMELEVEL) / log(2); i >= 0; i--){
		if (nstep % ((int)pow(2, i)) == ((int)pow(2, i)) - 1){
			if (nstep >= 2 * AMR_SWITCHTIMELEVEL) MPI_Barrier(row_comm[i]);
			break;
		}
	}

	if (rank == 0){
		begin2 = clock();
	}
	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			flux_rec1(F1, BufferF1_1, n_ord[n], 3);
			flux_rec2(F2, BufferF2_1, n_ord[n], 3);
			flux_rec3(F3, BufferF3_1, n_ord[n], 3);

			#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
			#if(!TIMESTEP_JET)
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
			E_rec1(E_corn, BufferE_1, n_ord[n], 3);
			E_rec2(E_corn, BufferE_1, n_ord[n], 3);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 3);
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 3);
			#endif
			#endif
			#endif
		}

		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);

			#if(WHICHPROBLEM!=DISRUPTION_PROBLEM)
			#if(!TIMESTEP_JET)
			E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			E_rec1(E_corn, BufferE_1, n_ord[n], 1);
			E_rec2(E_corn, BufferE_1, n_ord[n], 1);
			#if(N3G>0)
			#if(!TIMESTEP_JET)
			E_rec3(E_corn, BufferE_1, n_ord[n], 1);
			E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
			#endif
			#endif
			#endif
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE/BoundcomF \n");
	#endif


	rc = 0;
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, Bufferp_1, 0, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, Bufferph_1, 0, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
	}

	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, Bufferp_1, 0, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, Bufferph_1, 0, n_ord[n]);
	}

	if (N3 > 1){
		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n]);
		}

		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, Bufferp_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, Bufferph_1, 0, n_ord[n]);
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");
	//MPI communication
	mpi_synch();

	if (rank == 0){
		end2 = clock();
		time_spent3 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	}
	receive_tag = 0;

	#if (STAGGERED && COPY_BFIELD)
	/*rc = 0;
	if (nstep % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){ //watch out does this for both half and full timestep while only needed for full timestep
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_send3(ps, Bufferps_1, n_ord[n]);
		}
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_rec1(ps, Bufferps_1, n_ord[n]);
		for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1)B_rec2(ps, Bufferps_1, n_ord[n]);
		if (N3 > 1){
			for (n = 0; n < n_active; n++) if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) B_rec3(ps, Bufferps_1, n_ord[n]);
		}
	}*/
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomB \n");
	#endif
	nstep = temp;
}

void GPU_boundprim1(int flag, int n)
{
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[n]);
		}
		else{
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[n]);
		}
		if (status != 0) printf("Error boundprim1 %d\n", status);
	}
}

void GPU_boundprim2(int flag, int n)
{
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[n]);
		}
		else{
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[n]);
		}
		if (status != 0) printf("Error boundprim2 %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k, l, pg, d1, d2;
	status = cudaMemcpy(p_1[n], Bufferp_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	status += cudaMemcpy(ph_1[n], Bufferph_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	#if(STAGGERED)
	status += cudaMemcpy(ps_1[n], Bufferps_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	status += cudaMemcpy(psh_1[n], Bufferpsh_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(FTYPE2), cudaMemcpyDeviceToHost);
	#endif
	status += cudaMemcpy(failimage_GPU[n], Bufferfailimage[n], (int)((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(int), cudaMemcpyDeviceToHost);

	#pragma omp parallel private(i, j, z, k, l, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[n][index(n, i, j, z)][k] = p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				ph[n][index(n, i, j, z)][k] = ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];

				#if(GPU_DEBUG)
				ph[n][index(n, i, j, z)][k] = ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			#endif
			}
			for (k = 0; k < NFAIL; k++){
				failimage[n][index(n, i, j, z)][k] = failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps[n][index(n, i, j, z)][k] = ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				psh[n][index(n, i, j, z)][k] = psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
		#endif
		}
	}
	if (status != 0)printf("Error in GPU_read: %d \n", status);
}


void GPU_finish(int n)
{
	int i;

	//Select correct CUDA device
	cudaSetDevice(block[n][AMR_GPU]);

	//Destroy CUDA events associated with block
	for (i = 0; i < 600; i++) cudaEventDestroy(boundevent[n][i]);
	for (i = 0; i < 100; i++) cudaEventDestroy(boundevent1[n][i]);
	for (i = 0; i < 100; i++) cudaEventDestroy(boundevent2[n][i]);

	free(p_1[n]);
	free(dq_1[n]);
	#if(STAGGERED)
	free(ps_1[n]);
	free(psh_1[n]);
	#endif
	free(ph_1[n]);
	free(dU_GPU[n]);
	free(pbound_1[n]);
	free(pflag_GPU[n]);
	free(failimage_GPU[n]);
	free(Katm_GPU[n]);
	
	status += cudaFree(Bufferdtij[n]);
	status += cudaFree(BufferF1_1[n]);
	status += cudaFree(BufferF2_1[n]);
	status += cudaFree(BufferF3_1[n]);
	status += cudaFree(Bufferdq_1[n]);
	status += cudaFree(BufferE_1[n]);
	#if(LEER)
	status += cudaFree(BufferV[n]);
	#endif
	status += cudaFree(Bufferradius[n]);
	status += cudaFree(Bufferstorage1[n]);
	status += cudaFree(Bufferstorage2[n]);
	status += cudaFree(Bufferstorage3[n]);
	status += cudaFree(Bufferstorage4[n]);
	status += cudaFree(Bufferp_1[n]);
	status += cudaFree(Bufferph_1[n]);
	#if(STAGGERED)
	status += cudaFree(Bufferps_1[n]);
	status += cudaFree(Bufferpsh_1[n]);
	#endif
	status += cudaFree(Bufferpbound_1[n]);
	status += cudaFree(Bufferpflag[n]);
	status += cudaFree(Bufferfailimage[n]);
	status += cudaFree(Bufferdiagflux[n]);
	status += cudaFree(BufferKatm[n]);
	status += cudaFree(BufferdU[n]);
	status += cudaFree(Buffergcov[n]);
	status += cudaFree(Buffergcon[n]);
	status += cudaFree(Bufferconn[n]);
	status += cudaFree(Buffergdet[n]);
	status += cudaFree(Buffersend1[n]);
	status += cudaFree(Buffersend1_3[n]);
	status += cudaFree(Buffersend1_4[n]);
	status += cudaFree(Buffersend1_7[n]);
	status += cudaFree(Buffersend1_8[n]);
	status += cudaFree(Buffersend2[n]);
	status += cudaFree(Buffersend2_1[n]);
	status += cudaFree(Buffersend2_2[n]);
	status += cudaFree(Buffersend2_3[n]);
	status += cudaFree(Buffersend2_4[n]);
	status += cudaFree(Buffersend3[n]);
	status += cudaFree(Buffersend3_1[n]);
	status += cudaFree(Buffersend3_2[n]);
	status += cudaFree(Buffersend3_5[n]);
	status += cudaFree(Buffersend3_6[n]);
	status += cudaFree(Buffersend4[n]);
	status += cudaFree(Buffersend4_5[n]);
	status += cudaFree(Buffersend4_6[n]);
	status += cudaFree(Buffersend4_7[n]);
	status += cudaFree(Buffersend4_8[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5[n]);
	status += cudaFree(Buffersend5_1[n]);
	status += cudaFree(Buffersend5_3[n]);
	status += cudaFree(Buffersend5_5[n]);
	status += cudaFree(Buffersend5_7[n]);
	status += cudaFree(Buffersend6[n]);
	status += cudaFree(Buffersend6_2[n]);
	status += cudaFree(Buffersend6_4[n]);
	status += cudaFree(Buffersend6_6[n]);
	status += cudaFree(Buffersend6_8[n]);
	#endif
	status += cudaFree(Bufferrec1[n]);
	status += cudaFree(Bufferrec1_3[n]);
	status += cudaFree(Bufferrec1_4[n]);
	status += cudaFree(Bufferrec1_7[n]);
	status += cudaFree(Bufferrec1_8[n]);
	status += cudaFree(Bufferrec2[n]);
	status += cudaFree(Bufferrec2_1[n]);
	status += cudaFree(Bufferrec2_2[n]);
	status += cudaFree(Bufferrec2_3[n]);
	status += cudaFree(Bufferrec2_4[n]);
	status += cudaFree(Bufferrec3[n]);
	status += cudaFree(Bufferrec3_1[n]);
	status += cudaFree(Bufferrec3_2[n]);
	status += cudaFree(Bufferrec3_5[n]);
	status += cudaFree(Bufferrec3_6[n]);
	status += cudaFree(Bufferrec4[n]);
	status += cudaFree(Bufferrec4_5[n]);
	status += cudaFree(Bufferrec4_6[n]);
	status += cudaFree(Bufferrec4_7[n]);
	status += cudaFree(Bufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5[n]);
	status += cudaFree(Bufferrec5_1[n]);
	status += cudaFree(Bufferrec5_3[n]);
	status += cudaFree(Bufferrec5_5[n]);
	status += cudaFree(Bufferrec5_7[n]);
	status += cudaFree(Bufferrec6[n]);
	status += cudaFree(Bufferrec6_2[n]);
	status += cudaFree(Bufferrec6_4[n]);
	status += cudaFree(Bufferrec6_6[n]);
	status += cudaFree(Bufferrec6_8[n]);
	#endif
	status += cudaFree(tempBufferrec1[n]);
	status += cudaFree(tempBufferrec1_3[n]);
	status += cudaFree(tempBufferrec1_4[n]);
	status += cudaFree(tempBufferrec1_7[n]);
	status += cudaFree(tempBufferrec1_8[n]);
	status += cudaFree(tempBufferrec2[n]);
	status += cudaFree(tempBufferrec2_1[n]);
	status += cudaFree(tempBufferrec2_2[n]);
	status += cudaFree(tempBufferrec2_3[n]);
	status += cudaFree(tempBufferrec2_4[n]);
	status += cudaFree(tempBufferrec3[n]);
	status += cudaFree(tempBufferrec3_1[n]);
	status += cudaFree(tempBufferrec3_2[n]);
	status += cudaFree(tempBufferrec3_5[n]);
	status += cudaFree(tempBufferrec3_6[n]);
	status += cudaFree(tempBufferrec4[n]);
	status += cudaFree(tempBufferrec4_5[n]);
	status += cudaFree(tempBufferrec4_6[n]);
	status += cudaFree(tempBufferrec4_7[n]);
	status += cudaFree(tempBufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrec5[n]);
	status += cudaFree(tempBufferrec5_1[n]);
	status += cudaFree(tempBufferrec5_3[n]);
	status += cudaFree(tempBufferrec5_5[n]);
	status += cudaFree(tempBufferrec5_7[n]);
	status += cudaFree(tempBufferrec6[n]);
	status += cudaFree(tempBufferrec6_2[n]);
	status += cudaFree(tempBufferrec6_4[n]);
	status += cudaFree(tempBufferrec6_6[n]);
	status += cudaFree(tempBufferrec6_8[n]);
	#endif
	status += cudaFree(Buffersend1flux[n]);
	status += cudaFree(Buffersend2flux[n]);
	status += cudaFree(Buffersend3flux[n]);
	status += cudaFree(Buffersend4flux[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5flux[n]);
	status += cudaFree(Buffersend6flux[n]);
	#endif
	status += cudaFree(Bufferrec1flux[n]);
	status += cudaFree(Bufferrec2flux[n]);
	status += cudaFree(Bufferrec3flux[n]);
	status += cudaFree(Bufferrec4flux[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5flux[n]);
	status += cudaFree(Bufferrec6flux[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux[n]);
	status += cudaFree(Bufferrec1_4flux[n]);
	status += cudaFree(Bufferrec1_7flux[n]);
	status += cudaFree(Bufferrec1_8flux[n]);
	status += cudaFree(Bufferrec2_1flux[n]);
	status += cudaFree(Bufferrec2_2flux[n]);
	status += cudaFree(Bufferrec2_3flux[n]);
	status += cudaFree(Bufferrec2_4flux[n]);
	status += cudaFree(Bufferrec3_1flux[n]);
	status += cudaFree(Bufferrec3_2flux[n]);
	status += cudaFree(Bufferrec3_5flux[n]);
	status += cudaFree(Bufferrec3_6flux[n]);
	status += cudaFree(Bufferrec4_5flux[n]);
	status += cudaFree(Bufferrec4_6flux[n]);
	status += cudaFree(Bufferrec4_7flux[n]);
	status += cudaFree(Bufferrec4_8flux[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux[n]);
	status += cudaFree(Bufferrec5_3flux[n]);
	status += cudaFree(Bufferrec5_5flux[n]);
	status += cudaFree(Bufferrec5_7flux[n]);
	status += cudaFree(Bufferrec6_2flux[n]);
	status += cudaFree(Bufferrec6_4flux[n]);
	status += cudaFree(Bufferrec6_6flux[n]);
	status += cudaFree(Bufferrec6_8flux[n]);
	#endif
	status += cudaFree(Bufferrec1flux1[n]);
	status += cudaFree(Bufferrec2flux1[n]);
	status += cudaFree(Bufferrec3flux1[n]);
	status += cudaFree(Bufferrec4flux1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5flux1[n]);
	status += cudaFree(Bufferrec6flux1[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux1[n]);
	status += cudaFree(Bufferrec1_4flux1[n]);
	status += cudaFree(Bufferrec1_7flux1[n]);
	status += cudaFree(Bufferrec1_8flux1[n]);
	status += cudaFree(Bufferrec2_1flux1[n]);
	status += cudaFree(Bufferrec2_2flux1[n]);
	status += cudaFree(Bufferrec2_3flux1[n]);
	status += cudaFree(Bufferrec2_4flux1[n]);
	status += cudaFree(Bufferrec3_1flux1[n]);
	status += cudaFree(Bufferrec3_2flux1[n]);
	status += cudaFree(Bufferrec3_5flux1[n]);
	status += cudaFree(Bufferrec3_6flux1[n]);
	status += cudaFree(Bufferrec4_5flux1[n]);
	status += cudaFree(Bufferrec4_6flux1[n]);
	status += cudaFree(Bufferrec4_7flux1[n]);
	status += cudaFree(Bufferrec4_8flux1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux1[n]);
	status += cudaFree(Bufferrec5_3flux1[n]);
	status += cudaFree(Bufferrec5_5flux1[n]);
	status += cudaFree(Bufferrec5_7flux1[n]);
	status += cudaFree(Bufferrec6_2flux1[n]);
	status += cudaFree(Bufferrec6_4flux1[n]);
	status += cudaFree(Bufferrec6_6flux1[n]);
	status += cudaFree(Bufferrec6_8flux1[n]);
	#endif
	status += cudaFree(Bufferrec1_3flux2[n]);
	status += cudaFree(Bufferrec1_4flux2[n]);
	status += cudaFree(Bufferrec1_7flux2[n]);
	status += cudaFree(Bufferrec1_8flux2[n]);
	status += cudaFree(Bufferrec2_1flux2[n]);
	status += cudaFree(Bufferrec2_2flux2[n]);
	status += cudaFree(Bufferrec2_3flux2[n]);
	status += cudaFree(Bufferrec2_4flux2[n]);
	status += cudaFree(Bufferrec3_1flux2[n]);
	status += cudaFree(Bufferrec3_2flux2[n]);
	status += cudaFree(Bufferrec3_5flux2[n]);
	status += cudaFree(Bufferrec3_6flux2[n]);
	status += cudaFree(Bufferrec4_5flux2[n]);
	status += cudaFree(Bufferrec4_6flux2[n]);
	status += cudaFree(Bufferrec4_7flux2[n]);
	status += cudaFree(Bufferrec4_8flux2[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1flux2[n]);
	status += cudaFree(Bufferrec5_3flux2[n]);
	status += cudaFree(Bufferrec5_5flux2[n]);
	status += cudaFree(Bufferrec5_7flux2[n]);
	status += cudaFree(Bufferrec6_2flux2[n]);
	status += cudaFree(Bufferrec6_4flux2[n]);
	status += cudaFree(Bufferrec6_6flux2[n]);
	status += cudaFree(Bufferrec6_8flux2[n]);
	#endif
	/*
	status += cudaFree(Buffersend1fine[n]);
	status += cudaFree(Buffersend2fine[n]);
	status += cudaFree(Buffersend3fine[n]);
	status += cudaFree(Buffersend4fine[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5fine[n]);
	status += cudaFree(Buffersend6fine[n]);
	#endif
	status += cudaFree(Bufferrec1fine[n]);
	status += cudaFree(Bufferrec2fine[n]);
	status += cudaFree(Bufferrec3fine[n]);
	status += cudaFree(Bufferrec4fine[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5fine[n]);
	status += cudaFree(Bufferrec6fine[n]);
	#endif
	status += cudaFree(Bufferrec1_3fine[n]);
	status += cudaFree(Bufferrec1_4fine[n]);
	status += cudaFree(Bufferrec1_7fine[n]);
	status += cudaFree(Bufferrec1_8fine[n]);
	status += cudaFree(Bufferrec2_1fine[n]);
	status += cudaFree(Bufferrec2_2fine[n]);
	status += cudaFree(Bufferrec2_3fine[n]);
	status += cudaFree(Bufferrec2_4fine[n]);
	status += cudaFree(Bufferrec3_1fine[n]);
	status += cudaFree(Bufferrec3_2fine[n]);
	status += cudaFree(Bufferrec3_5fine[n]);
	status += cudaFree(Bufferrec3_6fine[n]);
	status += cudaFree(Bufferrec4_5fine[n]);
	status += cudaFree(Bufferrec4_6fine[n]);
	status += cudaFree(Bufferrec4_7fine[n]);
	status += cudaFree(Bufferrec4_8fine[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1fine[n]);
	status += cudaFree(Bufferrec5_3fine[n]);
	status += cudaFree(Bufferrec5_5fine[n]);
	status += cudaFree(Bufferrec5_7fine[n]);
	status += cudaFree(Bufferrec6_2fine[n]);
	status += cudaFree(Bufferrec6_4fine[n]);
	status += cudaFree(Bufferrec6_6fine[n]);
	status += cudaFree(Bufferrec6_8fine[n]);
	#endif
	*/
	status += cudaFree(Buffersend1E[n]);
	status += cudaFree(Buffersend2E[n]);
	status += cudaFree(Buffersend3E[n]);
	status += cudaFree(Buffersend4E[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5E[n]);
	status += cudaFree(Buffersend6E[n]);
	#endif
	status += cudaFree(Bufferrec1E[n]);
	status += cudaFree(Bufferrec2E[n]);
	status += cudaFree(Bufferrec3E[n]);
	status += cudaFree(Bufferrec4E[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5E[n]);
	status += cudaFree(Bufferrec6E[n]);
	#endif
	status += cudaFree(Bufferrec1_3E[n]);
	status += cudaFree(Bufferrec1_4E[n]);
	status += cudaFree(Bufferrec1_7E[n]);
	status += cudaFree(Bufferrec1_8E[n]);
	status += cudaFree(Bufferrec2_1E[n]);
	status += cudaFree(Bufferrec2_2E[n]);
	status += cudaFree(Bufferrec2_3E[n]);
	status += cudaFree(Bufferrec2_4E[n]);
	status += cudaFree(Bufferrec3_1E[n]);
	status += cudaFree(Bufferrec3_2E[n]);
	status += cudaFree(Bufferrec3_5E[n]);
	status += cudaFree(Bufferrec3_6E[n]);
	status += cudaFree(Bufferrec4_5E[n]);
	status += cudaFree(Bufferrec4_6E[n]);
	status += cudaFree(Bufferrec4_7E[n]);
	status += cudaFree(Bufferrec4_8E[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E[n]);
	status += cudaFree(Bufferrec5_3E[n]);
	status += cudaFree(Bufferrec5_5E[n]);
	status += cudaFree(Bufferrec5_7E[n]);
	status += cudaFree(Bufferrec6_2E[n]);
	status += cudaFree(Bufferrec6_4E[n]);
	status += cudaFree(Bufferrec6_6E[n]);
	status += cudaFree(Bufferrec6_8E[n]);
	#endif
	status += cudaFree(Bufferrec1E1[n]);
	status += cudaFree(Bufferrec2E1[n]);
	status += cudaFree(Bufferrec3E1[n]);
	status += cudaFree(Bufferrec4E1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5E1[n]);
	status += cudaFree(Bufferrec6E1[n]);
	#endif
	status += cudaFree(Bufferrec1_3E1[n]);
	status += cudaFree(Bufferrec1_4E1[n]);
	status += cudaFree(Bufferrec1_7E1[n]);
	status += cudaFree(Bufferrec1_8E1[n]);
	status += cudaFree(Bufferrec2_1E1[n]);
	status += cudaFree(Bufferrec2_2E1[n]);
	status += cudaFree(Bufferrec2_3E1[n]);
	status += cudaFree(Bufferrec2_4E1[n]);
	status += cudaFree(Bufferrec3_1E1[n]);
	status += cudaFree(Bufferrec3_2E1[n]);
	status += cudaFree(Bufferrec3_5E1[n]);
	status += cudaFree(Bufferrec3_6E1[n]);
	status += cudaFree(Bufferrec4_5E1[n]);
	status += cudaFree(Bufferrec4_6E1[n]);
	status += cudaFree(Bufferrec4_7E1[n]);
	status += cudaFree(Bufferrec4_8E1[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E1[n]);
	status += cudaFree(Bufferrec5_3E1[n]);
	status += cudaFree(Bufferrec5_5E1[n]);
	status += cudaFree(Bufferrec5_7E1[n]);
	status += cudaFree(Bufferrec6_2E1[n]);
	status += cudaFree(Bufferrec6_4E1[n]);
	status += cudaFree(Bufferrec6_6E1[n]);
	status += cudaFree(Bufferrec6_8E1[n]);
	#endif
	status += cudaFree(Bufferrec1_3E2[n]);
	status += cudaFree(Bufferrec1_4E2[n]);
	status += cudaFree(Bufferrec1_7E2[n]);
	status += cudaFree(Bufferrec1_8E2[n]);
	status += cudaFree(Bufferrec2_1E2[n]);
	status += cudaFree(Bufferrec2_2E2[n]);
	status += cudaFree(Bufferrec2_3E2[n]);
	status += cudaFree(Bufferrec2_4E2[n]);
	status += cudaFree(Bufferrec3_1E2[n]);
	status += cudaFree(Bufferrec3_2E2[n]);
	status += cudaFree(Bufferrec3_5E2[n]);
	status += cudaFree(Bufferrec3_6E2[n]);
	status += cudaFree(Bufferrec4_5E2[n]);
	status += cudaFree(Bufferrec4_6E2[n]);
	status += cudaFree(Bufferrec4_7E2[n]);
	status += cudaFree(Bufferrec4_8E2[n]);
	#if(N3G>0)
	status += cudaFree(Bufferrec5_1E2[n]);
	status += cudaFree(Bufferrec5_3E2[n]);
	status += cudaFree(Bufferrec5_5E2[n]);
	status += cudaFree(Bufferrec5_7E2[n]);
	status += cudaFree(Bufferrec6_2E2[n]);
	status += cudaFree(Bufferrec6_4E2[n]);
	status += cudaFree(Bufferrec6_6E2[n]);
	status += cudaFree(Bufferrec6_8E2[n]);
	#endif
	#if(N3G>0)
	status += cudaFree(BuffersendE1corn9[n]);
	status += cudaFree(BuffersendE1corn10[n]);
	status += cudaFree(BuffersendE1corn11[n]);
	status += cudaFree(BuffersendE1corn12[n]);
	status += cudaFree(BuffersendE2corn5[n]);
	status += cudaFree(BuffersendE2corn6[n]);
	status += cudaFree(BuffersendE2corn7[n]);
	status += cudaFree(BuffersendE2corn8[n]);
	#endif
	status += cudaFree(BuffersendE3corn1[n]);
	status += cudaFree(BuffersendE3corn2[n]);
	status += cudaFree(BuffersendE3corn3[n]);
	status += cudaFree(BuffersendE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9[n]);
	status += cudaFree(BufferrecE1corn10[n]);
	status += cudaFree(BufferrecE1corn11[n]);
	status += cudaFree(BufferrecE1corn12[n]);
	status += cudaFree(BufferrecE2corn5[n]);
	status += cudaFree(BufferrecE2corn6[n]);
	status += cudaFree(BufferrecE2corn7[n]);
	status += cudaFree(BufferrecE2corn8[n]);
	#endif
	status += cudaFree(BufferrecE3corn1[n]);
	status += cudaFree(BufferrecE3corn2[n]);
	status += cudaFree(BufferrecE3corn3[n]);
	status += cudaFree(BufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9_3[n]);
	status += cudaFree(BufferrecE1corn9_7[n]);
	status += cudaFree(BufferrecE1corn10_1[n]);
	status += cudaFree(BufferrecE1corn10_5[n]);
	status += cudaFree(BufferrecE1corn11_2[n]);
	status += cudaFree(BufferrecE1corn11_6[n]);
	status += cudaFree(BufferrecE1corn12_4[n]);
	status += cudaFree(BufferrecE1corn12_8[n]);
	status += cudaFree(BufferrecE2corn5_2[n]);
	status += cudaFree(BufferrecE2corn5_4[n]);
	status += cudaFree(BufferrecE2corn6_1[n]);
	status += cudaFree(BufferrecE2corn6_3[n]);
	status += cudaFree(BufferrecE2corn7_5[n]);
	status += cudaFree(BufferrecE2corn7_7[n]);
	status += cudaFree(BufferrecE2corn8_6[n]);
	status += cudaFree(BufferrecE2corn8_8[n]);
	#endif
	status += cudaFree(BufferrecE3corn1_3[n]);
	status += cudaFree(BufferrecE3corn1_4[n]);
	status += cudaFree(BufferrecE3corn2_1[n]);
	status += cudaFree(BufferrecE3corn2_2[n]);
	status += cudaFree(BufferrecE3corn3_5[n]);
	status += cudaFree(BufferrecE3corn3_6[n]);
	status += cudaFree(BufferrecE3corn4_7[n]);
	status += cudaFree(BufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrecE1corn9[n]);
	status += cudaFree(tempBufferrecE1corn10[n]);
	status += cudaFree(tempBufferrecE1corn11[n]);
	status += cudaFree(tempBufferrecE1corn12[n]);
	status += cudaFree(tempBufferrecE2corn5[n]);
	status += cudaFree(tempBufferrecE2corn6[n]);
	status += cudaFree(tempBufferrecE2corn7[n]);
	status += cudaFree(tempBufferrecE2corn8[n]);
	#endif
	status += cudaFree(tempBufferrecE3corn1[n]);
	status += cudaFree(tempBufferrecE3corn2[n]);
	status += cudaFree(tempBufferrecE3corn3[n]);
	status += cudaFree(tempBufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFree(tempBufferrecE1corn9_3[n]);
	status += cudaFree(tempBufferrecE1corn9_7[n]);
	status += cudaFree(tempBufferrecE1corn10_1[n]);
	status += cudaFree(tempBufferrecE1corn10_5[n]);
	status += cudaFree(tempBufferrecE1corn11_2[n]);
	status += cudaFree(tempBufferrecE1corn11_6[n]);
	status += cudaFree(tempBufferrecE1corn12_4[n]);
	status += cudaFree(tempBufferrecE1corn12_8[n]);
	status += cudaFree(tempBufferrecE2corn5_2[n]);
	status += cudaFree(tempBufferrecE2corn5_4[n]);
	status += cudaFree(tempBufferrecE2corn6_1[n]);
	status += cudaFree(tempBufferrecE2corn6_3[n]);
	status += cudaFree(tempBufferrecE2corn7_5[n]);
	status += cudaFree(tempBufferrecE2corn7_7[n]);
	status += cudaFree(tempBufferrecE2corn8_6[n]);
	status += cudaFree(tempBufferrecE2corn8_8[n]);
	#endif
	status += cudaFree(tempBufferrecE3corn1_3[n]);
	status += cudaFree(tempBufferrecE3corn1_4[n]);
	status += cudaFree(tempBufferrecE3corn2_1[n]);
	status += cudaFree(tempBufferrecE3corn2_2[n]);
	status += cudaFree(tempBufferrecE3corn3_5[n]);
	status += cudaFree(tempBufferrecE3corn3_6[n]);
	status += cudaFree(tempBufferrecE3corn4_7[n]);
	status += cudaFree(tempBufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFree(BufferrecE1corn9_32[n]);
	status += cudaFree(BufferrecE1corn9_72[n]);
	status += cudaFree(BufferrecE1corn10_12[n]);
	status += cudaFree(BufferrecE1corn10_52[n]);
	status += cudaFree(BufferrecE1corn11_22[n]);
	status += cudaFree(BufferrecE1corn11_62[n]);
	status += cudaFree(BufferrecE1corn12_42[n]);
	status += cudaFree(BufferrecE1corn12_82[n]);
	status += cudaFree(BufferrecE2corn5_22[n]);
	status += cudaFree(BufferrecE2corn5_42[n]);
	status += cudaFree(BufferrecE2corn6_12[n]);
	status += cudaFree(BufferrecE2corn6_32[n]);
	status += cudaFree(BufferrecE2corn7_52[n]);
	status += cudaFree(BufferrecE2corn7_72[n]);
	status += cudaFree(BufferrecE2corn8_62[n]);
	status += cudaFree(BufferrecE2corn8_82[n]);
	#endif
	status += cudaFree(BufferrecE3corn1_32[n]);
	status += cudaFree(BufferrecE3corn1_42[n]);
	status += cudaFree(BufferrecE3corn2_12[n]);
	status += cudaFree(BufferrecE3corn2_22[n]);
	status += cudaFree(BufferrecE3corn3_52[n]);
	status += cudaFree(BufferrecE3corn3_62[n]);
	status += cudaFree(BufferrecE3corn4_72[n]);
	status += cudaFree(BufferrecE3corn4_82[n]);

	if (status != 0) printf("Error in GPU_finish_1: %d \n", status);
}
}