#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}

void GPU_init(void)
{
	int j, pos;
	
	//Create concurrent commandqueues
	////cudaSetDevice(local_rank%N_GPU);
	//for (j = 0; j < NQ; j++) cudaStreamCreate(&commandQueue[j]);

	for (j = 0; j < numdevices; j++){
		//cudaDeviceCanAccessPeer(&pos, local_rank%numdevices, j);
		//if (pos==1) cudaDeviceEnablePeerAccess(j, 0);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting cache: %d \n", status);
}

void set_arrays_GPU(int n, int device){
	int i, j, z;

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
	nr_workgroups1[n] = (int)ceil((double)global_work_size1[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2[n] = (int)ceil((double)global_work_size2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_1[n] = (int)ceil((double)global_work_size2_1[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_2[n] = (int)ceil((double)global_work_size2_2[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups2_3[n] = (int)ceil((double)global_work_size2_3[n][0] / (double)LOCAL_WORK_SIZE);
	nr_workgroups3[n] = (int)ceil((double)global_work_size3[n][0] / (double)LOCAL_WORK_SIZE);

	//Select correct CUDA device
	cudaStreamCreate(&commandQueueGPU[n]);

	//Create events
	for (i = 0; i < 600; i++) cudaEventCreate(&boundevent[n][i]);
	for (i = 0; i < 100; i++) cudaEventCreate(&boundevent1[n][i]);

	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in creating events: %d \n", status);

	/*Allocate memory to 1D arrays*/
	p_1[n] = (double(*))malloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(double));
	dq_1[n] = (double(*))malloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(double)); //array to store temporary data
	#if(STAGGERED)
	ps_1[n] = (double(*))malloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(double));
	psh_1[n] = (double(*))malloc(NDIM * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(double));
	#endif
	ph_1[n] = (double(*))malloc(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(double));
	//dU_GPU[n] = (double(*))malloc(NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G))* sizeof(double));
	#if(!NONSYMMETRIC)
	gcov_GPU[n] = (double(*))malloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM* sizeof(double));
	gcon_GPU[n] = (double(*))malloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM* sizeof(double));
	conn_GPU[n] = (double(*))malloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM* sizeof(double));
	gdet_GPU[n] = (double(*))malloc(((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG* sizeof(double));
	#else
	gcov_GPU[n] = (double(*))mcalloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM* sizeof(double));
	gcon_GPU[n] = (double(*))malloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG*NDIM*NDIM* sizeof(double));
	conn_GPU[n] = (double(*))malloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NDIM*NDIM*NDIM* sizeof(double));
	gdet_GPU[n] = (double(*))malloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*NPG * sizeof(double));
	#endif
	//pflag_GPU[n] = (int(*))malloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])* sizeof(int));
	failimage_GPU[n] = (int(*))malloc(((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL *sizeof(int));
	Katm_GPU[n] = (double(*))malloc((N1_GPU[n] + 2 * N1G) * sizeof(double));

	/*Allocate memory to buffers on GPU*/
	cudaMalloc(&BufferF1_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&BufferF2_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&BufferF3_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&Bufferdq_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&BufferE_1[n], NDIM*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	#if(LEER)
	cudaMalloc(&BufferV[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	#endif
	cudaMalloc(&Bufferradius[n], (N1_GPU[n] + 2 * N1G)*sizeof(double));
	cudaMalloc(&Bufferstorage1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&Bufferp_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&Bufferph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	#if(STAGGERED)
	cudaMalloc(&Bufferps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	cudaMalloc(&Bufferpsh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double));
	#endif
	cudaMalloc(&Buffergdet[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*sizeof(double));
	cudaMalloc(&Buffergcov[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Buffergcon[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Bufferconn[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Bufferpflag[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int));
	cudaMalloc(&Bufferfailimage[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(int));
	cudaMalloc(&BufferKatm[n], (N1_GPU[n] + 2 * N1G)*sizeof(double));
	//cudaMalloc(&BufferdU[n], NPR*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G))*sizeof(double));
	if (cudaSuccess != cudaGetLastError() ) fprintf(stderr, "Error in setting kernel arguments 3: %d \n", cudaGetLastError());
	cudaHostAlloc(&dtij_GPU[n], (nr_workgroups[n] + 1)*sizeof(double), 0);
	//cudaMalloc(&Bufferdtij[n], (nr_workgroups[n] + 1)*sizeof(double));
	#if(GPU_DIRECT)
	//cudaMalloc(&NULL_POINTER[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Buffersend2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Buffersend3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Buffersend4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Buffersend4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	#if(N3G>0)
	cudaMalloc(&Buffersend5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend5_1[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend5_3[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend5_5[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend5_7[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	#endif
	cudaMalloc(&Buffersend6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Buffersend6_2[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend6_4[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend6_6[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Buffersend6_8[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#endif
	cudaMalloc(&Bufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#endif
	#if(N3G>0)
	cudaMalloc(&Bufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#endif
	#endif
	cudaMalloc(&tempBufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&tempBufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&tempBufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	cudaMalloc(&tempBufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&tempBufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	#if(N3G>0)
	cudaMalloc(&tempBufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	cudaMalloc(&tempBufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1)
	cudaMalloc(&tempBufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&tempBufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#endif
	cudaMalloc(&Buffersend1flux[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Buffersend2flux[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Buffersend3flux[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Buffersend4flux[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Buffersend5flux[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	cudaMalloc(&Buffersend6flux[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec1flux[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec2flux[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec3flux[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec4flux[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5flux[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec6flux[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_1flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_2flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_3flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_4flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_5flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_6flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_7flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_8flux[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	#endif
	#endif
	cudaMalloc(&Bufferrec1flux1[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec2flux1[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec3flux1[n], NPR*(N1_GPU[n])*(N3_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec4flux1[n], NPR*(N2_GPU[n])*(N3_GPU[n])*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5flux1[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	cudaMalloc(&Bufferrec6flux1[n], NPR*(N1_GPU[n])*(N2_GPU[n])*sizeof(double));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_1flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_2flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_3flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_4flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_5flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_6flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_7flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_8flux1[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec1_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec1_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_1flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_2flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_3flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec2_4flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec3_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_5flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_6flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_7flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	cudaMalloc(&Bufferrec4_8flux2[n], NPR*(N2_GPU[n] / (1 + REF_2))*(N3_GPU[n] / (1 + REF_3))*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec5_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	cudaMalloc(&Bufferrec6_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1))*(N2_GPU[n] / (1 + REF_2))*sizeof(double));
	#endif
	#endif

	cudaHostAlloc(&Buffersend1fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double), 0);
	//cudaHostAlloc(&Buffersend2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double), 0);
	//cudaHostAlloc(&Buffersend4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	//cudaHostAlloc(&Buffersend5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	//cudaHostAlloc(&Buffersend6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double), 0);
	//cudaHostAlloc(&Bufferrec2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double), 0);
	//cudaHostAlloc(&Bufferrec4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	//cudaHostAlloc(&Bufferrec5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	//cudaHostAlloc(&Bufferrec6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif

	/*#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec1_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_1fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_2fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_3fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2_4fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_5fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_6fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_7fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4_8fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec5_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#endif*/
	cudaMalloc(&Buffersend1E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Buffersend2E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Buffersend3E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Buffersend4E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Buffersend5E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&Buffersend6E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	#endif
	cudaMalloc(&Bufferrec1E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_4E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_7E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_8E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_1E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_2E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_3E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_4E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_1E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_2E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_5E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_6E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_5E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_6E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_7E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_8E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_3E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_5E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_7E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_2E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_4E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_6E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_8E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	#endif
	cudaMalloc(&Bufferrec1E1[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec2E1[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec3E1[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&Bufferrec4E1[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5E1[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&Bufferrec6E1[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double));
	#endif
	#if(N_LEVELS>1)
	cudaMalloc(&Bufferrec1_3E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_4E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_7E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_8E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_1E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_2E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_3E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_4E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_1E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_2E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_5E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_6E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_5E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_6E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_7E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_8E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_3E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_5E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_7E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_2E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_4E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_6E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_8E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	cudaMalloc(&Bufferrec1_3E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_4E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_7E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec1_8E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_1E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_2E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_3E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec2_4E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_1E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_2E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_5E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec3_6E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_5E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_6E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_7E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&Bufferrec4_8E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#if(N3G>0)
	cudaMalloc(&Bufferrec5_1E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_3E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_5E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec5_7E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_2E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_4E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_6E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&Bufferrec6_8E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	#endif
	#if(N3G>0)
	cudaMalloc(&BuffersendE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BuffersendE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BuffersendE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BuffersendE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BuffersendE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BuffersendE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BuffersendE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BuffersendE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	#endif
	cudaMalloc(&BuffersendE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BuffersendE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BuffersendE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BuffersendE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BufferrecE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BufferrecE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BufferrecE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&BufferrecE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BufferrecE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BufferrecE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&BufferrecE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	#endif
	cudaMalloc(&BufferrecE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BufferrecE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BufferrecE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&BufferrecE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9_3[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn9_7[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn10_1[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn10_5[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn11_2[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn11_6[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn12_4[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn12_8[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE2corn5_2[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn5_4[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn6_1[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn6_3[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn7_5[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn7_7[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn8_6[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn8_8[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	cudaMalloc(&BufferrecE3corn1_3[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn1_4[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn2_1[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn2_2[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn3_5[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn3_6[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn4_7[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn4_8[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#endif
	#if(N3G>0)
	cudaMalloc(&tempBufferrecE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&tempBufferrecE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&tempBufferrecE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&tempBufferrecE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double));
	cudaMalloc(&tempBufferrecE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&tempBufferrecE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&tempBufferrecE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	cudaMalloc(&tempBufferrecE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double));
	#endif
	cudaMalloc(&tempBufferrecE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&tempBufferrecE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&tempBufferrecE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	cudaMalloc(&tempBufferrecE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaMalloc(&tempBufferrecE1corn9_3[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn9_7[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn10_1[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn10_5[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn11_2[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn11_6[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn12_4[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE1corn12_8[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn5_2[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn5_4[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn6_1[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn6_3[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn7_5[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn7_7[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn8_6[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&tempBufferrecE2corn8_8[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	cudaMalloc(&tempBufferrecE3corn1_3[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn1_4[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn2_1[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn2_2[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn3_5[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn3_6[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn4_7[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&tempBufferrecE3corn4_8[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#if(N3G>0)
	cudaMalloc(&BufferrecE1corn9_32[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn9_72[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn10_12[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn10_52[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn11_22[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn11_62[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn12_42[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE1corn12_82[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double));
	cudaMalloc(&BufferrecE2corn5_22[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn5_42[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn6_12[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn6_32[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn7_52[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn7_72[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn8_62[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	cudaMalloc(&BufferrecE2corn8_82[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double));
	#endif
	cudaMalloc(&BufferrecE3corn1_32[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn1_42[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn2_12[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn2_22[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn3_52[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn3_62[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn4_72[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	cudaMalloc(&BufferrecE3corn4_82[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double));
	#endif
	#else
	//cudaHostAlloc(&NULL_POINTER[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Buffersend2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Buffersend3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Buffersend4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&Buffersend5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend5_1[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend5_3[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend5_5[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend5_7[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Buffersend6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Buffersend6_2[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6_4[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6_6[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6_8[n], NG * (NPR + 3)*(N1_GPU[n]/ (1+REF_1) + 2 * N1G)*(N2_GPU[n]/(1+REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&Bufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&tempBufferrec1[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec1_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec1_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec1_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec1_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrec2[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec2_1[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec2_2[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec2_3[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec2_4[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrec3[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec3_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec3_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec3_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec3_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrec4[n], NG * (NPR + 3)*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec4_5[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec4_6[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec4_7[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec4_8[n], NG * (NPR + 3)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&tempBufferrec5[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec5_1[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec5_3[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec5_5[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec5_7[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrec6[n], NG * (NPR + 3)*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#if(N_LEVELS>1)
	cudaHostAlloc(&tempBufferrec6_2[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec6_4[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec6_6[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrec6_8[n], NG * (NPR + 3)*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&Buffersend1flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Buffersend5flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4flux[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6flux[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8flux[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8flux[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&Bufferrec1flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4flux1[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6flux1[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8flux1[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8flux1[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8flux2[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8flux2[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	#endif

	cudaHostAlloc(&Buffersend1fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Buffersend5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4fine[n], NPR*(N2_GPU[n] + 2 * N2G)*(N3_GPU[n] + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6fine[n], NPR*(N1_GPU[n] + 2 * N1G)*(N2_GPU[n] + 2 * N2G)*sizeof(double),0);
	#endif

	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8fine[n], NPR*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*(N3_GPU[n] / (1 + REF_3) + 2 * N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8fine[n], NPR*(N1_GPU[n] / (1 + REF_1) + 2 * N1G)*(N2_GPU[n] / (1 + REF_2) + 2 * N2G)*sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&Buffersend1E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend2E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend3E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend4E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Buffersend5E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&Buffersend6E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3E[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4E[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6E[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8E[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8E[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	#endif
	cudaHostAlloc(&Bufferrec1E1[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec2E1[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec3E1[n], 2 * (N1_GPU[n] + N1G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec4E1[n], 2 * (N2_GPU[n] + N2G)*(N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5E1[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&Bufferrec6E1[n], 2 * (N1_GPU[n] + N1G)*(N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	#if(N_LEVELS>1)
	cudaHostAlloc(&Bufferrec1_3E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8E1[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8E1[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	cudaHostAlloc(&Bufferrec1_3E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_4E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_7E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec1_8E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_1E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_2E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_3E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec2_4E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_1E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_2E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_5E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec3_6E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_5E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_6E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_7E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec4_8E2[n], 2 * (N2_GPU[n] + N2G) / (1 + REF_2) *(N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&Bufferrec5_1E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_3E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_5E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec5_7E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_2E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_4E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_6E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&Bufferrec6_8E2[n], 2 * (N1_GPU[n] + N1G) / (1 + REF_1) *(N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	#endif
	#if(N3G>0)
	cudaHostAlloc(&BuffersendE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&BuffersendE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BuffersendE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&BufferrecE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&BufferrecE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostAlloc(&BufferrecE1corn9_3[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn9_7[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn10_1[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn10_5[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn11_2[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn11_6[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn12_4[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn12_8[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn5_2[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn5_4[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn6_1[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn6_3[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn7_5[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn7_7[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn8_6[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn8_8[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	cudaHostAlloc(&BufferrecE3corn1_3[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn1_4[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn2_1[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn2_2[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn3_5[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn3_6[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn4_7[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn4_8[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#endif
	#if(N3G>0)
	cudaHostAlloc(&tempBufferrecE1corn9[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn10[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn11[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn12[n], 1 * (N1_GPU[n] + N1G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn5[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn6[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn7[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn8[n], 1 * (N2_GPU[n] + N2G)*sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrecE3corn1[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn2[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn3[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn4[n], 1 * (N3_GPU[n] + N3G)*sizeof(double),0);
	#if(N_LEVELS>1)
	#if(N3G>0)
	cudaHostAlloc(&tempBufferrecE1corn9_3[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn9_7[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn10_1[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn10_5[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn11_2[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn11_6[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn12_4[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE1corn12_8[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn5_2[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn5_4[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn6_1[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn6_3[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn7_5[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn7_7[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn8_6[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE2corn8_8[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	cudaHostAlloc(&tempBufferrecE3corn1_3[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn1_4[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn2_1[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn2_2[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn3_5[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn3_6[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn4_7[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&tempBufferrecE3corn4_8[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#if(N3G>0)
	cudaHostAlloc(&BufferrecE1corn9_32[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn9_72[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn10_12[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn10_52[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn11_22[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn11_62[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn12_42[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE1corn12_82[n], 1 * (N1_GPU[n] + N1G) / (1 + REF_1) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn5_22[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn5_42[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn6_12[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn6_32[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn7_52[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn7_72[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn8_62[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE2corn8_82[n], 1 * (N2_GPU[n] + N2G) / (1 + REF_2) *sizeof(double),0);
	#endif
	cudaHostAlloc(&BufferrecE3corn1_32[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn1_42[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn2_12[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn2_22[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn3_52[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn3_62[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn4_72[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	cudaHostAlloc(&BufferrecE3corn4_82[n], 1 * (N3_GPU[n] + N3G) / (1 + REF_3) *sizeof(double),0);
	#endif
	#endif
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting kernel arguments 4.6: %d \n", status);

	/*Copy metric to GPU*/
	int pg, d1, d2;
	#pragma omp parallel private(i, j, z, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]){
			for (pg = 0; pg < NPG; pg++){
				#if(!NONSYMMETRIC)
				gdet_GPU[n][pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gdet[n][index_2D(n, i, j, z)][pg];
				#else
				#endif
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						gcov_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[n][index_2D(n, i, j, z)][pg][d1][d2];
						gcon_GPU[n][d1*NDIM*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + d2*NPG*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + pg*((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n]) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[n][index_2D(n, i, j, z)][pg][d1][d2];
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
							+ (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[n][index_2D(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}
			#if(LEER)
			for (k = 0; k < 6; k++){
				dq_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = V[n][index_3D(n, i, j, z)][k];
			}
			#endif
		}
	}

	#if(!NONSYMMETRIC)
	cudaMemcpyAsync(Buffergdet[n], gdet_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(Buffergcov[n], gcov_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(Buffergcon[n], gcon_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NPG*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(Bufferconn[n], conn_GPU[n], ((N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem2[n])*NDIM*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	#else
	#endif
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting kernel arguments 5: %d \n", cudaGetLastError());
}
double check = 1.0;

void GPU_write(int n)
{
	int i, j, z, k;
	double radius_GPU[(N1 + 2 * N1G)], r, th, phi, X[NDIM];
	for (i = N1_GPU_offset[n] - N1G; i < N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		coord(n, i, 0, 0, CENT, X);
		bl_coord(X, &r, &th, &phi);
		radius_GPU[(i - N1_GPU_offset[n] + N1G)] = r;
		Katm_GPU[n][i - N1_GPU_offset[n] + N1G] = Katm[n][i - N1_GPU_offset[n] + N1G];
	}

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = p[n][index_3D(n, i, j, z)][k];
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index_3D(n, i, j, z)][k];
				#if(GPU_DEBUG)
				ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ph[n][index_3D(n, i, j, z)][k];
				#endif
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = ps[n][index_3D(n, i, j, z)][k];
				psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = psh[n][index_3D(n, i, j, z)][k];

			}
			#endif
			for (k = 0; k < NFAIL; k++){
				failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
			}
			//pflag_GPU[n][(i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = 0;
		}
	}

	status = 0;
	#if (ELLIPTICAL2)
	for (i = N1_GPU_offset[n] - N1G; i<N1_GPU_offset[n] + N1_GPU[n] + N1G; i++){
		for (j = N2_GPU_offset[n] - N2G; j<N2_GPU_offset[n] + N2_GPU[n] + N2G; j++){
			for (k = 0; k < NPR; k++){
				dU_GPU[n][k*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + (i - N1_GPU_offset[n] + N1G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = dU_s[n][index_2D(n, i, j, 0)][k];
			}
		}
	}
	status = cudaMemcpyAsync(BufferdU[n], dU_GPU[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	#endif
	////cudaSetDevice(block[n][AMR_GPU]);
	/*Initialize memory items that have to be passed on to the GPU*/
	cudaMemcpyAsync(Bufferp_1[n], p_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double), cudaMemcpyHostToDevice,commandQueueGPU[n]);
	cudaMemcpyAsync(Bufferph_1[n], ph_1[n], NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	#if(STAGGERED)
	cudaMemcpyAsync(Bufferps_1[n], ps_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(Bufferpsh_1[n], psh_1[n], 3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	#endif
	//cudaMemcpyAsync(Bufferpflag[n], pflag_GPU[n], ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int), cudaMemcpyHostToDevice,commandQueueGPU[n]);
	cudaMemcpyAsync(Bufferfailimage[n], failimage_GPU[n], NFAIL*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n])*sizeof(int), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(BufferKatm[n], Katm_GPU[n], (N1_GPU[n] + 2 * N1G)*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);
	cudaMemcpyAsync(Bufferradius[n], radius_GPU, (N1_GPU[n] + 2 * N1G)*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[n]);

	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in GPU_write: %d \n", status);
}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	/*Set arguments of kernel*/
	if (dir == 1){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else if (dir == 2){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF2_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF2_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else{
		if (flag == 1){
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF3_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF3_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error Fluxcalcprep %d \n", status);
}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	////cudaSetDevice(block[n][AMR_GPU]);
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),  
				dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF2_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF3_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	else{
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				 dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF2_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF3_1[n], Bufferdq_1[n], Bufferstorage1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				  dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Fluxcalc2D2 %d\n", status);
}

/*Start reading timestep from GPU*/
void read_time_GPU(void){
	//int n;
	//for (n = 0; n < n_active; n++){
	//	if (prestep_full[n_ord[n]] == 1){
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
	if (dir == 1) nr = nr_workgroups2_1[n];
	else if (dir == 2) nr = nr_workgroups2_2[n];
	else if (dir == 3) nr = nr_workgroups2_3[n];
	cudaStreamSynchronize(commandQueueGPU[n]);
	status = cudaGetLastError();
	if (status != 0) fprintf(stderr, "Error fluxcalc_GPU %d\n", status);
	for (y = 0; y < nr; y++){
		if (dtij_GPU[n][y] < ndt && dtij_GPU[n][y] < 1.e9 && dtij_GPU[n][y] > 1.e-6){
			ndt = dtij_GPU[n][y];
		}
	}
	return(ndt);
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	//cudaSetDevice(block[n][AMR_GPU]);
	 fix_flux << < nr_workgroups_special[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error fixflux %d \n", status);
}

void GPU_consttransport_bound(void){
	int n;

	gpu = 1;
	#if(PRESTEP)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
	#elif(PRESTEP2)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[n_ord[n]] == 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
	#else
	#if(!TIMESTEP_JET)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send1(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec2(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
	}
	
	#if(N3G>0)
	#if(!TIMESTEP_JET)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send3(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}	
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferph_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport2 %d \n", status);
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferps_1[n], BufferE_1[n], Dt);
	}
	else{
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferpsh_1[n], BufferE_1[n], Dt);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status )fprintf(stderr, "Error constransport3 %d \n", status);
}

void GPU_consttransport3_post(double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) / LOCAL_WORK_SIZE;
	
	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	consttransport3_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferps_1[n], BufferE_1[n], Dt);
	
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport3_post %d \n", status);
}

void GPU_flux_ct1(int n)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct1 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct2 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	 //cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct2 %d\n", status);
}

void GPU_Utoprim(int flag, int n, double Dt)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	#if(!V100)
	if (flag == 0){
		Utoprim0 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		Utoprim0 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim0 %d\n", status);

	if (flag == 0){
		Utoprim1 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim1 %d\n", status);
	if (flag == 0){
		Utoprim2 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferph_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		Utoprim2 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim1 %d\n", status);
	#endif
}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) / LOCAL_WORK_SIZE;
	//cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 1){
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferph_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_fixup(int flag, int n, double Dt)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 0){
		fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fixup %d\n", status);
}

void GPU_cleanup_post(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + 2 * (BS_1 + 2 * N2G)*(BS_3 + 2 * N3G) + 2 * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) + 2 * (BS_1 + 2 * N2G)*(BS_3 + 2 * N3G) + 2 * (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)) / LOCAL_WORK_SIZE;
	//cudaSetDevice(block[n][AMR_GPU]);
	cleanup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], BufferE_1[n]);
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error cleanup_post %d\n", status);
}

void GPU_fixup_post(int n, double Dt)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2) % LOCAL_WORK_SIZE) + 2 * (BS_2)*(BS_3) + (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2)) / LOCAL_WORK_SIZE;
	//cudaSetDevice(block[n][AMR_GPU]);
	fixup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Bufferp_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
		Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, 1);
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error fixup_post %d\n", status);
}

void GPU_boundprim(int bound_force)
{
	int n;
	int temp = nstep;
	gpu = 1;

	if (bound_force == 1) nstep = -1;
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim1(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(0, n_ord[n]);
	}
	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim2(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	//For last timestep do not receive synchronized electrice fields 
	#if(PRESTEP)
	rc = 0;
	//MPI communication
	//mpi_synch();
	if (rank == 0){
		begin2 =clock();
	}
	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			flux_rec1(F1, BufferF1_1, n_ord[n], 1);
			flux_rec2(F2, BufferF2_1, n_ord[n], 1);
			flux_rec3(F3, BufferF3_1, n_ord[n], 1);

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
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomE/BoundcomF \n");
	#endif


	rc = 0;
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, Bufferp_1, 0, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, Bufferph_1, 0, n_ord[n], 0);
	}

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}

	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, Bufferp_1, 0, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, Bufferph_1, 0, n_ord[n], 0);
	}

	if (N3 > 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			//cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, Bufferp_1, 0, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, Bufferph_1, 0, n_ord[n], 0);
		}
	}
	if (rc != 0)fprintf(stderr, "Error in MPI in boundcomP \n");
	
	//MPI communication
	mpi_synch();

	if (rank == 0){
		end2 = clock();
		time_spent3 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	}

	nstep = temp;
}

void GPU_boundprim1(int flag, int n)
{
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[n]);
		}
		else{
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[n]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status ) fprintf(stderr, "Error boundprim1 %d\n", status);
	}
}

void GPU_boundprim2(int flag, int n)
{
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferph_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[n]);
		}
		else{
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (Bufferp_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[n]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error boundprim2.1 %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k;
	//cudaSetDevice(block[n][AMR_GPU]);
	cudaMemcpyAsync(p_1[n], Bufferp_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
	cudaMemcpyAsync(ph_1[n], Bufferph_1[n], (int)(NPR*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
	#if(STAGGERED)
	cudaMemcpyAsync(ps_1[n], Bufferps_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
	cudaMemcpyAsync(psh_1[n], Bufferpsh_1[n], (int)(3 * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
	#endif
	cudaMemcpyAsync(failimage_GPU[n], Bufferfailimage[n], (int)((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) * NFAIL * sizeof(int), cudaMemcpyDeviceToHost, commandQueueGPU[n]);
	cudaDeviceSynchronize();

	#pragma omp parallel private(i, j, z, k)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + N1_GPU[n] - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + N2_GPU[n] - 1 + N2G, N3_GPU_offset[n] - N3G, N3_GPU_offset[n] + N3_GPU[n] - 1 + N3G){
			for (k = 0; k < NPR; k++){
				p[n][index_3D(n, i, j, z)][k] = p_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				ph[n][index_3D(n, i, j, z)][k] = ph_1[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			for (k = 0; k < NFAIL; k++){
				failimage[n][index_3D(n, i, j, z)][k] = failimage_GPU[n][k*((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#if(STAGGERED)
			for (k = 1; k < NDIM; k++){
				ps[n][index_3D(n, i, j, z)][k] = ps_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
				psh[n][index_3D(n, i, j, z)][k] = psh_1[n][(k - 1) * ((N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G)*(N1_GPU[n] + 2 * N1G) + fix_mem[n]) + (i - N1_GPU_offset[n] + N1G)*(N3_GPU[n] + 2 * N3G)*(N2_GPU[n] + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(N3_GPU[n] + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)];
			}
			#endif
		}
	}
	status = cudaGetLastError();
	if (cudaSuccess != status )fprintf(stderr, "Error in GPU_read: %d \n", status);
}

void GPU_finish(int n)
{
	int i;

	//Select correct CUDA device
	//cudaSetDevice(block[n][AMR_GPU]);

	//Destroy CUDA events associated with block
	for (i = 0; i < 600; i++) cudaEventDestroy(boundevent[n][i]);
	for (i = 0; i < 100; i++) cudaEventDestroy(boundevent1[n][i]);
	cudaStreamDestroy(commandQueueGPU[n]);

	free(p_1[n]);
	#if(STAGGERED)
	free(ps_1[n]);
	free(psh_1[n]);
	#endif
	free(ph_1[n]);
	//free(dU_GPU[n]);
	//free(pflag_GPU[n]);
	free(failimage_GPU[n]);
	free(Katm_GPU[n]);
	free(dq_1[n]);
	free(gcov_GPU[n]);
	free(gcon_GPU[n]);
	free(conn_GPU[n]);
	free(gdet_GPU[n]);

	status += cudaFreeHost(dtij_GPU[n]);
	//status += cudaFree(Bufferdtij[n]);
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
	status += cudaFree(Bufferp_1[n]);
	status += cudaFree(Bufferph_1[n]);
	#if(STAGGERED)
	status += cudaFree(Bufferps_1[n]);
	status += cudaFree(Bufferpsh_1[n]);
	#endif
	status += cudaFree(Bufferpflag[n]);
	status += cudaFree(Bufferfailimage[n]);
	status += cudaFree(BufferKatm[n]);
	//status += cudaFree(BufferdU[n]);
	status += cudaFree(Buffergcov[n]);
	status += cudaFree(Buffergcon[n]);
	status += cudaFree(Bufferconn[n]);
	status += cudaFree(Buffergdet[n]);
	#if(GPU_DIRECT)
	//status += cudaFree(NULL_POINTER[n]);
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
	status += cudaFreeHost(Buffersend1fine[n]);
	status += cudaFreeHost(Buffersend3fine[n]);
	status += cudaFreeHost(Bufferrec1fine[n]);
	status += cudaFreeHost(Bufferrec3fine[n]);

	/*
	status += cudaFree(Buffersend2fine[n]);
	status += cudaFree(Buffersend4fine[n]);
	#if(N3G>0)
	status += cudaFree(Buffersend5fine[n]);
	status += cudaFree(Buffersend6fine[n]);
	#endif
	status += cudaFree(Bufferrec2fine[n]);
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
	#else
	//status += cudaFreeHost(NULL_POINTER[n]);
	status += cudaFreeHost(Buffersend1[n]);
	status += cudaFreeHost(Buffersend1_3[n]);
	status += cudaFreeHost(Buffersend1_4[n]);
	status += cudaFreeHost(Buffersend1_7[n]);
	status += cudaFreeHost(Buffersend1_8[n]);
	status += cudaFreeHost(Buffersend2[n]);
	status += cudaFreeHost(Buffersend2_1[n]);
	status += cudaFreeHost(Buffersend2_2[n]);
	status += cudaFreeHost(Buffersend2_3[n]);
	status += cudaFreeHost(Buffersend2_4[n]);
	status += cudaFreeHost(Buffersend3[n]);
	status += cudaFreeHost(Buffersend3_1[n]);
	status += cudaFreeHost(Buffersend3_2[n]);
	status += cudaFreeHost(Buffersend3_5[n]);
	status += cudaFreeHost(Buffersend3_6[n]);
	status += cudaFreeHost(Buffersend4[n]);
	status += cudaFreeHost(Buffersend4_5[n]);
	status += cudaFreeHost(Buffersend4_6[n]);
	status += cudaFreeHost(Buffersend4_7[n]);
	status += cudaFreeHost(Buffersend4_8[n]);
	#if(N3G>0)
	status += cudaFreeHost(Buffersend5[n]);
	status += cudaFreeHost(Buffersend5_1[n]);
	status += cudaFreeHost(Buffersend5_3[n]);
	status += cudaFreeHost(Buffersend5_5[n]);
	status += cudaFreeHost(Buffersend5_7[n]);
	status += cudaFreeHost(Buffersend6[n]);
	status += cudaFreeHost(Buffersend6_2[n]);
	status += cudaFreeHost(Buffersend6_4[n]);
	status += cudaFreeHost(Buffersend6_6[n]);
	status += cudaFreeHost(Buffersend6_8[n]);
	#endif
	status += cudaFreeHost(Bufferrec1[n]);
	status += cudaFreeHost(Bufferrec1_3[n]);
	status += cudaFreeHost(Bufferrec1_4[n]);
	status += cudaFreeHost(Bufferrec1_7[n]);
	status += cudaFreeHost(Bufferrec1_8[n]);
	status += cudaFreeHost(Bufferrec2[n]);
	status += cudaFreeHost(Bufferrec2_1[n]);
	status += cudaFreeHost(Bufferrec2_2[n]);
	status += cudaFreeHost(Bufferrec2_3[n]);
	status += cudaFreeHost(Bufferrec2_4[n]);
	status += cudaFreeHost(Bufferrec3[n]);
	status += cudaFreeHost(Bufferrec3_1[n]);
	status += cudaFreeHost(Bufferrec3_2[n]);
	status += cudaFreeHost(Bufferrec3_5[n]);
	status += cudaFreeHost(Bufferrec3_6[n]);
	status += cudaFreeHost(Bufferrec4[n]);
	status += cudaFreeHost(Bufferrec4_5[n]);
	status += cudaFreeHost(Bufferrec4_6[n]);
	status += cudaFreeHost(Bufferrec4_7[n]);
	status += cudaFreeHost(Bufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5[n]);
	status += cudaFreeHost(Bufferrec5_1[n]);
	status += cudaFreeHost(Bufferrec5_3[n]);
	status += cudaFreeHost(Bufferrec5_5[n]);
	status += cudaFreeHost(Bufferrec5_7[n]);
	status += cudaFreeHost(Bufferrec6[n]);
	status += cudaFreeHost(Bufferrec6_2[n]);
	status += cudaFreeHost(Bufferrec6_4[n]);
	status += cudaFreeHost(Bufferrec6_6[n]);
	status += cudaFreeHost(Bufferrec6_8[n]);
	#endif
	status += cudaFreeHost(tempBufferrec1[n]);
	status += cudaFreeHost(tempBufferrec1_3[n]);
	status += cudaFreeHost(tempBufferrec1_4[n]);
	status += cudaFreeHost(tempBufferrec1_7[n]);
	status += cudaFreeHost(tempBufferrec1_8[n]);
	status += cudaFreeHost(tempBufferrec2[n]);
	status += cudaFreeHost(tempBufferrec2_1[n]);
	status += cudaFreeHost(tempBufferrec2_2[n]);
	status += cudaFreeHost(tempBufferrec2_3[n]);
	status += cudaFreeHost(tempBufferrec2_4[n]);
	status += cudaFreeHost(tempBufferrec3[n]);
	status += cudaFreeHost(tempBufferrec3_1[n]);
	status += cudaFreeHost(tempBufferrec3_2[n]);
	status += cudaFreeHost(tempBufferrec3_5[n]);
	status += cudaFreeHost(tempBufferrec3_6[n]);
	status += cudaFreeHost(tempBufferrec4[n]);
	status += cudaFreeHost(tempBufferrec4_5[n]);
	status += cudaFreeHost(tempBufferrec4_6[n]);
	status += cudaFreeHost(tempBufferrec4_7[n]);
	status += cudaFreeHost(tempBufferrec4_8[n]);
	#if(N3G>0)
	status += cudaFreeHost(tempBufferrec5[n]);
	status += cudaFreeHost(tempBufferrec5_1[n]);
	status += cudaFreeHost(tempBufferrec5_3[n]);
	status += cudaFreeHost(tempBufferrec5_5[n]);
	status += cudaFreeHost(tempBufferrec5_7[n]);
	status += cudaFreeHost(tempBufferrec6[n]);
	status += cudaFreeHost(tempBufferrec6_2[n]);
	status += cudaFreeHost(tempBufferrec6_4[n]);
	status += cudaFreeHost(tempBufferrec6_6[n]);
	status += cudaFreeHost(tempBufferrec6_8[n]);
	#endif
	status += cudaFreeHost(Buffersend1flux[n]);
	status += cudaFreeHost(Buffersend2flux[n]);
	status += cudaFreeHost(Buffersend3flux[n]);
	status += cudaFreeHost(Buffersend4flux[n]);
	#if(N3G>0)
	status += cudaFreeHost(Buffersend5flux[n]);
	status += cudaFreeHost(Buffersend6flux[n]);
	#endif
	status += cudaFreeHost(Bufferrec1flux[n]);
	status += cudaFreeHost(Bufferrec2flux[n]);
	status += cudaFreeHost(Bufferrec3flux[n]);
	status += cudaFreeHost(Bufferrec4flux[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5flux[n]);
	status += cudaFreeHost(Bufferrec6flux[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3flux[n]);
	status += cudaFreeHost(Bufferrec1_4flux[n]);
	status += cudaFreeHost(Bufferrec1_7flux[n]);
	status += cudaFreeHost(Bufferrec1_8flux[n]);
	status += cudaFreeHost(Bufferrec2_1flux[n]);
	status += cudaFreeHost(Bufferrec2_2flux[n]);
	status += cudaFreeHost(Bufferrec2_3flux[n]);
	status += cudaFreeHost(Bufferrec2_4flux[n]);
	status += cudaFreeHost(Bufferrec3_1flux[n]);
	status += cudaFreeHost(Bufferrec3_2flux[n]);
	status += cudaFreeHost(Bufferrec3_5flux[n]);
	status += cudaFreeHost(Bufferrec3_6flux[n]);
	status += cudaFreeHost(Bufferrec4_5flux[n]);
	status += cudaFreeHost(Bufferrec4_6flux[n]);
	status += cudaFreeHost(Bufferrec4_7flux[n]);
	status += cudaFreeHost(Bufferrec4_8flux[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1flux[n]);
	status += cudaFreeHost(Bufferrec5_3flux[n]);
	status += cudaFreeHost(Bufferrec5_5flux[n]);
	status += cudaFreeHost(Bufferrec5_7flux[n]);
	status += cudaFreeHost(Bufferrec6_2flux[n]);
	status += cudaFreeHost(Bufferrec6_4flux[n]);
	status += cudaFreeHost(Bufferrec6_6flux[n]);
	status += cudaFreeHost(Bufferrec6_8flux[n]);
	#endif
	status += cudaFreeHost(Bufferrec1flux1[n]);
	status += cudaFreeHost(Bufferrec2flux1[n]);
	status += cudaFreeHost(Bufferrec3flux1[n]);
	status += cudaFreeHost(Bufferrec4flux1[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5flux1[n]);
	status += cudaFreeHost(Bufferrec6flux1[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3flux1[n]);
	status += cudaFreeHost(Bufferrec1_4flux1[n]);
	status += cudaFreeHost(Bufferrec1_7flux1[n]);
	status += cudaFreeHost(Bufferrec1_8flux1[n]);
	status += cudaFreeHost(Bufferrec2_1flux1[n]);
	status += cudaFreeHost(Bufferrec2_2flux1[n]);
	status += cudaFreeHost(Bufferrec2_3flux1[n]);
	status += cudaFreeHost(Bufferrec2_4flux1[n]);
	status += cudaFreeHost(Bufferrec3_1flux1[n]);
	status += cudaFreeHost(Bufferrec3_2flux1[n]);
	status += cudaFreeHost(Bufferrec3_5flux1[n]);
	status += cudaFreeHost(Bufferrec3_6flux1[n]);
	status += cudaFreeHost(Bufferrec4_5flux1[n]);
	status += cudaFreeHost(Bufferrec4_6flux1[n]);
	status += cudaFreeHost(Bufferrec4_7flux1[n]);
	status += cudaFreeHost(Bufferrec4_8flux1[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1flux1[n]);
	status += cudaFreeHost(Bufferrec5_3flux1[n]);
	status += cudaFreeHost(Bufferrec5_5flux1[n]);
	status += cudaFreeHost(Bufferrec5_7flux1[n]);
	status += cudaFreeHost(Bufferrec6_2flux1[n]);
	status += cudaFreeHost(Bufferrec6_4flux1[n]);
	status += cudaFreeHost(Bufferrec6_6flux1[n]);
	status += cudaFreeHost(Bufferrec6_8flux1[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3flux2[n]);
	status += cudaFreeHost(Bufferrec1_4flux2[n]);
	status += cudaFreeHost(Bufferrec1_7flux2[n]);
	status += cudaFreeHost(Bufferrec1_8flux2[n]);
	status += cudaFreeHost(Bufferrec2_1flux2[n]);
	status += cudaFreeHost(Bufferrec2_2flux2[n]);
	status += cudaFreeHost(Bufferrec2_3flux2[n]);
	status += cudaFreeHost(Bufferrec2_4flux2[n]);
	status += cudaFreeHost(Bufferrec3_1flux2[n]);
	status += cudaFreeHost(Bufferrec3_2flux2[n]);
	status += cudaFreeHost(Bufferrec3_5flux2[n]);
	status += cudaFreeHost(Bufferrec3_6flux2[n]);
	status += cudaFreeHost(Bufferrec4_5flux2[n]);
	status += cudaFreeHost(Bufferrec4_6flux2[n]);
	status += cudaFreeHost(Bufferrec4_7flux2[n]);
	status += cudaFreeHost(Bufferrec4_8flux2[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1flux2[n]);
	status += cudaFreeHost(Bufferrec5_3flux2[n]);
	status += cudaFreeHost(Bufferrec5_5flux2[n]);
	status += cudaFreeHost(Bufferrec5_7flux2[n]);
	status += cudaFreeHost(Bufferrec6_2flux2[n]);
	status += cudaFreeHost(Bufferrec6_4flux2[n]);
	status += cudaFreeHost(Bufferrec6_6flux2[n]);
	status += cudaFreeHost(Bufferrec6_8flux2[n]);
	#endif
	status += cudaFreeHost(Buffersend1fine[n]);
	status += cudaFreeHost(Buffersend3fine[n]);
	status += cudaFreeHost(Bufferrec1fine[n]);
	status += cudaFreeHost(Bufferrec3fine[n]);

	/*
	status += cudaFreeHost(Buffersend2fine[n]);
	status += cudaFreeHost(Buffersend4fine[n]);
	#if(N3G>0)
	status += cudaFreeHost(Buffersend5fine[n]);
	status += cudaFreeHost(Buffersend6fine[n]);
	#endif
	status += cudaFreeHost(Bufferrec2fine[n]);
	status += cudaFreeHost(Bufferrec4fine[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5fine[n]);
	status += cudaFreeHost(Bufferrec6fine[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3fine[n]);
	status += cudaFreeHost(Bufferrec1_4fine[n]);
	status += cudaFreeHost(Bufferrec1_7fine[n]);
	status += cudaFreeHost(Bufferrec1_8fine[n]);
	status += cudaFreeHost(Bufferrec2_1fine[n]);
	status += cudaFreeHost(Bufferrec2_2fine[n]);
	status += cudaFreeHost(Bufferrec2_3fine[n]);
	status += cudaFreeHost(Bufferrec2_4fine[n]);
	status += cudaFreeHost(Bufferrec3_1fine[n]);
	status += cudaFreeHost(Bufferrec3_2fine[n]);
	status += cudaFreeHost(Bufferrec3_5fine[n]);
	status += cudaFreeHost(Bufferrec3_6fine[n]);
	status += cudaFreeHost(Bufferrec4_5fine[n]);
	status += cudaFreeHost(Bufferrec4_6fine[n]);
	status += cudaFreeHost(Bufferrec4_7fine[n]);
	status += cudaFreeHost(Bufferrec4_8fine[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1fine[n]);
	status += cudaFreeHost(Bufferrec5_3fine[n]);
	status += cudaFreeHost(Bufferrec5_5fine[n]);
	status += cudaFreeHost(Bufferrec5_7fine[n]);
	status += cudaFreeHost(Bufferrec6_2fine[n]);
	status += cudaFreeHost(Bufferrec6_4fine[n]);
	status += cudaFreeHost(Bufferrec6_6fine[n]);
	status += cudaFreeHost(Bufferrec6_8fine[n]);
	#endif
	*/
	status += cudaFreeHost(Buffersend1E[n]);
	status += cudaFreeHost(Buffersend2E[n]);
	status += cudaFreeHost(Buffersend3E[n]);
	status += cudaFreeHost(Buffersend4E[n]);
	#if(N3G>0)
	status += cudaFreeHost(Buffersend5E[n]);
	status += cudaFreeHost(Buffersend6E[n]);
	#endif
	status += cudaFreeHost(Bufferrec1E[n]);
	status += cudaFreeHost(Bufferrec2E[n]);
	status += cudaFreeHost(Bufferrec3E[n]);
	status += cudaFreeHost(Bufferrec4E[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5E[n]);
	status += cudaFreeHost(Bufferrec6E[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3E[n]);
	status += cudaFreeHost(Bufferrec1_4E[n]);
	status += cudaFreeHost(Bufferrec1_7E[n]);
	status += cudaFreeHost(Bufferrec1_8E[n]);
	status += cudaFreeHost(Bufferrec2_1E[n]);
	status += cudaFreeHost(Bufferrec2_2E[n]);
	status += cudaFreeHost(Bufferrec2_3E[n]);
	status += cudaFreeHost(Bufferrec2_4E[n]);
	status += cudaFreeHost(Bufferrec3_1E[n]);
	status += cudaFreeHost(Bufferrec3_2E[n]);
	status += cudaFreeHost(Bufferrec3_5E[n]);
	status += cudaFreeHost(Bufferrec3_6E[n]);
	status += cudaFreeHost(Bufferrec4_5E[n]);
	status += cudaFreeHost(Bufferrec4_6E[n]);
	status += cudaFreeHost(Bufferrec4_7E[n]);
	status += cudaFreeHost(Bufferrec4_8E[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1E[n]);
	status += cudaFreeHost(Bufferrec5_3E[n]);
	status += cudaFreeHost(Bufferrec5_5E[n]);
	status += cudaFreeHost(Bufferrec5_7E[n]);
	status += cudaFreeHost(Bufferrec6_2E[n]);
	status += cudaFreeHost(Bufferrec6_4E[n]);
	status += cudaFreeHost(Bufferrec6_6E[n]);
	status += cudaFreeHost(Bufferrec6_8E[n]);
	#endif
	status += cudaFreeHost(Bufferrec1E1[n]);
	status += cudaFreeHost(Bufferrec2E1[n]);
	status += cudaFreeHost(Bufferrec3E1[n]);
	status += cudaFreeHost(Bufferrec4E1[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5E1[n]);
	status += cudaFreeHost(Bufferrec6E1[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3E1[n]);
	status += cudaFreeHost(Bufferrec1_4E1[n]);
	status += cudaFreeHost(Bufferrec1_7E1[n]);
	status += cudaFreeHost(Bufferrec1_8E1[n]);
	status += cudaFreeHost(Bufferrec2_1E1[n]);
	status += cudaFreeHost(Bufferrec2_2E1[n]);
	status += cudaFreeHost(Bufferrec2_3E1[n]);
	status += cudaFreeHost(Bufferrec2_4E1[n]);
	status += cudaFreeHost(Bufferrec3_1E1[n]);
	status += cudaFreeHost(Bufferrec3_2E1[n]);
	status += cudaFreeHost(Bufferrec3_5E1[n]);
	status += cudaFreeHost(Bufferrec3_6E1[n]);
	status += cudaFreeHost(Bufferrec4_5E1[n]);
	status += cudaFreeHost(Bufferrec4_6E1[n]);
	status += cudaFreeHost(Bufferrec4_7E1[n]);
	status += cudaFreeHost(Bufferrec4_8E1[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1E1[n]);
	status += cudaFreeHost(Bufferrec5_3E1[n]);
	status += cudaFreeHost(Bufferrec5_5E1[n]);
	status += cudaFreeHost(Bufferrec5_7E1[n]);
	status += cudaFreeHost(Bufferrec6_2E1[n]);
	status += cudaFreeHost(Bufferrec6_4E1[n]);
	status += cudaFreeHost(Bufferrec6_6E1[n]);
	status += cudaFreeHost(Bufferrec6_8E1[n]);
	#endif
	status += cudaFreeHost(Bufferrec1_3E2[n]);
	status += cudaFreeHost(Bufferrec1_4E2[n]);
	status += cudaFreeHost(Bufferrec1_7E2[n]);
	status += cudaFreeHost(Bufferrec1_8E2[n]);
	status += cudaFreeHost(Bufferrec2_1E2[n]);
	status += cudaFreeHost(Bufferrec2_2E2[n]);
	status += cudaFreeHost(Bufferrec2_3E2[n]);
	status += cudaFreeHost(Bufferrec2_4E2[n]);
	status += cudaFreeHost(Bufferrec3_1E2[n]);
	status += cudaFreeHost(Bufferrec3_2E2[n]);
	status += cudaFreeHost(Bufferrec3_5E2[n]);
	status += cudaFreeHost(Bufferrec3_6E2[n]);
	status += cudaFreeHost(Bufferrec4_5E2[n]);
	status += cudaFreeHost(Bufferrec4_6E2[n]);
	status += cudaFreeHost(Bufferrec4_7E2[n]);
	status += cudaFreeHost(Bufferrec4_8E2[n]);
	#if(N3G>0)
	status += cudaFreeHost(Bufferrec5_1E2[n]);
	status += cudaFreeHost(Bufferrec5_3E2[n]);
	status += cudaFreeHost(Bufferrec5_5E2[n]);
	status += cudaFreeHost(Bufferrec5_7E2[n]);
	status += cudaFreeHost(Bufferrec6_2E2[n]);
	status += cudaFreeHost(Bufferrec6_4E2[n]);
	status += cudaFreeHost(Bufferrec6_6E2[n]);
	status += cudaFreeHost(Bufferrec6_8E2[n]);
	#endif
	#if(N3G>0)
	status += cudaFreeHost(BuffersendE1corn9[n]);
	status += cudaFreeHost(BuffersendE1corn10[n]);
	status += cudaFreeHost(BuffersendE1corn11[n]);
	status += cudaFreeHost(BuffersendE1corn12[n]);
	status += cudaFreeHost(BuffersendE2corn5[n]);
	status += cudaFreeHost(BuffersendE2corn6[n]);
	status += cudaFreeHost(BuffersendE2corn7[n]);
	status += cudaFreeHost(BuffersendE2corn8[n]);
	#endif
	status += cudaFreeHost(BuffersendE3corn1[n]);
	status += cudaFreeHost(BuffersendE3corn2[n]);
	status += cudaFreeHost(BuffersendE3corn3[n]);
	status += cudaFreeHost(BuffersendE3corn4[n]);
	#if(N3G>0)
	status += cudaFreeHost(BufferrecE1corn9[n]);
	status += cudaFreeHost(BufferrecE1corn10[n]);
	status += cudaFreeHost(BufferrecE1corn11[n]);
	status += cudaFreeHost(BufferrecE1corn12[n]);
	status += cudaFreeHost(BufferrecE2corn5[n]);
	status += cudaFreeHost(BufferrecE2corn6[n]);
	status += cudaFreeHost(BufferrecE2corn7[n]);
	status += cudaFreeHost(BufferrecE2corn8[n]);
	#endif
	status += cudaFreeHost(BufferrecE3corn1[n]);
	status += cudaFreeHost(BufferrecE3corn2[n]);
	status += cudaFreeHost(BufferrecE3corn3[n]);
	status += cudaFreeHost(BufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFreeHost(BufferrecE1corn9_3[n]);
	status += cudaFreeHost(BufferrecE1corn9_7[n]);
	status += cudaFreeHost(BufferrecE1corn10_1[n]);
	status += cudaFreeHost(BufferrecE1corn10_5[n]);
	status += cudaFreeHost(BufferrecE1corn11_2[n]);
	status += cudaFreeHost(BufferrecE1corn11_6[n]);
	status += cudaFreeHost(BufferrecE1corn12_4[n]);
	status += cudaFreeHost(BufferrecE1corn12_8[n]);
	status += cudaFreeHost(BufferrecE2corn5_2[n]);
	status += cudaFreeHost(BufferrecE2corn5_4[n]);
	status += cudaFreeHost(BufferrecE2corn6_1[n]);
	status += cudaFreeHost(BufferrecE2corn6_3[n]);
	status += cudaFreeHost(BufferrecE2corn7_5[n]);
	status += cudaFreeHost(BufferrecE2corn7_7[n]);
	status += cudaFreeHost(BufferrecE2corn8_6[n]);
	status += cudaFreeHost(BufferrecE2corn8_8[n]);
	#endif
	status += cudaFreeHost(BufferrecE3corn1_3[n]);
	status += cudaFreeHost(BufferrecE3corn1_4[n]);
	status += cudaFreeHost(BufferrecE3corn2_1[n]);
	status += cudaFreeHost(BufferrecE3corn2_2[n]);
	status += cudaFreeHost(BufferrecE3corn3_5[n]);
	status += cudaFreeHost(BufferrecE3corn3_6[n]);
	status += cudaFreeHost(BufferrecE3corn4_7[n]);
	status += cudaFreeHost(BufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFreeHost(tempBufferrecE1corn9[n]);
	status += cudaFreeHost(tempBufferrecE1corn10[n]);
	status += cudaFreeHost(tempBufferrecE1corn11[n]);
	status += cudaFreeHost(tempBufferrecE1corn12[n]);
	status += cudaFreeHost(tempBufferrecE2corn5[n]);
	status += cudaFreeHost(tempBufferrecE2corn6[n]);
	status += cudaFreeHost(tempBufferrecE2corn7[n]);
	status += cudaFreeHost(tempBufferrecE2corn8[n]);
	#endif
	status += cudaFreeHost(tempBufferrecE3corn1[n]);
	status += cudaFreeHost(tempBufferrecE3corn2[n]);
	status += cudaFreeHost(tempBufferrecE3corn3[n]);
	status += cudaFreeHost(tempBufferrecE3corn4[n]);
	#if(N3G>0)
	status += cudaFreeHost(tempBufferrecE1corn9_3[n]);
	status += cudaFreeHost(tempBufferrecE1corn9_7[n]);
	status += cudaFreeHost(tempBufferrecE1corn10_1[n]);
	status += cudaFreeHost(tempBufferrecE1corn10_5[n]);
	status += cudaFreeHost(tempBufferrecE1corn11_2[n]);
	status += cudaFreeHost(tempBufferrecE1corn11_6[n]);
	status += cudaFreeHost(tempBufferrecE1corn12_4[n]);
	status += cudaFreeHost(tempBufferrecE1corn12_8[n]);
	status += cudaFreeHost(tempBufferrecE2corn5_2[n]);
	status += cudaFreeHost(tempBufferrecE2corn5_4[n]);
	status += cudaFreeHost(tempBufferrecE2corn6_1[n]);
	status += cudaFreeHost(tempBufferrecE2corn6_3[n]);
	status += cudaFreeHost(tempBufferrecE2corn7_5[n]);
	status += cudaFreeHost(tempBufferrecE2corn7_7[n]);
	status += cudaFreeHost(tempBufferrecE2corn8_6[n]);
	status += cudaFreeHost(tempBufferrecE2corn8_8[n]);
	#endif
	status += cudaFreeHost(tempBufferrecE3corn1_3[n]);
	status += cudaFreeHost(tempBufferrecE3corn1_4[n]);
	status += cudaFreeHost(tempBufferrecE3corn2_1[n]);
	status += cudaFreeHost(tempBufferrecE3corn2_2[n]);
	status += cudaFreeHost(tempBufferrecE3corn3_5[n]);
	status += cudaFreeHost(tempBufferrecE3corn3_6[n]);
	status += cudaFreeHost(tempBufferrecE3corn4_7[n]);
	status += cudaFreeHost(tempBufferrecE3corn4_8[n]);
	#if(N3G>0)
	status += cudaFreeHost(BufferrecE1corn9_32[n]);
	status += cudaFreeHost(BufferrecE1corn9_72[n]);
	status += cudaFreeHost(BufferrecE1corn10_12[n]);
	status += cudaFreeHost(BufferrecE1corn10_52[n]);
	status += cudaFreeHost(BufferrecE1corn11_22[n]);
	status += cudaFreeHost(BufferrecE1corn11_62[n]);
	status += cudaFreeHost(BufferrecE1corn12_42[n]);
	status += cudaFreeHost(BufferrecE1corn12_82[n]);
	status += cudaFreeHost(BufferrecE2corn5_22[n]);
	status += cudaFreeHost(BufferrecE2corn5_42[n]);
	status += cudaFreeHost(BufferrecE2corn6_12[n]);
	status += cudaFreeHost(BufferrecE2corn6_32[n]);
	status += cudaFreeHost(BufferrecE2corn7_52[n]);
	status += cudaFreeHost(BufferrecE2corn7_72[n]);
	status += cudaFreeHost(BufferrecE2corn8_62[n]);
	status += cudaFreeHost(BufferrecE2corn8_82[n]);
	#endif
	status += cudaFreeHost(BufferrecE3corn1_32[n]);
	status += cudaFreeHost(BufferrecE3corn1_42[n]);
	status += cudaFreeHost(BufferrecE3corn2_12[n]);
	status += cudaFreeHost(BufferrecE3corn2_22[n]);
	status += cudaFreeHost(BufferrecE3corn3_52[n]);
	status += cudaFreeHost(BufferrecE3corn3_62[n]);
	status += cudaFreeHost(BufferrecE3corn4_72[n]);
	status += cudaFreeHost(BufferrecE3corn4_82[n]);
	#endif
	
	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in GPU_finish_1: %d \n", status);
}
