#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}

void GPU_init(void)
{
	int i,j,pos,ranks_per_node;
	//Do some checks first
	if (N_GPU>numdevices){
		fprintf(stderr, "N_GPU is bigger than the number of devices! \n");
		exit(0);
	}

	//Enable peer access
	ranks_per_node = numdevices / N_GPU;
	gpu_offset = (rank % (ranks_per_node))*N_GPU;
	for (i = gpu_offset; i < gpu_offset + N_GPU; i++){
		cudaSetDevice(i);
		for (j = gpu_offset; j < gpu_offset + N_GPU; j++){
			if (i!=j) cudaDeviceEnablePeerAccess(j, 0);
		}
	}
	status = cudaGetLastError();
	if (cudaSuccess != status){
		exit(0);
		fprintf(stderr, "Error in setting peeraccess: %d \n", status);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting cache: %d \n", status);

	/*Set the global work size and make sure that it is a multiple of the group size. The Nvidia OpenCL framework crashes otherwise!*/
	fix_mem = LOCAL_WORK_SIZE - ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	fix_mem2 = LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)) % LOCAL_WORK_SIZE;
	local_work_size[0] = LOCAL_WORK_SIZE;

	/*Allocate memory to 1D arrays*/
	p_1[0] = (double(*))calloc(MAX_BLOCKS*NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(double));
	dq_1[0] = (double(*))calloc(MAX_BLOCKS*NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(double)); //array to store temporary data
	#if(STAGGERED)
	ps_1[0] = (double(*))calloc(MAX_BLOCKS*NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(double));
	psh_1[0] = (double(*))calloc(MAX_BLOCKS*NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(double));
	#endif
	ph_1[0] = (double(*))calloc(MAX_BLOCKS*NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(double));
	//dU_GPU[0] = (double(*))calloc(MAX_BLOCKS*NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G)), sizeof(double));
	#if(!NONSYMMETRIC)
	gcov_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG*NDIM*NDIM, sizeof(double));
	gcon_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG*NDIM*NDIM, sizeof(double));
	conn_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NDIM*NDIM*NDIM, sizeof(double));
	gdet_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG, sizeof(double));
	#else
	gcov_GPU[0] = (double(*))ccalloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*NPG*NDIM*NDIM, sizeof(double));
	gcon_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*NPG*NDIM*NDIM, sizeof(double));
	conn_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*NDIM*NDIM*NDIM, sizeof(double));
	gdet_GPU[0] = (double(*))calloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*NPG, sizeof(double));
	#endif
	//pflag_GPU[0] = (int(*))calloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem), sizeof(int));
	failimage_GPU[0] = (int(*))calloc(MAX_BLOCKS*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem) * NFAIL, sizeof(int));
	Katm_GPU[0] = (double(*))calloc(MAX_BLOCKS*(BS_1 + 2 * N1G), sizeof(double));

	/*Allocate memory to buffers on GPU*/
	cudaMalloc(&BufferF1_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&BufferF2_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&BufferF3_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&Bufferdq_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&BufferE_1[0], MAX_BLOCKS* NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&Bufferp_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&Bufferph_1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	#if(LEER)
	cudaMalloc(&BufferV[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	#endif
	cudaMalloc(&Bufferradius[0], MAX_BLOCKS* (BS_1 + 2 * N1G)*sizeof(double));
	#if(PPM || LEER)
	cudaMalloc(&Bufferstorage1[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	#endif
	#if(PRESTEP_P)
	cudaMalloc(&Bufferstorage2[0], MAX_BLOCKS* NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&Bufferstorage3[0], MAX_BLOCKS* NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	#else
	#endif
	#if(STAGGERED)
	cudaMalloc(&Bufferps_1[0], MAX_BLOCKS* 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	cudaMalloc(&Bufferpsh_1[0], MAX_BLOCKS* 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(double));
	#endif
	cudaMalloc(&Buffergdet[0], MAX_BLOCKS* ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG*sizeof(double));
	cudaMalloc(&Buffergcov[0], MAX_BLOCKS* ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Buffergcon[0], MAX_BLOCKS* ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NPG*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Bufferconn[0], MAX_BLOCKS* ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[0])*NDIM*NDIM*NDIM*sizeof(double));
	cudaMalloc(&Bufferpflag[0], MAX_BLOCKS* ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem)*sizeof(int));
	cudaMalloc(&Bufferfailimage[0], MAX_BLOCKS* ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem) * NFAIL * sizeof(int));
	cudaMalloc(&BufferKatm[0], MAX_BLOCKS* (BS_1 + 2 * N1G)*sizeof(double));
	//cudaMalloc(&BufferdU[0], MAX_BLOCKS* NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double));
	if (cudaSuccess != cudaGetLastError()) fprintf(stderr, "Error in setting kernel arguments 3: %d \n", cudaGetLastError());
	cudaHostAlloc(&dtij_GPU[0], MAX_BLOCKS* (nr_workgroups[0] + 1)*sizeof(double), 0);
	//cudaMalloc(&Bufferdtij[0], MAX_BLOCKS* (nr_workgroups[0] + 1)*sizeof(double));

	gpuMalloc(&Buffersend1[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuMalloc(&Buffersend1_3[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_3)gpuMalloc(&Buffersend1_4[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_1)gpuMalloc(&Buffersend1_7[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Buffersend1_8[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	gpuMalloc(&Buffersend2[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	gpuMalloc(&Buffersend2_1[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_3)gpuMalloc(&Buffersend2_2[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_2)gpuMalloc(&Buffersend2_3[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Buffersend2_4[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	gpuMalloc(&Buffersend3[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1)
	gpuMalloc(&Buffersend3_1[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_3)gpuMalloc(&Buffersend3_2[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_1)gpuMalloc(&Buffersend3_5[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Buffersend3_6[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	gpuMalloc(&Buffersend4[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double));
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	gpuMalloc(&Buffersend4_5[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_3)gpuMalloc(&Buffersend4_6[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_2)gpuMalloc(&Buffersend4_7[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Buffersend4_8[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#endif
	#if(N3G>0)
	gpuMalloc(&Buffersend5[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	gpuMalloc(&Buffersend5_1[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_2)gpuMalloc(&Buffersend5_3[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_1)gpuMalloc(&Buffersend5_5[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Buffersend5_7[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	gpuMalloc(&Buffersend6[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double));
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	gpuMalloc(&Buffersend6_2[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_2)gpuMalloc(&Buffersend6_4[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_1)gpuMalloc(&Buffersend6_6[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Buffersend6_8[0], MAX_BLOCKS* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#endif
	gpuMalloc(&Bufferrec1[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&Bufferrec2[0], MAX_BLOCKS* (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&Bufferrec3[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&Bufferrec4[0], MAX_BLOCKS* (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Bufferrec5[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	gpuMalloc(&Bufferrec6[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#if(PRESTEP || PRESTEP2)
	gpuMalloc(&tempBufferrec1[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&tempBufferrec2[0], MAX_BLOCKS* (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&tempBufferrec3[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	gpuMalloc(&tempBufferrec4[0], MAX_BLOCKS* (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&tempBufferrec5[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	gpuMalloc(&tempBufferrec6[0], MAX_BLOCKS* (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G)*sizeof(double));
	#endif
	#endif
	gpuMalloc(&Buffersend1flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Buffersend2flux[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuMalloc(&Buffersend3flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Buffersend4flux[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Buffersend5flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuMalloc(&Buffersend6flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	gpuMalloc(&Bufferrec1flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec2flux[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec3flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec4flux[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Bufferrec5flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuMalloc(&Bufferrec6flux[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	gpuMalloc(&Bufferrec1flux1[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec2flux1[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec3flux1[0], MAX_BLOCKS* NPR*(BS_1)*(BS_3)*sizeof(double));
	gpuMalloc(&Bufferrec4flux1[0], MAX_BLOCKS* NPR*(BS_2)*(BS_3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Bufferrec5flux1[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	gpuMalloc(&Bufferrec6flux1[0], MAX_BLOCKS* NPR*(BS_1)*(BS_2)*sizeof(double));
	#endif
	#if(N_LEVELS>1)
	gpuMalloc(&Bufferrec1_3flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec1_4flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec1_7flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Bufferrec1_8flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec2_1flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec2_2flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec2_3flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Bufferrec2_4flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	#endif
	gpuMalloc(&Bufferrec3_1flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec3_2flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec3_5flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Bufferrec3_6flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3))*sizeof(double));
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec4_5flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec4_6flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec4_7flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Bufferrec4_8flux2[0], MAX_BLOCKS* NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3))*sizeof(double));
	#endif
	#if(N3G>0)
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec5_1flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec5_3flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec5_5flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Bufferrec5_7flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	gpuMalloc(&Bufferrec6_2flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec6_4flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec6_6flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Bufferrec6_8flux2[0], MAX_BLOCKS* NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2))*sizeof(double));
	#endif
	#endif
	#endif

	cudaHostAlloc(&Buffersend1fine[0], MAX_BLOCKS* NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Buffersend3fine[0], MAX_BLOCKS* NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Bufferrec1fine[0], MAX_BLOCKS* NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);
	cudaHostAlloc(&Bufferrec3fine[0], MAX_BLOCKS* NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), 0);

	gpuMalloc(&Buffersend1E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Buffersend2E[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Buffersend3E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Buffersend4E[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Buffersend5E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&Buffersend6E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuMalloc(&Bufferrec1E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec2E[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec3E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec4E[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Bufferrec5E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&Bufferrec6E[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	#endif	
	gpuMalloc(&Bufferrec1E1[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec2E1[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec3E1[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&Bufferrec4E1[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&Bufferrec5E1[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&Bufferrec6E1[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*sizeof(double));
	#endif	
	#if(N_LEVELS>1)
	gpuMalloc(&Bufferrec1_3E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec1_4E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec1_7E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Bufferrec1_8E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec2_1E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec2_2E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec2_3E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Bufferrec2_4E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#endif
	gpuMalloc(&Bufferrec3_1E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec3_2E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec3_5E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_1 && REF_3)gpuMalloc(&Bufferrec3_6E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec4_5E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&Bufferrec4_6E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec4_7E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_2 && REF_3)gpuMalloc(&Bufferrec4_8E2[0], MAX_BLOCKS* 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#endif
	#if(N3G>0)
	#if(!DEREFINE_POLE)
	gpuMalloc(&Bufferrec5_1E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec5_3E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec5_5E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Bufferrec5_7E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&Bufferrec6_2E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&Bufferrec6_4E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_1)gpuMalloc(&Bufferrec6_6E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_1 && REF_2)gpuMalloc(&Bufferrec6_8E2[0], MAX_BLOCKS* 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	#endif
	#endif
	#endif
	#if(N3G>0)
	gpuMalloc(&BuffersendE1corn9[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BuffersendE1corn10[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BuffersendE1corn11[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BuffersendE1corn12[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BuffersendE2corn5[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BuffersendE2corn6[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BuffersendE2corn7[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BuffersendE2corn8[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuMalloc(&BuffersendE3corn1[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BuffersendE3corn2[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BuffersendE3corn3[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BuffersendE3corn4[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N3G>0)
	gpuMalloc(&BufferrecE1corn9[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BufferrecE1corn10[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BufferrecE1corn11[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BufferrecE1corn12[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&BufferrecE2corn5[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BufferrecE2corn6[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BufferrecE2corn7[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&BufferrecE2corn8[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuMalloc(&BufferrecE3corn1[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BufferrecE3corn2[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BufferrecE3corn3[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&BufferrecE3corn4[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	gpuMalloc(&BufferrecE1corn9_3[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn9_7[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn10_1[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn10_5[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn11_2[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn11_6[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn12_4[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn12_8[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE2corn5_2[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn5_4[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn6_1[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn6_3[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn7_5[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn7_7[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn8_6[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn8_8[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	#endif
	gpuMalloc(&BufferrecE3corn1_3[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn1_4[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn2_1[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn2_2[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn3_5[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn3_6[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn4_7[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn4_8[0], MAX_BLOCKS* 1 * (BS_3 + N3G) / (1 + REF_3) *sizeof(double));
	#endif
	#if(N3G>0)
	gpuMalloc(&tempBufferrecE1corn9[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&tempBufferrecE1corn10[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&tempBufferrecE1corn11[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&tempBufferrecE1corn12[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1)*sizeof(double));
	gpuMalloc(&tempBufferrecE2corn5[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&tempBufferrecE2corn6[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&tempBufferrecE2corn7[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	gpuMalloc(&tempBufferrecE2corn8[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2)*sizeof(double));
	#endif
	gpuMalloc(&tempBufferrecE3corn1[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&tempBufferrecE3corn2[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&tempBufferrecE3corn3[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	gpuMalloc(&tempBufferrecE3corn4[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3)*sizeof(double));
	#if(N_LEVELS>1)
	#if(N3G>0)
	gpuMalloc(&tempBufferrecE1corn9_3[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&tempBufferrecE1corn9_7[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&tempBufferrecE1corn10_1[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&tempBufferrecE1corn10_5[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&tempBufferrecE1corn11_2[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&tempBufferrecE1corn11_6[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&tempBufferrecE1corn12_4[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&tempBufferrecE1corn12_8[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&tempBufferrecE2corn5_2[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&tempBufferrecE2corn5_4[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&tempBufferrecE2corn6_1[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&tempBufferrecE2corn6_3[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&tempBufferrecE2corn7_5[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&tempBufferrecE2corn7_7[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&tempBufferrecE2corn8_6[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&tempBufferrecE2corn8_8[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	#endif
	gpuMalloc(&tempBufferrecE3corn1_3[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&tempBufferrecE3corn1_4[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&tempBufferrecE3corn2_1[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&tempBufferrecE3corn2_2[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&tempBufferrecE3corn3_5[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&tempBufferrecE3corn3_6[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&tempBufferrecE3corn4_7[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&tempBufferrecE3corn4_8[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#if(N3G>0)
	gpuMalloc(&BufferrecE1corn9_32[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn9_72[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn10_12[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn10_52[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn11_22[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn11_62[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE1corn12_42[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	if (REF_1)gpuMalloc(&BufferrecE1corn12_82[0], MAX_BLOCKS* 1 * (BS_1 + 2 * D1) / (1 + REF_1) *sizeof(double));
	gpuMalloc(&BufferrecE2corn5_22[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn5_42[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn6_12[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn6_32[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn7_52[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn7_72[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	gpuMalloc(&BufferrecE2corn8_62[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	if (REF_2)gpuMalloc(&BufferrecE2corn8_82[0], MAX_BLOCKS* 1 * (BS_2 + 2 * D2) / (1 + REF_2) *sizeof(double));
	#endif
	gpuMalloc(&BufferrecE3corn1_32[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn1_42[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn2_12[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn2_22[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn3_52[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn3_62[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	gpuMalloc(&BufferrecE3corn4_72[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	if (REF_3)gpuMalloc(&BufferrecE3corn4_82[0], MAX_BLOCKS* 1 * (BS_3 + 2 * D3) / (1 + REF_3) *sizeof(double));
	#endif

	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting kernel arguments: %d \n", status);
}

void gpuMalloc(void **devPtr, size_t size){
	#if(GPU_DIRECT)
	cudaMalloc(devPtr, size);
	#else
	cudaHostAlloc(devPtr, size, 0);
	#endif
	return
}

void set_arrays_GPU(int n, int device){
	int i;

	if (mem_spot_gpu[nl[n]] == device){
		block[n][AMR_GPU] = device;
		return;
	}
	else if (mem_spot_gpu[nl[n]] != device && mem_spot_gpu[nl[n]] != -1){
		GPU_finish(n, 1);
	}
	block[n][AMR_GPU] = device;
	cudaSetDevice(device);
	mem_spot_gpu[nl[n]] = device;

	//Select correct CUDA device
	cudaStreamCreate(&commandQueueGPU[nl[n]]);

	//Create events
	for (i = 0; i < 600; i++) cudaEventCreate(&boundevent[nl[n]][i]);
	for (i = 0; i < 100; i++) cudaEventCreate(&boundevent1[nl[n]][i]);

	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in creating events: %d \n", status);

	/*ASet pointers to point to other 1D arrays*/
	p_1[nl[n]] = p_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	dq_1[nl[n]] = dq_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]); //array to store temporary data
	#if(STAGGERED)
	ps_1[nl[n]] = ps_1[0] + nl[n] * NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	psh_1[nl[n]] = psh_1[0] + nl[n] * NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#endif
	ph_1[nl[n]] = ph_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	//dU_GPU[nl[n]] = dU_GPU[0] + NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G));
	#if(!NONSYMMETRIC)
	gcov_GPU[nl[n]] = gcov_GPU[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM;
	gcon_GPU[nl[n]] = gcon_GPU[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM;
	conn_GPU[nl[n]] = conn_GPU[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*NDIM*NDIM;
	gdet_GPU[nl[n]] = gdet_GPU[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG;
	#else
	gcov_GPU[nl[n]] = gcov_GPU[0] + nl[n] * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*NPG*NDIM*NDIM;
	gcon_GPU[nl[n]] = gcon_GPU[0] + nl[n] * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*NPG*NDIM*NDIM;
	conn_GPU[nl[n]] = conn_GPU[0] + nl[n] *  ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*NDIM*NDIM*NDIM;
	gdet_GPU[nl[n]] = gdet_GPU[0] + nl[n] *  ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*NPG ;
	#endif
	//pflag_GPU[nl[n]] = pflag_GPU + ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(int));
	failimage_GPU[nl[n]] = failimage_GPU[0] + nl[n] * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL);
	Katm_GPU[nl[n]] = Katm_GPU[0] + nl[n] * (BS_1 + 2 * N1G);

	/*Allocate memory to buffers on GPU*/
	BufferF1_1[nl[n]] = BufferF1_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	BufferF2_1[nl[n]] = BufferF2_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	BufferF3_1[nl[n]] = BufferF3_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	Bufferdq_1[nl[n]] = Bufferdq_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	BufferE_1[nl[n]] = BufferE_1[0] + nl[n] * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	Bufferp_1[nl[n]] = Bufferp_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	Bufferph_1[nl[n]] = Bufferph_1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#if(LEER)
	BufferV[nl[n]] = BufferV[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#endif
	Bufferradius[nl[n]] = Bufferradius[0] + nl[n] * (BS_1 + 2 * N1G);
	#if(PPM || LEER)
	Bufferstorage1[nl[n]] = Bufferstorage1[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#endif
	#if(PRESTEP_P)
	Bufferstorage2[nl[n]] = Bufferstorage2[0] + nl[n] * NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	Bufferstorage3[nl[n]] = Bufferstorage3[0] + nl[n] * NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#else
	Bufferstorage2[nl[n]] = Bufferp_1[nl[n]];
	Bufferstorage3[nl[n]] = Bufferdq_1[nl[n]];
	#endif
	#if(STAGGERED)
	Bufferps_1[nl[n]] = Bufferps_1[0] + nl[n] * 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	Bufferpsh_1[nl[n]] = Bufferpsh_1[0] + nl[n] * 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]);
	#endif
	Buffergdet[nl[n]] = Buffergdet[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG;
	Buffergcov[nl[n]] = Buffergcov[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM;
	Buffergcon[nl[n]] = Buffergcon[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM;
	Bufferconn[nl[n]] = Bufferconn[0] + nl[n] * ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*NDIM*NDIM;
	Bufferpflag[nl[n]] = Bufferpflag[0] + nl[n] * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int));
	Bufferfailimage[nl[n]] = Bufferfailimage[0] + nl[n] * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int));
	BufferKatm[nl[n]] = BufferKatm[0] + nl[n] * (BS_1 + 2 * N1G);
	//BufferdU[nl[n]] = BufferdU[0] + nl[n] * NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G));
	dtij_GPU[nl[n]] = dtij_GPU[0] + nl[n] * (nr_workgroups[nl[n]] + 1);
	//Bufferdtij[nl[n]] = Bufferdtij[0] + nl[n] * (nr_workgroups[nl[n]] + 1);
	
	Buffersend1[nl[n]] = Buffersend1[0] + nl[n] * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
	#if(N_LEVELS>1)
	Buffersend1_3[nl[n]] = Buffersend1_3[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_3)Buffersend1_4[nl[n]] = Buffersend1_4[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_1)Buffersend1_7[nl[n]] = Buffersend1_7[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_1 && REF_3)Buffersend1_8[nl[n]] = Buffersend1_8[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	#endif
	Buffersend2[nl[n]] = Buffersend2[0] + nl[n] * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G);
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	Buffersend2_1[nl[n]] = Buffersend2_1[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_3)Buffersend2_2[nl[n]] = Buffersend2_2[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_2)Buffersend2_3[nl[n]] = Buffersend2_3[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_2 && REF_3)Buffersend2_4[nl[n]] = Buffersend2_4[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	#endif
	Buffersend3[nl[n]] = Buffersend3[0] + nl[n] * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
	#if(N_LEVELS>1)
	Buffersend3_1[nl[n]] = Buffersend3_1[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_3)Buffersend3_2[nl[n]] = Buffersend3_2[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_1)Buffersend3_5[nl[n]] = Buffersend3_5[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_1 && REF_3)Buffersend3_6[nl[n]] = Buffersend3_6[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	#endif
	Buffersend4[nl[n]] = Buffersend4[0] + nl[n] * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G);
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	Buffersend4_5[nl[n]] = Buffersend4_5[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_3)Buffersend4_6[nl[n]] = Buffersend4_6[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_2)Buffersend4_7[nl[n]] = Buffersend4_7[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	if (REF_2 && REF_3)Buffersend4_8[nl[n]] = Buffersend4_8[0] + nl[n] * NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	#endif
	#if(N3G>0)
	Buffersend5[nl[n]] = Buffersend5[0] + nl[n] * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G);
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	Buffersend5_1[nl[n]] = Buffersend5_1[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_2)Buffersend5_3[nl[n]] = Buffersend5_3[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_1)Buffersend5_5[nl[n]] = Buffersend5_5[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_1 && REF_2)Buffersend5_7[nl[n]] = Buffersend5_7[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	#endif
	Buffersend6[nl[n]] = Buffersend6[0] + nl[n] * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G);
	#if(N_LEVELS>1 && !DEREFINE_POLE)
	Buffersend6_2[nl[n]] = Buffersend6_2[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_2)Buffersend6_4[nl[n]] = Buffersend6_4[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_1)Buffersend6_6[nl[n]] = Buffersend6_6[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	if (REF_1 && REF_2)Buffersend6_8[nl[n]] = Buffersend6_8[0] + nl[n] * NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	#endif
	#endif
	Bufferrec1[nl[n]] = Bufferrec1[0] + nl[n] * (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	Bufferrec1_3[nl[n]]= Bufferrec1[nl[n]]; 
	Bufferrec1_4[nl[n]]= Bufferrec1[nl[n]]+ REF_3*(NG * (NPR + 3)*(BS_1/ (1+REF_1) + 2 * N1G)*(BS_3/ (1+REF_3) + 2 * N3G)); 
	Bufferrec1_7[nl[n]]= Bufferrec1[nl[n]]+ (REF_3 + REF_1)*(NG * (NPR + 3)*(BS_1/ (1+REF_1) + 2 * N1G)*(BS_3/ (1+REF_3) + 2 * N3G)); 
	Bufferrec1_8[nl[n]] = Bufferrec1[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec2[nl[n]] = Bufferrec2[0] + nl[n] * (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	Bufferrec2_1[nl[n]] = Bufferrec2[nl[n]];
	Bufferrec2_2[nl[n]] = Bufferrec2[nl[n]] + REF_3*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec2_3[nl[n]] = Bufferrec2[nl[n]] + (REF_3 + REF_2)*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec2_4[nl[n]] = Bufferrec2[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec3[nl[n]] = Bufferrec3[0] + nl[n] * (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	Bufferrec3_1[nl[n]]= Bufferrec3[nl[n]]; 
	Bufferrec3_2[nl[n]]= Bufferrec3[nl[n]]+ REF_3*(NG * (NPR + 3)*(BS_1/ (1+REF_1) + 2 * N1G)*(BS_3/ (1+REF_3) + 2 * N3G)); 
	Bufferrec3_5[nl[n]]= Bufferrec3[nl[n]]+ (REF_3 + REF_1)*(NG * (NPR + 3)*(BS_1/ (1+REF_1) + 2 * N1G)*(BS_3/ (1+REF_3) + 2 * N3G)); 
	Bufferrec3_6[nl[n]] = Bufferrec3[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec4[nl[n]] = Bufferrec4[0] + nl[n] * (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	Bufferrec4_5[nl[n]] = Bufferrec4[nl[n]];
	Bufferrec4_6[nl[n]] = Bufferrec4[nl[n]] + REF_3*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec4_7[nl[n]] = Bufferrec4[nl[n]] + (REF_3 + REF_2)*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	Bufferrec4_8[nl[n]] = Bufferrec4[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G));
	#if(N3G>0)
	Bufferrec5[nl[n]] = Bufferrec5[0] + nl[n] * (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	Bufferrec5_1[nl[n]] = Bufferrec5[nl[n]];
	Bufferrec5_3[nl[n]] = Bufferrec5[nl[n]] + REF_2*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	Bufferrec5_5[nl[n]] = Bufferrec5[nl[n]] + (REF_2 + REF_1)*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	Bufferrec5_7[nl[n]] = Bufferrec5[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NG * (NPR + 3) * (BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	Bufferrec6[nl[n]] = Bufferrec6[0] + nl[n] * (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	Bufferrec6_2[nl[n]] = Bufferrec6[nl[n]];
	Bufferrec6_4[nl[n]] = Bufferrec6[nl[n]] + REF_2*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	Bufferrec6_6[nl[n]] = Bufferrec6[nl[n]] + (REF_2 + REF_1)*(NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	Bufferrec6_8[nl[n]] = Bufferrec6[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NG * (NPR + 3) * (BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G));
	#endif
	#if(PRESTEP || PRESTEP2)
	tempBufferrec1[nl[n]] = tempBufferrec1[0] + nl[n] * (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	tempBufferrec2[nl[n]] = tempBufferrec2[0] + nl[n] * (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	tempBufferrec3[nl[n]] = tempBufferrec3[0] + nl[n] * (1 + REF_1)*(1 + REF_3)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	tempBufferrec4[nl[n]] = tempBufferrec4[0] + nl[n] * (1 + REF_2)*(1 + REF_3)* NG * (NPR + 3)*(BS_2 / (1 + REF_2) + 2 * N2G)*(BS_3 / (1 + REF_3) + 2 * N3G);
	#if(N3G>0)
	tempBufferrec5[nl[n]] = tempBufferrec5[0] + nl[n] * (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	tempBufferrec6[nl[n]] = tempBufferrec6[0] + nl[n] * (1 + REF_1)*(1 + REF_2)* NG * (NPR + 3)*(BS_1 / (1 + REF_1) + 2 * N1G)*(BS_2 / (1 + REF_2) + 2 * N2G);
	#endif
	#endif
	Buffersend1flux[nl[n]] = Buffersend1flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Buffersend2flux[nl[n]] = Buffersend2flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	Buffersend3flux[nl[n]] = Buffersend3flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Buffersend4flux[nl[n]] = Buffersend4flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	#if(N3G>0)
	Buffersend5flux[nl[n]] = Buffersend5flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	Buffersend6flux[nl[n]] = Buffersend6flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	#endif
	Bufferrec1flux[nl[n]] = Bufferrec1flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Bufferrec2flux[nl[n]] = Bufferrec2flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	Bufferrec3flux[nl[n]] = Bufferrec3flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Bufferrec4flux[nl[n]] = Bufferrec4flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	#if(N3G>0)
	Bufferrec5flux[nl[n]] = Bufferrec5flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	Bufferrec6flux[nl[n]] = Bufferrec6flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	#endif
	Bufferrec1_3flux[nl[n]] = Bufferrec1flux[nl[n]];
	Bufferrec1_4flux[nl[n]] = Bufferrec1flux[nl[n]] + REF_3*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec1_7flux[nl[n]] = Bufferrec1flux[nl[n]] + (REF_3 + REF_1)*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec1_8flux[nl[n]] = Bufferrec1flux[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec2_1flux[nl[n]] = Bufferrec2flux[nl[n]];
	Bufferrec2_2flux[nl[n]] = Bufferrec2flux[nl[n]] + REF_3*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec2_3flux[nl[n]] = Bufferrec2flux[nl[n]] + (REF_3 + REF_2)*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec2_4flux[nl[n]] = Bufferrec2flux[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec3_1flux[nl[n]] = Bufferrec3flux[nl[n]];
	Bufferrec3_2flux[nl[n]] = Bufferrec3flux[nl[n]] + REF_3*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec3_5flux[nl[n]] = Bufferrec3flux[nl[n]] + (REF_3 + REF_1)*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec3_6flux[nl[n]] = Bufferrec3flux[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec4_5flux[nl[n]] = Bufferrec4flux[nl[n]];
	Bufferrec4_6flux[nl[n]] = Bufferrec4flux[nl[n]] + REF_3*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec4_7flux[nl[n]] = Bufferrec4flux[nl[n]] + (REF_3 + REF_2)*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec4_8flux[nl[n]] = Bufferrec4flux[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	#if(N3G>0)
	Bufferrec5_1flux[nl[n]] = Bufferrec5flux[nl[n]];
	Bufferrec5_3flux[nl[n]] = Bufferrec5flux[nl[n]] + REF_2*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec5_5flux[nl[n]] = Bufferrec5flux[nl[n]] + (REF_2 + REF_1)*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec5_7flux[nl[n]] = Bufferrec5flux[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_2flux[nl[n]] = Bufferrec6flux[nl[n]];
	Bufferrec6_4flux[nl[n]] = Bufferrec6flux[nl[n]] + REF_2*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_6flux[nl[n]] = Bufferrec6flux[nl[n]] + (REF_2 + REF_1)*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_8flux[nl[n]] = Bufferrec6flux[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	#endif
	Bufferrec1flux1[nl[n]] = Bufferrec1flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Bufferrec2flux1[nl[n]] = Bufferrec2flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	Bufferrec3flux1[nl[n]] = Bufferrec3flux[0] + nl[n] * NPR*(BS_1)*(BS_3);
	Bufferrec4flux1[nl[n]] = Bufferrec4flux[0] + nl[n] * NPR*(BS_2)*(BS_3);
	#if(N3G>0)
	Bufferrec5flux1[nl[n]] = Bufferrec5flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	Bufferrec6flux1[nl[n]] = Bufferrec6flux[0] + nl[n] * NPR*(BS_1)*(BS_2);
	#endif
	Bufferrec1_3flux1[nl[n]] = Bufferrec1flux1[nl[n]];
	Bufferrec1_4flux1[nl[n]] = Bufferrec1flux1[nl[n]] + REF_3*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec1_7flux1[nl[n]] = Bufferrec1flux1[nl[n]] + (REF_3 + REF_1)*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec1_8flux1[nl[n]] = Bufferrec1flux1[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec2_1flux1[nl[n]] = Bufferrec2flux1[nl[n]];
	Bufferrec2_2flux1[nl[n]] = Bufferrec2flux1[nl[n]] + REF_3*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec2_3flux1[nl[n]] = Bufferrec2flux1[nl[n]] + (REF_3 + REF_2)*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec2_4flux1[nl[n]] = Bufferrec2flux1[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec3_1flux1[nl[n]] = Bufferrec3flux1[nl[n]];
	Bufferrec3_2flux1[nl[n]] = Bufferrec3flux1[nl[n]] + REF_3*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec3_5flux1[nl[n]] = Bufferrec3flux1[nl[n]] + (REF_3 + REF_1)*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec3_6flux1[nl[n]] = Bufferrec3flux1[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3)));
	Bufferrec4_5flux1[nl[n]] = Bufferrec4flux1[nl[n]];
	Bufferrec4_6flux1[nl[n]] = Bufferrec4flux1[nl[n]] + REF_3*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec4_7flux1[nl[n]] = Bufferrec4flux1[nl[n]] + (REF_3 + REF_2)*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	Bufferrec4_8flux1[nl[n]] = Bufferrec4flux1[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3)));
	#if(N3G>0)
	Bufferrec5_1flux1[nl[n]] = Bufferrec5flux1[nl[n]];
	Bufferrec5_3flux1[nl[n]] = Bufferrec5flux1[nl[n]] + REF_2*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec5_5flux1[nl[n]] = Bufferrec5flux1[nl[n]] + (REF_2 + REF_1)*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec5_7flux1[nl[n]] = Bufferrec5flux1[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_2flux1[nl[n]] = Bufferrec6flux1[nl[n]];
	Bufferrec6_4flux1[nl[n]] = Bufferrec6flux1[nl[n]] + REF_2*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_6flux1[nl[n]] = Bufferrec6flux1[nl[n]] + (REF_2 + REF_1)*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	Bufferrec6_8flux1[nl[n]] = Bufferrec6flux1[nl[n]] + (REF_2 + REF_1 + (REF_2 && REF_1))*(NPR*(BS_2 / (1 + REF_2))*(BS_1 / (1 + REF_1)));
	#endif
	#if(N_LEVELS>1)
	Bufferrec1_3flux2[nl[n]] = Bufferrec1_3flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_3)Bufferrec1_4flux2[nl[n]] = Bufferrec1_4flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_1)Bufferrec1_7flux2[nl[n]] = Bufferrec1_7flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_1 && REF_3)Bufferrec1_8flux2[nl[n]] = Bufferrec1_8flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	#if(!DEREFINE_POLE)
	Bufferrec2_1flux2[nl[n]] = Bufferrec2_1flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_3)Bufferrec2_2flux2[nl[n]] = Bufferrec2_2flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_2)Bufferrec2_3flux2[nl[n]] = Bufferrec2_3flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_2 && REF_3)Bufferrec2_4flux2[nl[n]] = Bufferrec2_4flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	#endif
	Bufferrec3_1flux2[nl[n]] = Bufferrec3_1flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_3)Bufferrec3_2flux2[nl[n]] = Bufferrec3_2flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_1)Bufferrec3_5flux2[nl[n]] = Bufferrec3_5flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	if (REF_1 && REF_3)Bufferrec3_6flux2[nl[n]] = Bufferrec3_6flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_3 / (1 + REF_3));
	#if(!DEREFINE_POLE)
	Bufferrec4_5flux2[nl[n]] = Bufferrec4_5flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_3)Bufferrec4_6flux2[nl[n]] = Bufferrec4_6flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_2)Bufferrec4_7flux2[nl[n]] = Bufferrec4_7flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	if (REF_2 && REF_3)Bufferrec4_8flux2[nl[n]] = Bufferrec4_8flux2[0] + nl[n] * NPR*(BS_2 / (1 + REF_2))*(BS_3 / (1 + REF_3));
	#endif
	#if(N3G>0)
	#if(!DEREFINE_POLE)
	Bufferrec5_1flux2[nl[n]] = Bufferrec5_1flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_2)Bufferrec5_3flux2[nl[n]] = Bufferrec5_3flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_1)Bufferrec5_5flux2[nl[n]] = Bufferrec5_5flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_1 && REF_2)Bufferrec5_7flux2[nl[n]] = Bufferrec5_7flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	Bufferrec6_2flux2[nl[n]] = Bufferrec6_2flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_2)Bufferrec6_4flux2[nl[n]] = Bufferrec6_4flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_1)Bufferrec6_6flux2[nl[n]] = Bufferrec6_6flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	if (REF_1 && REF_2)Bufferrec6_8flux2[nl[n]] = Bufferrec6_8flux2[0] + nl[n] * NPR*(BS_1 / (1 + REF_1))*(BS_2 / (1 + REF_2));
	#endif
	#endif
	#endif

	Buffersend1fine[nl[n]] = Buffersend1fine[0] + nl[n] * NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
	Buffersend3fine[nl[n]] = Buffersend3fine[0] + nl[n] * NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
	Bufferrec1fine[nl[n]] = Bufferrec1fine[0] + nl[n] * NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);
	Bufferrec3fine[nl[n]] = Bufferrec3fine[0] + nl[n] * NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G);

	Buffersend1E[nl[n]] = Buffersend1E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Buffersend2E[nl[n]] = Buffersend2E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	Buffersend3E[nl[n]] = Buffersend3E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Buffersend4E[nl[n]] = Buffersend4E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	#if(N3G>0)
	Buffersend5E[nl[n]] = Buffersend5E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	Buffersend6E[nl[n]] = Buffersend6E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	#endif
	Bufferrec1E[nl[n]] = Bufferrec1E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Bufferrec2E[nl[n]] = Bufferrec2E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	Bufferrec3E[nl[n]] = Bufferrec3E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Bufferrec4E[nl[n]] = Bufferrec4E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	#if(N3G>0)
	Bufferrec5E[nl[n]] = Bufferrec5E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	Bufferrec6E[nl[n]] = Bufferrec6E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	#endif
	Bufferrec1_3E[nl[n]] = Bufferrec1E[nl[n]];
	Bufferrec1_4E[nl[n]] = Bufferrec1E[nl[n]] + REF_3*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec1_7E[nl[n]] = Bufferrec1E[nl[n]] + (REF_3 + REF_1)*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec1_8E[nl[n]] = Bufferrec1E[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_1E[nl[n]] = Bufferrec2E[nl[n]];
	Bufferrec2_2E[nl[n]] = Bufferrec2E[nl[n]] + REF_3*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_3E[nl[n]] = Bufferrec2E[nl[n]] + (REF_3 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_4E[nl[n]] = Bufferrec2E[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_1E[nl[n]] = Bufferrec3E[nl[n]];
	Bufferrec3_2E[nl[n]] = Bufferrec3E[nl[n]] + REF_3*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_5E[nl[n]] = Bufferrec3E[nl[n]] + (REF_3 + REF_1)*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_6E[nl[n]] = Bufferrec3E[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_5E[nl[n]] = Bufferrec4E[nl[n]];
	Bufferrec4_6E[nl[n]] = Bufferrec4E[nl[n]] + REF_3*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_7E[nl[n]] = Bufferrec4E[nl[n]] + (REF_3 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_8E[nl[n]] = Bufferrec4E[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	#if(N3G>0)
	Bufferrec5_1E[nl[n]] = Bufferrec5E[nl[n]];
	Bufferrec5_3E[nl[n]] = Bufferrec5E[nl[n]] + REF_1*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec5_5E[nl[n]] = Bufferrec5E[nl[n]] + (REF_1 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec5_7E[nl[n]] = Bufferrec5E[nl[n]] + (REF_1 + REF_2 + (REF_1 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_2E[nl[n]] = Bufferrec6E[nl[n]];
	Bufferrec6_4E[nl[n]] = Bufferrec6E[nl[n]] + REF_1*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_6E[nl[n]] = Bufferrec6E[nl[n]] + (REF_1 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_8E[nl[n]] = Bufferrec6E[nl[n]] + (REF_1 + REF_2 + (REF_1 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	#endif
	Bufferrec1E1[nl[n]] = Bufferrec1E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Bufferrec2E1[nl[n]] = Bufferrec2E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	Bufferrec3E1[nl[n]] = Bufferrec3E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3);
	Bufferrec4E1[nl[n]] = Bufferrec4E[0] + nl[n] * 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3);
	#if(N3G>0)
	Bufferrec5E1[nl[n]] = Bufferrec5E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	Bufferrec6E1[nl[n]] = Bufferrec6E[0] + nl[n] * 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2);
	#endif
	Bufferrec1_3E1[nl[n]] = Bufferrec1E1[nl[n]];
	Bufferrec1_4E1[nl[n]] = Bufferrec1E1[nl[n]] + REF_3*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec1_7E1[nl[n]] = Bufferrec1E1[nl[n]] + (REF_3 + REF_1)*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec1_8E1[nl[n]] = Bufferrec1E1[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_1E1[nl[n]] = Bufferrec2E1[nl[n]];
	Bufferrec2_2E1[nl[n]] = Bufferrec2E1[nl[n]] + REF_3*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_3E1[nl[n]] = Bufferrec2E1[nl[n]] + (REF_3 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec2_4E1[nl[n]] = Bufferrec2E1[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_1E1[nl[n]] = Bufferrec3E1[nl[n]];
	Bufferrec3_2E1[nl[n]] = Bufferrec3E1[nl[n]] + REF_3*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_5E1[nl[n]] = Bufferrec3E1[nl[n]] + (REF_3 + REF_1)*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec3_6E1[nl[n]] = Bufferrec3E1[nl[n]] + (REF_3 + REF_1 + (REF_3 && REF_1))*(2 * ((BS_1 + 2 * D1) / (1 + REF_1))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_5E1[nl[n]] = Bufferrec4E1[nl[n]];
	Bufferrec4_6E1[nl[n]] = Bufferrec4E1[nl[n]] + REF_3*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_7E1[nl[n]] = Bufferrec4E1[nl[n]] + (REF_3 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	Bufferrec4_8E1[nl[n]] = Bufferrec4E1[nl[n]] + (REF_3 + REF_2 + (REF_3 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_3 + 2 * D3) / (1 + REF_3)));
	#if(N3G>0)
	Bufferrec5_1E1[nl[n]] = Bufferrec5E1[nl[n]];
	Bufferrec5_3E1[nl[n]] = Bufferrec5E1[nl[n]] + REF_1*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec5_5E1[nl[n]] = Bufferrec5E1[nl[n]] + (REF_1 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec5_7E1[nl[n]] = Bufferrec5E1[nl[n]] + (REF_1 + REF_2 + (REF_1 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_2E1[nl[n]] = Bufferrec6E1[nl[n]];
	Bufferrec6_4E1[nl[n]] = Bufferrec6E1[nl[n]] + REF_1*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_6E1[nl[n]] = Bufferrec6E1[nl[n]] + (REF_1 + REF_2)*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	Bufferrec6_8E1[nl[n]] = Bufferrec6E1[nl[n]] + (REF_1 + REF_2 + (REF_1 && REF_2))*(2 * ((BS_2 + 2 * D2) / (1 + REF_2))*((BS_1 + 2 * D1) / (1 + REF_1)));
	#endif
	#if(N_LEVELS>1)
	Bufferrec1_3E2[nl[n]] = Bufferrec1_3E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)Bufferrec1_4E2[nl[n]] = Bufferrec1_4E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_1)Bufferrec1_7E2[nl[n]] = Bufferrec1_7E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_1 && REF_3)Bufferrec1_8E2[nl[n]] = Bufferrec1_8E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	#if(!DEREFINE_POLE)
	Bufferrec2_1E2[nl[n]] = Bufferrec2_1E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)Bufferrec2_2E2[nl[n]] = Bufferrec2_2E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_2)Bufferrec2_3E2[nl[n]] = Bufferrec2_3E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_2 && REF_3)Bufferrec2_4E2[nl[n]] = Bufferrec2_4E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	#endif
	Bufferrec3_1E2[nl[n]] = Bufferrec3_1E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)Bufferrec3_2E2[nl[n]] = Bufferrec3_2E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_1)Bufferrec3_5E2[nl[n]] = Bufferrec3_5E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_1 && REF_3)Bufferrec3_6E2[nl[n]] = Bufferrec3_6E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_3 + 2 * D3) / (1 + REF_3);
	#if(!DEREFINE_POLE)
	Bufferrec4_5E2[nl[n]] = Bufferrec4_5E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)Bufferrec4_6E2[nl[n]] = Bufferrec4_6E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_2)Bufferrec4_7E2[nl[n]] = Bufferrec4_7E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_2 && REF_3)Bufferrec4_8E2[nl[n]] = Bufferrec4_8E2[0] + nl[n] * 2 * (BS_2 + 2 * D2) / (1 + REF_2) *(BS_3 + 2 * D3) / (1 + REF_3);
	#endif
	#if(N3G>0)
	#if(!DEREFINE_POLE)
	Bufferrec5_1E2[nl[n]] = Bufferrec5_1E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)Bufferrec5_3E2[nl[n]] = Bufferrec5_3E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_1)Bufferrec5_5E2[nl[n]] = Bufferrec5_5E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_1 && REF_2)Bufferrec5_7E2[nl[n]] = Bufferrec5_7E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	Bufferrec6_2E2[nl[n]] = Bufferrec6_2E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)Bufferrec6_4E2[nl[n]] = Bufferrec6_4E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_1)Bufferrec6_6E2[nl[n]] = Bufferrec6_6E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_1 && REF_2)Bufferrec6_8E2[nl[n]] = Bufferrec6_8E2[0] + nl[n] * 2 * (BS_1 + 2 * D1) / (1 + REF_1) *(BS_2 + 2 * D2) / (1 + REF_2);
	#endif
	#endif
	#endif
	#if(N3G>0)
	BuffersendE1corn9[nl[n]] = BuffersendE1corn9[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BuffersendE1corn10[nl[n]] = BuffersendE1corn10[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BuffersendE1corn11[nl[n]] = BuffersendE1corn11[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BuffersendE1corn12[nl[n]] = BuffersendE1corn12[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BuffersendE2corn5[nl[n]] = BuffersendE2corn5[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BuffersendE2corn6[nl[n]] = BuffersendE2corn6[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BuffersendE2corn7[nl[n]] = BuffersendE2corn7[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BuffersendE2corn8[nl[n]] = BuffersendE2corn8[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	#endif
	BuffersendE3corn1[nl[n]] = BuffersendE3corn1[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BuffersendE3corn2[nl[n]] = BuffersendE3corn2[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BuffersendE3corn3[nl[n]] = BuffersendE3corn3[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BuffersendE3corn4[nl[n]] = BuffersendE3corn4[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	#if(N3G>0)
	BufferrecE1corn9[nl[n]] = BufferrecE1corn9[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BufferrecE1corn10[nl[n]] = BufferrecE1corn10[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BufferrecE1corn11[nl[n]] = BufferrecE1corn11[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BufferrecE1corn12[nl[n]] = BufferrecE1corn12[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	BufferrecE2corn5[nl[n]] = BufferrecE2corn5[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BufferrecE2corn6[nl[n]] = BufferrecE2corn6[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BufferrecE2corn7[nl[n]] = BufferrecE2corn7[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	BufferrecE2corn8[nl[n]] = BufferrecE2corn8[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	#endif
	BufferrecE3corn1[nl[n]] = BufferrecE3corn1[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BufferrecE3corn2[nl[n]] = BufferrecE3corn2[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BufferrecE3corn3[nl[n]] = BufferrecE3corn3[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	BufferrecE3corn4[nl[n]] = BufferrecE3corn4[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	#if(N_LEVELS>1)
	#if(N3G>0)
	BufferrecE1corn9_3[nl[n]] = BufferrecE1corn9_3[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn9_7[nl[n]] = BufferrecE1corn9_7[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn10_1[nl[n]] = BufferrecE1corn10_1[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn10_5[nl[n]] = BufferrecE1corn10_5[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn11_2[nl[n]] = BufferrecE1corn11_2[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn11_6[nl[n]] = BufferrecE1corn11_6[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn12_4[nl[n]] = BufferrecE1corn12_4[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn12_8[nl[n]] = BufferrecE1corn12_8[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE2corn5_2[nl[n]] = BufferrecE2corn5_2[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn5_4[nl[n]] = BufferrecE2corn5_4[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn6_1[nl[n]] = BufferrecE2corn6_1[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn6_3[nl[n]] = BufferrecE2corn6_3[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn7_5[nl[n]] = BufferrecE2corn7_5[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn7_7[nl[n]] = BufferrecE2corn7_7[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn8_6[nl[n]] = BufferrecE2corn8_6[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn8_8[nl[n]] = BufferrecE2corn8_8[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	#endif
	BufferrecE3corn1_3[nl[n]] = BufferrecE3corn1_3[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)BufferrecE3corn1_4[nl[n]] = BufferrecE3corn1_4[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	BufferrecE3corn2_1[nl[n]] = BufferrecE3corn2_1[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)BufferrecE3corn2_2[nl[n]] = BufferrecE3corn2_2[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	BufferrecE3corn3_5[nl[n]] = BufferrecE3corn3_5[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)BufferrecE3corn3_6[nl[n]] = BufferrecE3corn3_6[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	BufferrecE3corn4_7[nl[n]] = BufferrecE3corn4_7[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)BufferrecE3corn4_8[nl[n]] = BufferrecE3corn4_8[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	#endif

	#if(N3G>0)
	tempBufferrecE1corn9[nl[n]] = tempBufferrecE1corn9[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	tempBufferrecE1corn10[nl[n]] = tempBufferrecE1corn10[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	tempBufferrecE1corn11[nl[n]] = tempBufferrecE1corn11[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	tempBufferrecE1corn12[nl[n]] = tempBufferrecE1corn12[0] + nl[n] * 1 * (BS_1 + 2 * D1);
	tempBufferrecE2corn5[nl[n]] = tempBufferrecE2corn5[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	tempBufferrecE2corn6[nl[n]] = tempBufferrecE2corn6[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	tempBufferrecE2corn7[nl[n]] = tempBufferrecE2corn7[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	tempBufferrecE2corn8[nl[n]] = tempBufferrecE2corn8[0] + nl[n] * 1 * (BS_2 + 2 * D2);
	#endif
	tempBufferrecE3corn1[nl[n]] = tempBufferrecE3corn1[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	tempBufferrecE3corn2[nl[n]] = tempBufferrecE3corn2[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	tempBufferrecE3corn3[nl[n]] = tempBufferrecE3corn3[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	tempBufferrecE3corn4[nl[n]] = tempBufferrecE3corn4[0] + nl[n] * 1 * (BS_3 + 2 * D3);
	#if(N_LEVELS>1)
	#if(N3G>0)
	tempBufferrecE1corn9_3[nl[n]] = tempBufferrecE1corn9_3[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)tempBufferrecE1corn9_7[nl[n]] = tempBufferrecE1corn9_7[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	tempBufferrecE1corn10_1[nl[n]] = tempBufferrecE1corn10_1[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)tempBufferrecE1corn10_5[nl[n]] = tempBufferrecE1corn10_5[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	tempBufferrecE1corn11_2[nl[n]] = tempBufferrecE1corn11_2[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)tempBufferrecE1corn11_6[nl[n]] = tempBufferrecE1corn11_6[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	tempBufferrecE1corn12_4[nl[n]] = tempBufferrecE1corn12_4[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)tempBufferrecE1corn12_8[nl[n]] = tempBufferrecE1corn12_8[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	tempBufferrecE2corn5_2[nl[n]] = tempBufferrecE2corn5_2[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)tempBufferrecE2corn5_4[nl[n]] = tempBufferrecE2corn5_4[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	tempBufferrecE2corn6_1[nl[n]] = tempBufferrecE2corn6_1[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)tempBufferrecE2corn6_3[nl[n]] = tempBufferrecE2corn6_3[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	tempBufferrecE2corn7_5[nl[n]] = tempBufferrecE2corn7_5[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)tempBufferrecE2corn7_7[nl[n]] = tempBufferrecE2corn7_7[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	tempBufferrecE2corn8_6[nl[n]] = tempBufferrecE2corn8_6[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)tempBufferrecE2corn8_8[nl[n]] = tempBufferrecE2corn8_8[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	#endif
	tempBufferrecE3corn1_3[nl[n]] = tempBufferrecE3corn1_3[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)tempBufferrecE3corn1_4[nl[n]] = tempBufferrecE3corn1_4[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	tempBufferrecE3corn2_1[nl[n]] = tempBufferrecE3corn2_1[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)tempBufferrecE3corn2_2[nl[n]] = tempBufferrecE3corn2_2[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	tempBufferrecE3corn3_5[nl[n]] = tempBufferrecE3corn3_5[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)tempBufferrecE3corn3_6[nl[n]] = tempBufferrecE3corn3_6[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	tempBufferrecE3corn4_7[nl[n]] = tempBufferrecE3corn4_7[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	if (REF_3)tempBufferrecE3corn4_8[nl[n]] = tempBufferrecE3corn4_8[0] + nl[n] * 1 * (BS_3 + N3G) / (1 + REF_3);
	#endif

	#if(N_LEVELS>1)
	#if(N3G>0)
	BufferrecE1corn9_32[nl[n]] = BufferrecE1corn9_32[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn9_72[nl[n]] = BufferrecE1corn9_72[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn10_12[nl[n]] = BufferrecE1corn10_12[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn10_52[nl[n]] = BufferrecE1corn10_52[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn11_22[nl[n]] = BufferrecE1corn11_22[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn11_62[nl[n]] = BufferrecE1corn11_62[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE1corn12_42[nl[n]] = BufferrecE1corn12_42[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	if (REF_1)BufferrecE1corn12_82[nl[n]] = BufferrecE1corn12_82[0] + nl[n] * 1 * (BS_1 + 2 * D1) / (1 + REF_1);
	BufferrecE2corn5_22[nl[n]] = BufferrecE2corn5_22[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn5_42[nl[n]] = BufferrecE2corn5_42[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn6_12[nl[n]] = BufferrecE2corn6_12[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn6_32[nl[n]] = BufferrecE2corn6_32[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn7_52[nl[n]] = BufferrecE2corn7_52[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn7_72[nl[n]] = BufferrecE2corn7_72[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	BufferrecE2corn8_62[nl[n]] = BufferrecE2corn8_62[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	if (REF_2)BufferrecE2corn8_82[nl[n]] = BufferrecE2corn8_82[0] + nl[n] * 1 * (BS_2 + 2 * D2) / (1 + REF_2);
	#endif
	BufferrecE3corn1_32[nl[n]] = BufferrecE3corn1_32[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)BufferrecE3corn1_42[nl[n]] = BufferrecE3corn1_42[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	BufferrecE3corn2_12[nl[n]] = BufferrecE3corn2_12[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)BufferrecE3corn2_22[nl[n]] = BufferrecE3corn2_22[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	BufferrecE3corn3_52[nl[n]] = BufferrecE3corn3_52[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)BufferrecE3corn3_62[nl[n]] = BufferrecE3corn3_62[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	BufferrecE3corn4_72[nl[n]] = BufferrecE3corn4_72[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	if (REF_3)BufferrecE3corn4_82[nl[n]] = BufferrecE3corn4_82[0] + nl[n] * 1 * (BS_3 + 2 * D3) / (1 + REF_3);
	#endif
	
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in setting kernel arguments 4.6: %d \n", status);
}
double check = 1.0;

void GPU_write(int n)
{
	int i, j, z, k;
	double radius_GPU[(BS_1 + 2 * N1G)], r, th, phi, X[NDIM];
	cudaSetDevice(block[n][AMR_GPU]);

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
	int pg, d1, d2;
	#pragma omp parallel private(i, j, z, pg, d1, d2)
	{
		#pragma omp for collapse(2) schedule(dynamic)
		ZSLOOP3D(N1_GPU_offset[n] - N1G, N1_GPU_offset[n] + BS_1 - 1 + N1G, N2_GPU_offset[n] - N2G, N2_GPU_offset[n] + BS_2 - 1 + N2G, N3_GPU_offset[n], N3_GPU_offset[n]){
			for (pg = 0; pg < NPG; pg++){
				#if(!NONSYMMETRIC)
				gdet_GPU[nl[n]][pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gdet[nl[n]][index_2D(n, i, j, z)][pg];
				#else
				#endif
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						gcov_GPU[nl[n]][d1*NDIM*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + d2*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcov[nl[n]][index_2D(n, i, j, z)][pg][d1][d2];
						gcon_GPU[nl[n]][d1*NDIM*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + d2*NPG*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = gcon[nl[n]][index_2D(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}

			for (pg = 0; pg < NDIM; pg++){
				for (d1 = 0; d1 < NDIM; d1++){
					for (d2 = 0; d2 < NDIM; d2++){
						#if(!NONSYMMETRIC)
						conn_GPU[nl[n]][d1*NDIM*NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + d2*NDIM*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]]) + pg*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])
							+ (i - N1_GPU_offset[n] + N1G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)] = conn[nl[n]][index_2D(n, i, j, z)][pg][d1][d2];
						#else
						#endif
					}
				}
			}
			#if(LEER)
			for (k = 0; k < 6; k++){
				dq_1[nl[n]][k*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) + (i - N1_GPU_offset[n] + N1G)*(BS_3 + 2 * N3G)*(BS_2 + 2 * N2G) + (j - N2_GPU_offset[n] + N2G)*(BS_3 + 2 * N3G) + (z - N3_GPU_offset[n] + N3G)] = V[nl[n]][index_3D(n, i, j, z)][k];
			}
			#endif
		}
	}

	#if(!NONSYMMETRIC)
	cudaMemcpyAsync(Buffergdet[nl[n]], gdet_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcov[nl[n]], gcov_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Buffergcon[nl[n]], gcon_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	cudaMemcpyAsync(Bufferconn[nl[n]], conn_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*NDIM*NDIM*sizeof(double), cudaMemcpyHostToDevice, commandQueueGPU[nl[n]]);
	#else
	#endif

	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error in GPU_write: %d \n", status);
}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	/*Set arguments of kernel*/
	cudaSetDevice(block[n][AMR_GPU]);

	int nr_workgroups_local[1];
	if (poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1) * (BS_2 + 2 * D2) * (BS_3 + 2 * D3)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*(N1G + 2 * D1) + 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*(N2G + 2 * D2) + 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*(N3G + 2 * D3)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*(N1G + 2 * D1) + 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*(N2G + 2 * D2) + 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*(N3G + 2 * D3)) / LOCAL_WORK_SIZE;

	if (dir == 1){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
	}
	else if (dir == 2){
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
	}
	else{
		if (flag == 1){
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
		else{
			fluxcalcprep << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], dir, lim, ppm_solver, BufferV[nl[n]], poststep_p);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error Fluxcalcprep %d \n", status);
}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	cudaSetDevice(block[n][AMR_GPU]);

	int nr_workgroups_local[1];
	if (poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) % LOCAL_WORK_SIZE) + (BS_1 + 2 * D1 - (dir == 1)) * (BS_2 + 2 * D2 - (dir == 2)) * (BS_3 + 2 * D3 - (dir == 3))) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + 2 * D2 - (dir == 2))*(BS_3 + 2 * D3 - (dir == 3))*(N1G + 2 * D1 - (dir == 1)) + 2 * (BS_1 + 2 * D1 - (dir == 1))*(BS_3 + 2 * D3 - (dir == 3))*(N2G + 2 * D2 - (dir == 2)) + 2 * (BS_1 + 2 * D1 - (dir == 1))*(BS_2 + 2 * D2 - (dir == 2))*(N3G + 2 * D3 - (dir == 3))) % LOCAL_WORK_SIZE) + 2 * (BS_2 + 2 * D2 - (dir == 2))*(BS_3 + 2 * D3 - (dir == 3))*(N1G + 2 * D1 - (dir == 1)) + 2 * (BS_1 + 2 * D1 - (dir == 1))*(BS_3 + 2 * D3 - (dir == 3))*(N2G + 2 * D2 - (dir == 2)) + 2 * (BS_1 + 2 * D1 - (dir == 1))*(BS_2 + 2 * D2 - (dir == 2))*(N3G + 2 * D3 - (dir == 3))) / LOCAL_WORK_SIZE;
	
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),  
				dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
	}
	else{
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3),
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
				 lim, dir, gam, cour, dtij_GPU[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), 
				 dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], poststep_p);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Fluxcalc2D2 %d\n", status);
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
	int nr;
	if (dir == 1) nr = nr_workgroups2_1[nl[n]];
	else if (dir == 2) nr = nr_workgroups2_2[nl[n]];
	else if (dir == 3) nr = nr_workgroups2_3[nl[n]];
	cudaSetDevice(block[n][AMR_GPU]);
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
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) + (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) +
		(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	fix_flux << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error fixflux %d \n", status);
}

void GPU_consttransport_bound(void){
	int n;

	gpu = 1;
	#if(TRANS_BOUND)
	E_average();
	#endif

	#if(PRESTEP)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) == 2 * AMR_SWITCHTIMELEVEL - 1){
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
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1 && ((block[n_ord[n]][AMR_NSTEP] % (2 * AMR_SWITCHTIMELEVEL) != 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1))){ //
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
	#elif(PRESTEP2)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (prestep_full[nl[n_ord[n]]] == 1){
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
	#else
	#if(!TIMESTEP_JET)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E3_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send1(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec1(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send2(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec2(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec2(E_corn, BufferE_1, n_ord[n], 2);
	}
	
	#if(N3G>0)
	#if(!TIMESTEP_JET)
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_send3(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E_rec3(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}	
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E1_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_send_corn(E_corn, BufferE_1, n_ord[n]);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 1);
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		E2_receive_corn(E_corn, BufferE_1, n_ord[n], 2);
	}
	#endif
	#endif
	#endif

}

void GPU_consttransport1(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	if (poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) % LOCAL_WORK_SIZE) + (BS_1 + N1G) * (BS_2 + N2G) * (BS_3 + N3G)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*(N1G + 2 * D1) + 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*(N2G + 2 * D2) + 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*(N3G + 2 * D3)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + 2 * D2)*(BS_3 + 2 * D3)*(N1G + 2 * D1) + 2 * (BS_1 + 2 * D1)*(BS_3 + 2 * D3)*(N2G + 2 * D2) + 2 * (BS_1 + 2 * D1)*(BS_2 + 2 * D2)*(N3G + 2 * D3)) / LOCAL_WORK_SIZE;
	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], poststep_p);
	}
	else{
		consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferstorage3[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], poststep_p);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	if (poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2)*(BS_3 + D3)*(N1G + D1) + 2 * (BS_1 + D1)*(BS_3 + D3)*(N2G + D2) + 2 * (BS_1 + D1)*(BS_2 + D2)*(N3G + D3)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2)*(BS_3 + D3)*(N1G + D1) + 2 * (BS_1 + D1)*(BS_3 + D3)*(N2G + D2) + 2 * (BS_1 + D1)*(BS_2 + D2)*(N3G + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), poststep_p);
	}
	else{
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), poststep_p);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport2 %d \n", status);
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	if(poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2)*(BS_3 + D3)*(N1G + D1) + 2 * (BS_1 + D1)*(BS_3 + D3)*(N2G + D2) + 2 * (BS_1 + D1)*(BS_2 + D2)*(N3G + D3)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2)*(BS_3 + D3)*(N1G + D1) + 2 * (BS_1 + D1)*(BS_3 + D3)*(N2G + D2) + 2 * (BS_1 + D1)*(BS_2 + D2)*(N3G + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt, poststep_p);
	}
	else{
		consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferpsh_1[nl[n]], BufferE_1[nl[n]], Dt, poststep_p);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status )fprintf(stderr, "Error constransport3 %d \n", status);
}

void GPU_consttransport3_post(double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) % LOCAL_WORK_SIZE) + 2 * (BS_2 + D2)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_3 + D3) + 2 * (BS_1 + D1)*(BS_2 + D2)) / LOCAL_WORK_SIZE;
	
	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	consttransport3_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], Buffergdet[nl[n]], Bufferps_1[nl[n]], Bufferps_1[nl[n]], BufferE_1[nl[n]], Dt);
	
	status = cudaGetLastError();
	if (cudaSuccess != status)fprintf(stderr, "Error constransport3_post %d \n", status);
}

void GPU_flux_ct1(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	flux_ct1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) % LOCAL_WORK_SIZE) + (BS_1 + D1) * (BS_2 + D2) * (BS_3 + D3)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	flux_ct2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]]);
	 //cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fluxct2 %d\n", status);
}

void GPU_Utoprim(int flag, int n, double Dt)
{
	#if(!V100)
	int nr_workgroups_local[1];
	cudaSetDevice(block[n][AMR_GPU]);
	if (poststep_p==0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3*N1G + 2 * (BS_1 - 2 * N1G)*BS_3*N2G + 2 * (BS_1 - 2 * N1G)*(BS_2 - 2 * N2G)*N3G) % LOCAL_WORK_SIZE) + 2 * BS_2*BS_3*N1G + 2 * (BS_1 - 2 * N1G)*BS_3*N2G + 2 * (BS_1 - 2 * N1G)*(BS_2 - 2 * N2G)*N3G) / LOCAL_WORK_SIZE;
	if (flag == 0){
		Utoprim0 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	else{
		Utoprim0 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim0 %d\n", status);

	if (flag == 0){
		Utoprim1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim1 %d\n", status);
	if (flag == 0){
		Utoprim2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	else{
		Utoprim2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error Utoprim1 %d\n", status);
	#endif
}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G) * (BS_2 + 2 * N2G) * (BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 1){
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
	}
	else{
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fixuputoprim %d\n", status);
}

void GPU_fixup(int flag, int n, double Dt)
{
	int nr_workgroups_local[1];
	cudaSetDevice(block[n][AMR_GPU]);
	if (poststep_p == 0) nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;
	else nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3*N1G + 2 * (BS_1 - 2 * N1G)*BS_3*N2G + 2 * (BS_1 - 2 * N1G)*(BS_2 - 2 * N2G)*N3G) % LOCAL_WORK_SIZE) + 2 * BS_2*BS_3*N1G + 2 * (BS_1 - 2 * N1G)*BS_3*N2G + 2 * (BS_1 - 2 * N1G)*(BS_2 - 2 * N2G)*N3G) / LOCAL_WORK_SIZE;

	if (flag == 0){
		fixup << <nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferstorage2[nl[n]], Bufferpsh_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag, poststep_p);
	}
	else{
		fixup << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferph_1[nl[n]], Bufferp_1[nl[n]], Bufferstorage2[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
			Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, flag,poststep_p);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error fixup %d\n", status);
}

void GPU_cleanup_post(int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	cleanup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], BufferE_1[nl[n]]);
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) fprintf(stderr, "Error cleanup_post %d\n", status);
}

void GPU_fixup_post(int n, double Dt)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2) % LOCAL_WORK_SIZE) + 2 * (BS_2)*(BS_3) + (2 * BS_2*BS_3+2 * BS_1*BS_3+2 * BS_1*BS_2)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	fixup_post << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]], Bufferdq_1[nl[n]],
		Bufferradius[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], Bufferconn[nl[n]], BufferKatm[nl[n]], gam, dx[nl[n]][1], dx[nl[n]][2], dx[nl[n]][3], a, Dt, 1);
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
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim1(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim1(0, n_ord[n]);
	}

	#if(!TRANS_BOUND)
	for (n = 0; n < n_active; n++){
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) GPU_boundprim2(1, n_ord[n]);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) GPU_boundprim2(0, n_ord[n]);
	}
	#endif

	//cudaDeviceSynchronize();
	//mpi_synch();
	//For last timestep do not receive synchronized electrice fields 
	if (rank == 0){
		begin2 = clock();
	}
	#if(PRESTEP)
	rc = 0;
	//MPI communication

	if (nstep != -1 && nstep % (2 * AMR_SWITCHTIMELEVEL) != 2 * AMR_SWITCHTIMELEVEL - 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
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
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++)if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
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
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send1(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send1(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	if (PRESTEP_P == 0 || bound_force == 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec1(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec1(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
	}
	//#pragma omp parallel for schedule(dynamic,1) private(n,status)
	for (n = 0; n < n_active; n++){
		cudaSetDevice(block[n_ord[n]][AMR_GPU]);
		if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send2(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
		else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send2(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
	}
	if (PRESTEP_P == 0 || bound_force == 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec2(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec2(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
		}
	}
	if (N3 > 1){
		//#pragma omp parallel for schedule(dynamic,1) private(n,status)
		for (n = 0; n < n_active; n++){
			cudaSetDevice(block[n_ord[n]][AMR_GPU]);
			if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_send3(p, ps, Bufferp_1, Bufferps_1, n_ord[n], 0);
			else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_send3(ph, psh, Bufferph_1, Bufferpsh_1, n_ord[n], 0);
		}
		if (PRESTEP_P == 0 || bound_force == 1){
			//#pragma omp parallel for schedule(dynamic,1) private(n,status)
			for (n = 0; n < n_active; n++){
				cudaSetDevice(block[n_ord[n]][AMR_GPU]);
				if (nstep % (2 * block[n_ord[n]][AMR_TIMELEVEL]) == 2 * block[n_ord[n]][AMR_TIMELEVEL] - 1 || nstep == -1) bound_rec3(p, ps, Bufferp_1, Bufferps_1, 0, n_ord[n]);
				else if (nstep % (block[n_ord[n]][AMR_TIMELEVEL]) == block[n_ord[n]][AMR_TIMELEVEL] - 1) bound_rec3(ph, psh, Bufferph_1, Bufferpsh_1, 0, n_ord[n]);
			}
		}	
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
		end2 = clock();
		time_spent3 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	}

	nstep = temp;
}

void GPU_boundprim1(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);

	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			boundprim1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]);
		}
		else{
			boundprim1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status ) fprintf(stderr, "Error boundprim1 %d\n", status);
	}
}

void GPU_boundprim2(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			boundprim2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[nl[n]]);
		}
		else{
			boundprim2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error boundprim2.1 %d\n", status);
	}
}

void GPU_boundprim_trans(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)) % LOCAL_WORK_SIZE) + (BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)) / LOCAL_WORK_SIZE;

	cudaSetDevice(block[n][AMR_GPU]);
	if (block[n][AMR_POLE] != 0 ){
		if (flag == 0){
			boundprim_trans << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferpsh_1[nl[n]]);
		}
		else{
			boundprim_trans << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferps_1[nl[n]]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) fprintf(stderr, "Error boundprim2.1 %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k;
	//cudaDeviceSynchronize();
	cudaSetDevice(block[n][AMR_GPU]);
	cudaMemcpyAsync(p_1[nl[n]], Bufferp_1[nl[n]], (int)(NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]))*sizeof(double), cudaMemcpyDeviceToHost, commandQueueGPU[nl[n]]);
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
		return;
	}
	else if (mem_spot_gpu[nl[n]] == -1){
		return;
	}
	else{
		//Select correct CUDA device
		cudaSetDevice(mem_spot_gpu[nl[n]]);
		
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

	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) fprintf(stderr, "Error in GPU_finish_1: %d \n", status);
}
