#include "decsCUDA.h"
#include "include.h"
extern "C" {
#include "decs.h"
}
int gpuAlloc(void **devPtr, int size);
int gpuFree(void *devPtr, int trash1, int trash2);

#define GPU_SET (1)

//Wrapper for allocation of boundary cells
#if(GPU_DIRECT)
#define gpuAlloc(val1,val2, val3) if(val3) gpuMalloc((void**)val1,val2); else gpuMallocHost((void**)val1,val2)
#else
#define gpuAlloc(val1,val2, val3) gpuMallocHost((void**)val1,val2,0)
#endif

//Wrapper for calculation of GPU memory
#if(GPU_DIRECT)
#define gpuMem(val1,val2, val3) (double)(val2*(val3==1))
#else
#define gpuMem(val1,val2, val3) (double)(val2)
#endif

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
	#if(N_GPU>1)
	for (i = gpu_offset; i < gpu_offset + N_GPU; i++){
		gpuSetDevice(i);
		for (j = gpu_offset; j < gpu_offset + N_GPU; j++){
			if (i!=j) gpuDeviceEnablePeerAccess(j, 0);
		}
	}
	#endif
	status = gpuGetLastError();
	if (gpuSuccess != status){
		fprintf(stderr, "Error in setting peeraccess: %d \n", status);
		exit(0);
	}

	/*Set cache config, this is fastest on NVIDIA Kepler*/
	gpuDeviceSetCacheConfig(gpuFuncCachePreferL1);
	//gpuDeviceSetSharedMemConfig(gpuSharedMemBankSizeEightByte);

	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting cache: %d \n", status);
}

void set_arrays_GPU(int n, int device){
	int i;

	if (mem_spot_gpu[nl[n]] == device){
		block[n][AMR_GPU] = device;
		//alloc_bounds_GPU(n);
		return;
	}
	else if (mem_spot_gpu[nl[n]] != device && mem_spot_gpu[nl[n]] != -1){
		GPU_finish(n, 1);
	}
	block[n][AMR_GPU] = device;
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif
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
	gpuStreamCreate(&commandQueueGPU[nl[n]]);

	//Create events
	for (i = 0; i < 600; i++) gpuEventCreate(&boundevent[nl[n]][i]);
	for (i = 0; i < 100; i++) gpuEventCreate(&boundevent1[nl[n]][i]);

	status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error in creating events: %d \n", status);

	/*Allocate memory to 1D arrays*/
	gpuMallocHost((void**)&p_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMallocHost((void**)&ph_1[nl[n]], NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMallocHost((void**)&dq_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double)); //array to store temporary data
	#if(STAGGERED)
	gpuMallocHost((void**)&ps_1[nl[n]], NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMallocHost((void**)&psh_1[nl[n]], NDIM * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	#endif
	#if(!NSY)
	gpuMallocHost((void**)&gcov_GPU[nl[n]],((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10 * sizeof(double));
	gpuMallocHost((void**)&gcon_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10 * sizeof(double));
	gpuMallocHost((void**)&conn_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	gpuMallocHost((void**)&Mud_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	gpuMallocHost((void**)&Mud_inv_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	gpuMallocHost((void**)&gdet_GPU[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * sizeof(double));
	#else
	gpuMallocHost((void**)&gcov_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10 * sizeof(double));
	gpuMallocHost((void**)&gcon_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*10 * sizeof(double));
	gpuMallocHost((void**)&conn_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM*10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	gpuMallocHost((void**)&Mud_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	gpuMallocHost((void**)&Mud_inv_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	gpuMallocHost((void**)&gdet_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG  * sizeof(double));
	#endif
	//gpuMallocHost((void**)&pflag_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(int));
	//gpuMallocHost((void**)&pflag_RAD_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]), sizeof(int));
	#if(CARTESIAN_GR)
	gpuMallocHost((void**)&pflag_CART_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int));
	#endif
	gpuMallocHost((void**)&failimage_GPU[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int));
	gpuMallocHost((void**)&radius_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));

	/*Allocate memory to buffers on GPU*/
	gpuMalloc((void**)&BufferF1_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&BufferF2_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&BufferF3_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&Bufferdq_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&BufferE_1[nl[n]], NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&Bufferp_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&Bufferph_1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#if(LEER)
	gpuMalloc((void**)&BufferV[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#endif
	#if(CARTESIAN || CARTESIAN_GR)
	gpuMalloc((void**)&Bufferradius[nl[n]], ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double)); 
	#else
	gpuMalloc((void**)&Bufferradius[nl[n]], (BS_1 + 2 * N1G)*sizeof(double));
	#endif
	gpuMalloc((void**)&Bufferstorage1[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#if(DO_IMEX && RAD_M1)
	gpuMalloc((void**)&BufferU_n[nl[n]], NPR * ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMalloc((void**)&BufferU_0[nl[n]], NPR * ((BS_3 + 2 * N3G)* (BS_2 + 2 * N2G)* (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMalloc((void**)&BufferU_1[nl[n]], NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMalloc((void**)&BufferdU_RAD0[nl[n]], NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	gpuMalloc((void**)&BufferdU_RAD1[nl[n]], NPR * ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(double));
	#endif
	#if((N_LEVELS_1D_INT>0) || RAD_M1 || RESISTIVE || TWO_T)
	gpuMalloc((void**)&Bufferstorage2[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double)); //Temp storage for conserved quantities
	gpuMalloc((void**)&Bufferstorage3[nl[n]], NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double)); //Temp storage for cell centered electric field
	#else
	Bufferstorage2[nl[n]] = Bufferp_1[nl[n]]; //Temp storage for conserved quantities
	Bufferstorage3[nl[n]] = Bufferdq_1[nl[n]]; //Temp storage for cell centered electric field
	#endif
	#if(STAGGERED)
	gpuMalloc((void**)&Bufferps_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	gpuMalloc((void**)&Bufferpsh_1[nl[n]], 3 * ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double));
	#endif
	#if(!NSY)
	gpuMalloc((void**)&Buffergdet[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double));
	gpuMalloc((void**)&Buffergcov[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	gpuMalloc((void**)&Buffergcon[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	gpuMalloc((void**)&BufferMud[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	gpuMalloc((void**)&BufferMud_inv[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	gpuMalloc((void**)&Bufferconn[nl[n]], ((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM * 10 * sizeof(double));
	#else
	gpuMalloc((void**)&Buffergdet[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG*sizeof(double));
	gpuMalloc((void**)&Buffergcov[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	gpuMalloc((void**)&Buffergcon[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NPG * 10 * sizeof(double));
	#if(FRAME_TRANSFORM)
	gpuMalloc((void**)&BufferMud[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	gpuMalloc((void**)&BufferMud_inv[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NSOLVER * NDIM * NDIM * sizeof(double));
	#endif
	gpuMalloc((void**)&Bufferconn[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem2[nl[n]])*NDIM * 10 * sizeof(double));
	#endif
	gpuMalloc((void**)&Bufferpflag[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(int));
	#if(RAD_M1)
	gpuMalloc((void**)&Bufferpflag_RAD[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int));
	#endif
	#if(CARTESIAN_GR)
	gpuMalloc((void**)&Bufferpflag_CART[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int));
	#endif
	gpuMalloc((void**)&Bufferfailimage[nl[n]], ((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NFAIL * sizeof(int));
	//gpuMalloc((void**)&BufferdU[nl[n]], NPR*((BS_2 + 2 * N2G)*(BS_1 + 2 * N1G))*sizeof(double));
	gpuMallocHost((void**)&dtij1_GPU[nl[n]], (nr_workgroups[nl[n]] + 1) * sizeof(double));
	gpuMallocHost((void**)&dtij2_GPU[nl[n]], (nr_workgroups[nl[n]] + 1) * sizeof(double));
	gpuMallocHost((void**)&dtij3_GPU[nl[n]], (nr_workgroups[nl[n]] + 1) * sizeof(double));

	//alloc_bounds_GPU(n);

	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting kernel arguments 4.6: %d \n", status);
}

//Set arrays for boundary cell transfer on GPU
void alloc_bounds_GPU(int n){
	int ref1, ref2, ref3;

	//Send buffers primitive variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Buffersend2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Buffersend2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Buffersend2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Buffersend4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Buffersend4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Buffersend4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Buffersend3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Buffersend3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Buffersend3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Buffersend1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Buffersend1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Buffersend1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Buffersend5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Buffersend5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Buffersend5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Buffersend6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Buffersend6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Buffersend6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Buffersend6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers primitive variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G)*sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec5[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec6[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G)*sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if(ref3)gpuAlloc(&Bufferrec1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&Bufferrec5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	#if(PRESTEP || PRESTEP2)
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec1[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec2[nl[n]], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec3[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec4[nl[n]], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec5[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&tempBufferrec6[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec4_5[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec4_6[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec4_7[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec4_8[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec4_5[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&tempBufferrec4_6[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&tempBufferrec4_7[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&tempBufferrec4_8[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec2_1[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec2_2[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec2_3[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec2_4[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec2_1[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&tempBufferrec2_2[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&tempBufferrec2_3[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&tempBufferrec2_4[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec1_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec1_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec1_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec1_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec1_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&tempBufferrec1_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&tempBufferrec1_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&tempBufferrec1_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec3_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec3_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec3_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec3_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec3_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&tempBufferrec3_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&tempBufferrec3_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&tempBufferrec3_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec6_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec6_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec6_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec6_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec6_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&tempBufferrec6_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&tempBufferrec6_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&tempBufferrec6_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec5_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec5_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec5_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrec5_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&tempBufferrec5_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&tempBufferrec5_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&tempBufferrec5_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&tempBufferrec5_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif
	#endif

	//Send buffers flux variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers flux variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec5flux[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec6flux[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4flux[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7flux[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec1flux1[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec2flux1[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec3flux1[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec4flux1[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec5flux1[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec6flux1[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4flux1[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7flux1[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4flux2[nl[n]], (NPR*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7flux2[nl[n]], (NPR*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffers misc variables
	gpuMallocHost((void**)&Buffersend1fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	gpuMallocHost((void**)&Buffersend3fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	gpuMallocHost((void**)&Bufferrec1fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));
	gpuMallocHost((void**)&Bufferrec3fine[nl[n]], NPR*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G)*sizeof(double));

	//Send buffers E variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5E[nl[n]], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6E[nl[n]], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend1E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend2E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend3E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend4E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend5E[nl[n]], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuAlloc(&Buffersend6E[nl[n]], 2*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers E variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec1E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec2E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec3E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec4E[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec5E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec6E[nl[n]], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4E[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7E[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec1E1[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec2E1[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec3E1[nl[n]], 2*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec4E1[nl[n]], 2*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec5E1[nl[n]], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuAlloc(&Bufferrec6E1[nl[n]], 2*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4E1[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7E1[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec4_5E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec4_6E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec4_7E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec4_8E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec2_1E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec2_2E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec2_3E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuAlloc(&Bufferrec2_4E2[nl[n]], (2*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec1_3E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec1_4E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec1_7E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec1_8E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec3_1E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuAlloc(&Bufferrec3_2E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec3_5E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuAlloc(&Bufferrec3_6E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec6_2E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec6_4E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec6_6E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec6_8E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuAlloc(&Bufferrec5_1E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuAlloc(&Bufferrec5_3E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuAlloc(&Bufferrec5_5E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuAlloc(&Bufferrec5_7E2[nl[n]], (2*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN9P] >= 0 && block[block[n][AMR_CORN9P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10P] >= 0 && block[block[n][AMR_CORN10P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11P] >= 0 && block[block[n][AMR_CORN11P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12P] >= 0 && block[block[n][AMR_CORN12P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5P] >= 0 && block[block[n][AMR_CORN5P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6P] >= 0 && block[block[n][AMR_CORN6P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7P] >= 0 && block[block[n][AMR_CORN7P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8P] >= 0 && block[block[n][AMR_CORN8P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN1P] >= 0 && block[block[n][AMR_CORN1P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2P] >= 0 && block[block[n][AMR_CORN2P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3P] >= 0 && block[block[n][AMR_CORN3P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4P] >= 0 && block[block[n][AMR_CORN4P]][AMR_ACTIVE] == 1) gpuAlloc(&BuffersendE3corn4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	//Receive buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE3corn3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE3corn4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE3corn1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuAlloc(&BufferrecE3corn2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn3_5[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn3_6[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn4_7[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn4_8[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn1_3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn1_4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn2_1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn2_2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive tempbuffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE3corn3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE3corn4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE3corn1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuAlloc(&tempBufferrecE3corn2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE3corn3_5[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE3corn3_6[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE3corn4_7[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE3corn4_8[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE3corn1_3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE3corn1_4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&tempBufferrecE3corn2_1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&tempBufferrecE3corn2_2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive buffer E2 corn
	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn11_22[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn11_62[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn12_42[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn12_82[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn9_32[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn9_72[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE1corn10_12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE1corn10_52[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn7_52[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn7_72[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn8_62[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn8_82[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn5_22[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn5_42[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE2corn6_32[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE2corn6_12[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn3_52[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn3_62[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn4_72[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn4_82[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn1_32[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn1_42[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuAlloc(&BufferrecE3corn2_12[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuAlloc(&BufferrecE3corn2_22[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
}
double check = 1.0;

#if(DOHELM)
void eos_init_GPU(void) {
	int i, j;
	int eos_offset = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX) % LOCAL_WORK_SIZE;
	#if(N_GPU>1)
	gpuSetDevice(0);
	#endif

	// Setting EOS arrays: the dumbest way - to copy EOS table for each block individually
	gpuMallocHost((void**)&eos_table[0], ((EOSIMAX * EOSJMAX + eos_offset) * 21) * sizeof(double)); // should I add? OFFSET = LOCAL_WORK_SIZE - (EOSIMAX * EOSJMAX * 21) % LOCAL_WORK_SIZE
	gpuMalloc((void**)&GPU_eos_table[0], ((EOSIMAX * EOSJMAX + eos_offset) * 21) * sizeof(double)); // same here regarding the OFFSET

	// Check for errors: EOS array allocation
	status = gpuGetLastError();
	if (gpuSuccess != status) fprintf(stderr, "Error in setting EOS: %d \n", status);

	// Write EOS table for each block individually
	// fill in the host array
	for (i = 0; i < EOSIMAX; i++) for (j = 0; j < EOSJMAX; j++) {
		// helmholtz free energy table (total: 9 items)
		eos_table[0][0 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_f[i * EOSJMAX + j];
		eos_table[0][1 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fd[i * EOSJMAX + j];
		eos_table[0][2 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ft[i * EOSJMAX + j];
		eos_table[0][3 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdd[i * EOSJMAX + j];
		eos_table[0][4 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ftt[i * EOSJMAX + j];
		eos_table[0][5 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdt[i * EOSJMAX + j];
		eos_table[0][6 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fddt[i * EOSJMAX + j];
		eos_table[0][7 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fdtt[i * EOSJMAX + j];
		eos_table[0][8 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_fddtt[i * EOSJMAX + j];

		// pressure derivative with density table (total: 4 items)
		eos_table[0][9 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdf[i * EOSJMAX + j];
		eos_table[0][10 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdfd[i * EOSJMAX + j];
		eos_table[0][11 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdft[i * EOSJMAX + j];
		eos_table[0][12 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_dpdfdt[i * EOSJMAX + j];

		// electron chemical potential table (total: 4 items)
		eos_table[0][13 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_ef[i * EOSJMAX + j];
		eos_table[0][14 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_efd[i * EOSJMAX + j];
		eos_table[0][15 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_eft[i * EOSJMAX + j];
		eos_table[0][16 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_efdt[i * EOSJMAX + j];

		// number denisty table (total: 4 items)
		eos_table[0][17 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xf[i * EOSJMAX + j];
		eos_table[0][18 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xfd[i * EOSJMAX + j];
		eos_table[0][19 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xft[i * EOSJMAX + j];
		eos_table[0][20 * (EOSIMAX * EOSJMAX + eos_offset) + i * EOSJMAX + j] = eos_xfdt[i * EOSJMAX + j];
	}
	gpuMemcpy(GPU_eos_table[0], eos_table[0], ((EOSIMAX * EOSJMAX + eos_offset) * 21) * sizeof(double), gpuMemcpyHostToDevice);
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
	GPU_boundprim_cart(dir, flag, n);
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], 
					lim, dir, cour, dtij2_GPU[nl[n]], POLE_1, POLE_2, 
					dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF1_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij1_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][1],block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF2_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir,  cour, dtij2_GPU[nl[n]], POLE_1, POLE_2,
					 dx[nl[n]][2], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
					#if(CALC_MDOT)
					, mass_density_scale_cpu
					#endif
					);
				#elif(SCUDA)
				fluxcalc2D2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferF3_1[nl[n]], Bufferdq_1[nl[n]], Bufferstorage1[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]],
					 lim, dir, cour, dtij3_GPU[nl[n]], POLE_1, POLE_2,
					  dx[nl[n]][3], block[n][AMR_NSTEP] % (2 * AMR_MAXTIMELEVEL) == 2 * AMR_MAXTIMELEVEL - 1, flag
					#if (DOHELM) 
					, GPU_eos_table[0]
					#endif
					#if(CALC_MDOT)
					, mass_density_scale_cpu
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
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(consttransport2, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
		#elif(SCUDA)
		consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (BufferE_1[nl[n]], Bufferstorage3[nl[n]], BufferF1_1[nl[n]], BufferF2_1[nl[n]], BufferF3_1[nl[n]],
			Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], POLE_1, POLE_2);
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
		, mass_density_scale_cpu
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
		, mass_density_scale_cpu
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
		, mass_density_scale_cpu
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
		, mass_density_scale_cpu
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
		, mass_density_scale_cpu
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
		, mass_density_scale_cpu
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
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
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
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
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
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
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
			#if(RAD_M1)
			, Bufferpflag_RAD[nl[n]]
			#endif
			#if(CALC_MDOT)
			, mass_density_scale_cpu
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
		hipLaunchKernelGGL(fixuputoprim, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
		#endif
	}
	else{
		#if(SHIP)
		hipLaunchKernelGGL(fixuputoprim, nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
		#elif(SCUDA)
		fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpflag[nl[n]], Bufferfailimage[nl[n]]);
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
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
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
		#if(CARTESIAN_GR)
		, Bufferpflag_CART[nl[n]]
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
			hipLaunchKernelGGL(boundprim1_outflow, nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]);
			#elif(SCUDA)
			boundprim1_outflow << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[nl[n]]);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim1_outflow, nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]);
			#elif(SCUDA)
			boundprim1_outflow << < nr_workgroups_special1[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Buffergcov[nl[n]], Buffergcon[nl[n]], Buffergdet[nl[n]], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[nl[n]]);
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
			hipLaunchKernelGGL(boundprim_trans2, nr_workgroups_special2[nl[n]], local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Buffergdet[nl[n]], block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3, block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3, Bufferpsh_1[nl[n]]);
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
			hipLaunchKernelGGL(boundprim_cart, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_CART[nl[n]], dir);
			#elif(SCUDA)
			boundprim_cart << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferph_1[nl[n]], Bufferpsh_1[nl[n]], Bufferpflag_CART[nl[n]], dir);
			#endif
		}
		else{
			#if(SHIP)
			hipLaunchKernelGGL(boundprim_cart, nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]], Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_CART[nl[n]], dir);
			#elif(SCUDA)
			boundprim_cart << < nr_workgroups_local, local_work_size[0], 0, commandQueueGPU[nl[n]] >> > (Bufferp_1[nl[n]], Bufferps_1[nl[n]], Bufferpflag_CART[nl[n]], dir);
			#endif
		}
		//gpuDeviceSynchronize();
		status = gpuGetLastError();
		if (gpuSuccess != status ) fprintf(stderr, "Error boundprim_cart %d\n", status);
	}
}

void GPU_read(int n)
{
	int i, j, z, k;
	int nr_workgroups_local[1];

	//gpuDeviceSynchronize();
	#if(N_GPU>1)
	gpuSetDevice(block[n][AMR_GPU]);
	#endif

	#if(N_LEVELS_1D_INT>20)
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

void GPU_finish(int n, int force_delete)
{
	int i;

	//Select correct GPU device
	#if(N_GPU>1)
	gpuSetDevice(mem_spot_gpu[nl[n]]);
	#endif

	//Tell code no GPU
	block[n][AMR_GPU] = -1;

	if (mem_spot[nl[n]] == 0 && force_delete==0){
		return;
	}
	else if (mem_spot_gpu[nl[n]] == -1){
		fprintf(stderr, "Error, tries to deallocate empty GPU memory! \n");
		return;
	}
	else{
		//Tell the code that memory is deallocated on the GPU
		//free_bound_gpu(n);
		mem_spot_gpu[nl[n]] = -1;
	}

	//Destroy GPU events associated with block
	for (i = 0; i < 600; i++) gpuEventDestroy(boundevent[nl[n]][i]);
	for (i = 0; i < 100; i++) gpuEventDestroy(boundevent1[nl[n]][i]);
	gpuStreamDestroy(commandQueueGPU[nl[n]]);

	status += gpuFreeHost(p_1[nl[n]]);
	#if(STAGGERED)
	status += gpuFreeHost(ps_1[nl[n]]);
	status += gpuFreeHost(psh_1[nl[n]]);
	#endif
	status += gpuFreeHost(ph_1[nl[n]]);
	status += gpuFreeHost(failimage_GPU[nl[n]]);
	status += gpuFreeHost(radius_GPU[nl[n]]);
	status += gpuFreeHost(dq_1[nl[n]]);
	status += gpuFreeHost(gcov_GPU[nl[n]]);
	status += gpuFreeHost(gcon_GPU[nl[n]]);
	status += gpuFreeHost(conn_GPU[nl[n]]);
	#if(FRAME_TRANSFORM)
	status += gpuFreeHost(Mud_GPU[nl[n]]);
	status += gpuFreeHost(Mud_inv_GPU[nl[n]]);
	#endif
	status += gpuFreeHost(gdet_GPU[nl[n]]);
	//gpuFreeHost(pflag_GPU[nl[n]]);
	#if(CARTESIAN_GR)
	gpuFreeHost(pflag_CART_GPU[nl[n]]);
	#endif
    status += gpuFreeHost(dtij1_GPU[nl[n]]);
	status += gpuFreeHost(dtij2_GPU[nl[n]]);
	status += gpuFreeHost(dtij3_GPU[nl[n]]);
	//status += gpuFree(Bufferdtij[nl[n]]);
	status += gpuFree(BufferF1_1[nl[n]]);
	status += gpuFree(BufferF2_1[nl[n]]);
	status += gpuFree(BufferF3_1[nl[n]]);
	status += gpuFree(Bufferdq_1[nl[n]]);
	status += gpuFree(BufferE_1[nl[n]]);
	#if(LEER)
	status += gpuFree(BufferV[nl[n]]);
	#endif
	status += gpuFree(Bufferradius[nl[n]]);
	status += gpuFree(Bufferstorage1[nl[n]]);
	#if(DO_IMEX && RAD_M1)
	status += gpuFree(BufferU_n[nl[n]]);
	status += gpuFree(BufferU_0[nl[n]]);
	status += gpuFree(BufferU_1[nl[n]]);
	status += gpuFree(BufferdU_RAD0[nl[n]]);
	status += gpuFree(BufferdU_RAD1[nl[n]]);
	#endif
	#if((N_LEVELS_1D_INT>0) || RAD_M1 || RESISTIVE || TWO_T)
	status += gpuFree(Bufferstorage2[nl[n]]);
	status += gpuFree(Bufferstorage3[nl[n]]);
	#endif
	status += gpuFree(Bufferp_1[nl[n]]);
	status += gpuFree(Bufferph_1[nl[n]]);
	#if(STAGGERED)
	status += gpuFree(Bufferps_1[nl[n]]);
	status += gpuFree(Bufferpsh_1[nl[n]]);
	#endif
	status += gpuFree(Bufferpflag[nl[n]]);
	#if(RAD_M1)
	status += gpuFree(Bufferpflag_RAD[nl[n]]);
	#endif
	#if(CARTESIAN_GR)
	status += gpuFree(Bufferpflag_CART[nl[n]]);
	#endif
	status += gpuFree(Bufferfailimage[nl[n]]);
	//status += gpuFree(BufferdU[nl[n]]);
	status += gpuFree(Buffergcov[nl[n]]);
	status += gpuFree(Buffergcon[nl[n]]);
	status += gpuFree(Bufferconn[nl[n]]);
	#if(FRAME_TRANSFORM)
	status += gpuFree(BufferMud[nl[n]]);
	status += gpuFree(BufferMud_inv[nl[n]]);
	#endif
	status += gpuFree(Buffergdet[nl[n]]);

	//gpuDeviceSynchronize();
	status = gpuGetLastError();
	if (gpuSuccess != status ) fprintf(stderr, "Error in GPU_finish_1: %d \n", status);
}

void free_bound_gpu(int n){
	int ref1, ref2, ref3;

	//Send buffers primitive variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Buffersend1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Buffersend2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Buffersend3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Buffersend4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Buffersend5[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Buffersend6[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuFree(Buffersend1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuFree(Buffersend2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuFree(Buffersend3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuFree(Buffersend4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuFree(Buffersend5[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuFree(Buffersend6[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Buffersend2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Buffersend2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Buffersend2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Buffersend2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Buffersend4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Buffersend4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Buffersend4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Buffersend4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Buffersend3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Buffersend3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Buffersend3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Buffersend3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Buffersend1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Buffersend1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Buffersend1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Buffersend1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Buffersend5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Buffersend5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Buffersend5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Buffersend5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Buffersend6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Buffersend6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Buffersend6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Buffersend6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers primitive variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Bufferrec1[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Bufferrec2[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Bufferrec3[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Bufferrec4[nl[n]], NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Bufferrec5[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Bufferrec6[nl[n]], NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(Bufferrec5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	#if(PRESTEP || PRESTEP2)
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec1[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec2[nl[n]], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec3[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec4[nl[n]], 2 * NG * (NPR + 3)*(BS_2 + 2 * N2G)*(BS_3 + 2 * N3G) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec5[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(tempBufferrec6[nl[n]], 2 * NG * (NPR + 3)*(BS_1 + 2 * N1G)*(BS_2 + 2 * N2G) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR2P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec4_5[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec4_6[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec4_7[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec4_8[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec4_5[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(tempBufferrec4_6[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(tempBufferrec4_7[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(tempBufferrec4_8[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N2G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR4P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec2_1[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec2_2[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec2_3[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec2_4[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec2_1[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(tempBufferrec2_2[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(tempBufferrec2_3[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(tempBufferrec2_4[nl[n]], (2 * NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR3P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec1_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec1_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec1_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec1_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec1_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(tempBufferrec1_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(tempBufferrec1_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(tempBufferrec1_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR1P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec3_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec3_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec3_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec3_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec3_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(tempBufferrec3_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(tempBufferrec3_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(tempBufferrec3_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_3 / (1 + ref3) + 2 * N3G) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR5P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec6_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec6_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec6_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec6_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec6_2[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(tempBufferrec6_4[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(tempBufferrec6_6[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(tempBufferrec6_8[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1) {
		set_ref(block[n][AMR_NBR6P], n, &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec5_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec5_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec5_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrec5_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(tempBufferrec5_1[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(tempBufferrec5_3[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(tempBufferrec5_5[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(tempBufferrec5_7[nl[n]], (2 * NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * N1G)*(BS_2 / (1 + ref2) + 2 * N2G) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif
	#endif

	//Send buffers flux variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Buffersend1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Buffersend2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Buffersend3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Buffersend4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Buffersend5flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Buffersend6flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuFree(Buffersend1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuFree(Buffersend2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuFree(Buffersend3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuFree(Buffersend4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuFree(Buffersend5flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuFree(Buffersend6flux[nl[n]], NPR*(BS_2) * (BS_1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers flux variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Bufferrec1flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Bufferrec2flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Bufferrec3flux[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Bufferrec4flux[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Bufferrec5flux[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Bufferrec6flux[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4flux[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7flux[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Bufferrec1flux1[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Bufferrec2flux1[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Bufferrec3flux1[nl[n]], NPR*(BS_1)*(BS_3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Bufferrec4flux1[nl[n]], NPR*(BS_2)*(BS_3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Bufferrec5flux1[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Bufferrec6flux1[nl[n]], NPR*(BS_1)*(BS_2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4flux1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7flux1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers flux2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4flux2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_3 / (1 + ref3)) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7flux2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1))*(BS_2 / (1 + ref2)) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffers misc variables
	gpuFreeHost(Buffersend1fine[nl[n]]);
	gpuFreeHost(Buffersend3fine[nl[n]]);
	gpuFreeHost(Bufferrec1fine[nl[n]]);
	gpuFreeHost(Bufferrec3fine[nl[n]]);

	//Send buffers E variables
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Buffersend1E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Buffersend2E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Buffersend3E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Buffersend4E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Buffersend5E[nl[n]], NPR*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Buffersend6E[nl[n]], NPR*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_NBR1P] >= 0 && block[block[n][AMR_NBR1P]][AMR_ACTIVE] == 1)gpuFree(Buffersend1E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2P] >= 0 && block[block[n][AMR_NBR2P]][AMR_ACTIVE] == 1)gpuFree(Buffersend2E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR3P] >= 0 && block[block[n][AMR_NBR3P]][AMR_ACTIVE] == 1)gpuFree(Buffersend3E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4P] >= 0 && block[block[n][AMR_NBR4P]][AMR_ACTIVE] == 1)gpuFree(Buffersend4E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR5P] >= 0 && block[block[n][AMR_NBR5P]][AMR_ACTIVE] == 1)gpuFree(Buffersend5E[nl[n]], NPR*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR6P] >= 0 && block[block[n][AMR_NBR6P]][AMR_ACTIVE] == 1)gpuFree(Buffersend6E[nl[n]], NPR*(BS_2 + 2 * D2) * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_NBR6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	//Receive buffers E variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Bufferrec1E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Bufferrec2E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Bufferrec3E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Bufferrec4E[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Bufferrec5E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Bufferrec6E[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4E[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7E[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E1 variables
	if (block[n][AMR_NBR3] >= 0 && block[block[n][AMR_NBR3]][AMR_ACTIVE] == 1)gpuFree(Bufferrec1E1[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR4] >= 0 && block[block[n][AMR_NBR4]][AMR_ACTIVE] == 1)gpuFree(Bufferrec2E1[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR1] >= 0 && block[block[n][AMR_NBR1]][AMR_ACTIVE] == 1)gpuFree(Bufferrec3E1[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR2] >= 0 && block[block[n][AMR_NBR2]][AMR_ACTIVE] == 1)gpuFree(Bufferrec4E1[nl[n]], NPR*(BS_2 + 2 * D2)*(BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_NBR2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#if(N3G>0)
	if (block[n][AMR_NBR6] >= 0 && block[block[n][AMR_NBR6]][AMR_ACTIVE] == 1)gpuFree(Bufferrec5E1[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_NBR5] >= 0 && block[block[n][AMR_NBR5]][AMR_ACTIVE] == 1)gpuFree(Bufferrec6E1[nl[n]], NPR*(BS_1 + 2 * D1)*(BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_NBR5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif

	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4E1[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7E1[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Receive buffers E2 variables
	if (block[n][AMR_NBR2_1] >= 0 && block[block[n][AMR_NBR2_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR2_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec4_5E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec4_6E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec4_7E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR2_4], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec4_8E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR2_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR4_5] >= 0 && block[block[n][AMR_NBR4_5]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR4_5], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec2_1E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_6], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec2_2E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_7], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec2_3E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR4_8], &ref1, &ref2, &ref3);
		if (ref3 && ref2)gpuFree(Bufferrec2_4E2[nl[n]], (NG * (NPR + 3)*(BS_2 / (1 + ref2) + 2 * D2)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR4_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}


	if (block[n][AMR_NBR3_1] >= 0 && block[block[n][AMR_NBR3_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR3_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec1_3E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_2], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec1_4E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec1_7E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR3_6], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec1_8E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR3_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_NBR1_3] >= 0 && block[block[n][AMR_NBR1_3]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR1_3], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec3_1E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_4], &ref1, &ref2, &ref3);
		if (ref3)gpuFree(Bufferrec3_2E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_7], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec3_5E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR1_8], &ref1, &ref2, &ref3);
		if (ref3 && ref1)gpuFree(Bufferrec3_6E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_3 / (1 + ref3) + 2 * D3) * sizeof(double)), block[block[n][AMR_NBR1_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_NBR5_1] >= 0 && block[block[n][AMR_NBR5_1]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR5_1], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec6_2E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_3], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec6_4E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_5], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec6_6E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR5_7], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec6_8E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR5_7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	if (block[n][AMR_NBR6_2] >= 0 && block[block[n][AMR_NBR6_2]][AMR_ACTIVE] == 1) {
		set_ref(n, block[n][AMR_NBR6_2], &ref1, &ref2, &ref3);
		gpuFree(Bufferrec5_1E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_4], &ref1, &ref2, &ref3);
		if (ref2)gpuFree(Bufferrec5_3E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_6], &ref1, &ref2, &ref3);
		if (ref1)gpuFree(Bufferrec5_5E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		set_ref(n, block[n][AMR_NBR6_8], &ref1, &ref2, &ref3);
		if (ref2 && ref1)gpuFree(Bufferrec5_7E2[nl[n]], (NG * (NPR + 3)*(BS_1 / (1 + ref1) + 2 * D1)*(BS_2 / (1 + ref2) + 2 * D2) * sizeof(double)), block[block[n][AMR_NBR6_8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	//Send buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN9P] >= 0 && block[block[n][AMR_CORN9P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10P] >= 0 && block[block[n][AMR_CORN10P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11P] >= 0 && block[block[n][AMR_CORN11P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12P] >= 0 && block[block[n][AMR_CORN12P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5P] >= 0 && block[block[n][AMR_CORN5P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6P] >= 0 && block[block[n][AMR_CORN6P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7P] >= 0 && block[block[n][AMR_CORN7P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8P] >= 0 && block[block[n][AMR_CORN8P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN1P] >= 0 && block[block[n][AMR_CORN1P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN1P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2P] >= 0 && block[block[n][AMR_CORN2P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN2P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3P] >= 0 && block[block[n][AMR_CORN3P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN3P]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4P] >= 0 && block[block[n][AMR_CORN4P]][AMR_ACTIVE] == 1) gpuFree(BuffersendE3corn4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN4P]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	//Receive buffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuFree(BufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuFree(BufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuFree(BufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuFree(BufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuFree(BufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuFree(BufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuFree(BufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuFree(BufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuFree(BufferrecE3corn3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuFree(BufferrecE3corn4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuFree(BufferrecE3corn1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuFree(BufferrecE3corn2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

#	if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn3_5[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn3_6[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn4_7[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn4_8[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn1_3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn1_4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn2_1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn2_2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive tempbuffer E corn
	#if(N3G>0)
	if (block[n][AMR_CORN9] >= 0 && block[block[n][AMR_CORN9]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE1corn11[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN10] >= 0 && block[block[n][AMR_CORN10]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE1corn12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN11] >= 0 && block[block[n][AMR_CORN11]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE1corn9[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN12] >= 0 && block[block[n][AMR_CORN12]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE1corn10[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN5] >= 0 && block[block[n][AMR_CORN5]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE2corn7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN6] >= 0 && block[block[n][AMR_CORN6]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE2corn8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN7] >= 0 && block[block[n][AMR_CORN7]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE2corn5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN8] >= 0 && block[block[n][AMR_CORN8]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE2corn6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	#endif
	if (block[n][AMR_CORN1] >= 0 && block[block[n][AMR_CORN1]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE3corn3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN2] >= 0 && block[block[n][AMR_CORN2]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE3corn4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN3] >= 0 && block[block[n][AMR_CORN3]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE3corn1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN3]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	if (block[n][AMR_CORN4] >= 0 && block[block[n][AMR_CORN4]][AMR_ACTIVE] == 1) gpuFree(tempBufferrecE3corn2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN4]][AMR_NODE] / GPU_SET == rank / GPU_SET);

	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE1corn11_2[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE1corn11_6[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE1corn12_4[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE1corn12_8[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE1corn9_3[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE1corn9_7[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE1corn10_1[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE1corn10_5[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE2corn7_5[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE2corn7_7[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE2corn8_6[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE2corn8_8[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE2corn5_2[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE2corn5_4[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE2corn6_3[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE2corn6_1[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE3corn3_5[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE3corn3_6[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE3corn4_7[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE3corn4_8[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE3corn1_3[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE3corn1_4[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuFree(tempBufferrecE3corn2_1[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(tempBufferrecE3corn2_2[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	//Receive buffer E2 corn
	if (block[n][AMR_CORN9_1] >= 0 && block[block[n][AMR_CORN9_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn11_22[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn11_62[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN9_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN10_1] >= 0 && block[block[n][AMR_CORN10_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn12_42[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn12_82[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN10_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN11_1] >= 0 && block[block[n][AMR_CORN11_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn9_32[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn9_72[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN11_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN12_1] >= 0 && block[block[n][AMR_CORN12_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE1corn10_12[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE1corn10_52[nl[n]], 1 * (BS_1 + 2 * D1) * sizeof(double), block[block[n][AMR_CORN12_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

	#if(N3G>0)
	if (block[n][AMR_CORN5_1] >= 0 && block[block[n][AMR_CORN5_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn7_52[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn7_72[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN5_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN6_1] >= 0 && block[block[n][AMR_CORN6_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn8_62[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn8_82[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN6_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN7_1] >= 0 && block[block[n][AMR_CORN7_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn5_22[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn5_42[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN7_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN8_1] >= 0 && block[block[n][AMR_CORN8_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE2corn6_32[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE2corn6_12[nl[n]], 1 * (BS_2 + 2 * D2) * sizeof(double), block[block[n][AMR_CORN8_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	#endif

	if (block[n][AMR_CORN1_1] >= 0 && block[block[n][AMR_CORN1_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn3_52[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn3_62[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN1_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN2_1] >= 0 && block[block[n][AMR_CORN2_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn4_72[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn4_82[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN2_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN3_1] >= 0 && block[block[n][AMR_CORN3_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn1_32[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn1_42[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN3_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}
	if (block[n][AMR_CORN4_1] >= 0 && block[block[n][AMR_CORN4_1]][AMR_ACTIVE] == 1) {
		gpuFree(BufferrecE3corn2_12[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_1]][AMR_NODE] / GPU_SET == rank / GPU_SET);
		gpuFree(BufferrecE3corn2_22[nl[n]], 1 * (BS_3 + 2 * D3) * sizeof(double), block[block[n][AMR_CORN4_2]][AMR_NODE] / GPU_SET == rank / GPU_SET);
	}

}

int gpuFree(void *devPtr, int trash1, int val3){
	#if(GPU_DIRECT)
	if (val3) return gpuFree(devPtr);
	else return gpuFreeHost(devPtr);
	#else
	return gpuFreeHost(devPtr);
	#endif
}



