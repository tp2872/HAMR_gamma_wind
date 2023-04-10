
int gpuFree(void* devPtr, int trash1, int trash2);

#define GPU_SET (10000)


//Wrapper for allocation of boundary cells
#if(GPU_DIRECT)
int gpuAlloc(double** ptr, size_t size, int val);

int gpuAlloc(double** ptr, size_t size, int val3) {
	int error;
	if (val3) {
		error = gpuMalloc((void**)ptr, size);
	}
	else {
		error = gpuMallocHost((void**)ptr, size);
	}
	return error;
}

#else
#define gpuAlloc(val1,val2, val3) gpuMallocHost((void**)val1,val2,0)
#endif

//Wrapper for calculation of GPU memory
#if(GPU_DIRECT)
#define gpuMem(val1,val2, val3) (double)(val2*(val3==1))
#else
#define gpuMem(val1,val2, val3) (double)(val2)
#endif

//Wrapper for calculation of GPU memory
#if(GPU_DIRECT)
#define gpuMem(val1,val2, val3) (double)(val2*(val3==1))
#else
#define gpuMem(val1,val2, val3) (double)(val2)
#endif

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
	#if((N_LEVELS_1D_INT>0) || RAD_M1 || RESISTIVE || TWO_T || NEUTRINOS_M1)
	cudaMalloc(&Bufferstorage2[nl[n]], NPR*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double)); //Temp storage for conserved quantities
	cudaMalloc(&Bufferstorage3[nl[n]], NDIM*((BS_3 + 2 * N3G)*(BS_2 + 2 * N2G)*(BS_1 + 2 * N1G) + fix_mem[nl[n]])*sizeof(double)); //Temp storage for cell centered electric field
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
	#if(NEUTRINOS_M1)
	cudaMalloc(&Bufferpflag_NU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * sizeof(int));
		#if (NEUTRINOS_DEBUG)
		cudaMallocHost(&allflags_NU_GPU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NEUTRINOS_DEBUG_NFLAGS * NU_SPECIES * sizeof(double));
		cudaMalloc(&Bufferallflags_NU[nl[n]], ((BS_3 + 2 * N3G) * (BS_2 + 2 * N2G) * (BS_1 + 2 * N1G) + fix_mem[nl[n]]) * NEUTRINOS_DEBUG_NFLAGS * NU_SPECIES * sizeof(double));
		#endif
	#endif

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
	#if(NEUTRINOS_DEBUG)
	gpuFreeHost(allflags_NU_GPU[nl[n]]);
	#endif
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
	#if((N_LEVELS_1D_INT>0) || RAD_M1 || RESISTIVE || TWO_T || NEUTRINOS_M1)
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
	#if(NEUTRINOS_M1)
	status += gpuFree(Bufferpflag_NU[nl[n]]);
	#if (NEUTRINOS_DEBUG)
	status += gpuFree(Bufferallflags_NU[nl[n]]);
	#endif
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



